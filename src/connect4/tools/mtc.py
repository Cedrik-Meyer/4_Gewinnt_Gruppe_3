"""
tools/mtc.py

Die hyper-optimierte Multiprocessing Monte Carlo Tree Search (MCTS) Engine.
Nutzt 'Root Parallelization', Neural Network Caching (Transposition Table)
und Terminal State Caching, um in kürzester Zeit ein Maximum an
Simulationen aus den CPU-Kernen herauszuholen.

Optimierungen für den Live-Betrieb (wiederholte get_engine_move-Aufrufe):
  * Modell wird EINMALIG pro Worker geladen (Pool-Initializer) statt das
    state_dict bei jedem Zug an alle Kerne zu pickeln.
  * Das Modell wird per torch.jit für Batch-1-Inferenz optimiert
    (faltet u.a. BatchNorm in die Conv-Layer -> mehr Simulationen/Sekunde).
  * Die NN-Transposition-Table bleibt über Züge hinweg im Worker bestehen
    (Stellungen wiederholen sich von Zug zu Zug massiv).
  * Schwächeres, pro-Kern decorreliertes Rauschen für stärkeres Live-Spiel.
"""

import time
import math
import torch
import numpy as np
import multiprocessing as mp

from shared.data_structures import Move
from shared.game_logic import apply_move, check_winner, create_empty_board
from shared.state_encoder import encode_state, get_legal_mask
from training_system.neural_network.model import Connect4Model

# Rausch-Anteil an der Root-Policy. 0.25 ist der klassische AlphaZero-Wert für
# Self-Play-Training. Im Live-Betrieb wollen wir den stärksten Zug, brauchen aber
# etwas Decorrelation, damit die Root-Parallelization nicht auf jedem Kern denselben
# (deterministischen) Baum baut. Ein kleiner Wert diversifiziert die Kerne, ohne die
# aggregierte Zugwahl (Summe der Visit-Counts) nennenswert zu verzerren.
LIVE_NOISE_EPS = 0.10

# Obergrenze für die persistente Transposition-Table pro Worker (RAM-Schutz).
# Bei Überschreitung wird der Cache geleert. Eine komplette Partie besucht deutlich
# weniger Stellungen, der Cache überlebt also typischerweise das ganze Spiel.
_CACHE_CAP = 200_000

class TimeOutException(Exception):
    pass

# ==============================================================================
# 1. MCTS KNOTEN (Node)
# ==============================================================================

class MCTSNode:
    __slots__ = ['visit_count', 'value_sum', 'prior', 'children', 'is_expanded', 'terminal_value']

    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}  # Dictionary: Action -> MCTSNode
        self.is_expanded = False
        # Caching für bewiesene Siege/Niederlagen, spart extrem viele check_winner() Aufrufe!
        self.terminal_value = None

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

# ==============================================================================
# 2. WORKER-STATE & HILFSFUNKTIONEN
# ==============================================================================

# Pro Worker-Prozess EINMAL befülltes Modell + persistenter NN-Cache.
# Wird über den Pool-Initializer gesetzt und überlebt alle get_engine_move-Aufrufe.
_WORKER = {}


def _init_worker(state_dict, jit_path=None):
    """
    Initialisiert einen Worker-Prozess: Modell laden, für Inferenz optimieren, Cache anlegen.

    Zwei Modellquellen werden unterstützt:
      * jit_path != None: Ein vorkompiliertes TorchScript-Modell (.jit) wird direkt
        geladen. Solche Modelle sind in der Regel schon optimiert (C++-Backend) und
        liefern maximale Inferenz-Geschwindigkeit, ohne dass jeder Worker erst tracen muss.
      * sonst: Das state_dict wird in ein Connect4Model geladen und per torch.jit zur
        Laufzeit für Batch-1-Inferenz optimiert.
    """
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    if jit_path is not None:
        # Vorkompilierte, pfeilschnelle TorchScript-Version laden.
        model = torch.jit.load(jit_path, map_location='cpu')
        model.eval()
    else:
        model = Connect4Model()
        model.load_state_dict(state_dict)
        model.eval()

        # Für Batch-1-CPU-Inferenz optimieren: optimize_for_inference faltet BatchNorm
        # in die Conv-Layer und friert eval-Konstanten ein. Bei Fehlern: ungescriptetes
        # Modell als sicherer Fallback.
        try:
            example = torch.zeros(1, 2, 4, 4, 4)
            model = torch.jit.optimize_for_inference(torch.jit.trace(model, example))
        except Exception:
            pass

    _WORKER["model"] = model
    _WORKER["nn_cache"] = {}


def _masked_softmax(policy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Softmax über die Logits, wobei illegale Züge auf praktisch 0 gedrückt werden."""
    policy = policy.copy()
    policy[legal_mask == 0.0] = -1e9
    policy = policy - np.max(policy)
    exp = np.exp(policy)
    return exp / np.sum(exp)


def _evaluate(board: np.ndarray, player: int, legal_mask: np.ndarray):
    """
    Wertet eine Stellung mit dem NN aus (Policy + Value) und nutzt den persistenten
    Transposition-Cache. Die Perspektive (eigene/gegnerische Steine) ergibt sich aus
    dem Spieler am Zug; da Spieler strikt abwechseln, bestimmt die Board-Geometrie den
    Spieler eindeutig -> board.tobytes() ist als Cache-Key kollisionsfrei.
    """
    cache = _WORKER["nn_cache"]
    key = board.tobytes()
    cached = cache.get(key)
    if cached is not None:
        return cached

    state = encode_state(board, player - 1).unsqueeze(0)
    logits, val = _WORKER["model"](state)
    probs = _masked_softmax(logits.squeeze(0).numpy(), legal_mask)
    value = val.squeeze(0).item()

    if len(cache) >= _CACHE_CAP:
        cache.clear()
    cache[key] = (probs, value)
    return probs, value


def _expand(node: MCTSNode, probs: np.ndarray, legal_mask: np.ndarray):
    """Hängt für jeden legalen Zug ein Kind mit dem entsprechenden Prior an den Knoten."""
    for action in range(16):
        if legal_mask[action] == 1.0:
            node.children[action] = MCTSNode(prior=probs[action])
    node.is_expanded = True


# ==============================================================================
# 3. DER WORKER (Läuft isoliert auf einem CPU-Kern)
# ==============================================================================

def mcts_worker_task(args):
    """
    Diese Funktion läuft isoliert auf einem einzelnen CPU-Kern.
    Sie baut einen MCTS-Baum auf und nutzt das vorab geladene Modell sowie
    aggressives Caching aus dem Worker-State.
    """
    board, player, time_limit_sec, c_puct, noise_eps, seed = args

    # Pro-Kern-RNG, damit das Root-Rauschen zwischen den Kernen tatsächlich
    # decorreliert (wichtig auf fork-Systemen, wo der globale RNG-State geteilt wird).
    rng = np.random.default_rng(seed)

    # 5% Puffer für Inter-Process-Communication (IPC)
    end_time = time.monotonic() + (time_limit_sec * 0.95)

    root = MCTSNode(1.0)
    simulations = 0

    # --- ROOT EXPANSION ---
    legal_mask = get_legal_mask(board)
    probs, _ = _evaluate(board, player, legal_mask)

    # Dirichlet-Rauschen (Decorrelation für die Root Parallelization)
    if noise_eps > 0.0:
        legal_actions = np.flatnonzero(legal_mask == 1.0)
        noise = rng.dirichlet([0.3] * len(legal_actions))
        probs = probs.copy()
        probs[legal_actions] = (1.0 - noise_eps) * probs[legal_actions] + noise_eps * noise

    _expand(root, probs, legal_mask)

    # --- MCTS SCHLEIFE ---
    while time.monotonic() < end_time:
        node = root
        sim_board = np.copy(board)
        current_player = player
        search_path = [node]

        # 1. Selection (Auswahl des vielversprechendsten Astes)
        while node.is_expanded and len(node.children) > 0:
            best_score = -float('inf')
            best_action, best_child = None, None

            # Mathe-Optimierung: Wurzel nur EINMAL pro Ebene ziehen!
            sqrt_visits = math.sqrt(node.visit_count)

            for action, child in node.children.items():
                # UCB Formel
                score = child.value() + c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
                if score > best_score:
                    best_score, best_action, best_child = score, action, child

            apply_move(sim_board, Move(x=best_action % 4, z=best_action // 4), current_player)
            current_player = 2 if current_player == 1 else 1
            node = best_child
            search_path.append(node)

        # 2. Evaluation & Expansion
        if node.terminal_value is not None:
            # OPTIMIERUNG: Wir kennen diesen Knoten schon als Sieg/Niederlage!
            # Erspart uns die redundante check_winner() Prüfung.
            value = node.terminal_value
        else:
            opponent = 2 if current_player == 1 else 1

            if check_winner(sim_board, opponent):
                value = -1.0
                node.terminal_value = value # Dauerhaft cachen
            else:
                sub_mask = get_legal_mask(sim_board)
                if np.sum(sub_mask) == 0:
                    value = 0.0 # Remis
                    node.terminal_value = value
                else:
                    # NN Evaluation mit persistentem Caching!
                    sub_probs, value = _evaluate(sim_board, current_player, sub_mask)
                    _expand(node, sub_probs, sub_mask)

        # 3. Backpropagation (Perspektiven invertieren!)
        for n in reversed(search_path):
            value = -value
            n.value_sum += value
            n.visit_count += 1

        simulations += 1

    visit_counts = {action: child.visit_count for action, child in root.children.items()}
    return (visit_counts, simulations)


# ==============================================================================
# 4. HAUPT-ENGINE (Orchestriert das Multiprocessing)
# ==============================================================================

class MCTSEngine:
    def __init__(self, model_path: str, num_cores: int = None):
        self.model_path = model_path
        self.model_state_dict = None
        self.jit_path = None

        if model_path.endswith(".jit"):
            # Vorkompiliertes TorchScript-Modell: Pfad an die Worker durchreichen, die
            # es per torch.jit.load einlesen. Kein state_dict-Pickling nötig.
            try:
                # Vorab im Hauptprozess testweise laden, damit Fehler nicht erst still
                # in den Workern auftauchen.
                torch.jit.load(model_path, map_location='cpu')
            except Exception as e:
                print(f"[!] FEHLER beim Laden des TorchScript-Modells: {e}")
                raise e
            self.jit_path = model_path
        else:
            try:
                state_dict = torch.load(self.model_path, map_location='cpu', weights_only=True)
                # Sicherstellen, dass das Modell ladbar ist, bevor die Worker starten.
                probe = Connect4Model()
                probe.load_state_dict(state_dict)
                self.model_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            except Exception as e:
                print(f"[!] FEHLER beim Laden des Modells: {e}")
                raise e

        if num_cores is None:
            num_cores = mp.cpu_count()
        self.num_cores = max(1, min(num_cores, mp.cpu_count()))

        # Persistenter Pool: Modell wird EINMALIG pro Worker geladen und optimiert.
        self.pool = mp.Pool(
            processes=self.num_cores,
            initializer=_init_worker,
            initargs=(self.model_state_dict, self.jit_path),
        )
        self._warmup()

    def _warmup(self):
        empty_board = create_empty_board()
        seeds = np.random.SeedSequence().spawn(self.num_cores)
        warmup_tasks = [
            (empty_board, 1, 0.05, 1.5, 0.0, seeds[i])
            for i in range(self.num_cores)
        ]
        self.pool.map(mcts_worker_task, warmup_tasks)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def get_engine_move(self, board: np.ndarray, player: int, time_limit_ms: int) -> Move:
        legal_mask = get_legal_mask(board)
        legal_indices = [i for i in range(16) if legal_mask[i] == 1.0]
        if len(legal_indices) == 1:
            return Move(x=legal_indices[0] % 4, z=legal_indices[0] // 4)

        time_limit_sec = time_limit_ms / 1000.0

        # Pro Kern ein eigener Seed -> decorrelierte Bäume trotz deterministischem UCB.
        seeds = np.random.SeedSequence().spawn(self.num_cores)
        tasks = [
            (board, player, time_limit_sec, 1.5, LIVE_NOISE_EPS, seeds[i])
            for i in range(self.num_cores)
        ]

        # Den persistenten Pool wiederverwenden -- nur die Suche läuft jetzt unter Deadline.
        results = self.pool.map(mcts_worker_task, tasks)

        combined_visit_counts = {action: 0 for action in legal_indices}
        total_simulations = 0

        for worker_visits, worker_sims in results:
            total_simulations += worker_sims
            for action, count in worker_visits.items():
                if action in combined_visit_counts:
                    combined_visit_counts[action] += count

        # Optional: Print auskommentieren, falls ihr die pure Simulations-Zahl im Terminal sehen wollt
        # print(f"-> MCTS hat {total_simulations} Simulationen auf {self.num_cores} Kernen geschafft.")

        best_action = max(combined_visit_counts.items(), key=lambda item: item[1])[0]
        return Move(x=best_action % 4, z=best_action // 4)
