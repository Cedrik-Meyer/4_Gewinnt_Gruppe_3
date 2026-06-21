"""
tools/mtc.py

Die ultimative Multiprocessing Monte Carlo Tree Search (MCTS) Engine.
Nutzt 'Root Parallelization': Jeder CPU-Kern baut in der vorgegebenen Zeit
einen eigenen Suchbaum auf. Am Ende werden die Visit-Counts aller Bäume 
zusammenaddiert (Ensemble-Entscheidung), um den absolut besten Zug zu finden.
"""

import time
import math
import torch
import numpy as np
import multiprocessing as mp

from shared.data_structures import Move
from shared.game_logic import apply_move, check_winner
from shared.state_encoder import encode_state, get_legal_mask
from training_system.neural_network.model import Connect4Model

class TimeOutException(Exception):
    pass

# ==============================================================================
# 1. MCTS KNOTEN (Node)
# ==============================================================================

class MCTSNode:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior  
        self.children = {}  # Dictionary: Action -> MCTSNode
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0: 
            return 0.0
        return self.value_sum / self.visit_count

# ==============================================================================
# 2. DER WORKER (Läuft isoliert auf einem CPU-Kern)
# ==============================================================================

def mcts_worker_task(args):
    """
    Diese Funktion läuft auf einem einzelnen CPU-Kern. 
    Sie baut einen MCTS-Baum auf, bis die Zeit abläuft.
    """
    # Argumente entpacken
    board, player, model_state_dict, time_limit_sec, c_puct, add_noise = args
    
    # 1. Zeit-Management
    start_time = time.time()
    # Wir ziehen 5% Puffer ab, um das sichere Beenden und den Datentransfer zu garantieren
    end_time = start_time + (time_limit_sec * 0.95)
    
    # 2. Lokales Modell laden (Jeder Prozess braucht eine EIGENE CPU-Instanz des Netzes!)
    model = Connect4Model()
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Sicherstellen, dass PyTorch in diesem Sub-Prozess nur 1 Thread nutzt,
    # da wir die Multiprocessing-Parallelisierung ueber mp.Pool machen!
    torch.set_num_threads(1)
    
    root = MCTSNode(1.0)
    simulations = 0
    
    # --- ROOT EXPANSION ---
    legal_mask = get_legal_mask(board)
    player_slot = player - 1
    state_tensor = encode_state(board, player_slot).unsqueeze(0)
    
    with torch.no_grad():
        logits, val_tensor = model(state_tensor)
        
    policy = logits.squeeze(0).numpy()
    policy[legal_mask == 0.0] = -1e9
    policy = policy - np.max(policy) # Numerische Stabilität
    probs = np.exp(policy) / np.sum(np.exp(policy))
    
    # Optionales Dirichlet-Rauschen auf der Wurzel (Sorgt dafuer, dass 8 Kerne nicht exakt denselben Baum bauen)
    if add_noise:
        noise = np.random.dirichlet([0.3] * np.sum(legal_mask == 1.0))
        noise_idx = 0
        for action in range(16):
            if legal_mask[action] == 1.0:
                # 75% Netz-Prior, 25% Rauschen
                probs[action] = 0.75 * probs[action] + 0.25 * noise[noise_idx]
                noise_idx += 1
                
    for action in range(16):
        if legal_mask[action] == 1.0:
            root.children[action] = MCTSNode(prior=probs[action])
    root.is_expanded = True
    
    # --- MCTS SCHLEIFE ---
    while time.time() < end_time:
        node = root
        sim_board = np.copy(board)
        current_player = player
        search_path = [node]
        
        # 1. Selection
        while node.is_expanded and len(node.children) > 0:
            best_score = -float('inf')
            best_action, best_child = None, None
            
            for action, child in node.children.items():
                # UCB Formel
                score = child.value() + c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
                if score > best_score:
                    best_score, best_action, best_child = score, action, child
                    
            apply_move(sim_board, Move(x=best_action % 4, z=best_action // 4), current_player)
            current_player = 2 if current_player == 1 else 1
            node = best_child
            search_path.append(node)
            
        # 2. Evaluation & Expansion
        opponent = 2 if current_player == 1 else 1
        
        if check_winner(sim_board, opponent):
            value = -1.0 # Der Spieler, der VORHER dran war, hat gewonnen
        elif np.sum(get_legal_mask(sim_board)) == 0:
            value = 0.0 # Remis
        else:
            # Netz-Inferenz fuer das neue Blatt
            sub_mask = get_legal_mask(sim_board)
            sub_state = encode_state(sim_board, current_player - 1).unsqueeze(0)
            
            with torch.no_grad():
                sub_logits, sub_val = model(sub_state)
                
            sub_policy = sub_logits.squeeze(0).numpy()
            sub_policy[sub_mask == 0.0] = -1e9
            sub_policy = sub_policy - np.max(sub_policy)
            sub_probs = np.exp(sub_policy) / np.sum(np.exp(sub_policy))
            
            for action in range(16):
                if sub_mask[action] == 1.0:
                    node.children[action] = MCTSNode(prior=sub_probs[action])
            node.is_expanded = True
            value = sub_val.squeeze(0).item()
            
        # 3. Backpropagation (Perspektiven invertieren!)
        for n in reversed(search_path):
            value = -value 
            n.value_sum += value
            n.visit_count += 1
            
        simulations += 1
        
    # --- WORKER ERGEBNIS ---
    # Wir senden nicht den ganzen Baum zurueck (dauert zu lange ueber IPC),
    # sondern nur die Anzahl der Besuche (Visit Counts) fuer die Top-Level Zuege (0-15)
    visit_counts = {action: child.visit_count for action, child in root.children.items()}
    return (visit_counts, simulations)


# ==============================================================================
# 3. HAUPT-ENGINE (Orchestriert das Multiprocessing)
# ==============================================================================

class MCTSEngine:
    """
    Die Multiprocessing MCTS-Engine.
    """
    def __init__(self, model_path: str):
        self.name = f"MCTS (Root Parallelization)"
        self.model_path = model_path
        
        # Wir laden das Modell EINMAL in den Hauptprozess, um das state_dict 
        # an die Worker zu uebergeben. Das ist schneller, als wenn jeder Worker
        # die Datei von der Festplatte liest.
        self.model = Connect4Model()
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu', weights_only=True))
            self.model.eval()
            # Extrahieren des state_dict fuer IPC-Transfer
            self.model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        except Exception as e:
            print(f"[!] FEHLER beim Laden des Modells {self.model_path} für MCTS: {e}")
            raise e

    def get_engine_move(self, board: np.ndarray, player: int, time_limit_ms: int, num_cores: int) -> Move:
        """
        Spannt einen Worker-Pool auf, laesst die MCTS-Baeume wachsen und 
        merged am Ende die Ergebnisse, um den absolut besten Zug zu waehlen.
        """
        # Wenn nur noch 1 Zug moeglich ist, muessen wir nicht rechnen
        legal_mask = get_legal_mask(board)
        legal_indices = [i for i in range(16) if legal_mask[i] == 1.0]
        if len(legal_indices) == 1:
            return Move(x=legal_indices[0] % 4, z=legal_indices[0] // 4)
            
        active_cores = min(num_cores, mp.cpu_count())
        active_cores = max(1, active_cores)
        
        time_limit_sec = time_limit_ms / 1000.0
        
        # Argumente fuer die Worker vorbereiten
        # add_noise = True sorgt dafuer, dass jeder Kern ein leicht anderes Tree-Layout bekommt
        tasks = []
        for _ in range(active_cores):
            tasks.append((board, player, self.model_state_dict, time_limit_sec, 1.5, True))
            
        # Multiprocessing Pool oeffnen
        # ACHTUNG: Auf dem Mac M-Series ist MCTS auf der CPU meist deutlich schneller als auf MPS,
        # da MPS fuer Batch_size=1 (einzelne MCTS-Evaluations) zu viel Overhead hat.
        with mp.Pool(processes=active_cores) as pool:
            # pool.map wartet, bis alle Worker fertig sind und ihre Zeit abgelaufen ist
            results = pool.map(mcts_worker_task, tasks)
            
        # --- DER MERGE ---
        # Alle Baeume (Visit Counts) zusammenfuehren
        combined_visit_counts = {action: 0 for action in legal_indices}
        total_simulations = 0
        
        for worker_visits, worker_sims in results:
            total_simulations += worker_sims
            for action, count in worker_visits.items():
                if action in combined_visit_counts:
                    combined_visit_counts[action] += count
                    
        # Den Zug waehlen, der in der Summe aller Kerne am haeufigsten besucht wurde
        best_action = max(combined_visit_counts.items(), key=lambda item: item[1])[0]
        
        # Print-Statement (kann bei Bedarf auskommentiert werden)
        # print(f"-> MCTS Root Parallelization beendet: {total_simulations} Sims auf {active_cores} Kernen.")
        
        return Move(x=best_action % 4, z=best_action // 4)
