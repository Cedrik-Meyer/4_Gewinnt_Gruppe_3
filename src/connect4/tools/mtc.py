"""
tools/mtc.py

Die hyper-optimierte Multiprocessing Monte Carlo Tree Search (MCTS) Engine.
Nutzt 'Root Parallelization', Neural Network Caching (Transposition Table) 
und Terminal State Caching, um in kürzester Zeit ein Maximum an 
Simulationen aus den CPU-Kernen herauszuholen.
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
# 2. DER WORKER (Läuft isoliert auf einem CPU-Kern)
# ==============================================================================

def mcts_worker_task(args):
    """
    Diese Funktion läuft isoliert auf einem einzelnen CPU-Kern. 
    Sie baut einen MCTS-Baum auf und nutzt aggressives Caching.
    """
    board, player, model_state_dict, time_limit_sec, c_puct, add_noise = args
    
    start_time = time.time()
    # 5% Puffer für Inter-Process-Communication (IPC)
    end_time = start_time + (time_limit_sec * 0.95)
    
    # PyTorch global für diesen Worker auf Inference-Modus schalten
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    
    model = Connect4Model()
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # NN Caching (Transposition Table für das Neuronale Netz)
    # Verhindert, dass das Modell dieselbe Board-Geometrie mehrfach berechnen muss
    nn_cache = {}
    
    root = MCTSNode(1.0)
    simulations = 0
    
    # --- ROOT EXPANSION ---
    legal_mask = get_legal_mask(board)
    player_slot = player - 1
    state_tensor = encode_state(board, player_slot).unsqueeze(0)
    
    logits, val_tensor = model(state_tensor)
        
    policy = logits.squeeze(0).numpy()
    policy[legal_mask == 0.0] = -1e9
    policy = policy - np.max(policy) 
    probs = np.exp(policy) / np.sum(np.exp(policy))
    
    # Dirichlet-Rauschen (Varianz für die Root Parallelization)
    if add_noise:
        noise = np.random.dirichlet([0.3] * np.sum(legal_mask == 1.0))
        noise_idx = 0
        for action in range(16):
            if legal_mask[action] == 1.0:
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
                    # NN Evaluation mit Caching!
                    board_hash = sim_board.tobytes()
                    if board_hash in nn_cache:
                        sub_probs, value = nn_cache[board_hash]
                    else:
                        sub_state = encode_state(sim_board, current_player - 1).unsqueeze(0)
                        sub_logits, sub_val = model(sub_state)
                        
                        sub_policy = sub_logits.squeeze(0).numpy()
                        sub_policy[sub_mask == 0.0] = -1e9
                        sub_policy = sub_policy - np.max(sub_policy)
                        sub_probs = np.exp(sub_policy) / np.sum(np.exp(sub_policy))
                        value = sub_val.squeeze(0).item()
                        
                        # Im RAM speichern fuer den naechsten Durchlauf
                        nn_cache[board_hash] = (sub_probs, value)
                        
                    for action in range(16):
                        if sub_mask[action] == 1.0:
                            node.children[action] = MCTSNode(prior=sub_probs[action])
                    node.is_expanded = True
            
        # 3. Backpropagation (Perspektiven invertieren!)
        for n in reversed(search_path):
            value = -value 
            n.value_sum += value
            n.visit_count += 1
            
        simulations += 1
        
    visit_counts = {action: child.visit_count for action, child in root.children.items()}
    return (visit_counts, simulations)


# ==============================================================================
# 3. HAUPT-ENGINE (Orchestriert das Multiprocessing)
# ==============================================================================

class MCTSEngine:
    def __init__(self, model_path: str, num_cores: int = None):
        self.model_path = model_path
        self.model = Connect4Model()
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu', weights_only=True))
            self.model.eval()
            self.model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        except Exception as e:
            print(f"[!] FEHLER beim Laden des Modells: {e}")
            raise e

        if num_cores is None:
            num_cores = mp.cpu_count()
        self.num_cores = max(1, min(num_cores, mp.cpu_count()))
        self.pool = mp.Pool(processes=self.num_cores)
        self._warmup()

    def _warmup(self):
        empty_board = create_empty_board()
        warmup_tasks = [
            (empty_board, 1, self.model_state_dict, 0.05, 1.5, False)
            for _ in range(self.num_cores)
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

        tasks = []
        for _ in range(self.num_cores):
            # c_puct = 1.5 ist der Standard. add_noise=True erzwingt Varianz in den Kernen
            tasks.append((board, player, self.model_state_dict, time_limit_sec, 1.5, True))

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
        # print(f"-> MCTS hat {total_simulations} Simulationen auf {active_cores} Kernen geschafft.")
                    
        best_action = max(combined_visit_counts.items(), key=lambda item: item[1])[0]
        return Move(x=best_action % 4, z=best_action // 4)
