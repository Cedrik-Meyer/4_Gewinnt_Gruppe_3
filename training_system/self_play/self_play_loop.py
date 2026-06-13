"""
training_system/self_play/self_play_loop.py

Simuliert Spiele des Agenten gegen sich selbst (Self-Play), 
um Trajectories (Spielverläufe) für das Training zu generieren.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any

from shared.data_structures import Move
from shared.game_logic import create_empty_board, apply_move, check_winner
from shared.state_encoder import encode_state, get_legal_mask

def play_single_game(model: torch.nn.Module) -> Tuple[List[Dict[str, Any]], int]:
    """
    Lässt das übergebene Modell eine komplette Partie gegen sich selbst spielen.
    
    Args:
        model (torch.nn.Module): Das neuronale Netz (im .eval() Modus).
        
    Returns:
        Tuple[List, int]: 
            - trajectory: Eine Liste aller Züge (States und Action-Probs).
            - winner: Der finale Gewinner (1, 2 oder 0 für Unentschieden).
    """
    board = create_empty_board()
    current_player = 1
    trajectory = []
    
    while True:
        # 1. Spielfeld für das Netz übersetzen (Ebene B -> Ebene C)
        player_slot = current_player - 1
        state_tensor = encode_state(board, player_slot)
        legal_mask = get_legal_mask(board)
        
        # 2. Inferenz: Wir fügen eine Batch-Dimension hinzu [1, 2, 4, 4, 4]
        # (Da wir nicht trainieren, schalten wir die Gradienten ab)
        with torch.no_grad():
            policy_logits, _ = model(state_tensor.unsqueeze(0))
            
        logits = policy_logits.squeeze(0)  # Zurück auf Shape [16]
        
        # 3. Maskierung (Illegale Züge auf -1 Milliarde setzen)
        mask_tensor = torch.tensor(legal_mask, dtype=torch.float32)
        masked_logits = logits + (1.0 - mask_tensor) * -1e9
        
        # 4. Softmax-Aktivierung: Logits in echte Wahrscheinlichkeiten (0.0 bis 1.0) umwandeln
        action_probs = torch.softmax(masked_logits, dim=0).numpy()
        
        # 5. EXPLORATION (Training-Spezialität): 
        # Im Live-Spiel würden wir jetzt einfach Argmax (den besten Zug) nehmen.
        # Im Training ziehen wir stattdessen zufällig gewichtet nach den Wahrscheinlichkeiten,
        # damit das Modell ab und zu etwas Neues ausprobiert.
        action_index = np.random.choice(16, p=action_probs)
        
        # 6. Erfahrung festhalten (Noch ohne Reward, da das Spiel nicht vorbei ist)
        trajectory.append({
            "state": state_tensor,
            "action_probs": action_probs,
            "player": current_player
        })
        
        # 7. Zug in physikalische Koordinaten auflösen und ausführen
        x = int(action_index % 4)
        z = int(action_index // 4)
        apply_move(board, Move(x=x, z=z), current_player)
        
        # 8. Spielende prüfen
        if check_winner(board, current_player):
            return trajectory, current_player
            
        if not np.any(board == 0):
            return trajectory, 0  # Unentschieden
            
        # Spieler wechseln
        current_player = 2 if current_player == 1 else 1


def generate_self_play_data_mock(model: torch.nn.Module, num_games: int):
    """
    Platzhalter für Multiprocessing. 
    Hier werden später mehrere `play_single_game` Prozesse parallel gestartet.
    """
    # Hinweis für später: Wenn ihr hier Multiprocessing (z.B. concurrent.futures) 
    # einbaut, muss das Modell zwingend per torch.multiprocessing.set_start_method('spawn') 
    # geteilt werden, da CUDA/Mac-Tensor-Speicher sonst zu Deadlocks führt.
    pass
