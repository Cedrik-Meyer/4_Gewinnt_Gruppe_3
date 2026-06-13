"""
training_system/self_play/self_play_loop.py

Simuliert Spiele des Agenten gegen sich selbst (Self-Play) und
speichert die Spielverläufe (Trajectories) mit den entsprechenden
Belohnungen (Rewards) im Replay Buffer.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any

from shared.data_structures import Move
from shared.game_logic import create_empty_board, apply_move, check_winner
from shared.state_encoder import encode_state, get_legal_mask
from training_system.self_play.replay_buffer import ReplayBuffer  

def play_single_game(model: torch.nn.Module) -> Tuple[List[Dict[str, Any]], int]:
    """
    Lässt das übergebene Modell eine komplette Partie gegen sich selbst spielen.
    
    Args:
        model (torch.nn.Module): Das neuronale Netz (im .eval() Modus).
        
    Returns:
        Tuple[List, int]: 
            - trajectory: Eine Liste aller Züge (States, Action-Probs und Spieler-ID).
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
        with torch.no_grad():
            policy_logits, _ = model(state_tensor.unsqueeze(0))
            
        logits = policy_logits.squeeze(0)  # Zurück auf Shape [16]
        
        # 3. Maskierung (Illegale Züge auf -1 Milliarde setzen)
        mask_tensor = torch.tensor(legal_mask, dtype=torch.float32)
        masked_logits = logits + (1.0 - mask_tensor) * -1e9
        
        # 4. Softmax-Aktivierung: Logits in echte Wahrscheinlichkeiten umwandeln
        action_probs = torch.softmax(masked_logits, dim=0).numpy()
        
        # 5. Exploration: Zufällig gewichtet nach den Wahrscheinlichkeiten ziehen
        action_index = np.random.choice(16, p=action_probs)
        
        # 6. Erfahrung festhalten (inkl. Information, welcher Spieler den Zug gemacht hat)
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


def store_game_trajectory(trajectory: List[Dict[str, Any]], winner: int, replay_buffer: ReplayBuffer) -> None:
    """
    B5_04: Nimmt den Spielverlauf (Trajectory) einer beendeten Partie, berechnet den 
    Reward aus Sicht des jeweiligen Spielers und schiebt die Züge in den Replay Buffer.
    
    Args:
        trajectory (List[Dict]): Der aufgezeichnete Spielverlauf aus play_single_game.
        winner (int): Der Gewinner des Spiels (1, 2 oder 0 für Remis).
        replay_buffer (ReplayBuffer): Der zentrale Ringpuffer.
    """
    for step in trajectory:
        state = step["state"]
        action_probs = step["action_probs"]
        player = step["player"]
        
        # Berechnung des Werts (Value/Reward) aus der Perspektive des Spielers,
        # der in genau diesem Moment am Zug war:
        if winner == 0:
            # Unentschieden bedeutet für beide Seiten Frustration/Neutralität (0.0)
            value = 0.0
        elif player == winner:
            # Dieser Zug wurde vom späteren Gewinner gemacht -> Genial (+1.0)
            value = 1.0
        else:
            # Dieser Zug wurde vom späteren Verlierer gemacht -> Schlecht (-1.0)
            value = -1.0
            
        # Die fertige Erfahrung wird in den Ringpuffer geschoben
        replay_buffer.push(state, action_probs, value)
