"""
training_system/self_play/self_play_loop.py

Simuliert Spiele des Agenten gegen sich selbst (Self-Play) und
speichert die Spielverläufe (Trajectories) mit den entsprechenden
Belohnungen (Rewards) und korrigierten Policy-Targets im Replay Buffer.
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
            
        logits = policy_logits.squeeze(0)
        
        # 3. Maskierung (Illegale Züge auf -1 Milliarde setzen)
        mask_tensor = torch.tensor(legal_mask, dtype=torch.float32)
        masked_logits = logits + (1.0 - mask_tensor) * -1e9
        
         # 4. Softmax-Aktivierung: Logits in echte Wahrscheinlichkeiten umwandeln
        action_probs = torch.softmax(masked_logits, dim=0).numpy()
         # 5. Exploration: Zufällig gewichtet nach den Wahrscheinlichkeiten ziehen
        action_index = np.random.choice(16, p=action_probs)
        
        #Wir speichern den TATSÄCHLICHEN ZUG (action_index) und die Maske!
        trajectory.append({
            "state": state_tensor,
            "action": action_index,          
            "action_probs": action_probs,
            "legal_mask": legal_mask,        
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
            return trajectory, 0  
            
        current_player = 2 if current_player == 1 else 1


def store_game_trajectory(trajectory: List[Dict[str, Any]], winner: int, replay_buffer: ReplayBuffer) -> None:
    """
    B5_04: Nimmt den Spielverlauf und berechnet Reward und Policy-Target.
    Hier greift nun das pure Reflex-Lernen: Gewinner-Züge werden auf 100% gepusht,
    Verlierer-Züge auf 0% gesetzt.
    """
    for step in trajectory:
        state = step["state"]
        action = step["action"]
        original_probs = step["action_probs"]
        legal_mask = step["legal_mask"]
        player = step["player"]
        
        if winner == 0:
            # Unentschieden: Weder bestrafen noch belohnen, behalte alte Intuition
            value = 0.0
            target_probs = original_probs
            
        elif player == winner:
            # GEWINNER: Dieser Zug war Gold wert! 
            # Die Kreuzentropie soll das Netz zwingen, hier 100% vorherzusagen.
            value = 1.0
            target_probs = np.zeros(16, dtype=np.float32)
            target_probs[action] = 1.0
            
        else:
            # VERLIERER: Dieser Zug führte zum Untergang!
            value = -1.0
            target_probs = np.zeros(16, dtype=np.float32)
            legal_count = np.sum(legal_mask)
            
            if legal_count > 1:
                # Wir verteilen die Wahrscheinlichkeit auf alle ANDEREN legalen Züge.
                # Das zwingt das Netz dazu, diese Spalte in Zukunft zu meiden.
                target_probs += legal_mask / (legal_count - 1)
                target_probs[action] = 0.0 
            else:
                target_probs = original_probs
                
        replay_buffer.push(state, target_probs, value)
