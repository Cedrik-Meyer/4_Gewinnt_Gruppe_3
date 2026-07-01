"""
training_system/self_play/self_play_loop.py

Simuliert Partien des Agenten gegen sich selbst (Self-Play) auf der CPU.
Nach Abschluss eines Spiels werden die Spielverläufe (Trajectories) evaluiert
und mit den entsprechenden Belohnungen (Values) und korrigierten Wahrscheinlichkeiten
(Policy-Targets) im zentralen Replay Buffer hinterlegt.
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
    Nutzt stochastisches Sampling, um Exploration zu garantieren.
    
    Args:
        model (torch.nn.Module): Das neuronale Netzwerk (zwingend im .eval() Modus).
        
    Returns:
        Tuple[List[Dict[str, Any]], int]: 
            - trajectory: Eine chronologische Liste aller getätigten Züge und Zustände.
            - winner: Der finale Spielausgang (1, 2 oder 0 für Unentschieden).
    """
    board = create_empty_board()
    current_player = 1
    trajectory = []
    
    while True:
        # 1. Spielfeld in die invariante Tensor-Darstellung übersetzen (Ebene B -> Ebene C)
        player_slot = current_player - 1
        state_tensor = encode_state(board, player_slot)
        legal_mask = get_legal_mask(board)
        
        # 2. Inferenz: Berechnung der unmaskierten Logits. Hinzufügen der Batch-Dimension [1, 2, 4, 4, 4]
        with torch.no_grad():
            policy_logits, _ = model(state_tensor.unsqueeze(0))
            
        logits = policy_logits.squeeze(0)
        
        # 3. Maskierung: Die Wahrscheinlichkeit für illegale Züge auf mathematisch nahe Null (-1 Milliarde) setzen
        mask_tensor = torch.tensor(legal_mask, dtype=torch.float32)
        masked_logits = logits + (1.0 - mask_tensor) * -1e9
        
        # 4. Softmax-Aktivierung: Logits in eine valide Wahrscheinlichkeitsverteilung transformieren
        action_probs = torch.softmax(masked_logits, dim=0).numpy()
        
        # 5. Exploration: Einen Zug stochastisch (basierend auf den Wahrscheinlichkeiten) ziehen
        action_index = np.random.choice(16, p=action_probs)
        
        # 6. Zustand, Aktion und Netzwerk-Präferenz in die Trajectory aufnehmen
        trajectory.append({
            "state": state_tensor,
            "action": action_index,          
            "action_probs": action_probs,
            "legal_mask": legal_mask,        
            "player": current_player
        })
        
        # 7. Index in physikalische 3D-Koordinaten auflösen und auf dem Board ausführen
        x = int(action_index % 4)
        z = int(action_index // 4)
        apply_move(board, Move(x=x, z=z), current_player)
        
        # 8. Abbruchbedingungen (Sieg oder Unentschieden) prüfen
        if check_winner(board, current_player):
            return trajectory, current_player
            
        if not np.any(board == 0):
            return trajectory, 0  
            
        current_player = 2 if current_player == 1 else 1


def store_game_trajectory(trajectory: List[Dict[str, Any]], winner: int, replay_buffer: ReplayBuffer) -> None:
    """
    Evaluiert einen beendeten Spielverlauf und transferiert ihn in den Replay Buffer.
    
    Hier greift das Prinzip des Reflex-Lernens:
    Die Policy-Targets werden rückwirkend basierend auf dem Spielausgang (Win/Loss) 
    radikal überschrieben, um das Modell während der Backpropagation in die richtige
    Richtung zu zwingen.
    
    Args:
        trajectory (List): Die gesammelten Spielzüge einer Partie.
        winner (int): Der Sieger der Partie (0, 1 oder 2).
        replay_buffer (ReplayBuffer): Der globale Speicher für die GPU-Trainingsphase.
    """
    for step in trajectory:
        state = step["state"]
        action = step["action"]
        original_probs = step["action_probs"]
        legal_mask = step["legal_mask"]
        player = step["player"]
        
        if winner == 0:
            # Unentschieden: Weder Bestrafung noch Belohnung. 
            # Die ursprüngliche Netzwerk-Intuition wird als Ground Truth beibehalten.
            value = 0.0
            target_probs = original_probs
            
        elif player == winner:
            # GEWINNER: Dieser spezifische Zug hat langfristig zum Sieg geführt.
            # Ein 1-Hot Vektor wird erzeugt, um das Netz auf exakt diesen Zug zu trainieren.
            value = 1.0
            target_probs = np.zeros(16, dtype=np.float32)
            target_probs[action] = 1.0
            
        else:
            # VERLIERER: Dieser Zug war rückwirkend betrachtet ein Fehler.
            value = -1.0
            target_probs = np.zeros(16, dtype=np.float32)
            legal_count = np.sum(legal_mask)
            
            if legal_count > 1:
                # Strafmaßnahme: Die Wahrscheinlichkeit für den getätigten Zug wird auf 0% gesetzt.
                # Die freigewordene Wahrscheinlichkeitsmasse wird gleichmäßig auf alle ANDEREN 
                # legalen Züge verteilt, um Alternativen zu fördern.
                target_probs += legal_mask / (legal_count - 1)
                target_probs[action] = 0.0 
            else:
                target_probs = original_probs
                
        # Den aufbereiteten Datenpunkt in den RAM (Replay Buffer) schieben
        replay_buffer.push(state, target_probs, value)
