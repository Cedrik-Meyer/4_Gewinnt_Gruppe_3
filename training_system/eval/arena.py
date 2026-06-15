"""
training_system/eval/arena.py

Das Testgelände (Arena) für neu trainierte Modelle.
Lässt das amtierende Champion-Modell gegen das neu trainierte Kandidaten-Modell antreten.
"""

import torch
import numpy as np

from shared.data_structures import Move
from shared.game_logic import create_empty_board, apply_move, check_winner
from shared.state_encoder import encode_state, get_legal_mask
from training_system.neural_network.model import Connect4Model

def evaluate_candidate(champion: Connect4Model, candidate: Connect4Model, 
                       num_games: int = 100, win_threshold: float = 0.55) -> bool:
    """
    B6_01 & B6_02: Lässt Champion und Kandidat N Partien gegeneinander spielen.
    Nach der Hälfte der Partien werden die Seiten (Spieler 1 / Spieler 2) gewechselt.
    
    Args:
        champion (Connect4Model): Das aktuell beste Modell.
        candidate (Connect4Model): Das frisch trainierte Modell.
        num_games (int): Anzahl der Testspiele (Standard: 100).
        win_threshold (float): Die benötigte Siegrate für den Kandidaten (Standard: 55%).
        
    Returns:
        bool: True, wenn der Kandidat gewonnen hat und neuer Champion wird. Sonst False.
    """
    champion.eval()
    candidate.eval()
    
    candidate_wins = 0
    champion_wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        board = create_empty_board()
        current_player = 1
        
        # B6_01: Seitenwechsel nach der Hälfte der Partien, 
        # damit der Startspieler-Vorteil exakt ausgeglichen ist.
        if game_idx < (num_games // 2):
            p1_model = candidate
            p2_model = champion
            candidate_is_p1 = True
        else:
            p1_model = champion
            p2_model = candidate
            candidate_is_p1 = False
            
        while True:
            # Das Modell des aktuell ziehenden Spielers auswählen
            active_model = p1_model if current_player == 1 else p2_model
            
            # Zustand für das Netz codieren
            player_slot = current_player - 1
            state_tensor = encode_state(board, player_slot)
            legal_mask = get_legal_mask(board)
            
            # Inferenz (schnell, ohne Gradienten)
            with torch.no_grad():
                logits, _ = active_model(state_tensor.unsqueeze(0))
            logits = logits.squeeze(0)
            
            # Maskieren illegaler Züge
            mask_tensor = torch.tensor(legal_mask, dtype=torch.float32)
            masked_logits = logits + (1.0 - mask_tensor) * -1e9
            
            # -------------------------------------------------------------
            # DER KRITISCHE UNTERSCHIED ZUM SELF-PLAY (Exploitation)
            # Hier wird NICHT mehr gewürfelt. Wir wollen die reine Stärke 
            # testen, also wählen wir via argmax IMMER den absolut besten Zug.
            # -------------------------------------------------------------
            best_action = torch.argmax(masked_logits).item()
            
            x = int(best_action % 4)
            z = int(best_action // 4)
            apply_move(board, Move(x=x, z=z), current_player)
            
            # Gewinner-Überprüfung
            if check_winner(board, current_player):
                # B6_02: Wer war dieser aktuelle Spieler? Kandidat oder Champion?
                if (current_player == 1 and candidate_is_p1) or (current_player == 2 and not candidate_is_p1):
                    candidate_wins += 1
                else:
                    champion_wins += 1
                break
                
            # Unentschieden prüfen
            if not np.any(board == 0):
                draws += 1
                break
                
            # Spieler wechseln
            current_player = 2 if current_player == 1 else 1

    # B6_02: Winrate & Update Logik
    # Wir berechnen die Winrate des Kandidaten aus ALLEN Spielen.
    win_rate = candidate_wins / num_games
    
    print(f"--- Arena Ergebnis ---")
    print(f"Kandidat Siege: {candidate_wins} | Champion Siege: {champion_wins} | Remis: {draws}")
    print(f"Winrate des Kandidaten: {win_rate:.1%}")
    
    if win_rate >= win_threshold:
        print("RESULTAT: Kandidat ist der neue Champion!")
        return True
    else:
        print("RESULTAT: Kandidat wurde abgelehnt. Champion verteidigt Titel.")
        return False
