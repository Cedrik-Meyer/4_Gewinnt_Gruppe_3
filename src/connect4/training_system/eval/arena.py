"""
training_system/eval/arena.py

Das Testgelände für neu trainierte Modelle.
Lässt das amtierende Champion-Modell gegen das neu trainierte Kandidaten-Modell antreten.
"""

import torch
import numpy as np
from shared.data_structures import Move
from shared.game_logic import create_empty_board, apply_move, check_winner
from shared.state_encoder import encode_state, get_legal_mask
from training_system.neural_network.model import Connect4Model

def evaluate_candidate(champion: Connect4Model, candidate: Connect4Model, 
                       num_games: int = 100, win_threshold: float = 0.55,
                       verbose: bool = False) -> bool:
    """
    Lässt Champion und Kandidat N Partien gegeneinander spielen.
    Gibt True zurück, wenn der Kandidat die definierte Gewinnrate erreicht hat.
    """
    champion.eval()
    candidate.eval()
    
    candidate_wins = 0
    champion_wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        board = create_empty_board()
        current_player = 1
        
        # Fairness-Regelung: Jeder startet in der Hälfte der Spiele als Spieler 1
        if game_idx < (num_games // 2):
            p1_model = candidate
            p2_model = champion
            candidate_is_p1 = True
        else:
            p1_model = champion
            p2_model = candidate
            candidate_is_p1 = False
            
        while True:
            # Auswahl des Modells für den aktuell ziehenden Spieler
            active_model = p1_model if current_player == 1 else p2_model
            
            # Zustand für das Netz codieren
            player_slot = current_player - 1
            state_tensor = encode_state(board, player_slot)
            legal_mask = get_legal_mask(board)
            
            # Inferenz ohne Gradientenberechnung
            with torch.no_grad():
                logits, _ = active_model(state_tensor.unsqueeze(0))
            logits = logits.squeeze(0)
            
            # Maskierung illegaler Züge mit einem sehr großen negativen Wert
            mask_tensor = torch.tensor(legal_mask, dtype=torch.float32)
            masked_logits = logits + (1.0 - mask_tensor) * -1e9
            
            # Für die Evaluation wird eine deterministische Auswahl getroffen.
            # Da die absolute Spielstärke gemessen werden soll, wird der Zug mit 
            # der höchsten Wahrscheinlichkeit (Argmax) ohne stochastische Auswahl gewählt.
            best_action = torch.argmax(masked_logits).item()
            
            x = int(best_action % 4)
            z = int(best_action // 4)
            apply_move(board, Move(x=x, z=z), current_player)
            
            # Überprüfung auf Sieg nach dem Zug
            if check_winner(board, current_player):
                # Identifikation des Gewinners basierend auf der Spielerzuweisung in dieser Partie
                if (current_player == 1 and candidate_is_p1) or (current_player == 2 and not candidate_is_p1):
                    candidate_wins += 1
                else:
                    champion_wins += 1
                break
                
            # Überprüfung auf Unentschieden bei vollem Spielfeld
            if not np.any(board == 0):
                draws += 1
                break
                
            # Wechsel des Spielers
            current_player = 2 if current_player == 1 else 1

    # Berechnung der Gewinnrate des Kandidaten über alle Partien
    win_rate = candidate_wins / num_games
    
    if verbose:
        print(f"--- Arena Ergebnis ---")
        print(f"Kandidat Siege: {candidate_wins} | Champion Siege: {champion_wins} | Remis: {draws}")
        print(f"Gewinnrate des Kandidaten: {win_rate:.1%}")
        if win_rate >= win_threshold:
            print("RESULTAT: Kandidat ist der neue Champion.")
        else:
            print("RESULTAT: Kandidat wurde abgelehnt. Champion verteidigt Titel.")
            
    return win_rate >= win_threshold
