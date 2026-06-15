"""
tests/unit/training/test_arena.py

Prueft die Funktionalitaet des Arena-Match-Runners und die Winrate-Logik.
"""

import pytest
from unittest.mock import patch
from training_system.neural_network.model import Connect4Model
from training_system.eval.arena import evaluate_candidate

def test_evaluate_candidate_execution_and_shapes():
    """
    Prueft, ob die Arena mit untrainierten Modellen absturzfrei durchlaeuft
    und einen booleschen Wert zurueckgibt.
    """
    champion = Connect4Model()
    candidate = Connect4Model()
    
    # Wir lassen nur 2 Spiele laufen, um Zeit zu sparen
    result = evaluate_candidate(champion, candidate, num_games=2, win_threshold=0.55)
    
    assert isinstance(result, bool), "Die Funktion muss True oder False zurueckgeben."

@patch('training_system.eval.arena.check_winner')
def test_evaluate_candidate_winrate_logic(mock_check_winner):
    """
    Simuliert deterministische Siege, um den Seitenwechsel und 
    die Threshold-Auswertung exakt zu pruefen.
    """
    champion = Connect4Model()
    candidate = Connect4Model()
    
    # Wir manipulieren check_winner: Der Spieler, der den ersten Zug macht, gewinnt sofort.
    mock_check_winner.return_value = True
    
    # Bei 2 Spielen bedeutet das:
    # Spiel 1: Kandidat ist Spieler 1 -> Er macht den ersten Zug -> Kandidat gewinnt.
    # Spiel 2: Champion ist Spieler 1 -> Er macht den ersten Zug -> Champion gewinnt.
    # Resultierende Winrate fuer den Kandidaten: exakt 50.0%
    
    # Szenario A: Threshold liegt bei strengen 55%
    # 50% < 55%, also muss der Kandidat abgelehnt werden (False).
    result_rejected = evaluate_candidate(champion, candidate, num_games=2, win_threshold=0.55)
    assert not result_rejected
    
    # Szenario B: Threshold liegt bei lockeren 40%
    # 50% >= 40%, also muss der Kandidat der neue Champion werden (True).
    result_accepted = evaluate_candidate(champion, candidate, num_games=2, win_threshold=0.40)
    assert result_accepted
