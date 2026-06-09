"""
tests/unit/shared/test_game_logic.py

Unit-Tests für die physikalischen Spielregeln (Drop-Mechanik)
und die Gewinnerkennung.
"""

import pytest
import numpy as np
from shared.data_structures import Move
from shared.game_logic import create_empty_board, apply_move

def test_apply_move_stacking():
    """Prüft, ob Steine in derselben Säule korrekt übereinander gestapelt werden."""
    # Arrange
    board = create_empty_board()
    player1 = 1
    player2 = 2
    
    # Act: Erster Stein fällt nach ganz unten (y=0)
    move1 = Move(x=1, z=1)
    apply_move(board, move1, player1)
    
    # Act: Zweiter Stein fällt genau auf den ersten (y=1)
    move2 = Move(x=1, z=1)
    apply_move(board, move2, player2)
    
    # Assert: Werte müssen korrekt im Array liegen
    assert board[0][1][1] == player1
    assert board[1][1][1] == player2
    
    # Assert: Die y-Koordinaten in den Move-Objekten müssen von resolve_y befüllt worden sein
    assert move1.y == 0
    assert move2.y == 1


def test_apply_move_full_column_raises_error():
    """Prüft, ob ein Fehler geworfen wird, wenn man in eine volle Säule werfen will."""
    # Arrange
    board = create_empty_board()
    player = 1
    
    # Act: Wir füllen die Säule an Position x=2, z=2 komplett auf (4 Steine)
    for _ in range(4):
        apply_move(board, Move(x=2, z=2), player)
        
    # Assert: Der 5. Stein MUSS mit einem ValueError abgelehnt werden
    with pytest.raises(ValueError, match="Illegaler Zug"):
        apply_move(board, Move(x=2, z=2), player)


def test_apply_move_out_of_bounds_raises_error():
    """Prüft, ob Koordinaten außerhalb des 4x4 Rasters blockiert werden."""
    # Arrange
    board = create_empty_board()
    player = 1
    
    # Assert: x=4 existiert nicht (erlaubt sind nur 0, 1, 2, 3)
    with pytest.raises(ValueError, match="Ungültige Koordinaten"):
        apply_move(board, Move(x=4, z=1), player)

from shared.game_logic import check_winner

def test_winner_straight_vertical():
    """Testet einen Sieg durch einen 4er-Turm (Y-Achse)."""
    board = create_empty_board()
    player = 1
    
    # 4 Steine übereinander an x=0, z=0
    board[0][0][0] = player
    board[1][0][0] = player
    board[2][0][0] = player
    board[3][0][0] = player
    
    assert check_winner(board, player) is True
    # Wichtig: Der Gegner darf diesen Sieg nicht fälschlicherweise zugesprochen bekommen
    assert check_winner(board, 2) is False

def test_winner_2d_diagonal():
    """Testet einen Sieg auf einer flachen 2D-Ebene (Treppenmuster auf dem Boden)."""
    board = create_empty_board()
    player = 2
    
    # Eine Diagonale flach auf dem Boden (y=0) von links-vorne nach rechts-hinten
    board[0][0][0] = player
    board[0][1][1] = player
    board[0][2][2] = player
    board[0][3][3] = player
    
    assert check_winner(board, player) is True

def test_winner_3d_room_diagonal():
    """Testet die komplexeste Siegerkennung: Eine Raumdiagonale quer durch den Würfel."""
    board = create_empty_board()
    player = 1
    
    # Von unten-vorne-links (0,0,0) nach oben-hinten-rechts (3,3,3)
    board[0][0][0] = player
    board[1][1][1] = player
    board[2][2][2] = player
    board[3][3][3] = player
    
    assert check_winner(board, player) is True

def test_no_winner_if_interrupted():
    """Testet, dass eine 3er-Reihe oder eine unterbrochene 4er-Reihe KEIN Sieg ist."""
    board = create_empty_board()
    player = 1
    
    # Eine Reihe auf der X-Achse (Breite), die vom Gegner unterbrochen wird
    board[0][0][0] = player
    board[0][0][1] = player
    board[0][0][2] = 2       # Gegner blockiert!
    board[0][0][3] = player
    
    assert check_winner(board, player) is False