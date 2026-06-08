import numpy as np
from shared.data_structures import Move

"""
Enthält die zentralen Spielregeln, die Spielfeld-Initialisierung
und die Gewinnerkennung für den 3D Connect4 Agenten.
"""

def create_empty_board() -> np.ndarray:
    """
    Generiert ein komplett leeres 3D-Spielfeld.
    Shape: [Y, Z, X] -> [4, 4, 4] (Höhe, Tiefe, Breite)
    Datentyp: uint8 (sehr speichereffizient für 0, 1, 2)
    
    Returns:
        np.ndarray: Ein 4x4x4 Array gefüllt mit Nullen.
    """
    return np.zeros((4, 4, 4), dtype=np.uint8)

def apply_move(board: np.ndarray, move: Move, player_value: int) -> np.ndarray:
    """
    Lässt einen Stein auf das 3D-Spielfeld fallen (Drop-Mechanik) und
    aktualisiert das Array in-place.
    
    Args:
        board (np.ndarray): Das aktuelle 3D-Spielfeld.
        move (Move): Das Zug-Objekt (enthält x und z).
        player_value (int): Der Zellwert, der gesetzt werden soll (z. B. 1 oder 2).
        
    Returns:
        np.ndarray: Das aktualisierte Spielfeld.
        
    Raises:
        ValueError: Wenn die Säule voll ist (wird von move.resolve_y geworfen).
    """
    # 1. Berechne die niedrigste freie Höhe (y) über die Schwerkraft-Logik im Move-Objekt
    y = move.resolve_y(board)
    
    # 2. Setze den Stein des Spielers auf die berechnete 3D-Koordinate
    board[y][move.z][move.x] = player_value
    
    # 3. Gebe das aktualisierte Board zurück
    return board
