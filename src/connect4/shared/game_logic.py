"""
Enthält die zentralen Spielregeln, die Spielfeld-Initialisierung
und die Gewinnerkennung für den 3D Connect4 Agenten.
"""

import numpy as np
from shared.data_structures import Move


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

def check_winner_straight_axes(board: np.ndarray, player_value: int) -> bool:
    """
    Prüft, ob der übergebene Spieler eine 4er-Reihe entlang der geraden
    X- (Breite), Y- (Höhe) oder Z-Achsen (Tiefe) gebildet hat.
    
    Args:
        board (np.ndarray): Das 3D-Spielfeld mit Shape [4, 4, 4] und Format [y][z][x].
        player_value (int): Der Zellwert des Spielers (z. B. 1 oder 2).
        
    Returns:
        bool: True, wenn eine gerade 4er-Reihe gefunden wurde, sonst False.
    """
    # Erzeugt eine boolesche Maske (True wo der Spieler Steine hat, sonst False)
    player_mask = (board == player_value)
    
    # 1. Vertikale Gewinnerkennung (Entlang der Y-Achse / Höhe)
    # axis=0 reduziert die Y-Dimension. Wenn in einer (z, x) Säule alle 4 Werte True sind,
    # ergibt np.all dort True. np.any prüft, ob es überhaupt so eine Säule gibt.
    if np.any(np.all(player_mask, axis=0)):
        return True
        
    # 2. Tiefen-Gewinnerkennung (Entlang der Z-Achse / Tiefe)
    # axis=1 reduziert die Z-Dimension. Prüft alle geraden Reihen nach hinten.
    if np.any(np.all(player_mask, axis=1)):
        return True
        
    # 3. Horizontale Gewinnerkennung (Entlang der X-Achse / Breite)
    # axis=2 reduziert die X-Dimension. Prüft alle geraden Reihen nach rechts/links.
    if np.any(np.all(player_mask, axis=2)):
        return True
        
    return False

def check_winner_2d_diagonals(board: np.ndarray, player_value: int) -> bool:
    """
    Prüft, ob der übergebene Spieler eine 4er-Reihe auf einer der 2D-Diagonalen gebildet hat.
    Das bedeutet: Eine Diagonale, die flach auf einer bestimmten XY-, YZ- oder ZX-Ebene liegt.
    
    Args:
        board (np.ndarray): Das 3D-Spielfeld.
        player_value (int): Der Zellwert des Spielers (z. B. 1 oder 2).
        
    Returns:
        bool: True, wenn eine 2D-Diagonale gefunden wurde, sonst False.
    """
    player_mask = (board == player_value)
    
    # 1. XY-Ebenen prüfen (Wir schneiden den Würfel entlang der Z-Achse in 4 Scheiben)
    for z in range(4):
        plane = player_mask[:, z, :]  # Holt eine 4x4 Wand (Y und X)
        if np.all(np.diagonal(plane)) or np.all(np.diagonal(np.fliplr(plane))):
            return True
            
    # 2. YZ-Ebenen prüfen (Wir schneiden den Würfel entlang der X-Achse in 4 Scheiben)
    for x in range(4):
        plane = player_mask[:, :, x]  # Holt eine 4x4 Wand (Y und Z)
        if np.all(np.diagonal(plane)) or np.all(np.diagonal(np.fliplr(plane))):
            return True
            
    # 3. ZX-Ebenen prüfen (Wir schneiden den Würfel entlang der Y-Achse in 4 Scheiben)
    # Das entspricht flachen Diagonalen den einzelnen "Ebenen".
    for y in range(4):
        plane = player_mask[y, :, :]  # Holt einen 4x4 Boden (Z und X)
        if np.all(np.diagonal(plane)) or np.all(np.diagonal(np.fliplr(plane))):
            return True
            
    return False

def check_winner_3d_diagonals(board: np.ndarray, player_value: int) -> bool:
    """
    Prüft, ob der übergebene Spieler eine der 4 echten Raumdiagonalen 
    quer durch das Zentrum des 3D-Würfels gebildet hat.
    
    Args:
        board (np.ndarray): Das 3D-Spielfeld.
        player_value (int): Der Zellwert des Spielers (z. B. 1 oder 2).
        
    Returns:
        bool: True, wenn eine 3D-Raumdiagonale gefunden wurde, sonst False.
    """
    # Da wir wissen, dass die Kantenlänge exakt 4 ist, können wir die 4 Linien
    # direkt mathematisch über die Indizes abgreifen. Das ist extrem schnell.
    
    # 1. Raumdiagonale: Von (0,0,0) nach (3,3,3)
    if all(board[i, i, i] == player_value for i in range(4)):
        return True
        
    # 2. Raumdiagonale: Von (0,0,3) nach (3,3,0)
    if all(board[i, i, 3 - i] == player_value for i in range(4)):
        return True
        
    # 3. Raumdiagonale: Von (0,3,0) nach (3,0,3)
    if all(board[i, 3 - i, i] == player_value for i in range(4)):
        return True
        
    # 4. Raumdiagonale: Von (0,3,3) nach (3,0,0)
    if all(board[i, 3 - i, 3 - i] == player_value for i in range(4)):
        return True
        
    return False


def check_winner(board: np.ndarray, player_value: int) -> bool:
    """
    Die zentrale Hauptfunktion zur Gewinnerkennung.
    Überprüft das gesamte Board auf eine gültige 4er-Reihe für den Spieler.
    Geprüft werden: Gerade Achsen, flache 2D-Diagonalen und 3D-Raumdiagonalen.
    
    Args:
        board (np.ndarray): Das 3D-Spielfeld mit Shape [4, 4, 4].
        player_value (int): Der Zellwert des Spielers (1 oder 2).
        
    Returns:
        bool: True, wenn der Spieler gewonnen hat, sonst False.
    """
    # Wir nutzen die Kurzschluss-Auswertung (Short-Circuit Evaluation) von Python:
    # Wenn die geraden Achsen schon True sind, werden die rechenintensiveren 
    # Diagonalen gar nicht erst ausgeführt. Das spart massiv CPU-Zyklen beim Training!
    return (
        check_winner_straight_axes(board, player_value) or
        check_winner_2d_diagonals(board, player_value) or
        check_winner_3d_diagonals(board, player_value)
    )
