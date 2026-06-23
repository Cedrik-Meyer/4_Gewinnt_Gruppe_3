import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Move:
    """
    Repräsentiert einen Spielzug im 4-Gewinnt 3D.
    (Ebene E der Datenarchitektur)
    """
    x: int
    z: int
    y: Optional[int] = None  # Wird erst nach der Berechnung der Schwerkraft gesetzt

    def resolve_y(self, board: np.ndarray) -> int:
        """
        Berechnet die Drop-Mechanik (Schwerkraft) für diesen Zug.
        
        Args:
            board (np.ndarray): Das 3D-Spielfeld mit Shape [4][4][4] und Format [y][z][x].
            
        Returns:
            int: Die ermittelte y-Koordinate (0, 1, 2 oder 3).
            
        Raises:
            ValueError: Wenn die x/z-Koordinaten ungültig sind oder die Säule voll ist.
        """
        if not (0 <= self.x < 4 and 0 <= self.z < 4):
            raise ValueError(f"Ungültige Koordinaten: x={self.x}, z={self.z}. Nur 0-3 erlaubt.")

        # Von unten (y=0) nach oben (y=3) die niedrigste freie Ebene suchen
        for current_y in range(4):
            if board[current_y][self.z][self.x] == 0:
                self.y = current_y
                return current_y
                
        raise ValueError(f"Illegaler Zug: Die Säule an Position (x={self.x}, z={self.z}) ist bereits voll.")
    
@dataclass
class GameState:
    """
    Kapselt den gesamten aktuellen Spielzustand.
    (Ebene B der Datenarchitektur)
    """
    board: np.ndarray       # Shape [4, 4, 4], dtype=np.uint8 (0=leer, 1=playerSlot 0, 2=playerSlot 1)
    player_slot: int        # Eigener Slot (0 oder 1)
    current_player: int     # Wer laut Server gerade am Zug ist (0 oder 1)
    match_id: str           # UUID
    request_id: str         # UUID (wichtig für die Spiegelung in der Antwort)
    legal_mask: Optional[np.ndarray] = None  # Float32 Array der Größe 16
    deadline_ms: Optional[int] = None  # Unix-Zeit in ms, bis wann der Zug beim Server eingehen muss

    def __post_init__(self):
        """Wird nach der Initialisierung automatisch aufgerufen, um die Maske zu berechnen."""
        if self.legal_mask is None:
            self.update_legal_mask()

    def update_legal_mask(self):
        """
        Berechnet ein 1D-Array (Größe 16), das anzeigt, welche Spalten spielbar sind.
        1.0 = gültig (Säule hat noch Platz), 0.0 = ungültig (Säule ist voll).
        Index-Mapping: index = z * 4 + x
        """
        self.legal_mask = np.ones(16, dtype=np.float32)
        
        # Wir prüfen nur die oberste Ebene (y = 3) des Boards.
        # Wenn dort ein Stein liegt (!= 0), ist die komplette Säule darunter (aufgrund der Schwerkraft) voll.
        top_layer = self.board[3]
        
        for z in range(4):
            for x in range(4):
                if top_layer[z][x] != 0:
                    index = z * 4 + x
                    self.legal_mask[index] = 0.0
