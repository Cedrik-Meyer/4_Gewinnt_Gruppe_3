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
    