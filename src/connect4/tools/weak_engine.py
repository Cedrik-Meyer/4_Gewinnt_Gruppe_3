"""
tools/weak_engine.py

Eine einfache, heuristische Engine für 3D-Connect4.
Dient als "Leichter Gegner" für Menschen oder als Baseline-Test für KIs.
Die Engine blockt offensichtliche 1-Step-Bedrohungen und nimmt 1-Step-Siege mit,
spielt ansonsten aber völlig zufällig und baut keine tiefen Fallen auf.
"""

import numpy as np
from shared.data_structures import Move
from shared.game_logic import apply_move, check_winner
from shared.state_encoder import get_legal_mask

class WeakEngine:
    """
    Die einfache Engine. 
    Suchtiefe: 1 Ebene (Prüft nur den unmittelbaren nächsten Zug).
    """
    def __init__(self):
        self.name = "WeakEngine (1-Step)"

    def get_engine_move(self, board: np.ndarray, player: int) -> Move:
        """
        Berechnet den nächsten Zug basierend auf simplen Überlebensregeln.
        """
        opponent = 2 if player == 1 else 1
        
        # 1. Alle legalen Züge (0-15) ermitteln
        legal_mask = get_legal_mask(board)
        legal_moves = [i for i in range(16) if legal_mask[i] == 1.0]
        
        if not legal_moves:
            # Fallback, falls das Board voll ist (sollte im normalen Spiel Loop abgefangen werden)
            return Move(x=0, z=0)

        # ---------------------------------------------------------
        # Regel 1: Sofort-Sieg (Gibt es einen Zug, der mich sofort gewinnen lässt?)
        # ---------------------------------------------------------
        for action in legal_moves:
            move = Move(x=action % 4, z=action // 4)
            test_board = np.copy(board)
            apply_move(test_board, move, player)
            
            if check_winner(test_board, player):
                return move

        # ---------------------------------------------------------
        # Regel 2: Blocken (Würde der Gegner mit einem dieser Züge gewinnen?)
        # ---------------------------------------------------------
        for action in legal_moves:
            move = Move(x=action % 4, z=action // 4)
            test_board = np.copy(board)
            apply_move(test_board, move, opponent)
            
            if check_winner(test_board, opponent):
                return move

        # ---------------------------------------------------------
        # Regel 3: Zufälliger Zug (Keine Strategie, reines Raten)
        # ---------------------------------------------------------
        # Um die Engine wirklich "schwach" zu machen, verzichten wir hier 
        # absichtlich auf eine Zentrumskontrolle oder tiefere Vorausberechnung.
        random_action = np.random.choice(legal_moves)
        x = int(random_action % 4)
        z = int(random_action // 4)
        
        return Move(x=x, z=z)
