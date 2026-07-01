"""
tools/weak_engine.py

Eine heuristische Basis-Engine für 3D-Connect4 (Baseline-Agent).
Dient als Benchmark/leichter Gegner für die trainierten Modelle in play_terminal.py

Verhaltensprofil:
- Suchtiefe: 1 (Bewertet ausschließlich den unmittelbar nächsten Zustand).
- Vermeidet triviale Niederlagen (zwingender Block).
- Nutzt direkte Gewinnchancen (zwingender Sieg).
- Spielt ansonsten rein stochastisch (ohne Zentrumskontrolle oder Taktik).
"""

import numpy as np

from shared.data_structures import Move
from shared.game_logic import apply_move, check_winner
from shared.state_encoder import get_legal_mask


class WeakEngine:
    """
    Die deterministisch-stochastische Basis-Engine.
    """
    
    def __init__(self):
        self.name = "WeakEngine (1-Step)"

    def get_engine_move(self, board: np.ndarray, player: int) -> Move:
        """
        Berechnet den nächsten Spielzug basierend auf einer strikten Prioritätenfolge
        aus Überlebensregeln und Zufall.
        
        Args:
            board (np.ndarray): Der aktuelle 3D-Spielzustand (Shape: [4, 4, 4]).
            player (int): Die ID des Spielers, der am Zug ist (1 oder 2).
            
        Returns:
            Move: Das berechnete Zug-Objekt mit (x, z) Koordinaten.
        """
        opponent = 2 if player == 1 else 1
        
        # 1. Ermittlung des legalen Aktionsraums
        legal_mask = get_legal_mask(board)
        legal_moves = [i for i in range(16) if legal_mask[i] == 1.0]
        
        # Fallback-Sicherheit, falls das Board bereits voll ist
        if not legal_moves:
            return Move(x=0, z=0)

        # ---------------------------------------------------------
        # Regel 1: Offensive (Sofortigen Sieg erzwingen)
        # ---------------------------------------------------------
        # Simuliert alle legalen Züge für den eigenen Spieler. 
        # Führt einer zum direkten Sieg, wird die Suche sofort beendet.
        for action in legal_moves:
            move = Move(x=action % 4, z=action // 4)
            test_board = np.copy(board)
            apply_move(test_board, move, player)
            
            if check_winner(test_board, player):
                return move

        # ---------------------------------------------------------
        # Regel 2: Defensive (Sofortige Niederlage abwenden)
        # ---------------------------------------------------------
        # Simuliert alle legalen Züge aus der Perspektive des Gegners.
        # Würde der Gegner mit einem dieser Züge gewinnen, wird genau dort geblockt.
        for action in legal_moves:
            move = Move(x=action % 4, z=action // 4)
            test_board = np.copy(board)
            apply_move(test_board, move, opponent)
            
            if check_winner(test_board, opponent):
                return move

        # ---------------------------------------------------------
        # Regel 3: Stochastische Auswahl (Zufallszug)
        # ---------------------------------------------------------
        # Keine unmittelbare Gefahr und kein direkter Sieg in Sicht.
        # Um die Engine als "leichten" Gegner zu definieren, wird hier absichtlich 
        # auf Positionsbewertungen (Value) oder Baumsuche (MCTS) verzichtet.
        random_action = np.random.choice(legal_moves)
        x = int(random_action % 4)
        z = int(random_action // 4)
        
        return Move(x=x, z=z)
