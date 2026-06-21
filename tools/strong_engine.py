"""
tools/strong_engine.py

Eine hyper-optimierte Minimax-Engine mit Alpha-Beta-Pruning, 
Iterative Deepening und Transposition Table.
Dies ist ein "Anytime-Algorithmus", der solange immer tiefer in den Baum 
rechnet, bis das vorgegebene Zeitlimit (z.B. 180ms) exakt abgelaufen ist.

WICHTIG: Diese Engine läuft absichtlich auf einem Single-Core! 
Beim klassischen Alpha-Beta-Pruning frisst der Python-Multiprocessing-Overhead 
(IPC) bei kurzen Zeitlimits (180ms) zu viel Zeit, wodurch eine Single-Core 
Variante mit geteilter Transposition-Table massiv tiefer rechnen kann.
"""

import time
import numpy as np
from shared.data_structures import Move
from shared.game_logic import apply_move, check_winner
from shared.state_encoder import get_legal_mask

class TimeOutException(Exception):
    """Wird geworfen, wenn das Zeitlimit während der rekursiven Suche abläuft."""
    pass

class StrongEngine:
    def __init__(self):
        self.name = "StrongEngine (Single-Core Alpha-Beta)"
        # Move Ordering: Zentrum-Züge zuerst prüfen. 
        # Das beschleunigt Alpha-Beta-Pruning extrem!
        self.center_first = [5, 6, 9, 10, 1, 2, 4, 7, 8, 11, 13, 14, 0, 3, 12, 15]
        
        self.start_time = 0
        self.time_limit = 0
        
        # Transposition Table: Speichert (Tiefe, Bewertung) für bereits gesehene Bretter
        self.transposition_table = {}

    def check_time(self):
        """Prüft, ob die Zeit abgelaufen ist und bricht den Baum knallhart ab."""
        if time.time() - self.start_time > self.time_limit:
            raise TimeOutException()

    def evaluate_board(self, board: np.ndarray, player: int) -> float:
        """Statische Blatt-Bewertung: Fokus auf das 2x2x2 Zentrum."""
        score = 0
        opponent = 2 if player == 1 else 1
        for y in range(1, 3):
            for z in range(1, 3):
                for x in range(1, 3):
                    val = board[y][z][x]
                    if val == player: score += 10
                    elif val == opponent: score -= 10
        return score

    def get_engine_move(self, board: np.ndarray, player: int, time_limit_ms: int = 180, num_cores: int = 1) -> Move:
        """
        Iterative Deepening: Rechnet Tiefe 1, dann 2, dann 3... bis die Uhr abläuft.
        Das Argument 'num_cores' wird für Kompatibilität mit der API akzeptiert, aber ignoriert.
        """
        self.start_time = time.time()
        # 5% Puffer abziehen, damit der Abbruch sicher im Zeitfenster passiert
        self.time_limit = (time_limit_ms / 1000.0) * 0.95 
        
        self.transposition_table.clear() # RAM vor jedem neuen Zug leeren

        legal_mask = get_legal_mask(board)
        legal_moves = [i for i in range(16) if legal_mask[i] == 1.0]
        ordered_moves = [m for m in self.center_first if m in legal_moves]

        if not ordered_moves:
            return Move(x=0, z=0)

        best_move_overall = Move(x=ordered_moves[0] % 4, z=ordered_moves[0] // 4)
        
        try:
            # Iterative Tiefensuche (Wir versuchen so tief zu kommen, wie die Zeit erlaubt)
            for current_max_depth in range(1, 30): 
                alpha = -float('inf')
                beta = float('inf')
                best_score = -float('inf')
                best_move_for_depth = best_move_overall
                
                for action in ordered_moves:
                    self.check_time()
                    move = Move(x=action % 4, z=action // 4)
                    test_board = np.copy(board)
                    apply_move(test_board, move, player)
                    
                    # Sofort-Sieg auf Ebene 0
                    if check_winner(test_board, player):
                        return move 
                        
                    score = self.minimax(test_board, current_max_depth - 1, alpha, beta, False, player)
                    
                    if score > best_score:
                        best_score = score
                        best_move_for_depth = move
                    
                    alpha = max(alpha, score)
                    
                # Tiefe komplett abgeschlossen! Wir speichern diesen Zug als aktuell besten.
                best_move_overall = best_move_for_depth
                
                # Wenn wir eine Forcing-Line zum Sieg gefunden haben, stoppen wir sofort
                if best_score > 9000:
                    break
                    
        except TimeOutException:
            # Die Zeit ist mitten in einer Tiefe abgelaufen. 
            # Wir ignorieren die halbfertige Ebene und geben den besten Zug der LETZTEN fertigen Tiefe zurück.
            pass 
            
        return best_move_overall

    def minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, maximizingPlayer: bool, original_player: int) -> float:
        self.check_time()
        
        # Transposition Table Lookup (Haben wir das Brett schon berechnet?)
        board_hash = board.tobytes()
        if board_hash in self.transposition_table:
            stored_depth, stored_eval = self.transposition_table[board_hash]
            if stored_depth >= depth:
                return stored_eval
        
        opponent = 2 if original_player == 1 else 1
        current_player = original_player if maximizingPlayer else opponent
        prev_player = opponent if maximizingPlayer else original_player
        
        if check_winner(board, prev_player):
            # Sieg in weniger Zügen ist besser (deshalb addieren/subtrahieren wir die Tiefe)
            final_eval = -10000 - depth if maximizingPlayer else 10000 + depth
            self.transposition_table[board_hash] = (depth, final_eval)
            return final_eval
            
        legal_mask = get_legal_mask(board)
        legal_moves = [i for i in range(16) if legal_mask[i] == 1.0]
        
        if depth == 0 or len(legal_moves) == 0:
            final_eval = self.evaluate_board(board, original_player)
            self.transposition_table[board_hash] = (depth, final_eval)
            return final_eval
            
        ordered_moves = [m for m in self.center_first if m in legal_moves]

        if maximizingPlayer:
            maxEval = -float('inf')
            for action in ordered_moves:
                test_board = np.copy(board)
                apply_move(test_board, Move(x=action % 4, z=action // 4), current_player)
                eval = self.minimax(test_board, depth - 1, alpha, beta, False, original_player)
                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.transposition_table[board_hash] = (depth, maxEval)
            return maxEval
        else:
            minEval = float('inf')
            for action in ordered_moves:
                test_board = np.copy(board)
                apply_move(test_board, Move(x=action % 4, z=action // 4), current_player)
                eval = self.minimax(test_board, depth - 1, alpha, beta, True, original_player)
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.transposition_table[board_hash] = (depth, minEval)
            return minEval
