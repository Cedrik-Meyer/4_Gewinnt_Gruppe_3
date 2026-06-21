"""
tools/strong_engine.py

Eine extrem starke, zeitgesteuerte Minimax-Engine mit Alpha-Beta-Pruning.
Nutzt Multiprocessing (Root Splitting), um alle verfügbaren CPU-Kerne 
für die Berechnung zu verwenden. Schließt die Prozesse nach jedem Zug sauber ab.
"""

import time
import numpy as np
import multiprocessing as mp

from shared.data_structures import Move
from shared.game_logic import apply_move, check_winner
from shared.state_encoder import get_legal_mask

class TimeOutException(Exception):
    """Wird geworfen, wenn das Zeitlimit innerhalb eines Worker-Prozesses abläuft."""
    pass

# ==============================================================================
# 1. WORKER-LOGIK (Muss auf Modul-Ebene sein, damit Windows/Mac es parallelisieren kann)
# ==============================================================================

# Move Ordering: Zentrum zuerst prüfen, beschleunigt das Alpha-Beta-Pruning massiv.
CENTER_FIRST_MOVES = [5, 6, 9, 10, 1, 2, 4, 7, 8, 11, 13, 14, 0, 3, 12, 15]

def evaluate_board(board: np.ndarray, player: int) -> float:
    """Statische Blatt-Bewertung: Fokus auf das 2x2x2 Zentrum im 3D-Raum."""
    score = 0
    opponent = 2 if player == 1 else 1
    # Das innere Zentrum (y, z, x von 1 bis 2)
    for y in range(1, 3):
        for z in range(1, 3):
            for x in range(1, 3):
                val = board[y][z][x]
                if val == player: score += 10
                elif val == opponent: score -= 10
    return score

def minimax_worker(board: np.ndarray, depth: int, alpha: float, beta: float, 
                   maximizingPlayer: bool, original_player: int, end_time: float) -> float:
    """Die rekursive Minimax-Funktion, die auf den einzelnen Kernen läuft."""
    
    # 1. Zeit-Check: Wenn die Zeit abgelaufen ist, werfen wir sofort den Anker!
    if time.time() > end_time:
        raise TimeOutException()
        
    opponent = 2 if original_player == 1 else 1
    current_player = original_player if maximizingPlayer else opponent
    prev_player = opponent if maximizingPlayer else original_player
    
    # 2. Sieg-Erkennung (Je schneller der Sieg, desto höher die Punkte)
    if check_winner(board, prev_player):
        return -10000 - depth if maximizingPlayer else 10000 + depth
        
    legal_mask = get_legal_mask(board)
    legal_moves = [i for i in range(16) if legal_mask[i] == 1.0]
    
    # 3. Abbruchbedingung (Tiefe 0 oder Brett voll)
    if depth == 0 or len(legal_moves) == 0:
        return evaluate_board(board, original_player)
        
    ordered_moves = [m for m in CENTER_FIRST_MOVES if m in legal_moves]

    # 4. Rekursion (Alpha-Beta-Pruning)
    if maximizingPlayer:
        maxEval = -float('inf')
        for action in ordered_moves:
            test_board = np.copy(board)
            apply_move(test_board, Move(x=action % 4, z=action // 4), current_player)
            
            eval = minimax_worker(test_board, depth - 1, alpha, beta, False, original_player, end_time)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break # Beta-Cutoff (Ast abschneiden)
        return maxEval
    else:
        minEval = float('inf')
        for action in ordered_moves:
            test_board = np.copy(board)
            apply_move(test_board, Move(x=action % 4, z=action // 4), current_player)
            
            eval = minimax_worker(test_board, depth - 1, alpha, beta, True, original_player, end_time)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break # Alpha-Cutoff (Ast abschneiden)
        return minEval

def evaluate_single_move_task(args):
    """
    Die Task-Funktion für den Pool. Nimmt EINEN legalen Top-Level-Zug,
    führt ihn aus und berechnet den Minimax-Baum für diesen Ast.
    """
    board, first_move, player, depth, end_time = args
    
    test_board = np.copy(board)
    apply_move(test_board, first_move, player)
    
    # Wenn der Zug sofort gewinnt, geben wir maximale Punktzahl zurück
    if check_winner(test_board, player):
        return (first_move, 20000) # 20000 ist höher als jeder Minimax-Sieg
        
    try:
        score = minimax_worker(
            board=test_board, 
            depth=depth - 1, 
            alpha=-float('inf'), 
            beta=float('inf'), 
            maximizingPlayer=False, 
            original_player=player, 
            end_time=end_time
        )
        return (first_move, score)
    except TimeOutException:
        # Die Zeit ist abgelaufen, dieses Ergebnis ist unvollständig!
        return (first_move, None)


# ==============================================================================
# 2. HAUPT-ENGINE (Orchestriert das Multiprocessing)
# ==============================================================================

class StrongEngine:
    """
    Die Multiprocessing Minimax-Engine.
    """
    def __init__(self):
        self.name = "StrongEngine (Multicore Alpha-Beta)"

    def get_engine_move(self, board: np.ndarray, player: int, time_limit_ms: int, num_cores: int) -> Move:
        """
        Zustandslose Funktion: Öffnet einen Pool, berechnet den besten Zug, schließt den Pool.
        """
        # Puffer abziehen (5%), da das Schließen des Pools und Interprozess-Kommunikation minimal Zeit kostet
        end_time = time.time() + (time_limit_ms / 1000.0) * 0.95
        
        legal_mask = get_legal_mask(board)
        legal_moves = [i for i in range(16) if legal_mask[i] == 1.0]
        
        # Sortieren nach Zentrum (wichtig für den Fall, dass die Zeit früh abläuft)
        ordered_actions = [m for m in CENTER_FIRST_MOVES if m in legal_moves]
        
        if not ordered_actions:
            return Move(x=0, z=0)
            
        best_move_overall = Move(x=ordered_actions[0] % 4, z=ordered_actions[0] // 4)
        
        # Multiprocessing Pool erstellen (wird durch das 'with' sauber geschlossen!)
        # Wir limitieren die Kerne auf das Maximum der legalen Züge
        active_cores = min(num_cores, len(ordered_actions), mp.cpu_count())
        active_cores = max(1, active_cores)
        
        with mp.Pool(processes=active_cores) as pool:
            # Iterative Deepening: Wir fangen bei Tiefe 1 an und gehen bis max 20
            for current_depth in range(1, 21):
                
                # Wenn schon VOR dem Start der neuen Ebene die Zeit knapp ist -> Abbruch
                if time.time() >= end_time:
                    break
                    
                # Vorbereiten der Argumente für die Worker
                tasks = []
                for action in ordered_actions:
                    move = Move(x=action % 4, z=action // 4)
                    tasks.append((board, move, player, current_depth, end_time))
                    
                # Verteile die Aufgaben auf die Kerne
                results = pool.map(evaluate_single_move_task, tasks)
                
                # Checken, ob ein Timeout aufgetreten ist (None im Ergebnis)
                timeout_occurred = any(score is None for move, score in results)
                
                if timeout_occurred:
                    # Diese Tiefen-Ebene ist unvollständig. Wir verwerfen sie.
                    break
                    
                # Wenn alle Worker fertig sind, werten wir die Ebene aus
                best_score_this_depth = -float('inf')
                best_move_this_depth = None
                
                for move, score in results:
                    if score > best_score_this_depth:
                        best_score_this_depth = score
                        best_move_this_depth = move
                        
                # Wir haben eine komplette Ebene erfolgreich berechnet!
                if best_move_this_depth:
                    best_move_overall = best_move_this_depth
                    
                # Wenn wir einen garantierten "Forcing-Sieg" gefunden haben, können wir stoppen
                if best_score_this_depth > 9000:
                    break
                    
        # Der 'with'-Block ist hier zu Ende -> Pool wird sofort zerstört, CPUs sind wieder frei!
        return best_move_overall
