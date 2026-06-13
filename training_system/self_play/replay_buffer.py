"""
training_system/self_play/replay_buffer.py

Speichert die simulierten Spiele (Erfahrungen) des Agenten.
Dient als kontinuierlicher Datensatz für das Training des neuronalen Netzes.
"""

from collections import deque
from typing import Any

class ReplayBuffer:
    """
    Ein Ringpuffer (First-In-First-Out) mit fester Maximalkapazität für ML-Training.
    Wenn der Puffer voll ist, werden die ältesten Erinnerungen automatisch überschrieben.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialisiert den Buffer.
        
        Args:
            capacity (int): Die maximale Anzahl an Zügen, die im RAM gehalten werden.
                            Standard: 100.000 Züge.
        """
        # deque mit maxlen ist ein nativer Ringpuffer in C-Geschwindigkeit.
        # Wenn das Limit erreicht ist und rechts ein neues Element angehängt wird, 
        # fällt das älteste Element auf der linken Seite automatisch heraus.
        self.buffer = deque(maxlen=capacity)

    def push(self, state: Any, action_probs: Any, value: float):
        """
        Speichert einen einzelnen Spielzug (eine Erfahrung) im Puffer.
        
        Args:
            state: Das codierte Board (Tensor oder Numpy-Array aus Ebene C).
            action_probs: Die Ziel-Wahrscheinlichkeiten für die 16 Züge (vom MCTS berechnet).
            value: Der finale Ausgang des Spiels aus Sicht des Spielers (+1.0, 0.0, -1.0).
        """
        # Wir speichern die Erfahrung als einfaches Tuple
        self.buffer.append((state, action_probs, value))

    def __len__(self) -> int:
        """
        Gibt die aktuelle Anzahl der gespeicherten Züge zurück.
        Ermöglicht den Aufruf von len(buffer) im restlichen Code.
        """
        return len(self.buffer)