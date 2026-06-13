"""
training_system/self_play/replay_buffer.py

Speichert die simulierten Spiele (Erfahrungen) des Agenten.
Dient als kontinuierlicher Datensatz für das Training des neuronalen Netzes.
"""

import random
from collections import deque
from typing import Any, Tuple, List

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
        self.buffer = deque(maxlen=capacity)

    def push(self, state: Any, action_probs: Any, value: float):
        """
        Speichert einen einzelnen Spielzug (eine Erfahrung) im Puffer.
        
        Args:
            state: Das codierte Board (Tensor oder Numpy-Array aus Ebene C).
            action_probs: Die Ziel-Wahrscheinlichkeiten für die 16 Züge (vom MCTS berechnet).
            value: Der finale Ausgang des Spiels aus Sicht des Spielers (+1.0, 0.0, -1.0).
        """
        self.buffer.append((state, action_probs, value))

    def sample_batch(self, batch_size: int = 64) -> Tuple[List[Any], List[Any], List[float]]:
        """
        Zieht eine zufällige Stichprobe (Mini-Batch) von alten Zügen aus dem Puffer.
        
        Args:
            batch_size (int): Die Anzahl der Züge, die für einen Trainingsschritt 
                              benötigt werden (Standard: 64).
                              
        Returns:
            Tuple[List, List, List]: Drei separate Listen für States, Action-Probs und Values.
            
        Raises:
            ValueError: Wenn der Puffer weniger Elemente enthält als die angeforderte batch_size.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Nicht genug Daten im Buffer. Erwartet: {batch_size}, Aktuell: {len(self.buffer)}"
            )
            
        # random.sample zieht eine Stichprobe OHNE Zurücklegen (kein Element wird doppelt gezogen)
        batch = random.sample(self.buffer, batch_size)
        
        # Entpackt die Liste von Tupeln: [(s1, a1, v1), (s2, a2, v2), ...]
        # in drei separate Listen: [s1, s2, ...], [a1, a2, ...], [v1, v2, ...]
        states, action_probs, values = zip(*batch)
        
        return list(states), list(action_probs), list(values)

    def __len__(self) -> int:
        """
        Gibt die aktuelle Anzahl der gespeicherten Züge zurück.
        """
        return len(self.buffer)
    