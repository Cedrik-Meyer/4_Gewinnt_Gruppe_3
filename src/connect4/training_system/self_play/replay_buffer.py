"""
training_system/self_play/replay_buffer.py

Das Experience Replay Memory für den Connect4-Agenten.
Speichert gespielte Züge, Netzwerk-Präferenzen und den finalen Spielausgang.
Dient als hochperformanter, kontinuierlich rotierender Datensatz (Ringpuffer)
für die GPU-Trainingsphase.
"""

import random
from collections import deque
from typing import Tuple, List, Union

import torch
import numpy as np


class ReplayBuffer:
    """
    Ein rotierender Ringpuffer (First-In-First-Out) mit einer harten Obergrenze.
    Das Limit garantiert, dass das Netz während des RL-Trainings stets von 
    seinen qualitativ besten (neuesten) Spielen lernt und veraltetes Anfängerwissen 
    automatisch aussortiert wird.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialisiert den Replay Buffer.
        
        Args:
            capacity (int): Maximale Anzahl an Zügen (Erfahrungen), die im RAM 
                            gehalten werden sollen.
        """
        # deque(maxlen=...) schiebt das älteste Element automatisch aus dem Puffer,
        # sobald die Maximalkapazität erreicht ist.
        self.buffer = deque(maxlen=capacity)

    def push(self, state: Union[torch.Tensor, np.ndarray], action_probs: np.ndarray, value: float):
        """
        Speichert einen einzelnen Datenpunkt (Transition) im Puffer.
        
        Args:
            state: Die 3D-Board-Repräsentation (Ebene C).
            action_probs: Die maskierte und vom MCTS/Self-Play korrigierte Zielverteilung 
                          für die 16 möglichen Spalten.
            value: Die Bewertung des Zugs (+1.0 Sieg, 0.0 Draw, -1.0 Niederlage).
        """
        self.buffer.append((state, action_probs, value))

    def sample_batch(self, batch_size: int = 64) -> Tuple[List[torch.Tensor], List[np.ndarray], List[float]]:
        """
        Zieht einen zufälligen Mini-Batch aus dem Puffer für die GPU-Backpropagation.
        Das stochastische Ziehen aus dem gesamten Puffer (Decorrelation) bricht die chronologische
        Abhängigkeit der aufeinanderfolgenden Züge einer Partie auf, was extremes 
        Overfitting verhindert und das Lernen verallgemeinert.
        
        Args:
            batch_size (int): Die Anzahl der benötigten Datenpunkte für den Forward/Backward Pass.
                              
        Returns:
            Tuple[List, List, List]: 
                Drei aufgespaltene Listen (States, Action_Probs, Values), fertig 
                zur Weiterverarbeitung als PyTorch Tensoren.
            
        Raises:
            ValueError: Wenn der Puffer noch nicht genügend Daten für einen vollständigen Batch enthält.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Batch-Ziehung fehlgeschlagen. Der Puffer benötigt mindestens {batch_size} "
                f"Elemente, enthält aber aktuell nur {len(self.buffer)}."
            )
            
        # Ziehung OHNE Zurücklegen, um Duplikate innerhalb desselben Batches zu vermeiden.
        batch = random.sample(self.buffer, batch_size)
        
        # Entpackt die Liste von Tripeln: [(s1, a1, v1), (s2, a2, v2), ...]
        # in drei getrennte Tuple: (s1, s2, ...), (a1, a2, ...), (v1, v2, ...)
        states, action_probs, values = zip(*batch)
        
        return list(states), list(action_probs), list(values)

    def __len__(self) -> int:
        """
        Gibt die aktuelle Füllmenge des Puffers zurück.
        """
        return len(self.buffer)
