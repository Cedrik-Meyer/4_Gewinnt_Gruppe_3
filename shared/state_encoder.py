"""
shared/state_encoder.py

Zuständig für die Transformation des logischen Spielzustands (Ebene B)
in die mathematische Tensor-Repräsentation für das Neuronale Netz (Ebene C).
"""

import numpy as np

def generate_feature_channels(board: np.ndarray, player_slot: int) -> np.ndarray:
    """
    Invertiert das Spielfeld dynamisch aus der Perspektive des aktuellen Spielers
    (Invarianz-Transformation).
    
    Args:
        board (np.ndarray): Das 3D-Spielfeld mit Shape [4, 4, 4] (Werte: 0, 1, 2).
        player_slot (int): Der Slot des Agenten (0 oder 1).
        
    Returns:
        np.ndarray: Ein Numpy-Array der Shape [2, 4, 4, 4] vom Typ float32.
                    Kanal 0: Eigene Steine (1.0 = Stein, 0.0 = Leer).
                    Kanal 1: Gegnerische Steine (1.0 = Stein, 0.0 = Leer).
    """
    # 1. Ermittle, welche Zahlenwerte auf dem Board wem gehören
    # player_slot 0 -> Eigene Steine sind 1, Gegner hat 2
    # player_slot 1 -> Eigene Steine sind 2, Gegner hat 1
    own_value = player_slot + 1
    opp_value = 2 if player_slot == 0 else 1
    
    # 2. Erzeuge boolesche Masken (True/False) und wandle sie 
    # direkt in float32 um (1.0 oder 0.0), da neuronale Netze
    # mit Fließkommazahlen arbeiten.
    own_channel = (board == own_value).astype(np.float32)
    opp_channel = (board == opp_value).astype(np.float32)
    
    # 3. Stapele die beiden 3D-Matrizen übereinander.
    # Aus zwei [4, 4, 4] Arrays wird ein [2, 4, 4, 4] Array.
    return np.stack([own_channel, opp_channel])
