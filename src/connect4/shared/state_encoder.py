"""
shared/state_encoder.py

Zuständig für die Transformation des logischen Spielzustands (Ebene B)
in die mathematische Tensor-Repräsentation für das Neuronale Netz (Ebene C).
"""

import numpy as np
import torch

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

def encode_state(board: np.ndarray, player_slot: int) -> torch.Tensor:
    """
    Kombiniert die Invarianz-Transformation mit der PyTorch-Konvertierung.
    Wandelt das Numpy-Array in einen float32 PyTorch-Tensor um (Ebene C).
    
    Args:
        board (np.ndarray): Das logische 3D-Spielfeld.
        player_slot (int): Der Slot des Agenten (0 oder 1).
        
    Returns:
        torch.Tensor: Ein Tensor der Shape [2, 4, 4, 4] mit dtype torch.float32.
    """
    # 1. Feature Channels via Numpy generieren (Logik aus B4_01)
    channels_array = generate_feature_channels(board, player_slot)
    
    # 2. In PyTorch Tensor umwandeln
    # torch.from_numpy ist extrem speichereffizient (Zero-Copy), da es den  
    # RAM-Speicher mit Numpy teilt, anstatt das Array komplett neu zu kopieren.
    state_tensor = torch.from_numpy(channels_array)
    
    return state_tensor

def get_legal_mask(board: np.ndarray) -> np.ndarray:
    """
    Erstellt ein 1D-Array (Größe 16), das für das Maskieren der Modell-Ausgaben genutzt wird.
    Prüft die oberste Ebene (y=3) des Boards.
    
    Args:
        board (np.ndarray): Das 3D-Spielfeld mit Shape [4, 4, 4].
        
    Returns:
        np.ndarray: Ein 1D float32 Array. 1.0 bedeutet "Säule frei", 0.0 bedeutet "Säule voll".
    """
    # 1. Wir schneiden die oberste Ebene (Dachgeschoss) ab. Shape ist jetzt [4, 4] für (z, x)
    top_layer = board[3]
    
    # 2. Überall wo eine 0 steht, ist die Säule noch NICHT voll (True). 
    # Wo ein Stein liegt, ist sie voll (False).
    valid_positions = (top_layer == 0)
    
    # 3. Wir wandeln True/False in 1.0 und 0.0 um (für PyTorch Multiplikationen) 
    # und machen das 4x4 Grid mit flatten() zu einem 1D-Array der Länge 16.
    legal_mask = valid_positions.astype(np.float32).flatten()
    
    return legal_mask
