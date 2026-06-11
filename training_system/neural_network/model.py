"""
training_system/neural_network/model.py

Das neuronale Netzwerk für den Connect4 3D Agenten.
Es nutzt eine Actor-Critic-Architektur (AlphaZero-Ansatz), um aus dem 3D-Tensor 
gleichzeitig Zug-Wahrscheinlichkeiten (Policy) und eine Stellungsbewertung (Value) zu berechnen.
"""

import torch
import torch.nn as nn

class Connect4Model(nn.Module):
    """
    Das neuronale Netzwerk zur Evaluierung von 3D-Connect4-Spielzuständen.
    
    Input-Shape (Ebene C):
        [Batch, Channels, Y, Z, X] -> [B, 2, 4, 4, 4]
        - Channels: 2 (Kanal 0: Eigene Steine, Kanal 1: Gegnerische Steine)
        - Y, Z, X: Jeweils 4 (Höhe, Tiefe, Breite des 3D-Boards)
    """
    
    def __init__(self):
        # Der Aufruf von super() ist zwingend erforderlich, damit PyTorch 
        # die Klasse korrekt als neuronales Netz registriert.
        super().__init__()
        
        # ---------------------------------------------------------
        # Platzhalter: Hier kommen in B4_05 die 3D-Faltungsschichten hin.
        # Wir wissen bereits: in_channels=2 (aufgrund unseres State-Encoders).
        # ---------------------------------------------------------
        pass

    def forward(self, x: torch.Tensor):
        """
        Der Vorwärtspass durch das Netzwerk.
        
        Args:
            x (torch.Tensor): Der Input-Tensor der Shape [B, 2, 4, 4, 4].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - policy_logits: Rohwerte für die 16 möglichen Züge (Shape: [B, 16])
                - value: Stellungsbewertung zwischen -1.0 und 1.0 (Shape: [B, 1])
        """
        # ---------------------------------------------------------
        # Platzhalter: Hier werden in B4_05 bis B4_07 die Layer aufgerufen.
        # ---------------------------------------------------------
        pass
    