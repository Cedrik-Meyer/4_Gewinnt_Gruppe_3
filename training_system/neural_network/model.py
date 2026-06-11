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
    """
    
    def __init__(self):
        super().__init__()
        
        # ---------------------------------------------------------
        # B4_05: Feature Extractor (3D-Faltungsschichten)
        # ---------------------------------------------------------
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        self.flattened_size = 64 * 4 * 4 * 4  # 4096
        
        # ---------------------------------------------------------
        # B4_06: Policy-Head (Wahrscheinlichkeiten der 16 Züge)
        # ---------------------------------------------------------
        self.policy_head = nn.Sequential(
            nn.Linear(self.flattened_size, 16)
        )
        
        # ---------------------------------------------------------
        # B4_07: Value-Head (Stellungsbewertung)
        # ---------------------------------------------------------
        # Ein linearer Layer komprimiert die 4096 Features auf genau 1 Wert.
        # Die Tanh-Aktivierungsfunktion drückt das Ergebnis zwingend in den
        # Bereich zwischen -1.0 und +1.0.
        self.value_head = nn.Sequential(
            nn.Linear(self.flattened_size, 1),
            nn.Tanh()
        )

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
        # 1. Feature Extraktion: [B, 2, 4, 4, 4] -> [B, 4096]
        features = self.conv_layers(x)
        
        # 2. Policy berechnen (Welcher Zug ist gut?): [B, 4096] -> [B, 16]
        policy_logits = self.policy_head(features)
        
        # 3. Value berechnen (Wer gewinnt das Spiel?): [B, 4096] -> [B, 1]
        value = self.value_head(features)
        
        return policy_logits, value
    