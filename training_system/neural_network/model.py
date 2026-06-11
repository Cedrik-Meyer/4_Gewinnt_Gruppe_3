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
        # Wir schicken den [2, 4, 4, 4] Input durch drei 3D-Scanner-Schichten.
        # padding=1 sorgt dafür, dass die Kanten des 4x4x4 Würfels nicht 
        # abgeschnitten werden und die räumliche Dimension erhalten bleibt.
        self.conv_layers = nn.Sequential(
            # Schicht 1: Aus 2 Kanälen werden 32 Feature-Karten
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            # Schicht 2: Aus 32 Karten werden 64
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            # Schicht 3: Komplexe Mustererkennung auf 64 Karten
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            # Flatten: Presst den 3D-Tensor für die nachfolgenden Linear-Layer flach.
            # Shape-Rechnung: 64 Channels * 4 Höhe * 4 Tiefe * 4 Breite = 4096
            nn.Flatten()
        )
        
        # Diese Variable speichern wir uns, damit die Output-Heads (B4_06/B4_07)
        # exakt wissen, wie viele Neuronen bei ihnen ankommen.
        self.flattened_size = 64 * 4 * 4 * 4
        
        # Platzhalter für B4_06 und B4_07
        self.policy_head = None
        self.value_head = None

    def forward(self, x: torch.Tensor):
        """
        Der Vorwärtspass durch das Netzwerk.
        
        Args:
            x (torch.Tensor): Der Input-Tensor der Shape [B, 2, 4, 4, 4].
            
        Returns:
            Ein Placeholder (Feature-Vektor) bis die Heads implementiert sind.
        """
        # 1. Feature Extraktion: [B, 2, 4, 4, 4] -> [B, 4096]
        features = self.conv_layers(x)
        
        # (Später: Die features fließen in den Policy- und Value-Head)
        return features
    