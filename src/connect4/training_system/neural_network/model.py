"""
training_system/neural_network/model.py

Implementierung der neuronalen Netzwerkarchitektur für den Connect4-3D Agenten.

Dieses Modul definiert ein Dual-Head Convolutional Neural Network (CNN) in Anlehnung
an die AlphaZero-Architektur. Das Netzwerk erhält den normalisierten Spielzustand 
als 3D-Tensor und liefert simultan zwei Auswertungen:
1. Policy (Zug-Präferenzen für die MCTS-Suche)
2. Value (Bewertung der aktuellen Brettstellung)
"""

import torch
import torch.nn as nn


class Connect4Model(nn.Module):
    """
    Neuronales Netzwerk zur Evaluierung von 3D-Connect4-Spielzuständen.
    
    Die Architektur basiert auf einem gemeinsamen Feature-Extractor (3D-Faltungen),
    der in zwei unabhängigen Köpfen (Policy und Value) mündet.
    """
    
    def __init__(self):
        super().__init__()
        
        # ---------------------------------------------------------
        # Gemeinsame Feature-Extraktion (Shared Body)
        # ---------------------------------------------------------
        # Verarbeitet den Input-Tensor der Form [Batch, Channels, Y, Z, X]
        # Channels: 2 (Eigene Steine, Gegnerische Steine)
        # Dimension: 4x4x4 Spielfeld
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
        
        # Die Vektor-Dimension nach der Faltung: 64 Channels * 4(Y) * 4(Z) * 4(X)
        self.flattened_size = 64 * 4 * 4 * 4  # 4096
        
        # ---------------------------------------------------------
        # Policy-Head (Handlungsstrategie)
        # ---------------------------------------------------------
        # Transformiert die 4096 extrahierten Merkmale in 16 unmaskierte Logits.
        # Diese repräsentieren die Präferenz für jeden der 16 möglichen Züge (4x4 Raster).
        self.policy_head = nn.Sequential(
            nn.Linear(self.flattened_size, 16)
        )
        
        # ---------------------------------------------------------
        # Value-Head (Stellungsbewertung)
        # ---------------------------------------------------------
        # Komprimiert die 4096 Merkmale auf einen einzigen Skalarwert.
        # Die Tanh-Aktivierungsfunktion normiert das Ergebnis zwingend auf den 
        # Wertebereich zwischen -1.0 (Niederlage) und +1.0 (Sieg).
        self.value_head = nn.Sequential(
            nn.Linear(self.flattened_size, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        """
        Führt den Vorwärtspass (Forward Pass) durch das Netzwerk aus.
        
        Args:
            x (torch.Tensor): Der Input-Tensor der Dimension [Batch, 2, 4, 4, 4].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - policy_logits: Rohwerte für die 16 möglichen Züge (Shape: [B, 16]).
                - value: Evaluierung der Brettstellung von -1.0 bis 1.0 (Shape: [B, 1]).
        """
        # 1. Extrahieren der räumlichen Muster aus dem Brettzustand
        features = self.conv_layers(x)
        
        # 2. Parallele Berechnung der beiden Ausgabeköpfe
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value
    