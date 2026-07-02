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
    
    Die Architektur basiert auf insgesamt 13 sequentiellen Schichten (Layern),
    die in einen gemeinsamen Feature-Extractor (3D-Faltungen) und zwei 
    unabhängige Köpfe (Policy und Value) strukturiert sind.
    """
    
    def __init__(self):
        """
        PHASE 1: INSTANZIIERUNG DER NETZWERKSCHICHTEN (Speicher-Allokation)
        
        Der Konstruktor wird einmalig beim Start aufgerufen. Hier werden alle 13
        mathematischen Schichten (Layer) des Modells im Speicher angelegt und die
        trainierbaren Parameter (Weights und Biases) initialisiert.
        """
        super().__init__()
        
        # ---------------------------------------------------------
        # Block A: Gemeinsame Feature-Extraktion (Layer 1 bis 10)
        # ---------------------------------------------------------
        # Verarbeitet den Input-Tensor der Form [Batch, Channels, Y, Z, X]
        self.conv_layers = nn.Sequential(
            # Layer 1-3: Erste räumliche Merkmalsextraktion
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            # Layer 4-6: Tiefere Merkmalsextraktion
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            # Layer 7-9: Hochabstrakte Mustererkennung
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            # Layer 10: Dimensionale Transformation
            # Wandelt den 5D-Tensor zwingend in einen flachen 1D-Vektor um.
            # Dies ist notwendig, da Fully-Connected-Layer (wie in den Köpfen)
            # keine räumlichen Matrizen verarbeiten können.
            nn.Flatten()
        )
        
        # Die Vektor-Dimension nach dem Flatten-Layer: 64 Channels * 4(Y) * 4(Z) * 4(X)
        self.flattened_size = 64 * 4 * 4 * 4  # 4096
        
        # ---------------------------------------------------------
        # Block B: Policy-Head (Layer 11)
        # ---------------------------------------------------------
        self.policy_head = nn.Sequential(
            # Layer 11: Transformiert die 4096 extrahierten Merkmale in 16 Logits
            nn.Linear(self.flattened_size, 16)
        )
        
        # ---------------------------------------------------------
        # Block C: Value-Head (Layer 12 bis 13)
        # ---------------------------------------------------------
        self.value_head = nn.Sequential(
            # Layer 12: Komprimiert die 4096 Merkmale auf einen Skalarwert
            nn.Linear(self.flattened_size, 1),
            
            # Layer 13: Wertebereichs-Normierung
            # Die Tanh-Aktivierungsfunktion presst das Ergebnis zwingend in den 
            # definierten Wertebereich zwischen -1.0 (Niederlage) und +1.0 (Sieg).
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        """
        PHASE 2: DATENFLUSS (Forward Pass)
        
        Diese Funktion wird bei jeder Vorhersage (Inferenz/Training) aufgerufen.
        Sie definiert den Berechnungsgraphen, also in welcher Reihenfolge der 
        Eingabe-Tensor 'x' durch die zuvor instanziierten 13 Schichten propagiert.
        
        Args:
            x (torch.Tensor): Der Input-Tensor der Dimension [Batch, 2, 4, 4, 4].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - policy_logits: Rohwerte für die 16 möglichen Züge (Shape: [B, 16]).
                - value: Evaluierung der Brettstellung von -1.0 bis 1.0 (Shape: [B, 1]).
        """
        
        # 1. Propagation durch Layer 1 bis 10:
        # Das Spielfeld fließt durch die 3D-Faltungen und wird zu einem flachen 
        # Vektor aus 4096 abstrakten Features komprimiert.
        features = self.conv_layers(x)
        
        # 2. Parallele Propagation durch die Ausgabeköpfe:
        # Die extrahierten Features werden simultan an Layer 11 sowie an Layer 12-13 übergeben.
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value
