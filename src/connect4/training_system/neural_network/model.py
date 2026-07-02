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
        
        Hier werden alle 13 mathematischen Schichten (Layer) des Modells im 
        Speicher angelegt und die trainierbaren Parameter initialisiert.
        """
        super().__init__()
        
        # ---------------------------------------------------------
        # Block A: Gemeinsame Feature-Extraktion (Layer 1 bis 10)
        # ---------------------------------------------------------
        # Verarbeitet den Input-Tensor der Form [Batch, Channels, Y, Z, X]
        self.conv_layers = nn.Sequential(
            
            # --- Hierarchie-Stufe 1: Low-Level Features ---
            # Extraktion lokaler, fundamentaler Merkmale (z. B. Kanten, isolierte Spielsteine).
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, padding=1), # Layer 1: 3D-Faltungsoperation
            nn.BatchNorm3d(32),                                                  # Layer 2: Batch-Normalisierung
            nn.ReLU(),                                                           # Layer 3: Nichtlineare Aktivierung
            
            # --- Hierarchie-Stufe 2: Mid-Level Features ---
            # Kombination lokaler Features zu regionalen Strukturen (z. B. offene Zweierreihen).
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # Layer 4: 3D-Faltungsoperation
            nn.BatchNorm3d(64),                                                  # Layer 5: Batch-Normalisierung
            nn.ReLU(),                                                           # Layer 6: Nichtlineare Aktivierung
            
            # --- Hierarchie-Stufe 3: High-Level Features ---
            # Abstraktion regionaler Strukturen zu globalen, taktischen Mustern (z. B. Zwickmühlen).
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # Layer 7: 3D-Faltungsoperation
            nn.BatchNorm3d(64),                                                  # Layer 8: Batch-Normalisierung
            nn.ReLU(),                                                           # Layer 9: Nichtlineare Aktivierung
            
            # --- Dimensionale Transformation ---
            # Fully-Connected-Layer erfordern zwingend eindimensionale Vektoren als Eingabe.
            # Transformiert den 5D-Tensor aus den Faltungsschichten in einen 1D-Vektor.
            nn.Flatten()                                                         # Layer 10: Dimensionale Reduktion
        )
        
        # Die Vektor-Dimension nach dem Flatten-Layer: 64 Channels * 4(Y) * 4(Z) * 4(X)
        self.flattened_size = 64 * 4 * 4 * 4  # 4096 abstrakte Merkmale
        
        # ---------------------------------------------------------
        # Block B: Policy-Head (Layer 11)
        # ---------------------------------------------------------
        self.policy_head = nn.Sequential(
            # Berechnet aus dem abstrakten Feature-Vektor die unnormalisierten 
            # Wahrscheinlichkeiten (Logits) für den gesamten Aktionsraum (16 Spalten).
            nn.Linear(self.flattened_size, 16)                                   # Layer 11: POLICY-Ausgabe (16 Zug-Logits)
        )
        
        # ---------------------------------------------------------
        # Block C: Value-Head (Layer 12 bis 13)
        # ---------------------------------------------------------
        self.value_head = nn.Sequential(
            # Aggregiert denselben Feature-Vektor zu einer skalaren Stellungsbewertung.
            nn.Linear(self.flattened_size, 1),                                   # Layer 12: VALUE-Aggregation (Skalar)
            
            # Normiert den berechneten Skalar strikt auf das Intervall 
            # zwischen -1.0 (deterministische Niederlage) und +1.0 (deterministischer Sieg).
            nn.Tanh()                                                            # Layer 13: VALUE-Aktivierung (Normierung auf [-1.0, 1.0])
        )

    def forward(self, x: torch.Tensor):
        """
        PHASE 2: DATENFLUSS (Forward Pass)
        
        Diese Funktion wird bei jeder Inferenz/Trainingsepoche aufgerufen.
        Sie definiert den Berechnungsgraphen, also die Propagationsreihenfolge
        des Eingabe-Tensors 'x' durch die instanziierten Schichten.
        
        Args:
            x (torch.Tensor): Der Input-Tensor der Dimension [Batch, 2, 4, 4, 4].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - policy_logits: Rohwerte für die 16 möglichen Züge (Shape: [B, 16]).
                - value: Evaluierung der Brettstellung im Intervall [-1.0, 1.0] (Shape: [B, 1]).
        """
        
        # 1. Sequentielle Feature-Extraktion:
        # Propagiert den Eingabe-Tensor durch die 3D-Faltungsblöcke zur 
        # Generierung des hochdimensionalen Feature-Vektors (4096 Dimensionen).
        features = self.conv_layers(x)
        
        # 2. Parallele Output-Generierung:
        # Der Vektor wird simultan durch die linearen Ausgabeschichten propagiert.
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value
