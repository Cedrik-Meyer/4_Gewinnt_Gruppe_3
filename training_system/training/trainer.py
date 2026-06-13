"""
training_system/training/trainer.py

Verantwortlich für das Trainieren des Connect4Model.
Berechnet die Loss-Funktionen (Fehler) für den Policy- und Value-Head,
führt den Optimizer aus und passt die Gewichte des Netzwerks an.
"""

import torch
import torch.nn as nn
from training_system.neural_network.model import Connect4Model

class Connect4Trainer:
    """
    Klasse zur Verwaltung des Trainingsprozesses für das Connect4-Netzwerk.
    Berechnet den Gesamt-Fehler (Loss) basierend auf dem AlphaZero-Ansatz.
    """
    
    def __init__(self, model: Connect4Model):
        """
        Initialisiert den Trainer mit dem zu trainierenden Modell und den Loss-Funktionen.
        
        Args:
            model (Connect4Model): Das zu trainierende neuronale Netzwerk.
        """
        self.model = model
        
        # ---------------------------------------------------------
        # B5_05: Definition der Loss-Funktionen
        # ---------------------------------------------------------
        # CrossEntropyLoss (Kreuzentropie) misst den Fehler bei Klassifikationen.
        # Es vergleicht die 16 vorhergesagten Logits mit den idealen Wahrscheinlichkeiten.
        self.policy_loss_fn = nn.CrossEntropyLoss()
        
        # MSELoss (Mean Squared Error / Mittlerer quadratischer Fehler) ist perfekt für Regressionen.
        # Es misst den Abstand zwischen der geschätzten Stellungsbewertung und dem echten Ausgang.
        self.value_loss_fn = nn.MSELoss()
        
        # Platzhalter für B5_06 (Optimizer & Scheduler)
        self.optimizer = None

    def compute_loss(self, predicted_logits: torch.Tensor, predicted_value: torch.Tensor, 
                     target_probs: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        """
        B5_05: Berechnet den kombinierten Gesamt-Fehler (Total Loss) aus Policy- und Value-Loss.
        
        Args:
            predicted_logits (torch.Tensor): Vom Modell vorhergesagte Logits [Batch, 16].
            predicted_value (torch.Tensor): Vom Modell vorhergesagter Zustandswert [Batch, 1].
            target_probs (torch.Tensor): Die Ziel-Wahrscheinlichkeiten aus dem Buffer [Batch, 16].
            target_value (torch.Tensor): Der tatsächliche Spielausgang aus dem Buffer [Batch, 1].
            
        Returns:
            torch.Tensor: Der kombinierte Gesamt-Loss als differenzierbarer PyTorch-Skalar.
        """
        # 1. Policy Loss berechnen
        # Vergleicht die Vorhersage der 16 Züge mit der Verteilung, die im Buffer liegt.
        # PyTorch kann seit Version 1.10 direkt soft targets (Wahrscheinlichkeitsverteilungen) verarbeiten.
        policy_loss = self.policy_loss_fn(predicted_logits, target_probs)
        
        # 2. Value Loss berechnen
        # Quadriert den Abstand zwischen Schätzung (z.B. 0.2) und Realität (z.B. -1.0).
        # Das Quadrieren sorgt dafür, dass extreme Fehlprognosen besonders hart bestraft werden!
        value_loss = self.value_loss_fn(predicted_value, target_value)
        
        # 3. Gesamt-Loss kombinieren (AlphaZero-Standard)
        # Die Summe beider Fehler bildet den Gesamt-Loss. PyTorch merkt sich hierbei 
        # den mathematischen Pfad zu beiden Köpfen für das spätere Update.
        total_loss = policy_loss + value_loss
        
        return total_loss
