"""
training_system/training/trainer.py

Verantwortlich für das Trainieren des Connect4Model.
Berechnet die Loss-Funktionen (Fehler) für den Policy- und Value-Head,
führt den Optimizer aus und passt die Gewichte des Netzwerks an.
"""

import torch
import torch.nn as nn
import torch.optim as optim  
from training_system.neural_network.model import Connect4Model

class Connect4Trainer:
    """
    Klasse zur Verwaltung des Trainingsprozesses für das Connect4-Netzwerk.
    Berechnet den Gesamt-Fehler (Loss) basierend auf dem AlphaZero-Ansatz.
    """
    
    def __init__(self, model: Connect4Model, learning_rate: float = 1e-3):
        """
        Initialisiert den Trainer mit dem zu trainierenden Modell und den Loss-Funktionen.
        
        Args:
            model (Connect4Model): Das zu trainierende neuronale Netzwerk.
            learning_rate (float): Die Schrittweite für den Lernprozess (Standard: 0.001).
        """
        self.model = model
        
        # ---------------------------------------------------------
        # B5_05: Definition der Loss-Funktionen
        # ---------------------------------------------------------
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # ---------------------------------------------------------
        # B5_06: Optimizer & Scheduler
        # ---------------------------------------------------------
        # Der Adam-Optimizer sammelt die Fehler-Gradienten und passt die Parameter an.
        # weight_decay (L2-Regularisierung) bestraft zu große Gewichte und verhindert Overfitting.
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4
        )
        
        # Der Scheduler fungiert als Feinjustierung für die Learning Rate.
        # StepLR verkleinert die Learning Rate nach einer bestimmten Anzahl von Trainingsschritten.
        # Hier: Nach 10.000 Trainingsschritten wird die Lernrate mit 0.9 multipliziert.
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10000, 
            gamma=0.9
        )

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
        policy_loss = self.policy_loss_fn(predicted_logits, target_probs)
        value_loss = self.value_loss_fn(predicted_value, target_value)
        total_loss = policy_loss + value_loss
        
        return total_loss
