"""
training_system/training/trainer.py

Verantwortlich für das Trainieren des Connect4Model.
Berechnet die Loss-Funktionen (Fehler) für den Policy- und Value-Head,
führt den Optimizer aus und passt die Gewichte des Netzwerks an.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        
        # CrossEntropyLoss misst den Fehler bei Klassifikationen (welcher Zug ist der beste).
        self.policy_loss_fn = nn.CrossEntropyLoss()
        
        # MSELoss misst den quadratischen Abstand bei Regressionen (Bewertung der Stellung).
        self.value_loss_fn = nn.MSELoss()
        
        # Der Adam-Optimizer sammelt die Fehler-Gradienten und passt die Parameter an.
        # weight_decay (L2-Regularisierung) bestraft zu große Gewichte und verhindert Overfitting.
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4
        )
        
        # Der Scheduler fungiert als Feinjustierung für die Learning Rate.
        # Er verkleinert die Lernrate nach 100.000 Trainingsschritten (Batches) um 10% (gamma=0.9).
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=100000, 
            gamma=0.9
        )

    def compute_loss(self, predicted_logits: torch.Tensor, predicted_value: torch.Tensor, 
                     target_probs: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        """
        B5_05: Berechnet den kombinierten Gesamt-Fehler (Total Loss) aus Policy- und Value-Loss.
        """
        policy_loss = self.policy_loss_fn(predicted_logits, target_probs)
        value_loss = self.value_loss_fn(predicted_value, target_value)
        total_loss = policy_loss + value_loss
        
        return total_loss

    def train_step(self, states: list, target_probs: list, target_values: list) -> tuple:
        """
        B5_07: Führt einen kompletten Trainingsschritt (Backpropagation) auf einem Batch aus.
        
        Args:
            states (list): Liste von Tensors (Input-Spielfelder).
            target_probs (list): Liste von Numpy-Arrays (Ziel-Wahrscheinlichkeiten).
            target_values (list): Liste von Floats (Tatsächlicher Spielausgang).
            
        Returns:
            tuple: (total_loss, policy_loss, value_loss) für detailliertes CSV-Logging.
        """
        # 1. Daten für die Grafikkarte/CPU vorbereiten (Listen -> PyTorch Batches)
        states_tensor = torch.stack(states)
        probs_tensor = torch.tensor(np.array(target_probs), dtype=torch.float32)
        values_tensor = torch.tensor(target_values, dtype=torch.float32).unsqueeze(1)
        
        # 2. Modell in den Trainingsmodus versetzen (Aktiviert BatchNorm)
        self.model.train()
        
        # 3. Alte Fehler-Notizen löschen, bevor wir neu rechnen
        self.optimizer.zero_grad()
        
        # 4. FORWARD PASS: Modell trifft seine Vorhersagen für die Spielfelder
        predicted_logits, predicted_value = self.model(states_tensor)
        
        # 5. LOSS: Die Strafe für falsche Vorhersagen berechnen (getrennt für das Log)
        policy_loss = self.policy_loss_fn(predicted_logits, probs_tensor)
        value_loss = self.value_loss_fn(predicted_value, values_tensor)
        total_loss = policy_loss + value_loss
        
        # 6. BACKWARD PASS: PyTorch sucht rückwärts durch das Netz und berechnet die Gradienten.
        total_loss.backward()
        
        # 7. OPTIMIZER STEP: Gewichte anpassen und Scheduler aktualisieren.
        self.optimizer.step()
        self.scheduler.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
