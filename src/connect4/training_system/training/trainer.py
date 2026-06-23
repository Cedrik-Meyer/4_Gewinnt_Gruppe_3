"""
training_system/training/trainer.py

Verantwortlich für das Trainieren des Connect4Model.
Berechnet die Loss-Funktionen (Fehler) für den Policy- und Value-Head,
führt den Optimizer aus und passt die Gewichte des Netzwerks an.
Unterstützt nun automatische Hardware-Beschleunigung (CUDA/GPU).
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
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4
        )
        
        # Der Scheduler fungiert als Feinjustierung für die Learning Rate.
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
        Wurde für CUDA-Beschleunigung optimiert.
        """
        # Wir prüfen, wo das Modell gerade liegt (CPU oder NVIDIA GPU)
        device = next(self.model.parameters()).device
        
        # 1. Daten vorbereiten und direkt in den Speicher der Grafikkarte (VRAM) schieben
        states_tensor = torch.stack(states).to(device)
        probs_tensor = torch.tensor(np.array(target_probs), dtype=torch.float32).to(device)
        values_tensor = torch.tensor(target_values, dtype=torch.float32).unsqueeze(1).to(device)
        
        # 2. Modell in den Trainingsmodus versetzen
        self.model.train()
        
        # 3. Alte Fehler-Notizen löschen
        self.optimizer.zero_grad()
        
        # 4. FORWARD PASS: Vorhersagen treffen
        predicted_logits, predicted_value = self.model(states_tensor)
        
        # 5. LOSS: Fehler berechnen
        policy_loss = self.policy_loss_fn(predicted_logits, probs_tensor)
        value_loss = self.value_loss_fn(predicted_value, values_tensor)
        total_loss = policy_loss + value_loss
        
        # 6. BACKWARD PASS: Gradienten berechnen
        total_loss.backward()
        
        # 7. OPTIMIZER STEP: Gewichte anpassen
        self.optimizer.step()
        self.scheduler.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
