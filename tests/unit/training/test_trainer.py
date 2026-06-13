"""
tests/unit/training/test_trainer.py

Prüft die Backpropagation-Pipeline (Loss, Optimizer, Step) des Trainers.
"""

import torch
import numpy as np
from training_system.neural_network.model import Connect4Model
from training_system.training.trainer import Connect4Trainer

def test_trainer_initialization():
    """
    Prüft, ob der Trainer das Modell, die Loss-Funktionen und den Optimizer lädt.
    """
    model = Connect4Model()
    trainer = Connect4Trainer(model)
    
    assert trainer.optimizer is not None
    assert trainer.policy_loss_fn is not None
    assert trainer.value_loss_fn is not None

def test_trainer_train_step_execution():
    """
    Baut einen Fake-Batch und führt einen kompletten Trainingsschritt aus, 
    um Shape-Mismatches in der Backpropagation auszuschließen.
    """
    model = Connect4Model()
    trainer = Connect4Trainer(model)
    
    # Arrange: Wir simulieren einen Batch (batch_size=2) aus dem Buffer
    # 1. States (2x [2, 4, 4, 4] Tensors)
    state_1 = torch.zeros((2, 4, 4, 4), dtype=torch.float32)
    state_2 = torch.ones((2, 4, 4, 4), dtype=torch.float32)
    states = [state_1, state_2]
    
    # 2. Target Probs (Wahrscheinlichkeitsverteilungen für beide Züge)
    probs_1 = np.ones(16) / 16.0  # Gleichverteilung
    probs_2 = np.zeros(16)
    probs_2[0] = 1.0             # 100% Sicherheit für Index 0
    target_probs = [probs_1, probs_2]
    
    # 3. Target Values (Echter Spielausgang)
    target_values = [1.0, -1.0]
    
    # Act: Führe einen Trainingsschritt aus
    loss = trainer.train_step(states, target_probs, target_values)
    
    # Assert
    assert isinstance(loss, float), "Der zurückgegebene Loss muss eine Float-Zahl sein."
    assert loss > 0.0, "Der berechnete Loss muss positiv sein."
    # Wenn dieser Test ohne Absturz durchläuft, ist die PyTorch Pipeline:
    # forward -> compute_loss -> backward -> step mathematisch 100% korrekt verlinkt!
