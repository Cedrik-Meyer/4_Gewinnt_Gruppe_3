"""
tests/unit/training/test_neural_network.py

Unit-Tests für das neuronale Netzwerk und die Tensor-Codierung.
"""

import pytest
import torch
import numpy as np
from shared.state_encoder import encode_state, get_legal_mask
from training_system.neural_network.model import Connect4Model

def test_model_forward_pass_shapes():
    """
    Prüft, ob das ungelernte Modell einen Fake-Tensor verarbeiten kann
    und exakt die erwarteten Output-Shapes (16 Logits, 1 Value) zurückgibt.
    """
    # Arrange: Modell instanziieren
    model = Connect4Model()
    model.eval()  # Setzt das Modell in den Inferenz-Modus (wichtig für BatchNorm)
    
    # Arrange: Fake-Tensor erstellen
    # Shape: [Batch=1, Channels=2, Y=4, Z=4, X=4]
    # Wir füllen ihn einfach mit Nullen (ein komplett leeres Spielfeld)
    fake_input = torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32)
    
    # Act: Forward Pass durchführen
    # Mit torch.no_grad() sparen wir uns die Berechnung von Gradienten, 
    # da wir hier nicht trainieren, sondern nur testen wollen.
    with torch.no_grad():
        policy_logits, value = model(fake_input)
        
    # Assert: Output-Shapes prüfen
    assert policy_logits.shape == (1, 16), f"Erwartete Policy Shape (1, 16), bekam {policy_logits.shape}"
    assert value.shape == (1, 1), f"Erwartete Value Shape (1, 1), bekam {value.shape}"
    
    # Assert: Value-Wertebereich prüfen (Tanh muss zwischen -1.0 und 1.0 liegen)
    assert value.item() >= -1.0 and value.item() <= 1.0, "Value muss zwischen -1.0 und 1.0 liegen"

def test_state_encoder_and_mask():
    """
    Prüft, ob der State-Encoder aus einem Numpy-Board einen korrekten 
    Tensor baut und ob die Legal Mask richtig funktioniert.
    """
    # Arrange: Wir bauen ein kleines Fake-Board (4x4x4 mit Nullen)
    board = np.zeros((4, 4, 4), dtype=np.uint8)
    
    # Wir platzieren einen eigenen Stein (player_slot=0 -> Stein 1) ganz unten links
    board[0][0][0] = 1
    # Wir platzieren einen gegnerischen Stein ganz oben rechts (y=3, z=3, x=3)
    board[3][3][3] = 2
    
    # Act 1: Tensor codieren
    state_tensor = encode_state(board, player_slot=0)
    
    # Act 2: Legal Mask erstellen
    legal_mask = get_legal_mask(board)
    
    # Assert 1: Tensor Shape prüfen (hier noch ohne Batch-Dimension!)
    assert state_tensor.shape == (2, 4, 4, 4)
    
    # Assert 2: Invarianz prüfen
    # Kanal 0 (Eigene Steine) muss an Position [0,0,0] eine 1.0 haben
    assert state_tensor[0, 0, 0, 0] == 1.0
    # Kanal 1 (Gegner) muss an Position [3,3,3] eine 1.0 haben
    assert state_tensor[1, 3, 3, 3] == 1.0
    
    # Assert 3: Maske prüfen
    assert legal_mask.shape == (16,)
    # Da y=3 an Position [3][3] blockiert ist, muss der letzte Eintrag der Maske 0.0 sein
    assert legal_mask[-1] == 0.0
    # Alle anderen 15 Säulen müssen frei (1.0) sein
    assert sum(legal_mask) == 15.0
    