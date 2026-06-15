"""
tests/unit/training/test_checkpoints.py

Prueft das Speichern und Laden der Modellgewichte (State Dictionary).
"""

import os
import torch
import tempfile
from training_system.neural_network.model import Connect4Model

def test_checkpoint_save_and_load():
    """
    Prueft, ob das state_dict korrekt auf die Festplatte geschrieben 
    und von einem anderen Modell fehlerfrei gelesen werden kann.
    """
    # Arrange: Ein "trainiertes" Modell erstellen
    original_model = Connect4Model()
    
    # Wir erzeugen einen temporaeren Ordner, der nach dem Test automatisch geloescht wird
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "test_champion.pt")
        
        # Act 1: Modell speichern (B6_04 Logik)
        torch.save(original_model.state_dict(), filepath)
        
        # Assert 1: Pruefen, ob die Datei physisch existiert
        assert os.path.exists(filepath), "Die Checkpoint-Datei wurde nicht erstellt."
        
        # Act 2: Ein komplett neues, leeres Modell erzeugen und die Gewichte laden
        loaded_model = Connect4Model()
        loaded_model.load_state_dict(torch.load(filepath, weights_only=True))
        
        # Assert 2: Vergleichen, ob die Millionen von Parametern mathematisch identisch sind
        for param_original, param_loaded in zip(original_model.parameters(), loaded_model.parameters()):
            assert torch.equal(param_original, param_loaded), "Geladene Gewichte weichen vom Original ab!"
