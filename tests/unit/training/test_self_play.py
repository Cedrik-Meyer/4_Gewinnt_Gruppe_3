"""
tests/unit/training/test_self_play.py

Prüft die Simulation der Spiele und die korrekte Zuweisung von +1/-1 Rewards.
"""

import torch
import numpy as np
from training_system.neural_network.model import Connect4Model
from training_system.self_play.self_play_loop import play_single_game, store_game_trajectory
from training_system.self_play.replay_buffer import ReplayBuffer

def test_play_single_game_execution():
    """
    Prüft, ob ein untrainiertes Netz eine Partie bis zum Ende spielen kann 
    und die generierte Trajectory das korrekte Format besitzt.
    """
    model = Connect4Model()
    model.eval()  # Wichtig, damit das Model beim Spielen nicht im Train-Modus hängt
    
    trajectory, winner = play_single_game(model)
    
    assert isinstance(trajectory, list)
    assert len(trajectory) >= 7, "Ein Connect4 Spiel muss mindestens 7 Züge haben (4 für den Sieger, 3 für den Verlierer)."
    assert winner in [0, 1, 2], "Der Gewinner muss 1, 2 oder 0 (Unentschieden) sein."
    
    # Prüfe den Inhalt des allerersten aufgezeichneten Zugs
    first_step = trajectory[0]
    assert "state" in first_step and isinstance(first_step["state"], torch.Tensor)
    assert "action_probs" in first_step and isinstance(first_step["action_probs"], np.ndarray)
    assert "player" in first_step and first_step["player"] == 1

def test_store_game_trajectory_rewards():
    """
    Prüft die B5_04 Logik: Züge des Gewinners müssen +1.0 erhalten, Züge des Verlierers -1.0.
    """
    buffer = ReplayBuffer(capacity=10)
    
    # Fake-Spielverlauf: Spieler 2 gewinnt das Spiel
    fake_trajectory = [
        {"state": "Zug_von_P1", "action_probs": [0.1]*16, "player": 1},
        {"state": "Zug_von_P2", "action_probs": [0.1]*16, "player": 2},
    ]
    
    # Wir übergeben Spieler 2 als Sieger an den Speichervorgang
    store_game_trajectory(trajectory=fake_trajectory, winner=2, replay_buffer=buffer)
    
    states, _, values = buffer.sample_batch(batch_size=2)
    
    # Die Reihenfolge beim Sampling ist zufällig, daher prüfen wir über das Mapping
    for s, v in zip(states, values):
        if s == "Zug_von_P1":
            assert v == -1.0, "Spieler 1 hat verloren, Reward muss -1.0 sein."
        elif s == "Zug_von_P2":
            assert v == 1.0, "Spieler 2 hat gewonnen, Reward muss 1.0 sein."
