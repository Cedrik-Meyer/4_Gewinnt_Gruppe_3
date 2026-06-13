"""
tests/unit/training/test_replay_buffer.py

Prüft die Ringpuffer-Logik und das Sampling-Verhalten.
"""

import pytest
import numpy as np
from training_system.self_play.replay_buffer import ReplayBuffer

def test_replay_buffer_capacity_fifo():
    """
    Prüft, ob der Buffer bei Erreichen der Kapazität die ältesten Elemente überschreibt.
    """
    # Arrange: Sehr kleiner Buffer mit Platz für 3 Züge
    buffer = ReplayBuffer(capacity=3)
    
    # Act: Wir fügen 5 Züge hinzu (Züge 1 und 2 sollten herausfallen)
    for i in range(1, 6):
        buffer.push(state=f"State_{i}", action_probs=np.zeros(16), value=1.0)
        
    # Assert
    assert len(buffer) == 3, "Der Buffer darf seine Maximalkapazität nicht überschreiten."
    
    states, _, _ = buffer.sample_batch(batch_size=3)
    # Die States 3, 4 und 5 müssen enthalten sein. 1 und 2 wurden gelöscht.
    assert set(states) == {"State_3", "State_4", "State_5"}

def test_replay_buffer_sampling_insufficient_data():
    """
    Prüft, ob der Buffer einen sauberen Fehler wirft, wenn wir einen Batch anfordern, 
    der größer ist als die gespeicherte Datenmenge.
    """
    buffer = ReplayBuffer(capacity=100)
    buffer.push(state="A", action_probs=np.zeros(16), value=-1.0)
    buffer.push(state="B", action_probs=np.zeros(16), value=0.0)
    
    with pytest.raises(ValueError):
        buffer.sample_batch(batch_size=10)

def test_replay_buffer_sampling_format():
    """
    Prüft, ob die Daten beim Ziehen eines Batches korrekt in 3 separate Listen getrennt werden.
    """
    buffer = ReplayBuffer(capacity=100)
    for i in range(10):
        buffer.push(state=i, action_probs=np.array([0.1] * 16), value=0.5)
        
    states, probs, values = buffer.sample_batch(batch_size=4)
    
    assert isinstance(states, list)
    assert isinstance(probs, list)
    assert isinstance(values, list)
    assert len(states) == 4
    assert len(probs) == 4
    assert len(values) == 4
