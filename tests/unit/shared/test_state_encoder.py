"""
tests/unit/shared/test_state_encoder.py

Unit-Tests fuer die Umwandlung des 3D-Spielbretts in die Tensor-Darstellung
des neuronalen Netzes.
"""

import numpy as np
import torch

from shared.state_encoder import generate_feature_channels, encode_state, get_legal_mask


def test_generate_feature_channels_player_one_perspective():
    """Spieler 1 muss eigene Steine in Kanal 0 und gegnerische in Kanal 1 sehen."""
    board = np.zeros((4, 4, 4), dtype=np.int8)
    board[0][0][0] = 1
    board[1][2][3] = 2

    channels = generate_feature_channels(board, player_slot=0)

    assert channels.shape == (2, 4, 4, 4)
    assert channels.dtype == np.float32
    assert channels[0][0][0][0] == 1.0
    assert channels[1][1][2][3] == 1.0
    assert channels[0][1][2][3] == 0.0
    assert channels[1][0][0][0] == 0.0


def test_generate_feature_channels_player_two_perspective():
    """Spieler 2 muss dieselbe Brettlage aus umgekehrter Perspektive sehen."""
    board = np.zeros((4, 4, 4), dtype=np.int8)
    board[0][0][0] = 1
    board[1][2][3] = 2

    channels = generate_feature_channels(board, player_slot=1)

    assert channels[0][1][2][3] == 1.0
    assert channels[1][0][0][0] == 1.0
    assert channels[0][0][0][0] == 0.0
    assert channels[1][1][2][3] == 0.0


def test_encode_state_returns_float_tensor_with_expected_shape():
    """encode_state muss die Feature Channels als PyTorch Tensor liefern."""
    board = np.zeros((4, 4, 4), dtype=np.int8)
    board[0][1][1] = 1

    state = encode_state(board, player_slot=0)

    assert isinstance(state, torch.Tensor)
    assert state.shape == (2, 4, 4, 4)
    assert state.dtype == torch.float32
    assert state[0][0][1][1].item() == 1.0


def test_get_legal_mask_marks_full_columns_as_illegal():
    """Eine Spalte ist illegal, sobald auf der obersten Ebene ein Stein liegt."""
    board = np.zeros((4, 4, 4), dtype=np.int8)
    board[3][0][0] = 1
    board[3][2][3] = 2

    legal_mask = get_legal_mask(board)

    assert legal_mask.shape == (16,)
    assert legal_mask.dtype == np.float32
    assert legal_mask[0] == 0.0
    assert legal_mask[11] == 0.0
    assert legal_mask[1] == 1.0


def test_get_legal_mask_uses_z_x_flatten_order():
    """Der flache Action-Index muss zur x/z-Aufloesung im Self-Play passen."""
    board = np.zeros((4, 4, 4), dtype=np.int8)
    x = 2
    z = 1
    board[3][z][x] = 1

    legal_mask = get_legal_mask(board)

    assert legal_mask[z * 4 + x] == 0.0
