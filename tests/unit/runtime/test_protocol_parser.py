import re
from datetime import datetime

import numpy as np
import pytest

from runtime_system.parser.protocol_parser import build_move_submit, parse_turn_request
from shared.data_structures import Move


def server_turn_request_example() -> dict:
    return {
        "version": 1,
        "type": "turn.request",
        "requestId": "req-turn-1",
        "matchId": "550e8400-e29b-41d4-a716-446655440000",
        "agentId": "550e8400-e29b-41d4-a716-446655440001",
        "payload": {
            "match": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "gameId": "connect-four-3d",
                "status": "active",
                "state": {
                    "size": 4,
                    "height": 4,
                    "board": [
                        [
                            [0, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 2, 0],
                            [0, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                    ],
                    "currentPlayer": 0,
                    "moves": [
                        {"playerSlot": 0, "x": 1, "y": 0, "z": 1},
                        {"playerSlot": 1, "x": 2, "y": 0, "z": 2},
                    ],
                },
                "players": [
                    {
                        "slot": 0,
                        "agentId": "550e8400-e29b-41d4-a716-446655440001",
                        "agentName": "Gruppe A Agent",
                    },
                    {
                        "slot": 1,
                        "agentId": "550e8400-e29b-41d4-a716-446655440002",
                        "agentName": "Gruppe B Agent",
                    },
                ],
            },
            "playerSlot": 0,
            "deadlineMs": 1779883205000,
        },
        "timestamp": "2026-05-27T12:00:00.000Z",
    }


def parse_valid_example():
    game_state = parse_turn_request(server_turn_request_example())
    assert game_state is not None
    return game_state


def test_parse_turn_request_builds_expected_game_state():
    game_state = parse_valid_example()

    assert game_state.board.shape == (4, 4, 4)
    assert game_state.board.dtype == np.uint8
    assert game_state.board[0][1][1] == 1
    assert game_state.board[0][2][2] == 2
    assert game_state.player_slot == 0
    assert game_state.current_player == 0
    assert game_state.match_id == "550e8400-e29b-41d4-a716-446655440000"
    assert game_state.request_id == "req-turn-1"
    assert game_state.legal_mask.shape == (16,)
    assert game_state.legal_mask.dtype == np.float32
    assert np.all(game_state.legal_mask == 1.0)


def test_parse_turn_request_ignores_unknown_extra_fields():
    message = server_turn_request_example()
    message["payload"]["extraField"] = "wird ignoriert"
    message["payload"]["match"]["state"]["unknownStateField"] = 123

    game_state = parse_turn_request(message)

    assert game_state is not None
    assert game_state.request_id == "req-turn-1"
    assert game_state.board[0][1][1] == 1


def test_parse_turn_request_computes_legal_mask_for_full_columns():
    message = server_turn_request_example()
    board = message["payload"]["match"]["state"]["board"]
    board[3][0][0] = 1
    board[3][2][3] = 2

    game_state = parse_turn_request(message)

    assert game_state is not None
    assert game_state.legal_mask[0] == 0.0
    assert game_state.legal_mask[11] == 0.0
    assert game_state.legal_mask.sum() == 14.0


@pytest.mark.parametrize(
    "remove_path",
    [
        ("payload",),
        ("matchId",),
        ("requestId",),
        ("payload", "playerSlot"),
        ("payload", "match"),
        ("payload", "match", "state"),
        ("payload", "match", "state", "board"),
        ("payload", "match", "state", "currentPlayer"),
    ],
)
def test_parse_turn_request_rejects_missing_required_fields(remove_path):
    message = server_turn_request_example()
    current = message
    for key in remove_path[:-1]:
        current = current[key]
    del current[remove_path[-1]]

    assert parse_turn_request(message) is None


@pytest.mark.parametrize(
    "message",
    [
        None,
        "kein dict",
        [],
        {"type": "move.accepted"},
        {"type": "error", "payload": {}},
        {**server_turn_request_example(), "type": "turn.request.unbekannt"},
    ],
)
def test_parse_turn_request_rejects_invalid_message_types(message):
    assert parse_turn_request(message) is None


@pytest.mark.parametrize(
    "board",
    [
        [[0]],
        np.zeros((4, 4), dtype=np.uint8).tolist(),
        np.zeros((5, 4, 4), dtype=np.uint8).tolist(),
        np.zeros((4, 5, 4), dtype=np.uint8).tolist(),
        np.zeros((4, 4, 5), dtype=np.uint8).tolist(),
    ],
)
def test_parse_turn_request_rejects_invalid_board_shapes(board):
    message = server_turn_request_example()
    message["payload"]["match"]["state"]["board"] = board

    assert parse_turn_request(message) is None


@pytest.mark.parametrize("invalid_value", [-1, 3, 9, "x", None])
def test_parse_turn_request_rejects_invalid_board_values(invalid_value):
    message = server_turn_request_example()
    message["payload"]["match"]["state"]["board"][0][0][0] = invalid_value

    assert parse_turn_request(message) is None


@pytest.mark.parametrize("field", ["playerSlot", "currentPlayer"])
@pytest.mark.parametrize("value", [-1, 2, 99, "abc", None])
def test_parse_turn_request_rejects_invalid_player_values(field, value):
    message = server_turn_request_example()
    if field == "playerSlot":
        message["payload"]["playerSlot"] = value
    else:
        message["payload"]["match"]["state"]["currentPlayer"] = value

    assert parse_turn_request(message) is None


@pytest.mark.parametrize("value", ["0", "1"])
def test_parse_turn_request_accepts_numeric_player_strings(value):
    message = server_turn_request_example()
    message["payload"]["playerSlot"] = value
    message["payload"]["match"]["state"]["currentPlayer"] = value

    game_state = parse_turn_request(message)

    assert game_state is not None
    assert game_state.player_slot == int(value)
    assert game_state.current_player == int(value)


def test_build_move_submit_uses_server_envelope_format():
    game_state = parse_valid_example()
    message = build_move_submit(Move(x=1, z=2, y=0), game_state)

    assert message["version"] == 1
    assert message["type"] == "move.submit"
    assert message["requestId"] == "req-turn-1"
    assert message["matchId"] == "550e8400-e29b-41d4-a716-446655440000"
    assert message["payload"] == {"x": 1, "z": 2}
    assert "y" not in message["payload"]
    assert "agentId" not in message
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", message["timestamp"])
    parsed_timestamp = datetime.fromisoformat(message["timestamp"].replace("Z", "+00:00"))
    assert parsed_timestamp.tzinfo is not None


@pytest.mark.parametrize(
    "move",
    [
        Move(x=0, z=0),
        Move(x=3, z=3),
        Move(x="1", z="2"),
    ],
)
def test_build_move_submit_accepts_valid_coordinate_edges(move):
    game_state = parse_valid_example()

    message = build_move_submit(move, game_state)

    assert message["payload"] == {"x": int(move.x), "z": int(move.z)}


@pytest.mark.parametrize(
    "move",
    [
        Move(x=-1, z=0),
        Move(x=4, z=0),
        Move(x=0, z=-1),
        Move(x=0, z=4),
        Move(x="abc", z=0),
    ],
)
def test_build_move_submit_rejects_invalid_coordinates(move):
    game_state = parse_valid_example()

    with pytest.raises(ValueError):
        build_move_submit(move, game_state)
