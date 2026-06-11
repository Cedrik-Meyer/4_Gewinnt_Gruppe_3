import numpy as np

from shared.data_structures import GameState


def parse_turn_request(message: dict) -> GameState:
    """
    Wandelt einen turn.request-Umschlag des Servers in den internen GameState um.
    """
    payload = message["payload"]
    match = payload["match"]
    state = match["state"]

    board = np.array(state["board"], dtype=np.uint8)

    return GameState(
        board=board,
        player_slot=int(payload["playerSlot"]),
        current_player=int(state["currentPlayer"]),
        match_id=str(message["matchId"]),
        request_id=str(message["requestId"]),
    )
