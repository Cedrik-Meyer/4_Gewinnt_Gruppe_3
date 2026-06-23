import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from shared.data_structures import GameState, Move

logger = logging.getLogger(__name__)


def parse_turn_request(message: dict) -> Optional[GameState]:
    """
    Wandelt einen turn.request-Umschlag des Servers in den internen GameState um.
    """
    try:
        if not isinstance(message, dict):
            raise TypeError("Nachricht muss ein Dictionary sein.")

        if message.get("type") != "turn.request":
            raise ValueError(f"Unerwarteter Nachrichtentyp: {message.get('type')}")

        payload = message["payload"]
        match = payload["match"]
        state = match["state"]

        board = _parse_board(state["board"])

        raw_player_slot = int(payload["playerSlot"])
        raw_current_player = int(state["currentPlayer"])

        if raw_player_slot not in (1, 2):
            raise ValueError(f"Ungültiger playerSlot: {raw_player_slot}")

        if raw_current_player not in (1, 2):
            raise ValueError(f"Ungültiger currentPlayer: {raw_current_player}")

        player_slot = raw_player_slot - 1
        current_player = raw_current_player - 1

        return GameState(
            board=board,
            player_slot=player_slot,
            current_player=current_player,
            match_id=str(message["matchId"]),
            request_id=str(message["requestId"]),
        )

    except KeyError as error:
        logger.warning("Ungültige turn.request Nachricht: fehlendes Feld %s", error)
        return None
    except (TypeError, ValueError) as error:
        logger.warning("Ungültige turn.request Nachricht: %s", error)
        return None
    except Exception:
        logger.exception("Unerwarteter Fehler beim Parsen der turn.request Nachricht.")
        return None


def build_move_submit(move: Move, game_state: GameState) -> dict:
    """
    Baut aus einem internen Move den move.submit-Umschlag für den Server.

    Der Server erwartet nur x und z im Payload. Die Höhe y wird serverseitig
    berechnet und darf deshalb nicht mitgesendet werden.
    """
    x = int(move.x)
    z = int(move.z)

    if not (0 <= x < 4 and 0 <= z < 4):
        raise ValueError(f"Ungültige Zugkoordinaten: x={x}, z={z}. Erlaubt sind Werte von 0 bis 3.")

    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    return {
        "version": 1,
        "type": "move.submit",
        "requestId": game_state.request_id,
        "matchId": game_state.match_id,
        "payload": {
            "x": x,
            "z": z,
        },
        "timestamp": timestamp,
    }


def _parse_board(raw_board) -> np.ndarray:
    """Validiert das Server-Board und wandelt es in ein uint8-Array um."""
    board = np.asarray(raw_board)

    if board.shape != (4, 4, 4):
        raise ValueError(f"Ungültige Board-Form: {board.shape}, erwartet (4, 4, 4)")

    if not np.isin(board, [0, 1, 2]).all():
        raise ValueError("Board enthält ungültige Werte.")

    return board.astype(np.uint8)
