import logging
from typing import Optional

import numpy as np

from shared.data_structures import GameState

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
        player_slot = int(payload["playerSlot"])
        current_player = int(state["currentPlayer"])

        if player_slot not in (0, 1):
            raise ValueError(f"Ungültiger playerSlot: {player_slot}")

        if current_player not in (0, 1):
            raise ValueError(f"Ungültiger currentPlayer: {current_player}")

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


def _parse_board(raw_board) -> np.ndarray:
    """Validiert das Server-Board und wandelt es in ein uint8-Array um."""
    board = np.asarray(raw_board)

    if board.shape != (4, 4, 4):
        raise ValueError(f"Ungültige Board-Form: {board.shape}, erwartet (4, 4, 4)")

    if not np.isin(board, [0, 1, 2]).all():
        raise ValueError("Board enthält ungültige Werte.")

    return board.astype(np.uint8)
