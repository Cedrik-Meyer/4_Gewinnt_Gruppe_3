import logging
import os
from typing import Optional

from shared.data_structures import GameState, Move
from shared.game_logic import apply_move, check_winner
from tools.mtc import MCTSEngine

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "training_system", "checkpoints", "best_champion.pt",
)


class LiveAgent:

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, cores: int = None):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Kein Checkpoint gefunden unter '{checkpoint_path}'. "
                "Der Live-Agent benötigt ein trainiertes Modell (best_champion.pt)."
            )

        self.engine = MCTSEngine(checkpoint_path, cores)
        logger.info(
            "LiveAgent mit MCTS-Engine initialisiert (Checkpoint '%s', %d Kerne).",
            checkpoint_path, self.engine.num_cores,
        )

    def select_move(self, game_state: GameState, time_limit_ms: int) -> Move:

        forced_move = self._find_forced_move(game_state)
        if forced_move is not None:
            logger.info("Erzwungener Zug erkannt (Sieg/Block): x=%d z=%d", forced_move.x, forced_move.z)
            return forced_move

        player = game_state.player_slot + 1
        return self.engine.get_engine_move(game_state.board, player, time_limit_ms)

    def close(self):
        self.engine.close()

    @staticmethod
    def _find_forced_move(game_state: GameState) -> Optional[Move]:

        own_value = game_state.player_slot + 1
        opponent_value = 2 if game_state.player_slot == 0 else 1

        block_move = None
        for z in range(4):
            for x in range(4):
                index = z * 4 + x
                if game_state.legal_mask[index] == 0:
                    continue

                own_board = game_state.board.copy()
                apply_move(own_board, Move(x=x, z=z), own_value)
                if check_winner(own_board, own_value):
                    return Move(x=x, z=z)

                if block_move is None:
                    opponent_board = game_state.board.copy()
                    apply_move(opponent_board, Move(x=x, z=z), opponent_value)
                    if check_winner(opponent_board, opponent_value):
                        block_move = Move(x=x, z=z)

        return block_move
