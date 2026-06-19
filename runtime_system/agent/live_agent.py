import logging
import os
import torch

from shared.data_structures import GameState, Move
from shared.game_logic import apply_move, check_winner
from shared.state_encoder import encode_state
from training_system.neural_network.model import Connect4Model

logger = logging.getLogger(__name__)

ILLEGAL_MOVE_PENALTY = 1e9
OWN_WIN_BONUS = 2e6
BLOCK_WIN_BONUS = 1e6

DEFAULT_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "training_system", "checkpoints", "best_champion.pt",
)


class LiveAgent:
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> Connect4Model:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Kein Checkpoint gefunden unter '{checkpoint_path}'. "
                "Der Live-Agent benötigt ein trainiertes Modell (best_champion.pt)."
            )

        model = Connect4Model()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Champion Modell wird geladen
        # Zwingt auf CPU zu laufen, um Absturz zu vermeiden
        # Verhindert laden vom Code beim Laden vom PyTorch Modell
        model.load_state_dict(state_dict)
        model.eval() # Aktiviert Wettkampfsmodus

        logger.info("Modell erfolgreich geladen aus '%s'.", checkpoint_path)
        return model

    def predict(self, game_state: GameState):

        state_tensor = encode_state(game_state.board, game_state.player_slot)
        # Ebene B (Spielbrett & aktueller Spieler) in Ebene C (mathematischer Tensor) übersetzen
        batch_tensor = state_tensor.unsqueeze(0)
        # fügt eine Dimension hinzu: aus 1 Brett wird 1 Batch <-- PyTorch erwartet Batch

        with torch.no_grad():
        # keine Gradienten-Berechnung (Gedächtnis), da nicht weiter gelernt wird
            policy_logits, value = self.model(batch_tensor)
            # Logits (Bewertung der einzelnen Spalten) werden ausgegeben

        return policy_logits.squeeze(0), value.squeeze(0)
        # Value: Bewertung des gesamten Bretts
        # Batch wird wieder entfernt

    def mask_illegal_moves(self, policy_logits: torch.Tensor, legal_mask) -> torch.Tensor:

        legal_mask_tensor = torch.as_tensor(legal_mask, dtype=torch.float32)
        # volle Spalten werden mit 0 ausgegeben 
        illegal_positions = (legal_mask_tensor == 0)


        masked_logits = policy_logits.clone()
        # Clone wird erstellt, um Fehler bei der Speicherverwaltung zu verhindern
        masked_logits[illegal_positions] -= ILLEGAL_MOVE_PENALTY

        return masked_logits

    def apply_forced_moves(self, logits: torch.Tensor, game_state: GameState) -> torch.Tensor:

        own_value = game_state.player_slot + 1
        # Index Spieler: 0,1; Index Steine: 1,2
        # player_slot: Index des Spielers der am Zug ist (wir) + 1 = Index unserer Steine
        opponent_value = 2 if game_state.player_slot == 0 else 1
        # Index der Steine des Gegners

        boosted_logits = logits.clone()

        for z in range(4):
            for x in range(4):
                index = z * 4 + x
                if game_state.legal_mask[index] == 0:
                    continue
                # Abbruch, wenn Spalte voll ist

                own_board = game_state.board.copy()
                apply_move(own_board, Move(x=x, z=z), own_value)
                if check_winner(own_board, own_value):
                    boosted_logits[index] += OWN_WIN_BONUS
                    continue

                opponent_board = game_state.board.copy()
                apply_move(opponent_board, Move(x=x, z=z), opponent_value)
                if check_winner(opponent_board, opponent_value):
                    boosted_logits[index] += BLOCK_WIN_BONUS

        return boosted_logits

    def select_action(self, masked_logits: torch.Tensor) -> Move:

        best_index = int(torch.argmax(masked_logits).item())
        z = best_index // 4 # 9 // 4 = 2
        x = best_index % 4 # 9 // 4 = 2: Rest 1

        return Move(x=x, z=z)