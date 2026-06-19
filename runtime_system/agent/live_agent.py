import logging
import os
import torch

from training_system.neural_network.model import Connect4Model

logger = logging.getLogger(__name__)

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