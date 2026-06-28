import torch
import os
import sys

CURRENT_FILE = os.path.abspath(__file__)
SRC_CONNECT4_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE))
sys.path.append(SRC_CONNECT4_DIR)

from training_system.neural_network.model import Connect4Model

def main():
    model_path = os.path.join(SRC_CONNECT4_DIR, "training_system", "checkpoints", "v12_champion.pt")
    save_path = os.path.join(SRC_CONNECT4_DIR, "training_system", "checkpoints", "v12_champion.jit")
    
    print("Lade V12 Modell...")
    model = Connect4Model()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    print("Kompiliere Modell nach C++ (TorchScript)...")
    # Wandelt das Modell in statischen, optimierten Graphen um
    jit_model = torch.jit.script(model)
    jit_model.save(save_path)
    
    print(f"Erfolg! Kompiliertes Modell gespeichert unter: {save_path}")

if __name__ == "__main__":
    main()
