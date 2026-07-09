import torch
import os
import sys
import glob

# ==============================================================================
# 1. PFAD-KONFIGURATION
# ==============================================================================
CURRENT_FILE = os.path.abspath(__file__)
SRC_CONNECT4_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE))
sys.path.append(SRC_CONNECT4_DIR)

from training_system.neural_network.model import Connect4Model

# ==============================================================================
# 2. HILFSFUNKTIONEN
# ==============================================================================

def get_int_input(prompt: str) -> int:
    """Sichere Funktion zum Einlesen von ganzen Zahlen."""
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("[!] Exception: Bitte eine ganze Zahl eingeben.")

def choose_model_path():
    """Lässt den Nutzer ein Modell aus dem Checkpoint-Ordner wählen."""
    chk_dir = os.path.join(SRC_CONNECT4_DIR, "training_system", "checkpoints")
    files = sorted(glob.glob(os.path.join(chk_dir, "*.pt")))
    
    if not files:
        print(f"[!] Keine .pt Modelle gefunden im Ordner: {chk_dir}")
        sys.exit(1)
    
    print("\nVerfügbare Modelle zur Kompilierung:")
    for idx, f in enumerate(files):
        print(f"[{idx + 1}] {os.path.basename(f)}")
        
    while True:
        choice = get_int_input("Modell-Nummer wählen: ")
        if 1 <= choice <= len(files):
            selected = files[choice - 1]
            return selected
        print("[!] Ungültige Auswahl.")

# ==============================================================================
# 3. HAUPTLOGIK
# ==============================================================================

def main():
    print("=== Connect4 JIT-Export Tool ===")
    
    # 1. Modell auswählen
    model_path = choose_model_path()
    
    # 2. Zielpfad automatisch bestimmen (.pt -> .jit)
    save_path = model_path.replace(".pt", ".jit")
    
    print(f"\nLade Modell: {os.path.basename(model_path)}...")
    
    try:
        model = Connect4Model()
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        
        print("Kompiliere Modell nach C++ (TorchScript)...")
        # Wandelt das Modell in statischen, optimierten Graphen um
        jit_model = torch.jit.script(model)
        
        print(f"Speichere unter: {save_path}...")
        jit_model.save(save_path)
        
        print(f"\n[OK] Erfolg! Modell erfolgreich kompiliert.")
        
    except Exception as e:
        print(f"\n[!] Fehler beim Export: {e}")

if __name__ == "__main__":
    main()
