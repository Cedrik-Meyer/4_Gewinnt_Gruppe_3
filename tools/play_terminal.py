"""
tools/play_terminal.py

Ein interaktives Kommandozeilen-Spiel (CLI) für Mensch vs. Mensch 
oder Mensch vs. KI. Dient als interaktiver Integrationstest.
"""

import sys
import os
import glob
import torch
import numpy as np

# Root-Verzeichnis zum Python-Pfad hinzufuegen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_structures import Move
from shared.game_logic import create_empty_board, apply_move, check_winner
from shared.state_encoder import encode_state, get_legal_mask
from training_system.neural_network.model import Connect4Model

def print_board(board: np.ndarray):
    """
    Druckt das 3D-Spielfeld menschenlesbar in das Terminal.
    Druckt von oben (y=3) nach unten (y=0).
    """
    COLORS = {
        0: ".",                     
        1: "\033[91mX\033[0m",      # Spieler 1 (Rot)
        2: "\033[94mO\033[0m"       # Spieler 2 (Blau)
    }
    
    print("\n" + "="*30)
    print("4-GEWINNT 3D")
    print("="*30)
    
    for y in reversed(range(4)):
        print(f"\n[ Ebene y={y} ] (z=0 ist oben, z=3 ist unten)")
        for z in range(4):
            row_str = "  "
            for x in range(4):
                val = board[y][z][x]
                row_str += COLORS[val] + " "
            print(f"{row_str}  (z={z})")
            
    print("  ^ ^ ^ ^")
    print(" x:0 1 2 3\n")

def get_ai_move(board: np.ndarray, model: Connect4Model, current_player: int) -> Move:
    """
    Lässt das neuronale Netz den besten Zug berechnen (Inferenz-Pipeline).
    """
    player_slot = current_player - 1
    
    # 1. Spielfeld fuer das Netz codieren
    state_tensor = encode_state(board, player_slot)
    legal_mask = get_legal_mask(board)
    
    # 2. Forward-Pass durch das Netz
    with torch.no_grad():
        logits, _ = model(state_tensor.unsqueeze(0))
    logits = logits.squeeze(0)
    
    # 3. Illegale Zuege extrem stark bestrafen (Logit-Maskierung)
    mask_tensor = torch.tensor(legal_mask, dtype=torch.float32)
    masked_logits = logits + (1.0 - mask_tensor) * -1e9
    
    # 4. Den Index mit dem hoechsten Wert auswaehlen (0 bis 15)
    best_action = torch.argmax(masked_logits).item()
    
    # 5. Index in x- und z-Koordinaten umwandeln
    x = int(best_action % 4)
    z = int(best_action // 4)
    
    return Move(x=x, z=z)

def choose_model() -> Connect4Model:
    """
    Durchsucht den Checkpoint-Ordner und lässt den Nutzer ein Modell auswaehlen.
    """
    checkpoint_dir = os.path.join("training_system", "checkpoints")
    if not os.path.exists(checkpoint_dir):
        print(f"\n[!] FEHLER: Der Ordner {checkpoint_dir} existiert nicht.")
        sys.exit(1)
        
    files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not files:
        print(f"\n[!] FEHLER: Keine Modelle (.pt Dateien) im Ordner {checkpoint_dir} gefunden.")
        sys.exit(1)
        
    # Dateien alphabetisch sortieren fuer eine saubere Anzeige
    files.sort()
    
    print("\nVerfuegbare KI-Modelle:")
    for idx, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        print(f"[{idx + 1}] {filename}")
        
    while True:
        try:
            choice = int(input("\nWelches Modell möchtest du laden? (Nummer eingeben): "))
            if 1 <= choice <= len(files):
                selected_file = files[choice - 1]
                break
            else:
                print("Ungültige Nummer.")
        except ValueError:
            print("Bitte eine Zahl eingeben.")
            
    print(f"\nLade Modell: {os.path.basename(selected_file)} ...")
    model = Connect4Model()
    model.load_state_dict(torch.load(selected_file, weights_only=True))
    model.eval()  # Wichtig: Modell in den Inferenz-Modus setzen
    return model

def main():
    print("Willkommen zum Terminal-Test fuer Connect4 3D!")
    print("[1] Mensch vs. Mensch")
    print("[2] Mensch vs. KI")
    
    while True:
        mode = input("Wähle den Spielmodus (1 oder 2): ").strip()
        if mode in ['1', '2']:
            break
        print("Ungültige Eingabe.")
        
    ai_model = None
    if mode == '2':
        ai_model = choose_model()
        print("\nDu bist Spieler 1 (Rot/X). Die KI ist Spieler 2 (Blau/O).")
    
    print("Geben Sie Ihre Züge im Format 'x z' ein (z. B. '1 2').")
    
    board = create_empty_board()
    current_player = 1
    
    while True:
        print_board(board)
        
        if not np.any(board == 0):
            print("Das Spielfeld ist voll! Unentschieden!")
            break
            
        print(f"Spieler {current_player} ist am Zug.")
        
        # Pruefen, ob die KI am Zug ist
        is_ai_turn = (mode == '2' and current_player == 2)
        
        if is_ai_turn and ai_model is not None:
            print("KI überlegt...")
            move = get_ai_move(board, ai_model, current_player)
            print(f"KI spielt: x={move.x}, z={move.z}")
            apply_move(board, move, current_player)
        else:
            # Menschlicher Zug
            user_input = input("Dein Zug (x z) oder 'q' zum Beenden: ").strip().lower()
            
            if user_input == 'q':
                print("Spiel abgebrochen.")
                break
                
            try:
                parts = user_input.split()
                if len(parts) != 2:
                    print("\n[!] FEHLER: Bitte genau zwei Zahlen mit Leerzeichen eingeben (z. B. '0 0').")
                    continue
                    
                x, z = int(parts[0]), int(parts[1])
                move = Move(x=x, z=z)
                apply_move(board, move, current_player)
                
            except ValueError as e:
                print(f"\n[!] UNGÜLTIGER ZUG: {e}")
                continue
                
        # Siegerkennung
        if check_winner(board, current_player):
            print_board(board)
            print("="*30)
            if is_ai_turn:
                print(" DIE KI HAT GEWONNEN! Skynet lästt grüßen.")
            else:
                print(f" SPIELER {current_player} HAT GEWONNEN! Glückwunsch!")
            print("="*30)
            break
            
        # Spieler wechseln
        current_player = 2 if current_player == 1 else 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSpiel beendet.")
