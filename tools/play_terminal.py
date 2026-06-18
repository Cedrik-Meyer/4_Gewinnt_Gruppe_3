"""
tools/play_terminal.py

Ein interaktives Kommandozeilen-Spiel (CLI) für Mensch vs. Mensch,
Mensch vs. KI oder KI vs. KI. Dient als interaktiver Integrationstest.
Der Code ist strukturell in die 4 Haupt-Spielmodi unterteilt.
"""

import sys
import os
import glob
import time
import torch
import numpy as np

# Root-Verzeichnis zum Python-Pfad hinzufuegen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_structures import Move
from shared.game_logic import create_empty_board, apply_move, check_winner
from shared.state_encoder import encode_state, get_legal_mask
from training_system.neural_network.model import Connect4Model


# ==============================================================================
# 1. HILFSFUNKTIONEN (Spielfeld, Inferenz & Modell-Auswahl)
# ==============================================================================

def print_board(board: np.ndarray):
    """
    Druckt das 3D-Spielfeld menschenlesbar in das Terminal.
    Zeigt oben eine cleane Matrix-Hilfe für die (x z) Eingabe.
    Druckt von oben (y=3) nach unten (y=0).
    """
    COLORS = {
        0: ".",                     
        1: "\033[91mX\033[0m",      # Spieler 1 (Rot)
        2: "\033[94mO\033[0m"       # Spieler 2 (Blau)
    }
    
    print("\n" + "="*30)
    print("      4-GEWINNT 3D      ")
    print("="*30)
    
    # Koordinaten-Matrix (x z) ueber dem Spielfeld anzeigen
    print("\nEingabe-Koordinaten (x z):")
    print("0 0      1 0      2 0      3 0")
    print("0 1      1 1      2 1      3 1")
    print("0 2      1 2      2 2      3 2")
    print("0 3      1 3      2 3      3 3\n")
    
    # Die 4 Ebenen von oben nach unten drucken
    for y in reversed(range(4)):
        print(f"Ebene y = {y}")
        for z in range(4):
            row_str = ""
            for x in range(4):
                val = board[y][z][x]
                row_str += COLORS[val] + " "
            print(row_str.strip())
        print()

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

def choose_model(custom_prompt: str = "\nWelches Modell möchtest du laden? (Nummer eingeben): "):
    """
    Durchsucht den Checkpoint-Ordner und lässt den Nutzer ein Modell auswaehlen.
    Nimmt einen dynamischen Text entgegen, um bei KI vs. KI besser unterscheiden zu können.
    Gibt das geladene Modell und den Dateinamen zurueck.
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
    
    print("\nVerfügbare KI-Modelle:")
    for idx, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        print(f"[{idx + 1}] {filename}")
        
    while True:
        try:
            choice = int(input(custom_prompt))
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
    return model, os.path.basename(selected_file)

def get_human_move() -> Move:
    """Holt einen gueltigen Zug vom menschlichen Spieler und validiert das Format."""
    while True:
        user_input = input("Dein Zug (x z) oder 'q' zum Beenden: ").strip().lower()
        if user_input == 'q':
            print("Spiel abgebrochen.")
            sys.exit(0)
            
        try:
            parts = user_input.split()
            if len(parts) != 2:
                print("\n[!] FEHLER: Bitte genau zwei Zahlen mit Leerzeichen eingeben (z. B. '0 0').")
                continue
            x, z = int(parts[0]), int(parts[1])
            return Move(x=x, z=z)
        except ValueError as e:
            print(f"\n[!] UNGÜLTIGER ZUG: {e}")


# ==============================================================================
# 2. SPIELMODI LOGIK
# ==============================================================================

def play_human_vs_human():
    """Modus 1: Zwei Menschen spielen an der Konsole gegeneinander."""
    print("\nMensch vs. Mensch ausgewählt. Spieler 1 (Rot/X) fängt an.")
    board = create_empty_board()
    current_player = 1
    
    while True:
        print_board(board)
        if not np.any(board == 0):
            print("Das Spielfeld ist voll! Unentschieden!")
            break
            
        print(f"Spieler {current_player} ist am Zug.")
        move = get_human_move()
        apply_move(board, move, current_player)
            
        # Siegerkennung
        if check_winner(board, current_player):
            print_board(board)
            print("="*30)
            print(f" SPIELER {current_player} HAT GEWONNEN! Glückwunsch!")
            print("="*30)
            break
            
        # Spieler wechseln
        current_player = 2 if current_player == 1 else 1


def play_human_vs_ai():
    """Modus 2: Mensch spielt gegen ein ausgewaehltes KI-Modell."""
    ai_model, ai_filename = choose_model("\nWähle das KI-Modell (Nummer eingeben): ")
    
    while True:
        first = input("\nWer soll den ersten Zug machen? [1] Mensch [2] KI: ").strip()
        if first in ['1', '2']:
            break
        print("Ungültige Eingabe. Bitte 1 oder 2 wählen.")
        
    p1_is_ai = (first == '2')
    p2_is_ai = not p1_is_ai
    
    if p1_is_ai:
        print(f"\nDie KI ({ai_filename}) ist Spieler 1 (Rot/X) und fängt an. Du bist Spieler 2 (Blau/O).")
    else:
        print(f"\nDu bist Spieler 1 (Rot/X) und fängst an. Die KI ({ai_filename}) ist Spieler 2 (Blau/O).")
        
    board = create_empty_board()
    current_player = 1
    
    while True:
        print_board(board)
        if not np.any(board == 0):
            print("Das Spielfeld ist voll! Unentschieden!")
            break
            
        # Pruefen, ob die KI am Zug ist
        is_current_ai = (current_player == 1 and p1_is_ai) or (current_player == 2 and p2_is_ai)
        player_name = ai_filename if is_current_ai else "Mensch"
        print(f"{player_name} ist am Zug.")
        
        if is_current_ai:
            print("KI überlegt...")
            move = get_ai_move(board, ai_model, current_player)
            print(f"KI spielt: x={move.x}, z={move.z}")
            apply_move(board, move, current_player)
        else:
            move = get_human_move()
            apply_move(board, move, current_player)
            
        # Siegerkennung
        if check_winner(board, current_player):
            print_board(board)
            print("="*30)
            if is_current_ai:
                print(f" DIE KI ({ai_filename}) HAT GEWONNEN!")
            else:
                print(" DU HAST GEWONNEN! Glückwunsch!")
            print("="*30)
            break
            
        # Spieler wechseln
        current_player = 2 if current_player == 1 else 1


def play_ai_vs_ai_live():
    """Modus 3: Zwei KI-Modelle spielen live mit Print-Ausgabe im Terminal gegeneinander."""
    print("\n--- Auswahl KI A ---")
    model_A, filename_A = choose_model("\nWähle das erste KI-Modell (Nummer eingeben): ")
    print("\n--- Auswahl KI B ---")
    model_B, filename_B = choose_model("\nWähle das zweite KI-Modell (Nummer eingeben): ")
    
    while True:
        first = input(f"\nWer soll den ersten Zug machen? [1] {filename_A} oder [2] {filename_B}: ").strip()
        if first in ['1', '2']:
            break
        print("Ungültige Eingabe. Bitte 1 oder 2 wählen.")
        
    if first == '1':
        p1_model, p1_name = model_A, filename_A
        p2_model, p2_name = model_B, filename_B
    else:
        p1_model, p1_name = model_B, filename_B
        p2_model, p2_name = model_A, filename_A
        
    print(f"\n{p1_name} ist Spieler 1 (Rot/X) und fängt an. {p2_name} ist Spieler 2 (Blau/O).")
    
    board = create_empty_board()
    current_player = 1
    
    while True:
        print_board(board)
        if not np.any(board == 0):
            print("Das Spielfeld ist voll! Unentschieden!")
            break
            
        active_model = p1_model if current_player == 1 else p2_model
        active_name = p1_name if current_player == 1 else p2_name
        
        print(f"{active_name} ist am Zug. (Warte 1 Sekunde...)")
        time.sleep(1)  # Timer, damit das Spiel fuer den Menschen mitverfolgbar bleibt
        
        move = get_ai_move(board, active_model, current_player)
        print(f"KI spielt: x={move.x}, z={move.z}")
        apply_move(board, move, current_player)
            
        # Siegerkennung
        if check_winner(board, current_player):
            print_board(board)
            print("="*30)
            winner_filename = p1_name if current_player == 1 else p2_name
            loser_filename = p2_name if current_player == 1 else p1_name
            print(f"{winner_filename}  - Gewinner")
            print(f"{loser_filename}  - Verlierer")
            print("="*30)
            break
            
        # Spieler wechseln
        current_player = 2 if current_player == 1 else 1


def play_ai_vs_ai_auto():
    """Modus 4: Zwei KIs spielen 100 Runden im Hintergrund (Benchmark)."""
    print("\n--- Auswahl KI A ---")
    model_A, filename_A = choose_model("\nWähle das erste KI-Modell (Nummer eingeben): ")
    print("\n--- Auswahl KI B ---")
    model_B, filename_B = choose_model("\nWähle das zweite KI-Modell (Nummer eingeben): ")
    
    print(f"\nSpiele 100 Runden automatisiert ({filename_A} vs {filename_B})...")
    wins_A = 0
    wins_B = 0
    draws = 0
    
    for game_idx in range(100):
        board = create_empty_board()
        current_player = 1
        
        # Die ersten 50 Spiele faengt Modell A an, die naechsten 50 Spiele faengt Modell B an.
        # Dies verhindert den First-Mover-Advantage und sorgt fuer faire Winrates.
        if game_idx < 50:
            p1_m, p2_m = model_A, model_B
            p1_is_A = True
        else:
            p1_m, p2_m = model_B, model_A
            p1_is_A = False
            
        while True:
            active_model = p1_m if current_player == 1 else p2_m
            move = get_ai_move(board, active_model, current_player)
            apply_move(board, move, current_player)
            
            # Siegerkennung im Hintergrund
            if check_winner(board, current_player):
                if current_player == 1:
                    if p1_is_A: wins_A += 1
                    else: wins_B += 1
                else:
                    if p1_is_A: wins_B += 1
                    else: wins_A += 1
                break
                
            if not np.any(board == 0):
                draws += 1
                break
                
            current_player = 2 if current_player == 1 else 1
            
    winrate_A = (wins_A / 100) * 100
    winrate_B = (wins_B / 100) * 100
    
    print("\n--- Endergebnis (100 Spiele) ---")
    print(f"{filename_A} - Winrate = {winrate_A:.0f}%")
    print(f"{filename_B} - Winrate = {winrate_B:.0f}%")
    if draws > 0:
        print(f"Remis = {draws}%")


# ==============================================================================
# 3. HAUPTMENÜ
# ==============================================================================

def main():
    print("Willkommen zum Terminal-Test für Connect4 3D!")
    print("[1] Mensch vs. Mensch")
    print("[2] Mensch vs. KI")
    print("[3] KI vs. KI live")
    print("[4] KI vs. KI automatisiert")
    
    while True:
        mode = input("\nWähle den Spielmodus (1, 2, 3 oder 4): ").strip()
        if mode == '1':
            play_human_vs_human()
            break
        elif mode == '2':
            play_human_vs_ai()
            break
        elif mode == '3':
            play_ai_vs_ai_live()
            break
        elif mode == '4':
            play_ai_vs_ai_auto()
            break
        else:
            print("Ungültige Eingabe.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSpiel beendet.")
