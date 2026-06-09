"""
tools/play_terminal.py

Ein interaktives Kommandozeilen-Spiel (CLI) für 2 menschliche Spieler.
Dient als interaktiver Integrationstest für Block 1 (Spiellogik).
"""

import sys
import numpy as np

# Wir fügen das Root-Verzeichnis zum Python-Pfad hinzu, 
# falls jemand das Skript direkt aus dem Ordner startet.
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_structures import Move
from shared.game_logic import create_empty_board, apply_move, check_winner

def print_board(board: np.ndarray):
    """
    Druckt das 3D-Spielfeld menschenlesbar in das Terminal.
    Druckt von oben (y=3) nach unten (y=0).
    """
    # ANSI Color Codes für die Steine
    COLORS = {
        0: ".",                     # Leer
        1: "\033[91mX\033[0m",      # Spieler 1 (Rot)
        2: "\033[94mO\033[0m"       # Spieler 2 (Blau)
    }
    
    print("\n" + "="*30)
    print("      4-GEWINNT 3D      ")
    print("="*30)
    
    # Iteriere von Ebene 3 abwärts bis 0
    for y in reversed(range(4)):
        print(f"\n[ Ebene y={y} ] (z=0 ist oben, z=3 ist unten)")
        for z in range(4):
            row_str = "  "
            for x in range(4):
                val = board[y][z][x]
                row_str += COLORS[val] + " "
            # Zeige die z-Koordinate als kleine Hilfe am Rand
            print(f"{row_str}  (z={z})")
            
    print("  ^ ^ ^ ^")
    print(" x:0 1 2 3\n")

def main():
    print("Willkommen zum Terminal-Test für Connect4 3D!")
    print("Geben Sie Ihre Züge im Format 'x z' ein (z. B. '1 2').")
    
    board = create_empty_board()
    current_player = 1
    
    while True:
        print_board(board)
        
        # Unentschieden prüfen (Gibt es noch leere Felder auf dem ganzen Board?)
        if not np.any(board == 0):
            print("Das Spielfeld ist voll! Unentschieden!")
            break
            
        print(f"Spieler {current_player} ist am Zug.")
        user_input = input("Dein Zug (x z) oder 'q' zum Beenden: ").strip().lower()
        
        if user_input == 'q':
            print("Spiel abgebrochen.")
            break
            
        try:
            # Eingabe splitten und in Integers umwandeln
            parts = user_input.split()
            if len(parts) != 2:
                print("\n[!] FEHLER: Bitte genau zwei Zahlen mit Leerzeichen eingeben (z. B. '0 0').")
                continue
                
            x, z = int(parts[0]), int(parts[1])
            
            # Zug erstellen und anwenden
            move = Move(x=x, z=z)
            apply_move(board, move, current_player)
            
        except ValueError as e:
            # Fängt Buchstaben, Koordinaten > 3 oder volle Säulen ab
            print(f"\n[!] UNGÜLTIGER ZUG: {e}")
            continue
            
        # Siegerkennung prüfen
        if check_winner(board, current_player):
            print_board(board)
            print("="*30)
            print(f" SPIELER {current_player} HAT GEWONNEN!")
            print("="*30)
            break
            
        # Spieler wechseln
        current_player = 2 if current_player == 1 else 1

if __name__ == "__main__":
    # Verhindert Fehler, wenn der User mit Strg+C abbrechen will
    try:
        main()
    except KeyboardInterrupt:
        print("\nSpiel beendet.")
