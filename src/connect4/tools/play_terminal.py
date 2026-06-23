"""
tools/play_terminal.py

Ein interaktives Kommandozeilen-Spiel (CLI) für das finale Benchmarking
und Testen verschiedener KI-Ansätze.
Unterstützt Mensch, Pures Modell, Modell + MTC, Einfache Engine und Starke Engine.
Ressourcen (CPU-Kerne und Bedenkzeit) werden global verwaltet.
"""

import sys
import os
import glob
import time
import datetime
import torch
import numpy as np

# Root-Verzeichnis zum Python-Pfad hinzufügen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_structures import Move
from shared.game_logic import create_empty_board, apply_move, check_winner
from shared.state_encoder import encode_state, get_legal_mask
from training_system.neural_network.model import Connect4Model

# Externe Agenten-Engines importieren
from tools.weak_engine import WeakEngine
from tools.strong_engine import StrongEngine
from tools.mtc import MCTSEngine

# Sicherstellen, dass das log-Verzeichnis existiert
os.makedirs("logs", exist_ok=True)


# ==============================================================================
# 1. AGENTEN WRAPPER (Einheitliche Schnittstelle)
# ==============================================================================

class Agent:
    """Basisklasse für alle Spielertypen."""
    def __init__(self, name: str):
        self.name = name
        
    def get_move(self, board: np.ndarray, player: int) -> Move:
        raise NotImplementedError

class HumanAgent(Agent):
    """Liest die Eingabe über das Terminal ein und zeigt die Koordinaten-Hilfe."""
    def __init__(self):
        super().__init__("Mensch")
        
    def get_move(self, board: np.ndarray, player: int) -> Move:
        # Die optische Hilfe wird NUR gedruckt, wenn der Mensch am Zug ist!
        print("\n   --- EINGABE-HILFE (x z) ---")
        print("  [0 0]  [1 0]  [2 0]  [3 0]")
        print("  [0 1]  [1 1]  [2 1]  [3 1]")
        print("  [0 2]  [1 2]  [2 2]  [3 2]")
        print("  [0 3]  [1 3]  [2 3]  [3 3]")
        print("   ---------------------------\n")
        
        while True:
            user_input = input("Dein Zug (x z) oder 'q' zum Beenden: ").strip().lower()
            if user_input == 'q': 
                sys.exit(0)
            try:
                parts = user_input.split()
                if len(parts) != 2: 
                    raise ValueError("Bitte exakt zwei Zahlen (x z) eingeben.")
                return Move(x=int(parts[0]), z=int(parts[1]))
            except ValueError as e:
                print(f"[!] Ungültig: {e}")

class PureNNAgent(Agent):
    """Pures neuronales Netz (Forward-Pass ohne Baumsuche)."""
    def __init__(self, filepath: str, filename: str):
        super().__init__(f"Modell({filename})")
        self.model = Connect4Model()
        self.model.load_state_dict(torch.load(filepath, map_location='cpu', weights_only=True))
        self.model.eval()
        
    def get_move(self, board: np.ndarray, player: int) -> Move:
        legal_mask = get_legal_mask(board)
        state_tensor = encode_state(board, player - 1)
        with torch.no_grad():
            logits, _ = self.model(state_tensor.unsqueeze(0))
            
        masked_logits = logits.squeeze(0) + (1.0 - torch.tensor(legal_mask, dtype=torch.float32)) * -1e9
        action = torch.argmax(masked_logits).item()
        return Move(x=int(action % 4), z=int(action // 4))

class MTCAgent(Agent):
    """Monte Carlo Tree Search kombiniert mit dem neuronalen Netz."""
    def __init__(self, filepath: str, filename: str, time_limit: int, cores: int):
        super().__init__(f"Modell+MTC({filename})")
        self.engine = MCTSEngine(filepath)
        self.time_limit = time_limit
        self.cores = cores
        
    def get_move(self, board: np.ndarray, player: int) -> Move:
        return self.engine.get_engine_move(board, player, self.time_limit, self.cores)

class WeakEngineAgent(Agent):
    """Die schwache 1-Step Heuristik-Engine."""
    def __init__(self):
        self.engine = WeakEngine()
        super().__init__("Einfache Engine")
        
    def get_move(self, board: np.ndarray, player: int) -> Move:
        return self.engine.get_engine_move(board, player)

class StrongEngineAgent(Agent):
    """Die hochoptimierte Minimax-Engine mit Alpha-Beta-Pruning."""
    def __init__(self, time_limit: int, cores: int):
        self.engine = StrongEngine()
        super().__init__("Starke Engine")
        self.time_limit = time_limit
        self.cores = cores
        
    def get_move(self, board: np.ndarray, player: int) -> Move:
        return self.engine.get_engine_move(board, player, self.time_limit, self.cores)


# ==============================================================================
# 2. HILFSFUNKTIONEN (Input, Logging, UI)
# ==============================================================================

def get_int_input(prompt: str) -> int:
    """Sichere Funktion zum Einlesen von ganzen Zahlen."""
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("[!] Exception: Bitte eine ganze Zahl eingeben.")

def print_board(board: np.ndarray):
    """Druckt das Spielfeld im Terminal (ohne nervige Koordinaten-Texte)."""
    COLORS = {0: ".", 1: "\033[91mX\033[0m", 2: "\033[94mO\033[0m"}
    print("\n" + "="*30 + "\n      4-GEWINNT 3D      \n" + "="*30)
    
    for y in reversed(range(4)):
        print(f"Ebene y = {y}")
        for z in range(4):
            # Mit etwas mehr Abstand gedruckt, damit es zur Eingabe-Hilfe passt
            print("  " + "      ".join([COLORS[board[y][z][x]] for x in range(4)]))
        print()

def get_y_drop(board: np.ndarray, x: int, z: int) -> int:
    """Bestimmt die y-Ebene, auf der ein Stein landen wird."""
    for y in range(4):
        if board[y][z][x] == 0: 
            return y
    return -1

def board_to_string(board: np.ndarray) -> str:
    """Wandelt das Spielfeld in einen String für die Log-Datei um."""
    res = ""
    for y in reversed(range(4)):
        res += f"Ebene y = {y}\n"
        for z in range(4):
            res += " ".join(["." if board[y][z][x]==0 else "X" if board[y][z][x]==1 else "O" for x in range(4)]) + "\n"
        res += "\n"
    return res

def save_game_log(p1_name: str, p2_name: str, history: list, winner: int, final_board: np.ndarray, log_file: str, game_idx: int = 1):
    """Speichert ein einzelnes Spiel im übergebenen Log-Dokument."""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- SPIEL {game_idx} ---\n")
        f.write(f"Spieler 1 (Rot/X): {p1_name}\n")
        f.write(f"Spieler 2 (Blau/O): {p2_name}\n")
        f.write("--------------------\n")
        for i, (p_name, p_idx, x, z, y) in enumerate(history):
            f.write(f"Zug {i+1:02d} | {p_name} (Sp. {p_idx}) spielt -> x={x}, z={z}, y={y}\n")
        
        f.write("\n>>> GEWINNER: ")
        if winner == 1: f.write(f"{p1_name} (Spieler 1) <<<\n\n")
        elif winner == 2: f.write(f"{p2_name} (Spieler 2) <<<\n\n")
        else: f.write("UNENTSCHIEDEN <<<\n\n")
        
        f.write("Finales Spielfeld:\n")
        f.write(board_to_string(final_board))
        f.write("="*50 + "\n")

def choose_model_path():
    """Lässt den Nutzer ein Modell aus dem Checkpoint-Ordner wählen."""
    chk_dir = os.path.join("training_system", "checkpoints")
    files = sorted(glob.glob(os.path.join(chk_dir, "*.pt")))
    if not files:
        print("[!] Keine .pt Modelle gefunden im Ordner: training_system/checkpoints/")
        sys.exit(1)
    
    print("\nVerfügbare Modelle:")
    for idx, f in enumerate(files):
        print(f"[{idx + 1}] {os.path.basename(f)}")
        
    while True:
        choice = get_int_input("Modell-Nummer: ")
        if 1 <= choice <= len(files):
            selected = files[choice - 1]
            return selected, os.path.basename(selected)
        print("[!] Ungültige Auswahl.")

def setup_player(prompt_text: str, cores: int, time_limit: int) -> Agent:
    """Menü-Führung zur Auswahl eines Agenten."""
    print(f"\n{prompt_text}")
    print("\t[1] Mensch")
    print("\t[2] Modell")
    print("\t[3] Modell + MTC")
    print("\t[4] Einfache Engine")
    print("\t[5] Starke Engine")
    
    while True:
        choice = get_int_input("Eingabe: ")
        if choice == 1: 
            return HumanAgent()
        elif choice == 2: 
            filepath, fname = choose_model_path()
            return PureNNAgent(filepath, fname)
        elif choice == 3: 
            filepath, fname = choose_model_path()
            return MTCAgent(filepath, fname, time_limit, cores)
        elif choice == 4: 
            return WeakEngineAgent()
        elif choice == 5: 
            return StrongEngineAgent(time_limit, cores)
        else:
            print("[!] Exception: Bitte eine Zahl von 1 bis 5 eingeben.")

def clean_filename(name: str) -> str:
    """Entfernt Leerzeichen und Pfad-Fragmente für einen sauberen Dateinamen."""
    return name.split("(")[0].strip().replace(" ", "_").replace("/", "_").replace("+", "plus")


# ==============================================================================
# 3. SPIEL-STEUERUNG
# ==============================================================================

def play_game_logic(agent1: Agent, agent2: Agent, show_board: bool, log_file: str = None, game_idx: int = 1) -> int:
    """
    Führt ein einzelnes Spiel zwischen zwei Agenten durch.
    Gibt die Spielernummer des Gewinners zurück (1, 2) oder 0 für Unentschieden.
    """
    board = create_empty_board()
    current = 1
    history = []
    
    while True:
        if show_board: 
            print_board(board)
            
        if np.sum(get_legal_mask(board)) == 0:
            if show_board: print("Unentschieden!")
            if log_file: save_game_log(agent1.name, agent2.name, history, 0, board, log_file, game_idx)
            return 0
            
        active_agent = agent1 if current == 1 else agent2
        
        move = active_agent.get_move(board, current)
        y = get_y_drop(board, move.x, move.z)
        
        # Validierung des Zuges
        if y == -1 or get_legal_mask(board)[move.x + (move.z * 4)] == 0.0:
            if show_board: print(f"[!] ILLEGALER ZUG von {active_agent.name}! Disqualifikation.")
            winner = 2 if current == 1 else 1 
            if log_file: save_game_log(agent1.name, agent2.name, history, winner, board, log_file, game_idx)
            return winner
            
        history.append((active_agent.name, current, move.x, move.z, y))
        apply_move(board, move, current)
        
        if check_winner(board, current):
            if show_board:
                print_board(board)
            if log_file: save_game_log(agent1.name, agent2.name, history, current, board, log_file, game_idx)
            return current
            
        current = 2 if current == 1 else 1


# ==============================================================================
# 4. HAUPTMENÜ
# ==============================================================================

def main():
    print("Willkommen zum Terminal-Test für Connect4 3D!\n")
    
    print("Wie viele CPU-Kerne für jeden Spieler?")
    cores = get_int_input("Eingabe: ")
    torch.set_num_threads(cores)
    
    print("\nWie viel Zeit pro Zug in ms?")
    time_limit = get_int_input("Eingabe: ")
    
    print("\n[1]\tEinzelspiel (Live zuschauen)")
    print("[2]\tAuto-Benchmark (X Spiele simulieren)")
    
    while True:
        mode = get_int_input("Eingabe: ")
        if mode in [1, 2]: break
        print("[!] Exception: Bitte 1 oder 2 eingeben.")
        
    # --- MODUS 1: EINZELSPIEL ---
    if mode == 1:
        a1 = setup_player("Spieler 1 wählen:", cores, time_limit)
        print(f"Gewählt wurde {a1.name}")
        
        a2 = setup_player("Spieler 2 wählen:", cores, time_limit)
        print(f"Gewählt wurde {a2.name}")
        
        print(f"\nWer soll anfangen? [1] {a1.name} oder [2] {a2.name} ?")
        while True:
            starter = get_int_input("Eingabe: ")
            if starter in [1, 2]: break
            print("[!] Exception: Bitte 1 oder 2 eingeben.")
            
        # Tauschen, falls Spieler 2 anfangen soll
        if starter == 2:
            a1, a2 = a2, a1
            
        print("\nLivespiel.")
        winner = play_game_logic(a1, a2, show_board=True, log_file=None, game_idx=1)
        
        if winner == 1: print(f"Gewinner! ({a1.name})")
        elif winner == 2: print(f"Gewinner! ({a2.name})")
        else: print("Unentschieden!")
        
    # --- MODUS 2: AUTO-BENCHMARK ---
    elif mode == 2:
        print("\nWie viele Spiele sollen simuliert werden?")
        games = get_int_input("Eingabe: ")
        
        a1 = setup_player("Spieler 1 wählen:", cores, time_limit)
        print(f"Gewählt wurde {a1.name}")
        
        a2 = setup_player("Spieler 2 wählen:", cores, time_limit)
        print(f"Gewählt wurde {a2.name}")
        
        print("\nSimulation startet jetzt:")
        
        # Log-Datei vorbereiten
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n1 = clean_filename(a1.name)
        n2 = clean_filename(a2.name)
        log_filename = f"logs/benchmark_{n1}_vs_{n2}_{timestamp}.txt"
        
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(f"BENCHMARK START: {games} Spiele\n")
            f.write(f"Zeitlimit: {time_limit}ms | Kerne: {cores}\n")
            f.write(f"Agent 1: {a1.name}\nAgent 2: {a2.name}\n")
            f.write("="*50 + "\n")
        
        wins1, wins2, draws = 0, 0, 0
        
        for i in range(games):
            # Abwechselndes Startrecht für Balance
            if i % 2 == 0:
                w = play_game_logic(a1, a2, show_board=False, log_file=log_filename, game_idx=i+1)
                if w == 1: wins1 += 1
                elif w == 2: wins2 += 1
                else: draws += 1
            else:
                w = play_game_logic(a2, a1, show_board=False, log_file=log_filename, game_idx=i+1)
                if w == 1: wins2 += 1
                elif w == 2: wins1 += 1
                else: draws += 1
                
            sys.stdout.write(f"\rSimulation: {i+1}/{games} ")
            sys.stdout.flush()
            
        winrate1 = (wins1 / games) * 100
        winrate2 = (wins2 / games) * 100
        
        print("\n\nSimulation:")
        print(f"Winrate von {a1.name}: {winrate1:.1f}%")
        print(f"Winrate von {a2.name}: {winrate2:.1f}%")
        print(f"Spiel wurde unter dem Namen {os.path.basename(log_filename)} in logs/ gespeichert")
        
        # Endergebnis ans Ende der Log-Datei schreiben
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write("\n--- BENCHMARK ENDERGEBNIS ---\n")
            f.write(f"Winrate von {a1.name}: {winrate1:.1f}%\n")
            f.write(f"Winrate von {a2.name}: {winrate2:.1f}%\n")
            f.write(f"Remis: {(draws/games)*100:.1f}%\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbbruch.")
