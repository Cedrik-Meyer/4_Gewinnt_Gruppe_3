"""
training_system/supervised_train.py

Automatisierte Supervised-Learning-Pipeline für das 3D-Connect4-Modell.
Dieses Skript sucht eigenständig nach der aktuellsten Modellversion im 
Checkpoint-Ordner und trainiert iterativ die nächste Version (v+1).

Die Datengenerierung basiert auf dem "Master Mix"-Ansatz, einer Hierarchie aus:
1. Regelbasierten Zügen (Verhindern von trivialen Fehlern / 1-Turn-Loss).
2. Behavioral Cloning (Klonen von taktischen Engine-Zügen).
3. Knowledge Distillation (Erhalten der strategischen Intuition des Basis-Modells).

Hinweis zur Projekt-Timeline: 
Dieses Skript wurde in den Trainingsphasen 2 (v1-v9) und 4 (v10-v12) eingesetzt.
Die Laufzeit beträgt bei 15.000 generierten Spielen mehrere Stunden pro Iteration.
"""

import os
import re
import glob
import time
import torch
import numpy as np
import multiprocessing as mp
import sys
from torch.utils.data import Dataset, DataLoader

# Root-Verzeichnis zum Python-Pfad hinzufügen, um absolute Importe zu ermöglichen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_structures import Move
from shared.game_logic import create_empty_board, apply_move, check_winner
from shared.state_encoder import encode_state, get_legal_mask
from training_system.neural_network.model import Connect4Model
from tools.strong_engine import StrongEngine


# ==============================================================================
# 1. AUTOMATISCHE VERSIONSVERWALTUNG
# ==============================================================================

def get_latest_model_info():
    """
    Durchsucht den definierten Checkpoint-Ordner nach existierenden Modellen
    im Format 'vX_champion.pt'. 
    Gibt den Dateipfad des aktuellsten Modells und den Integer-Wert für 
    die darauffolgende Version zurück.
    """
    checkpoint_dir = "training_system/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(checkpoint_dir, "v*_champion.pt"))
    if not files:
        # Falls keine Modelle existieren, beginnt der Trainingszyklus bei Version 1.
        return None, 1  
        
    max_version = 0
    latest_file = ""
    
    # Extraktion der höchsten Versionsnummer via Regular Expressions
    for f in files:
        match = re.search(r'v(\d+)_champion\.pt', os.path.basename(f))
        if match:
            version = int(match.group(1))
            if version > max_version:
                max_version = version
                latest_file = f
                
    return latest_file, max_version + 1


def load_sim_model(checkpoint_path: str):
    """
    Lädt die Gewichte (Weights) eines existierenden PyTorch-Modells.
    Wird auf die CPU gemappt, da die Worker-Prozesse unabhängige Kopien benötigen.
    """
    model = Connect4Model()
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


# ==============================================================================
# 2. DATEN-GENERIERUNG (The Master Mix)
# ==============================================================================

def get_critical_move(board: np.ndarray, player: int):
    """
    Prüft das Spielfeld auf unmittelbare (1-Step) Bedrohungen oder Siegchancen.
    Dient als harter Filter, damit das neuronale Netz lernt, einfache 
    Flüchtigkeitsfehler unter allen Umständen zu vermeiden.
    """
    opponent = 2 if player == 1 else 1
    legal_mask = get_legal_mask(board)
    
    # Prüfung 1: Gibt es einen legalen Zug, der das Spiel sofort gewinnt?
    for i in range(16):
        if legal_mask[i] == 1.0:
            test_board = np.copy(board)
            apply_move(test_board, Move(x=i % 4, z=i // 4), player)
            if check_winner(test_board, player): 
                return i
            
    # Prüfung 2: Würde der Gegner im nächsten Zug gewinnen? (Zwingender Block)
    for i in range(16):
        if legal_mask[i] == 1.0:
            test_board = np.copy(board)
            apply_move(test_board, Move(x=i % 4, z=i // 4), opponent)
            if check_winner(test_board, opponent): 
                return i
            
    return None


def worker_play_master_mix(num_games: int, checkpoint_path: str):
    """
    Ausführungslogik für einen einzelnen CPU-Kern im Multiprocessing-Pool.
    Generiert selbstständig Spiele und zeichnet die Züge gemäß der Hierarchie auf.
    """
    model = load_sim_model(checkpoint_path)
    engine = StrongEngine()
    data = []
    
    # PyTorch nutzt standardmäßig Multi-Threading. Um Konflikte mit dem
    # Multiprocessing-Pool zu vermeiden, wird die Thread-Anzahl hier isoliert.
    torch.set_num_threads(1)
    
    for _ in range(num_games):
        board = create_empty_board()
        # Zufallsauswahl, welcher Spieler in dieser Partie die Engine darstellt
        engine_player = np.random.choice([1, 2])
        current_player = 1
        
        game_memory = []
        winner = 0
        
        while True:
            legal_mask = get_legal_mask(board)
            if np.sum(legal_mask) == 0:
                break
                
            critical_idx = get_critical_move(board, current_player)
            player_slot = current_player - 1
            state_tensor = encode_state(board, player_slot)
            
            # ---------------------------------------------------------
            # DIE DATEN-HIERARCHIE (Auswahl des optimalen Ziel-Vektors)
            # ---------------------------------------------------------
            
            if critical_idx is not None:
                # Priorität 1 (Hard Rules): Erzwingt einen 1-Hot-Vektor auf den Sieges-/Blockzug.
                target_probs = np.zeros(16, dtype=np.float32)
                target_probs[critical_idx] = 1.0
                action_idx = critical_idx
                data_type = "trap"
                
            elif current_player == engine_player:
                # Priorität 2 (Behavioral Cloning): Die Engine rechnet unter hohem Zeitdruck (2000ms).
                # Dies zwingt sie zu taktischen 2-bis-3-Step Lookaheads, welche das Modell kopiert.
                engine_move = engine.get_engine_move(board, current_player, time_limit_ms=2000, num_cores=1)
                action_idx = engine_move.x + (engine_move.z * 4)
                target_probs = np.zeros(16, dtype=np.float32)
                target_probs[action_idx] = 1.0
                data_type = "clone"
                
            else:
                # Priorität 3 (Knowledge Distillation): Wenn keine Gefahr besteht und die Engine
                # nicht am Zug ist, wird die weiche Wahrscheinlichkeitsverteilung des alten Netzes genutzt.
                # Dies schützt vor dem Überschreiben des grundlegenden Spielverständnisses (Catastrophic Forgetting).
                with torch.no_grad():
                    logits, _ = model(state_tensor.unsqueeze(0))
                    policy = logits.squeeze(0).numpy()
                    policy[legal_mask == 0.0] = -1e9
                    target_probs = torch.softmax(torch.tensor(policy), dim=0).numpy()
                    
                    # Boltzmann Exploration für Varianz in den Trainingsdaten
                    exp_preds = np.exp(policy / 1.0)
                    action_probs = exp_preds / np.sum(exp_preds)
                    action_idx = np.random.choice(16, p=action_probs)
                data_type = "distill"
            
            # Speichern des Status für die spätere Zuweisung des finalen Spielausgangs (Value)
            game_memory.append((state_tensor, target_probs, current_player, data_type))
            
            # Zug auf das reale Spielfeld anwenden
            x, z = action_idx % 4, action_idx // 4
            apply_move(board, Move(x=x, z=z), current_player)
            
            if check_winner(board, current_player):
                winner = current_player
                break
                
            current_player = 2 if current_player == 1 else 1
            
        # ---------------------------------------------------------
        # VALUE ZUWEISUNG (Post-Game Processing)
        # ---------------------------------------------------------
        for state, probs, p, d_type in game_memory:
            # Bewertung aus Perspektive des Spielers, der diesen Zug getätigt hat
            if winner == p: 
                val = 1.0
            elif winner == 0: 
                val = 0.0
            else: 
                val = -1.0
                
            # Filterung: Wenn ein von der Engine geklonter Zug zu einer Niederlage geführt hat,
            # wird dieser Datenpunkt verworfen, um fehlerhafte Taktiken nicht zu erlernen.
            if d_type == "clone" and val < 0:
                continue
                
            data.append((state, probs, val))
            
    return data


class MasterMixDataset(Dataset):
    """Wickelt die generierten Spieldaten in ein PyTorch-kompatibles Dataset-Format."""
    def __init__(self, data): 
        self.data = data
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx):
        state, probs, value = self.data[idx]
        return state, torch.tensor(probs), torch.tensor([value], dtype=torch.float32)


# ==============================================================================
# 3. HAUPTSCHLEIFE (Trainings-Orchestrierung)
# ==============================================================================

def main():
    latest_checkpoint, next_version = get_latest_model_info()
    
    print("==================================================")
    print(" 4-GEWINNT 3D - SUPERVISED LEARNING PIPELINE ")
    print("==================================================")
    print(f"Ziel-Version   : v{next_version}_champion.pt")
    
    if latest_checkpoint:
        print(f"Basis-Modell   : {os.path.basename(latest_checkpoint)}")
    else:
        print("Basis-Modell   : [Keines gefunden] Initialisiere mit Random Weights.")
    print("==================================================\n")
    
    # ---------------------------------------------------------
    # HARDWARE ERKENNUNG (NVIDIA CUDA / CPU)
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Nutze NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Keine NVIDIA GPU gefunden. Nutze CPU (Erheblich langsamer!).")
        
    model = Connect4Model()
    if latest_checkpoint:
        model.load_state_dict(torch.load(latest_checkpoint, map_location='cpu', weights_only=True))
        
    # ---------------------------------------------------------
    # TRAININGS-HYPERPARAMETER
    # ---------------------------------------------------------
    TOTAL_GAMES = 15000
    BATCH_SIZE = 512
    EPOCHS = 4
    LEARNING_RATE = 2e-5       # Niedrige Lernrate, um Feinjustierung (Finetuning) zu gewährleisten
    EARLY_STOP_LOSS = 0.35     # Schwellenwert zur Vermeidung von Overfitting
    
    # Reserviert einen Thread für das System, nutzt den Rest für Daten-Generierung
    cpu_cores = max(1, mp.cpu_count() - 1)
    
    print(f"\nSimuliere {TOTAL_GAMES} Trainings-Spiele auf {cpu_cores} logischen Kernen...")
    start_time = time.time()
    
    # Aufteilung der zu spielenden Partien auf die verfügbaren CPU-Kerne
    games_per_worker = TOTAL_GAMES // cpu_cores
    args = [(games_per_worker, latest_checkpoint) for _ in range(cpu_cores)]
    all_data = []
    
    # Start des synchronisierten Multiprocessing-Pools
    with mp.Pool(processes=cpu_cores) as pool:
        results = pool.starmap(worker_play_master_mix, args)
        for res in results:
            all_data.extend(res)
            
    print(f"Datensatz-Generierung abgeschlossen in {time.time() - start_time:.1f}s. (Züge im RAM: {len(all_data)})\n")
    
    print(f"Starte Training für v{next_version}_champion.pt...")
    dataset = MasterMixDataset(all_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model.to(device)
    model.train()
    
    # Initialisierung von Optimierer und Fehlerfunktionen (Loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    policy_loss_fn = torch.nn.CrossEntropyLoss() # Bewertet die Genauigkeit der Zug-Vorhersage
    value_loss_fn = torch.nn.MSELoss()           # Bewertet die Genauigkeit der Gewinn-Wahrscheinlichkeit
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for batch_states, batch_probs, batch_values in dataloader:
            batch_states = batch_states.to(device)
            batch_probs = batch_probs.to(device)
            batch_values = batch_values.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass: Das Modell berechnet Züge und Gewinnwahrscheinlichkeiten
            predicted_logits, predicted_values = model(batch_states)
            
            p_loss = policy_loss_fn(predicted_logits, batch_probs)
            v_loss = value_loss_fn(predicted_values, batch_values)
            
            # Kombination beider Verluste für die Backpropagation
            loss = p_loss + v_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoche {epoch:02d}/{EPOCHS} | Total Loss: {avg_loss:.4f}")
        
        # Frühzeitiger Abbruch, wenn das Modell konvergiert ist
        if avg_loss < EARLY_STOP_LOSS:
            print("\n[!] EARLY STOPPING ausgelöst (Ziel-Loss erreicht).")
            break
            
    # Rückführung auf die CPU und Speicherung der Gewichte
    model.cpu()
    save_path = f"training_system/checkpoints/v{next_version}_champion.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining erfolgreich abgeschlossen! Modell gespeichert als: '{save_path}'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer.")
