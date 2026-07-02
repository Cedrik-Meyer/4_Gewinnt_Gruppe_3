"""
training_system/supervised_train.py

Automatisierte Supervised-Learning-Pipeline für das 3D-Connect4-Modell.

Dieses Modul implementiert einen hybriden Datengenerierungs- und Optimierungszyklus
zur Vermeidung von 'Catastrophic Forgetting' taktischer Grundprinzipien. 
Die Architektur extrahiert die aktuellste Modelliteration aus dem Persistenz-Layer
und trainiert iterativ die Nachfolgeversion (v+1).

Die Datengenerierung basiert auf dem "Master Mix"-Ansatz, einer hierarchischen Synthese aus:
1. Deterministischen Heuristiken (Prävention von terminalen 1-Turn-Loss-Zuständen).
2. Behavioral Cloning (Approximation tiefer algorithmischer Suchbäume durch Imitation).
3. Knowledge Distillation (Erhalt der Soft-Target-Verteilung des prä-trainierten Basis-Modells).

Hinweis zur Projekt-Timeline: 
Dieses Skript definierte die massiven Supervised-Finetuning-Phasen 2 (v1-v9) und 4 (v10-v12).
Die Laufzeit beträgt bei 15.000 simulierten Trajectories mehrere Stunden pro Iteration.
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

# Modulpfad-Registrierung für absolute Import-Referenzen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_structures import Move
from shared.game_logic import create_empty_board, apply_move, check_winner
from shared.state_encoder import encode_state, get_legal_mask
from training_system.neural_network.model import Connect4Model
from tools.strong_engine import StrongEngine


# ==============================================================================
# 1. AUTOMATISCHE VERSIONSVERWALTUNG & PERSISTENZ
# ==============================================================================

def get_latest_model_info():
    """
    Analysiert das Checkpoint-Verzeichnis nach existierenden Modell-Iterationen
    im Format 'vX_champion.pt'.
    
    Returns:
        Tuple[str, int]: Relativer Dateipfad der aktuellsten Iteration und 
                         die determinierte Ziel-Versionsnummer (v+1).
    """
    checkpoint_dir = "training_system/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(checkpoint_dir, "v*_champion.pt"))
    if not files:
        # Fallback: Bootstrapping bei Iteration 1 ohne Pre-Trained State
        return None, 1  
        
    max_version = 0
    latest_file = ""
    
    # Extraktion der maximalen Versions-ID via Regular Expressions
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
    Deserialisiert die trainierbaren Parameter eines PyTorch-Modells.
    Allokiert das Modell zwingend im Host-RAM (CPU), um Speicherzugriffskonflikte 
    während der asynchronen Subprozess-Generierung zu verhindern.
    """
    model = Connect4Model()
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


# ==============================================================================
# 2. DATEN-GENERIERUNG (Hybrider 'Master Mix')
# ==============================================================================

def get_critical_move(board: np.ndarray, player: int):
    """
    Deterministische Heuristik zur Vermeidung von 1-Turn-Loss-Szenarien.
    Evaluiert den aktuellen Zustandsraum auf unmittelbare terminale Folgezustände 
    (direkte Siege oder zwingende Blöcke).
    """
    opponent = 2 if player == 1 else 1
    legal_mask = get_legal_mask(board)
    
    # 1. Prädiktion terminaler Gewinne
    for i in range(16):
        if legal_mask[i] == 1.0:
            test_board = np.copy(board)
            apply_move(test_board, Move(x=i % 4, z=i // 4), player)
            if check_winner(test_board, player): 
                return i
            
    # 2. Prädiktion zwingender Verlust-Präventionen
    for i in range(16):
        if legal_mask[i] == 1.0:
            test_board = np.copy(board)
            apply_move(test_board, Move(x=i % 4, z=i // 4), opponent)
            if check_winner(test_board, opponent): 
                return i
            
    return None


def worker_play_master_mix(num_games: int, checkpoint_path: str):
    """
    Subprozess-Logik zur stochastischen Generierung von Trainingsdaten.
    Konstruiert die Ziel-Verteilungen (Policy Targets) basierend auf einer strikten 
    3-stufigen Hierarchie.
    """
    model = load_sim_model(checkpoint_path)
    engine = StrongEngine()
    data = []
    
    # Isolierung der PyTorch-Threads zur Vermeidung von Multiprocessing-Deadlocks
    torch.set_num_threads(1)
    
    for _ in range(num_games):
        board = create_empty_board()
        # Stochastische Zuweisung der algorithmischen Engine zu einer Spieler-Rolle
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
            # HIERARCHISCHE ZIEL-SYNTHESE (Target Vector Generation)
            # ---------------------------------------------------------
            
            if critical_idx is not None:
                # Priorität 1: Deterministische Terminalitäts-Prüfung (Hard Rules)
                # Generiert einen absoluten 1-Hot-Vektor zur Forcierung des korrekten Zuges.
                target_probs = np.zeros(16, dtype=np.float32)
                target_probs[critical_idx] = 1.0
                action_idx = critical_idx
                data_type = "trap"
                
            elif current_player == engine_player:
                # Priorität 2: Imitation Learning (Behavioral Cloning)
                # Die Engine berechnet unter Restriktion (2000ms) einen hoch-taktischen Zug.
                # Das Netzwerk lernt, diese baumbasierte Suchtiefe zu approximieren.
                engine_move = engine.get_engine_move(board, current_player, time_limit_ms=2000, num_cores=1)
                action_idx = engine_move.x + (engine_move.z * 4)
                target_probs = np.zeros(16, dtype=np.float32)
                target_probs[action_idx] = 1.0
                data_type = "clone"
                
            else:
                # Priorität 3: Knowledge Distillation
                # Transferiert die 'weichen' Vorhersagen (Soft Targets) des Basis-Modells.
                # Dies stabilisiert die Loss-Funktion und erhält die bereits erlernte Generalisierung.
                with torch.no_grad():
                    logits, _ = model(state_tensor.unsqueeze(0))
                    policy = logits.squeeze(0).numpy()
                    # Maskierung nicht-legaler Aktionen vor der Softmax-Transformation
                    policy[legal_mask == 0.0] = -1e9
                    target_probs = torch.softmax(torch.tensor(policy), dim=0).numpy()
                    
                    # Boltzmann Exploration zur Gewährleistung der Zustandsraum-Abdeckung (State-Space Coverage)
                    exp_preds = np.exp(policy / 1.0)
                    action_probs = exp_preds / np.sum(exp_preds)
                    action_idx = np.random.choice(16, p=action_probs)
                data_type = "distill"
            
            # Temporäre Aggregation der Transition für das finale Credit Assignment
            game_memory.append((state_tensor, target_probs, current_player, data_type))
            
            # Ausführung des physikalischen Zustandsübergangs
            x, z = action_idx % 4, action_idx // 4
            apply_move(board, Move(x=x, z=z), current_player)
            
            if check_winner(board, current_player):
                winner = current_player
                break
                
            current_player = 2 if current_player == 1 else 1
            
        # ---------------------------------------------------------
        # CREDIT ASSIGNMENT & TRAJECTORY-FILTERUNG
        # ---------------------------------------------------------
        for state, probs, p, d_type in game_memory:
            # Skalare Repräsentation des finalen Spielausgangs
            if winner == p: 
                val = 1.0
            elif winner == 0: 
                val = 0.0
            else: 
                val = -1.0
                
            # False-Negative Filterung: Verwurf von imitierten Engine-Datenpunkten,
            # falls diese wider Erwarten in einer deterministischen Niederlage resultierten.
            if d_type == "clone" and val < 0:
                continue
                
            data.append((state, probs, val))
            
    return data


class MasterMixDataset(Dataset):
    """
    Datenstruktur zur Bereitstellung der Transitions für die PyTorch DataLoader-API.
    Konvertiert die iterierbaren Numpy-Strukturen in hardware-optimierte Tensoren.
    """
    def __init__(self, data): 
        self.data = data
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx):
        state, probs, value = self.data[idx]
        return state, torch.tensor(probs), torch.tensor([value], dtype=torch.float32)


# ==============================================================================
# 3. HAUPTSCHLEIFE (Trainings-Orchestrierung & Joint Optimization)
# ==============================================================================

def main():
    latest_checkpoint, next_version = get_latest_model_info()
    
    print("==================================================")
    print(" 4-GEWINNT 3D - SUPERVISED LEARNING PIPELINE ")
    print("==================================================")
    print(f"Ziel-Iteration : v{next_version}_champion.pt")
    
    if latest_checkpoint:
        print(f"Basis-Modell   : {os.path.basename(latest_checkpoint)}")
    else:
        print("Basis-Modell   : [Pre-Trained State fehlt] Initialisiere Stochastisch.")
    print("==================================================\n")
    
    # ---------------------------------------------------------
    # HARDWARE-ALLOKATION (NVIDIA CUDA / CPU)
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Hardwarebeschleunigung aktiv: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Keine CUDA-kompatible GPU detektiert. Führe Optimierung auf CPU aus.")
        
    model = Connect4Model()
    if latest_checkpoint:
        model.load_state_dict(torch.load(latest_checkpoint, map_location='cpu', weights_only=True))
        
    # ---------------------------------------------------------
    # TRAININGS-HYPERPARAMETER (Finetuning Regime)
    # ---------------------------------------------------------
    TOTAL_GAMES = 15000
    BATCH_SIZE = 512
    EPOCHS = 4
    LEARNING_RATE = 2e-5       # Niedrige Lernrate zur Vermeidung der Gradienten-Destabilisierung beim Finetuning
    EARLY_STOP_LOSS = 0.35     # Schwellenwert zur Prävention von Überanpassung (Overfitting)
    
    # Reservierung eines System-Threads, Nutzung der restlichen Kapazität für die Datengenerierung
    cpu_cores = max(1, mp.cpu_count() - 1)
    
    print(f"\nGeneriere {TOTAL_GAMES} Trajectories verteilt auf {cpu_cores} asynchrone Threads...")
    start_time = time.time()
    
    games_per_worker = TOTAL_GAMES // cpu_cores
    args = [(games_per_worker, latest_checkpoint) for _ in range(cpu_cores)]
    all_data = []
    
    # Synchronisierte Ausführung der asynchronen Datenakquise
    with mp.Pool(processes=cpu_cores) as pool:
        results = pool.starmap(worker_play_master_mix, args)
        for res in results:
            all_data.extend(res)
            
    print(f"Datenakquise abgeschlossen in {time.time() - start_time:.1f}s. (Transitions im RAM: {len(all_data)})\n")
    
    print(f"Initiiere Gradient Descent Optimierung für v{next_version}_champion.pt...")
    dataset = MasterMixDataset(all_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model.to(device)
    model.train()
    
    # Optimierer mit L2-Regularisierung (Weight Decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Formulierung der kombinierten Fehlerfunktion (Joint Loss)
    policy_loss_fn = torch.nn.CrossEntropyLoss() # Klassifikations-Loss für die Stochastik der Zugwahl
    value_loss_fn = torch.nn.MSELoss()           # Regressions-Loss (Mean Squared Error) für die skalare Bewertung
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for batch_states, batch_probs, batch_values in dataloader:
            batch_states = batch_states.to(device)
            batch_probs = batch_probs.to(device)
            batch_values = batch_values.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass: Berechnung der Prädiktionen des Dual-Head CNN
            predicted_logits, predicted_values = model(batch_states)
            
            p_loss = policy_loss_fn(predicted_logits, batch_probs)
            v_loss = value_loss_fn(predicted_values, batch_values)
            
            # Lineare Kombination beider Verluste für die Backpropagation
            loss = p_loss + v_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoche {epoch:02d}/{EPOCHS} | Aggregierter Joint Loss: {avg_loss:.4f}")
        
        # Regularisierung durch Early Stopping bei empirisch definierter Konvergenz
        if avg_loss < EARLY_STOP_LOSS:
            print("\n[!] Konvergenzschwellenwert unterschritten (Early Stopping ausgelöst).")
            break
            
    # Rückführung des optimierten Modells in den Host-Speicher und Persistierung
    model.cpu()
    save_path = f"training_system/checkpoints/v{next_version}_champion.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nTrainingsiteration erfolgreich abgeschlossen. Artefakt persistiert als: '{save_path}'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nManuelle Prozess-Terminierung durch den Benutzer.")
