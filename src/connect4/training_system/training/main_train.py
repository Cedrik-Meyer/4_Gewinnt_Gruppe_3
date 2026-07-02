"""
training_system/main_train.py

Zentraler Orchestrator für die Reinforcement-Learning-Pipeline (AlphaZero-Architektur)
des Connect4-3D-Agenten. 

Dieses Modul verwaltet den kontinuierlichen Verbesserungszyklus bestehend aus:
1. Asynchroner Datengenerierung via Self-Play (CPU-Multiprocessing).
2. Mini-Batch Gradient Descent Training (GPU/CUDA-Beschleunigung).
3. Rigoroser Candidate-Evaluierung in der Arena.
"""

import os
import glob
import re
import csv
import time
import torch
import multiprocessing as mp

from training_system.neural_network.model import Connect4Model
from training_system.self_play.replay_buffer import ReplayBuffer
from training_system.self_play.self_play_loop import play_single_game, store_game_trajectory
from training_system.training.trainer import Connect4Trainer
from training_system.eval.arena import evaluate_candidate


def worker_self_play(state_dict: dict, num_games: int):
    """
    Subprozess für die dezentrale Generierung von Trainingsdaten.
    
    Jeder Worker isoliert sein eigenes Inferenz-Modell im Arbeitsspeicher,
    um Race Conditions bei der Gewichtungsabfrage durch die parallelen 
    MCTS-Instanzen zu vermeiden.
    
    Args:
        state_dict (dict): Die serialisierten Parameter (Gewichte) des aktuellen Champion-Modells.
        num_games (int): Die Anzahl der pro Thread zu absolvierenden Partien.
        
    Returns:
        list: Eine chronologische Aufzeichnung der Spielverläufe (Trajectories) 
              und deren finale Resultate.
    """
    # 1. Instanziierung eines lokalen, unabhängigen Evaluators
    local_model = Connect4Model()
    local_model.load_state_dict(state_dict)
    local_model.eval()
    
    results = []
    # 2. Ausführung der stochastischen Self-Play-Spiele
    for _ in range(num_games):
        trajectory, winner = play_single_game(local_model)
        results.append((trajectory, winner))
        
    return results


def main_training_loop():
    """
    Die zentrale Steuerungsschleife (Main Event Loop) für das Reinforcement Learning.
    Verwaltet das Speicher-Management, das Checkpointing und die Hardware-Allokation
    für die Iterationen des AlphaZero-Zyklus.
    """
    # Sicherstellung der Prozess-Isolation, um Deadlocks unter Unix/Windows zu verhindern
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
        
    print("Initialisiere verteilte ML-Trainingsumgebung ...")
    
    # ---------------------------------------------------------
    # 1. Hardware-Allokation (CUDA / CPU Fallback)
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[!] Hardwarebeschleunigung aktiv: {torch.cuda.get_device_name(0)} allokiert.\n")
    else:
        print("[!] Keine CUDA-kompatible GPU detektiert. Führe Backpropagation auf CPU aus.\n")

    # ---------------------------------------------------------
    # 2. Persistenz und Logging-Struktur
    # ---------------------------------------------------------
    checkpoint_dir = "training_system/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "best_champion.pt")
    metrics_path = os.path.join(checkpoint_dir, "training_metrics.csv")
    
    # Anlage der zentralen Telemetrie-Datei (falls nicht existent)
    if not os.path.isfile(metrics_path):
        with open(metrics_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Buffer_Size", "Total_Loss", "Policy_Loss", "Value_Loss", "New_Champion"])
    
    # ---------------------------------------------------------
    # 3. Modell-Initialisierung & State Recovery
    # ---------------------------------------------------------
    champion_model = Connect4Model()
    start_iteration = 1
    
    # Prüft, ob ein trainiertes Modell (Pre-Trained State) zur Verfügung steht
    if os.path.exists(checkpoint_path):
        champion_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Gewichte des Basis-Modells geladen aus: {checkpoint_path}")
        
        # Iterations-Synchronisation anhand der Modell-Historie
        history_files = glob.glob(os.path.join(checkpoint_dir, "champion_iter_*.pt"))
        if history_files:
            iterations = []
            for f in history_files:
                match = re.search(r"champion_iter_(\d+)\.pt", f)
                if match:
                    iterations.append(int(match.group(1)))
            if iterations:
                start_iteration = max(iterations) + 1
                print(f"Zustandshistorie synchronisiert. Fortsetzung bei Iteration {start_iteration}.")
    else:
        print("Kein Pre-Trained State gefunden. Initialisiere Modell mit stochastischen Gewichten.")
        
    # Der Candidate ist anfangs ein exakter parametrischer Klon des Champions
    candidate_model = Connect4Model()
    candidate_model.load_state_dict(champion_model.state_dict())
    
    # =========================================================
    # 4. Hyperparameter-Konfiguration
    # =========================================================
    ADDITIONAL_ITERATIONS = 2000 
    
    # Thread-Allokation: Reserviert 2 logische Kerne für das Host-OS, nutzt den Rest für MCTS
    cpu_cores = max(1, mp.cpu_count() - 2)
    
    # Hyperparameter Phase 1: Datengenerierung
    SELF_PLAY_GAMES = 50 * cpu_cores  # Dynamische Skalierung der Batch-Generierung
    REPLAY_BUFFER_CAPACITY = 500000   # Minimiert das Risiko des 'Catastrophic Forgetting'
    
    # Hyperparameter Phase 2: Modell-Optimierung (Backpropagation)
    TRAINING_BATCHES = 1000     
    BATCH_SIZE = 1024             
    LEARNING_RATE = 1e-3
    
    # Hyperparameter Phase 3: Validierung
    ARENA_GAMES = 100
    WIN_THRESHOLD = 0.55          # Der Candidate muss eine signifikante Überlegenheit beweisen
    # =========================================================
    
    # Instanziierung des Experience-Memorys und des Optimizers
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)
    trainer = Connect4Trainer(candidate_model, learning_rate=LEARNING_RATE)
    
    end_iteration = start_iteration + ADDITIONAL_ITERATIONS - 1
    session_start_time = time.time()
    session_champions_found = 0
    final_loss = 0.0
    
    print(f"\nBeginne RL-Zyklus für {ADDITIONAL_ITERATIONS} Iterationen...")
    print(f"Nutze {cpu_cores} Threads für asynchrone Datengenerierung.")
    print("Telemetrie-Daten werden fortlaufend in 'training_metrics.csv' aggregiert.\n")
    
    # ---------------------------------------------------------
    # 5. DER ALPHAZERO KERNZYKLUS
    # ---------------------------------------------------------
    for iteration in range(start_iteration, end_iteration + 1):
        
        # --- PHASE 1: ASYNCHRONES SELF-PLAY (CPU) ---
        # Sperrt das Modell für Gradientenberechnungen (Inferenz-Modus)
        champion_model.eval()
        champion_state = champion_model.state_dict()
        
        # Gleichmäßige Verteilung der Spiellast auf die verbleibenden Worker-Threads
        games_per_worker = SELF_PLAY_GAMES // cpu_cores
        remainder = SELF_PLAY_GAMES % cpu_cores
        
        args_list = []
        for i in range(cpu_cores):
            games_to_play = games_per_worker + (remainder if i == 0 else 0)
            args_list.append((champion_state, games_to_play))
            
        with mp.Pool(processes=cpu_cores) as pool:
            worker_results = pool.starmap(worker_self_play, args_list)
            
        # Aggregation der dezentralen Spielverläufe in den zentralen Ringpuffer
        for result_batch in worker_results:
            for trajectory, winner in result_batch:
                store_game_trajectory(trajectory, winner, replay_buffer)
            
        # Puffer-Sicherheit: Verhindert asymmetrische Batches in frühen Trainingsphasen
        if len(replay_buffer) < BATCH_SIZE:
            print(f"Iter {iteration:04d} / {end_iteration:04d} | Puffer-Initialisierung ({len(replay_buffer)}/{BATCH_SIZE})...")
            continue
            
        # --- PHASE 2: NEURALES TRAINING (GPU) ---
        # Transferiert den Candidate in den VRAM und aktiviert die Gradientenverfolgung
        candidate_model.to(device)
        candidate_model.train()
        
        total_loss_sum, pol_loss_sum, val_loss_sum = 0.0, 0.0, 0.0
        
        # Ausführung des Mini-Batch Gradient Descent
        for _ in range(TRAINING_BATCHES):
            # Stochastisches Ziehen (Decorrelation) zur Vermeidung von Sequenz-Overfitting
            states, action_probs, values = replay_buffer.sample_batch(BATCH_SIZE)
            t_loss, p_loss, v_loss = trainer.train_step(states, action_probs, values)
            
            total_loss_sum += t_loss
            pol_loss_sum += p_loss
            val_loss_sum += v_loss
            
        # Mittelwertbildung der Verlustfunktionen über die Epoche
        avg_loss = total_loss_sum / TRAINING_BATCHES
        avg_pol = pol_loss_sum / TRAINING_BATCHES
        avg_val = val_loss_sum / TRAINING_BATCHES
        final_loss = avg_loss
        
        # Rückführung des aktualisierten Candidates in den Host-RAM für die CPU-Arena
        candidate_model.cpu()
        
        # --- PHASE 3: EVALUIERUNG IN DER ARENA (CPU) ---
        # Rigorose Überprüfung, ob die angepassten Gewichte strategisch wertvoll sind
        is_new_champion = evaluate_candidate(
            champion=champion_model, 
            candidate=candidate_model, 
            num_games=ARENA_GAMES, 
            win_threshold=WIN_THRESHOLD,
            verbose=False
        )
        
        # --- METRIKEN & ZUSTANDSSPEICHERUNG ---
        with open(metrics_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, len(replay_buffer), f"{avg_loss:.4f}", f"{avg_pol:.4f}", f"{avg_val:.4f}", is_new_champion])
        
        if is_new_champion:
            session_champions_found += 1
            # Der Candidate verdrängt den bisherigen Champion
            champion_model.load_state_dict(candidate_model.state_dict())
            
            # Persistierung des neuen Benchmark-Modells
            torch.save(champion_model.state_dict(), checkpoint_path)
            
            # Archivierung der historischen Iteration für eventuelle Rollbacks
            history_filename = f"champion_iter_{iteration:04d}.pt"
            history_path = os.path.join(checkpoint_dir, history_filename)
            torch.save(champion_model.state_dict(), history_path)
            
            print(f"Iter {iteration:04d} / {end_iteration:04d} | Loss: {avg_loss:.4f} | Validation: ERFOLG -> {history_filename}")
        else:
            # Die Modifikationen des Candidates wurden als ineffizient gewertet.
            # Rollback: Der Candidate wird durch den unangetasteten Champion überschrieben.
            candidate_model.load_state_dict(champion_model.state_dict())
            print(f"Iter {iteration:04d} / {end_iteration:04d} | Loss: {avg_loss:.4f} | Validation: FEHLSCHLAG (Rollback)")

    # ---------------------------------------------------------
    # ABSCHLUSS-ZUSAMMENFASSUNG
    # ---------------------------------------------------------
    duration_minutes = (time.time() - session_start_time) / 60.0
    print("\n" + "#"*50)
    print(" TRAINING SESSION SUMMARY ")
    print("#"*50)
    print(f" Dauer der Session:       {duration_minutes:.1f} Minuten")
    print(f" Iterationen absolviert:  {ADDITIONAL_ITERATIONS}")
    print(f" Architektonische Updates:{session_champions_found} (Neue Champions)")
    print(f" Finaler Puffer-Status:   {len(replay_buffer)} Transitions")
    print(f" Finaler Total Loss:      {final_loss:.4f}")
    print("#"*50)
    print("RL-Zyklus regulär terminiert.")


if __name__ == "__main__":
    main_training_loop()
