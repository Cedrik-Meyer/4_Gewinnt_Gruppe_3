"""
training_system/main_train.py

Das Hauptskript für den Trainingsprozess.
Orchestriert die Endlosschleife aus Self-Play, neuronalem Training und Arena-Evaluation.
Nutzt Multiprocessing für CPU-Auslastung, schreibt detaillierte CSV-Metriken und
unterstützt das automatische Fortsetzen (Resume) bestehender Trainingsläufe.
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

def worker_self_play(state_dict, num_games):
    """
    Diese Funktion wird auf jedem CPU-Kern separat ausgeführt.
    Jeder Kern baut ein eigenes lokales Modell auf, um Speicher-Konflikte zu vermeiden.
    """
    # Eigenes, isoliertes Modell für diesen Kern erstellen
    local_model = Connect4Model()
    local_model.load_state_dict(state_dict)
    local_model.eval()
    
    # Die zugewiesene Anzahl an Spielen absolvieren
    results = []
    for _ in range(num_games):
        trajectory, winner = play_single_game(local_model)
        results.append((trajectory, winner))
        
    return results

def main_training_loop():
    """
    B6_03 & B6_04: Endlosschleife für das AlphaZero-Training.
    """
    # PyTorch Multiprocessing Start-Methode sicherstellen (verhindert Deadlocks auf MacOS/Linux)
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
        
    print("Initialisiere ML-Fabrik (Multiprocessing Edition)...")
    
    # 1. Checkpoint-Ordner und Pfade sicherstellen
    checkpoint_dir = "training_system/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_champion.pt")
    metrics_path = os.path.join(checkpoint_dir, "training_metrics.csv")
    
    # CSV Header anlegen, falls Datei noch nicht existiert
    if not os.path.isfile(metrics_path):
        with open(metrics_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Buffer_Size", "Total_Loss", "Policy_Loss", "Value_Loss", "New_Champion"])
    
    # 2. Modell aufsetzen
    champion_model = Connect4Model()
    start_iteration = 1
    
    # Intelligentes Laden: Prüfen, ob bereits ein trainiertes Gehirn existiert
    if os.path.exists(checkpoint_path):
        champion_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Bestehendes Gehirn erfolgreich geladen: {checkpoint_path}")
        
        # Historien-Dateien scannen, um die richtige Iterationsnummer zu ermitteln
        history_files = glob.glob(os.path.join(checkpoint_dir, "champion_iter_*.pt"))
        if history_files:
            iterations = []
            for f in history_files:
                match = re.search(r"champion_iter_(\d+)\.pt", f)
                if match:
                    iterations.append(int(match.group(1)))
            
            if iterations:
                # Wir machen genau eine Nummer nach dem bisherigen Maximum weiter
                start_iteration = max(iterations) + 1
                print(f"Historie gefunden. Training wird bei Iteration {start_iteration} fortgesetzt.")
    else:
        print("Kein bestehender Checkpoint gefunden. Starte mit neuem Basis-Gehirn bei Iteration 1.")
        
    # Der Kandidat startet immer als Kopie des geladenen/neuen Champions
    candidate_model = Connect4Model()
    candidate_model.load_state_dict(champion_model.state_dict())
    
    # 3. Infrastruktur aufsetzen
    # Der Buffer muss riesig sein, um das "Vergessen" von alten Taktiken zu verhindern.
    replay_buffer = ReplayBuffer(capacity=500000)
    trainer = Connect4Trainer(candidate_model, learning_rate=1e-3)
    
    # =========================================================
    # HYPERPARAMETER FÜR EINEN LEISTUNGSSTARKEN RECHNER
    # (Diese Werte garantieren ein viel stabileres Lernen)
    # =========================================================
    ADDITIONAL_ITERATIONS = 1000
    SELF_PLAY_GAMES = 400
    TRAINING_BATCHES = 1000
    BATCH_SIZE = 256
    ARENA_GAMES = 100
    # =========================================================
    
    end_iteration = start_iteration + ADDITIONAL_ITERATIONS - 1
    cpu_cores = max(1, mp.cpu_count())
    
    session_start_time = time.time()
    session_champions_found = 0
    final_loss = 0.0
    
    print(f"\nStartschuss! Die Fabrik läuft für {ADDITIONAL_ITERATIONS} Iterationen...")
    print(f"Nutze {cpu_cores} CPU-Kerne für die parallele Datengenerierung.")
    print("Detaillierte Metriken werden im Hintergrund in 'training_metrics.csv' gespeichert.\n")
    
    for iteration in range(start_iteration, end_iteration + 1):
        
        # ---------------------------------------------------------
        # PHASE 1: SELF-PLAY (Multiprocessing)
        # ---------------------------------------------------------
        champion_model.eval()
        champion_state = champion_model.state_dict()
        
        games_per_worker = SELF_PLAY_GAMES // cpu_cores
        remainder = SELF_PLAY_GAMES % cpu_cores
        
        args_list = []
        for i in range(cpu_cores):
            games_to_play = games_per_worker + (remainder if i == 0 else 0)
            args_list.append((champion_state, games_to_play))
            
        with mp.Pool(processes=cpu_cores) as pool:
            worker_results = pool.starmap(worker_self_play, args_list)
            
        for result_batch in worker_results:
            for trajectory, winner in result_batch:
                store_game_trajectory(trajectory, winner, replay_buffer)
            
        if len(replay_buffer) < BATCH_SIZE:
            print(f"Iter {iteration:04d} / {end_iteration:04d} | Buffer wird gefüllt ({len(replay_buffer)}/{BATCH_SIZE})...")
            continue
            
        # ---------------------------------------------------------
        # PHASE 2: TRAINING (Kandidat lernt aus dem Buffer)
        # ---------------------------------------------------------
        candidate_model.train()
        total_loss_sum, pol_loss_sum, val_loss_sum = 0.0, 0.0, 0.0
        
        for _ in range(TRAINING_BATCHES):
            states, action_probs, values = replay_buffer.sample_batch(BATCH_SIZE)
            t_loss, p_loss, v_loss = trainer.train_step(states, action_probs, values)
            total_loss_sum += t_loss
            pol_loss_sum += p_loss
            val_loss_sum += v_loss
            
        avg_loss = total_loss_sum / TRAINING_BATCHES
        avg_pol = pol_loss_sum / TRAINING_BATCHES
        avg_val = val_loss_sum / TRAINING_BATCHES
        final_loss = avg_loss
        
        # ---------------------------------------------------------
        # PHASE 3: ARENA (Der ultimative Test)
        # ---------------------------------------------------------
        is_new_champion = evaluate_candidate(
            champion=champion_model, 
            candidate=candidate_model, 
            num_games=ARENA_GAMES, 
            win_threshold=0.55,
            verbose=False
        )
        
        # Logging
        with open(metrics_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, len(replay_buffer), f"{avg_loss:.4f}", f"{avg_pol:.4f}", f"{avg_val:.4f}", is_new_champion])
        
        if is_new_champion:
            session_champions_found += 1
            champion_model.load_state_dict(candidate_model.state_dict())
            
            torch.save(champion_model.state_dict(), checkpoint_path)
            history_filename = f"champion_iter_{iteration:04d}.pt"
            history_path = os.path.join(checkpoint_dir, history_filename)
            torch.save(champion_model.state_dict(), history_path)
            
            print(f"Iter {iteration:04d} / {end_iteration:04d} | Loss: {avg_loss:.4f} | Neuer Champion: JA -> {history_filename}")
        else:
            candidate_model.load_state_dict(champion_model.state_dict())
            print(f"Iter {iteration:04d} / {end_iteration:04d} | Loss: {avg_loss:.4f} | Neuer Champion: Nein")

    # ---------------------------------------------------------
    # ABSCHLUSS-ZUSAMMENFASSUNG
    # ---------------------------------------------------------
    duration_minutes = (time.time() - session_start_time) / 60.0
    print("\n" + "#"*50)
    print(" TRAINING SESSION SUMMARY ")
    print("#"*50)
    print(f" Dauer der Session:      {duration_minutes:.1f} Minuten")
    print(f" Iterationen absolviert: {ADDITIONAL_ITERATIONS}")
    print(f" Neue Champions gekrönt: {session_champions_found}")
    print(f" Letzter Buffer Status:  {len(replay_buffer)} Züge")
    print(f" Letzter Total Loss:     {final_loss:.4f}")
    print("#"*50)
    print("Training erfolgreich und sicher beendet!")

if __name__ == "__main__":
    main_training_loop()
