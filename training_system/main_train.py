"""
training_system/main_train.py

Das Hauptskript fuer den Trainingsprozess.
Orchestriert die Endlosschleife aus Self-Play, neuronalem Training und Arena-Evaluation.
Unterstuetzt das automatische Fortsetzen (Resume) bestehender Trainingslaeufe.
"""

import os
import glob
import re
import torch
from training_system.neural_network.model import Connect4Model
from training_system.self_play.replay_buffer import ReplayBuffer
from training_system.self_play.self_play_loop import play_single_game, store_game_trajectory
from training_system.training.trainer import Connect4Trainer
from training_system.eval.arena import evaluate_candidate

def main_training_loop():
    """
    B6_03 & B6_04: Endlosschleife fuer das AlphaZero-Training mit automatischem Resume.
    """
    print("Initialisiere ML-Fabrik...")
    
    # 1. Checkpoint-Ordner und Pfade sicherstellen
    checkpoint_dir = "training_system/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_champion.pt")
    
    # 2. Modell aufsetzen
    champion_model = Connect4Model()
    start_iteration = 1
    
    # Intelligentes Laden: Pruefen, ob bereits ein trainiertes Gehirn existiert
    if os.path.exists(checkpoint_path):
        # Laedt die Gewichte des letzten Champions von der Festplatte
        champion_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Bestehendes Gehirn erfolgreich geladen: {checkpoint_path}")
        
        # Historien-Dateien scannen, um die richtige Iterationsnummer zu ermitteln
        history_files = glob.glob(os.path.join(checkpoint_dir, "champion_iter_*.pt"))
        if history_files:
            iterations = []
            for f in history_files:
                # Extrahiert die Nummer aus dem Dateinamen mittels Regular Expression
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
    replay_buffer = ReplayBuffer(capacity=100000)
    trainer = Connect4Trainer(candidate_model, learning_rate=1e-3)
    
    # Hyperparameter für diesen Lauf
    ADDITIONAL_ITERATIONS = 20
    SELF_PLAY_GAMES = 50       
    TRAINING_BATCHES = 100     
    BATCH_SIZE = 64
    ARENA_GAMES = 40           
    
    # Das Ende der Schleife berechnet sich dynamisch aus dem Startpunkt
    end_iteration = start_iteration + ADDITIONAL_ITERATIONS - 1
    
    print("Startschuss! Die Fabrik laeuft...")
    
    for iteration in range(start_iteration, end_iteration + 1):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration} / {end_iteration}")
        print(f"{'='*50}")
        
        # ---------------------------------------------------------
        # PHASE 1: SELF-PLAY (Daten generieren)
        # ---------------------------------------------------------
        print(f"Phase 1: Self-Play ({SELF_PLAY_GAMES} Partien)...")
        champion_model.eval()
        for i in range(SELF_PLAY_GAMES):
            trajectory, winner = play_single_game(champion_model)
            store_game_trajectory(trajectory, winner, replay_buffer)
            
        print(f" -> Buffer enthaelt jetzt {len(replay_buffer)} Zuege.")
        
        if len(replay_buffer) < BATCH_SIZE:
            print(" -> Noch nicht genug Daten fuer einen Batch. Ueberspringe Training.")
            continue
            
        # ---------------------------------------------------------
        # PHASE 2: TRAINING (Kandidat lernt aus dem Buffer)
        # ---------------------------------------------------------
        print(f"Phase 2: Training ({TRAINING_BATCHES} Batches a {BATCH_SIZE} Zuege)...")
        candidate_model.train()
        total_loss = 0.0
        
        for _ in range(TRAINING_BATCHES):
            states, action_probs, values = replay_buffer.sample_batch(BATCH_SIZE)
            loss = trainer.train_step(states, action_probs, values)
            total_loss += loss
            
        avg_loss = total_loss / TRAINING_BATCHES
        print(f" -> Durchschnittlicher Loss: {avg_loss:.4f}")
        
        # ---------------------------------------------------------
        # PHASE 3: ARENA (Der ultimative Test)
        # ---------------------------------------------------------
        print(f"Phase 3: Arena Evaluation ({ARENA_GAMES} Partien)...")
        is_new_champion = evaluate_candidate(
            champion=champion_model, 
            candidate=candidate_model, 
            num_games=ARENA_GAMES, 
            win_threshold=0.55
        )
        
        if is_new_champion:
            champion_model.load_state_dict(candidate_model.state_dict())
            
            # 1. Standard-Checkpoint ueberschreiben
            torch.save(champion_model.state_dict(), checkpoint_path)
            
            # 2. Historien-Checkpoint mit der korrekten fortlaufenden Nummer anlegen
            history_filename = f"champion_iter_{iteration:04d}.pt"
            history_path = os.path.join(checkpoint_dir, history_filename)
            torch.save(champion_model.state_dict(), history_path)
            
            print(f"Checkpoint ueberschrieben: {checkpoint_path}")
            print(f"Historie gesichert als:    {history_path}")
            
        else:
            candidate_model.load_state_dict(champion_model.state_dict())

    print("\nTraining erfolgreich und sicher beendet!")

if __name__ == "__main__":
    main_training_loop()
