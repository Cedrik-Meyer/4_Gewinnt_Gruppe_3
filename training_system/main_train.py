"""
training_system/main_train.py

Das Hauptskript für den Trainingsprozess.
Orchestriert die Endlosschleife aus Self-Play, neuronalem Training und Arena-Evaluation.
Speichert sowohl das aktuellste Modell als auch eine Historie aller bisherigen Champions.
"""

import os
import torch
from training_system.neural_network.model import Connect4Model
from training_system.self_play.replay_buffer import ReplayBuffer
from training_system.self_play.self_play_loop import play_single_game, store_game_trajectory
from training_system.training.trainer import Connect4Trainer
from training_system.eval.arena import evaluate_candidate

def main_training_loop():
    """
    B6_03 & B6_04: Die Endlosschleife für das AlphaZero-Training inkl. Checkpoints und Historie.
    """
    print("Initialisiere ML-Fabrik...")
    
    # 1. Modelle aufsetzen
    champion_model = Connect4Model()
    candidate_model = Connect4Model()
    candidate_model.load_state_dict(champion_model.state_dict())
    
    # 2. Infrastruktur aufsetzen
    replay_buffer = ReplayBuffer(capacity=100000)
    trainer = Connect4Trainer(candidate_model, learning_rate=1e-3)
    
    # 3. Checkpoint-Ordner sicherstellen
    checkpoint_dir = "training_system/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_champion.pt")
    
    # Hyperparameter für die Loop
    MAX_ITERATIONS = 100       
    SELF_PLAY_GAMES = 50       
    TRAINING_BATCHES = 100     
    BATCH_SIZE = 64
    ARENA_GAMES = 40           
    
    print("Startschuss! Die Fabrik laeuft...")
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration} / {MAX_ITERATIONS}")
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
            # Kandidat wird neuer Champion
            champion_model.load_state_dict(candidate_model.state_dict())
            
            # 1. Standard-Checkpoint fuer den Live-Betrieb aktualisieren
            torch.save(champion_model.state_dict(), checkpoint_path)
            
            # 2. Historien-Checkpoint anlegen (z.B. champion_iter_0005.pt)
            history_filename = f"champion_iter_{iteration:04d}.pt"
            history_path = os.path.join(checkpoint_dir, history_filename)
            torch.save(champion_model.state_dict(), history_path)
            
            print(f"Checkpoint ueberschrieben: {checkpoint_path}")
            print(f"Historie gesichert als:    {history_path}")
            
        else:
            # Kandidat wird zurueckgesetzt
            candidate_model.load_state_dict(champion_model.state_dict())

    print("\nTraining erfolgreich und sicher beendet!")

if __name__ == "__main__":
    main_training_loop()
