"""
training_system/main_train.py

Das Hauptskript für den Trainingsprozess.
Orchestriert die Endlosschleife aus Self-Play, neuronalem Training und Arena-Evaluation.
"""

import torch
from training_system.neural_network.model import Connect4Model
from training_system.self_play.replay_buffer import ReplayBuffer
from training_system.self_play.self_play_loop import play_single_game, store_game_trajectory
from training_system.training.trainer import Connect4Trainer
from training_system.eval.arena import evaluate_candidate

def main_training_loop():
    """
    B6_03: Die Endlosschleife für das AlphaZero-Training.
    """
    print("Initialisiere ML-Fabrik...")
    
    # 1. Modelle aufsetzen
    champion_model = Connect4Model()
    candidate_model = Connect4Model()
    
    # Der Kandidat startet als exakter Klon des Champions
    candidate_model.load_state_dict(champion_model.state_dict())
    
    # 2. Infrastruktur aufsetzen
    replay_buffer = ReplayBuffer(capacity=100000)
    trainer = Connect4Trainer(candidate_model, learning_rate=1e-3)
    
    # Hyperparameter für den Loop
    MAX_ITERATIONS = 100
    SELF_PLAY_GAMES = 50       # N Runden Daten generieren
    TRAINING_BATCHES = 100     # M Batches trainieren
    BATCH_SIZE = 64
    ARENA_GAMES = 40           # Anzahl der Arena-Spiele
    
    iteration = 1
    
    print("Startschuss! Die Fabrik läuft...")
    
    # Die AlphaZero-Schleife
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
            
        print(f" -> Buffer enthält jetzt {len(replay_buffer)} Züge.")
        
        if len(replay_buffer) < BATCH_SIZE:
            print(" -> Noch nicht genug Daten für einen Batch. Überspringe Training.")
            iteration += 1
            continue
            
        # ---------------------------------------------------------
        # PHASE 2: TRAINING (Kandidat lernt aus dem Buffer)
        # ---------------------------------------------------------
        print(f"Phase 2: Training ({TRAINING_BATCHES} Batches à {BATCH_SIZE} Züge)...")
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
        
        # Die evaluate_candidate Funktion (aus B6_01) übernimmt die Spiele und die Winrate-Logik
        is_new_champion = evaluate_candidate(
            champion=champion_model, 
            candidate=candidate_model, 
            num_games=ARENA_GAMES, 
            win_threshold=0.55
        )
        
        if is_new_champion:
            # Der Kandidat hat gesiegt! Er überschreibt das Gehirn des Champions.
            champion_model.load_state_dict(candidate_model.state_dict())
            
            # ---------------------------------------------------------
            # Platzhalter für B6_04: Checkpoint speichern
            # ---------------------------------------------------------
            pass
            
        else:
            # Der Kandidat hat versagt. Sein Gehirn wird gelöscht und auf den 
            # bewährten Stand des Champions zurückgesetzt, um es in Iteration X+1 neu zu versuchen.
            candidate_model.load_state_dict(champion_model.state_dict())
            
        iteration += 1
    print("\n Training erfolgreich und sicher beendet!")
if __name__ == "__main__":
    main_training_loop()
