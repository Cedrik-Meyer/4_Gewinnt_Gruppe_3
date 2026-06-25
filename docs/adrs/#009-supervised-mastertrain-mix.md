# 009 - Supervised "Master Mix" Trainings-Pipeline

## Status

Akzeptiert

## Kontext

Während der AlphaZero Self-Play-Trainingsphase stießen wir auf ein strategisches Plateau. Das reine Verstärkungslernen (RL) stagnierte bei einem Policy-Loss von ca. 2.70. Das Netzwerk tat sich ohne die Hilfe von MCTS extrem schwer, eine scharfe Unterscheidung zwischen guten und schlechten Zügen (Säulen 0-15) zu treffen. Um das Modell auf ein kompetitives Niveau für das Turnier zu heben, reichte das ungesteuerte Self-Play nicht mehr aus.

## Entscheidung

Wir implementieren zusätzlich zum Self-Play eine überwachte Trainingspipeline (`supervised_train.py`). Diese generiert Offline-Daten basierend auf einer strikten Hierarchie ("Master Mix"): 1. Hardcoded Traps (Sofortsieg), 2. Behavioral Cloning (die Züge der tief suchenden `StrongEngine` werden imitiert) und 3. Knowledge Distillation (das Modell spielt gegen sich selbst in ruhigen Stellungen, um sein eigenes Positionsverständnis nicht zu vergessen).

## Konsequenzen

Diese Intervention durchbrach das Trainingsplateau erfolgreich (Resultat: `v9_champion.pt`). Das Modell erhielt eine extrem scharfe Intuition für gute Züge, wodurch MCTS im späteren Verlauf wesentlich effizienter arbeiten konnte. Der gravierende Nachteil ist die Laufzeit: Die Generierung eines Datensatzes über die Alpha-Beta-Engine erfordert extrem viel CPU-Zeit (ca. 60 Stunden bis zum v9-Modell). Zudem übernimmt das neuronale Netz durch das Cloning unweigerlich die systematischen blinden Flecke (Bias) der heuristischen Engine.
