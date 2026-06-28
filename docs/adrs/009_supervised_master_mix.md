# 009 - Supervised "Master Mix" Trainings-Pipeline

## Status

Akzeptiert

## Kontext

Während der AlphaZero Self-Play-Trainingsphase entstand ein strategisches Plateau. Das reine Verstärkungslernen (RL) stagnierte bei einem Policy-Loss von ca. 2.70. Ohne MCTS unterschied das Netzwerk gute und schlechte Züge (Säulen 0-15) nur schwach. Für den Turniereinsatz reichte das ungesteuerte Self-Play nicht mehr aus.

## Entscheidung

Zusätzlich zum Self-Play wird eine überwachte Trainingspipeline (`supervised_train.py`) implementiert. Diese generiert Offline-Daten auf Basis einer Hierarchie ("Master Mix"): 1. Hardcoded Traps (Sofortsieg), 2. Behavioral Cloning (Imitation der `StrongEngine`) und 3. Knowledge Distillation (Selbstspiel in ruhigen Stellungen, um zuvor gelernte Positionsbewertungen zu erhalten).

## Konsequenzen

Die Pipeline überwand das Trainingsplateau (Resultat: `v9_champion.pt`). Das Modell erhielt deutlichere Policy-Zielverteilungen für taktisch relevante Züge, wodurch MCTS effizienter arbeiten konnte. Der Nachteil ist die Laufzeit: Die Generierung eines Datensatzes über die Alpha-Beta-Engine erfordert viel CPU-Zeit (ca. 60 Stunden bis zum v9-Modell). Zudem kann das neuronale Netz durch Cloning systematische Verzerrungen (Bias) der heuristischen Engine übernehmen.
