# 004 - Relative Perspektiven-Codierung für den Modell-Input

## Status

Akzeptiert

## Kontext

Der WebSocket-Server übermittelt das Spielfeld mit absoluten Spieler-IDs. Wert `1` steht für den ersten Spieler-Slot, Wert `2` für den zweiten Spieler-Slot. Würden diese IDs direkt in das neuronale Netzwerk eingegeben, müsste das Modell Strategien für beide absoluten Slots getrennt lernen. Dies würde die Trainingszeit erhöhen und die Daten im Replay-Buffer ineffizient nutzen.

## Entscheidung

Es wird eine relative Perspektivencodierung in `src/connect4/shared/state_encoder.py` implementiert. Unabhängig vom Spieler-Slot transformiert der Encoder das 3D-Spielfeld vor der Übergabe an das Netzwerk in einen relativen, zweikanaligen Tensor. Kanal 0 enthält die Steine des am Zug befindlichen Agenten (Own Stones), Kanal 1 die Steine des Gegners (Opponent Stones). Der Ansatz orientiert sich an Verfahren wie AlphaZero.

## Konsequenzen

Diese Transformation reduziert den Problemraum für das neuronale Netzwerk, da ein Siegpfad unabhängig vom absoluten Spieler-Slot gleich codiert wird. Der Nachteil liegt im zusätzlichen Rechenaufwand und in einer Fehlerquelle im Laufzeitsystem, da der State-Encoder die Serverdaten anhand des Spieler-Slots transformieren muss.
