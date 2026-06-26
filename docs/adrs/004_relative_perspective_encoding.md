# 004 - Relative Perspektiven-Codierung für den Modell-Input

## Status

Akzeptiert

## Kontext

Der WebSocket-Server übermittelt das Spielfeld mit absoluten Spieler-IDs, wobei der Wert `1` für den ersten Spieler-Slot und der Wert `2` für den zweiten Spieler-Slot steht. Wenn wir diese absoluten IDs direkt in das neuronale Netzwerk einspeisen würden, müsste das Modell zwei völlig unterschiedliche Strategien erlernen: "Wie gewinne ich als Spieler 1" und "Wie gewinne ich als Spieler 2". Dies würde die benötigte Trainingszeit effektiv verdoppeln, die Datenmenge im Replay-Buffer ineffizient nutzen und das Modell insgesamt anfälliger für strategische Fehler machen.

## Entscheidung

Wir werden eine relative Perspektivencodierung innerhalb des `shared/state_encoder.py` implementieren. Unabhängig davon, ob unser Agent in einem Match als Spieler 1 oder Spieler 2 zugewiesen wird, transformiert der Encoder das 3D-Spielfeld vor der Übergabe an das Netzwerk immer in einen relativen, zweikanaligen Tensor. Kanal 0 enthält dabei ausschließlich die Steine des aktuell am Zug befindlichen, eigenen Agenten (Own Stones), während Kanal 1 die Steine des Gegners abbildet (Opponent Stones). Dieser Ansatz orientiert sich an bewährten Konzepten moderner Spiele-KIs wie AlphaZero.

## Konsequenzen

Diese Transformation halbiert den mathematischen Problemraum für das neuronale Netzwerk drastisch, wodurch das Modell strategische Gewinnmuster wesentlich schneller und robuster erlernt, da ein Siegpfad aus Sicht des Netzwerks immer identisch aussieht. Der Nachteil liegt im erhöhten Rechenaufwand und einer potenziellen Fehlerquelle im Oberbau, da der State-Encoder die absoluten Serverdaten im Live-Betrieb basierend auf dem aktuellen Spieler-Slot absolut fehlerfrei und dynamisch invertieren muss.