# 008 - Single-Core Ausführung der Alpha-Beta Engine

## Status

Akzeptiert

## Kontext

Zur Evaluierung und für die Datengenerierung nutzen wir eine handgeschriebene Minimax-Engine mit Alpha-Beta-Pruning (`StrongEngine`). Da Multiprocessing beim MCTS-Algorithmus (siehe ADR-007) so erfolgreich war, stand die Überlegung im Raum, auch den Suchbaum der Alpha-Beta-Engine auf mehrere Kerne aufzuteilen, um noch tiefer suchen zu können.

## Entscheidung

Wir verwerfen Multiprocessing für die `StrongEngine` und beschränken sie hart auf einen einzigen CPU-Kern. Alpha-Beta-Pruning lebt davon, dass Informationen über gute Züge sofort genutzt werden, um andere Äste abzuschneiden (Pruning). Die Synchronisation einer gemeinsamen Transposition-Table zwischen mehreren Prozessen (IPC) dauert in Python zu lange. Besonders bei den kurzen Zeitlimits (< 200ms) frisst der IPC-Overhead mehr Zeit, als die parallele Berechnung einspart.

## Konsequenzen

Die Implementierung bleibt deutlich simpler. Durch das Center-First Move Ordering und eine geteilte Hash-Tabelle im RAM eines einzelnen Kerns erreicht die Engine extrem schnell hohe Suchtiefen. Der Nachteil ist offensichtlich: Wir können die rohe Rechenkraft von modernen Multi-Core-CPUs für diese spezifische Komponente nicht ausnutzen. Die Engine bleibt an die Single-Core-Taktfrequenz der CPU gebunden.
