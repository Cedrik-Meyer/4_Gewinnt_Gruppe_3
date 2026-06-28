# 006 - Hybride Entscheidungsfindung im Live-Betrieb (Forced Moves)

## Status

Akzeptiert

## Kontext

Neuronale Netze ohne ausreichende MCTS-Tiefe agieren probabilistisch. In frühen oder mittleren Trainingsphasen kann das Netzwerk 1-Step-Siege übersehen oder eine unmittelbare 4er-Reihe des Gegners nicht blockieren. In einem Live-Turnier unter Zeitdruck kann dies zu vermeidbaren Niederlagen führen.

## Entscheidung

Im `live_agent.py` wird ein hybrider Entscheidungsansatz implementiert. Vor der Modell- oder MCTS-Entscheidung führt `apply_forced_moves` einen deterministischen 1-Step-Lookahead durch. Führt ein Zug sofort zum Sieg, werden dessen Policy-Logits erhöht (`+2e6`). Gleiches gilt für Block-Züge (`+1e6`).

## Konsequenzen

Durch diese Heuristik erhält der Agent eine regelbasierte Baseline. 1-Step-Fehler werden ausgeschlossen. Der Nachteil ist zusätzliche Rechenzeit, da pro Zug bis zu 16 mögliche Züge simuliert werden.
