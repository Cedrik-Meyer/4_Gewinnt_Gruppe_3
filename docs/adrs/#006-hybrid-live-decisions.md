# 006 - Hybride Entscheidungsfindung im Live-Betrieb (Forced Moves)

## Status

Akzeptiert

## Kontext

Reine neuronale Netze (ohne extrem tiefe MCTS-Suchen) agieren probabilistisch. Gerade in frühen oder mittleren Trainingsphasen kann es passieren, dass das Netzwerk "halluziniert" und offensichtliche 1-Step-Siege übersieht oder vergisst, eine unmittelbare 4er-Reihe des Gegners zu blockieren. In einem Live-Turnier unter Zeitdruck (wo die MCTS-Tiefe durch die `deadline_ms` stark limitiert ist) würde blindes Vertrauen in das Modell zu peinlichen und vermeidbaren Niederlagen führen.

## Entscheidung

Wir implementieren einen hybriden Entscheidungsansatz im `live_agent.py`. Bevor das neuronale Netzwerk oder MCTS den finalen Zug diktiert, führt eine hartcodierte Heuristik (`apply_forced_moves`) einen deterministischen 1-Step-Lookahead durch. Wenn ein Zug gefunden wird, der sofort zum Sieg führt, werden dessen Policy-Logits künstlich massiv erhöht (`+2e6`). Gleiches gilt für zwingende Block-Züge (`+1e6`).

## Konsequenzen

Durch diese Heuristik erhält der Agent ein garantiertes, unfehlbares Basisniveau (Baseline) an Spielstärke. Triviale 1-Step-Blunder werden komplett ausgeschlossen. Der Nachteil ist, dass wir domänenspezifisches Wissen (Spielregeln) künstlich in den sonst rein statistischen Machine-Learning-Ablauf injizieren und pro Zug zusätzliche Rechenzeit für das simulierte Ausführen der 16 möglichen Züge im RAM aufwenden müssen.

