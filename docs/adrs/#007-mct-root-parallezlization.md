# 007 - MCTS Root Parallelization via Multiprocessing

## Status

Akzeptiert

## Kontext

Der Monte Carlo Tree Search (MCTS) Algorithmus profitiert massiv von vielen Simulationen. Je mehr Pfade berechnet werden, desto stärker der Agent. Python nutzt jedoch den Global Interpreter Lock (GIL), der verhindert, dass Threads echten parallelen Code auf mehreren CPU-Kernen ausführen. Bei einem CPU-intensiven Suchbaum würde einfaches Multi-Threading keinen Geschwindigkeitsvorteil bringen, was die Hardware (wie den 14-Kern Ryzen) ungenutzt ließe.

## Entscheidung

Wir nutzen das `multiprocessing`-Modul von Python und implementieren eine "Root Parallelization". Dabei generiert die `MCTSEngine` nicht einen großen Baum auf einem Kern, sondern lässt jeden logischen CPU-Kern einen völlig unabhängigen Suchbaum generieren. Um zu verhindern, dass alle Kerne dieselben Pfade berechnen, injizieren wir Dirichlet-Rauschen in die Root-Node jedes Baumes. Am Ende der Zeitlimits werden die Visit-Counts (`visit_counts`) aller Kerne einfach aufsummiert.

## Konsequenzen

Diese Entscheidung ermöglicht eine annähernd lineare Skalierung der Simulationsrate mit der Anzahl der CPU-Kerne. Wir können in der gleichen Bedenkzeit drastisch tiefer rechnen. Der Nachteil ist ein sehr hoher Speicherverbrauch, da jeder Prozess eine eigene, isolierte Kopie des neuronalen Netzes (inklusive PyTorch-Overhead) im RAM halten muss. Zudem entsteht ein leichter Performance-Verlust durch die Inter-Process-Communication (IPC) beim Sammeln der Ergebnisse.
