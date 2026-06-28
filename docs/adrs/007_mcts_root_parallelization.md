# 007 - MCTS Root Parallelization via Multiprocessing

## Status

Akzeptiert

## Kontext

Der Monte Carlo Tree Search (MCTS) Algorithmus profitiert von vielen Simulationen. Je mehr Pfade berechnet werden, desto stärker der Agent. Python nutzt jedoch den Global Interpreter Lock (GIL), der parallele CPU-Ausführung mit Threads begrenzt. Bei einem CPU-intensiven Suchbaum bringt Multi-Threading daher keinen relevanten Geschwindigkeitsvorteil.

## Entscheidung

Wir nutzen das `multiprocessing`-Modul von Python und implementieren eine "Root Parallelization". Dabei generiert die `MCTSEngine` nicht einen Baum auf einem Kern, sondern lässt jeden logischen CPU-Kern einen unabhängigen Suchbaum generieren. Um identische Pfade zu reduzieren, wird Dirichlet-Rauschen in die Root-Node jedes Baumes injiziert. Am Ende des Zeitlimits werden die Visit-Counts (`visit_counts`) aller Kerne aufsummiert.

## Konsequenzen

Diese Entscheidung ermöglicht eine annähernd lineare Skalierung der Simulationsrate mit der Anzahl der CPU-Kerne. In gleicher Bedenkzeit können mehr Simulationen berechnet werden. Der Nachteil ist höherer Speicherverbrauch, da jeder Prozess eine Kopie des neuronalen Netzes im RAM hält. Zudem entsteht Performance-Verlust durch Inter-Process-Communication (IPC) beim Sammeln der Ergebnisse.
