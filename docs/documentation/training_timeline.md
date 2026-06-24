# Trainings-Historie & Timeline (Connect4 3D Agent)

Dieses Dokument protokolliert den chronologischen Ablauf des Modell-Trainings. Da das Training für 4-Gewinnt im 3D-Raum (4x4x4) enorm rechenintensiv ist, wurde der Prozess in drei große Phasen unterteilt: Initiale Self-Play-Datengenerierung, gezieltes Supervised Finetuning (um Stagnation zu überwinden) und abschließendes Self-Play für den Feinschliff.

---

## Hardware-Setup

Alle maßgeblichen Trainingsläufe wurden auf folgendem dedizierten System durchgeführt:
* **CPU:** AMD Ryzen 7 7800X3D (Nutzung von 14 logischen Kernen für MCTS & Datengenerierung)
* **GPU:** NVIDIA RTX 4070 Super (Nutzung für hochparallele Tensor-Kalkulationen / Backpropagation)
* **VRAM:** 12 GB 
* **RAM:** 32 GB


---

## Phase 1: Initiales AlphaZero Self-Play (Iterationen 1 – 2000)

Das Training begann von Grund auf neu (Zufallsgewichte) mit der klassischen Reinforcement-Learning-Schleife (`main_train.py`). Das Modell spielte ununterbrochen gegen sich selbst, sammelte Trajektorien im `ReplayBuffer` und trainierte mittels Kreuzentropie (Policy) und MSE (Value).

### Hyperparameter (RL Phase)
| Parameter | Wert | Beschreibung |
| :--- | :--- | :--- |
| **Iterationen** | 2000 | Zyklen aus Self-Play $\rightarrow$ Training $\rightarrow$ Arena. |
| **Spiele pro Iteration** | ~700 | 50 Spiele verteilt auf 14 CPU-Worker. |
| **Buffer Capacity** | 500.000 | Großer Sliding-Window-Speicher. |
| **Training Batches** | 1.000 | Anzahl der Netz-Updates pro Iteration auf der GPU. |
| **Batch Size** | 1.024 | Optimiert für die RTX 4070 Super. |
| **Learning Rate** | `1e-3` | Mit StepLR-Scheduler (Gamma 0.9 alle 100.000 Steps). |

**Erkenntnisse der Phase 1:** 
Zwischen Iteration 1 und 500 gab es ein starkes initiales Wachstum. Ab Iteration 500 bis ca. 1500 fiel jedoch auf, dass der **Policy Loss** auf einem Niveau von ca. `2.69 - 2.71` stagnierte. Das Netzwerk ordnete allen 16 Säulen sehr ähnliche Wahrscheinlichkeiten zu und übersah teilweise simple 1-Step-Bedrohungen. Zwar wurden durch einen stark verbesserten Value-Head weiterhin neue Champions gekürt, die reine Zug-Intuition stagnierte jedoch.

---

## Phase 2: Die "Master Mix" Intervention (Supervised Learning v1 bis v9)

Um die Policy-Stagnation aufzubrechen und fundamentale Flüchtigkeitsfehler abzutrainieren, wurde das Training pausiert. Das Modell aus Phase 1 wurde in eine überwachte Trainingspipeline (`supervised_train.py`) übergeben. 

Dabei wurde eine dedizierte "Master Mix"-Hierarchie genutzt, um dem Netz taktisches Wissen von außen "einzutrichtern":
1. **Trap:** Hardcoded One-Hot-Vektor bei direkten 1-Step-Siegen/Blocks.
2. **Clone:** Klonen von Zügen der Alpha-Beta `StrongEngine` (2000ms Bedenkzeit).
3. **Distill:** Boltzmann-Exploration des Basismodells (Knowledge Distillation) bei ruhigen Stellungen, um das bisherige RL-Wissen nicht zu vergessen.

### Hyperparameter (Supervised Phase)
| Parameter | Wert | Beschreibung |
| :--- | :--- | :--- |
| **Trainings-Zyklen** | 9 | Es wurden 9 vollständige Durchläufe generiert (v1 bis v9). |
| **Total Games** | 15.000 | Anzahl der simulierten Partien pro Zyklus. |
| **Batch Size** | 512 | Tensor-Stapelgröße für die GPU. |
| **Epochs** | 4 | Trainingsdurchläufe über den gesamten Datensatz pro Zyklus. |
| **Learning Rate** | `2e-5` | Sehr gering gewählt für behutsames Finetuning. |
| **Early Stop Loss** | `0.35` | Schwellenwert, um Overfitting auf die Engine-Züge zu vermeiden. |

**Resultat der Phase 2:** Nach insgesamt 9 Zyklen (Laufzeit: ca. 60 Stunden) entstand das Modell **`v9_champion.pt`**. Das Netz besaß nun eine messerscharfe Policy-Intuition für Fallen und Zentrumskontrolle.

---

## Phase 3: Zweite Self-Play RL-Phase (Iterationen 2001 – 4000)

Mit dem extrem starken `v9_champion.pt` als neuer Baseline kehrte das System in den AlphaZero-Modus (`main_train.py`) zurück. Ziel war es nun, die geklonten Heuristiken der Engine durch tiefes, eigenes RL-Wissen (Beyond Human/Engine Level) zu übertreffen.

Die Hyperparameter blieben identisch zur Phase 1.

**Erkenntnisse der Phase 3 (Deep Finetuning):**
In dieser fortgeschrittenen RL-Phase lief das Training extrem stabil weiter. Da das Ausgangsniveau von `v9` bereits sehr hoch war, wurden neue Champions erwartungsgemäß seltener (z.B. bei Iterationen 2852, 3130, 3341, etc.). Das Modell lernte in dieser Phase vor allem "Micro-Optimizations" (kleinste Positionsvorteile) und glich verbleibende Schwächen aus, die die klassische Alpha-Beta-Engine in Phase 2 nicht abdecken konnte.

---

## Aktueller Stand

Das produktiv genutzte Modell im Live-Betrieb ist **`best_champion.pt`** (intern referenziert als der letzte erfolgreiche Kandidat aus der Endphase kurz vor Iteration 4000). 

Zusammen mit der asynchronen Echtzeit-Pipeline, der MCTS-Ressourcenverteilung und den Hardcoded-Heuristiken (Forced Moves) bildet dieses Gewichts-Set das derzeit stabilste und spielstärkste Modell + MTC
