# Glossar – Maschinelles Lernen & Connect4 3D

Dieses Glossar definiert die zentralen Fachbegriffe der künstlichen Intelligenz und erklärt sie im spezifischen Kontext unseres PyTorch-basierten Connect4 3D Agenten (AlphaZero-Architektur).

**A**
* **Action / Aktion:** Ein möglicher Spielzug. Im 3D-Vier-Gewinnt entspricht eine Aktion einer der 16 Säulen (x, z-Koordinaten) auf dem 4x4-Raster.
* **Actor-Critic:** Eine duale Netzwerkarchitektur. Das `Connect4Model` fungiert gleichzeitig als "Akteur" (Policy: Welcher Zug ist gut?) und "Kritiker" (Value: Wer gewinnt das Spiel?).
* **AlphaZero:** Das zugrundeliegende Reinforcement-Learning-Paradigma des Projekts, bei dem der Agent ausschließlich durch das Spielen gegen sich selbst (Self-Play) kombiniert mit MCTS lernt.
* **Arena / Evaluation:** Das Vergleichssystem (`eval/arena.py`), in dem ein neues Kandidatenmodell gegen den aktuellen Champion spielt. Züge werden hier strikt nach Vorhersage-Maximum (Exploitation) gewählt, um die reine Spielstärke zu messen.

**B**
* **Backpropagation:** Das Lernverfahren neuronaler Netzwerke in PyTorch. Nach der Loss-Berechnung wird der Fehler rückwärts durch das Netz propagiert, um die Gewichte mittels eines Optimizers (z.B. Adam) anzupassen.
* **Batch Size:** Die Anzahl der Trainingsbeispiele, die auf einmal parallel verarbeitet werden (z. B. ein Batch von 1024 Tensor-Boards, um moderne NVIDIA-GPUs optimal auszulasten).
* **Behavioral Cloning:** Eine Technik in der Supervised-Learning-Pipeline (`supervised_train.py`). Das neuronale Netz lernt, indem es taktische Züge der extrem starken Minimax-Engine imitiert (Klonen von Verhalten).

**C**
* **Candidate:** Ein neu trainiertes Modell (z. B. in `main_train.py`), das sich in der Arena beweisen muss, bevor es zum neuen Champion werden kann.
* **Champion:** Das aktuell stärkste und freigegebene Modell (`best_champion.pt`), das vom Runtime-System für Live-Turnierspiele verwendet wird.
* **Checkpoint (.pt):** Eine Speicherdatei, die den Trainingszustand (die Gewichte/Weights) eines PyTorch-Modells enthält.
* **CNN (Convolutional Neural Network):** Ein Faltungsnetzwerk. In unserem Agenten als 3D-CNN (`nn.Conv3d`) implementiert, um dreidimensionale, räumliche Muster (wie Blockaden oder 4er-Reihen) im Spielfeld zu erkennen.

**D**
* **Dirichlet-Rauschen (Dirichlet Noise):** Ein mathematisches Rauschen, das während der Self-Play-Datengenerierung in die Basis des MCTS-Baums injiziert wird. Es zwingt den Agenten dazu, neue, unbekannte Zugwege zu explorieren.

**E**
* **Encoder (State Encoder):** Das Modul (`state_encoder.py`), das den logischen Spielzustand in eine maschinenlesbare Tensorrepräsentation (Shape `[2, 4, 4, 4]`) umwandelt.
* **Exploration / Exploitation:** Der grundlegende ML-Konflikt. *Exploration* bedeutet das bewusste Ausprobieren sub-optimaler Züge, um Neues zu lernen (wichtig im Self-Play). *Exploitation* bedeutet das rigorose Ausnutzen des bereits gelernten Wissens (wichtig im Live-Betrieb und in der Arena).

**F**
* **Feature:** Ein relevantes Merkmal der Eingabedaten. Im Projekt nutzt das Netz zwei Features (Kanäle): Position der eigenen Steine (Kanal 0) und Position der gegnerischen Steine (Kanal 1).
* **Forward-Pass (Inference):** Der Vorwärtspass durch das Netz (`predict`), bei dem Eingabedaten in Ausgaben umgewandelt werden. Im Live-Betrieb wird dies speicherschonend ohne Gradientenberechnung (`torch.no_grad()`) ausgeführt.

**G**
* **Game State / State:** Der vollständige logische Zustand des Spiels, bestehend aus dem 3D-Array, der Info über den aktuellen Spieler, Zeitlimits und UUIDs vom Server.
* **Gewichte (Weights):** Die internen numerischen Parameter des neuronalen Netzwerks (Lern-Matrix), die durch Backpropagation kontinuierlich verschoben werden, bis die Vorhersagen präzise sind.

**I**
* **Input / Eingabe:** Die Daten, die in das Modell fließen. Im Projekt ist dies der vorverarbeitete Tensor der Form `[Batch, 2, 4, 4, 4]`.
* **Invarianz-Transformation:** Ein Konzept im Encoder. Das Spielfeld wird immer aus der relativen "Ich"-Perspektive des aktuell ziehenden Spielers codiert, um dem Netz die Unterscheidung zwischen absoluten Spieler-IDs abzunehmen.

**K**
* **Knowledge Distillation:** Eine ML-Technik im Supervised Learning. Bei simplen Zügen sagt das alte Basis-Modell eine Wahrscheinlichkeitsverteilung voraus, die das neue Modell lernen soll. Dies schützt das Modell davor, sein generelles Positionsverständnis zu vergessen (Catastrophic Forgetting).

**L**
* **Learning Rate:** Die Lernrate (z. B. `1e-3` oder `2e-5`). Steuert die Schrittweite, mit der der PyTorch-Optimizer die Gewichte anpasst.
* **Legal Mask:** Ein float32-Filter (1D-Array der Größe 16). Volle Säulen werden mit `0.0` markiert. Der Filter straft unzulässige Aktionen ab (`-1e9`), damit das Netz niemals illegale Züge an den Server schickt.
* **Loss (Total, Policy, Value):** Der Fehlerwert des Modells, den es zu minimieren gilt. Er kombiniert sich aus dem Policy-Loss (Kreuzentropie: Wie falsch war der Zug?) und dem Value-Loss (MSE: Wie falsch war die Siegvorhersage?).

**M**
* **Maschinelles Lernen (ML):** Bereich der Informatik, bei dem der Agent aus Trajektorien (gespielten Partien) Muster ableitet, statt alle taktischen Spielregeln manuell per if/else programmiert zu bekommen.
* **MCTS (Monte Carlo Tree Search):** Ein asymmetrischer Suchalgorithmus (`mtc.py`), der das Netz durch Vorwärtssimulationen im Suchbaum unterstützt. Verbessert die Qualität der Trainingsdaten maßgeblich.
* **Modell (`Connect4Model`):** Die eigentliche KI-Komponente des Systems. Sie verarbeitet den 3D-Tensor und berechnet daraus Entscheidungen oder Bewertungen.

**N**
* **Neuronales Netzwerk:** Siehe Modell / CNN. Ein Konstrukt aus verbundenen Faltungs- und linearen Schichten.

**O**
* **Output / Ausgabe:** Das Ergebnis des neuronalen Netzes. Im Projekt sind das exakt zwei Tensoren: Policy (Zugwahrscheinlichkeiten) und Value (Siegchance).

**P**
* **Policy (Akteur):** Der Ausgabekopf (Policy-Head) des Modells, der bewertet, welche der 16 Säulen die beste Handlungsoption ist.
* **Policy Logits:** Die unformatierten, reinen Rohwerte, die aus dem Netz kommen. Erst durch eine Softmax-Aktivierung werden diese in verständliche Prozent-Wahrscheinlichkeiten (0.0 bis 1.0) umgewandelt.

**R**
* **Replay Buffer:** Ein FIFO-Ringspeicher (`replay_buffer.py`), in dem Hunderttausende Züge aus dem Self-Play gespeichert werden. Aus ihm werden beim Training durchmischte Batches gezogen (Sliding Window Prinzip).
* **Reproduzierbarkeit:** Die Eigenschaft, durch feste Startparameter und Seeds möglichst identische Netze trainieren zu können.
* **Root Parallelization:** Eine massive Performance-Optimierung im MCTS. Mehrere CPU-Kerne bauen isoliert eigene kleine Suchbäume auf, deren Ergebnisse am Ende vereint werden.

**S**
* **Self-Play:** Die primäre Methode der Datengenerierung. Das Modell spielt ununterbrochen gegen sich selbst, um aus eigenen Fehlern Taktiken zu entwickeln.
* **Supervised Learning:** Offline-Lernmethode (`supervised_train.py`), bei der das Modell durch einen "Master Mix" aus bekannten Best-Practice-Daten (Klonen von starken Engines) trainiert wird.

**T**
* **Tensor:** Die multidimensionale Zahlenstruktur für die PyTorch-Maschine. Grafikkarten benötigen Daten als Tensoren, um effizient Matrixmultiplikationen durchzuführen.
* **Training Loop:** Der zyklische Dauerbetrieb (`main_train.py`) aus Datengenerierung (Self-Play), Backpropagation und Arena-Evaluation.
* **Trajectory:** Die vollständige Aufzeichnung eines einzelnen Spielzugs (Status-Tensor, MCTS-Wahrscheinlichkeiten, finaler Spielausgang), die in den Replay Buffer fließt.
* **Transposition Table:** Ein Hash-Speicher (`nn_cache`), in dem MCTS und Alpha-Beta-Suchbäume sich bereits berechnete Brettzustände merken, um wiederholte Neural-Net-Inferenzen zu vermeiden.

**V**
* **Value (Kritiker):** Der Ausgabekopf (Value-Head) des Modells. Eine Tanh-Aktivierungsfunktion presst die Bewertung der Spielsituation strikt in einen Bereich zwischen `-1.0` (Sichere Niederlage) und `+1.0` (Sicherer Sieg).

**W**
* **WebSocket:** Die asynchrone, dauerhafte Netzwerkverbindung zwischen dem Live-Agenten und dem Turnier-Server (`websocket_client.py`).
* **Winrate:** Der prozentuale Anteil der gewonnenen Spiele in der Arena. Oft an einen Threshold gekoppelt (z.B. > 55%), um zu entscheiden, ob ein Modell gut genug für das Runtime-System ist.

**Z**
* **Zero-Copy:** Ein speichereffizienter Vorgang. Bei `torch.from_numpy()` wird das Numpy-Spielfeld in einen PyTorch-Tensor übersetzt, ohne den RAM durch eine neue Kopie der Daten doppelt zu belasten.