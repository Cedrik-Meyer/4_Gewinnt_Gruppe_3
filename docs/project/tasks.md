# Detaillierter Aufgabenplan (Connect4 3D Agent)

Dieses Dokument beschreibt alle Entwicklungsaufgaben im Detail. Es dient als Referenz für den Allex-Terminplan. Die Zuordnungen zu den Datenebenen (A-F) beziehen sich auf [`data_layers.md`](../system/data_layers.md).

---

## BLOCK 1: Fundament & Datenstrukturen (Gemeinsame Basis)
*Ziel: Erstellung einer konsistenten Regelimplementierung, die sowohl vom Live-Agenten als auch vom Trainings-Simulator genutzt wird.*

* **B1_01 | `Move` definieren (`src/connect4/shared/data_structures.py`):**
    Definiere eine Datenklasse (z. B. dataclass/Pydantic) für einen Spielzug. Sie muss `x` und `z` speichern. Implementiere eine Hilfsfunktion, die anhand eines übergebenen Boards die Drop-Mechanik berechnet, um das korrekte `y` (die niedrigste freie Ebene) zu ermitteln.
* **B1_02 | `GameState` definieren (`src/connect4/shared/data_structures.py`):**
    Definiere die zentrale Datenstruktur für den aktuellen Spielzustand (Ebene B). Sie enthält das `board` (3D NumPy-Array), den eigenen `player_slot`, `current_player`, `match_id`, `request_id` und ein Feld für die `legal_mask`.
* **B1_03 | Board-Init & Zug-Ausführung (`src/connect4/shared/game_logic.py`):**
    Schreibe Funktionen, um ein leeres 4x4x4 Board zu generieren und eine Funktion `apply_move(board, move, player)`, die einen Stein real in das 3D-Array fallen lässt und das Array aktualisiert.
* **B1_04 | Gewinnerkennung: Horizontal/Vertikal (`src/connect4/shared/game_logic.py`):**
    Implementiere die Such-Algorithmen, die das 3D-Array nach 4er-Reihen entlang der geraden X-, Y- und Z-Achsen durchsuchen.
* **B1_05 | Gewinnerkennung: 2D-Diagonalen (`src/connect4/shared/game_logic.py`):**
    Erweitere die Siegerkennung auf Diagonalen, die flach auf einer Ebene liegen (z. B. eine Diagonale nur auf Ebene y=0 oder entlang einer bestimmten x-Scheibe).
* **B1_06 | Gewinnerkennung: Raumdiagonalen (`src/connect4/shared/game_logic.py`):**
    Implementiere die komplexeste Siegerkennung: Die 4 Haupt-Raumdiagonalen, die quer durch den gesamten 3D-Würfel verlaufen (z. B. von x=0,y=0,z=0 nach x=3,y=3,z=3).
* **B1_07 | Unit-Tests: Züge (`tests/unit/shared/test_game_logic.py`):**
    Schreibe Tests, die prüfen, ob die Drop-Mechanik Steine korrekt stapelt und Fehler wirft, wenn versucht wird, in eine Säule zu werfen, die bereits 4 Steine (y=3) enthält.
* **B1_08 | Unit-Tests: Sieg-Szenarien (`tests/unit/shared/test_game_logic.py`):**
    Erstelle statische Test-Boards für *alle* entwickelten Siegmuster (H/V, 2D-Diag, 3D-Diag) und prüfe, ob die Logik den Gewinner zu 100 % korrekt erkennt.
* **B1_09 | CLI Mensch vs. Mensch (`src/connect4/tools/play_terminal.py`):**
    Implementiere ein Terminal-Skript, bei dem zwei Nutzer abwechselnd "x z" eingeben können. Das Skript nutzt die `game_logic.py`, gibt das Board in der Konsole aus und meldet einen Sieg.

---

## BLOCK 2: OBERBAU, Protokoll & Parser (Laufzeit-System)
*Ziel: Die fehlerfreie Übersetzung zwischen dem Server-JSON und euren internen Python-Objekten.*

* **B2_01 | JSON Parsing (`src/connect4/runtime_system/parser/protocol_parser.py`):**
    Schreibe eine Funktion, die das Server-JSON (`turn.request`, Ebene A) annimmt und daraus das `GameState`-Objekt (Ebene B) befüllt (inklusive Auslesen des `playerSlot`).
* **B2_02 | JSON Validierung (`src/connect4/runtime_system/parser/protocol_parser.py`):**
    Füge Try-Except-Blöcke hinzu. Wenn der Server ungültige Daten schickt (z. B. fehlende Felder), muss der Parser das sauber abfangen und loggen, ohne dass der Agent abstürzt.
* **B2_03 | Envelope Builder (`src/connect4/runtime_system/parser/protocol_parser.py`):**
    Schreibe die Funktion `build_move_submit(move, game_state)`. Sie nimmt den Agenten-Zug (Ebene E) und formt das strikte Server-Format (Ebene F) inklusive Zeitstempel und gespiegelter `requestId`.
* **B2_04 | Parser Mock-Tests (`tests/unit/runtime/test_protocol_parser.py`):**
    Nutze das Server-Beispiel-JSON aus der Doku, verarbeite es mit dem Parser und prüfe via Assertions, ob das `GameState`-Objekt am Ende exakt die richtigen Werte hat.

---

## BLOCK 3: OBERBAU, WebSocket & Netzwerk (Laufzeit-System)
*Ziel: Eine stabile, ausfallsichere Verbindung zum Spielserver.*

* **B3_01 | WS-Verbindung aufbauen (`src/connect4/runtime_system/network/websocket_client.py`):**
    Nutze eine Bibliothek wie `websockets`. Schreibe den Code, der sich mit der Server-URL verbindet und das Token als URL-Parameter übergibt.
* **B3_02 | WS Event-Handler (`src/connect4/runtime_system/network/websocket_client.py`):**
    Implementiere eine Endlosschleife (`async for message in websocket`), die eingehende Nachrichten auf ihren `type` prüft und sie entsprechend weiterleitet (z. B. `turn.request` an den Parser geben).
* **B3_03 | Heartbeat (`src/connect4/runtime_system/network/websocket_client.py`):**
    Schreibe einen asynchronen Task, der (falls vom Server gefordert) in regelmäßigen Abständen ein `heartbeat`-JSON sendet, damit die Verbindung nicht wegen Inaktivität gekappt wird.
* **B3_04 | Auto-Reconnect (`src/connect4/runtime_system/network/websocket_client.py`):**
    Ergänze Fehlerbehandlung (Exception Catching) für Verbindungsabbrüche. Der Client muss automatisch nach z. B. 5 Sekunden versuchen, sich neu zu verbinden.
* **B3_05 | Integrationstest Netzwerk (`tests/use_cases/test_server_communication.py`):**
    Implementiere einen lokalen Dummy-Server (mit `websockets`), starte den Client dagegen und prüfe, ob Nachrichten korrekt übertragen werden.

---

## BLOCK 4: UNTERBAU, Modell & Tensor-Codierung (Trainingssystem)
*Ziel: Die mathematische Repräsentation des Spielzustands für das Neuronale Netz und dessen Modellarchitektur.*

* **B4_01 | Invarianz-Transformation (`src/connect4/shared/state_encoder.py`):**
    Schreibe die Funktion, die das Board dynamisch auf Basis des `player_slot` invertiert. Eigene Steine werden in Kanal 0 isoliert, gegnerische in Kanal 1.
* **B4_02 | Tensor-Konvertierung (`src/connect4/shared/state_encoder.py`):**
    Wandle die getrennten Arrays in einen PyTorch-Tensor (`float32`) mit der exakten Shape `[2, 4, 4, 4]` (Ebene C) um.
* **B4_03 | Legal Mask (`src/connect4/shared/state_encoder.py`):**
    Erstelle ein 1D-Array (Größe 16). Prüfe die Höhe jeder der 16 Säulen des Boards. Ist $y<4$, setze 1 (gültig). Ist $y=4$ (voll), setze 0 (ungültig).
* **B4_04 | PyTorch Model Base (`src/connect4/training_system/neural_network/model.py`):**
    Erstelle die Klasse `class Connect4Model(nn.Module):` und richte den Input für die Größe `[2, 4, 4, 4]` ein.
* **B4_05 | 3D-Faltungs-Schichten (`src/connect4/training_system/neural_network/model.py`):**
    Füge `nn.Conv3d` Layer hinzu, um räumliche Muster (Features) zu extrahieren. Flatten die Ausgabe am Ende für die nachfolgenden linearen Schichten.
* **B4_06 | Policy-Head (`src/connect4/training_system/neural_network/model.py`):**
    Baue den ersten Output-Zweig: Ein linearer Layer (`nn.Linear`), der die extrahierten Features auf die 16 möglichen Züge (Logits) abbildet (Ebene D).
* **B4_07 | Value-Head (`src/connect4/training_system/neural_network/model.py`):**
    Baue den zweiten Output-Zweig: Ein linearer Layer, der die Features auf exakt 1 Wert (Stellungsbewertung) komprimiert und via `tanh`-Aktivierung auf den Bereich -1.0 bis +1.0 clippt.
* **B4_08 | Tensor-Tests (`tests/unit/training/test_neural_network.py`):**
    Schreibe einen Test, der einen Fake-Tensor in das ungelernte Modell wirft und prüft, ob exakt 16 Logits und 1 Value ohne Absturz zurückkommen.

---

## BLOCK 5: UNTERBAU, Datengenerierung & Trainer (Trainingssystem)
*Ziel: Der Lernprozess (Self-Play) und das Anpassen der Modell-Gewichte.*

* **B5_01 | Replay Buffer Base (`src/connect4/training_system/self_play/replay_buffer.py`):**
    Erstelle eine Klasse mit fester Maximalkapazität (z. B. 100.000 Züge), die alte Daten überschreibt, wenn sie voll ist (Ringpuffer).
* **B5_02 | Mini-Batch Sampling (`src/connect4/training_system/self_play/replay_buffer.py`):**
    Schreibe die Funktion, die zufällig eine vorgegebene Anzahl an Zügen (z. B. 64 State-Action-Reward Tuple) als Paket (`Batch`) für das Training bereitstellt.
* **B5_03 | Self-Play Simulator (`src/connect4/training_system/self_play/self_play_loop.py`):**
    Das Modell spielt ein komplettes Spiel gegen sich selbst. Es fragt das Netz nach Zügen, führt sie via `game_logic.py` aus, bis das Spiel endet (Multiprcessing berücksichtigen).
* **B5_04 | Trajectory Speicherung (`src/connect4/training_system/self_play/self_play_loop.py`):**
    Am Ende eines simulierten Spiels: Nimm alle gemachten Züge, verknüpfe sie mit dem Gewinner (Reward +1 für Sieger-Züge, -1 für Verlierer-Züge) und pushe sie in den Replay Buffer.
* **B5_05 | Loss-Funktionen (`src/connect4/training_system/training/trainer.py`):**
    Definiere, wie das Netz aus Fehlern lernt: `CrossEntropyLoss` für den Policy Head (Klassifikation des besten Zuges) und `MSELoss` für den Value Head.
* **B5_06 | Optimizer & Scheduler (`src/connect4/training_system/training/trainer.py`):**
    Richte den `torch.optim.Adam` Optimizer ein und definiere eine Learning-Rate.
* **B5_07 | Backpropagation (`src/connect4/training_system/training/trainer.py`):**
    Schreibe die Kernfunktion: Batches aus dem Buffer laden, mit dem Netz verarbeiten, Loss berechnen, `.backward()` aufrufen und Gewichte anpassen (Neues Kandidaten-Modell entsteht).

---

## BLOCK 6: UNTERBAU, Arena & Orchestrierung (Trainingssystem)
*Ziel: Qualitätskontrolle der neu trainierten Modelle und Automatisierung der Endlosschleife.*

* **B6_01 | Arena Match-Runner (`src/connect4/training_system/eval/arena.py`):**
    Lass das alte Champion-Modell und das neue Kandidaten-Modell 100 Partien gegeneinander spielen (Seitenwechsel nach 50 Partien).
* **B6_02 | Winrate & Update (`src/connect4/training_system/eval/arena.py`):**
    Zähle die Siege. Liegt die Winrate des Kandidaten über einem Threshold (z. B. 55%), gibt die Funktion `True` zurück (Kandidat wird neuer Champion).
* **B6_03 | Trainings-Loop (`src/connect4/training_system/main_train.py`):**
    Schreibe die Endlos-Schleife: 1. N Runden Self-Play, 2. M Batches trainieren, 3. Arena eval.
* **B6_04 | Checkpoint Manager (`src/connect4/training_system/main_train.py`):**
    Logik zum Speichern der Netze. Wird ein Kandidat Champion, speichere ihn per `torch.save` unter `checkpoints/best_champion.pt`.

---

## BLOCK 7: OBERBAU, Inferenz & Live-Agent (Live-Inferenz)
*Ziel: Ein schlanker Agent, der das Modell aus dem Speicher holt und für das Online-Match nutzt.*

* **B7_01 | Model Loader (`src/connect4/runtime_system/agent/live_agent.py`):**
    Initialisiere das Modell anhand der `model.py` Struktur und lade das Wissen aus `checkpoints/best_champion.pt` via `torch.load` hinein. Setze es auf `model.eval()`.
* **B7_02 | Inferenz-Pipeline (`src/connect4/runtime_system/agent/live_agent.py`):**
    Verbinde Ebene B mit dem Tensor-Encoder, verarbeite den Tensor (Ebene C) mit dem Netz und extrahiere die Logits (Ebene D).
* **B7_03 | Logit-Maskierung (`src/connect4/runtime_system/agent/live_agent.py`):**
    Maskiere die Logits, indem bei allen illegalen Zügen (`legal_mask == 0`) ein großer negativer Wert addiert wird (z. B. `-1e9`), um illegale Züge rechnerisch auszuschließen.
    Explizit überprüfen, dass ein Gewinnzug definitiv einen höheren Wert hat.
* **B7_04 | Action-Selection (`src/connect4/runtime_system/agent/live_agent.py`):**
    Wende `torch.argmax` auf die maskierten Logits an, nimm den resultierenden Index (0-15) und rechne ihn wieder in `x` und `z` zurück (Ebene E).
* **B7_05 | Zugzwang (`src/connect4/runtime_system/agent/live_agent.py`):**
    Identifiziere alle kritischen Züge, bei denen unmittelbarer Zugzwang besteht (direkter eigener Gewinn im aktuellen Zug oder das zwingende Blockieren eines gegnerischen
    Gewinns im nächsten Zug). Addiere bei diesen Spalten einen hohen positiven Wert (z. B. +1e6) auf die entsprechenden Logits, um diese Züge zu priorisieren.
    Stelle dabei explizit sicher, dass ein eigener Gewinnzug höher gewichtet wird als ein reiner Blockierzug (z. B. Gewinnzug +2e6, Blockierzug +1e6).
* **B7_06 | App Assembly (`src/connect4/runtime_system/main_live.py`):**
    Schreibe den Startcode. Initialisiere den WebSocket-Client, übergib ihm Parser und Agent, und starte die asynchrone Endlosschleife, die auf Turn-Requests des Servers lauscht.

---

## BLOCK 8: Verifikation, CLI-Game & Turniervorbereitung
*Ziel: Testen, Testen, Testen und die umfangreiche Trainingsphase starten.*

* **B8_01 | CLI Mensch vs. Modell (`src/connect4/tools/play_terminal.py`):**
    Rüste das Terminal-Spiel aus B1_09 auf: Spieler 1 bist du über die Tastatur, Spieler 2 ist der `live_agent.py`, der das Modell aus dem Checkpoint-Ordner lädt.
* **B8_02 | Trainings-Loop Mock Test (`tests/use_cases/test_training_loop_mock.py`):**
    Ein Integrationstest, der genau 1 Spiel simuliert, 1 Batch trainiert und 1 Arena-Spiel durchführt, um sicherzustellen, dass die gesamte Pipeline absturzfrei läuft.
* **MS_01 | MEILENSTEIN: 200h Self-Play:**
    *Manuelle Aufgabe:* Startet die `main_train.py` auf einem performanten Rechner und lasst das System iterativ Daten generieren und sich selbst verbessern.
* **B8_04 | Live-Server Test (`src/connect4/runtime_system/main_live.py`):**
    Startet das System gegen den echten Turnier-Server der Hochschule (Test-Modus) und spielt eine vollständige Online-Partie, um die Envelope-Struktur live zu verifizieren.
