# Connect4 3D Agent (Gruppe 3)

Autonomer, Machine Learning basierender Agent für 4-Gewinnt 3D.

## Live-Agent: Verbindung zum Turnier-/Testserver

### Voraussetzungen

In `.env` im Projekt-Hauptverzeichnis müssen Server-URL und Agent-Token stehen. `AGENT_TOKEN_2` ist optional und wird nur für einen zweiten, gleichzeitig laufenden eigenen Agenten benötigt:

```
AGENT_SERVER_URL=wss://phwt-3d4gewinnt.lennardpreusker.com
AGENT_TOKEN=ag_...
AGENT_TOKEN_2=ag_...
```

Der Default-Checkpoint-Pfad ist als `DEFAULT_CHECKPOINT_PATH` in `runtime_system/main_live.py` hinterlegt (aktuell `training_system/checkpoints/old_best_champion.pt`). Er kann beim Start per zweitem Kommandozeilen-Argument überschrieben werden (siehe unten).

### Verbinden

```
.venv\Scripts\python.exe -m runtime_system.main_live
```

aus dem Projekt-Hauptverzeichnis (PowerShell oder Bash). Das Skript:

- liest `AGENT_SERVER_URL` und `AGENT_TOKEN` aus `.env`,
- lädt das Modell aus dem Checkpoint,
- verbindet sich per WebSocket und läuft in einer Endlosschleife, bis es beendet wird.

### Trennen

**Strg+C** im Terminal, in dem es läuft. Das beendet den Prozess; der Server merkt das automatisch über das geschlossene WebSocket.

### Zweiten eigenen Agenten gleichzeitig starten

Z. B. um zwei Modelle gegeneinander zu testen, optional auch mit einem anderen Modell. `AGENT_TOKEN_2` in `.env` eintragen (siehe oben) und beim Start angeben:

```
.venv\Scripts\python.exe -m runtime_system.main_live <TOKEN_ENV_VAR> <CHECKPOINT_PATH>
```

- 1. Argument: Name der Umgebungsvariable, aus der der Token gelesen wird (Default `AGENT_TOKEN`).
- 2. Argument: Pfad zur Checkpoint-Datei (Default `training_system/checkpoints/old_best_champion.pt`).

Beispiel — zweiter Agent mit eigenem Modell:

```
.venv\Scripts\python.exe -m runtime_system.main_live AGENT_TOKEN_2 training_system/checkpoints/champion_iter_3341.pt
```

Beide Prozesse parallel in zwei Terminals starten, jeweils mit **Strg+C** einzeln beenden.
