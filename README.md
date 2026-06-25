# Connect4 3D Agent (Gruppe 3)

Ein autonomer, auf Machine Learning basierender Agent für **4-Gewinnt 3D (4x4x4)**.

---

# Übersicht

Dieses Projekt beinhaltet die vollständige Entwicklungsumgebung für einen 3D-Connect4-Agenten. Die Architektur nutzt maschinelles Lernen, um ein tiefes neuronales Modell zu trainieren. Dieses Modell bildet zusammen mit einer Monte Carlo Tree Search (MCTS) und dem übergeordneten Runtime-System den eigentlichen Spiele-Agenten. Zusätzlich steht eine klassische, iterative Alpha-Beta-Suche (Minimax) als Benchmark-Gegner zur Verfügung.

Das System ist in drei logische Hauptkomponenten unterteilt:

- **Training System**: Generierung von Datensätzen und Training des neuronalen Netzes (Supervised Learning & Self-Play)
- **Tools & Evaluierung**: Lokale CLI-Werkzeuge zum interaktiven Testen und für hochperformante Benchmarks (z.B. Modell + MCTS vs. Minimax)
- **Runtime System**: Ein WebSocket-Client, der das fertige Modell mit dem Live-Turnierserver verbindet

---

# Voraussetzungen und Installation

Das Projekt erfordert:

- Python >= 3.10

Der verwendete Tech-Stack basiert zentral auf Python und nutzt Pytorch für Deep Learning, Numpy für schnelle mathematische Matrix-Operationen sowie asynchrone Netzwerkkommunikation für die Live-Spiele.

Die Abhängigkeiten des Projekts werden vollständig und automatisiert über die `pyproject.toml` verwaltet.

---

# Setup-Anleitung

Es wird dringend empfohlen, eine virtuelle Umgebung (`.venv`) zu verwenden.

## Virtuelle Umgebung erstellen und aktivieren

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Abhängigkeiten installieren

Durch den folgenden Befehl wird das Projekt als lokales Paket installiert. Dadurch werden alle internen Import-Pfade korrekt aufgelöst.

```bash
pip install -e .
```

---

# Lokales Testen und Benchmarking

Für die Evaluierung der Modelle und Engines steht ein interaktives Terminal-Werkzeug zur Verfügung. Es erlaubt Einzelspiele (Mensch gegen KI) sowie vollautomatisierte Turniere (Benchmarking) über hunderte Runden zur statistischen Auswertung.

## Starten des Terminals

```bash
python src/connect4/tools/play_terminal.py
```

Anschließend werden Sie interaktiv nach folgenden Parametern gefragt:

- Anzahl der zu verwendenden CPU-Kerne
- Zeitlimit pro Zug in Millisekunden (z.B. 180 ms)

Danach können die Agenten ausgewählt werden:

- Pures Modell
- Modell + MCTS
- Einfache Engine
- Starke Engine

Die Ergebnisse automatisierter Benchmarks werden im Ordner `logs/` gespeichert.

---

# Training und Modell-Generierung

Das Training des neuronalen Netzes kann über zwei verschiedene Ansätze erfolgen.

## 1. Normales Training (Self-Play / Reinforcement Learning)

Dies ist der primäre Weg, um dem Agenten komplexe Strategien beizubringen.

Beim Self-Play spielt das Modell kontinuierlich unzählige Partien gegen sich selbst. Das MCTS dient dabei als Lehrer, der in jeder Stellung tiefere Suchbäume aufbaut und bessere Züge findet, als das Modell initial vorschlägt.

Das Modell lernt aus diesen Entdeckungen und verbessert sein strategisches Verständnis von Iteration zu Iteration.

### Start des Self-Play-Trainings

```bash
python src/connect4/training_system/training/main_train.py
```

---

## 2. Supervised Learning Pipeline

Im Gegensatz zum eigenständigen Explorieren im Self-Play lernt das Modell hier zielgerichtet anhand generierter Muster-Daten ("Behavioral Cloning" und Heuristiken).

Dieses Skript sucht automatisch nach der aktuellsten Modellversion, generiert einen neuen Datensatz und trainiert die nächste Modell-Iteration.

Beispiele:

- Klonen extrem schneller taktischer Züge einer Minimax-Engine
- Blockieren offensichtlicher 1-Step-Verluste
- Gezieltes Beseitigen taktischer Schwächen

Dieses Verfahren eignet sich besonders für gezieltes "Bugfixing" am Modell.

### Start des Supervised Trainings

```bash
python src/connect4/training_system/training/supervised_train.py
```

---

# Live-Agent: Verbindung zum Turnier-/Testserver

## Konfiguration

In der Datei `.env` im Projekt-Hauptverzeichnis müssen die Server-URL und der Agent-Token hinterlegt werden.

`AGENT_TOKEN_2` ist optional und wird nur für einen zweiten, gleichzeitig laufenden Agenten benötigt.

```env
AGENT_SERVER_URL=wss://phwt-3d4gewinnt.lennardpreusker.com
AGENT_TOKEN=ag_...
AGENT_TOKEN_2=ag_...
```

Der Standard-Checkpoint-Pfad ist als `DEFAULT_CHECKPOINT_PATH` in `runtime_system/main_live.py` hinterlegt und kann beim Start überschrieben werden.

---

## Verbinden

```bash
python -m runtime_system.main_live
```

Das Skript:

1. liest `AGENT_SERVER_URL` und `AGENT_TOKEN` aus `.env`
2. lädt das Modell aus dem konfigurierten Checkpoint
3. verbindet sich per WebSocket
4. läuft in einer Endlosschleife bis zur Beendigung

---

## Trennen

Zum Beenden genügt:

```text
Strg + C
```

Der Server erkennt die Trennung automatisch über die geschlossene WebSocket-Verbindung.

---

# Zweiten eigenen Agenten gleichzeitig starten

Zum Testen mehrerer Modelle auf dem Server kann ein zweiter Agent mit eigenem Token und optional anderem Modell gestartet werden.

Tragen Sie dazu `AGENT_TOKEN_2` in die `.env` ein und starten Sie den Agenten mit:

```bash
python -m runtime_system.main_live <TOKEN_ENV_VAR> <CHECKPOINT_PATH>
```

## Parameter

### 1. Argument

Name der Umgebungsvariable, aus der der Token gelesen wird.

Beispiel:

```text
AGENT_TOKEN_2
```

### 2. Argument

Pfad zur gewünschten Checkpoint-Datei.

---

## Beispiel

Zweiter Agent mit eigenem Modell:

```bash
python -m runtime_system.main_live AGENT_TOKEN_2 src/connect4/training_system/checkpoints/v9_champion.pt
```

Beide Prozesse können parallel in separaten Terminals gestartet und unabhängig voneinander mit `Strg + C` beendet werden.

---

# Projektstruktur

```text
src/connect4/
├── runtime_system/                 # Live-Agent und Serveranbindung
├── shared/                         # Gemeinsame Datenstrukturen und Spiellogik
├── training_system/
│   ├── training/
│   │   ├── main_train.py           # Reinforcement Learning / Self-Play
│   │   └── supervised_train.py     # Supervised Learning Pipeline
│   └── checkpoints/                # Trainierte Modelle (*.pt)
└── tools/                          # Benchmarking und lokale Testwerkzeuge
logs/                                # Benchmark- und Analyse-Logs (Projekt-Root)
```

---
# Dokumentation

Die **[Dokumentation](/docs/documentation/docu.md)** zum Projekt, der Architektur und der Implementierung

---

# Entwicklerhinweis

Für reproduzierbare Ergebnisse wird empfohlen:

- ausschließlich innerhalb einer virtuellen Umgebung zu arbeiten
- Checkpoint-Dateien (`*.pt`) nicht in Git zu versionieren
- Benchmark-Ergebnisse und Log-Dateien lokal zu halten
- neue Modelle vor dem Live-Einsatz zunächst gegen die Benchmark-Engines zu testen
