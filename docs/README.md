# Dokumentationsübersicht: Connect4 3D Agent

Willkommen im zentralen Dokumentationsverzeichnis der Gruppe 3. 

Um die Komplexität unseres Systems – bestehend aus asynchroner Netzwerkkommunikation, hochoptimierten Suchbäumen und tiefen neuronalen Netzen – beherrschbar zu machen, ist diese Dokumentation modular aufgebaut. Sie orientiert sich an einem ebenenbasierten Abstraktionsmodell (ähnlich dem C4-Modell) und dient als strukturierter Einstiegspunkt zu allen technischen Spezifikationen und historischen Entscheidungen.

---

## Systemarchitektur (`system/`)
Dieser Bereich beschreibt den aktuellen technischen Zustand des Projekts – von der High-Level-Architektur bis tief in die Implementierungsdetails.

* **[Level 1: Systemkontext & Hauptkomponenten](system/architecture_lvl1.md)** — Ein abstrakter High-Level-Überblick. Erklärt die vier Hauptkomponenten (`shared`, `runtime`, `tools`, `training`).
* **[Level 2: Modul- & Dateiebene](system/architecture_lvl2.md)** — Verbindet die Architektur mit der konkreten Ordnerstruktur und erläutert die Rolle der Dateien im systemweiten Datenfluss.
* **[Level 3: Tiefe Implementierungsdetails](system/architecture_lvl3.md)** — Der Deep-Dive: Mathematische Tensor-Transformationen, Multiprocessing-Optimierungen in den Suchbäumen und der Inferenz-Datenfluss.
* **[Datenebenen & Formate](system/data_layers.md)** — Der genaue Weg eines Spielfelds vom rohen Server-JSON, über den float32-Tensor, zurück in den JSON-Umschlag (Ebenen A bis F).
* **[Training Timeline](system/training_timeline.md)** — Das Logbuch der KI-Entwicklung: Historie der Modell-Iterationen, genutzte Hardware und Strategien.
* **[Glossar – Maschinelles Lernen](system/glossar.md)** — Ein Nachschlagewerk für fachspezifische Begriffe (z. B. *AlphaZero*, *Backpropagation*, *Dirichlet-Rauschen*).

---

## Architekturentscheidungen (`adrs/`)
Die Architecture Decision Records (ADRs) dokumentieren das *Warum* hinter unserem Systemaufbau und beleuchten getroffene technische Kompromisse.

* **[ADR-001: Ansatz (AlphaZero / Actor-Critic RL)](adrs/001_approach.md)**
* **[ADR-002: Strikte Trennung von Runtime- und Trainingssystem](adrs/002_separation_runtime_training.md)**
* **[ADR-003: Zentrale Spiel-Logik und lokales Paketmanagement](adrs/003_centralized_game_logic.md)**
* **[ADR-004: Relative Perspektiven-Codierung für den Modell-Input](adrs/004_relative_perspective_encoding.md)**
* **[ADR-005: Zustandsloser Live-Agent (Stateless Runtime)](adrs/005_stateless_agent.md)**
* **[ADR-006: Hybride Entscheidungsfindung im Live-Betrieb (Forced Moves)](adrs/006_hybrid_live_decisions.md)**
* **[ADR-007: MCTS Root Parallelization via Multiprocessing](adrs/007_mcts_root_parallelization.md)**
* **[ADR-008: Single-Core Ausführung der Alpha-Beta Engine](adrs/008_single_core_engine.md)**
* **[ADR-009: Supervised "Master Mix" Trainings-Pipeline](adrs/009_supervised_master_mix.md)**

*(Neue Entscheidungen werden auf Basis des [ADR Templates](adrs/template.md) hinzugefügt).*

---

## API & Schnittstellen (`api/`)
Spezifikationen zur asynchronen Kommunikation zwischen unserem Agenten und dem offiziellen Turnierserver.

* **[API Übersicht](api/README.md)** — Zentrale Informationen zum Turnier-Server.
* **[Agenten Protokoll](api/agent_protocol.md)** — Erwarteter WebSocket-Ablauf, JSON-Envelopes und Event-Types (`turn.request`, `move.submit`).
* **[Board Format](api/board_format.md)** — Das 3D-Koordinatensystem und die Übertragungsregeln des Spielfelds.

---

## Projektmanagement (`project/`)
Interne Organisation und Task-Tracking.

* **[Tasks & Todos](project/tasks.md)** — Aktuelle Aufgaben und Meilensteine.
* **[Allex Tasks](project/tasks_allex.txt)** — Roher Task-Export aus der Projektmanagement-Software.
