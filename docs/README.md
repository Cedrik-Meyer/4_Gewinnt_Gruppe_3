# Dokumentationsübersicht: Connect4 3D Agent

Willkommen im zentralen Dokumentationsverzeichnis der Gruppe 3. 

Um die Komplexität unseres Systems, bestehend aus asynchroner Netzwerkkommunikation, hochoptimierten Suchbäumen und tiefen neuronalen Netzen, beherrschbar zu machen, ist diese Dokumentation modular aufgebaut. Sie orientiert sich an einem ebenenbasierten Abstraktionsmodell (ähnlich dem C4-Modell) und dient als strukturierter Einstiegspunkt zu allen technischen Spezifikationen und historischen Entscheidungen.

---

## Systemarchitektur (`system/`)
Dieser Bereich beschreibt den aktuellen technischen Zustand des Projekts, von der High-Level-Architektur bis zu den Implementierungsdetails.

* **[Level 1: Systemkontext & Hauptkomponenten](system/architecture_lvl1.md)**. Abstrakter Überblick über die vier Hauptkomponenten (`shared`, `runtime_system`, `tools`, `training_system`).
* **[Level 2: Modul- & Dateiebene](system/architecture_lvl2.md)**. Zuordnung der Architektur zur Ordnerstruktur und Beschreibung der Dateien im systemweiten Datenfluss.
* **[Level 3: Tiefe Implementierungsdetails](system/architecture_lvl3.md)**. Detailanalyse der Tensor-Transformationen, Multiprocessing-Optimierungen und des Inferenz-Datenflusses.
* **[Datenebenen & Formate](system/data_layers.md)**. Beschreibung der Repräsentationswechsel vom Server-JSON über den `float32`-Tensor bis zum Antwort-JSON.
* **[Training Timeline](system/training_timeline.md)**. Chronologische Dokumentation der Modell-Iterationen, Hardware-Konfigurationen und Trainingsstrategien.
* **[Glossar, Maschinelles Lernen](system/glossar.md)**. Nachschlagewerk für fachspezifische Begriffe, z. B. *AlphaZero*, *Backpropagation* und *Dirichlet-Rauschen*.

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

* **[API Übersicht](api/README.md)**. Zentrale Informationen zum Turnier-Server.
* **[Agenten Protokoll](api/agent_protocol.md)**. Erwarteter WebSocket-Ablauf, JSON-Envelopes und Event-Types (`turn.request`, `move.submit`).
* **[Board Format](api/board_format.md)**. 3D-Koordinatensystem und Übertragungsregeln des Spielfelds.

---

## Projektmanagement (`project/`)
Interne Organisation und Task-Tracking.

* **[Tasks & Todos](project/tasks.md)**. Aktuelle Aufgaben und Meilensteine.
* **[Allex Tasks](project/tasks_allex.txt)**. Roher Task-Export aus der Projektmanagement-Software.
