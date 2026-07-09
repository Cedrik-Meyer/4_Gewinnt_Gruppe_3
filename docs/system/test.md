# Testdokumentation

Dieses Dokument beschreibt die testbasierte Qualitätssicherung des Connect4
3D Agenten. Es dokumentiert Testziele, Teststruktur, abgedeckte Komponenten
und bekannte Grenzen der Testsuite.

## 1. Zielsetzung

Die Tests prüfen die funktionale Korrektheit zentraler Systembestandteile.
Der Schwerpunkt liegt auf Komponenten, deren Fehlverhalten Auswirkungen auf
Training, Runtime oder Spielentscheidung hätte.

Geprüft werden insbesondere:

- Spielregeln für 4-Gewinnt 3D
- Transformation des Spielbretts in Modell-Eingaben
- Verarbeitung des Server-Protokolls
- Trainingskomponenten wie Replay Buffer, Self-Play und Trainer
- grundlegende Runtime-Kommunikation

Die Testsuite kombiniert isolierte Unit-Tests mit Use-Case-Tests für
modulübergreifende Abläufe.

## 2. Testframework

Als Testframework wird `pytest` verwendet. Die Abhängigkeiten sind in
`pyproject.toml` definiert.

Ausführung im Projektverzeichnis:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Bei aktivierter virtueller Umgebung kann alternativ ausgeführt werden:

```powershell
pytest
```

## 3. Teststruktur

Die Tests befinden sich im Verzeichnis `tests/`.

### Unit-Tests

Unit-Tests prüfen einzelne Module isoliert. Sie dienen der schnellen
Regressionserkennung nach Codeaenderungen.

| `shared` | Spielregeln, State Encoding, Legal-Mask |
| `runtime` | Parser, Servernachrichten, Protokollformate |
| `training` | Modell, Replay Buffer, Self-Play, Trainer, Arena, Checkpoints |

### Use-Case-Tests

Use-Case-Tests prüfen Ablaufketten über mehrere Module hinweg.

| `tests/use_cases/test_server_communication.py` | grundlegende Runtime-Kommunikation |
| `tests/use_cases/test_training_loop_mock.py` | zentrale Schritte der Trainingsschleife in vereinfachter Form |

## 4. Abgedeckte Komponenten

### Shared Core

Der Shared Core enthält Spiellogik, Datenstrukturen und State Encoding. Da
Runtime und Training diese Funktionen gemeinsam verwenden, ist dieser Bereich
besonders sicherheitsrelevant.

Abgedeckt sind:

- Erzeugung eines leeren 4x4x4-Spielbretts
- Stapelmechanik innerhalb einer Spalte
- Ablehnung ungültiger Koordinaten
- Ablehnung voller Spalten
- Gewinnerkennung auf Achsen, Flächendiagonalen und Raumdiagonalen
- relative Perspektiven-Codierung für beide Spieler
- Tensor-Shape und Datentyp der Modell-Eingaben
- Legal-Mask für freie und volle Spalten
- Flatten-Reihenfolge der Action-Indizes

### Runtime System

Das Runtime System verarbeitet externe Serverdaten. Die Tests konzentrieren
sich daher auf Parser-Verhalten und Protokollvalidierung.

Abgedeckt sind:

- gueltige Servernachrichten
- erwartete Event-Typen
- Umwandlung von Board-Daten in interne Strukturen
- Fehlerbehandlung bei unvollstaendigen Nachrichten
- Fehlerbehandlung bei ungueltigen Nachrichten

### Training System

Das Training System umfasst Modell, Datenspeicher, Self-Play und Optimierung.
Die Tests prüfen die strukturelle und mathematische Ausfuehrbarkeit dieser
Bausteine.

Abgedeckt sind:

- Initialisierung des neuronalen Netzes
- Forward Pass mit erwarteten Output-Shapes
- Speichern und Laden von Checkpoints
- Replay Buffer mit Push- und Sample-Operationen
- Self-Play bis zu einem terminalen Spielzustand
- Reward-Zuweisung für Gewinner und Verlierer
- Trainingsschritt mit Forward Pass, Loss, Backpropagation und Optimizer Step
- Arena-Auswertung zwischen Modellen

## 5. Aktueller Teststand

Aktueller Stand der Testsuite: 76 passed

Damit bestehen alle vorhandenen Unit- und Use-Case-Tests.

## 6. Beitrag zu Qualitaetskriterien

| Fehlerfreiheit | automatische Prügung zentraler Spiellogik, Parser und Trainingsschritte |
| Robustheit | Abdeckung ungültiger Eingaben und definierter Sonderfaelle |
| Wartbarkeit | schnelle Regressionserkennung nach Aenderungen |
| Modularitaet | getrennte Tests für `shared`, `runtime_system` und `training_system` |
| Nachvollziehbarkeit | Zuordnung von Testdateien zu Systemkomponenten |

## 7. Grenzen der Testsuite

Die Testsuite bewertet die technische Korrektheit der Implementierung. Die
Spielstärke des trainierten Modells wird dadurch nicht vollständig
nachgewiesen.

Nicht vollständig abgedeckt sind:

- langfristige Modellqualität nach vielen Trainingsiterationen
- statistische Spielstärke gegen starke Gegner
- Laufzeitverhalten unter Turnierbedingungen
- Netzwerkverhalten bei realen Verbindungsabbrüchen
- Verhalten bei serverseitigen Störungen

Diese Aspekte werden durch Benchmarks, Arena-Spiele und Live-Server-Tests
ergänzt.

## 8. Empfohlener Prüfablauf

Vor grösseren Commits, Abgaben oder Live-Einsaetzen wird folgender Ablauf
verwendet:

1. Virtuelle Umgebung aktivieren.
2. Vollständige Testsuite ausführen.
3. Fehlerhafte Komponente isoliert testen.
4. Vollständige Testsuite erneut ausführen.
5. Optional Benchmark über `play_terminal.py` starten.

Empfohlener Befehl:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```
