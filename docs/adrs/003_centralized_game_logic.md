# 003 - Zentrale Spiel-Logik und lokales Paketmanagement

## Status

Akzeptiert

## Kontext

Laufzeitsystem und Trainingssystem benötigen dieselben 3D-Vier-Gewinnt-Regeln. Die Runtime nutzt diese Logik zur Validierung der Server-Zustände, das Trainingssystem zur Simulation von Self-Play-Partien. Separate Implementierungen könnten zu abweichendem Verhalten führen. Wenn das Modell Regeln lernt, die sich von den Server-Regeln unterscheiden, können illegale Züge entstehen.

## Entscheidung

Die Datenstrukturen (`Move`, `GameState`), der State-Encoder und das Regelwerk (`game_logic.py`) werden in `src/connect4/shared/` gebündelt. Um Python-Importfehler (`ModuleNotFoundError`) beim Zugriff aus Laufzeit- und Trainingsordnern zu verhindern, wird eine `pyproject.toml`-Datei im Root-Verzeichnis genutzt. Über diese Datei wird das Projekt lokal als editierbares Paket mittels `pip install -e .` installiert.

## Konsequenzen

Durch die Zentralisierung werden Regeländerungen oder Fehlerkorrekturen an der 3D-Gewinnerkennung an einer Stelle vorgenommen. Das Modell wird damit auf Basis derselben Regeln trainiert, die der Live-Agent zur Laufzeit verwendet. Voraussetzung ist die lokale Installation über `pip install -e .`.

---
