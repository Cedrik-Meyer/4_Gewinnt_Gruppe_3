# 003 - Zentrale Spiel-Logik und lokales Paketmanagement

## Status

Akzeptiert

## Kontext

Sowohl das Laufzeitsystem als auch das Trainingssystem benötigen ein identisches und exaktes Verständnis der 3D-Vier-Gewinnt-Regeln. Die Runtime braucht diese Logik zur Validierung der Server-Zustände, während das Trainingssystem sie zur Simulation der Millionen Self-Play-Partien benötigt. Würden wir die Spielregeln in beiden Systemen separat implementieren, bestünde die große Gefahr eines logischen Auseinanderdriftens. Wenn das ML-Modell Regeln lernt, die sich minimal von den offiziellen Server-Regeln unterscheiden, führt dies unweigerlich zu illegalen Zügen und Disqualifikationen im offiziellen Turnierbetrieb.

## Entscheidung

Wir haben uns dafür entschieden, alle zentralen Datenstrukturen (`Move`, `GameState`), den State-Encoder sowie das komplette Regelwerk (`game_logic.py`) in einem einzigen, gemeinsamen Verzeichnis (`shared/`) zu bündeln. Um die typischen Python-Importfehler (`ModuleNotFoundError`) beim Zugriff aus den parallel liegenden Laufzeit- und Trainingsordnern zu verhindern, nutzen wir eine `pyproject.toml`-Datei im Root-Verzeichnis. Über diese Datei installieren wir das gesamte Projekt lokal als editierbares Paket mittels des Befehls `pip install -e .`.

## Konsequenzen

Durch die Zentralisierung müssen Regeländerungen oder Fehlerkorrekturen an der 3D-Gewinnerkennung nur noch an einer einzigen Stelle im Code vorgenommen werden. Dies garantiert, dass das KI-Modell auf Basis exakt derselben Regeln trainiert wird, die der Live-Agent später zur Laufzeit verwendet. Als einzige technische Hürde müssen alle Entwickler des Teams zu Projektbeginn einmalig die lokale Installation über das Terminal ausführen, um eine konsistente Entwicklungsumgebung sicherzustellen.

---