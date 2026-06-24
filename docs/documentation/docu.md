# Dokumentationsübersicht: Connect4 3D Agent

Willkommen in der Dokumentation für den Connect4 3D Agenten (Gruppe 3). Um die Komplexität des Systems – bestehend aus asynchroner Netzwerkkommunikation, hochoptimierten Suchbäumen und tiefen neuronalen Netzen – beherrschbar zu machen, ist diese Dokumentation modular aufgebaut. 

Sie orientiert sich an einem ebenenbasierten Abstraktionsmodell (ähnlich dem C4-Modell). Bitte wählen Sie den gewünschten Detailgrad:

## 1. Architektur & Systemdesign

Die Kernarchitektur des Agenten ist in drei aufeinander aufbauende Detailstufen (Levels) unterteilt:

* **[Level 1: Systemkontext & Hauptkomponenten](level1.md)**
  Ein abstrakter High-Level-Überblick. Ideal als Einstiegspunkt. Erklärt die vier Hauptkomponenten (`shared`, `runtime_system`, `tools`, `training_system`), deren Verantwortlichkeiten und die groben Abhängigkeiten.
* **[Level 2: Modul- & Dateiebene](level2.md)**
  Verbindet die Architektur mit der konkreten Ordnerstruktur. Listet die wichtigsten Skripte pro Komponente auf und erläutert deren Kernfunktionen sowie die Rolle der Dateien im systemweiten Datenfluss.
* **[Level 3: Tiefe Implementierungsdetails](level3.md)**
  Der Deep-Dive für Entwickler. Enthält genaue Erklärungen zu den mathematischen Tensor-Transformationen, den Multiprocessing-Optimierungen in den MCTS/Alpha-Beta-Suchbäumen und dem chronologischen Inferenz-Datenfluss im Live-Betrieb.

## 2. Daten & interne Schnittstellen

* **[Datenformate & interne Schnittstellen](data.md)**
  Definiert die sechs Datenebenen (A bis F) des Systems. Beschreibt den genauen Weg eines Spielfelds vom rohen Server-JSON (`turn.request`), über die interne Python-Repräsentation, hinein in den float32-Tensor für das neuronale Netz und zurück in den JSON-Umschlag (`move.submit`).

## 3. Maschinelles Lernen & Training

* **[Glossar – Maschinelles Lernen](glossar.md)**
  Ein alphabetisches Nachschlagewerk für alle fachspezifischen Begriffe (z. B. *AlphaZero*, *Backpropagation*, *Invarianz-Transformation* oder *Dirichlet-Rauschen*), die im Code und in dieser Dokumentation verwendet werden.
* **[Training Timeline](training_timeline.md)**
  Das Logbuch der KI-Entwicklung. Dokumentiert die chronologische Historie der Modell-Iterationen: Wann wurde trainiert, welche Hardware wurde genutzt und welche Strategien (Self-Play vs. Supervised "Master Mix") führten zu welchen Fortschritten in der Spielstärke.