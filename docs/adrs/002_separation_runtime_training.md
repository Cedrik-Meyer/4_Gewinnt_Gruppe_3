# 002 - Strikte Trennung von Runtime- und Trainingssystem

## Status

Akzeptiert

## Kontext

Das Projekt erfordert einen Agenten für maschinelles Lernen, der zwei Rollen erfüllt. Zum einen muss er Live-Spiele gegen andere Agenten über eine WebSocket-Verbindung bestreiten, was kurze Antwortzeiten, schlanke Abhängigkeiten und Stabilität voraussetzt. Zum anderen muss er das Spiel durch Self-Play erlernen, was Rechenleistung, Datenspeicher im RAM und Machine-Learning-Frameworks erfordert. Werden beide Verantwortlichkeiten in einer Architektur vermischt, steigen Komplexität und Risiko von Performance-Problemen während der Turniermatches.

## Entscheidung

Das Projekt wird in ein Laufzeitsystem (`runtime_system`) und ein Trainingssystem (`training_system`) aufgeteilt. Das Laufzeitsystem übernimmt WebSocket-Kommunikation, Parsing des Server-Protokolls und Modell-Inferenz zur Vorhersage des nächsten Zuges. Das Trainingssystem verwaltet Self-Play, Replay-Buffer und Modell-Updates per Backpropagation. Die Kommunikation zwischen beiden Systemen erfolgt über exportierte Modell-Dateien (`.pt`-Checkpoints).

## Konsequenzen

Diese Trennung vereinfacht die Bereitstellung des Live-Agenten, da der Live-Server keine Trainings-Bibliotheken laden muss. Zudem ermöglicht sie parallele Arbeit an Netzwerkprotokollen und Training. Der Nachteil liegt in der Schnittstellenpflege, da die Datenstrukturen in [`data_layers.md`](../system/data_layers.md) definiert und eingehalten werden müssen, damit das Laufzeitsystem die exportierten Modelle des Trainingssystems interpretieren kann.

---
