# 002 - Strikte Trennung von Runtime- und Trainingssystem

## Status

Akzeptiert

## Kontext

Das Projekt erfordert einen Agenten für maschinelles Lernen, der zwei grundlegend unterschiedliche Rollen erfüllen muss. Zum einen muss er Live-Spiele gegen andere Agenten über eine WebSocket-Verbindung bestreiten, was schnelle Antwortzeiten, schlanke Abhängigkeiten und hohe Stabilität voraussetzt. Zum anderen muss er das Spiel durch Millionen von Partien gegen sich selbst von Grund auf erlernen, was intensive Rechenleistung, große Datenspeicher im RAM und tiefe Machine-Learning-Frameworks erfordert. Wenn beide Verantwortlichkeiten in einer einzigen, monolithischen Architektur vermischt werden, wird das Live-System überladen und das Risiko von Speicherlecks oder Performance-Problemen während der offiziellen Turniermatches steigt drastisch.

## Entscheidung

Wir sind zu dem Entschluss gekommen, das Projekt strikt in ein Laufzeitsystem (`runtime_system` als Oberbau) und ein Trainingssystem (`training_system` als Unterbau) aufzuteilen. Der Oberbau übernimmt ausschließlich die WebSocket-Kommunikation, das Parsen des Server-Protokolls sowie die schlanke Modell-Inferenz zur Vorhersage des nächsten Zuges. Der Unterbau verwaltet die Selbstspiel-Schleife (Self-Play), den Replay-Buffer und die eigentlichen Modell-Updates per Backpropagation. Die Kommunikation zwischen beiden Systemen erfolgt ausschließlich über exportierte Modell-Dateien (`.pt`-Checkpoints), die in einem dafür vorgesehenen Verzeichnis abgelegt werden.

## Konsequenzen

Diese Trennung macht die Bereitstellung des Live-Agenten extrem einfach und ressourcenschonend, da der Live-Server keine schweren Trainings-Bibliotheken laden muss. Zudem ermöglicht es dem Team, parallel an den Netzwerkprotokollen zu arbeiten, ohne den ML-Trainingslauf zu gefährden. Auf der anderen Seite erhöht dieser Ansatz die Komplexität bei der Schnittstellenpflege, da wir die Datenstrukturen in der `data.md` strikt definieren und einhalten müssen, damit das Laufzeitsystem die exportierten Modelle des Trainingssystems jederzeit fehlerfrei interpretieren kann.

---