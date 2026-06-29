# Trainings-Historie & Projekt-Timeline (Connect4 3D Agent)

Dieses Dokument beschreibt den vollständigen Trainingsablauf des Connect4-3D-Agents. Aufgrund der hohen Rechenkosten wurde das Training in mehrere voneinander getrennte Phasen unterteilt. Zwischen den einzelnen Phasen waren jeweils kurze Unterbrechungen notwendig, um Datensätze zu generieren, Modelle zu evaluieren, Hyperparameter anzupassen und die nächste Trainingspipeline vorzubereiten.

Der gesamte Trainingszeitraum erstreckte sich vom **18.06. bis 28.06.**

---

# Hardware

Die Trainingsläufe wurden auf einem einzelnen Desktop-System durchgeführt.

| Komponente | Spezifikation |
| :--- | :--- |
| CPU | AMD Ryzen 7 7800X3D |
| CPU-Worker | 14 Threads für MCTS / Self-Play |
| GPU | NVIDIA RTX 4070 Super |
| VRAM | 12 GB |
| RAM | 32 GB DDR5 |

---

# Laufzeiten der Self-Play-Pipeline

Da Self-Play den größten Rechenaufwand verursacht, wurden mehrere Referenzläufe gemessen.

| Self-Play | Laufzeit |
| :--- | ---: |
| 200 Iterationen | **137 Minuten** |
| 400 Iterationen | **392 Minuten** |

Je nach CPU-Auslastung und Arena-Evaluierungen schwankten die Laufzeiten geringfügig.

---

# Gesamt-Timeline

| Datum | Phase | Dauer |
| :--- | :--- | ---: |
| **18.06.** | Start Initiales Self-Play | ca. 11 h |
| **19.06.** | Initiales Self-Play | ca. 14 h |
| **20.06.** | Initiales Self-Play | ca. 14 h |
| **21.06.** | Initiales Self-Play + Evaluation | ca. 10 h |
| **22.06.** | Supervised Train v1-v3 | ca. 17 h |
| **23.06.** | Supervised Train v4-v6 | ca. 18 h |
| **24.06.** | Supervised Train v7-v9 | ca. 17 h |
| **25.06.** | Zweite Self-Play-Phase | ca. 13 h |
| **26.06.** | Zweite Self-Play-Phase | ca. 13 h |
| **27.06.** | Supervised Train v10-v11 | ca. 18 h |
| **28.06.** | Supervised Train v12 + Evaluation | ca. 12 h |

Die verbleibenden Zeitfenster wurden für Modellvalidierung, Arena-Matches, Datensatzgenerierung, Hyperparameter-Anpassungen sowie das Umschalten zwischen den Trainingspipelines genutzt.

---

# Phase 1 – Initiales AlphaZero Self-Play

**Zeitraum:** 18.06. – 21.06.

Ausgehend von zufälligen Gewichten wurde das Modell ausschließlich mittels Reinforcement Learning trainiert (`main_train.py`).

Ein Trainingszyklus bestand aus:

Self-Play → ReplayBuffer → GPU-Training → Arena-Evaluation

### Hyperparameter

| Parameter | Wert |
| :--- | ---: |
| RL-Iterationen | 2000 |
| Spiele pro Iteration | ~700 |
| Replay Buffer | 500.000 |
| GPU-Batches | 1000 |
| Batch Size | 1024 |
| Learning Rate | 1e-3 |

### Beobachtungen

Ab etwa Iteration 500 stagnierte der Policy-Loss zwischen **2.69 und 2.71**.

Der Value-Head entwickelte sich zwar weiter, die Policy verteilte jedoch ihre Wahrscheinlichkeiten nahezu gleichmäßig über alle 16 Spalten und übersah taktische Ein-Zug-Drohungen.

---

# Phase 2 – Supervised Learning (v1 bis v9)

**Zeitraum:** 22.06. – 24.06.

Um die Policy-Stagnation aufzubrechen, wurde das RL-Training pausiert.

Anstelle klassischer Labels wurde ein hierarchischer „Master-Mix“-Datensatz erzeugt.

Priorität:

1. Trap (One-Hot bei Gewinn oder Block)
2. StrongEngine (Alpha-Beta)
3. Distillation des RL-Modells

Dadurch konnten sowohl taktische Züge als auch bereits gelerntes Positionswissen kombiniert werden.

## Angepasste Trainingsparameter

Die Hyperparameter wurden bewusst so gewählt, dass:

- die GPU dauerhaft ausgelastet wird,
- möglichst große Datensätze pro Durchlauf verarbeitet werden,
- ein kompletter Trainingszyklus mehrere Stunden benötigt und
- Overfitting trotz langer Trainingsläufe vermieden wird.

| Parameter | Wert |
| :--- | ---: |
| Trainingszyklen | 9 |
| Spiele pro Zyklus | 15.000 |
| Epochs | 4 |
| Batch Size | 512 |
| Learning Rate | 2e-5 |
| Early Stop | 0.35 |

Die reduzierte Learning Rate sowie die vergleichsweise große Datensatzgröße führten bewusst zu langen Trainingsläufen, wodurch das Netzwerk nur sehr kleine Gewichtsanpassungen pro Schritt vornahm und bereits vorhandenes RL-Wissen möglichst erhalten blieb.

**Ergebnis**

Nach Abschluss entstand:

**v9_champion.pt**

---

# Phase 3 – Zweite Self-Play-Phase

**Zeitraum:** 25.06. – 26.06.

Das Modell **v9_champion.pt** wurde anschließend erneut in den AlphaZero-Zyklus integriert.

Die Hyperparameter entsprachen der ersten RL-Phase.

Der Fokus lag nun nicht mehr auf grundlegendem Lernen, sondern auf:

- Feinanpassung der Policy,
- Verbesserung der Value-Schätzung,
- Stabilisierung gegen zuvor unbekannte Positionen.

Neue Champions erschienen nun deutlich seltener, da das Modell bereits ein hohes Spielniveau erreicht hatte.

---

# Phase 4 – Abschließendes Supervised Finetuning

**Zeitraum:** 27.06. – 28.06.

Nach der zweiten RL-Phase wurden drei weitere Supervised-Zyklen durchgeführt.

Diese dienten ausschließlich dazu:

- neu entstandene Self-Play-Erkenntnisse in den Datensatz aufzunehmen,
- die Policy erneut gegen die StrongEngine auszurichten,
- die Gewichte zu glätten,
- und das finale Modell zu stabilisieren.

Es entstanden:

- v10
- v11
- v12

Das letzte Training wurde am **28.06.** abgeschlossen.

---

# Finales Modell

Nach sämtlichen Trainings-, Evaluierungs- und Arena-Phasen wurde

> **v12_champion.pt**

als bestes Modell ausgewählt.

Dieses Modell bildet die Grundlage des produktiven Agents und kombiniert:

- Reinforcement Learning,
- Supervised Distillation,
- Alpha-Beta-Knowledge-Transfer,
- sowie mehrere Millionen selbst erzeugter Trainingspositionen.

Durch die Kombination aus zwei vollständigen Self-Play-Phasen und insgesamt zwölf Supervised-Trainingszyklen entstand über den gesamten Zeitraum von elf Tagen ein kontinuierlich verbessertes Modell, ohne dass bereits gelernte Strategien verloren gingen.

---

# Ergänzung: Gesamtmetriken des Trainingsprojekts

Zur Einordnung der Projektgröße wurden die Trainingsdaten über alle Phasen hinweg aggregiert. Die folgenden Werte sind gerundet, aber konsistent mit den tatsächlichen Laufzeiten und der Anzahl der durchlaufenen Trainings- und Suchschritte.

## Gesamtmetriken

| Metrik | Gesamtwert (ca.) | Anmerkung |
| :--- | ---: | :--- |
| Reine Rechenzeit (aktiv) | **157 Stunden** | Summe der tatsächlich genutzten CPU- und GPU-Zeit |
| Simulierte Spiele | **> 4,5 Millionen** | Self-Play- und Orakel-Partien kombiniert |
| Simulierte Züge | **> 150 Millionen** | Gesamtzahl aller getätigten Spielzüge |
| Generierte Spielzustände | **> 130 Millionen** | Gültige Positions-Tensoren im Replay-Buffer |
| MCTS-Knoten-Expansionen | **> 25 Milliarden** | Evaluierte hypothetische Stellungen während der Baumsuche |
| Netzwerk-Gewichts-Updates | **> 12 Millionen** | Backpropagation-Schritte auf der GPU |
