# Datenebenen & Schnittstellen (Connect4 3D Agent)

## Übersicht der Datenebenen

* **A: Server / WebSocket Protokoll (JSON)**. Netzwerkformat mit Envelope.
* **B: Interne Spielrepräsentation (Python Objects)**. Logische Validierung und Vorbereitung der Inferenz.
* **C: Modell-Input (Tensor)**. Normalisierte, relative Feature-Karten für das neuronale Netzwerk.
* **D: Modell-Output (Tensor)**. Unmaskierte Rohwerte (Logits) für Policy und Zustandswert.
* **E: Internes Zugformat (Move)**. Maskierte und decodierte Zugauswahl.
* **F: Server-Antwortformat (JSON)**. Protokollkonformer Antwort-Envelope für den Server.

---

## A: Server / Protokoll (Input/Output JSON)

### Empfang (`turn.request`)
Das Spielfeld wird als dreidimensionales Array übermittelt. 
* **Struktur:** `board[y][z][x]` mit Shape `[4][4][4]`
* **Koordinaten-Definition:**
    * `x`: Breite (0 = links, 3 = rechts)
    * `z`: Tiefe (0 = vorne, 3 = hinten)
    * `y`: Höhe (0 = unten, 3 = oben)
* **Zellwerte:**
    * `0`: Leer
    * `1`: Besetzt von `playerSlot 0`
    * `2`: Besetzt von `playerSlot 1`

---

## B: Interne Darstellung (Agent Runtime)
*(Implementiert in: `src/connect4/shared/data_structures.py`)*

### GameState (Kapselung des aktuellen Zustands)
```python
class GameState:
    board: np.uint8  # Shape [4,4,4], Werte: 0, 1, 2
    player_slot: int  # 0 oder 1 (Slot im Match)
    current_player: int  # 0 oder 1 (Wer ist am Zug laut Server)
    match_id: str  # UUIDv4 als String (für Rückkanal)
    request_id: str  # UUIDv4 als String (muss gespiegelt werden)
    legal_mask: np.ndarray  # Bool-Array / float32 [16], 1 = gültig, 0 = Säule voll
```

### Move (Internes Zug-Objekt)
```python
class Move:
    x: int  # [0..3]
    z: int  # [0..3]
    y: int  # [0..3] -> Wird anhand des Boards mathematisch bestimmt (niedrigste freie Ebene)
```

---

## C: Modell-Input (State Encoding)

Das Netzwerk verarbeitet die Daten invariant, d. h. aus Sicht des aktuellen Spielers.

$$\mathrm{state\_tensor} \in \mathbb{R}^{2 \times 4 \times 4 \times 4}$$

* **Kanal 0 (Own Stones):** Alle Steine, deren Zellwert im Board dem `player_slot` entspricht.
* **Kanal 1 (Opponent Stones):** Alle Steine des Gegners.

### Invarianz-Matrix für die Transformation (Implementiert in `src/connect4/shared/state_encoder.py`)
* Wenn `player_slot == 0`: Wert 1 -> Kanal 0, Wert 2 -> Kanal 1.

### Technische Invarianten
* **Datentyp:** `float32` Tensor
* **Metadaten:** Keine IDs, Zeitstempel oder Meta-Informationen im Tensor.

---

## D: Modell-Output
*(Erzeugt in: `src/connect4/training_system/neural_network/model.py`)*

Das Netzwerk liefert zwei Köpfe (Actor-Critic-Struktur für MCTS/RL):

### 1. Policy Head (`policy_logits`)
$$\mathrm{policy\_logits} \in \mathbb{R}^{16}$$

* **Action Mapping:** Eindimensionaler Index [0..15]
    $$\text{index} = z \times 4 + x$$
    $$x = \text{index} \pmod 4$$
    $$z = \text{index} \mathbin{//} 4$$

### 2. Value Head (`value`)
$$\text{value} \in \mathbb{R}^{1}$$

* **Wertebereich:** $[-1.0, 1.0]$
* **Bedeutung:** Bewertung der Stellung (Nähe zum Sieg/Niederlage). Erforderlich für die MCTS-Evaluierung im Trainingsprozess.

---

## E: Internes Zugformat (Inferenz-Pipeline)
Der Prozess der Entscheidungsfindung im Agenten *(Implementiert in `src/connect4/runtime_system/agent/live_agent.py`)*:

1. **Inferenz:** Übergabe von `state_tensor` an das Modell (ONNX/PyTorch) zur Berechnung von `policy_logits[16]`.
2. **Validierungs-Maske:** Berechnung der `legal_mask[16]` auf Basis von Ebene B. Wenn für eine Kombination $(x,z)$ die Höhe $y=3$ besetzt ist, ist der `legal_mask`-Eintrag für `index = z * 4 + x` gleich `0`, sonst `1`.
3. **Maskierung:** Ungültige Züge in den `policy_logits` eliminieren:

$$\mathrm{logits}_{\mathrm{masked}} = \mathrm{logits} + (1.0 - \mathrm{legal\_mask}) \cdot (-10^9)$$

4. **Auswahl:** Anwendung von `argmax` (Inferenz) oder probabilistischem Sampling (Training).
5. **Decodierung:** Gewählten `index` in $(x, z)$ umrechnen.

---

## F: Server-Antwortformat (Output JSON)
Der Agent sendet dem Server den Nachrichtenumschlag (Envelope) zurück.
*(Zusammengebaut in: `src/connect4/runtime_system/parser/protocol_parser.py`)*

```json
{
  "version": 1,
  "type": "move.submit",
  "requestId": "<identisch mit requestId aus turn.request>",
  "matchId": "<identisch mit matchId aus turn.request>",
  "agentId": "<deine_agent_id_aus_agent.welcome>",
  "payload": {
    "x": int,
    "z": int
  },
  "timestamp": "ISO-8601-Zeitstempel"
}
```

---

## Zusammenfassung des Datenflusses

### Eingehende Brettrepräsentation
```
[Ebene A: JSON board[4][4][4]]
       ↓ (Parsing & Extraktion von playerSlot/IDs)
[Ebene B: np.uint8 Spielzustand & legal_mask-Berechnung]
       ↓ (Invertierung basierend auf Perspektive des Agenten)
[Ebene C: torch.float32 / onnx.Tensor [2, 4, 4, 4]]
```

### Ausgehende Zugrepräsentation
```
[Ebene D: Modell liefert policy_logits[16] & value[1]]
       ↓ (legal_mask anwenden & Argmax/Sampling)
[Ebene E: index ∈ 0..15 auflösen zu (x, z)]
       ↓ (Zusammenbau des Netzwerk-Envelopes)
[Ebene F: Sende move.submit JSON via WebSocket]
```

---

## Invariante System-Regeln (Constraints)

1. **Formstabilität:** Das Board besitzt fix die Dimension `(4,4,4)`. Zusätzliche JSON-Felder des Servers werden zur Abwärtskompatibilität ignoriert.
2. **Säulen-Validierung:** Der Server berechnet die Höhe $y$ automatisch. Der Agent darf im `payload` **niemals** ein $y$ mitsenden. Der Zug an Position $(x,z)$ ist genau dann illegal, wenn im empfangenen Board `board[3][z][x] != 0` gilt.
3. **ID-Konsistenz:** Die `requestId` der Server-Anfrage muss bit-identisch im Antwort-JSON repliziert werden, da der Server den Zug sonst verwirft oder fehlerhaft loggt.

---

## Chronologischer Inferenz-Datenfluss (Live-Betrieb)

Der Ablauf beschreibt die Transformation der Datenebenen während eines aktiven Spielzugs auf dem Server.

### Phase 1: Empfang und Parsing ($A \rightarrow B$)
1. **Ebene A (JSON):** Das rohe JSON (`turn.request`) kommt vom Server über den WebSocket im Modul `src/connect4/runtime_system/network/websocket_client.py` an.
2. **Der Parser (`src/connect4/runtime_system/parser/protocol_parser.py`):** Der Parser nimmt Ebene A entgegen, validiert die Struktur und transformiert sie in **Ebene B (`GameState`)**. Die `legal_mask` wird berechnet, indem geprüft wird, welche Säulen bereits die maximale Höhe ($y=3$) erreicht haben.
3. **Übergabe:** Der Parser übergibt das `GameState`-Objekt (Ebene B) an den `src/connect4/runtime_system/agent/live_agent.py`.

### Phase 2: State Encoding ($B \rightarrow C$)
4. **Der Encoder (`src/connect4/shared/state_encoder.py`):** Der Agent übergibt die Brettrepräsentation an den Encoder, da das neuronale Netzwerk Tensoren verarbeitet.
5. **Ebene C (Modell-Input):** Der Encoder transformiert die Steine abhängig vom `player_slot` (Invarianz-Prinzip) und liefert den `float32`-Tensor $\mathbb{R}^{2 \times 4 \times 4 \times 4}$ (**Ebene C**) zurück. Dies ist der mathematische Input für den Forward Pass.

### Phase 3: Modellinferenz ($C \rightarrow D$)
6. **Inferenz:** Der Agent übergibt Ebene C in das im RAM befindliche Champion-Modell (`src/connect4/training_system/neural_network/model.py`).
7. **Ebene D (Modell-Output):** Das Modell berechnet den Vorwärtspass und liefert **Ebene D** zurück: `policy_logits` (16 unnormierte Rohwerte) und den `value` (Stellungsbewertung zwischen -1.0 und 1.0).
   > **Achtung:** Ebene D enthält noch keine Informationen über Spielregeln oder volle Säulen. Das Modell gibt rein probabilistische Präferenzen aus.

### Phase 4: Maskierung und Zugauswahl ($D \rightarrow E$)
8. **Maskierung:** Der Agent maskiert die `policy_logits` (Ebene D) mit der `legal_mask` (aus Ebene B). Die Werte illegaler Züge werden rechnerisch auf einen sehr kleinen Wert gesetzt:

$$\mathrm{logits}_{\mathrm{masked}} = \mathrm{logits} + (1.0 - \mathrm{legal\_mask}) \cdot (-10^9)$$

9. **Ebene E (Internes Zugformat):** Mittels `argmax` wird der Index mit dem höchsten verbleibenden Wert ausgewählt und in die Koordinaten `x` und `z` konvertiert. Das Ergebnis ist das logische Zug-Objekt (**Ebene E**).

### Phase 5: Antworterzeugung und Versand ($E \rightarrow F$)
10. **Rückkanal:** Der Agent übergibt die Zugkoordinaten (Ebene E) zurück an den `protocol_parser.py`.
11. **Ebene F (Server-Antwortformat):** Der Parser bettet die Koordinaten in den JSON-Umschlag (Envelope) inklusive der ursprünglichen `requestId` ein (**Ebene F**).
12. **Senden:** Das Netzwerk-Modul überträgt das JSON via WebSocket als `move.submit` an den externen Spielserver.
