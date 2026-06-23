# Agent-Protokoll

Diese Datei beschreibt die Schnittstelle für Agenten.

## Grundidee

Agenten verbinden sich per WebSocket mit dem Server. Die Agenten kommunizieren nicht direkt miteinander. Der Server ist Schiedsrichter, verwaltet den Spielzustand und fragt immer genau den Agenten an, der am Zug ist.

```txt
Agent A -- WebSocket -- Server -- WebSocket -- Agent B
```

## Verbindung

Ein Agent verbindet sich mit seinem Agent-Token:

```txt
ws://<server-host>/ws/agent?token=<AGENT_TOKEN>
```

Beispiel lokal:

```txt
ws://localhost:3000/ws/agent?token=ag_example_token
```

```

Beispiel im gleichen WLAN/LAN:

```txt
ws://192.168.178.42:3000/ws/agent?token=ag_example_token
```

Server:

```txt
wss://<server-domain>/ws/agent?token=<AGENT_TOKEN>
```

Der Token gehört zu genau einem Agenten. Er wird im Server erzeugt und in der Datenbank nur gehasht gespeichert.

Im MVP darf ein Agent genau eine aktive WebSocket-Verbindung haben. Wenn derselbe Agent bereits verbunden ist und eine zweite Verbindung mit einem gültigen Token öffnet, sendet der Server `agent.goodbye` mit `reason: "duplicate_connection"` und schließt die zweite Verbindung. Der bereits verbundene Agent bleibt online.

## Agent-Tokens im MVP

Für die erste Version gibt es zwei Arten von Tokens:

| Modus        | Token-Erstellung                                                             | Zweck                                                             |
| ------------ | ---------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Testmodus    | Gruppen erstellen ihre Tokens selbst im eingeloggten Gruppenbereich.         | Eigene Agenten lokal oder gegen eigene Test-Agenten ausprobieren. |
| Turniermodus | Turnier-Tokens werden später separat durch die Server-Gruppe bereitgestellt. | Offizielle Matches und Turnierbetrieb.                            |

Im MVP gilt für den Testmodus:

- Jede Gruppe kann maximal zwei eigene Test-Agent-Tokens erstellen.
- Diese Tokens gehören zur eingeloggten Gruppe.
- Diese Tokens sind dafür gedacht, zwei eigene Agenten gegeneinander testen zu lassen.
- Test-Tokens werden nicht automatisch für den Turniermodus verwendet.
- Turnier-Tokens werden später getrennt verwaltet und nicht frei durch die Gruppen erstellt.

## Envelope

Alle WebSocket-Nachrichten verwenden denselben Envelope:

```json
{
  "version": 1,
  "type": "turn.request",
  "requestId": "uuid",
  "matchId": "uuid",
  "agentId": "uuid",
  "payload": {},
  "timestamp": "2026-05-27T12:00:00.000Z"
}
```

Felder:

| Feld        | Bedeutung                                                           |
| ----------- | ------------------------------------------------------------------- |
| `version`   | Protokollversion. Für diese Spezifikation immer `1`.                |
| `type`      | Nachrichtentyp, z. B. `turn.request` oder `move.submit`.            |
| `requestId` | Eindeutige ID zur Zuordnung von Anfrage, Antwort, Fehlern und Logs. |
| `matchId`   | ID des Matches. Bei match-unabhängigen Nachrichten optional.        |
| `agentId`   | ID des Agenten. Wird vom Server gesetzt.                            |
| `payload`   | Inhalt der Nachricht. Die Struktur hängt vom `type` ab.             |
| `timestamp` | Zeitpunkt, zu dem die Nachricht erzeugt wurde.                      |

Der Agent muss die `requestId` aus `turn.request` in seiner Antwort `move.submit` wiederverwenden. Dadurch kann der Server verspätete, doppelte oder nicht passende Antworten eindeutig zuordnen.

## Nachrichtentypen

Server an Agent:

| Type            | Bedeutung                                                |
| --------------- | -------------------------------------------------------- |
| `agent.welcome` | Verbindung wurde akzeptiert.                             |
| `agent.goodbye` | Der Server beendet die Verbindung absichtlich.           |
| `turn.request`  | Der Agent ist am Zug und soll einen Zug senden.          |
| `move.accepted` | Der Server hat den gesendeten Zug akzeptiert.            |
| `error`         | Der Server lehnt eine Nachricht ab oder meldet Probleme. |
| `heartbeat.ack` | Antwort auf einen Heartbeat des Agenten.                 |

Agent an Server:

| Type          | Bedeutung                                       |
| ------------- | ----------------------------------------------- |
| `heartbeat`   | Agent meldet, dass die Verbindung noch lebt.    |
| `move.submit` | Agent sendet seinen Zug für den aktuellen Turn. |

## Match-Status

Ein Match kann im übertragenen `match.status` diese Werte haben:

| Status     | Bedeutung                                         |
| ---------- | ------------------------------------------------- |
| `created`  | Match wurde erstellt, aber noch nicht gestartet.  |
| `active`   | Match läuft. Agenten können Züge erhalten.        |
| `finished` | Match ist regulär beendet, z. B. durch Sieg/Draw. |
| `aborted`  | Match wurde abgebrochen, z. B. durch Timeout.     |

Agenten bekommen nur während `active` einen `turn.request`. Wenn ein Match beendet ist, sendet der Server für dieses Match keinen weiteren `turn.request`.

Ein beendetes Match beendet nicht automatisch die WebSocket-Verbindung des Agenten. Der Agent darf verbunden bleiben und später neue `turn.request`-Nachrichten für andere Matches erhalten. Die Verbindung wird nur beendet, wenn der Server den WebSocket schließt oder der Agent selbst trennt.

## agent.welcome

Direkt nach erfolgreicher Authentifizierung sendet der Server:

```json
{
  "version": 1,
  "type": "agent.welcome",
  "requestId": "req-welcome-1",
  "agentId": "550e8400-e29b-41d4-a716-446655440001",
  "payload": {
    "agent": {
      "id": "550e8400-e29b-41d4-a716-446655440001",
      "name": "Gruppe A Agent",
      "groupId": "550e8400-e29b-41d4-a716-446655440010",
      "tokenType": "test"
    }
  },
  "timestamp": "2026-05-27T12:00:00.000Z"
}
```

`tokenType` ist `test` oder `tournament`. Für den MVP werden Gruppen-Agenten normalerweise mit `test` verbunden.

Wenn der Token ungültig ist, sendet der Server eine `error`-Nachricht und schließt danach die WebSocket-Verbindung.

## agent.goodbye

Wenn der Server eine Agent-Verbindung absichtlich beendet, kann er vorher `agent.goodbye` senden.

Wichtig: `agent.goodbye` ist eine Best-Effort-Nachricht. Bei harten Netzwerkabbrüchen, Server-Crashes oder sofortigen WebSocket-Close-Events kann diese Nachricht fehlen. Agenten müssen deshalb zusätzlich immer normale WebSocket-Close-Events behandeln.

```json
{
  "version": 1,
  "type": "agent.goodbye",
  "requestId": "req-goodbye-1",
  "agentId": "550e8400-e29b-41d4-a716-446655440001",
  "payload": {
    "reason": "server_shutdown",
    "message": "Server is shutting down.",
    "reconnect": true,
    "retryAfterMs": 5000
  },
  "timestamp": "2026-05-27T12:00:10.000Z"
}
```

Payload-Felder:

| Feld           | Bedeutung                                                      |
| -------------- | -------------------------------------------------------------- |
| `reason`       | Maschinenlesbarer Grund für das Beenden der Verbindung.        |
| `message`      | Kurzer menschenlesbarer Hinweis.                               |
| `reconnect`    | Gibt an, ob der Agent später erneut verbinden darf/soll.       |
| `retryAfterMs` | Empfohlene Wartezeit vor erneutem Verbindungsaufbau, optional. |

Mögliche `reason`-Werte:

| Reason                 | Bedeutung                                    |
| ---------------------- | -------------------------------------------- |
| `server_shutdown`      | Server fährt herunter.                       |
| `maintenance`          | Server geht in Wartung.                      |
| `token_revoked`        | Agent-Token wurde widerrufen.                |
| `duplicate_connection` | Derselbe Agent ist bereits verbunden.        |
| `protocol_error`       | Agent hat das Protokoll wiederholt verletzt. |
| `idle_timeout`         | Agent war zu lange inaktiv.                  |
| `match_aborted`        | Zugehöriges Match wurde abgebrochen.         |

Nach `agent.goodbye` schließt der Server die WebSocket-Verbindung.

`agent.goodbye` hat nichts mit einem normalen Match-Ende zu tun. Wenn ein Match den Status `finished` oder `aborted` erreicht, bleibt die Agent-Verbindung normalerweise offen.

## heartbeat

Agenten können regelmäßig einen Heartbeat senden:

```json
{
  "version": 1,
  "type": "heartbeat",
  "requestId": "req-heartbeat-1",
  "payload": {},
  "timestamp": "2026-05-27T12:00:05.000Z"
}
```

Der Server antwortet:

```json
{
  "version": 1,
  "type": "heartbeat.ack",
  "requestId": "req-heartbeat-1",
  "agentId": "550e8400-e29b-41d4-a716-446655440001",
  "payload": {},
  "timestamp": "2026-05-27T12:00:05.020Z"
}
```

Das Heartbeat-`payload` muss ein JSON-Objekt sein. Ein leeres Objekt reicht aus. Bei einem ungültigen Heartbeat-Payload sendet der Server `error` und kein `heartbeat.ack`.

## turn.request

Wenn ein Agent am Zug ist, sendet der Server `turn.request`.

```json
{
  "version": 1,
  "type": "turn.request",
  "requestId": "req-turn-1",
  "matchId": "550e8400-e29b-41d4-a716-446655440000",
  "agentId": "550e8400-e29b-41d4-a716-446655440001",
  "payload": {
    "match": {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "gameId": "connect-four-3d",
      "status": "active",
      "state": {
        "size": 4,
        "height": 4,
        "board": [
          [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0]
          ],
          [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ],
          [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ],
          [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ]
        ],
        "currentPlayer": 1,
        "moves": [
          { "playerSlot": 1, "x": 1, "y": 0, "z": 1 },
          { "playerSlot": 2, "x": 2, "y": 0, "z": 2 }
        ]
      },
      "players": [
        {
          "slot": 1,
          "agentId": "550e8400-e29b-41d4-a716-446655440001",
          "agentName": "Gruppe A Agent"
        },
        {
          "slot": 2,
          "agentId": "550e8400-e29b-41d4-a716-446655440002",
          "agentName": "Gruppe B Agent"
        }
      ]
    },
    "playerSlot": 1,
    "deadlineMs": 1779883205000
  },
  "timestamp": "2026-05-27T12:00:00.000Z"
}
```

Wichtige Felder im Payload:

| Feld         | Bedeutung                                                       |
| ------------ | --------------------------------------------------------------- |
| `match`      | Öffentlicher Match-Zustand inklusive Spielzustand und Spielern. |
| `playerSlot` | Slot des Agenten in diesem Match. Erlaubt sind `1` und `2`.     |
| `deadlineMs` | Unix-Zeit in Millisekunden, bis wann der Agent antworten muss.  |

## Board-Format für 3D 4 Gewinnt

Das Board wird als 3D-Array übertragen:

```txt
board[y][z][x]
```

Koordinaten:

| Koordinate | Bedeutung                           | Werte |
| ---------- | ----------------------------------- | ----- |
| `x`        | links/rechts, von links nach rechts | 0-3   |
| `z`        | Tiefe, von vorne nach hinten        | 0-3   |
| `y`        | Höhe, von unten nach oben           | 0-3   |

Festlegung:

- `x = 0` ist links, `x = 3` ist rechts.
- `z = 0` ist vorne, `z = 3` ist hinten.
- `y = 0` ist unten, `y = 3` ist oben.

Zellwerte:

| Wert | Bedeutung      |
| ---- | -------------- |
| `0`  | leer           |
| `1`  | `playerSlot 1` |
| `2`  | `playerSlot 2` |

Beispiel:

```txt
board[0][1][2] = 1
```

Das bedeutet: Auf der untersten Ebene (`y = 0`), in Reihe `z = 1`, Spalte `x = 2`, liegt ein Stein von `playerSlot 1`.

## move.submit

Der Agent sendet nur `x` und `z`. Der Server berechnet `y` automatisch als niedrigste freie Höhe in dieser Säule.

```json
{
  "version": 1,
  "type": "move.submit",
  "requestId": "req-turn-1",
  "matchId": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "x": 1,
    "z": 2
  },
  "timestamp": "2026-05-27T12:00:01.000Z"
}
```

Regeln für `move.submit`:

- `x` muss eine Zahl von `0` bis `3` sein.
- `z` muss eine Zahl von `0` bis `3` sein.
- Der Agent sendet kein `y`.
- Der Agent darf nur antworten, wenn er eine `turn.request` erhalten hat.
- Der Agent muss dieselbe `requestId` verwenden wie in der `turn.request`.
- Eine fehlende oder falsche `requestId` kann als Protokollfehler behandelt werden.

## move.accepted

Wenn der Zug gültig war, bestätigt der Server. Das Feld `match.status` kann danach weiterhin `active` sein oder bereits `finished`, falls der Zug das Spiel beendet hat.

```json
{
  "version": 1,
  "type": "move.accepted",
  "requestId": "req-turn-1",
  "matchId": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "match": {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "gameId": "connect-four-3d",
      "status": "active"
    }
  },
  "timestamp": "2026-05-27T12:00:01.050Z"
}
```

Danach sendet der Server automatisch den nächsten `turn.request` an den nächsten Agenten.

Wenn der akzeptierte Zug das Spiel beendet hat, sendet der Server keinen nächsten `turn.request` für dieses Match.

## error

Bei ungültigen Nachrichten oder ungültigen Zügen sendet der Server:

```json
{
  "version": 1,
  "type": "error",
  "requestId": "req-turn-1",
  "matchId": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "status": 400,
    "reason": "column_full",
    "message": "Selected column is full."
  },
  "timestamp": "2026-05-27T12:00:01.050Z"
}
```

Payload-Felder:

| Feld      | Bedeutung                                                                  |
| --------- | -------------------------------------------------------------------------- |
| `status`  | Optionaler HTTP-ähnlicher Statuscode, z. B. `400`, `401`, `404` oder `409`. |
| `reason`  | Optionaler maschinenlesbarer Fehlergrund.                                  |
| `message` | Menschenlesbare Fehlermeldung.                                             |

Mögliche Fehlerfälle:

- ungültiges JSON
- unbekannter Nachrichtentyp
- falsches Nachrichtenformat
- ungültiger Zug
- Agent ist nicht am Zug
- Match existiert nicht
- Zeitlimit überschritten (`turn_timeout`)

Typische `reason`-Werte:

| Reason                      | Bedeutung                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| `invalid_agent_token`       | Der Token ist ungültig, widerrufen oder gehört zu einem deaktivierten Agenten.                     |
| `rate_limit_exceeded`       | Eine Aktion wurde zu häufig versucht. Der Agent sollte langsamer erneut versuchen.                 |
| `invalid_message_format`    | Die WebSocket-Nachricht entspricht nicht dem Envelope- oder Payload-Format.                        |
| `invalid_message_type`      | Der Nachrichtentyp ist für den Agent-WebSocket nicht bekannt oder nicht erlaubt.                   |
| `invalid_heartbeat_payload` | Das Heartbeat-Payload ist kein gültiges JSON-Objekt.                                               |
| `unknown_turn_request`      | `matchId` oder `requestId` passen zu keiner aktuell offenen `turn.request`.                        |
| `match_mismatch`            | Die `requestId` gehört zu einer anderen Match-Anfrage als die gesendete `matchId`.                 |
| `agent_not_current_player`  | Der Agent ist aktuell nicht am Zug.                                                               |
| `invalid_move_payload`      | `payload` enthält keine ganzzahligen Koordinaten `x` und `z` im Bereich `0` bis `3`.               |
| `invalid_move`              | Der gesendete Zug ist nach Spielregeln ungültig.                                                  |
| `not_your_turn`             | Der Agent ist aktuell nicht am Zug. Wird in v1 vor allem in älteren Beispielen synonym verwendet.  |
| `turn_timeout`              | Der Agent hat nicht vor `deadlineMs` geantwortet.                                                 |
| `request_mismatch`          | Die `requestId` passt nicht zur offenen `turn.request`. Wird in v1 als `unknown_turn_request` gemeldet. |
| `match_not_found`           | Das referenzierte Match existiert nicht oder ist beendet.                                         |
| `unknown_game_id`           | Für das Match ist serverseitig kein passender Game-Adapter bekannt.                               |

## Timeouts

Jede `turn.request` enthält `deadlineMs`.

Für v1 gilt:

- Antwortet der Agent nicht rechtzeitig, wird das Match abgebrochen.
- Der maschinenlesbare Abbruchgrund lautet `turn_timeout`.
- Der Server informiert Zuschauer über `match.updated`.

## Kompatibilität

Agenten müssen zusätzliche unbekannte Felder ignorieren. Dadurch kann der Server später neue Informationen ergänzen, ohne alte Agenten sofort zu brechen.

Breaking Changes erhöhen später `version`.

## Minimaler Agent-Ablauf

1. WebSocket mit Token öffnen.
2. `agent.welcome` abwarten.
3. Optional regelmäßig `heartbeat` senden.
4. Auf `turn.request` warten.
5. Aus `payload.match.state.board` einen Zug wählen.
6. `move.submit` mit `{ "x": ..., "z": ... }` senden.
7. `move.accepted` oder `error` loggen.
8. Wieder auf `turn.request` warten.
9. `agent.goodbye` und WebSocket-Close-Events sauber loggen.

## Mindestanforderung für euren Agenten

Ein Agent ist für die v1 ausreichend, wenn er:

- per WebSocket verbinden kann,
- `agent.welcome` akzeptiert,
- `turn.request` parsen kann,
- das Board aus `payload.match.state.board` liest,
- einen gültigen Zug als `move.submit` mit `x` und `z` sendet,
- `error`-Nachrichten loggt,
- `agent.goodbye` und WebSocket-Close-Events behandelt,
- unbekannte zusätzliche Felder ignoriert.