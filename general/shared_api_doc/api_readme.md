# WINF2025 Shared Docs

Gemeinsame Dokumentation für die Agenten-Gruppen.

## Empfohlene Lesereihenfolge

1. [Agent-Protokoll](agent-protocol.md)

   Vollständige Spezifikation für externe Agenten:

   - Verbindung per WebSocket
   - Agent-Token
   - Message-Envelope
   - `agent.welcome`
   - `turn.request`
   - `move.submit`
   - `move.accepted`
   - `error`
   - `agent.goodbye`
   - Timeouts

2. [Board-Format](board-format.md)

   Kurzreferenz zum 3D-Spielbrett:

   - `board[y][z][x]`
   - Bedeutung von `x`, `y`, `z`
   - Zellwerte `0`, `1`, `2`
   - Beispiel `board[0][1][2] = 1`

## Wichtigster Ablauf

1. Agent verbindet sich mit `/ws/agent?token=<AGENT_TOKEN>`.
2. Server sendet `agent.welcome`.
3. Server sendet `turn.request`, wenn der Agent am Zug ist.
4. Agent liest das Board aus `payload.match.state.board`.
5. Agent sendet `move.submit` mit nur `x` und `z`.
6. Server berechnet `y`, validiert den Zug und sendet `move.accepted` oder `error`.

## Kurzfassung Board

```txt
board[y][z][x]
```

- `x`: links/rechts, `0` links bis `3` rechts
- `z`: vorne/hinten, `0` vorne bis `3` hinten
- `y`: Höhe, `0` unten bis `3` oben

Agenten senden bei einem Zug nur:

```json
{
  "x": 1,
  "z": 2
}
```

Der Server berechnet die Höhe `y` automatisch.