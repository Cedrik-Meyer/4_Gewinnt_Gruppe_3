# Board-Format für 3D 4 Gewinnt

## Übersicht

Das Board ist ein **dreifach verschachteltes Array**, das ein 4x4x4-Spielfeld repräsentiert.

Im Agent-Protokoll liegt dieses Board in:

```txt
payload.match.state.board
```

```
board[y][z][x]
```

| Ebene               | Achse | Größe | Beschreibung                     |
| ------------------- | ----- | ----- | -------------------------------- |
| 1 (äußerstes Array) | **Y** | 4     | Höhe, `0` unten bis `3` oben     |
| 2 (mittleres Array) | **Z** | 4     | Tiefe, `0` vorne bis `3` hinten  |
| 3 (innerstes Array) | **X** | 4     | Breite, `0` links bis `3` rechts |

Festlegung:

- `x = 0` ist links, `x = 3` ist rechts.
- `z = 0` ist vorne, `z = 3` ist hinten.
- `y = 0` ist unten, `y = 3` ist oben.

---

## Struktur

```txt
[                          // Y-Achse (y=0..3)
  [                        // Z-Achse (z=0..3)
    [0, 0, 0, 0],          // X-Achse: board[0][0][0..3]
    [0, 0, 1, 0],          // X-Achse: board[0][1][0..3]
    [0, 0, 0, 0],          // X-Achse: board[0][2][0..3]
    [0, 0, 0, 0]           // X-Achse: board[0][3][0..3]
  ],
  [ ... ],                 // y=1
  [ ... ],                 // y=2
  [ ... ]                  // y=3
]
```

---

## Zugriff auf eine Zelle

```
board[y][z][x]
```

**Beispiel:** Zelle bei `x=2, y=0, z=1`

```
board[0][1][2]
```

Das ist die unterste Ebene (`y = 0`), die zweite Reihe von vorne (`z = 1`) und die dritte Spalte von links (`x = 2`).

---

## Mögliche Zellwerte

| Wert | Bedeutung              |
| ---- | ---------------------- |
| `0`  | Leer                   |
| `1`  | playerSlot 1/Spieler 1 |
| `2`  | playerSlot 2/Spieler 2 |

---

## Vollständige Beispiel-Antwort

Dieses Beispiel enthält genau einen Stein: `board[0][1][2] = 1`.

```json
[
  [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
]
```

Wichtig: Agenten senden bei einem Zug nur `x` und `z`. Der Server berechnet `y` automatisch.