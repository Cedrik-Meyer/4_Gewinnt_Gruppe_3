# 005 - Zustandsloser Live-Agent (Stateless Runtime)

## Status

Akzeptiert

## Kontext

Im Live-Turnierbetrieb kommuniziert der Agent über eine asynchrone WebSocket-Verbindung mit dem Server. Dabei können Pakete verloren gehen oder Verbindungsabbrüche auftreten. Wenn der Agent den Spielzustand lokal mitverfolgt, kann ein verlorenes Server-Paket zu einer Asynchronität (Desync) zwischen Agent und Server führen. Der Agent würde auf Basis eines falschen Spielfelds agieren und illegale Züge generieren.

## Entscheidung

Der Live-Agent wird zustandslos (stateless) entworfen. Das Laufzeitsystem speichert keine Spielhistorie und keine alten Spielfelder. Stattdessen parst der Agent bei jedem eingehenden `turn.request` das vom Server gesendete 3D-Array neu und nutzt diese Daten als Referenzzustand für die Inferenz.

## Konsequenzen

Dieser Ansatz reduziert das Risiko dauerhafter Desynchronisationen bei Paketverlusten oder Verbindungsabbrüchen. Nach einem Reconnect kann der Agent den vom Server gelieferten Zustand erneut als Referenz verwenden. Der Nachteil ist ein zusätzlicher Overhead, da das Board bei jedem Zug neu geparst und der `legal_mask`-Filter neu berechnet werden muss. Bei der Spielfeldgröße von 4x4x4 bleibt dieser Rechenaufwand gering.
