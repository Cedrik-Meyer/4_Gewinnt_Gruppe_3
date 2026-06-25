# 005 - Zustandsloser Live-Agent (Stateless Runtime)

## Status

Akzeptiert

## Kontext

Im Live-Turnierbetrieb kommuniziert unser Agent über eine asynchrone WebSocket-Verbindung mit dem Server. Bei Netzwerkverbindungen können Pakete verloren gehen oder Verbindungsabbrüche (Disconnects) auftreten. Wenn der Agent den Spielzustand lokal mitverfolgen würde (indem er das Spielfeld nach jedem eigenen oder gegnerischen Zug selbst aktualisiert), würde ein einziges verlorenes Server-Paket zu einer dauerhaften Asynchronität (Desync) zwischen Agent und Server führen. Der Agent würde auf Basis eines falschen Spielfelds agieren und illegale Züge generieren.

## Entscheidung

Wir entwerfen den Live-Agenten vollständig zustandslos (stateless). Das Laufzeitsystem merkt sich zu keinem Zeitpunkt eine Spielhistorie oder alte Spielfelder. Stattdessen parst der Agent bei jedem eingehenden `turn.request` das vom Server gesendete, vollständige 3D-Array komplett neu und nutzt ausschließlich diese Daten als Ground Truth für den Inferenz-Prozess.

## Konsequenzen

Dieser Ansatz macht das System zu 100 % resistent gegenüber Desyncs und Verbindungsabbrüchen. Selbst nach einem Reconnect kann der Agent nahtlos weiterspielen. Der Nachteil ist ein minimaler Overhead, da das Board bei jedem Zug neu geparst und der `legal_mask`-Filter von Grund auf neu berechnet werden muss. Bei der Spielfeldgröße von 4x4x4 ist dieser Rechenaufwand jedoch absolut vernachlässigbar.
