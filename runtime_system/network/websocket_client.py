import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
import websockets

logger = logging.getLogger(__name__)


class AgentWebSocketClient:
    """
        Managt die WebSocket-Verbindung zum 4-Gewinnt-Server.
        Kapselt Reconnect-Logik, Heartbeats und das Dispatching der eingehenden Nachrichten.
        """
    def __init__(self, base_url: str, token: str, heartbeat_interval: float = 30.0, reconnect_delay: float = 5.0):
        self.base_url = base_url.rstrip('/') #Slashes entfernen
        self.token = token
        self.ws_url = f"{self.base_url}/ws/agent?token={self.token}" #URL direkt mit Token zusammenbauen
        self.websocket = None
        self.handlers = {} # Dictionary für Event-Handler
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_task = None
        self.reconnect_delay = reconnect_delay # Wartezeit in Sekunden, bevor wir nach einem Disconnect neu verbinden
        self._running = False # Flag um die Endlosschleife abbrechen zu können
        self._next_reconnect_delay = None # Einmalige Override-Wartezeit aus agent.goodbye.retryAfterMs

    def register_handler(self, msg_type: str, handler_func):
        self.handlers[msg_type] = handler_func

    async def disconnect(self): # Fährt den Client runter und beendet die Auto-Reconnect-Schleife
        self._running = False
        self.stop_heartbeat()
        if self.websocket is not None:
            try:
                await self.websocket.close()
            except Exception:
                logger.debug("Fehler beim Schließen der Verbindung in disconnect()", exc_info=True)
            self.websocket = None

    def start_heartbeat(self): # Startet Background-Task für Heartbeat
        if self.heartbeat_task is None:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def stop_heartbeat(self): # Stoppt Heartbeat-Task
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
            self.heartbeat_task = None

    async def _heartbeat_loop(self): # Schickt alle X Sekunden ein Ping (Heartbeat) an den Server
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                if self.websocket is not None:
                    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                    heartbeat_msg = {
                        "version": 1,
                        "type": "heartbeat",
                        "requestId": str(uuid.uuid4()),
                        "payload": {},
                        "timestamp": timestamp
                    }
                    await self.send_message(heartbeat_msg)
        except asyncio.CancelledError:
            pass

    async def listen(self): # Baut Verbindung auf, wartet auf Nachrichten und versucht bei Fehlern einen Reconnect.
        self._running = True
        while self._running:
            try:
                self.websocket = await websockets.connect(self.ws_url)
                self.start_heartbeat()

                async for message in self.websocket:
                    try: # Versucht Verbindung aufzubauen
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "agent.goodbye": # reconnect/retryAfterMs aus dem Protokoll respektieren
                            self._apply_goodbye(data.get("payload") or {})

                        if msg_type in self.handlers: # Lese Nachrichten aus offenen Socket
                            if asyncio.iscoroutinefunction(self.handlers[msg_type]):
                                await self.handlers[msg_type](data)
                            else:
                                self.handlers[msg_type](data)
                        else:
                            await self.handle_unregistered_message(data)
                    except json.JSONDecodeError:
                        logger.warning("Ungültiges JSON empfangen: %s", message)

            except (websockets.exceptions.WebSocketException, OSError) as exc: # Verbindungs- und Handshake-Fehler abfangen
                logger.warning("Verbindung verloren: %s", exc)
            finally: # Verbindung schließen
                self.stop_heartbeat()
                if self.websocket is not None:
                    try:
                        await self.websocket.close()
                    except Exception:
                        logger.debug("Fehler beim Schließen der Verbindung", exc_info=True)
                    self.websocket = None

            if self._running: # Reconnect
                delay = self._next_reconnect_delay if self._next_reconnect_delay is not None else self.reconnect_delay
                self._next_reconnect_delay = None
                await asyncio.sleep(delay)

    def _apply_goodbye(self, payload: dict):
        """Wertet reconnect/retryAfterMs aus agent.goodbye aus (s. agent_protocol.md)."""
        if payload.get("reconnect") is False:
            self._running = False
        retry_after_ms = payload.get("retryAfterMs")
        if isinstance(retry_after_ms, (int, float)):
            self._next_reconnect_delay = retry_after_ms / 1000

    async def handle_unregistered_message(self, data: dict):
        pass

    async def send_message(self, message: dict):
        if self.websocket is not None:
            try:
                await self.websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                logger.debug("Nachricht konnte nicht gesendet werden, Verbindung bereits geschlossen.")