import asyncio
import json
import uuid
from datetime import datetime, timezone
import websockets


class AgentWebSocketClient:
    def __init__(self, base_url: str, token: str, heartbeat_interval: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.ws_url = f"{self.base_url}/ws/agent?token={self.token}"
        self.websocket = None
        self.handlers = {}
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_task = None

    def register_handler(self, msg_type: str, handler_func):
        self.handlers[msg_type] = handler_func

    async def connect(self):
        self.websocket = await websockets.connect(self.ws_url)
        self.start_heartbeat()

    async def disconnect(self):
        self.stop_heartbeat()
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None

    def start_heartbeat(self):
        if self.heartbeat_task is None:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def stop_heartbeat(self):
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
            self.heartbeat_task = None

    async def _heartbeat_loop(self):
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                if self.websocket is not None and self.websocket.open:
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

    async def listen(self):
        if self.websocket is None:
            return

        async for message in self.websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type in self.handlers:
                    if asyncio.iscoroutinefunction(self.handlers[msg_type]):
                        await self.handlers[msg_type](data)
                    else:
                        self.handlers[msg_type](data)
                else:
                    await self.handle_unregistered_message(data)
            except json.JSONDecodeError:
                pass

    async def handle_unregistered_message(self, data: dict):
        pass

    async def send_message(self, message: dict):
        if self.websocket is not None and self.websocket.open:
            await self.websocket.send(json.dumps(message))