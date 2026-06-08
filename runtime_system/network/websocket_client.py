import asyncio
import json
import websockets


class AgentWebSocketClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.ws_url = f"{self.base_url}/ws/agent?token={self.token}"
        self.websocket = None
        self.handlers = {}

    def register_handler(self, msg_type: str, handler_func):
        self.handlers[msg_type] = handler_func

    async def connect(self):
        self.websocket = await websockets.connect(self.ws_url)

    async def disconnect(self):
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None

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
        if self.websocket is not None:
            await self.websocket.send(json.dumps(message))