import asyncio
import websockets

class AgentWebSocketClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.ws_url = f"{self.base_url}/ws/agent?token={self.token}"
        self.websocket = None

    async def connect(self):
        self.websocket = await websockets.connect(self.ws_url)

    async def disconnect(self):
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None