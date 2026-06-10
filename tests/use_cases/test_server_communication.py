import asyncio
import json
import pytest
import websockets
from runtime_system.network.websocket_client import AgentWebSocketClient


async def dummy_server(websocket):
    await websocket.send(json.dumps({
        "type": "turn.request",
        "payload": {"test": True}
    }))

    async for message in websocket:
        data = json.loads(message)
        if data.get("type") == "move.submit":
            await websocket.send(json.dumps({"type": "move.accepted"}))
            break


@pytest.mark.asyncio
async def test_server_communication():
    server = await websockets.serve(dummy_server, "127.0.0.1", 8765)

    client = AgentWebSocketClient("ws://127.0.0.1:8765", "test_token", heartbeat_interval=1.0)

    turn_request_received = asyncio.Event()

    async def on_turn_request(data):
        turn_request_received.set()
        await client.send_message({
            "type": "move.submit",
            "payload": {"x": 1, "z": 2}
        })

    client.register_handler("turn.request", on_turn_request)

    client_task = asyncio.create_task(client.listen())

    try:
        await asyncio.wait_for(turn_request_received.wait(), timeout=2.0)
    finally:
        await client.disconnect()
        client_task.cancel()
        server.close()
        await server.wait_closed()

    assert turn_request_received.is_set()