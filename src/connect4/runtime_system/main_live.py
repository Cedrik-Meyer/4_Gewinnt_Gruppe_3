import asyncio
import logging
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

from runtime_system.agent.live_agent import LiveAgent
from runtime_system.network.websocket_client import AgentWebSocketClient
from runtime_system.parser.protocol_parser import build_move_submit, parse_turn_request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = os.path.join( # Pfad des aktuellen Modells
    os.path.dirname(os.path.abspath(__file__)), "..",
    "training_system", "checkpoints", "best_champion.pt",
)

# Puffer für Netzwerklaufzeit
DEADLINE_SAFETY_MARGIN_MS = 100
# Untergrenze
MIN_TIME_LIMIT_MS = 5
# Fallback bei keiner Deadline
FALLBACK_TIME_LIMIT_MS = 2000

# Versatz zwischen Server- und lokaler Zeit berechnen
def estimate_clock_skew_ms(server_timestamp: str, local_receipt_ms: float) -> float:
    server_ms = datetime.fromisoformat(server_timestamp.replace("Z", "+00:00")).timestamp() * 1000
    return server_ms - local_receipt_ms

# Berechnung der tatsächlichen Zugzeit
def compute_time_limit_ms(game_state, clock_skew_ms: float = 0.0) -> int:
    if game_state.deadline_ms is None:
        return FALLBACK_TIME_LIMIT_MS

    now_ms = time.time() * 1000
    remaining_ms = game_state.deadline_ms - now_ms - clock_skew_ms
    budget_ms = remaining_ms - DEADLINE_SAFETY_MARGIN_MS
    return int(max(MIN_TIME_LIMIT_MS, budget_ms))


def handle_agent_welcome(data: dict):
    agent_info = data.get("payload", {}).get("agent", {})
    logger.info("Verbunden als Agent '%s' (id=%s)", agent_info.get("name"), agent_info.get("id"))

def handle_move_accepted(data: dict):
    status = data.get("payload", {}).get("match", {}).get("status")
    logger.info("Zug akzeptiert (matchId=%s, status=%s)", data.get("matchId"), status)

def handle_error(data: dict):
    payload = data.get("payload", {})
    logger.warning(
        "Server-Fehler: %s (status=%s, reason=%s)",
        payload.get("message"), payload.get("status"), payload.get("reason"),
    )

def handle_agent_goodbye(data: dict):
    payload = data.get("payload", {})
    logger.info("agent.goodbye erhalten: %s (reconnect=%s)", payload.get("message"), payload.get("reconnect"))


def make_turn_request_handler(client: AgentWebSocketClient, agent: LiveAgent): # Äußere Funktion um Agent und Client aufzurufen
    async def handle_turn_request(data: dict):
        local_receipt_ms = time.time() * 1000

        game_state = parse_turn_request(data)
        if game_state is None:
            return

        clock_skew_ms = 0.0
        server_timestamp = data.get("timestamp")
        if server_timestamp:
            try:
                clock_skew_ms = estimate_clock_skew_ms(server_timestamp, local_receipt_ms)
            except (ValueError, TypeError):
                logger.warning("Server-Zeitstempel nicht interpretierbar: %s", server_timestamp)

        time_limit_ms = compute_time_limit_ms(game_state, clock_skew_ms)
        logger.info("Zeitbudget für diesen Zug: %dms (Uhren-Versatz %.0fms)", time_limit_ms, clock_skew_ms)

        loop = asyncio.get_running_loop() # Greift das Loop ab
        move = await loop.run_in_executor( # Zugberechnung wird in einem anderen Thread verschoben, um andere Funktionen nicht zu blockieren
            None, agent.select_move, game_state, time_limit_ms
        )

        envelope = build_move_submit(move, game_state)
        await client.send_message(envelope)
        logger.info("Zug gesendet: x=%d z=%d (requestId=%s)", move.x, move.z, game_state.request_id)

    return handle_turn_request


def main():
    load_dotenv() # Lädt .env

    # Optionale Argumente für den Aufruf eines zweiten Argenten
    token_env_var = sys.argv[1] if len(sys.argv) > 1 else "AGENT_TOKEN"
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CHECKPOINT_PATH
    cores = int(sys.argv[3]) if len(sys.argv) > 3 else (os.cpu_count() or 1)

    base_url = os.environ["AGENT_SERVER_URL"]
    token = os.environ[token_env_var]

    logger.info("Live-Agent startet mit %d CPU-Kernen für die MCTS (Worker-Pool wird aufgewärmt) ...", cores)
    agent = LiveAgent(checkpoint_path=checkpoint_path, cores=cores) # Modell wird geladen und Worker vorbereitet
    client = AgentWebSocketClient(base_url, token)

    client.register_handler("agent.welcome", handle_agent_welcome)
    client.register_handler("move.accepted", handle_move_accepted)
    client.register_handler("error", handle_error)
    client.register_handler("agent.goodbye", handle_agent_goodbye)
    client.register_handler("turn.request", make_turn_request_handler(client, agent))

    try:
        asyncio.run(client.listen())
    finally:
        agent.close()


if __name__ == "__main__":
    main()