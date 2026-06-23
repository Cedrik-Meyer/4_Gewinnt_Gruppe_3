import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from runtime_system.agent.live_agent import LiveAgent
from runtime_system.network.websocket_client import AgentWebSocketClient
from runtime_system.parser.protocol_parser import build_move_submit, parse_turn_request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "training_system", "checkpoints", "old_best_champion.pt",
)


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


def make_turn_request_handler(client: AgentWebSocketClient, agent: LiveAgent):
    async def handle_turn_request(data: dict):
        game_state = parse_turn_request(data)
        if game_state is None:
            return

        policy_logits, _value = agent.predict(game_state)
        masked_logits = agent.mask_illegal_moves(policy_logits, game_state.legal_mask)
        boosted_logits = agent.apply_forced_moves(masked_logits, game_state)
        move = agent.select_action(boosted_logits)

        envelope = build_move_submit(move, game_state)
        await client.send_message(envelope)
        logger.info("Zug gesendet: x=%d z=%d (requestId=%s)", move.x, move.z, game_state.request_id)

    return handle_turn_request


def main():
    load_dotenv()

    token_env_var = sys.argv[1] if len(sys.argv) > 1 else "AGENT_TOKEN"
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CHECKPOINT_PATH

    base_url = os.environ["AGENT_SERVER_URL"]
    token = os.environ[token_env_var]

    agent = LiveAgent(checkpoint_path=checkpoint_path)
    client = AgentWebSocketClient(base_url, token)

    client.register_handler("agent.welcome", handle_agent_welcome)
    client.register_handler("move.accepted", handle_move_accepted)
    client.register_handler("error", handle_error)
    client.register_handler("agent.goodbye", handle_agent_goodbye)
    client.register_handler("turn.request", make_turn_request_handler(client, agent))

    asyncio.run(client.listen())


if __name__ == "__main__":
    main()
