import time
import math
import torch
import numpy as np
import multiprocessing as mp

from shared.data_structures import Move
from shared.game_logic import apply_move, check_winner, create_empty_board
from shared.state_encoder import encode_state, get_legal_mask
from training_system.neural_network.model import Connect4Model

LIVE_NOISE_EPS = 0.10
_CACHE_CAP = 2_000_000

class TimeOutException(Exception):
    pass

# 1. MCTS KNOTEN (Node)
class MCTSNode:
    __slots__ = ['visit_count', 'value_sum', 'prior', 'children', 'is_expanded', 'terminal_value']

    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}  # Dictionary: Action -> MCTSNode
        self.is_expanded = False
        # Caching für bewiesene Siege/Niederlagen, spart extrem viele check_winner() Aufrufe!
        self.terminal_value = None

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

# 2. WORKER-STATE & HILFSFUNKTIONEN
_WORKER = {}

def _init_worker(state_dict, jit_path=None):
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    if jit_path is not None:
        model = torch.jit.load(jit_path, map_location='cpu')
        model.eval()
    else:
        model = Connect4Model()
        model.load_state_dict(state_dict)
        model.eval()

        try:
            example = torch.zeros(1, 2, 4, 4, 4)
            model = torch.jit.optimize_for_inference(torch.jit.trace(model, example))
        except Exception:
            pass

    _WORKER["model"] = model
    _WORKER["nn_cache"] = {}


def _masked_softmax(policy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    policy = policy.copy()
    policy[legal_mask == 0.0] = -1e9
    policy = policy - np.max(policy)
    exp = np.exp(policy)
    return exp / np.sum(exp)


def _evaluate(board: np.ndarray, player: int, legal_mask: np.ndarray):
    cache = _WORKER["nn_cache"]
    key = board.tobytes()
    cached = cache.get(key)
    if cached is not None:
        return cached

    state = encode_state(board, player - 1).unsqueeze(0)
    logits, val = _WORKER["model"](state)
    probs = _masked_softmax(logits.squeeze(0).numpy(), legal_mask)
    value = val.squeeze(0).item()

    if len(cache) >= _CACHE_CAP:
        cache.clear()
    cache[key] = (probs, value)
    return probs, value


def _expand(node: MCTSNode, probs: np.ndarray, legal_mask: np.ndarray):
    for action in range(16):
        if legal_mask[action] == 1.0:
            node.children[action] = MCTSNode(prior=probs[action])
    node.is_expanded = True

# 3. DER WORKER (Läuft isoliert auf einem CPU-Kern)
def mcts_worker_task(args):
    board, player, time_limit_sec, c_puct, noise_eps, seed, root_mask = args

    rng = np.random.default_rng(seed)

    end_time = time.monotonic() + (time_limit_sec * 0.95)

    root = MCTSNode(1.0)
    simulations = 0

    legal_mask = get_legal_mask(board)
    probs, _ = _evaluate(board, player, legal_mask)

    if root_mask is None:
        root_mask = legal_mask

    if noise_eps > 0.0:
        root_actions = np.flatnonzero(root_mask == 1.0)
        noise = rng.dirichlet([0.3] * len(root_actions))
        probs = probs.copy()
        probs[root_actions] = (1.0 - noise_eps) * probs[root_actions] + noise_eps * noise

    _expand(root, probs, root_mask)

    while time.monotonic() < end_time:
        node = root
        sim_board = np.copy(board)
        current_player = player
        search_path = [node]

        while node.is_expanded and len(node.children) > 0:
            best_score = -float('inf')
            best_action, best_child = None, None

            sqrt_visits = math.sqrt(node.visit_count)

            for action, child in node.children.items():
                score = child.value() + c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
                if score > best_score:
                    best_score, best_action, best_child = score, action, child

            apply_move(sim_board, Move(x=best_action % 4, z=best_action // 4), current_player)
            current_player = 2 if current_player == 1 else 1
            node = best_child
            search_path.append(node)

        if node.terminal_value is not None:
            value = node.terminal_value
        else:
            opponent = 2 if current_player == 1 else 1

            if check_winner(sim_board, opponent):
                value = -1.0
                node.terminal_value = value
            else:
                sub_mask = get_legal_mask(sim_board)
                if np.sum(sub_mask) == 0:
                    value = 0.0
                    node.terminal_value = value
                else:
                    sub_probs, value = _evaluate(sim_board, current_player, sub_mask)
                    _expand(node, sub_probs, sub_mask)

        for n in reversed(search_path):
            value = -value
            n.value_sum += value
            n.visit_count += 1

        simulations += 1

    visit_counts = {action: child.visit_count for action, child in root.children.items()}
    return (visit_counts, simulations)

# 4. HAUPT-ENGINE (Orchestriert das Multiprocessing)
class MCTSEngine:
    def __init__(self, model_path: str, num_cores: int = None):
        self.model_path = model_path
        self.model_state_dict = None
        self.jit_path = None

        if model_path.endswith(".jit"):
            try:
                torch.jit.load(model_path, map_location='cpu')
            except Exception as e:
                print(f"[!] FEHLER beim Laden des TorchScript-Modells: {e}")
                raise e
            self.jit_path = model_path
        else:
            try:
                state_dict = torch.load(self.model_path, map_location='cpu', weights_only=True)
                probe = Connect4Model()
                probe.load_state_dict(state_dict)
                self.model_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            except Exception as e:
                print(f"[!] FEHLER beim Laden des Modells: {e}")
                raise e

        if num_cores is None:
            num_cores = mp.cpu_count()
        self.num_cores = max(1, min(num_cores, mp.cpu_count()))

        self.pool = mp.Pool(
            processes=self.num_cores,
            initializer=_init_worker,
            initargs=(self.model_state_dict, self.jit_path),
        )
        self._warmup()

    def _warmup(self):
        empty_board = create_empty_board()
        seeds = np.random.SeedSequence().spawn(self.num_cores)
        warmup_tasks = [
            (empty_board, 1, 0.05, 1.5, 0.0, seeds[i], None)
            for i in range(self.num_cores)
        ]
        self.pool.map(mcts_worker_task, warmup_tasks)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    @staticmethod
    def _safe_indices(board: np.ndarray, player: int, legal_indices: list) -> list:
        opponent = 2 if player == 1 else 1
        safe = []
        for action in legal_indices:
            test_board = np.copy(board)
            apply_move(test_board, Move(x=action % 4, z=action // 4), player)

            opp_mask = get_legal_mask(test_board)
            opponent_can_win = False
            for opp_action in range(16):
                if opp_mask[opp_action] == 1.0:
                    reply_board = np.copy(test_board)
                    apply_move(reply_board, Move(x=opp_action % 4, z=opp_action // 4), opponent)
                    if check_winner(reply_board, opponent):
                        opponent_can_win = True
                        break

            if not opponent_can_win:
                safe.append(action)

        return safe if len(safe) > 0 else legal_indices

    def get_engine_move(self, board: np.ndarray, player: int, time_limit_ms: int) -> Move:
        legal_mask = get_legal_mask(board)
        legal_indices = [i for i in range(16) if legal_mask[i] == 1.0]
        if len(legal_indices) == 1:
            return Move(x=legal_indices[0] % 4, z=legal_indices[0] // 4)

        search_indices = self._safe_indices(board, player, legal_indices)

        if len(search_indices) == 1:
            return Move(x=search_indices[0] % 4, z=search_indices[0] // 4)

        root_mask = np.zeros(16, dtype=np.float32)
        for action in search_indices:
            root_mask[action] = 1.0

        time_limit_sec = time_limit_ms / 1000.0

        seeds = np.random.SeedSequence().spawn(self.num_cores)
        tasks = [
            (board, player, time_limit_sec, 1.5, LIVE_NOISE_EPS, seeds[i], root_mask)
            for i in range(self.num_cores)
        ]

        results = self.pool.map(mcts_worker_task, tasks)

        combined_visit_counts = {action: 0 for action in search_indices}
        total_simulations = 0

        for worker_visits, worker_sims in results:
            total_simulations += worker_sims
            for action, count in worker_visits.items():
                if action in combined_visit_counts:
                    combined_visit_counts[action] += count

        best_action = max(combined_visit_counts.items(), key=lambda item: item[1])[0]
        return Move(x=best_action % 4, z=best_action // 4)