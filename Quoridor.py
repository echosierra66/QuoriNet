import pygame
import sys
import os
import json
import random
import time
import threading
import argparse
import math
import io
from collections import deque
from collections.abc import Sequence
from functools import lru_cache
import torch
import torch.nn as nn
import torch.optim as optim

BOARD_SIZE = 9
STATE_CHANNELS = 8
STATE_SHAPE = (STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
VALUE_WIDTH = 1


class ReplayBuffer(Sequence):
    """Bounded ring with O(1) append and random access for random.sample."""

    def __init__(self, maxlen):
        if maxlen < 1:
            raise ValueError("replay capacity must be positive")
        self.maxlen = maxlen
        self._items = []
        self._start = 0

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._items[(self._start + index) % len(self)]

    def append(self, item):
        if len(self) < self.maxlen:
            self._items.append(item)
        else:
            self._items[self._start] = item
            self._start = (self._start + 1) % self.maxlen

    def extend(self, items):
        for item in items:
            self.append(item)

    def clear(self):
        self._items.clear()
        self._start = 0


def mirror_state_batch(states):
    """Reflect columns; wall anchors occupy an 8x8 grid, pawns a 9x9 grid."""
    mirrored = states.flip(-1).clone()
    mirrored[:, 2:4] = 0
    mirrored[:, 2:4, :8, :8] = states[:, 2:4, :8, :8].flip(-1)
    return mirrored


def validate_state_tensor(tensor, name="state"):
    """Enforce the unbatched neural-state contract: float32 [8, 9, 9]."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tuple(tensor.shape) != STATE_SHAPE:
        raise ValueError(
            f"{name} must have shape {STATE_SHAPE}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains NaN or infinite values")
    return tensor


def make_state_batch(state_tensors, device, name="states"):
    """Create one contiguous float32 [B, 8, 9, 9] batch on model device."""
    if not state_tensors:
        raise ValueError(f"{name} cannot be empty")
    checked = [
        validate_state_tensor(tensor, f"{name}[{index}]")
        for index, tensor in enumerate(state_tensors)
    ]
    batch = torch.stack(checked, dim=0).to(device=device, dtype=torch.float32)
    expected = (len(checked), *STATE_SHAPE)
    if tuple(batch.shape) != expected:
        raise RuntimeError(f"{name} batch must have shape {expected}, got {tuple(batch.shape)}")
    return batch.contiguous()


def make_value_batch(values, device, name="targets"):
    """Create one float32 [B, 1] value batch without allowing broadcasting."""
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        raise ValueError(f"{name} cannot be empty")
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError(f"{name} contains NaN or infinite values")
    batch = torch.tensor(numeric_values, dtype=torch.float32, device=device).reshape(-1, VALUE_WIDTH)
    expected = (len(numeric_values), VALUE_WIDTH)
    if tuple(batch.shape) != expected:
        raise RuntimeError(f"{name} must have shape {expected}, got {tuple(batch.shape)}")
    return batch


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("value model has no parameters") from exc


def predict_value_scalar(model, state_tensor, name="state"):
    """Evaluate exactly one [8, 9, 9] state and return its one scalar value."""
    device = model_device(model)
    state_batch = make_state_batch([state_tensor], device=device, name=name)
    with torch.no_grad():
        output = model(state_batch)
    if tuple(output.shape) != (1, VALUE_WIDTH):
        raise RuntimeError(
            f"value model must return shape (1, {VALUE_WIDTH}), got {tuple(output.shape)}"
        )
    value = float(output[0, 0].item())
    if not math.isfinite(value):
        raise FloatingPointError("value model returned NaN or infinity")
    return value


def validate_model_contract(model, name="value model"):
    """Smoke-test the complete [B, 8, 9, 9] -> [B, 1] contract."""
    device = model_device(model)
    was_training = model.training
    try:
        model.eval()
        samples = [torch.zeros(STATE_SHAPE, dtype=torch.float32) for _ in range(3)]
        batch = make_state_batch(samples, device=device, name=f"{name} smoke states")
        with torch.no_grad():
            output = model(batch)
        expected = (3, VALUE_WIDTH)
        if tuple(output.shape) != expected:
            raise RuntimeError(
                f"{name} returned {tuple(output.shape)}; expected {expected}"
            )
        if not torch.isfinite(output).all().item():
            raise FloatingPointError(f"{name} smoke output contains NaN or infinity")
    finally:
        model.train(was_training)


def validate_optimizer_state_shapes(optimizer, name="optimizer"):
    """Reject stale optimizer buffers whose shapes no longer match parameters."""
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim > 0 and tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"{name} state '{key}' has shape {tuple(value.shape)} for "
                    f"parameter shape {tuple(parameter.shape)}"
                )
            if not torch.isfinite(value).all().item():
                raise FloatingPointError(f"{name} state '{key}' contains NaN or infinity")


def checked_value_loss(loss_fn, predictions, targets, name="value loss"):
    """Compute scalar loss only after proving [B, 1] shapes match exactly."""
    if predictions.ndim != 2 or predictions.shape[1] != VALUE_WIDTH:
        raise ValueError(
            f"{name} predictions must have shape [B, {VALUE_WIDTH}], "
            f"got {tuple(predictions.shape)}"
        )
    if targets.ndim != 2 or targets.shape[1] != VALUE_WIDTH:
        raise ValueError(
            f"{name} targets must have shape [B, {VALUE_WIDTH}], "
            f"got {tuple(targets.shape)}"
        )
    if predictions.shape != targets.shape:
        raise ValueError(
            f"{name} shape mismatch: predictions {tuple(predictions.shape)} "
            f"vs targets {tuple(targets.shape)}"
        )
    loss = loss_fn(predictions, targets)
    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise FloatingPointError(
            f"{name} must be one finite scalar, got shape {tuple(loss.shape)} "
            f"and value {loss.detach().cpu()}"
        )
    return loss

class TerminalLogger:
    def __init__(self, stream):
        self.stream = stream
        self.logs = deque(maxlen=50)

    def write(self, message):
        self.stream.write(message)
        if message.strip():
            for line in message.splitlines():
                if line.strip():
                    self.logs.append(line)

    def flush(self):
        self.stream.flush()

sys.stdout = TerminalLogger(sys.stdout)

# ==========================================
# NEW NEURAL NETWORK ARCHITECTURE
# ==========================================
class SpatialQuoriNetV2(nn.Module):
    def __init__(self, in_channels=STATE_CHANNELS):
        # Channels: P1 pos, P2 pos, H-wall, V-wall, P1 walls-left, P2 walls-left, P1 path-dist, P2 path-dist
        super().__init__()
        if in_channels != STATE_CHANNELS:
            raise ValueError(
                f"SpatialQuoriNetV2 requires {STATE_CHANNELS} input channels, "
                f"got {in_channels}"
            )
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.value_head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, x):
        expected = (self.in_channels, BOARD_SIZE, BOARD_SIZE)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"network input must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"network input must have shape [B, {self.in_channels}, "
                f"{BOARD_SIZE}, {BOARD_SIZE}], got {tuple(x.shape)}"
            )
        if x.shape[0] < 1:
            raise ValueError("network input batch cannot be empty")
        if x.dtype != torch.float32:
            raise TypeError(f"network input must have dtype torch.float32, got {x.dtype}")
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        value = self.value_head(x)
        expected_output = (x.shape[0], VALUE_WIDTH)
        if tuple(value.shape) != expected_output:
            raise RuntimeError(
                f"network output must have shape {expected_output}, got {tuple(value.shape)}"
            )
        return value

# Global Helper for encode_state distance calcs
def get_adjacent_global(state, c, r):
    moves = []
    if r > 0 and (c, r - 1) not in state['walls_h'] and (c - 1, r - 1) not in state['walls_h']: moves.append((c, r - 1))
    if r < 8 and (c, r) not in state['walls_h'] and (c - 1, r) not in state['walls_h']: moves.append((c, r + 1))
    if c > 0 and (c - 1, r) not in state['walls_v'] and (c - 1, r - 1) not in state['walls_v']: moves.append((c - 1, r))
    if c < 8 and (c, r) not in state['walls_v'] and (c, r - 1) not in state['walls_v']: moves.append((c + 1, r))
    return moves

@lru_cache(maxsize=8192)
def _goal_distances(walls_h, walls_v, target_row):
    """One reverse BFS answers distances from every square for a wall layout."""
    state = {'walls_h': frozenset(walls_h), 'walls_v': frozenset(walls_v)}
    distances = [999] * (BOARD_SIZE * BOARD_SIZE)
    queue = deque((c, target_row) for c in range(BOARD_SIZE))
    for c in range(BOARD_SIZE):
        distances[target_row * BOARD_SIZE + c] = 0
    while queue:
        pos = queue.popleft()
        dist = distances[pos[1] * BOARD_SIZE + pos[0]]
        for nxt in get_adjacent_global(state, *pos):
            index = nxt[1] * BOARD_SIZE + nxt[0]
            if distances[index] == 999:
                distances[index] = dist + 1
                queue.append(nxt)
    return tuple(distances)


def get_distance_field(state, pid):
    return _goal_distances(frozenset(state['walls_h']), frozenset(state['walls_v']),
                           0 if pid == 1 else 8)


def get_shortest_path_dist(state, pid):
    c, r = state[f'p{pid}_pos']
    return get_distance_field(state, pid)[r * BOARD_SIZE + c]

def encode_state(state, active_pid):
    if active_pid not in (1, 2):
        raise ValueError(f"active_pid must be 1 or 2, got {active_pid}")
    # 8 Channels exactly as specified
    tensor = torch.zeros(STATE_SHAPE, dtype=torch.float32)
    
    my_pos = state['p1_pos'] if active_pid == 1 else state['p2_pos']
    opp_pos = state['p2_pos'] if active_pid == 1 else state['p1_pos']
    
    if active_pid == 2:
        my_pos = (8 - my_pos[0], 8 - my_pos[1])
        opp_pos = (8 - opp_pos[0], 8 - opp_pos[1])
        
    tensor[0, my_pos[1], my_pos[0]] = 1.0
    tensor[1, opp_pos[1], opp_pos[0]] = 1.0
    
    for c, r in state['walls_h']:
        if active_pid == 2: c, r = 7 - c, 7 - r
        tensor[2, r, c] = 1.0
            
    for c, r in state['walls_v']:
        if active_pid == 2: c, r = 7 - c, 7 - r
        tensor[3, r, c] = 1.0
            
    my_walls = state['p1_walls'] if active_pid == 1 else state['p2_walls']
    opp_walls = state['p2_walls'] if active_pid == 1 else state['p1_walls']
    my_dist = get_shortest_path_dist(state, active_pid)
    opp_dist = get_shortest_path_dist(state, 2 if active_pid == 1 else 1)
    
    tensor[4, :, :] = my_walls / 10.0
    tensor[5, :, :] = opp_walls / 10.0
    tensor[6, :, :] = my_dist / 81.0
    tensor[7, :, :] = opp_dist / 81.0
    
    return validate_state_tensor(tensor, "encoded state")

def get_state_key(state):
    return (state['p1_pos'], state['p2_pos'], tuple(sorted(state['walls_h'])), tuple(sorted(state['walls_v'])))

class QuoridorBot:
    # How many of a player's own most-recent squares to remember, purely to
    # discourage short A<->B (or small-loop) oscillation when two paths are
    # otherwise perfectly tied. This is a soft nudge (see _oscillation_penalty),
    # not a hard rule, so a truly forced backtrack is still allowed.
    RECENT_MEMORY_LEN = 8
    OSCILLATION_PENALTY = 45.0
    TOP_MOVE_POOL_SIZE = 4
    WIDE_MOVE_POOL_SIZE = 10
    WIDE_MOVE_SAMPLE_CHANCE = 0.10

    def __init__(self, bot_id=2, max_depth=3, nn_model=None, heuristic_weight=1.0):
        self.bot_id = bot_id
        self.max_depth = max_depth
        self.nn_model = nn_model
        self.heuristic_weight = heuristic_weight
        self.was_last_random = False
        self._value_cache = {}

    def get_move_and_eval(self, state, epsilon=0.0):
        # Parameters may change between moves. Never reuse neural values across
        # optimizer steps, but do share them throughout this search.
        self._value_cache.clear()
        for move in self.get_valid_moves(state, self.bot_id):
            if move[1] == self.get_target_row(self.bot_id):
                self.was_last_random = False
                return ('move', move), 10000.0
        if epsilon > 0.0 and random.random() < epsilon:
            raw_actions = self.generate_actions(state, self.bot_id)
            if raw_actions:
                chosen_action = random.choice(raw_actions)
                next_state = self.apply_action(state, self.bot_id, chosen_action, is_random=True)
                self.was_last_random = True
                return chosen_action, self.evaluate(next_state, self.bot_id)

        self.was_last_random = False
        best_move, max_eval = self.minimax(state, self.max_depth, float('-inf'), float('inf'), True)
        if best_move is None:
            raw_actions = self.generate_actions(state, self.bot_id)
            if raw_actions:
                ranked = self.rank_and_prune(state, raw_actions, self.bot_id, top_k=1)
                if ranked:
                    return ranked[0], self.evaluate(state, self.bot_id)
            return None, 0.0
        return best_move, max_eval

    def get_move(self, state):
        action, _ = self.get_move_and_eval(state, epsilon=0.0)
        return action

    def get_opponent(self, pid):
        return 1 if pid == 2 else 2

    def get_target_row(self, pid):
        return 8 if pid == 2 else 0

    def get_shortest_path(self, state, pid):
        pos = state[f'p{pid}_pos']
        distances = get_distance_field(state, pid)
        distance = distances[pos[1] * BOARD_SIZE + pos[0]]
        if distance == 999:
            return None
        path = [pos]
        while distance:
            pos = next(nxt for nxt in self.get_adjacent(state, *pos)
                       if distances[nxt[1] * BOARD_SIZE + nxt[0]] == distance - 1)
            path.append(pos)
            distance -= 1
        return path

    def get_adjacent(self, state, c, r):
        return get_adjacent_global(state, c, r)

    def get_valid_moves(self, state, pid):
        pos = state['p1_pos'] if pid == 1 else state['p2_pos']
        other_pos = state['p2_pos'] if pid == 1 else state['p1_pos']
        moves = self.get_adjacent(state, *pos)
        
        if other_pos in moves:
            moves.remove(other_pos)
            dc, dr = other_pos[0] - pos[0], other_pos[1] - pos[1]
            jump_straight = (other_pos[0] + dc, other_pos[1] + dr)
            other_adj = self.get_adjacent(state, *other_pos)
            
            if jump_straight in other_adj:
                moves.append(jump_straight)
            else:
                if dr != 0:
                    for side in [(other_pos[0]-1, other_pos[1]), (other_pos[0]+1, other_pos[1])]:
                        if side in other_adj: moves.append(side)
                elif dc != 0:
                    for side in [(other_pos[0], other_pos[1]-1), (other_pos[0], other_pos[1]+1)]:
                        if side in other_adj: moves.append(side)
        return moves

    def is_valid_wall(self, state, c, r, orient):
        if not (0 <= c <= 7 and 0 <= r <= 7): return False
        if orient not in ('H', 'V'): return False
        if orient == 'H':
            if (c, r) in state['walls_h'] or (c-1, r) in state['walls_h'] or (c+1, r) in state['walls_h']: return False
            if (c, r) in state['walls_v']: return False
        else:
            if (c, r) in state['walls_v'] or (c, r-1) in state['walls_v'] or (c, r+1) in state['walls_v']: return False
            if (c, r) in state['walls_h']: return False
        candidate = state.copy()
        key = 'walls_h' if orient == 'H' else 'walls_v'
        candidate[key] = state[key] | {(c, r)}
        return (get_shortest_path_dist(candidate, 1) < 999
                and get_shortest_path_dist(candidate, 2) < 999)

    def generate_actions(self, state, pid):
        actions = []
        valid_moves = self.get_valid_moves(state, pid)
        
        def get_path_dist(m):
            sim_state = state.copy()
            if pid == 1: sim_state['p1_pos'] = m
            else: sim_state['p2_pos'] = m
            return get_shortest_path_dist(sim_state, pid)

        valid_moves.sort(key=get_path_dist)
        for move in valid_moves: actions.append(('move', move))
            
        walls_left = state['p1_walls'] if pid == 1 else state['p2_walls']
        if walls_left <= 0: return actions

        opponent_id = self.get_opponent(pid)
        opp_path = self.get_shortest_path(state, opponent_id)
        my_path = self.get_shortest_path(state, pid)
        if not opp_path or not my_path: return actions

        base_my_dist, base_opp_dist = len(my_path) - 1, len(opp_path) - 1
        candidate_walls = set()
        
        for p in (opp_path, my_path):
            for i in range(len(p) - 1):
                c1, r1 = p[i]
                c2, r2 = p[i+1]
                if c1 == c2: 
                    r_min = min(r1, r2)
                    candidate_walls.update([(c1, r_min, 'H'), (c1-1, r_min, 'H')])
                else:
                    c_min = min(c1, c2)
                    candidate_walls.update([(c_min, r1, 'V'), (c_min, r1-1, 'V')])

        for placed_walls in (state['walls_h'], state['walls_v']):
            for c, r in placed_walls:
                for dc in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        nc, nr = c + dc, r + dr
                        if 0 <= nc <= 7 and 0 <= nr <= 7:
                            candidate_walls.add((nc, nr, 'H'))
                            candidate_walls.add((nc, nr, 'V'))
                
        scored_walls = []
        for c, r, orient in sorted(candidate_walls):
            if self.is_valid_wall(state, c, r, orient):
                candidate = state.copy()
                key = 'walls_h' if orient == 'H' else 'walls_v'
                candidate[key] = state[key] | {(c, r)}
                delay = ((get_shortest_path_dist(candidate, opponent_id) - base_opp_dist)
                         - (get_shortest_path_dist(candidate, pid) - base_my_dist))
                scored_walls.append((delay, ('wall', (c, r, orient))))

        scored_walls.sort(key=lambda x: x[0], reverse=True)
        quota = int(4 + (1.0 - self.heuristic_weight) * 20)
        
        for i, (delay, wall_action) in enumerate(scored_walls):
            if i < quota:
                if delay > 0 or random.random() > self.heuristic_weight:
                    actions.append(wall_action)
            else: break
        return actions

    def rank_and_prune(self, state, actions, pid, top_k=TOP_MOVE_POOL_SIZE):
        if not actions: return []
        actions_copy = list(actions)
        random.shuffle(actions_copy)

        next_states = [self.apply_action(state, pid, action) for action in actions_copy]
        values = self.evaluate_batch(next_states, pid)
        opponent = self.get_opponent(pid)
        for index, candidate in enumerate(next_states):
            if candidate[f'p{pid}_pos'][1] == self.get_target_row(pid):
                continue
            if any(move[1] == self.get_target_row(opponent)
                   for move in self.get_valid_moves(candidate, opponent)):
                # Never prune a necessary defense in favor of a position that
                # loses immediately, even if an immature network prefers it.
                values[index] = -10000.0
        scored_actions = list(zip(values, actions_copy))

        scored_actions.sort(key=lambda x: x[0], reverse=True)
        return [act for _, act in scored_actions[:top_k]]

    def _oscillation_penalty(self, state, pid):
        """Soft penalty for landing on a square this player occupied recently.

        Two genuinely equal-length paths (e.g. a fork around a wall) produce
        an *exact* tie in evaluate()'s distance/progress terms, so without a
        tiebreaker the bot can happily bounce between them forever -- the
        joint-state 'is_repeated' check almost never catches this because the
        opponent is also moving, so the full board rarely repeats exactly.
        Weighting by recency means a single necessary backtrack barely costs
        anything, while repeatedly ping-ponging over the same square or two
        gets progressively more expensive and reliably breaks the tie.
        """
        my_pos = state['p1_pos'] if pid == 1 else state['p2_pos']
        recent = state.get('p1_recent', ()) if pid == 1 else state.get('p2_recent', ())
        if not recent:
            return 0.0
        penalty = 0.0
        n = len(recent)
        for i, pos in enumerate(recent):
            if pos == my_pos:
                recency_weight = (i + 1) / n  # most recent entries weigh close to 1.0
                penalty += self.OSCILLATION_PENALTY * recency_weight
        return penalty

    def evaluate(self, state, eval_pid):
        return self.evaluate_batch([state], eval_pid)[0]

    def evaluate_batch(self, states, eval_pid):
        """Score candidate moves in one neural forward pass, in turn perspective."""
        scores = []
        neural_requests = []
        missing = {}
        for index, state in enumerate(states):
            if state['p1_pos'][1] == 0 or state['p2_pos'][1] == 8:
                winner = 1 if state['p1_pos'][1] == 0 else 2
                scores.append(10000.0 if eval_pid == winner else -10000.0)
                continue
            d1 = get_shortest_path_dist(state, 1)
            d2 = get_shortest_path_dist(state, 2)
            progress = (8 - state['p1_pos'][1]) - state['p2_pos'][1]
            heuristic = (d2 - d1) * 25.0 + progress * 5.0
            heuristic += (state['p1_walls'] - state['p2_walls']) * 2.0
            if eval_pid == 2:
                heuristic = -heuristic
            penalty = self._oscillation_penalty(state, eval_pid)
            if get_state_key(state) in state.get('history', set()):
                penalty += 8000.0
            if self.nn_model is None or not (state['p1_walls'] or state['p2_walls']):
                scores.append(heuristic - penalty)
                continue

            # A useful prior prevents a fresh random network from generating
            # aimless practice games; the learned value still supplies 65%.
            scores.append(0.35 * heuristic - penalty)
            active_pid = state['turn']
            key = (get_state_key(state), state['p1_walls'], state['p2_walls'], active_pid)
            sign = 1.0 if active_pid == eval_pid else -1.0
            neural_requests.append((index, key, sign))
            if key not in self._value_cache and key not in missing:
                missing[key] = encode_state(state, active_pid)

        if missing:
            batch = make_state_batch(list(missing.values()), model_device(self.nn_model))
            with torch.inference_mode():
                output = self.nn_model(batch)
            if tuple(output.shape) != (len(missing), 1) or not torch.isfinite(output).all():
                raise ValueError("bot network must return finite [B, 1] values")
            self._value_cache.update(zip(missing, output[:, 0].cpu().tolist()))
        for index, key, sign in neural_requests:
            scores[index] += 65.0 * sign * self._value_cache[key]
        return scores

    def apply_action(self, state, pid, action, is_random=False):
        new_state = {
            'p1_pos': state['p1_pos'], 'p2_pos': state['p2_pos'],
            'p1_walls': state['p1_walls'], 'p2_walls': state['p2_walls'],
            'walls_h': state['walls_h'].copy(), 'walls_v': state['walls_v'].copy(),
            'turn': self.get_opponent(pid),
            'turns_played': state.get('turns_played', 0) + 1,
            'history': state.get('history', set()).copy(),
            'p1_recent': state.get('p1_recent', ()),
            'p2_recent': state.get('p2_recent', ()),
            'random_walls': state.get('random_walls', set()).copy(),
            'p1_last_random': state.get('p1_last_random', False),
            'p2_last_random': state.get('p2_last_random', False)
        }
        new_state['history'].add(get_state_key(state))

        if action[0] == 'move':
            old_pos = state['p1_pos'] if pid == 1 else state['p2_pos']
            new_pos = action[1]
            if pid == 1:
                new_state['p1_pos'] = new_pos
                new_state['p1_recent'] = (state.get('p1_recent', ()) + (old_pos,))[-self.RECENT_MEMORY_LEN:]
                new_state['p1_last_random'] = is_random
            else:
                new_state['p2_pos'] = new_pos
                new_state['p2_recent'] = (state.get('p2_recent', ()) + (old_pos,))[-self.RECENT_MEMORY_LEN:]
                new_state['p2_last_random'] = is_random
        else:
            c, r, orient = action[1]
            if orient == 'H': new_state['walls_h'].add((c, r))
            else: new_state['walls_v'].add((c, r))
            if pid == 1:
                new_state['p1_walls'] -= 1
                new_state['p1_last_random'] = False
            else:
                new_state['p2_walls'] -= 1
                new_state['p2_last_random'] = False

            if is_random:
                new_state['random_walls'].add((c, r, orient))

        return new_state

    def minimax(self, state, depth, alpha, beta, maximizing):
        p1_win = (state['p1_pos'][1] == 0)
        p2_win = (state['p2_pos'][1] == 8)
        
        if p1_win or p2_win:
            if (p1_win and self.bot_id == 1) or (p2_win and self.bot_id == 2):
                return None, 10000 + depth
            else:
                return None, -10000 - depth

        if depth == 0: return None, self.evaluate(state, self.bot_id)

        pid = self.bot_id if maximizing else self.get_opponent(self.bot_id)
        raw_actions = self.generate_actions(state, pid)
        if not raw_actions: return None, self.evaluate(state, self.bot_id)
        
        # Usually search the strongest four candidates, but occasionally widen
        # the pool to ten so training explores credible alternatives.
        top_k = (
            self.WIDE_MOVE_POOL_SIZE
            if random.random() < self.WIDE_MOVE_SAMPLE_CHANCE
            else self.TOP_MOVE_POOL_SIZE
        )
        actions = self.rank_and_prune(state, raw_actions, pid, top_k=top_k)
        if not actions: return None, self.evaluate(state, self.bot_id)

        if maximizing:
            max_eval = float('-inf')
            best_action = None
            for action in actions:
                new_state = self.apply_action(state, pid, action)
                _, eval_score = self.minimax(new_state, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return best_action, max_eval
        else:
            min_eval = float('inf')
            best_action = None
            for action in actions:
                new_state = self.apply_action(state, pid, action)
                _, eval_score = self.minimax(new_state, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return best_action, min_eval

SCALE = 0.92

BG_COLOR, BOARD_BG = (242, 244, 247), (231, 234, 238)
SIDEBAR_BG, CARD_BG = (235, 238, 242), (255, 255, 255)
TERMINAL_BG, TERMINAL_TEXT = (26, 29, 36), (120, 220, 150)
CELL_COLOR, CELL_HOVER = (255, 255, 255), (246, 248, 252)
P1_COLOR, P2_COLOR = (238, 92, 105), (82, 143, 219)
WALL_COLOR, WALL_SHADOW = (82, 88, 98), (210, 214, 220)
VALID_MOVE_DOT = (205, 211, 220)
TEXT_COLOR, MUTED_TEXT = (78, 84, 94), (145, 151, 160)
ORANGE_COLOR = (255, 140, 0)

CELL_SIZE = int(55 * SCALE)
WALL_THICKNESS = int(14 * SCALE)
PADDING = int(25 * SCALE)
BOARD_PIXEL_SIZE = (BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * WALL_THICKNESS + (PADDING * 2))

SIDEBAR_WIDTH = int(380 * SCALE)
LEFT_SIDEBAR_WIDTH = int(370 * SCALE)  # Extended transparency UI space
WIDTH = BOARD_PIXEL_SIZE + SIDEBAR_WIDTH + LEFT_SIDEBAR_WIDTH
HEIGHT = BOARD_PIXEL_SIZE + int(120 * SCALE)

class Quoridor:
    def __init__(self, headless=False, train_depth=2, play_depth=4, train_updates=16):
        pygame.init()
        self.headless = headless
        self.train_depth = train_depth
        self.play_depth = play_depth
        self.is_fullscreen = False
        
        if not self.headless:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE | pygame.SRCALPHA)
            pygame.display.set_caption("Quoridor - QuoriNet AI Engine [Press F11 for Fullscreen]")
            self.virtual_screen = pygame.Surface((WIDTH, HEIGHT))
            
            self.font = pygame.font.SysFont("sans-serif", int(24 * SCALE), bold=True)
            self.small_font = pygame.font.SysFont("sans-serif", int(16 * SCALE), bold=True)
            self.tiny_font = pygame.font.SysFont("sans-serif", int(13 * SCALE), bold=False)
            self.title_font = pygame.font.SysFont("sans-serif", int(38 * SCALE), bold=True)
            self.console_font = pygame.font.SysFont("monospace", int(12 * SCALE), bold=False)
        else:
            self.screen = pygame.Surface((WIDTH, HEIGHT))
            self.virtual_screen = self.screen

        # Constants from Section 1
        self.RECENT_WINDOW_SIZE = 5
        # These two masses now divide only the non-latest support pool. The
        # latest teacher receives its own adaptive majority share below.
        self.RECENT_BUCKET_MASS = 0.60
        self.HISTORIC_BUCKET_MASS = 0.40
        self.RECENT_DECAY_RATIO = 2.0
        self.LOG_WINDOW_SIZE = 30
        self.MIN_SAMPLES_TRUSTED = 15
        self.GATE_MIN_GAMES = 30
        self.GATE_WIN_RATE_THRESHOLD = 0.57
        self.GATE_TREND_WINDOW_SIZE = 15
        self.FINAL_TEST_GAMES_PER_MODEL = 10
        self.ESCALATION_LOSS_THRESHOLD = 0.50
        self.ESCALATION_RECOVER_THRESHOLD = 0.65
        self.ESCALATION_STEP_MULTIPLIER = 1.5
        self.ESCALATION_MAX_MULTIPLIER = 8.0
        self.ESCALATION_STALL_LIMIT = 5
        self.ESCALATION_MAX_SHARE_OF_HISTORIC_BUCKET = 0.15

        # Gate-deficit curriculum. Every loaded teacher uses the same required
        # win rate; a uniform floor prevents total forgetting, and the remaining
        # mass follows whichever teacher is furthest behind that shared target.
        self.LATEST_TEACHER_LOG_WINDOW = 60
        self.CURRICULUM_UNIFORM_MASS = 0.25

        # Replay remains split between latest and historical experience, but its
        # batch ratio follows the same live gate deficits as opponent selection.
        self.REPLAY_LATEST_CAPACITY = 140000
        self.REPLAY_HISTORIC_CAPACITY = 60000
        self.REPLAY_WARMUP = 64
        self.REPLAY_BATCH_SIZE = 64
        self.TRAIN_UPDATES_PER_GAME = train_updates

        # Sparse terminal reward is propagated through every completed matchup and
        # blended with the stable teacher bootstrap already used by this model.
        self.OUTCOME_BACKFILL_MIX = 0.90
        self.OUTCOME_BACKFILL_DISCOUNT = 0.995

        # Explicit LR bounds. Legacy policy constants are retained for the
        # isolated diagnostic; production holds the loaded optimizer LR fixed.
        self.LR_MIN = 0.00001
        self.LR_MAX = 0.0008
        self.LR_TUNE_INTERVAL_LATEST_GAMES = 15
        self.LR_MIN_SAMPLES = 15
        self.LR_PLATEAU_EPS = 0.015
        self.LR_REGRESSION_EPS = 0.03
        self.LR_UP_FACTOR = 1.25
        self.LR_DOWN_FACTOR = 0.80
        self.LR_CONVERGE_FACTOR = 0.85
        self.GRAD_CLIP_NORM = 1.0

        # State vars
        self.student_net = SpatialQuoriNetV2(in_channels=STATE_CHANNELS)
        self.teacher_net = SpatialQuoriNetV2(in_channels=STATE_CHANNELS)
        self.teacher_net.load_state_dict(self.student_net.state_dict())
        self.student_net.eval()
        self.teacher_net.eval()
        self.optimizer = optim.Adam(self.student_net.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()

        self.games_played = 0
        self.teacher_gen = 1
        self.heuristic_weight = 1.0
        self.heuristic_lr_floor = self.optimizer.param_groups[0]['lr']

        # Kept for backward-compatible metadata/UI. Promotion no longer uses
        # this mixed-opponent history.
        self.student_eval_history = deque(maxlen=100)
        self.latest_teacher_history = deque(maxlen=self.LATEST_TEACHER_LOG_WINDOW)
        self.student_win_rate_str = "Pending live WR"
        self.student_id = 2

        self.p1_gen_str = "Gen #1"
        self.p2_gen_str = "Gen #1"

        self.latest_replay_buffer = ReplayBuffer(self.REPLAY_LATEST_CAPACITY)
        self.historic_replay_buffer = ReplayBuffer(self.REPLAY_HISTORIC_CAPACITY)
        self.episode_trajectory = []
        self.last_loss = 0.0
        self.optimizer_steps = 0
        self.replay_samples_latest = 0
        self.replay_samples_historic = 0
        self._save_requested = False

        self.match_history = deque(maxlen=6)
        self.opponent_pool = []
        self.opponent_weights = []
        self.opponent_support_base_weights = []

        # New State from Section 2
        self.checkpoint_log = {}
        self.escalation_state = {}
        self.wawr_history = []
        self.lr_last_reduced_at_gen = 0  # legacy metadata compatibility
        self.current_opponent_gen = 1
        self.current_opponent_is_latest = True
        self.curriculum_latest_credit = 1.0

        # Latest-teacher LR search state. Reset whenever a new teacher becomes
        # the target so metrics from different difficulty levels never mix.
        self.latest_teacher_games_seen = 0
        self.lr_training_games_seen = 0
        self.lr_metric_history = deque(maxlen=12)
        self.lr_last_tuned_at_training_game = 0
        self.lr_best_metric = -1.0
        self.lr_best_value = self.optimizer.param_groups[0]['lr']
        self.lr_search_direction = 0

        self.is_evaluating = False
        self.eval_vs_gen = 0
        self.eval_game_curr = 0
        self.eval_game_total = 0
        
        self.load_checkpoint_if_exists()
        self.update_opponent_pool()

        self.mode = "MENU"
        self.is_training_active = True
        self.toast_msg = ""
        self.toast_timer = 0
        
        self.rect_play, self.rect_train, self.rect_new_game, self.rect_menu, self.rect_train_toggle = [pygame.Rect(0, 0, 0, 0) for _ in range(5)]
        self.game_start_time = time.time()
        self.reset_game()
        if hasattr(self, '_restored_rng'):
            python_rng, torch_rng, cuda_rng = self._restored_rng
            random.setstate(python_rng)
            torch.set_rng_state(torch_rng)
            if torch.cuda.is_available() and cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
            del self._restored_rng


    # ==========================================
    # SECTION 4: OPPONENT POOL
    # ==========================================
    def update_opponent_pool(self):
        self.opponent_pool = []
        self.opponent_support_base_weights = []

        # Latest teacher is explicit and always present. Its actual probability
        # is computed dynamically from latest-teacher competence.
        self.opponent_pool.append((self.teacher_net, self.teacher_gen))
        self.opponent_support_base_weights.append(0.0)

        recent_start = max(1, self.teacher_gen - self.RECENT_WINDOW_SIZE + 1)
        recent_support_gens = list(range(recent_start, self.teacher_gen))
        recent_entries = []

        for i, gen in enumerate(recent_support_gens, start=1):
            net = self._load_net_for_gen(gen)
            if net is None:
                continue
            raw_weight = self.RECENT_DECAY_RATIO ** i
            recent_entries.append((net, gen, raw_weight))

        recent_raw_sum = sum(e[2] for e in recent_entries) or 1.0
        for net, gen, raw_weight in recent_entries:
            self.opponent_pool.append((net, gen))
            self.opponent_support_base_weights.append(
                raw_weight / recent_raw_sum * self.RECENT_BUCKET_MASS
            )

        historic_gens = [
            g for g in range(1, recent_start)
            if os.path.exists(f'quorinet_teacher_gen_{g}.pth')
        ]
        if historic_gens:
            hist_weights = {}
            for g in historic_gens:
                mult = self.escalation_state.get(g, {}).get("multiplier", 1.0)
                hist_weights[g] = 1.0 * mult

            hist_weights = self._apply_escalation_cap(
                hist_weights,
                self.HISTORIC_BUCKET_MASS,
                self.ESCALATION_MAX_SHARE_OF_HISTORIC_BUCKET
            )

            for g in historic_gens:
                net = self._load_net_for_gen(g)
                if net is None:
                    continue
                self.opponent_pool.append((net, g))
                self.opponent_support_base_weights.append(hist_weights[g])

        self._refresh_opponent_weights()
        latest_share = self._latest_teacher_share()
        print(
            f"Opponent pool updated. Latest target: Gen #{self.teacher_gen} "
            f"({latest_share*100:.1f}%) | Support models: {len(self.opponent_pool)-1} "
            f"| Total weight: {sum(self.opponent_weights):.4f}"
        )

    def _latest_teacher_win_rate(self):
        if not self.latest_teacher_history:
            return None
        return sum(self.latest_teacher_history) / len(self.latest_teacher_history)

    def _training_generations(self):
        """Return each loaded opponent generation exactly once."""
        return sorted({gen for _, gen in self.opponent_pool})

    def _required_win_rate(self, gen):
        """Return the same target win rate for every loaded teacher."""
        return self.GATE_WIN_RATE_THRESHOLD

    def _generation_win_rate(self, gen):
        log = self.checkpoint_log.get(gen)
        if not log:
            return None
        return sum(log) / len(log)

    def _passes_generation_gate(self, gen, win_rate):
        required = self._required_win_rate(gen)
        return win_rate >= required

    def _generation_gate_trend(self, gen):
        """Compare the closing live window with the equally sized prior window."""
        log = list(self.checkpoint_log.get(gen, ()))
        window = self.GATE_TREND_WINDOW_SIZE
        if len(log) < window * 2:
            return None
        prior_win_rate = sum(log[-2 * window:-window]) / window
        recent_win_rate = sum(log[-window:]) / window
        return prior_win_rate, recent_win_rate

    def _passes_training_generation_gate(self, gen):
        """Require both the 30-game WR target and a strictly rising finish."""
        log = self.checkpoint_log.get(gen)
        if not log or len(log) < self.GATE_MIN_GAMES:
            return False
        if not self._passes_generation_gate(gen, sum(log) / len(log)):
            return False
        trend = self._generation_gate_trend(gen)
        return trend is not None and trend[1] > trend[0]

    def _all_training_gates_met(self):
        generations = self._training_generations()
        if not generations:
            return False
        for gen in generations:
            if not self._passes_training_generation_gate(gen):
                return False
        return True

    def _format_training_win_rates(self):
        parts = []
        sampling_by_gen = {}
        for (_, pool_gen), weight in zip(
            self.opponent_pool, self.opponent_weights
        ):
            sampling_by_gen[pool_gen] = (
                sampling_by_gen.get(pool_gen, 0.0) + weight
            )
        for gen in self._training_generations():
            log = self.checkpoint_log.get(gen)
            sampling = sampling_by_gen.get(gen, 0.0) * 100
            if log:
                trend = self._generation_gate_trend(gen)
                if trend is None:
                    trend_text = (
                        f"Trend pending {len(log)}/{self.GATE_TREND_WINDOW_SIZE * 2}"
                    )
                else:
                    prior, recent = trend
                    direction = "RISING" if recent > prior else "NOT RISING"
                    trend_text = (
                        f"Trend {prior * 100:.1f}%->{recent * 100:.1f}% "
                        f"{direction}"
                    )
                parts.append(
                    f"Gen #{gen}: WR {sum(log) / len(log) * 100:.1f}% "
                    f"| {trend_text} | Sample {sampling:.1f}%"
                )
            else:
                parts.append(
                    f"Gen #{gen}: WR pending | Trend pending 0/"
                    f"{self.GATE_TREND_WINDOW_SIZE * 2} | Sample {sampling:.1f}%"
                )
        return " | ".join(parts) if parts else "No training opponents"

    def _weighted_pool_metric_and_trend(self):
        """Equal-weight live summary; descriptive only, never an LR signal.

        Require every teacher to be observed so missing matchups cannot silently
        change the population. Adaptive sampling weights are deliberately ignored.
        """
        generations = self._training_generations()
        histories = [list(self.checkpoint_log.get(gen, ())) for gen in generations]
        if not histories or any(not history for history in histories):
            return None
        metrics, trends = [], []
        for history in histories:
            metrics.append(sum(history) / len(history))
            half = min(15, len(history) // 2)
            trends.append((sum(history[-half:]) - sum(history[-2*half:-half])) / half
                          if half else 0.0)
        count = len(generations)
        return (sum(metrics) / count,
                sum(self._required_win_rate(gen) for gen in generations) / count,
                sum(trends) / count, sum(map(len, histories)))

    def _latest_teacher_share(self):
        if not self.opponent_pool:
            return 1.0
        if len(self.opponent_weights) != len(self.opponent_pool):
            return 1.0 / len(self.opponent_pool)
        return sum(
            weight for (_, gen), weight in zip(
                self.opponent_pool, self.opponent_weights
            ) if gen == self.teacher_gen
        )

    def _refresh_opponent_weights(self):
        if not self.opponent_pool:
            self.opponent_weights = []
            return

        count = len(self.opponent_pool)
        uniform_weight = self.CURRICULUM_UNIFORM_MASS / count
        deficit_mass = 1.0 - self.CURRICULUM_UNIFORM_MASS
        deficit_scores = []

        for _, gen in self.opponent_pool:
            log = self.checkpoint_log.get(gen)
            samples = len(log) if log else 0
            sample_deficit = max(
                0.0,
                1.0 - (samples / self.GATE_MIN_GAMES)
            )
            if samples:
                win_rate = sum(log) / samples
                required = self._required_win_rate(gen)
                performance_deficit = max(
                    0.0,
                    (required - win_rate) / max(required, 1e-9)
                )
            else:
                performance_deficit = 1.0
            deficit_scores.append(sample_deficit + performance_deficit)

        total_deficit = sum(deficit_scores)
        if total_deficit <= 0.0:
            self.opponent_weights = [1.0 / count] * count
            return

        self.opponent_weights = [
            uniform_weight + deficit_mass * (score / total_deficit)
            for score in deficit_scores
        ]

    def _select_training_opponent(self):
        self._refresh_opponent_weights()

        latest_idx = next(
            (i for i, (_, gen) in enumerate(self.opponent_pool)
             if gen == self.teacher_gen),
            None
        )
        support_indices = [
            i for i, (_, gen) in enumerate(self.opponent_pool)
            if gen != self.teacher_gen and self.opponent_weights[i] > 0.0
        ]

        if latest_idx is None:
            return self.teacher_net, self.teacher_gen

        if not support_indices:
            self.curriculum_latest_credit = 0.0
            return self.opponent_pool[latest_idx]

        # Deficit/credit scheduling realizes the live weights without forcing
        # pathological one-game alternation between latest and support models.
        latest_share = self.opponent_weights[latest_idx]
        self.curriculum_latest_credit += latest_share

        if self.curriculum_latest_credit >= 1.0:
            self.curriculum_latest_credit -= 1.0
            return self.opponent_pool[latest_idx]

        support_pool = [self.opponent_pool[i] for i in support_indices]
        support_weights = [self.opponent_weights[i] for i in support_indices]
        return random.choices(support_pool, weights=support_weights, k=1)[0]

    def _append_replay_sample(self, state_tensor, target, opponent_gen):
        state_tensor = validate_state_tensor(state_tensor, "replay state")
        target = float(target)
        if not math.isfinite(target):
            raise ValueError("replay target must be finite")
        # Replay is deliberately canonicalized on CPU. Batches are moved to the
        # student device in one place, which prevents mixed-device surprises.
        replay_state = state_tensor.detach().to(device="cpu", dtype=torch.float32).clone()
        item = (replay_state, float(max(-1.0, min(1.0, target))))
        if opponent_gen == self.teacher_gen:
            self.latest_replay_buffer.append(item)
        else:
            self.historic_replay_buffer.append(item)

    def _sample_replay_batch(self):
        total_samples = (
            len(self.latest_replay_buffer) + len(self.historic_replay_buffer)
        )
        if total_samples < self.REPLAY_WARMUP:
            return None

        # Match replay emphasis to the same live gate deficits that choose
        # opponents. A failing historical gate must receive real optimizer
        # capacity instead of being capped at 30% of every batch.
        latest_share = self._latest_teacher_share()
        desired_latest = int(round(self.REPLAY_BATCH_SIZE * latest_share))

        n_latest = min(len(self.latest_replay_buffer), desired_latest)
        n_historic = min(
            len(self.historic_replay_buffer),
            self.REPLAY_BATCH_SIZE - n_latest
        )

        remaining = self.REPLAY_BATCH_SIZE - n_latest - n_historic
        if remaining > 0:
            extra_latest = min(
                len(self.latest_replay_buffer) - n_latest,
                remaining
            )
            n_latest += extra_latest
            remaining -= extra_latest

        if remaining > 0:
            extra_historic = min(
                len(self.historic_replay_buffer) - n_historic,
                remaining
            )
            n_historic += extra_historic
            remaining -= extra_historic

        if n_latest + n_historic < self.REPLAY_BATCH_SIZE:
            return None

        batch = []
        if n_latest:
            batch.extend(random.sample(self.latest_replay_buffer, n_latest))
        if n_historic:
            batch.extend(random.sample(self.historic_replay_buffer, n_historic))
        random.shuffle(batch)
        self._last_replay_counts = (n_latest, n_historic)
        return batch

    def _optimize_student_from_replay(self):
        """Run exactly one checked optimizer step, or return False during warmup."""
        batch = self._sample_replay_batch()
        if batch is None:
            return False

        self.student_net.train()
        try:
            device = model_device(self.student_net)
            states = make_state_batch(
                [item[0] for item in batch],
                device=device,
                name="replay states"
            )
            targets = make_value_batch(
                [item[1] for item in batch],
                device=device,
                name="replay targets"
            )
            # Half the batch is reflected, keeping compute and memory fixed.
            reflect = torch.rand(states.shape[0], device=device) < 0.5
            states[reflect] = mirror_state_batch(states[reflect])
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.student_net(states)
            loss = checked_value_loss(
                self.loss_fn,
                predictions,
                targets,
                name="student replay MSE"
            )
            loss.backward()
            for name, parameter in self.student_net.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                    raise FloatingPointError(f"non-finite gradient in {name}")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student_net.parameters(),
                self.GRAD_CLIP_NORM
            )
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError("student gradient norm is NaN or infinite")
            self.optimizer.step()
            self.optimizer_steps = getattr(self, 'optimizer_steps', 0) + 1
            latest, historic = self._last_replay_counts
            self.replay_samples_latest += latest
            self.replay_samples_historic += historic
            self.last_loss = loss.item()
            return True
        finally:
            self.student_net.eval()

    def _roll_replay_buffers_for_new_teacher(self):
        # The old "latest" data is still useful for anti-forgetting, but after
        # promotion it must no longer occupy the protected current-target bucket.
        self.historic_replay_buffer.extend(self.latest_replay_buffer)
        self.latest_replay_buffer.clear()

    def _set_learning_rate(self, requested_lr, reason):
        if not math.isfinite(requested_lr) or requested_lr <= 0:
            raise ValueError('learning rate must be finite and positive')
        old_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(self.LR_MIN, min(self.LR_MAX, requested_lr))
        if abs(new_lr - old_lr) < 1e-12:
            return False

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(f"LR: {old_lr:.6f} -> {new_lr:.6f} | {reason}")
        return True

    def _reset_lr_search_for_new_teacher(self):
        self.latest_teacher_games_seen = 0
        self.lr_training_games_seen = 0
        self.lr_metric_history.clear()
        self.lr_last_tuned_at_training_game = 0
        self.lr_best_metric = -1.0
        self.lr_best_value = self.optimizer.param_groups[0]['lr']
        self.lr_search_direction = 0

    def _tune_learning_rate_from_pool(self, force=False):
        """Compatibility hook: training results no longer change optimizer LR.

        The former policy used noisy, changing training matchups as experiments.
        Keep the loaded optimizer LR until a controlled evaluation supports an
        explicit change. Adam's moment adaptation remains active.
        """
        return

    def _load_net_for_gen(self, gen):
        if gen == self.teacher_gen: return self.teacher_net
        path = f'quorinet_teacher_gen_{gen}.pth'
        if not os.path.exists(path): return None
        try:
            net = SpatialQuoriNetV2(in_channels=STATE_CHANNELS)
            net.load_state_dict(torch.load(path, weights_only=True))
            net.eval()
            return net
        except Exception as e:
            print(f"Failed to load pool model {gen}: {e}")
            return None

    def _apply_escalation_cap(self, weights: dict, total_mass: float, max_share: float) -> dict:
        raw_total = sum(weights.values()) or 1.0
        normed = {g: w / raw_total * total_mass for g, w in weights.items()}
        for _ in range(len(normed)):
            over = {g: w for g, w in normed.items() if w > total_mass * max_share}
            if not over: break
            for g in over: normed[g] = total_mass * max_share
            remaining_mass = total_mass - sum(normed[g] for g in over)
            remaining_gens = [g for g in normed if g not in over]
            remaining_raw_total = sum(normed[g] for g in remaining_gens) or 1.0
            for g in remaining_gens:
                normed[g] = normed[g] / remaining_raw_total * remaining_mass
        return normed

    # ==========================================
    # SECTION 5: ESCALATION STATE
    # ==========================================
    def update_escalation_state(self):
        flags = []
        for gen, log in self.checkpoint_log.items():
            if gen >= self.teacher_gen - self.RECENT_WINDOW_SIZE + 1: continue
            if len(log) < self.MIN_SAMPLES_TRUSTED: continue
            
            win_rate = sum(log) / len(log)
            state = self.escalation_state.setdefault(gen, {"multiplier": 1.0, "stall_count": 0})

            if win_rate < self.ESCALATION_LOSS_THRESHOLD:
                state["multiplier"] = min(state["multiplier"] * self.ESCALATION_STEP_MULTIPLIER, self.ESCALATION_MAX_MULTIPLIER)
                if state["multiplier"] >= self.ESCALATION_MAX_MULTIPLIER:
                    state["stall_count"] += 1
                else:
                    state["stall_count"] = 0
                if state["stall_count"] >= self.ESCALATION_STALL_LIMIT:
                    flags.append(gen)
            elif win_rate >= self.ESCALATION_RECOVER_THRESHOLD:
                state["multiplier"] = 1.0
                state["stall_count"] = 0

        for gen in flags:
            print(f"REVIEW_NEEDED: checkpoint gen #{gen} has not recovered above "
                  f"{self.ESCALATION_RECOVER_THRESHOLD*100:.0f}% after {self.ESCALATION_STALL_LIMIT} "
                  f"escalated intervals. Weight is capped — investigate encode_state() coverage.")

    def print_headless_metrics(self):
        if not self.headless: return
        eval_score = round(self.bot1.evaluate(self.state, self.state['turn']), 1)
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"\n--- MATCH #{self.games_played} TERMINAL METRICS ---")
        print(f"Red Player (P1):  {self.p1_gen_str}")
        print(f"Blue Player (P2): {self.p2_gen_str}")
        print(f"Latest Teacher:   Gen #{self.teacher_gen}")
        print(f"Wall filter:    {self.heuristic_weight:.3f} | LR policy: fixed")
        print(f"Optimizer steps: {self.optimizer_steps}")
        print(f"Studt Win Rate:  {self.student_win_rate_str}")
        print(f"Training WRs:    {self._format_training_win_rates()}")
        print(f"AI Board Eval:   {eval_score} | LR: {current_lr:.6f} | Loss: {self.last_loss:.5f}")
        print("----------------------------------------\n")

    def get_virtual_mouse_pos(self):
        mx, my = pygame.mouse.get_pos()
        sw, sh = self.screen.get_size()
        if sw <= 0 or sh <= 0: return mx, my
        return int(mx * (WIDTH / sw)), int(my * (HEIGHT / sh))

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.screen = pygame.display.set_mode((0, 0) if self.is_fullscreen else (WIDTH, HEIGHT), pygame.FULLSCREEN if self.is_fullscreen else pygame.RESIZABLE)

    def render_and_flip(self):
        if self.headless: return
        scaled_surface = pygame.transform.smoothscale(self.virtual_screen, self.screen.get_size())
        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

    def reset_game(self):
        self.state = {
            'p1_pos': (4, 8), 'p2_pos': (4, 0),
            'p1_walls': 10, 'p2_walls': 10,
            'walls_h': set(), 'walls_v': set(),
            'turn': 1, 'turns_played': 0, 'history': set(),
            'p1_recent': (), 'p2_recent': (),
            'random_walls': set(), 'p1_last_random': False, 'p2_last_random': False
        }
        self.wall_orientation = 'H'
        self.game_over = False
        self._result_recorded = False
        self._game_start_optimizer_steps = self.optimizer_steps
        self.winner = None
        self.winner_id = 0
        self.game_start_time = time.time()
        self.episode_trajectory = []

        if getattr(self, 'mode', 'MENU') == 'TRAIN':
            if hasattr(self, 'opponent_pool') and self.opponent_pool:
                pool_opponent, pool_gen = self._select_training_opponent()
                self.current_opponent_gen = pool_gen
            else:
                pool_opponent, pool_gen = self.teacher_net, self.teacher_gen
                self.current_opponent_gen = pool_gen

            self.current_opponent_is_latest = (pool_gen == self.teacher_gen)

            if self.games_played % 2 == 0:
                self.bot1 = QuoridorBot(bot_id=1, max_depth=self.train_depth, nn_model=pool_opponent, heuristic_weight=self.heuristic_weight)
                self.bot2 = QuoridorBot(bot_id=2, max_depth=self.train_depth, nn_model=self.student_net, heuristic_weight=self.heuristic_weight)
                self.student_id = 2
                self.p1_gen_str, self.p2_gen_str = f"Gen #{pool_gen} (Teacher)", f"Gen #{self.teacher_gen + 1} (Student)"
            else:
                self.bot1 = QuoridorBot(bot_id=1, max_depth=self.train_depth, nn_model=self.student_net, heuristic_weight=self.heuristic_weight)
                self.bot2 = QuoridorBot(bot_id=2, max_depth=self.train_depth, nn_model=pool_opponent, heuristic_weight=self.heuristic_weight)
                self.student_id = 1
                self.p1_gen_str, self.p2_gen_str = f"Gen #{self.teacher_gen + 1} (Student)", f"Gen #{pool_gen} (Teacher)"
        else:
            self.bot1 = QuoridorBot(bot_id=1, max_depth=self.play_depth, nn_model=self.teacher_net, heuristic_weight=self.heuristic_weight)
            self.bot2 = QuoridorBot(bot_id=2, max_depth=self.play_depth, nn_model=self.teacher_net, heuristic_weight=self.heuristic_weight)
            self.p1_gen_str, self.p2_gen_str = "Human", f"Gen #{self.teacher_gen} (Teacher)"

        self.p1_draw_x, self.p1_draw_y = self._get_center_pixels(self.state['p1_pos'])
        self.p2_draw_x, self.p2_draw_y = self._get_center_pixels(self.state['p2_pos'])

    def _load_network_checkpoint(self, path, destination, label):
        """Load strictly through a temporary model so a bad file cannot half-load."""
        candidate = SpatialQuoriNetV2(in_channels=STATE_CHANNELS)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        candidate.load_state_dict(state_dict, strict=True)
        validate_model_contract(candidate, f"{label} checkpoint")
        destination.load_state_dict(candidate.state_dict(), strict=True)
        validate_model_contract(destination, label)

    def load_checkpoint_if_exists(self):
        # One atomic bundle is authoritative over all legacy sidecar files.
        bundle = None
        if os.path.exists('quorinet_training_state.pth'):
            bundle = torch.load('quorinet_training_state.pth', map_location='cpu',
                                weights_only=True)
            if bundle.get('format_version') != 1:
                raise ValueError('unsupported training checkpoint format')
            self.teacher_net.load_state_dict(bundle['teacher'], strict=True)
            self.student_net.load_state_dict(bundle['student'], strict=True)
            self.optimizer.load_state_dict(bundle['optimizer'])
            self.latest_replay_buffer.extend(bundle['latest_replay'])
            self.historic_replay_buffer.extend(bundle['historic_replay'])
            self.last_loss = bundle['last_loss']
            self._restored_rng = (bundle['python_rng'], bundle['torch_rng'], bundle.get('cuda_rng'))
        elif os.path.exists('quorinet_meta.json'):
            print('Legacy checkpoint: model/Adam files are authoritative; '
                  'JSON counters may be newer and replay is unavailable.')
        teacher_loaded = False
        if bundle is None and os.path.exists('quorinet_teacher_weights.pth'):
            try:
                self._load_network_checkpoint(
                    'quorinet_teacher_weights.pth',
                    self.teacher_net,
                    'teacher network'
                )
                teacher_loaded = True
            except Exception as exc:
                raise RuntimeError(f"Teacher checkpoint rejected: {exc}") from exc

        if bundle is None and os.path.exists('quorinet_student_weights.pth'):
            try:
                self._load_network_checkpoint(
                    'quorinet_student_weights.pth',
                    self.student_net,
                    'student network'
                )
            except Exception as exc:
                raise RuntimeError(f"Student checkpoint rejected: {exc}") from exc
        elif teacher_loaded:
            self.student_net.load_state_dict(self.teacher_net.state_dict(), strict=True)

        if bundle is None and os.path.exists('quorinet_optimizer.pth'):
            try:
                candidate_optimizer = optim.Adam(
                    self.student_net.parameters(),
                    lr=self.optimizer.param_groups[0]['lr']
                )
                candidate_optimizer.load_state_dict(
                    torch.load(
                        'quorinet_optimizer.pth',
                        map_location=model_device(self.student_net),
                        weights_only=True
                    )
                )
                validate_optimizer_state_shapes(candidate_optimizer)
                self.optimizer = candidate_optimizer
            except Exception as exc:
                raise RuntimeError(f"Optimizer checkpoint rejected: {exc}") from exc

        validate_model_contract(self.teacher_net, "teacher network")
        validate_model_contract(self.student_net, "student network")
        validate_optimizer_state_shapes(self.optimizer)
        print(
            f"Neural tensor contract verified: state {STATE_SHAPE}, "
            f"batch [B, {STATE_CHANNELS}, {BOARD_SIZE}, {BOARD_SIZE}], "
            f"value [B, {VALUE_WIDTH}]."
        )

        self.heuristic_weight = 1.0
        if bundle is not None or os.path.exists('quorinet_meta.json'):
            try:
                source = (io.StringIO(json.dumps(bundle['meta'])) if bundle is not None
                          else open('quorinet_meta.json', 'r'))
                with source as f:
                    meta = json.load(f)
                    self.games_played = meta.get('games_played', 0)
                    self.teacher_gen = meta.get('teacher_gen', 1)
                    self.heuristic_weight = meta.get('heuristic_weight', 1.0)

                    eval_hist = meta.get('student_eval_history', [])
                    self.student_eval_history = deque(eval_hist, maxlen=100)

                    self.checkpoint_log = {
                        int(k): deque(v, maxlen=self.LOG_WINDOW_SIZE)
                        for k, v in meta.get('checkpoint_log', {}).items()
                    }
                    self.escalation_state = {
                        int(k): v for k, v in meta.get('escalation_state', {}).items()
                    }
                    self.wawr_history = meta.get('wawr_history', [])
                    self.lr_last_reduced_at_gen = meta.get('lr_last_reduced_at_gen', 0)

                    # New metadata is optional so every old checkpoint remains
                    # loadable. If absent, seed the latest-only metric from the
                    # already generation-tagged checkpoint log.
                    latest_hist = meta.get('latest_teacher_history')
                    if latest_hist is None:
                        latest_hist = list(self.checkpoint_log.get(self.teacher_gen, []))
                    self.latest_teacher_history = deque(
                        latest_hist,
                        maxlen=self.LATEST_TEACHER_LOG_WINDOW
                    )

                    self.latest_teacher_games_seen = meta.get(
                        'latest_teacher_games_seen',
                        len(self.latest_teacher_history)
                    )
                    self.lr_training_games_seen = meta.get(
                        'lr_training_games_seen',
                        self.latest_teacher_games_seen
                    )
                    self.lr_metric_history = deque(
                        meta.get('lr_metric_history', []),
                        maxlen=12
                    )
                    self.lr_last_tuned_at_training_game = meta.get(
                        'lr_last_tuned_at_training_game',
                        meta.get('lr_last_tuned_at_latest_game', 0)
                    )
                    self.lr_best_metric = meta.get('lr_best_metric', -1.0)
                    self.lr_best_value = meta.get(
                        'lr_best_value',
                        self.optimizer.param_groups[0]['lr']
                    )
                    self.lr_search_direction = meta.get('lr_search_direction', 0)
                    self.curriculum_latest_credit = meta.get(
                        'curriculum_latest_credit',
                        1.0
                    )

                    # The optimizer is authoritative: legacy per-game JSON may
                    # be newer than its model/Adam snapshot. Never replay its LR.
                    self.heuristic_lr_floor = meta.get(
                        'heuristic_lr_floor', self.optimizer.param_groups[0]['lr'])
                    self.optimizer_steps = meta.get('optimizer_steps', 0)
                    self.replay_samples_latest = meta.get('replay_samples_latest', 0)
                    self.replay_samples_historic = meta.get('replay_samples_historic', 0)

                    latest_log = self.checkpoint_log.get(self.teacher_gen)
                    if latest_log:
                        wins = sum(latest_log)
                        games = len(latest_log)
                        self.student_win_rate_str = (
                            f"{wins / games * 100:.1f}% Latest ({wins}/{games})"
                        )
                    elif self.student_eval_history:
                        wins = sum(self.student_eval_history)
                        games = len(self.student_eval_history)
                        self.student_win_rate_str = f"{wins/games * 100:.1f}% ({wins}/{games})"
            except Exception as e:
                raise RuntimeError(f"Metadata checkpoint rejected: {e}") from e

    def _checkpoint_metadata(self):
        return {
                'games_played': self.games_played,
                'teacher_gen': self.teacher_gen,
                'heuristic_weight': self.heuristic_weight,
                'student_eval_history': list(self.student_eval_history),
                'latest_teacher_history': list(self.latest_teacher_history),
                'checkpoint_log': {k: list(v) for k, v in self.checkpoint_log.items()},
                'escalation_state': self.escalation_state,
                'wawr_history': self.wawr_history,
                'lr_last_reduced_at_gen': self.lr_last_reduced_at_gen,
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'latest_teacher_games_seen': self.latest_teacher_games_seen,
                'lr_training_games_seen': self.lr_training_games_seen,
                'lr_metric_history': list(self.lr_metric_history),
                'lr_last_tuned_at_training_game': self.lr_last_tuned_at_training_game,
                'lr_best_metric': self.lr_best_metric,
                'lr_best_value': self.lr_best_value,
                'lr_search_direction': self.lr_search_direction,
                'curriculum_latest_credit': self.curriculum_latest_credit,
                'heuristic_lr_floor': self.heuristic_lr_floor,
                'optimizer_steps': self.optimizer_steps,
                'replay_samples_latest': self.replay_samples_latest,
                'replay_samples_historic': self.replay_samples_historic
            }

    def save_meta(self):
        # Observational sidecar only; restoration uses the bundle's metadata.
        temporary = 'quorinet_meta.json.tmp'
        with open(temporary, 'w') as f:
            json.dump(self._checkpoint_metadata(), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, 'quorinet_meta.json')

    def save_checkpoint(self, manual=False):
        # A training episode owns deferred targets and updates. Save only after
        # it commits; the input thread requests a save instead of racing Adam.
        if not self.game_over and (getattr(self, 'mode', None) == 'TRAIN'
                                   or self.episode_trajectory):
            self._save_requested = True
            print('Checkpoint requested; saving after this game finishes.')
            return
        validate_model_contract(self.teacher_net, "teacher network before save")
        validate_model_contract(self.student_net, "student network before save")
        validate_optimizer_state_shapes(self.optimizer, "optimizer before save")
        bundle = {
            'format_version': 1,
            'teacher': self.teacher_net.state_dict(),
            'student': self.student_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'meta': self._checkpoint_metadata(),
            'latest_replay': list(self.latest_replay_buffer),
            'historic_replay': list(self.historic_replay_buffer),
            'last_loss': self.last_loss,
            'python_rng': random.getstate(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        temporary = 'quorinet_training_state.pth.tmp'
        with open(temporary, 'wb') as f:
            torch.save(bundle, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, 'quorinet_training_state.pth')
        self._save_requested = False
        # Compatibility exports. Interrupted exports cannot supersede the bundle.
        torch.save(self.teacher_net.state_dict(), f'quorinet_teacher_gen_{self.teacher_gen}.pth')
        torch.save(self.teacher_net.state_dict(), 'quorinet_teacher_weights.pth')
        torch.save(self.student_net.state_dict(), 'quorinet_student_weights.pth')
        torch.save(self.optimizer.state_dict(), 'quorinet_optimizer.pth')
        self.save_meta()
        print(f"[{'Manual' if manual else 'Auto'} Saved] Gen #{self.teacher_gen} saved.")
        self.toast_msg, self.toast_timer = f"Saved Checkpoint! (Gen #{self.teacher_gen})", 120

    def play_eval_game(self, bot1, bot2, game_curr, game_total, vs_gen):
        eval_state = {
            'p1_pos': (4, 8), 'p2_pos': (4, 0),
            'p1_walls': 10, 'p2_walls': 10,
            'walls_h': set(), 'walls_v': set(),
            'turn': 1, 'turns_played': 0, 'history': set(),
            'p1_recent': (), 'p2_recent': (),
            'random_walls': set(), 'p1_last_random': False, 'p2_last_random': False
        }
        self.eval_game_curr, self.eval_game_total, self.eval_vs_gen = game_curr, game_total, vs_gen
        saved_state = self.state
        self.state = eval_state
        result = 0
        try:
            for _ in range(200):
                if not self.headless:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                    self.p1_draw_x, self.p1_draw_y = self._get_center_pixels(eval_state['p1_pos'])
                    self.p2_draw_x, self.p2_draw_y = self._get_center_pixels(eval_state['p2_pos'])
                    self.draw_board((-1, -1))
                    pygame.time.delay(10)

                if eval_state['p1_pos'][1] == 0: result = 1; break
                if eval_state['p2_pos'][1] == 8: result = 2; break
                
                pid = eval_state['turn']
                active_bot = bot1 if pid == 1 else bot2
                action, _ = active_bot.get_move_and_eval(eval_state, epsilon=0.0)
                if not action:
                    result = 2 if pid == 1 else 1
                    break
                eval_state = active_bot.apply_action(eval_state, pid, action, is_random=getattr(active_bot, 'was_last_random', False))
                self.state = eval_state
        finally:
            self.state = saved_state
        return result

    # ==========================================
    # SECTION 6: FINAL NON-TRAINING TEST
    # ==========================================
    def run_final_promotion_test(self):
        opponents = []
        seen = set()
        for net, gen in self.opponent_pool:
            if gen not in seen:
                opponents.append((net, gen))
                seen.add(gen)

        if not opponents:
            return False

        games_per_model = self.FINAL_TEST_GAMES_PER_MODEL
        total_games = len(opponents) * games_per_model
        completed_games = 0
        results = {}
        print(
            f"\n--- FINAL NON-TRAINING TEST: Candidate Gen "
            f"#{self.teacher_gen + 1} ---"
        )
        print(
            f"10 frozen games against each of {len(opponents)} training models."
        )

        self.is_evaluating = True
        try:
            for past_net, gen in opponents:
                gen_results = []
                for game_index in range(games_per_model):
                    completed_games += 1
                    if game_index % 2 == 0:
                        eval_bot1 = QuoridorBot(
                            bot_id=1, max_depth=2, nn_model=self.student_net,
                            heuristic_weight=self.heuristic_weight
                        )
                        eval_bot2 = QuoridorBot(
                            bot_id=2, max_depth=2, nn_model=past_net,
                            heuristic_weight=self.heuristic_weight
                        )
                        result = self.play_eval_game(
                            eval_bot1, eval_bot2, completed_games, total_games, gen
                        )
                        student_won = 1 if result == 1 else 0
                    else:
                        eval_bot1 = QuoridorBot(
                            bot_id=1, max_depth=2, nn_model=past_net,
                            heuristic_weight=self.heuristic_weight
                        )
                        eval_bot2 = QuoridorBot(
                            bot_id=2, max_depth=2, nn_model=self.student_net,
                            heuristic_weight=self.heuristic_weight
                        )
                        result = self.play_eval_game(
                            eval_bot1, eval_bot2, completed_games, total_games, gen
                        )
                        student_won = 1 if result == 2 else 0

                    gen_results.append(student_won)
                    print(
                        f"Final test {completed_games}/{total_games} vs Gen #{gen}: "
                        f"{'WIN' if student_won else 'LOSS/DRAW'}"
                    )
                results[gen] = gen_results
        finally:
            self.is_evaluating = False

        all_passed = True
        passed_by_gen = {}
        for _, gen in opponents:
            gen_results = results[gen]
            win_rate = sum(gen_results) / len(gen_results)
            passed = self._passes_generation_gate(gen, win_rate)
            passed_by_gen[gen] = passed
            all_passed = all_passed and passed
            print(
                f"Final vs Gen #{gen}: {sum(gen_results)}/{len(gen_results)} "
                f"({win_rate * 100:.1f}%) | Required "
                f">= {self._required_win_rate(gen) * 100:.1f}% "
                f"[{'PASS' if passed else 'FAIL'}]"
            )

        if not all_passed:
            # A failed generation starts its qualification window from the
            # genuine frozen-test results. This guarantees more training before
            # another final test instead of retriggering one game later on a
            # still-high pre-test rolling window.
            for gen, gen_results in results.items():
                if passed_by_gen[gen]:
                    log = self.checkpoint_log.setdefault(
                        gen, deque(maxlen=self.LOG_WINDOW_SIZE)
                    )
                    log.extend(gen_results)
                else:
                    self.checkpoint_log[gen] = deque(
                        gen_results,
                        maxlen=self.LOG_WINDOW_SIZE
                    )
                if gen == self.teacher_gen:
                    self.latest_teacher_history.extend(gen_results)
            latest_log = self.checkpoint_log.get(self.teacher_gen)
            if latest_log:
                latest_wins = sum(latest_log)
                latest_games = len(latest_log)
                self.student_win_rate_str = (
                    f"{latest_wins / latest_games * 100:.1f}% Latest "
                    f"({latest_wins}/{latest_games})"
                )
            self._refresh_opponent_weights()
            print("Final test failed. Candidate returns to training unchanged.")

        return all_passed

    def train_step(self, state_tensor, turn_pid, next_state_tensor, next_turn_pid, done, winner_id):
        validate_state_tensor(state_tensor, "train_step state")
        validate_state_tensor(next_state_tensor, "train_step next_state")
        if turn_pid not in (1, 2) or next_turn_pid not in (1, 2):
            raise ValueError(
                f"turn_pid and next_turn_pid must be 1 or 2, got "
                f"{turn_pid} and {next_turn_pid}"
            )
        if winner_id not in (0, 1, 2):
            raise ValueError(f"winner_id must be 0, 1, or 2, got {winner_id}")

        # Keep the game policy fixed while collecting a trajectory. Label each
        # position once, when its outcome is known, rather than storing both a
        # speculative bootstrap and a contradictory outcome for the same move.
        self.episode_trajectory.append((
            state_tensor.detach().cpu().clone(), turn_pid,
            next_state_tensor.detach().cpu().clone(), next_turn_pid,
        ))
        if not done:
            return

        trajectory = self.episode_trajectory
        count = len(trajectory)
        bootstrap_values = [0.0] * count
        if winner_id and count > 1:
            next_states = make_state_batch(
                [entry[2] for entry in trajectory[:-1]],
                model_device(self.teacher_net), name="episode bootstrap states"
            )
            with torch.inference_mode():
                values = self.teacher_net(next_states)
            if tuple(values.shape) != (count - 1, 1) or not torch.isfinite(values).all():
                raise ValueError("teacher bootstrap must return finite [B, 1] values")
            bootstrap_values[:-1] = values[:, 0].cpu().tolist()

        for index, (episode_state, pid, _, next_pid) in enumerate(trajectory):
            outcome = 0.0 if winner_id == 0 else (1.0 if pid == winner_id else -1.0)
            horizon = count - 1 - index
            if horizon == 0 or winner_id == 0:
                target = outcome
            else:
                bootstrap = 0.99 * bootstrap_values[index] * (1 if pid == next_pid else -1)
                target = (self.OUTCOME_BACKFILL_MIX * outcome
                          * self.OUTCOME_BACKFILL_DISCOUNT ** horizon
                          + (1.0 - self.OUTCOME_BACKFILL_MIX) * bootstrap)
            self._append_replay_sample(episode_state, target, self.current_opponent_gen)
        self.episode_trajectory.clear()

        # Reuse each completed game, without spending hundreds of updates on a
        # single long match. Wins, losses and draws against every generation help.
        updates = min(self.TRAIN_UPDATES_PER_GAME, max(4, math.ceil(count / 4)))
        for _ in range(updates):
            if not self._optimize_student_from_replay():
                break

    def log_match_result(self, winner_id):
        duration = round(time.time() - self.game_start_time, 1)
        if winner_id == 0: winner_str = "Draw"
        elif getattr(self, 'mode', 'TRAIN') == "TRAIN": winner_str = "Student" if winner_id == self.student_id else "Teacher"
        else: winner_str = "Human" if winner_id == 1 else "Teacher"
            
        self.match_history.appendleft({
            'game': self.games_played + 1, 'winner': winner_str, 'winner_id': winner_id,
            'turns': self.state['turns_played'], 'time': duration,
            'p1_w': self.state['p1_walls'], 'p2_w': self.state['p2_walls']
        })

    def _get_center_pixels(self, pos):
        offset_x, offset_y = PADDING + LEFT_SIDEBAR_WIDTH, int(85 * SCALE) + PADDING
        x = offset_x + pos[0] * (CELL_SIZE + WALL_THICKNESS) + CELL_SIZE / 2
        y = offset_y + pos[1] * (CELL_SIZE + WALL_THICKNESS) + CELL_SIZE / 2
        return x, y

    def draw_menu(self):
        if self.headless: return
        self.virtual_screen.fill(BG_COLOR)
        title = self.title_font.render("QuoriNet AI Engine", True, TEXT_COLOR)
        self.virtual_screen.blit(title, (WIDTH // 2 - title.get_width() // 2, int(100 * SCALE)))

        btn_w, btn_h = int(240 * SCALE), int(60 * SCALE)
        self.rect_play = pygame.Rect(WIDTH // 2 - btn_w // 2, int(240 * SCALE), btn_w, btn_h)
        pygame.draw.rect(self.virtual_screen, P1_COLOR, self.rect_play, border_radius=int(10 * SCALE))
        play_txt = self.font.render("PLAY WITH BOT", True, (255, 255, 255))
        self.virtual_screen.blit(play_txt, (self.rect_play.centerx - play_txt.get_width() // 2, self.rect_play.centery - play_txt.get_height() // 2))

        self.rect_train = pygame.Rect(WIDTH // 2 - btn_w // 2, int(330 * SCALE), btn_w, btn_h)
        pygame.draw.rect(self.virtual_screen, P2_COLOR, self.rect_train, border_radius=int(10 * SCALE))
        train_txt = self.font.render("TRAIN BOT (WATCH)", True, (255, 255, 255))
        self.virtual_screen.blit(train_txt, (self.rect_train.centerx - train_txt.get_width() // 2, self.rect_train.centery - train_txt.get_height() // 2))

        hint_txt = self.tiny_font.render("[Press F11 or F to toggle Fullscreen]", True, MUTED_TEXT)
        self.virtual_screen.blit(hint_txt, (WIDTH // 2 - hint_txt.get_width() // 2, int(420 * SCALE)))
        self.render_and_flip()

    def draw_left_sidebar(self):
        sb_x = 0
        sb_w = LEFT_SIDEBAR_WIDTH
        sidebar_rect = pygame.Rect(sb_x, 0, sb_w, HEIGHT)
        pygame.draw.rect(self.virtual_screen, SIDEBAR_BG, sidebar_rect)
        pygame.draw.line(self.virtual_screen, (215, 220, 228), (sb_w, 0), (sb_w, HEIGHT), int(2 * SCALE))

        padding = int(12 * SCALE)
        card_w = sb_w - (padding * 2)
        card_rect = pygame.Rect(
            padding, int(15 * SCALE), card_w, HEIGHT - int(30 * SCALE)
        )
        pygame.draw.rect(
            self.virtual_screen, CARD_BG, card_rect,
            border_radius=int(10 * SCALE)
        )

        title = self.small_font.render(
            "CURRENT TRAINING WIN RATES", True, TEXT_COLOR
        )
        self.virtual_screen.blit(
            title, (card_rect.x + int(10 * SCALE), card_rect.y + int(8 * SCALE))
        )
        pygame.draw.line(
            self.virtual_screen, (235, 238, 242),
            (card_rect.x + int(10 * SCALE), card_rect.y + int(28 * SCALE)),
            (card_rect.right - int(10 * SCALE), card_rect.y + int(28 * SCALE))
        )

        headers = [
            ("Generation", 10),
            ("Win Rate", 120),
            ("Sampling", 235)
        ]
        for header, x_offset in headers:
            self.virtual_screen.blit(
                self.tiny_font.render(header, True, MUTED_TEXT),
                (
                    card_rect.x + int(x_offset * SCALE),
                    card_rect.y + int(34 * SCALE)
                )
            )

        sampling_by_gen = {}
        for (_, pool_gen), weight in zip(
            self.opponent_pool, self.opponent_weights
        ):
            sampling_by_gen[pool_gen] = (
                sampling_by_gen.get(pool_gen, 0.0) + weight
            )

        y_offset = card_rect.y + int(52 * SCALE)
        for gen in reversed(self._training_generations()):
            if y_offset > card_rect.bottom - int(20 * SCALE):
                break
            win_rate = self._generation_win_rate(gen)
            value = "Pending" if win_rate is None else f"{win_rate * 100:.1f}%"
            gen_color = P1_COLOR if gen == self.current_opponent_gen else TEXT_COLOR
            self.virtual_screen.blit(
                self.tiny_font.render(f"Gen #{gen}", True, gen_color),
                (card_rect.x + int(10 * SCALE), y_offset)
            )
            value_surf = self.tiny_font.render(value, True, TEXT_COLOR)
            self.virtual_screen.blit(
                value_surf,
                (card_rect.x + int(120 * SCALE), y_offset)
            )
            sampling_surf = self.tiny_font.render(
                f"{sampling_by_gen.get(gen, 0.0) * 100:.1f}%",
                True,
                TEXT_COLOR
            )
            self.virtual_screen.blit(
                sampling_surf,
                (card_rect.x + int(235 * SCALE), y_offset)
            )
            y_offset += int(18 * SCALE)

    def draw_right_sidebar(self):
        sb_x = BOARD_PIXEL_SIZE + LEFT_SIDEBAR_WIDTH
        sb_w = SIDEBAR_WIDTH
        
        sidebar_rect = pygame.Rect(sb_x, 0, sb_w, HEIGHT)
        pygame.draw.rect(self.virtual_screen, SIDEBAR_BG, sidebar_rect)
        pygame.draw.line(self.virtual_screen, (215, 220, 228), (sb_x, 0), (sb_x, HEIGHT), int(2 * SCALE))

        padding = int(12 * SCALE)
        card_x = sb_x + padding
        card_w = sb_w - (padding * 2)

        card1_rect = pygame.Rect(card_x, int(15 * SCALE), card_w, int(225 * SCALE))
        pygame.draw.rect(self.virtual_screen, CARD_BG, card1_rect, border_radius=int(10 * SCALE))
        
        m_title = self.small_font.render("LIVE METRICS & ANALYTICS", True, TEXT_COLOR)
        self.virtual_screen.blit(m_title, (card1_rect.x + int(10 * SCALE), card1_rect.y + int(8 * SCALE)))
        pygame.draw.line(self.virtual_screen, (235, 238, 242), (card1_rect.x + int(10 * SCALE), card1_rect.y + int(28 * SCALE)), (card1_rect.right - int(10 * SCALE), card1_rect.y + int(28 * SCALE)))

        eval_score = round(self.bot1.evaluate(self.state, self.state['turn']), 1)
        current_lr = self.optimizer.param_groups[0]['lr']
        final_test_str = f"Vs Gen #{self.eval_vs_gen} ({self.eval_game_curr}/{self.eval_game_total})" if self.is_evaluating else "Waiting on live gates"

        metrics = [
            ("Red Player (P1)", self.p1_gen_str),
            ("Blue Player (P2)", self.p2_gen_str),
            ("Latest Teacher Gen", f"Gen #{self.teacher_gen}"),
            ("Candidate Student Gen", f"Gen #{self.teacher_gen + 1}"),
            ("Wall filter", f"{self.heuristic_weight:.3f}"),
            ("Student Win Rate", self.student_win_rate_str),
            ("Final Test", final_test_str),
            ("AI Board Evaluation", f"{eval_score}"),
            ("Learning Rate", f"{current_lr:.6f}"),
            ("Student Training Loss", f"{self.last_loss:.5f}")
        ]

        y_offset = card1_rect.y + int(31 * SCALE)
        for label, val in metrics:
            lbl_surf = self.tiny_font.render(label, True, MUTED_TEXT)
            val_color = P1_COLOR if label == "Red Player (P1)" else (P2_COLOR if label == "Blue Player (P2)" else TEXT_COLOR)
            val_surf = self.tiny_font.render(val, True, val_color)
            self.virtual_screen.blit(lbl_surf, (card1_rect.x + int(10 * SCALE), y_offset))
            self.virtual_screen.blit(val_surf, (card1_rect.right - val_surf.get_width() - int(10 * SCALE), y_offset))
            y_offset += int(14 * SCALE)

        card2_rect = pygame.Rect(card_x, int(248 * SCALE), card_w, int(130 * SCALE))
        pygame.draw.rect(self.virtual_screen, CARD_BG, card2_rect, border_radius=int(10 * SCALE))

        h_title = self.small_font.render("MATCH LOG HISTORY", True, TEXT_COLOR)
        self.virtual_screen.blit(h_title, (card2_rect.x + int(10 * SCALE), card2_rect.y + int(8 * SCALE)))
        pygame.draw.line(self.virtual_screen, (235, 238, 242), (card2_rect.x + int(10 * SCALE), card2_rect.y + int(28 * SCALE)), (card2_rect.right - int(10 * SCALE), card2_rect.y + int(28 * SCALE)))

        headers = [("Game", 10), ("Winner", 60), ("Turns", 150), ("Time", 210)]
        for h_text, h_pos in headers:
            self.virtual_screen.blit(self.tiny_font.render(h_text, True, MUTED_TEXT), (card2_rect.x + int(h_pos * SCALE), card2_rect.y + int(31 * SCALE)))

        hy_offset = card2_rect.y + int(48 * SCALE)
        for log in reversed(list(self.match_history)[:4]):
            w_color = MUTED_TEXT if log.get('winner_id') == 0 else TEXT_COLOR
            self.virtual_screen.blit(self.tiny_font.render(f"#{log['game']}", True, TEXT_COLOR), (card2_rect.x + int(10 * SCALE), hy_offset))
            self.virtual_screen.blit(self.tiny_font.render(log['winner'], True, w_color), (card2_rect.x + int(60 * SCALE), hy_offset))
            self.virtual_screen.blit(self.tiny_font.render(str(log['turns']), True, TEXT_COLOR), (card2_rect.x + int(150 * SCALE), hy_offset))
            self.virtual_screen.blit(self.tiny_font.render(f"{log['time']}s", True, TEXT_COLOR), (card2_rect.x + int(210 * SCALE), hy_offset))
            hy_offset += int(18 * SCALE)

        term_y = int(388 * SCALE)
        term_h = HEIGHT - term_y - int(15 * SCALE)
        card3_rect = pygame.Rect(card_x, term_y, card_w, term_h)
        pygame.draw.rect(self.virtual_screen, TERMINAL_BG, card3_rect, border_radius=int(10 * SCALE))

        t_title = self.console_font.render("> LIVE TERMINAL FEED", True, (150, 160, 180))
        self.virtual_screen.blit(t_title, (card3_rect.x + int(10 * SCALE), card3_rect.y + int(6 * SCALE)))
        pygame.draw.line(self.virtual_screen, (40, 45, 55), (card3_rect.x + int(10 * SCALE), card3_rect.y + int(22 * SCALE)), (card3_rect.right - int(10 * SCALE), card3_rect.y + int(22 * SCALE)))

        logs_to_show = list(sys.stdout.logs)[-12:]
        ty_offset = card3_rect.y + int(26 * SCALE)
        for line in logs_to_show:
            if ty_offset + int(13 * SCALE) > card3_rect.bottom - int(5 * SCALE): break
            line_str = line[:38] + ".." if len(line) > 40 else line
            self.virtual_screen.blit(self.console_font.render(line_str, True, TERMINAL_TEXT), (card3_rect.x + int(10 * SCALE), ty_offset))
            ty_offset += int(14 * SCALE)

    def draw_board(self, mouse_pos):
        if self.headless: return 'none', -1, -1, self.wall_orientation
        self.virtual_screen.fill(BG_COLOR)
        
        c_p1_w, c_p2_w = self.state['p1_walls'], self.state['p2_walls']
        p1_txt = self.small_font.render(f"Red ({self.p1_gen_str}) · {c_p1_w}", True, TEXT_COLOR)
        p2_txt = self.small_font.render(f"{c_p2_w} · Blue ({self.p2_gen_str})", True, TEXT_COLOR)
        self.virtual_screen.blit(p1_txt, (LEFT_SIDEBAR_WIDTH + int(20 * SCALE), int(35 * SCALE)))
        self.virtual_screen.blit(p2_txt, (LEFT_SIDEBAR_WIDTH + BOARD_PIXEL_SIZE - p2_txt.get_width() - int(20 * SCALE), int(35 * SCALE)))

        center_x = LEFT_SIDEBAR_WIDTH + BOARD_PIXEL_SIZE // 2
        status_text = "Draw" if self.winner == "Draw" else f"{self.winner} won!" if self.game_over else ("Red's turn" if self.state['turn'] == 1 else "Blue's turn")
        status_color = MUTED_TEXT if self.winner == "Draw" else (TEXT_COLOR if self.game_over else (P1_COLOR if self.state['turn'] == 1 else P2_COLOR))
        status_surf = self.small_font.render(status_text, True, status_color)
        self.virtual_screen.blit(status_surf, status_surf.get_rect(center=(center_x, int(35 * SCALE))))
        
        if self.mode == "TRAIN":
            if self.is_evaluating:
                rem = self.eval_game_total - self.eval_game_curr
                games_txt = self.small_font.render(f"EVAL: vs Gen #{self.eval_vs_gen} | {self.eval_game_curr}/{self.eval_game_total} ({rem} left)", True, (230, 110, 20))
            else:
                lbl = "TRAINING PAUSED" if not self.is_training_active else f"Games: {self.games_played} | Turns: {self.state['turns_played']}"
                games_txt = self.small_font.render(lbl, True, (180, 50, 50) if not self.is_training_active else P2_COLOR)
            self.virtual_screen.blit(games_txt, (center_x - games_txt.get_width() // 2, int(58 * SCALE)))

        board_rect = pygame.Rect(LEFT_SIDEBAR_WIDTH + int(12 * SCALE), int(78 * SCALE), BOARD_PIXEL_SIZE - int(24 * SCALE), BOARD_PIXEL_SIZE - int(12 * SCALE))
        pygame.draw.rect(self.virtual_screen, (224, 227, 232), board_rect.move(0, int(4 * SCALE)), border_radius=int(18 * SCALE))
        pygame.draw.rect(self.virtual_screen, BOARD_BG, board_rect, border_radius=int(18 * SCALE))

        offset_x, offset_y = PADDING + LEFT_SIDEBAR_WIDTH, int(85 * SCALE) + PADDING
        valid_moves = []
        if not self.game_over and self.mode == "PLAY":
            valid_moves = self.bot2.get_valid_moves(self.state, self.state['turn'])

        mx, my = mouse_pos
        local_x, local_y = mx - offset_x, my - offset_y
        unit = CELL_SIZE + WALL_THICKNESS
        hover_type, hover_c, hover_r = 'none', -1, -1
        active_orient = self.wall_orientation

        if 0 <= local_x <= (BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * WALL_THICKNESS) and 0 <= local_y <= (BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * WALL_THICKNESS) and not self.game_over:
            c_idx, r_idx = int(local_x // unit), int(local_y // unit)
            rx, ry = local_x % unit, local_y % unit
            if c_idx < 9 and r_idx < 9:
                in_c, in_r = rx < CELL_SIZE, ry < CELL_SIZE
                if in_c and in_r: hover_type, hover_c, hover_r = 'cell', c_idx, r_idx
                elif not in_c and in_r and c_idx < 8: hover_type, hover_c, hover_r, active_orient = 'wall', c_idx, r_idx, 'V'
                elif in_c and not in_r and r_idx < 8: hover_type, hover_c, hover_r, active_orient = 'wall', c_idx, r_idx, 'H'
                else: hover_type, hover_c, hover_r = 'wall', min(c_idx, 7), min(r_idx, 7)

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x, y = offset_x + c * unit, offset_y + r * unit
                is_hovered = (hover_type == 'cell' and hover_c == c and hover_r == r and self.mode == "PLAY" and (c, r) in valid_moves)
                pygame.draw.rect(self.virtual_screen, CELL_HOVER if is_hovered else CELL_COLOR, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE), border_radius=int(9 * SCALE))
                if (c, r) in valid_moves and self.state['turn'] == 1:
                    pygame.draw.circle(self.virtual_screen, VALID_MOVE_DOT, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), int(7 * SCALE))

        for (c, r) in self.state['walls_h']:
            x, y = offset_x + c * unit, offset_y + r * unit + CELL_SIZE
            rct = pygame.Rect(x, y, CELL_SIZE * 2 + WALL_THICKNESS, WALL_THICKNESS)
            pygame.draw.rect(self.virtual_screen, WALL_SHADOW, rct.move(0, int(2 * SCALE)), border_radius=int(6 * SCALE))
            wall_col = ORANGE_COLOR if (c, r, 'H') in self.state.get('random_walls', set()) else WALL_COLOR
            pygame.draw.rect(self.virtual_screen, wall_col, rct, border_radius=int(6 * SCALE))

        for (c, r) in self.state['walls_v']:
            x, y = offset_x + c * unit + CELL_SIZE, offset_y + r * unit
            rct = pygame.Rect(x, y, WALL_THICKNESS, CELL_SIZE * 2 + WALL_THICKNESS)
            pygame.draw.rect(self.virtual_screen, WALL_SHADOW, rct.move(int(2 * SCALE), 0), border_radius=int(6 * SCALE))
            wall_col = ORANGE_COLOR if (c, r, 'V') in self.state.get('random_walls', set()) else WALL_COLOR
            pygame.draw.rect(self.virtual_screen, wall_col, rct, border_radius=int(6 * SCALE))

        if hover_type == 'wall' and not self.game_over and self.mode == "PLAY" and self.state['turn'] == 1:
            if self.bot2.is_valid_wall(self.state, hover_c, hover_r, active_orient) and self.state['p1_walls'] > 0:
                hx, hy = offset_x + hover_c * unit, offset_y + hover_r * unit
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                if active_orient == 'H': pygame.draw.rect(overlay, (82, 88, 98, 115), (hx, hy + CELL_SIZE, CELL_SIZE * 2 + WALL_THICKNESS, WALL_THICKNESS), border_radius=int(6 * SCALE))
                else: pygame.draw.rect(overlay, (82, 88, 98, 115), (hx + CELL_SIZE, hy, WALL_THICKNESS, CELL_SIZE * 2 + WALL_THICKNESS), border_radius=int(6 * SCALE))
                self.virtual_screen.blit(overlay, (0, 0))

        t1_x, t1_y = self._get_center_pixels(self.state['p1_pos'])
        t2_x, t2_y = self._get_center_pixels(self.state['p2_pos'])

        speed = 1.0 if (self.mode == "TRAIN" or self.is_evaluating) else 0.3
        self.p1_draw_x += (t1_x - self.p1_draw_x) * speed
        self.p1_draw_y += (t1_y - self.p1_draw_y) * speed
        self.p2_draw_x += (t2_x - self.p2_draw_x) * speed
        self.p2_draw_y += (t2_y - self.p2_draw_y) * speed

        radius = CELL_SIZE // 2 - int(7 * SCALE)
        pygame.draw.circle(self.virtual_screen, (218, 221, 226), (int(self.p1_draw_x), int(self.p1_draw_y + int(3 * SCALE))), radius)
        p1_color = ORANGE_COLOR if self.state.get('p1_last_random') else P1_COLOR
        pygame.draw.circle(self.virtual_screen, p1_color, (int(self.p1_draw_x), int(self.p1_draw_y)), radius)
        pygame.draw.circle(self.virtual_screen, (218, 221, 226), (int(self.p2_draw_x), int(self.p2_draw_y + int(3 * SCALE))), radius)
        p2_color = ORANGE_COLOR if self.state.get('p2_last_random') else P2_COLOR
        pygame.draw.circle(self.virtual_screen, p2_color, (int(self.p2_draw_x), int(self.p2_draw_y)), radius)

        btn_y = HEIGHT - int(46 * SCALE)
        btn_h, btn_w = int(34 * SCALE), int(110 * SCALE)
        
        self.rect_menu = pygame.Rect(LEFT_SIDEBAR_WIDTH + BOARD_PIXEL_SIZE - int(125 * SCALE), btn_y, btn_w, btn_h)
        m_hover = self.rect_menu.collidepoint(mouse_pos)
        pygame.draw.rect(self.virtual_screen, (200, 205, 212) if m_hover else (220, 224, 230), self.rect_menu, border_radius=int(8 * SCALE))
        menu_txt = self.small_font.render("Menu", True, TEXT_COLOR)
        self.virtual_screen.blit(menu_txt, menu_txt.get_rect(center=self.rect_menu.center))

        self.rect_new_game = pygame.Rect(LEFT_SIDEBAR_WIDTH + BOARD_PIXEL_SIZE - int(245 * SCALE), btn_y, btn_w, btn_h)
        ng_hover = self.rect_new_game.collidepoint(mouse_pos)
        pygame.draw.rect(self.virtual_screen, (200, 205, 212) if ng_hover else (220, 224, 230), self.rect_new_game, border_radius=int(8 * SCALE))
        ng_txt = self.small_font.render("New Game", True, TEXT_COLOR)
        self.virtual_screen.blit(ng_txt, ng_txt.get_rect(center=self.rect_new_game.center))

        if self.mode == "TRAIN":
            self.rect_train_toggle = pygame.Rect(LEFT_SIDEBAR_WIDTH + int(15 * SCALE), btn_y, int(160 * SCALE), btn_h)
            tt_hover = self.rect_train_toggle.collidepoint(mouse_pos)
            
            if self.is_training_active:
                btn_color = (238, 92, 105) if not tt_hover else (218, 72, 85)
                lbl = "Stop Training"
            else:
                btn_color = (66, 175, 90) if not tt_hover else (56, 155, 80)
                lbl = "Continue Training"
                
            pygame.draw.rect(self.virtual_screen, btn_color, self.rect_train_toggle, border_radius=int(8 * SCALE))
            tt_txt = self.small_font.render(lbl, True, (255, 255, 255))
            self.virtual_screen.blit(tt_txt, tt_txt.get_rect(center=self.rect_train_toggle.center))

        if self.toast_timer > 0:
            self.toast_timer -= 1
            toast_surf = self.small_font.render(self.toast_msg, True, (255, 255, 255))
            t_w, t_h = toast_surf.get_width() + int(24 * SCALE), int(30 * SCALE)
            toast_rect = pygame.Rect(LEFT_SIDEBAR_WIDTH + BOARD_PIXEL_SIZE // 2 - t_w // 2, int(8 * SCALE), t_w, t_h)
            pygame.draw.rect(self.virtual_screen, (46, 125, 50), toast_rect, border_radius=int(15 * SCALE))
            self.virtual_screen.blit(toast_surf, toast_surf.get_rect(center=toast_rect.center))

        self.draw_left_sidebar()
        self.draw_right_sidebar()
        self.render_and_flip()
        return hover_type, hover_c, hover_r, active_orient

    # ==========================================
    # SECTION 7: CHECK WIN CALL SEQUENCE
    # ==========================================
    def _detect_terminal(self):
        self.game_over = False
        winner_id = 0
        if self.state['p1_pos'][1] == 0:
            self.game_over, winner_id = True, 1
        elif self.state['p2_pos'][1] == 8:
            self.game_over, winner_id = True, 2
        elif self.state['turns_played'] >= 200:
            self.game_over, winner_id = True, 0

        self.winner_id = winner_id
        return self.game_over, winner_id

    def _commit_training_action(self, state_tensor, turn_pid):
        done, winner_id = self._detect_terminal()
        next_pid = self.state['turn']
        self.train_step(state_tensor, turn_pid, encode_state(self.state, next_pid),
                        next_pid, done, winner_id)
        self.check_win()

    def check_win(self):
        self._detect_terminal()
        if getattr(self, '_result_recorded', False):
            return
        winner_id = self.winner_id
        if self.game_over:
            self._result_recorded = True
            self.winner_id = winner_id
            if winner_id == 0:
                self.winner = "Draw"
            elif getattr(self, 'mode', 'TRAIN') == "TRAIN":
                self.winner = "Student" if winner_id == self.student_id else "Teacher"
            else:
                self.winner = "Human" if winner_id == 1 else "Teacher"

            print(
                f"Game #{self.games_played + 1} Winner: "
                f"{self.winner.upper()} ({self.state['turns_played']} turns)"
            )
            self.log_match_result(winner_id)

            if self.mode == "TRAIN":
                self.games_played += 1

                student_won = 1 if winner_id == self.student_id else 0
                self.student_eval_history.append(student_won)

                gen = self.current_opponent_gen
                if gen not in self.checkpoint_log:
                    self.checkpoint_log[gen] = deque(maxlen=self.LOG_WINDOW_SIZE)
                self.checkpoint_log[gen].append(student_won)
                self.lr_training_games_seen += 1

                if self.current_opponent_is_latest:
                    self.latest_teacher_history.append(student_won)
                    self.latest_teacher_games_seen += 1

                # Keep the displayed and next-game sampling weights current
                # after every result, including historical opponents.
                self._refresh_opponent_weights()

                latest_log = self.checkpoint_log.get(self.teacher_gen)
                if latest_log:
                    latest_wins = sum(latest_log)
                    latest_games = len(latest_log)
                    self.student_win_rate_str = (
                        f"{latest_wins / latest_games * 100:.1f}% Latest "
                        f"({latest_wins}/{latest_games})"
                    )

                self._tune_learning_rate_from_pool(force=False)
                print(f"Learning: {self.optimizer_steps - self._game_start_optimizer_steps} "
                      f"steps this game / {self.optimizer_steps} total | "
                      f"fixed LR={self.optimizer.param_groups[0]['lr']:.8f} | "
                      f"wall filter={self.heuristic_weight:.3f} | "
                      f"replay samples latest/historic="
                      f"{self.replay_samples_latest}/{self.replay_samples_historic}")
                print(f"Current training WRs: {self._format_training_win_rates()}")
                self.save_meta()

                if self._all_training_gates_met():
                    print(
                        f"Candidate Gen #{self.teacher_gen + 1} cleared every "
                        f"live training gate. Starting final frozen test."
                    )
                    if self.run_final_promotion_test():
                        print(
                            f"Candidate Gen #{self.teacher_gen + 1} passed every "
                            f"final gate. Installing new teacher."
                        )

                        # Preserve the teacher that was actually trained against
                        # before the live teacher network is overwritten.
                        torch.save(
                            self.teacher_net.state_dict(),
                            f'quorinet_teacher_gen_{self.teacher_gen}.pth'
                        )

                        self.update_escalation_state()
                        self.teacher_gen += 1
                        self.teacher_net.load_state_dict(
                            self.student_net.state_dict()
                        )

                        self._roll_replay_buffers_for_new_teacher()
                        self.latest_teacher_history.clear()
                        self.checkpoint_log.clear()
                        self._reset_lr_search_for_new_teacher()
                        self.curriculum_latest_credit = 1.0

                        self.update_opponent_pool()
                        self.student_eval_history.clear()
                        self.student_win_rate_str = "Pending live WR"
                        self.save_checkpoint(manual=True)
                    else:
                        print(f"Current training WRs: {self._format_training_win_rates()}")
                        self.save_meta()

            if self.mode == "TRAIN" and (
                self.games_played % 250 == 0 or self._save_requested
            ):
                self.save_checkpoint(manual=self._save_requested)

    def run_headless_single_game_overfit_check(self, epochs=500, learning_rate=0.001):
        """Play one legal game, then prove an isolated value net can memorize it.

        This is intentionally isolated from normal training: it does not touch
        replay, optimizer state, generation counters, metadata, or checkpoints.
        """
        if epochs < 1:
            raise ValueError(f"overfit epochs must be positive, got {epochs}")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                f"overfit learning rate must be finite and positive, got {learning_rate}"
            )

        print("\n--- SINGLE-GAME NEURAL OVERFIT CHECK ---")
        print(
            f"Shape contract: state {STATE_SHAPE} | batch [B, {STATE_CHANNELS}, "
            f"{BOARD_SIZE}, {BOARD_SIZE}] | value [B, {VALUE_WIDTH}]"
        )
        validate_model_contract(self.student_net, "source student network")

        # A deterministic pawn-only race guarantees that the single diagnostic
        # game terminates. Every position and transition is still produced by
        # the real legality, apply_action(), and encode_state() paths.
        diagnostic_state = {
            'p1_pos': (4, 8), 'p2_pos': (4, 0),
            'p1_walls': 10, 'p2_walls': 10,
            'walls_h': set(), 'walls_v': set(),
            'turn': 1, 'turns_played': 0, 'history': set(),
            'p1_recent': (), 'p2_recent': (),
            'random_walls': set(), 'p1_last_random': False, 'p2_last_random': False
        }
        rules_bot = QuoridorBot(bot_id=1, max_depth=1, nn_model=None)
        trajectory = []
        winner_id = 0

        for _ in range(200):
            pid = diagnostic_state['turn']
            trajectory.append((encode_state(diagnostic_state, pid), pid))
            valid_moves = rules_bot.get_valid_moves(diagnostic_state, pid)
            if not valid_moves:
                winner_id = rules_bot.get_opponent(pid)
                break

            def remaining_distance(move):
                candidate = rules_bot.apply_action(
                    diagnostic_state,
                    pid,
                    ('move', move)
                )
                path = rules_bot.get_shortest_path(candidate, pid)
                return len(path) if path is not None else 999

            chosen_move = min(valid_moves, key=remaining_distance)
            diagnostic_state = rules_bot.apply_action(
                diagnostic_state,
                pid,
                ('move', chosen_move)
            )
            if diagnostic_state['p1_pos'][1] == 0:
                winner_id = 1
                break
            if diagnostic_state['p2_pos'][1] == 8:
                winner_id = 2
                break

        if not trajectory:
            raise RuntimeError("diagnostic game produced no training states")

        result = "Draw" if winner_id == 0 else f"Player {winner_id}"
        print(
            f"Diagnostic game: {len(trajectory)} plies | winner: {result} | "
            f"final board turn count: {diagnostic_state['turns_played']}"
        )

        final_index = len(trajectory) - 1
        state_tensors = []
        target_values = []
        for index, (state_tensor, active_pid) in enumerate(trajectory):
            if winner_id == 0:
                outcome = 0.0
            else:
                outcome = 1.0 if active_pid == winner_id else -1.0
            horizon = final_index - index
            target = outcome * (self.OUTCOME_BACKFILL_DISCOUNT ** horizon)
            state_tensors.append(state_tensor)
            target_values.append(target)

        device = model_device(self.student_net)
        # Use a deterministic fresh instance so an already-trained student
        # cannot make the diagnostic pass without demonstrating optimization.
        torch.manual_seed(0)
        diagnostic_net = SpatialQuoriNetV2(in_channels=STATE_CHANNELS).to(device)
        validate_model_contract(diagnostic_net, "diagnostic network")
        diagnostic_net.train()

        states = make_state_batch(
            state_tensors,
            device=device,
            name="single-game states"
        )
        targets = make_value_batch(
            target_values,
            device=device,
            name="single-game targets"
        )
        if states.shape[0] != targets.shape[0]:
            raise RuntimeError(
                f"diagnostic sample count mismatch: {states.shape[0]} states vs "
                f"{targets.shape[0]} targets"
            )

        diagnostic_optimizer = optim.Adam(
            diagnostic_net.parameters(),
            lr=learning_rate
        )
        loss_fn = nn.MSELoss()
        parameter_start = {
            name: parameter.detach().clone()
            for name, parameter in diagnostic_net.named_parameters()
        }

        with torch.no_grad():
            initial_predictions = diagnostic_net(states)
            initial_loss = checked_value_loss(
                loss_fn,
                initial_predictions,
                targets,
                name="initial diagnostic MSE"
            ).item()
            initial_max_error = (initial_predictions - targets).abs().max().item()

        report_every = max(1, epochs // 5)
        final_loss = initial_loss
        epochs_run = 0
        for epoch in range(1, epochs + 1):
            diagnostic_optimizer.zero_grad(set_to_none=True)
            predictions = diagnostic_net(states)
            loss = checked_value_loss(
                loss_fn,
                predictions,
                targets,
                name="diagnostic training MSE"
            )
            loss.backward()
            for name, parameter in diagnostic_net.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                    raise FloatingPointError(f"diagnostic gradient is non-finite in {name}")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                diagnostic_net.parameters(),
                self.GRAD_CLIP_NORM
            )
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError("diagnostic gradient norm is NaN or infinite")
            diagnostic_optimizer.step()
            final_loss = loss.item()
            epochs_run = epoch

            if epoch == 1 or epoch % report_every == 0:
                print(f"Overfit epoch {epoch:4d}/{epochs}: MSE={final_loss:.8f}")
            if final_loss <= 0.001:
                print(f"Overfit reached early-stop MSE at epoch {epoch}.")
                break

        with torch.no_grad():
            final_predictions = diagnostic_net(states)
            final_loss = checked_value_loss(
                loss_fn,
                final_predictions,
                targets,
                name="final diagnostic MSE"
            ).item()
            final_max_error = (final_predictions - targets).abs().max().item()

        squared_delta = 0.0
        for name, parameter in diagnostic_net.named_parameters():
            difference = parameter.detach() - parameter_start[name]
            squared_delta += difference.pow(2).sum().item()
        parameter_delta = math.sqrt(squared_delta)
        reduction = (
            1.0
            if initial_loss <= 1e-12 and final_loss <= initial_loss + 1e-12
            else max(0.0, 1.0 - (final_loss / max(initial_loss, 1e-12)))
        )
        learned = parameter_delta > 1e-10 and reduction >= 0.90
        overfit = final_loss <= 0.02
        passed = overfit and learned

        print(f"Tensor shapes: states={tuple(states.shape)}, targets={tuple(targets.shape)}")
        print(
            f"Initial MSE={initial_loss:.8f}, max error={initial_max_error:.6f} | "
            f"Final MSE={final_loss:.8f}, max error={final_max_error:.6f}"
        )
        print(
            f"Loss reduction={reduction * 100:.2f}% | parameter L2 delta="
            f"{parameter_delta:.8f} | epochs={epochs_run}"
        )
        print(
            "OVERFIT CHECK PASSED: the neural network changed parameters and "
            "memorized the single game."
            if passed else
            "OVERFIT CHECK FAILED: inspect the reported loss, gradients, and shapes."
        )
        print("Production student, optimizer, replay, and checkpoints were not modified.")
        return passed

    def _training_epsilon(self, active_bot):
        is_teacher = (active_bot.nn_model is not self.student_net)
        if is_teacher:
            return 0.0
        return max(0.05, 0.30 * (0.85 ** self.state['turns_played']))

    def run_headless_training(self, num_games):
        print(f"Starting Headless Turbo Training for {num_games} games...")
        self.mode = "TRAIN"
        self.is_training_active = True
        self.reset_game()
        self._quit_requested = False
        
        def command_listener():
            while True:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'p': self.is_training_active = not self.is_training_active
                elif cmd == 's': self._save_requested = True
                elif cmd == 'q': self._quit_requested = True
                elif not cmd: return
                    
        threading.Thread(target=command_listener, daemon=True).start()
        target_games, last_print = self.games_played + num_games, self.games_played
        
        while self.games_played < target_games:
            if self._quit_requested and self.game_over:
                break
            if not self.is_training_active and not self._quit_requested:
                time.sleep(0.5)
                continue
                
            if self.game_over:
                self.print_headless_metrics()
                if self.games_played - last_print >= 10:
                    print(f"Training Progress: {self.games_played} / {target_games} games")
                    last_print = self.games_played
                self.reset_game()
                continue

            active_bot = self.bot1 if self.state['turn'] == 1 else self.bot2
            pre_action_state_tensor = encode_state(self.state, self.state['turn'])
            pre_action_pid = self.state['turn']
            
            # The student explores against every generation; teachers stay greedy.
            eps = self._training_epsilon(active_bot)
            
            action, _ = active_bot.get_move_and_eval(self.state, epsilon=eps)
            if action: self.state = active_bot.apply_action(self.state, pre_action_pid, action, is_random=getattr(active_bot, 'was_last_random', False))
            else: self.state['turns_played'] = 200
                
            self._commit_training_action(pre_action_state_tensor, pre_action_pid)
                
        print(f"Headless training complete! Total games played: {self.games_played}")
        self.save_checkpoint(manual=True)
        pygame.quit()
        sys.exit()

    def play(self):
        clock = pygame.time.Clock()
        running, bot_thinking, bot_result = True, False, None

        def think_bot(current_state):
            nonlocal bot_result, bot_thinking
            try:
                bot_result = self.bot2.get_move(current_state)
                if bot_result is None:
                    moves = self.bot2.get_valid_moves(current_state, 2)
                    if moves: bot_result = ('move', moves[0])
            except Exception:
                moves = self.bot2.get_valid_moves(current_state, 2)
                if moves: bot_result = ('move', moves[0])
            finally: bot_thinking = False

        while running:
            virtual_mouse_pos = self.get_virtual_mouse_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_F11, pygame.K_f): self.toggle_fullscreen()
                    elif event.key == pygame.K_c: self.save_checkpoint(manual=True)

                if self.mode == "MENU" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.rect_play.collidepoint(virtual_mouse_pos):
                        self.mode, self.is_training_active = "PLAY", False
                        self.reset_game()
                    elif self.rect_train.collidepoint(virtual_mouse_pos):
                        self.mode, self.is_training_active = "TRAIN", True
                        self.reset_game()

                elif self.mode in ("PLAY", "TRAIN") and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.rect_menu.collidepoint(virtual_mouse_pos): self.mode = "MENU"
                    elif self.rect_new_game.collidepoint(virtual_mouse_pos): self.reset_game()
                    elif self.mode == "TRAIN" and self.rect_train_toggle.collidepoint(virtual_mouse_pos):
                        self.is_training_active = not self.is_training_active
                        if not self.is_training_active: self.save_checkpoint(manual=True)

                if self.mode == "PLAY" and not self.game_over and not bot_thinking and self.state['turn'] == 1:
                    if event.type == pygame.KEYDOWN and event.key in (pygame.K_r, pygame.K_SPACE):
                        self.wall_orientation = 'V' if self.wall_orientation == 'H' else 'H'
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 3: self.wall_orientation = 'V' if self.wall_orientation == 'H' else 'H'
                        elif event.button == 1:
                            hover_type, hover_c, hover_r, active_orient = self.draw_board(virtual_mouse_pos)
                            if hover_type == 'cell':
                                if (hover_c, hover_r) in self.bot2.get_valid_moves(self.state, 1):
                                    self.state = self.bot2.apply_action(self.state, 1, ('move', (hover_c, hover_r)))
                                    self.check_win()
                            elif hover_type == 'wall':
                                if self.state['p1_walls'] > 0 and self.bot2.is_valid_wall(self.state, hover_c, hover_r, active_orient):
                                    self.state = self.bot2.apply_action(self.state, 1, ('wall', (hover_c, hover_r, active_orient)))
                                    self.check_win()

            if self.mode == "TRAIN" and self.is_training_active and not self.is_evaluating:
                if self.game_over:
                    self.reset_game()
                else:
                    active_bot = self.bot1 if self.state['turn'] == 1 else self.bot2
                    pre_action_state_tensor = encode_state(self.state, self.state['turn'])
                    pre_action_pid = self.state['turn']

                    eps = self._training_epsilon(active_bot)
                    
                    action, _ = active_bot.get_move_and_eval(self.state, epsilon=eps)
                    if action: self.state = active_bot.apply_action(self.state, pre_action_pid, action, is_random=getattr(active_bot, 'was_last_random', False))
                    else: self.state['turns_played'] = 200

                    self._commit_training_action(pre_action_state_tensor, pre_action_pid)

            elif self.mode == "PLAY" and self.state['turn'] == 2 and not self.game_over:
                if not bot_thinking and bot_result is None:
                    bot_thinking = True
                    bot_state = {
                        'p1_pos': self.state['p1_pos'], 'p2_pos': self.state['p2_pos'],
                        'p1_walls': self.state['p1_walls'], 'p2_walls': self.state['p2_walls'],
                        'walls_h': self.state['walls_h'].copy(), 'walls_v': self.state['walls_v'].copy(),
                        'turn': 2, 'turns_played': self.state['turns_played'], 'history': self.state['history'].copy(),
                        'p1_recent': self.state.get('p1_recent', ()), 'p2_recent': self.state.get('p2_recent', ())
                    }
                    threading.Thread(target=think_bot, args=(bot_state,), daemon=True).start()
                
                if not bot_thinking and bot_result is not None:
                    self.state = self.bot2.apply_action(self.state, 2, bot_result, is_random=getattr(self.bot2, 'was_last_random', False))
                    bot_result = None
                    self.check_win()

            if self.mode == "MENU": self.draw_menu()
            else: self.draw_board(virtual_mouse_pos if self.mode == "PLAY" else (-1, -1))
            clock.tick(60 if self.mode == "PLAY" else 1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for a fresh run (resumes restore saved RNG state)')
    parser.add_argument('--threads', type=int, default=min(4, os.cpu_count() or 1),
                        help='PyTorch CPU threads; small boards often benefit from 1-4')
    parser.add_argument('--train-depth', type=int, default=2,
                        help='search depth during self-play (default: 2)')
    parser.add_argument('--play-depth', type=int, default=4,
                        help='search depth against a human (default: 4)')
    parser.add_argument('--train-updates', type=int, default=16,
                        help='maximum optimizer batches per completed game (default: 16)')
    parser.add_argument('--checkpoint-dir', default=None,
                        help='directory for all training saves; defaults to the working directory')
    parser.add_argument(
        "--headless",
        type=int,
        default=0,
        metavar="GAMES",
        help="run normal headless training for this many games"
    )
    parser.add_argument(
        "--overfit-one-game",
        action="store_true",
        help=(
            "play one isolated legal game and overfit a fresh network on it; "
            "exits nonzero if the neural learning check fails"
        )
    )
    parser.add_argument(
        "--overfit-epochs",
        type=int,
        default=500,
        metavar="N",
        help="maximum epochs for --overfit-one-game (default: 500)"
    )
    parser.add_argument(
        "--overfit-lr",
        type=float,
        default=0.001,
        metavar="RATE",
        help="learning rate for --overfit-one-game (default: 0.001)"
    )
    args = parser.parse_args()

    if min(args.threads, args.train_depth, args.play_depth, args.train_updates) < 1:
        parser.error('threads, search depths and train-updates must be positive')
    torch.set_num_threads(args.threads)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.chdir(args.checkpoint_dir)

    if args.headless < 0:
        parser.error("--headless GAMES cannot be negative")
    if args.overfit_epochs < 1:
        parser.error("--overfit-epochs must be positive")
    if not math.isfinite(args.overfit_lr) or args.overfit_lr <= 0.0:
        parser.error("--overfit-lr must be finite and positive")
    headless_mode = args.headless > 0 or args.overfit_one_game
    if headless_mode:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    game = Quoridor(headless=headless_mode, train_depth=args.train_depth,
                    play_depth=args.play_depth, train_updates=args.train_updates)
    if args.overfit_one_game:
        passed = game.run_headless_single_game_overfit_check(
            epochs=args.overfit_epochs,
            learning_rate=args.overfit_lr
        )
        pygame.quit()
        sys.exit(0 if passed else 1)
    if args.headless > 0:
        game.run_headless_training(args.headless)
    else:
        game.play()
