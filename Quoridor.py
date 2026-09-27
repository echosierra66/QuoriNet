"""QuoriNet rules, search, neural network, curriculum, and trainer."""

import pygame
import sys
import os
import copy
import hashlib
import inspect
import json
import random
import time
import threading
import argparse
import math
import io
import re
import shutil
import textwrap
import warnings
import queue
from collections import deque
from functools import lru_cache
import torch
import torch.nn as nn
import torch.optim as optim

BOARD_SIZE = 9
STATE_CHANNELS = 9
STATE_SHAPE = (STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
VALUE_WIDTH = 1
POLICY_SIZE = 81 + 64 + 64
MANUAL_WEIGHTS_PATTERN = re.compile(
    r"^quorinet_gen_(\d+)_manual_(\d+)\.pth$"
)
OFFICIAL_WEIGHTS_PATTERN = re.compile(r"^quorinet_teacher_gen_(\d+)\.pth$")
PACKED_MODEL_ARCHIVE_PATTERN = re.compile(
    r"^quorinet_gen_(\d+)-(\d+)(?:_([0-9a-f]{8}))?\.pth$"
)
MODEL_ARCHIVE_PATTERN = re.compile(
    r"^gen (\d+)-(\d+)(?: ([0-9a-f]{8}))?$"
)
ARCHIVED_OFFICIAL_PATTERN = re.compile(r"^gen (\d+)\.pth$")
ARCHIVED_MANUAL_PATTERN = re.compile(r"^gen (\d+) manual (\d+)\.pth$")
ARCHITECTURE_MANIFEST = "NN architecture.json"
ARCHITECTURE_SOURCE = "NN architecture.py"
ARCHITECTURE_EXECUTABLE = "NN architecture.pt"
MODEL_ARCHIVE_FORMAT = "quorinet_architecture_archive"
MODEL_ARCHIVE_VERSION = 1
TRAINING_STATE_FILE = "quorinet_training_state.pth"
STUDENT_WEIGHTS_FILE = "quorinet_student_weights.pth"
TEACHER_WEIGHTS_FILE = "quorinet_teacher_weights.pth"
OPTIMIZER_FILE = "quorinet_optimizer.pth"
META_FILE = "quorinet_meta.json"
ACTIVE_CHECKPOINT_FILES = (
    TRAINING_STATE_FILE,
    STUDENT_WEIGHTS_FILE,
    TEACHER_WEIGHTS_FILE,
    OPTIMIZER_FILE,
    META_FILE,
)
REPLAY_FORMAT_VERSION = 1  # Whole accepted games, partitioned by teacher.
CURRENT_ENCODER_SPEC = {
    "id": "relative_9ch_path_moves_v2",
    "shape": [STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE],
    "dtype": "float32",
    "perspective": "active_player_rotated_180_for_player_2",
    "channels": [
        "active_player_position",
        "opponent_position",
        "horizontal_walls",
        "vertical_walls",
        "active_player_walls_remaining_div_10",
        "opponent_walls_remaining_div_10",
        "active_player_shortest_path_div_81",
        "opponent_shortest_path_div_81",
        "total_moves_played_div_200",
    ],
}


def mirror_state_batch(states):
    """Reflect columns while preserving the 8x8 wall-anchor geometry."""
    mirrored = states.flip(-1).clone()
    mirrored[:, 2:4] = 0
    mirrored[:, 2:4, :8, :8] = states[:, 2:4, :8, :8].flip(-1)
    return mirrored


def validate_state_tensor(tensor, name="state", shape=STATE_SHAPE):
    """Enforce the unbatched neural-state shape, dtype, and finiteness."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains NaN or infinite values")
    return tensor


def make_state_batch(state_tensors, device, name="states"):
    """Create one contiguous current-encoder batch on model device."""
    if not state_tensors:
        raise ValueError(f"{name} cannot be empty")
    for index, tensor in enumerate(state_tensors):
        if (not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != STATE_SHAPE
                or tensor.dtype != torch.float32):
            validate_state_tensor(tensor, f"{name}[{index}]")
    batch = torch.stack(state_tensors, dim=0).to(device=device, dtype=torch.float32)
    if not torch.isfinite(batch).all().item():
        # Keep precise failure messages without launching a check per state.
        for index, tensor in enumerate(state_tensors):
            validate_state_tensor(tensor, f"{name}[{index}]")
    expected = (len(state_tensors), *STATE_SHAPE)
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
    """Evaluate one model-encoded state and return its scalar value."""
    state_batch = make_model_state_batch(model, [state_tensor], name=name)
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
    """Smoke-test the model's input shape and [B, 1] value contract."""
    device = model_device(model)
    was_training = model.training
    try:
        model.eval()
        shape = tuple(getattr(model, 'input_shape', STATE_SHAPE))
        samples = [torch.zeros(shape, dtype=torch.float32) for _ in range(3)]
        batch = make_model_state_batch(model, samples, name=f"{name} smoke states")
        with torch.no_grad():
            output = model(batch)
        expected = (3, VALUE_WIDTH)
        if tuple(output.shape) != expected:
            raise RuntimeError(
                f"{name} returned {tuple(output.shape)}; expected {expected}"
            )
        if not torch.isfinite(output).all().item():
            raise FloatingPointError(f"{name} smoke output contains NaN or infinity")
        if getattr(model, 'supports_policy', False):
            with torch.no_grad():
                logits, value = model.policy_value(batch)
            if tuple(logits.shape) != (3, POLICY_SIZE) or not torch.isfinite(logits).all():
                raise ValueError(f'{name} invalid policy output')
            if not torch.equal(value, output):
                raise ValueError(f'{name} two-head and scalar value paths differ')
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


def checked_value_loss(loss_fn, predictions, targets, name="value loss", sample_weights=None):
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
    if sample_weights is not None:
        if loss.shape != targets.shape or sample_weights.shape != targets.shape:
            raise ValueError(f"{name} per-sample loss and weights must match targets")
        if (not torch.isfinite(sample_weights).all().item()
                or not (sample_weights > 0).all().item()):
            raise ValueError(f"{name} weights must be finite and positive")
        loss = (sample_weights * loss).sum() / sample_weights.sum()
    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise FloatingPointError(
            f"{name} must be one finite scalar, got shape {tuple(loss.shape)} "
            f"and value {loss.detach().cpu()}"
        )
    return loss

# ==========================================
# NEW NEURAL NETWORK ARCHITECTURE
# ==========================================
class SpatialQuoriNetV2(nn.Module):
    def __init__(self, in_channels=STATE_CHANNELS):
        # Channels: P1 pos, P2 pos, H-wall, V-wall, P1 walls-left, P2 walls-left, P1 path-dist, P2 path-dist
        super().__init__()
        if in_channels not in (8, STATE_CHANNELS):
            raise ValueError(
                f"SpatialQuoriNetV2 requires {STATE_CHANNELS} input channels, "
                f"got {in_channels}"
            )
        self.in_channels = in_channels
        self.input_shape = (in_channels, BOARD_SIZE, BOARD_SIZE)
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

class LegacyPolicyValueQuoriNet(SpatialQuoriNetV2):
    """Keep the complete V2 value path; learn spatial action priors beside it.

    forward remains scalar for existing UI, diagnostics, and value opponents.
    policy_value is the shared-trunk, two-head training/search interface.
    """
    supports_policy = True

    def __init__(self, in_channels=STATE_CHANNELS):
        super().__init__(in_channels)
        self.policy_head = nn.Conv2d(64, 3, 1)
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def policy_value(self, states):
        features = self.conv(states)
        spatial = self.policy_head(features)
        logits = torch.cat((spatial[:, 0].flatten(1),
                            spatial[:, 1, :8, :8].flatten(1),
                            spatial[:, 2, :8, :8].flatten(1)), dim=1)
        return logits, self.value_head(self.pool(features).flatten(1))


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
        )

    def forward(self, features):
        return torch.relu(features + self.layers(features))


class PolicyValueQuoriNet(nn.Module):
    """Final two-head network: eight residual blocks and spatially aware heads."""
    supports_policy = True

    def __init__(self, in_channels=STATE_CHANNELS):
        super().__init__()
        if in_channels not in (8, STATE_CHANNELS):
            raise ValueError(f"network requires {STATE_CHANNELS} input channels")
        self.in_channels = in_channels
        self.input_shape = (in_channels, BOARD_SIZE, BOARD_SIZE)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            *(ResidualBlock() for _ in range(8)),
        )
        self.policy_features = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.policy_head = nn.Conv2d(32, 3, 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Flatten(), nn.Linear(8 * BOARD_SIZE * BOARD_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def _features(self, states):
        if not isinstance(states, torch.Tensor):
            raise TypeError("network input must be a torch.Tensor")
        if states.ndim != 4 or tuple(states.shape[1:]) != self.input_shape or states.shape[0] < 1:
            raise ValueError(f"network input must have shape [B, {self.input_shape}] with B >= 1")
        if states.dtype != torch.float32:
            raise TypeError("network input must have dtype torch.float32")
        return self.conv(states)

    def forward(self, states):
        return self.value_head(self._features(states))

    def policy_value(self, states):
        features = self._features(states)
        spatial = self.policy_head(self.policy_features(features))
        logits = torch.cat((spatial[:, 0].flatten(1),
                            spatial[:, 1, :8, :8].flatten(1),
                            spatial[:, 2, :8, :8].flatten(1)), dim=1)
        return logits, self.value_head(features)


def network_from_weights(weights):
    in_channels = weights['conv.0.weight'].shape[1]
    if 'policy_features.0.weight' in weights:
        return PolicyValueQuoriNet(in_channels)
    if 'policy_head.weight' in weights:
        return LegacyPolicyValueQuoriNet(in_channels)
    return SpatialQuoriNetV2(in_channels)


def policy_action_index(action, pid):
    """209 actions in the same active-player coordinates as the encoder."""
    if action[0] == 'move':
        c, r = action[1]
        if pid == 2:
            c, r = 8 - c, 8 - r
        return r * 9 + c
    c, r, orient = action[1]
    if pid == 2:
        c, r = 7 - c, 7 - r
    return 81 + (64 if orient == 'V' else 0) + r * 8 + c


def mirror_policy_batch(policy):
    return torch.cat((policy[:, :81].reshape(-1, 9, 9).flip(-1).flatten(1),
                      policy[:, 81:145].reshape(-1, 8, 8).flip(-1).flatten(1),
                      policy[:, 145:].reshape(-1, 8, 8).flip(-1).flatten(1)), dim=1)


def restore_student_optimizer(optimizer, saved):
    """Append uninitialized policy slots while retaining every old Adam slot."""
    saved = copy.deepcopy(saved)
    current = optimizer.state_dict()
    if len(saved['param_groups']) != len(current['param_groups']):
        raise ValueError('optimizer parameter groups changed')
    for old, new in zip(saved['param_groups'], current['param_groups']):
        if len(old['params']) > len(new['params']):
            raise ValueError('optimizer has more parameters than the student')
        if old['params'] != new['params'][:len(old['params'])]:
            raise ValueError('optimizer parameter ordering changed')
        old['params'] = new['params']
    optimizer.load_state_dict(saved)
    validate_optimizer_state_shapes(optimizer)


# Global Helper for encode_state distance calcs
def get_adjacent_global(state, c, r):
    moves = []
    if r > 0 and (c, r - 1) not in state['walls_h'] and (c - 1, r - 1) not in state['walls_h']: moves.append((c, r - 1))
    if r < 8 and (c, r) not in state['walls_h'] and (c - 1, r) not in state['walls_h']: moves.append((c, r + 1))
    if c > 0 and (c - 1, r) not in state['walls_v'] and (c - 1, r - 1) not in state['walls_v']: moves.append((c - 1, r))
    if c < 8 and (c, r) not in state['walls_v'] and (c, r - 1) not in state['walls_v']: moves.append((c + 1, r))
    return moves

_OPEN_BOARD_ADJACENCY = tuple(
    tuple(
        nr * BOARD_SIZE + nc
        for nc, nr in ((c, r - 1), (c, r + 1), (c - 1, r), (c + 1, r))
        if 0 <= nc < BOARD_SIZE and 0 <= nr < BOARD_SIZE
    )
    for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
)


@lru_cache(maxsize=4096)
def _wall_adjacency(walls_h, walls_v):
    """Share integer-square connectivity between both goal-distance fields."""
    adjacent = [list(neighbors) for neighbors in _OPEN_BOARD_ADJACENCY]
    for walls, offset, span in ((walls_h, BOARD_SIZE, 1), (walls_v, 1, BOARD_SIZE)):
        for c, r in walls:
            start = r * BOARD_SIZE + c
            for square in (start, start + span):
                other = square + offset
                if other in adjacent[square]:
                    adjacent[square].remove(other)
                    adjacent[other].remove(square)
    return tuple(tuple(neighbors) for neighbors in adjacent)


@lru_cache(maxsize=8192)
def _goal_distances(walls_h, walls_v, target_row):
    """One reverse BFS supplies every square's distance for a wall layout."""
    adjacent = _wall_adjacency(walls_h, walls_v)
    distances = [999] * (BOARD_SIZE * BOARD_SIZE)
    queue = list(range(target_row * BOARD_SIZE, (target_row + 1) * BOARD_SIZE))
    for square in queue:
        distances[square] = 0
    for square in queue:
        next_distance = distances[square] + 1
        for neighbor in adjacent[square]:
            if distances[neighbor] == 999:
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return tuple(distances)


def get_distance_field(state, pid):
    return _goal_distances(
        frozenset(state['walls_h']),
        frozenset(state['walls_v']),
        0 if pid == 1 else 8
    )


def get_shortest_path_dist(state, pid):
    c, r = state[f'p{pid}_pos']
    return get_distance_field(state, pid)[r * BOARD_SIZE + c]


@lru_cache(maxsize=8192)
def _wall_path_witnesses(walls_h, walls_v, p1_pos, p2_pos):
    """Keep one existing route per pawn; an untouched route proves legality."""
    adjacent = _wall_adjacency(walls_h, walls_v)
    paths = []
    for (c, r), target in ((p1_pos, 0), (p2_pos, 8)):
        distances = _goal_distances(walls_h, walls_v, target)
        square = r * BOARD_SIZE + c
        if distances[square] == 999:
            paths.append(None)
            continue
        edges = set()
        while distances[square]:
            other = next(n for n in adjacent[square]
                         if distances[n] == distances[square] - 1)
            edges.add((square, other))
            square = other
        paths.append(frozenset(edges))
    return adjacent, tuple(paths)


def _reaches_goal_without_edges(adjacent, position, target_row, blocked):
    square = position[1] * BOARD_SIZE + position[0]
    seen = {square}
    pending = [square]
    for square in pending:
        if square // BOARD_SIZE == target_row:
            return True
        for other in adjacent[square]:
            if other not in seen and (square, other) not in blocked:
                seen.add(other)
                pending.append(other)
    return False

def encode_relative_8ch_path_v1(state, active_pid):
    if active_pid not in (1, 2):
        raise ValueError(f"active_pid must be 1 or 2, got {active_pid}")
    # 8 Channels exactly as specified
    tensor = torch.zeros((8, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    
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
    
    return validate_state_tensor(tensor, "encoded state", shape=(8, BOARD_SIZE, BOARD_SIZE))


def encode_relative_9ch_path_moves_v2(state, active_pid):
    tensor = torch.cat((encode_relative_8ch_path_v1(state, active_pid),
                        torch.full((1, BOARD_SIZE, BOARD_SIZE),
                                   state.get('turns_played', 0) / 200.0,
                                   dtype=torch.float32)))
    return validate_state_tensor(tensor, "encoded state")


def encode_state(state, active_pid):
    return encode_relative_9ch_path_moves_v2(state, active_pid)


def encode_model_state_batch(model, states, active_pids):
    """Build the current encoder's sparse planes and constants in one batch."""
    if not states or len(states) != len(active_pids):
        raise ValueError("model states and active players must have matching nonempty lengths")
    if (
        (getattr(model, 'encode_game_state', None)
         and not getattr(model, '_uses_current_encoder', False))
        or tuple(getattr(model, 'input_shape', STATE_SHAPE)) != STATE_SHAPE
    ):
        return make_model_state_batch(model, [
            encode_state_for_model(model, state, pid)
            for state, pid in zip(states, active_pids)
        ], name="bot evaluation states")

    sample_ids, channels, rows, columns, constants = [], [], [], [], []
    for index, (state, pid) in enumerate(zip(states, active_pids)):
        if pid not in (1, 2):
            raise ValueError(f"active_pid must be 1 or 2, got {pid}")
        opponent = 3 - pid
        for channel, pos in enumerate((state[f'p{pid}_pos'], state[f'p{opponent}_pos'])):
            c, r = pos
            sample_ids.append(index)
            channels.append(channel)
            rows.append(8 - r if pid == 2 else r)
            columns.append(8 - c if pid == 2 else c)
        for channel, walls in ((2, state['walls_h']), (3, state['walls_v'])):
            for c, r in walls:
                sample_ids.append(index)
                channels.append(channel)
                rows.append(7 - r if pid == 2 else r)
                columns.append(7 - c if pid == 2 else c)
        constants.append([
            state[f'p{pid}_walls'] / 10.0,
            state[f'p{opponent}_walls'] / 10.0,
            get_shortest_path_dist(state, pid) / 81.0,
            get_shortest_path_dist(state, opponent) / 81.0,
            state.get('turns_played', 0) / 200.0,
        ])
    constant_batch = torch.tensor(constants, dtype=torch.float32).view(-1, 5, 1, 1)
    if not torch.isfinite(constant_batch).all().item():
        raise ValueError("encoded states contain NaN or infinite values")
    batch = torch.zeros((len(states), *STATE_SHAPE), dtype=torch.float32)
    batch[sample_ids, channels, rows, columns] = 1.0
    batch[:, 4:] = constant_batch
    return batch.to(device=model_device(model))


def _state_dict_schema(state_dict):
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in state_dict.items()
    }


def _cpu_state_dict(state_dict):
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def _encoder_implementation_source(channels=STATE_CHANNELS):
    source = textwrap.dedent(inspect.getsource(encode_relative_8ch_path_v1))
    if channels == 9:
        source += '\n' + textwrap.dedent(inspect.getsource(encode_relative_9ch_path_moves_v2))
    return source


def _current_encoder_spec(channels=STATE_CHANNELS):
    spec = copy.deepcopy(CURRENT_ENCODER_SPEC)
    if channels == 8:
        spec.update(id="relative_8ch_path_v1", shape=[8, BOARD_SIZE, BOARD_SIZE])
        spec['channels'] = spec['channels'][:8]
    source = _encoder_implementation_source(channels)
    spec["implementation_source"] = source
    spec["implementation_sha256"] = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()
    return spec


def _architecture_descriptor(architecture):
    descriptor = {
        "torchscript_code": architecture["torchscript_code"],
        "state_schema": architecture["state_schema"],
        "input_encoder": architecture["input_encoder"],
        "value_output": architecture["value_output"],
    }
    if "torchscript_modules" in architecture:
        descriptor["torchscript_modules"] = architecture["torchscript_modules"]
    if "policy_value_code" in architecture:
        descriptor["policy_value_code"] = architecture["policy_value_code"]
    return descriptor


def _scripted_module_codes(scripted):
    return {name: module.code for name, module in scripted.named_modules()}


def _architecture_id(descriptor):
    return hashlib.sha256(
        json.dumps(descriptor, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _model_architecture(model):
    archived = getattr(model, "archived_architecture", None)
    return copy.deepcopy(archived) if archived else capture_model_architecture(model)


def _model_state_dict(model):
    archived = getattr(model, "archived_state_dict", None)
    return archived() if archived else model.state_dict()


def capture_model_architecture(model):
    """Return a weight-independent, self-contained inference definition."""
    state_schema = _state_dict_schema(model.state_dict())
    encoder_spec = _current_encoder_spec(getattr(model, 'in_channels', STATE_CHANNELS))
    template = copy.deepcopy(model).cpu().eval()
    sample = torch.zeros(
        (1, *encoder_spec["shape"]), dtype=torch.float32
    )
    with torch.inference_mode():
        output = template(sample)
    if isinstance(output, torch.Tensor) and tuple(output.shape) == (1, 1):
        value_output = {"kind": "tensor"}
    elif isinstance(output, (tuple, list)):
        candidates = [
            index for index, value in enumerate(output)
            if isinstance(value, torch.Tensor) and tuple(value.shape) == (1, 1)
        ]
        if len(candidates) != 1:
            raise ValueError("model archive needs exactly one [B, 1] value output")
        value_output = {"kind": "sequence", "index": candidates[0]}
    else:
        raise ValueError("model archive cannot identify the value output")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if getattr(model, 'supports_policy', False):
            scripted = torch.jit.trace_module(
                template, {'forward': sample, 'policy_value': sample}, check_trace=False)
        else:
            scripted = torch.jit.trace(template, sample, check_trace=False)
    script_buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.jit.save(scripted, script_buffer)
    script_code = scripted.code
    descriptor = {
        "torchscript_code": script_code,
        "torchscript_modules": _scripted_module_codes(scripted),
        "state_schema": state_schema,
        "input_encoder": encoder_spec,
        "value_output": value_output,
    }
    if getattr(model, 'supports_policy', False):
        descriptor['policy_value_code'] = scripted.policy_value.code
    architecture_id = _architecture_id(descriptor)
    architecture = {
        "id": architecture_id,
        "class_name": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "class_source": (inspect.getsource(ResidualBlock) + '\n'
                         if any(isinstance(layer, ResidualBlock) for layer in model.modules())
                         else inspect.getsource(SpatialQuoriNetV2) + '\n'
                         if isinstance(model, LegacyPolicyValueQuoriNet) else '')
                        + inspect.getsource(model.__class__),
        "torchscript": script_buffer.getvalue(),
        **descriptor,
    }
    return architecture


class ArchivedValueNet(nn.Module):
    """Value opponent reconstructed entirely from one architecture archive."""

    def __init__(self, architecture, state_dict):
        super().__init__()
        self.archived_architecture = copy.deepcopy(architecture)
        self._archived_weights = _cpu_state_dict(state_dict)
        self.architecture_id = architecture["id"]
        self.encoder_spec = architecture["input_encoder"]
        self.input_shape = tuple(self.encoder_spec["shape"])
        self.value_output = architecture["value_output"]
        self.supports_policy = 'policy_value_code' in architecture
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.module = torch.jit.load(
                io.BytesIO(architecture["torchscript"]), map_location="cpu"
            )
        self.module.load_state_dict(state_dict, strict=True)
        self.module.eval()

        source = self.encoder_spec.get("implementation_source")
        if not source:
            raise ValueError("archived input encoder has no preserved implementation")
        self._uses_current_encoder = source == _encoder_implementation_source()
        namespace = dict(globals())
        # Old encoder sources refer to these globals; keep their saved contract.
        namespace.update(STATE_SHAPE=self.input_shape, STATE_CHANNELS=self.input_shape[0])
        namespace['validate_state_tensor'] = lambda tensor, name="state", shape=self.input_shape: validate_state_tensor(tensor, name, shape)
        exec(source, namespace)
        self._archived_encoder = namespace['encode_' + self.encoder_spec['id']]

    def archived_state_dict(self):
        return _cpu_state_dict(self._archived_weights)

    def encode_game_state(self, state, active_pid):
        encoder_id = self.encoder_spec.get("id")
        if encoder_id in ("relative_8ch_path_v1", "relative_9ch_path_moves_v2"):
            return self._archived_encoder(state, active_pid)
        raise ValueError(f"unsupported archived input encoder: {encoder_id}")

    def forward(self, states):
        output = self.module(states)
        if self.value_output["kind"] == "sequence":
            output = output[self.value_output["index"]]
        return output

    def policy_value(self, states):
        return self.module.policy_value(states)


def encode_state_for_model(model, state, active_pid):
    encoder = getattr(model, "encode_game_state", None)
    if encoder:
        return encoder(state, active_pid)
    if tuple(getattr(model, 'input_shape', STATE_SHAPE)) == (8, BOARD_SIZE, BOARD_SIZE):
        return encode_relative_8ch_path_v1(state, active_pid)
    return encode_state(state, active_pid)


def make_model_state_batch(model, state_tensors, name="states"):
    expected_shape = tuple(getattr(model, "input_shape", STATE_SHAPE))
    if not state_tensors:
        raise ValueError(f"{name} cannot be empty")
    for index, tensor in enumerate(state_tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}[{index}] must be a torch.Tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name}[{index}] must have shape {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name}[{index}] must be finite float32")
    return torch.stack(state_tensors).to(
        device=model_device(model), dtype=torch.float32
    ).contiguous()

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
        self.last_policy = None
        # Preserve depth controls: each level buys 64 leaf evaluations.
        self.mcts_simulations = max(1, max_depth) * 64

    def get_move_and_eval(self, state, epsilon=0.0):
        self.last_policy = None
        if getattr(self.nn_model, 'supports_policy', False):
            was_training = self.nn_model.training
            try:
                self.nn_model.eval()
                return self.mcts_move(state, epsilon)
            finally:
                self.nn_model.train(was_training)
        # Parameters can change between moves, but not during one search.
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

    def mcts_move(self, state, epsilon=0.0):
        """Rollout-free PUCT; each node stores value for its side to move."""
        predictions = {}  # Only this search: weights may change before the next.

        def node(position, prior=1.0, parent=None, action=None):
            return {'state': position, 'prior': prior, 'n': 0, 'w': 0.0,
                    'children': None, 'parent_state': parent, 'action': action}

        def materialize(current):
            if current['state'] is None:
                parent = current.pop('parent_state')
                current['state'] = self.apply_action(
                    parent, parent['turn'], current.pop('action'))
            return current['state']

        def expand(current):
            position = materialize(current)
            pid = position['turn']
            for winner, row in ((1, 0), (2, 8)):
                if position[f'p{winner}_pos'][1] == row:
                    return 1.0 if winner == pid else -1.0
            if position.get('turns_played', 0) >= 200:
                return 0.0
            actions = self.generate_actions(position, pid)
            winning = [a for a in actions if a[0] == 'move'
                       and a[1][1] == self.get_target_row(pid)]
            if winning:
                actions = winning
            elif any(m[1] == self.get_target_row(3 - pid)
                     for m in self.get_valid_moves(position, 3 - pid)):
                # Retain the existing immediate forced-defense safeguard.
                safe = [a for a in actions if not any(
                    m[1] == self.get_target_row(3 - pid)
                    for m in self.get_valid_moves(self.apply_action(position, pid, a), 3 - pid))]
                if safe:
                    actions = safe
            encoded = encode_state_for_model(self.nn_model, position, pid)
            batch = make_model_state_batch(self.nn_model, [encoded])
            # Key the actual input, including custom archived encoders. Cache
            # raw outputs, never legal masks, visits, terminal values or history.
            key = encoded.detach().cpu().contiguous().numpy().tobytes()
            if key not in predictions:
                with torch.inference_mode():
                    logits, value = self.nn_model.policy_value(batch)
                if (tuple(logits.shape) != (1, POLICY_SIZE)
                        or tuple(value.shape) != (1, 1)
                        or not torch.isfinite(logits).all()
                        or not torch.isfinite(value).all()):
                    raise FloatingPointError('invalid MCTS policy/value output')
                predictions[key] = (logits, value)
            logits, value = predictions[key]
            indices = [policy_action_index(a, pid) for a in actions]
            priors = torch.softmax(logits[0, indices], dim=0).cpu().tolist()
            current['children'] = {
                action: node(None, prior, position, action)
                for action, prior in zip(actions, priors)}
            return float(value[0, 0])

        if state['turn'] != self.bot_id:
            raise ValueError('MCTS bot must be the side to move')
        root = node(state)
        root_value = expand(root)
        if not root['children']:
            self.was_last_random = False
            return None, root_value * 10000.0
        explore = epsilon > 0 and not any(
            action[0] == 'move' and action[1][1] == self.get_target_row(self.bot_id)
            for action in root['children'])
        if explore:
            noise = torch.distributions.Dirichlet(
                torch.full((len(root['children']),), 0.3)).sample().tolist()
            for child, sample in zip(root['children'].values(), noise):
                child['prior'] = 0.75 * child['prior'] + 0.25 * sample
        for _ in range(self.mcts_simulations):
            current, path = root, [root]
            while current['children']:
                scale = math.sqrt(max(1, current['n']))
                current = max(current['children'].values(), key=lambda child:
                    (-child['w'] / child['n'] if child['n'] else 0.0)
                    + 1.5 * child['prior'] * scale / (1 + child['n']))
                path.append(current)
            value = expand(current)
            for visited in reversed(path):
                visited['n'] += 1
                visited['w'] += value
                value = -value
        children = root['children']
        self.last_policy = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        for action, child in children.items():
            self.last_policy[policy_action_index(action, self.bot_id)] = child['n']
        self.last_policy /= self.last_policy.sum()
        # Keep epsilon exploration and the UI marker; targets remain visits.
        self.was_last_random = explore and random.random() < epsilon
        if self.was_last_random:
            chosen = random.choice(list(children))
        else:
            most_visits = max(child['n'] for child in children.values())
            choices = [a for a, child in children.items() if child['n'] == most_visits]
            chosen = min(choices, key=lambda a:
                         self._oscillation_penalty(materialize(children[a]), self.bot_id))
        child = children[chosen]
        score = -child['w'] / child['n'] if child['n'] else root_value
        return chosen, score * 10000.0

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
            pos = next(
                nxt for nxt in self.get_adjacent(state, *pos)
                if distances[nxt[1] * BOARD_SIZE + nxt[0]] == distance - 1
            )
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
        adjacent, paths = _wall_path_witnesses(
            frozenset(state['walls_h']), frozenset(state['walls_v']),
            state['p1_pos'], state['p2_pos'])
        start = r * BOARD_SIZE + c
        offset, span = (BOARD_SIZE, 1) if orient == 'H' else (1, BOARD_SIZE)
        blocked = frozenset(
            edge for square in (start, start + span)
            for edge in ((square, square + offset), (square + offset, square)))
        for pid, target, path in zip((1, 2), (0, 8), paths):
            if path is None:
                return False
            if not path.isdisjoint(blocked) and not _reaches_goal_without_edges(
                    adjacent, state[f'p{pid}_pos'], target, blocked):
                return False
        return True

    def generate_actions(self, state, pid):
        valid_moves = self.get_valid_moves(state, pid)
        distances = get_distance_field(state, pid)
        valid_moves.sort(key=lambda move: distances[move[1] * BOARD_SIZE + move[0]])
        actions = [('move', move) for move in valid_moves]
        walls_left = state['p1_walls'] if pid == 1 else state['p2_walls']
        if walls_left <= 0: return actions

        # Every rules-legal wall reaches neural ranking, including walls that
        # do not touch a current shortest path or immediately delay a pawn.
        for c in range(8):
            for r in range(8):
                for orient in ('H', 'V'):
                    if self.is_valid_wall(state, c, r, orient):
                        actions.append(('wall', (c, r, orient)))
        return actions

    def rank_and_prune(self, state, actions, pid, top_k=TOP_MOVE_POOL_SIZE, return_states=False):
        if not actions: return []
        actions_copy = list(actions)
        random.shuffle(actions_copy)

        next_states = [self.apply_action(state, pid, action) for action in actions_copy]
        values = self.evaluate_batch(next_states, pid)
        opponent = self.get_opponent(pid)
        for index, candidate in enumerate(next_states):
            if candidate[f'p{pid}_pos'][1] == self.get_target_row(pid):
                continue
            if any(
                move[1] == self.get_target_row(opponent)
                for move in self.get_valid_moves(candidate, opponent)
            ):
                values[index] = -10000.0
        scored_actions = list(zip(values, actions_copy, next_states))

        scored_actions.sort(key=lambda x: x[0], reverse=True)
        if return_states:
            return [(act, candidate) for _, act, candidate in scored_actions[:top_k]]
        return [act for _, act, _ in scored_actions[:top_k]]

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
        """Score candidates in one forward pass using the side-to-move contract."""
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
            if d1 == 999 or d2 == 999:
                winner = 2 if d1 == 999 else 1
                scores.append(10000.0 if eval_pid == winner else -10000.0)
                continue

            penalty = self._oscillation_penalty(state, eval_pid)
            state_key = get_state_key(state)
            if state_key in state.get('history', set()):
                penalty += 8000.0

            if self.nn_model is None or not (state['p1_walls'] or state['p2_walls']):
                progress = (8 - state['p1_pos'][1]) - state['p2_pos'][1]
                heuristic = (d2 - d1) * 25.0 + progress * 5.0
                if eval_pid == 2:
                    heuristic = -heuristic
                scores.append((2.0 if self.nn_model is not None else 1.0) * heuristic - penalty)
                continue

            # Training encodes the active player. Convert that value back to
            # the caller's perspective instead of feeding an off-contract view.
            scores.append(-penalty)
            active_pid = state['turn']
            key = (
                state_key,
                state['p1_walls'],
                state['p2_walls'],
                state.get('turns_played', 0),
                active_pid
            )
            sign = 1.0 if active_pid == eval_pid else -1.0
            neural_requests.append((index, key, sign))
            if key not in self._value_cache and key not in missing:
                missing[key] = state

        if missing:
            batch = encode_model_state_batch(self.nn_model, list(missing.values()),
                                            [key[-1] for key in missing])
            with torch.inference_mode():
                output = self.nn_model(batch)
            if tuple(output.shape) != (len(missing), 1):
                raise ValueError(
                    f"bot network must return [B, 1], got {tuple(output.shape)}"
                )
            if not torch.isfinite(output).all().item():
                raise ValueError("bot network returned NaN or infinite values")
            self._value_cache.update(zip(missing, output[:, 0].cpu().tolist()))

        for index, key, sign in neural_requests:
            scores[index] += 100.0 * sign * self._value_cache[key]
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
        children = self.rank_and_prune(state, raw_actions, pid, top_k=top_k, return_states=True)
        if not children: return None, self.evaluate(state, self.bot_id)

        if maximizing:
            max_eval = float('-inf')
            best_action = None
            for action, new_state in children:
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
            for action, new_state in children:
                _, eval_score = self.minimax(new_state, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return best_action, min_eval

SCALE = 1.0
BG_COLOR, BOARD_BG = (245, 245, 245), (225, 225, 225)
SIDEBAR_BG, CELL_COLOR = (235, 235, 235), (255, 255, 255)
P1_COLOR, P2_COLOR = (210, 55, 65), (55, 105, 190)
WALL_COLOR, TEXT_COLOR, MUTED_TEXT = (70, 70, 70), (25, 25, 25), (100, 100, 100)
ORANGE_COLOR = (230, 125, 20)

CELL_SIZE = 42
WALL_THICKNESS = 8
PADDING = 16
BOARD_PIXEL_SIZE = BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * WALL_THICKNESS + PADDING * 2
SIDEBAR_WIDTH = 360
LEFT_SIDEBAR_WIDTH = 330
WIDTH = BOARD_PIXEL_SIZE + SIDEBAR_WIDTH + LEFT_SIDEBAR_WIDTH
HEIGHT = max(650, BOARD_PIXEL_SIZE + 80)

class Quoridor:
    def __init__(self, headless=False, train_depth=2, play_depth=4, train_updates=None):
        pygame.init()
        self.headless = headless
        self.train_depth = train_depth
        self.play_depth = play_depth
        self.is_fullscreen = False
        
        if not self.headless:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE | pygame.SRCALPHA)
            pygame.display.set_caption("QuoriNet Trainer [F11: Fullscreen, C: Save]")
            self.virtual_screen = pygame.Surface((WIDTH, HEIGHT))
            
            self.font = pygame.font.SysFont("sans-serif", int(24 * SCALE), bold=True)
            self.small_font = pygame.font.SysFont("sans-serif", int(16 * SCALE), bold=True)
            self.tiny_font = pygame.font.SysFont("sans-serif", int(13 * SCALE), bold=False)
            self.title_font = pygame.font.SysFont("sans-serif", int(38 * SCALE), bold=True)
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
        self.LOG_WINDOW_SIZE = 50
        self.MIN_SAMPLES_TRUSTED = 15
        self.GATE_MIN_GAMES = 30
        self.GATE_WIN_RATE_THRESHOLD = 0.57
        self.GATE_TREND_WINDOW_SIZE = 25
        self.FINAL_TEST_GAMES_PER_MODEL = 10
        self.ESCALATION_LOSS_THRESHOLD = 0.50
        self.ESCALATION_RECOVER_THRESHOLD = 0.65
        self.ESCALATION_STEP_MULTIPLIER = 1.5
        self.ESCALATION_MAX_MULTIPLIER = 8.0
        self.ESCALATION_STALL_LIMIT = 5
        self.ESCALATION_MAX_SHARE_OF_HISTORIC_BUCKET = 0.15

        # Performance curriculum. A uniform floor prevents total forgetting,
        # and the remaining mass follows lower win rates even above the gate.
        self.LATEST_TEACHER_LOG_WINDOW = 60
        self.CURRICULUM_UNIFORM_MASS = 0.25

        # Fresh positions get exactly one pass; replay never adds extra batches.
        self.REPLAY_GAMES_PER_TEACHER = 100
        self.REPLAY_BATCH_SIZE = 128
        self.GENERAL_REPLAY_SLOTS = 38
        self.TACTICAL_LEAD_PLIES = 8
        self.TACTICAL_MIN_LEAD_PLIES = 4
        self.TACTICAL_LOSS_WEIGHT = 3.0
        if train_updates is not None:
            warnings.warn('train_updates is ignored: every fresh position receives one pass.')

        # Used only by the isolated one-game overfit diagnostic.
        self.OUTCOME_BACKFILL_DISCOUNT = 0.995

        # Explicit LR bounds. Legacy constants remain for the retired-policy
        # diagnostic; production uses fresh, seat-balanced decline evidence.
        self.LR_TUNE_INTERVAL_LATEST_GAMES = 15
        self.LR_MIN_SAMPLES = 15
        self.LR_PLATEAU_EPS = 0.015
        self.LR_REGRESSION_EPS = 0.03
        self.LR_UP_FACTOR = 1.25
        self.LR_DOWN_FACTOR = 0.80
        self.LR_CONVERGE_FACTOR = 0.85
        self.LR_BLOCK_PER_SEAT = 16
        self.LR_DECLINE_BLOCKS = 3
        self.LR_COOLDOWN_BLOCKS = 3
        self.LR_MIN_UPDATE_STEPS = 64
        self.GRAD_CLIP_NORM = 1.0

        # State vars
        self.student_net = PolicyValueQuoriNet(in_channels=STATE_CHANNELS)
        self.teacher_net = PolicyValueQuoriNet(in_channels=STATE_CHANNELS)
        self.teacher_net.load_state_dict(self.student_net.state_dict())
        self.student_net.eval()
        self.teacher_net.eval()
        self.optimizer = optim.Adam(self.student_net.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss(reduction='none')

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

        self.replay_games = {}
        self.next_replay_game_id = 1
        self.last_game_batches = []
        self.episode_trajectory = []
        self.last_loss = 0.0
        self.optimizer_steps = 0
        self.replay_samples_latest = 0
        self.replay_samples_historic = 0
        self.replay_samples_tactical = 0
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
        self.rematch_teacher_gen = None
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

        self._reset_lr_inertia()

        self.is_evaluating = False
        self.eval_vs_gen = 0
        self.eval_game_curr = 0
        self.eval_game_total = 0
        self._manual_promotion_test_requested = False

        self._archive_groups = {}
        self._archived_official = {}
        self._archived_manual = {}
        self._checkpoint_dir = "."
        self._checkpoint_source_dir = None
        self._loaded_student_architecture = None
        self._loaded_student_architecture_id = None
        self._checkpoint_needs_initial_save = False
        self.architecture_transition_status = "same_architecture"
        self._discover_model_archives()
        self.load_checkpoint_if_exists()
        self._sync_model_archives()
        self.update_opponent_pool()

        self.mode = "MENU"
        self.is_training_active = True
        self.toast_msg = ""
        self.toast_timer = 0
        
        (
            self.rect_train, self.rect_save_manual, self.rect_new_game,
            self.rect_menu, self.rect_train_toggle, self.rect_test_promotion,
            self.rect_lr_up, self.rect_lr_down,
        ) = [pygame.Rect(0, 0, 0, 0) for _ in range(8)]
        self.manual_snapshots = self._find_manual_snapshots()
        self.manual_snapshot_index = max(0, len(self.manual_snapshots) - 1)
        self.game_start_time = time.time()
        self.reset_game()
        if hasattr(self, '_restored_rng'):
            python_rng, torch_rng, cuda_rng = self._restored_rng
            random.setstate(python_rng)
            torch.set_rng_state(torch_rng)
            if torch.cuda.is_available() and cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
            del self._restored_rng
        if self._checkpoint_needs_initial_save:
            self.save_checkpoint(manual=True)


    # ==========================================
    # SECTION 4: OPPONENT POOL
    # ==========================================
    def _validate_model_archive_payload(self, payload):
        if (
            payload.get("format") != MODEL_ARCHIVE_FORMAT
            or payload.get("format_version") != MODEL_ARCHIVE_VERSION
        ):
            raise ValueError("unsupported model archive format")
        architecture = copy.deepcopy(payload.get("architecture", {}))
        payload = dict(payload, architecture=architecture)
        required = {
            "id", "class_name", "class_source", "torchscript",
            "torchscript_code", "state_schema", "input_encoder",
            "value_output",
        }
        if not required.issubset(architecture):
            raise ValueError("model archive is missing its architecture definition")
        descriptor = _architecture_descriptor(architecture)
        if architecture["id"] != _architecture_id(descriptor):
            raise ValueError("model archive architecture fingerprint changed")
        # Older fingerprints omitted child modules. Recover their exact code
        # from the saved executable, never from the current network class.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scripted = torch.jit.load(
                io.BytesIO(architecture["torchscript"]), map_location="cpu"
            )
        module_codes = _scripted_module_codes(scripted)
        if ('policy_value_code' in architecture
                and architecture['policy_value_code'] != scripted.policy_value.code):
            raise ValueError('model archive policy executable differs from its architecture')
        if architecture.get("torchscript_modules", module_codes) != module_codes:
            raise ValueError("model archive executable differs from its architecture")
        architecture["torchscript_modules"] = module_codes
        encoder = architecture["input_encoder"]
        if "implementation_source" not in encoder:
            # Upgrade the immediately preceding archive format while its exact
            # encoder implementation is still the live implementation.
            encoder = copy.deepcopy(encoder)
            source = _encoder_implementation_source()
            encoder["implementation_source"] = source
            encoder["implementation_sha256"] = hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest()
            architecture["input_encoder"] = encoder
        architecture["id"] = _architecture_id(
            _architecture_descriptor(architecture)
        )
        source = encoder["implementation_source"]
        if encoder.get("implementation_sha256") != hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest():
            raise ValueError("model archive encoder fingerprint changed")
        state_dicts = list(payload.get("official", {}).values())
        state_dicts.extend(
            entry["weights"] for entry in payload.get("manual", {}).values()
        )
        if any(
            _state_dict_schema(state_dict) != architecture["state_schema"]
            for state_dict in state_dicts
        ):
            raise ValueError("model archive contains weights for another architecture")
        return payload

    def _discover_model_archives(self):
        """Read existing folders before any checkpoint tensors are loaded."""
        groups = {}
        for filename in os.listdir('.'):
            if not MODEL_ARCHIVE_PATTERN.fullmatch(filename) or not os.path.isdir(filename):
                continue
            payload = self._read_model_archive(filename)
            architecture = payload["architecture"]
            group = groups.setdefault(architecture["id"], {
                "architecture": architecture,
                "official": {}, "manual": {}, "paths": set(),
            })
            group["official"].update(payload.get("official", {}))
            group["manual"].update(payload.get("manual", {}))
            group["paths"].add(filename)
            if not group["official"] and not group["manual"]:
                generation_range = payload.get("generation_range") or []
                if generation_range:
                    group["checkpoint_generation"] = int(generation_range[-1])
            group["active_checkpoint"] = (
                group.get("active_checkpoint", False)
                or payload.get("active_checkpoint", False)
            )
            if payload.get("active_checkpoint", False):
                group["active_path"] = filename
        self._archive_groups = groups
        self._rebuild_archive_indexes()

    def _read_packed_model_archive(self, filename):
        payload = torch.load(filename, map_location="cpu", weights_only=True)
        return self._validate_model_archive_payload(payload)

    def _read_model_archive(self, directory):
        manifest_path = os.path.join(directory, ARCHITECTURE_MANIFEST)
        source_path = os.path.join(directory, ARCHITECTURE_SOURCE)
        executable_path = os.path.join(directory, ARCHITECTURE_EXECUTABLE)
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        with open(source_path, encoding="utf-8") as f:
            class_source = f.read()
        with open(executable_path, "rb") as f:
            torchscript = f.read()

        architecture = dict(manifest["architecture"])
        architecture["class_source"] = class_source
        architecture["torchscript"] = torchscript
        official = {}
        manual = {}
        for filename in os.listdir(directory):
            official_match = ARCHIVED_OFFICIAL_PATTERN.fullmatch(filename)
            manual_match = ARCHIVED_MANUAL_PATTERN.fullmatch(filename)
            if not official_match and not manual_match:
                continue
            path = os.path.join(directory, filename)
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            if official_match:
                official[int(official_match.group(1))] = state_dict
            else:
                generation = int(manual_match.group(1))
                iteration = int(manual_match.group(2))
                name = f"gen_{generation}_manual_{iteration}"
                manual[name] = {
                    "generation": generation,
                    "iteration": iteration,
                    "weights": state_dict,
                }
        payload = {
            "format": manifest.get("format"),
            "format_version": manifest.get("format_version"),
            "title": manifest.get("title"),
            "generation_range": manifest.get("generation_range"),
            "active_checkpoint": bool(manifest.get("active_checkpoint", False)),
            "architecture": architecture,
            "official": official,
            "manual": manual,
        }
        return self._validate_model_archive_payload(payload)

    def _write_model_archives(self):
        ranges = {}
        for architecture_id, group in self._archive_groups.items():
            generations = list(group["official"])
            generations.extend(
                entry["generation"] for entry in group["manual"].values()
            )
            if group.get("checkpoint_generation") is not None:
                generations.append(group["checkpoint_generation"])
            if not generations:
                continue
            ranges[architecture_id] = (min(generations), max(generations))
        range_counts = {
            generation_range: list(ranges.values()).count(generation_range)
            for generation_range in set(ranges.values())
        }

        for architecture_id, group in self._archive_groups.items():
            if architecture_id not in ranges:
                continue
            first_gen, last_gen = ranges[architecture_id]
            suffix = (
                f" {architecture_id[:8]}"
                if range_counts[(first_gen, last_gen)] > 1 else ""
            )
            directory = f"gen {first_gen}-{last_gen}{suffix}"
            old_directories = {
                path for path in group.get("paths", set())
                if os.path.isdir(path)
            }
            if not os.path.isdir(directory):
                source_directory = next(iter(old_directories), None)
                if source_directory is not None:
                    os.replace(source_directory, directory)
                    old_directories.remove(source_directory)
                else:
                    os.makedirs(directory)

            architecture = group["architecture"]
            manifest_architecture = {
                key: value for key, value in architecture.items()
                if key not in ("class_source", "torchscript")
            }
            manifest_architecture.update({
                "source_file": ARCHITECTURE_SOURCE,
                "executable_file": ARCHITECTURE_EXECUTABLE,
            })
            manifest = {
                "format": MODEL_ARCHIVE_FORMAT,
                "format_version": MODEL_ARCHIVE_VERSION,
                "title": f"gen {first_gen}-{last_gen}",
                "generation_range": [first_gen, last_gen],
                "active_checkpoint": (
                    architecture_id == getattr(self, "_active_architecture_id", None)
                ),
                "architecture": manifest_architecture,
                "official_generations": sorted(group["official"]),
                "manual_snapshots": sorted(group["manual"]),
            }

            def atomic_write(path, writer, binary=False):
                temporary = f"{path}.tmp"
                try:
                    mode = "wb" if binary else "w"
                    kwargs = {} if binary else {"encoding": "utf-8"}
                    with open(temporary, mode, **kwargs) as f:
                        writer(f)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.remove(temporary)

            atomic_write(
                os.path.join(directory, ARCHITECTURE_MANIFEST),
                lambda f: json.dump(manifest, f, indent=2, sort_keys=True),
            )
            atomic_write(
                os.path.join(directory, ARCHITECTURE_SOURCE),
                lambda f: f.write(architecture["class_source"]),
            )
            atomic_write(
                os.path.join(directory, ARCHITECTURE_EXECUTABLE),
                lambda f: f.write(architecture["torchscript"]),
                binary=True,
            )
            for generation, state_dict in group["official"].items():
                atomic_write(
                    os.path.join(directory, f"gen {generation}.pth"),
                    lambda f, state_dict=state_dict: torch.save(state_dict, f),
                    binary=True,
                )
            for entry in group["manual"].values():
                filename = (
                    f"gen {entry['generation']} manual {entry['iteration']}.pth"
                )
                atomic_write(
                    os.path.join(directory, filename),
                    lambda f, state_dict=entry["weights"]: torch.save(state_dict, f),
                    binary=True,
                )

            verified = self._read_model_archive(directory)
            if (
                verified["architecture"]["id"] != architecture_id
                or set(verified["official"]) != set(group["official"])
                or set(verified["manual"]) != set(group["manual"])
            ):
                raise RuntimeError(f"model folder {directory} failed verification")

            for old_directory in old_directories:
                if (
                    old_directory != directory
                    and MODEL_ARCHIVE_PATTERN.fullmatch(old_directory)
                    and os.path.isdir(old_directory)
                ):
                    shutil.rmtree(old_directory)
            group["paths"] = {directory}
            group["path"] = directory

    def _rebuild_archive_indexes(self):
        official = {}
        manual = {}
        for group in self._archive_groups.values():
            architecture = group["architecture"]
            for gen, state_dict in group["official"].items():
                gen = int(gen)
                if gen in official:
                    raise ValueError(f"generation {gen} exists in two archives")
                official[gen] = (architecture, state_dict)
            for name, entry in group["manual"].items():
                key = (int(entry["generation"]), int(entry["iteration"]))
                if key in manual:
                    raise ValueError(f"manual snapshot {name} exists in two archives")
                manual[key] = (name, architecture, entry["weights"])
        self._archived_official = official
        self._archived_manual = manual

    def _sync_model_archives(self):
        """Consolidate every known model by exact architecture at startup."""
        groups = {}
        cleanup_files = set()
        for filename in os.listdir('.'):
            if not MODEL_ARCHIVE_PATTERN.fullmatch(filename) or not os.path.isdir(filename):
                continue
            try:
                payload = self._read_model_archive(filename)
            except Exception as exc:
                raise RuntimeError(f"model archive {filename} rejected: {exc}") from exc
            architecture = payload["architecture"]
            group = groups.setdefault(architecture["id"], {
                "architecture": architecture,
                "official": {}, "manual": {}, "paths": set(),
            })
            group["official"].update({
                int(gen): state_dict
                for gen, state_dict in payload.get("official", {}).items()
            })
            group["manual"].update(payload.get("manual", {}))
            group["paths"].add(filename)
            if not group["official"] and not group["manual"]:
                generation_range = payload.get("generation_range") or []
                if generation_range:
                    group["checkpoint_generation"] = int(generation_range[-1])
            group["active_checkpoint"] = (
                group.get("active_checkpoint", False)
                or payload.get("active_checkpoint", False)
            )

        # Migrate the earlier packed archive format into the browsable folders.
        for filename in os.listdir('.'):
            if not PACKED_MODEL_ARCHIVE_PATTERN.fullmatch(filename):
                continue
            try:
                payload = self._read_packed_model_archive(filename)
            except Exception as exc:
                raise RuntimeError(f"packed model archive {filename} rejected: {exc}") from exc
            architecture = payload["architecture"]
            group = groups.setdefault(architecture["id"], {
                "architecture": architecture,
                "official": {}, "manual": {}, "paths": set(),
            })
            group["official"].update({
                int(gen): state_dict
                for gen, state_dict in payload.get("official", {}).items()
            })
            group["manual"].update(payload.get("manual", {}))
            cleanup_files.add(filename)

        current_architecture = _model_architecture(self.teacher_net)
        current_group = groups.setdefault(current_architecture["id"], {
            "architecture": current_architecture,
            "official": {}, "manual": {}, "paths": set(),
        })

        existing_official = {
            int(gen): architecture_id
            for architecture_id, group in groups.items()
            for gen in group["official"]
        }
        existing_manual = {
            name: architecture_id
            for architecture_id, group in groups.items()
            for name in group["manual"]
        }

        def matching_group(state_dict, preferred_id=None):
            schema = _state_dict_schema(state_dict)
            if preferred_id is not None:
                preferred = groups.get(preferred_id)
                if preferred and preferred["architecture"]["state_schema"] == schema:
                    return preferred
            matches = [
                group for group in groups.values()
                if group["architecture"]["state_schema"] == schema
            ]
            if len(matches) == 1:
                return matches[0]
            if current_group in matches:
                return current_group
            return None

        for filename in os.listdir('.'):
            official_match = OFFICIAL_WEIGHTS_PATTERN.fullmatch(filename)
            manual_match = MANUAL_WEIGHTS_PATTERN.fullmatch(filename)
            if not official_match and not manual_match:
                continue
            try:
                state_dict = torch.load(
                    filename, map_location="cpu", weights_only=True
                )
            except Exception as exc:
                print(f"Skipped legacy weights {filename}: {exc}")
                continue

            if official_match:
                gen = int(official_match.group(1))
                group = matching_group(state_dict, existing_official.get(gen))
                if group is None:
                    print(f"Skipped architecture-ambiguous weights {filename}")
                    continue
                group["official"][gen] = _cpu_state_dict(state_dict)
                cleanup_files.add(filename)
            else:
                generation = int(manual_match.group(1))
                iteration = int(manual_match.group(2))
                name = f"gen_{generation}_manual_{iteration}"
                group = matching_group(state_dict, existing_manual.get(name))
                if group is None:
                    print(f"Skipped architecture-ambiguous weights {filename}")
                    continue
                group["manual"][name] = {
                    "generation": generation,
                    "iteration": iteration,
                    "weights": _cpu_state_dict(state_dict),
                }
                cleanup_files.add(filename)

        # The restored bundle is authoritative for the current official teacher.
        current_group["official"][self.teacher_gen] = _cpu_state_dict(
            _model_state_dict(self.teacher_net)
        )
        if self._loaded_student_architecture is not None:
            loaded_group = groups.setdefault(
                self._loaded_student_architecture_id,
                {
                    "architecture": self._loaded_student_architecture,
                    "official": {}, "manual": {}, "paths": set(),
                },
            )
            if not loaded_group["official"] and not loaded_group["manual"]:
                loaded_group["checkpoint_generation"] = self.teacher_gen + 1
        active_architecture = capture_model_architecture(self.student_net)
        active_group = groups.setdefault(active_architecture["id"], {
            "architecture": active_architecture,
            "official": {}, "manual": {}, "paths": set(),
        })
        self._active_architecture_id = active_architecture["id"]
        if not active_group["official"] and not active_group["manual"]:
            active_group["checkpoint_generation"] = self.teacher_gen + 1
        self._archive_groups = groups
        self._write_model_archives()
        self._rebuild_archive_indexes()
        self._checkpoint_dir = self._archive_groups[
            self._active_architecture_id
        ]["path"]
        self._migrate_loaded_checkpoint()
        for filename in cleanup_files:
            if (
                os.path.isfile(filename)
                and (
                    OFFICIAL_WEIGHTS_PATTERN.fullmatch(filename)
                    or MANUAL_WEIGHTS_PATTERN.fullmatch(filename)
                    or PACKED_MODEL_ARCHIVE_PATTERN.fullmatch(filename)
                )
            ):
                os.remove(filename)

    def _available_official_generations(self):
        generations = set(self._archived_official)
        for filename in os.listdir('.'):
            match = OFFICIAL_WEIGHTS_PATTERN.fullmatch(filename)
            if match:
                generations.add(int(match.group(1)))
        return sorted(generations)

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

        available_generations = set(self._available_official_generations())
        historic_gens = [
            g for g in range(1, recent_start) if g in available_generations
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
        """Require enough games and the WR target; trend is informational only."""
        log = self.checkpoint_log.get(gen)
        if not log or len(log) < self.GATE_MIN_GAMES:
            return False
        return self._passes_generation_gate(gen, sum(log) / len(log))

    def _all_training_gates_met(self):
        generations = self._training_generations()
        if not generations:
            return False
        # Keep the latest-teacher-only opener's normal qualification window.
        if not self._passes_training_generation_gate(self.teacher_gen):
            return False
        for gen in generations:
            log = self.checkpoint_log.get(gen)
            if not log or not self._passes_generation_gate(
                gen, sum(log) / len(log)
            ):
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
                    direction = (
                        "PERFECT" if recent == 1.0 else
                        "RISING" if recent > prior else "NOT RISING"
                    )
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
        """Prioritize lower win rates, independent of history length."""
        if not self.opponent_pool:
            self.opponent_weights = []
            return

        count = len(self.opponent_pool)
        uniform_weight = self.CURRICULUM_UNIFORM_MASS / count
        deficit_mass = 1.0 - self.CURRICULUM_UNIFORM_MASS
        deficit_scores = []
        win_rates = [
            self._generation_win_rate(gen) for _, gen in self.opponent_pool
        ]
        performance_target = max(
            (win_rate for win_rate in win_rates if win_rate is not None),
            default=0.0,
        )

        for win_rate in win_rates:
            if win_rate is not None:
                performance_deficit = max(
                    0.0,
                    (performance_target - win_rate) / max(performance_target, 1e-9)
                )
            else:
                performance_deficit = 1.0
            deficit_scores.append(performance_deficit)

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

        if self.rematch_teacher_gen is not None:
            for opponent, gen in self.opponent_pool:
                if gen == self.rematch_teacher_gen:
                    return opponent, gen
            raise RuntimeError(f"Rematch teacher Gen #{self.rematch_teacher_gen} is unavailable")

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

        if not self._passes_training_generation_gate(self.teacher_gen):
            return self.opponent_pool[latest_idx]

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

    def _tactical_replay_indices(self, winner_id):
        """Find the final lasting race advantage, then tag only its lead-up.

        Goal-arrival plies are 2 * distance - 1 for the side to move and
        2 * distance for the other side. A positive loser-minus-winner gap
        includes a one-ply edge without oscillating as both pawns advance.
        This is a retrospective path heuristic, not a forced-win claim.
        """
        if winner_id not in (1, 2) or not self.episode_trajectory:
            return set()
        positions = [(entry[0], entry[1]) for entry in self.episode_trajectory]
        positions.append(self.episode_trajectory[-1][2:4])
        advantages = []
        for state, active_pid in positions:
            active_distance = round(float(state[6, 0, 0]) * 81)
            other_distance = round(float(state[7, 0, 0]) * 81)
            active_arrival = max(0, 2 * active_distance - 1)
            other_arrival = 2 * other_distance
            gap = other_arrival - active_arrival
            advantages.append(gap > 0 if active_pid == winner_id else gap < 0)
        if not advantages[-1]:
            return set()
        onset = len(advantages) - 1
        while onset > 0 and advantages[onset - 1]:
            onset -= 1
        # No observed onset, fewer than four lead plies, or terminal-only edge.
        if onset < self.TACTICAL_MIN_LEAD_PLIES or onset >= len(self.episode_trajectory):
            return set()
        return set(range(max(0, onset - self.TACTICAL_LEAD_PLIES), onset))

    def _training_batches(self, fresh, opponent_gen):
        """Uniform position draws, with no replacement across either replay draw."""
        general_pool = [sample for games in self.replay_games.values()
                        for game in games for sample in game['positions']]
        same_pool = [sample for sample in general_pool
                     if sample['teacher_gen'] == opponent_gen]
        fresh = list(fresh)
        random.shuffle(fresh)
        fresh_slots = self.REPLAY_BATCH_SIZE - self.GENERAL_REPLAY_SLOTS
        for start in range(0, len(fresh), fresh_slots):
            new = fresh[start:start + fresh_slots]
            same = random.sample(same_pool, min(fresh_slots - len(new), len(same_pool)))
            selected = {(sample['game_id'], sample['ply_index']) for sample in same}
            available = ([sample for sample in general_pool
                          if (sample['game_id'], sample['ply_index']) not in selected]
                         if selected else general_pool)
            general = random.sample(available, min(self.GENERAL_REPLAY_SLOTS, len(available)))
            batch = new + same + general
            random.shuffle(batch)
            replay = same + general
            counts = {
                'batch_size': len(batch), 'fresh': len(new),
                'same_teacher': len(same), 'general': len(general),
                'fresh_tactical': sum(s['tactical'] for s in new),
                'same_teacher_tactical': sum(s['tactical'] for s in same),
                'general_tactical': sum(s['tactical'] for s in general),
                'replay_latest': sum(s['teacher_gen'] == self.teacher_gen for s in replay),
                'replay_historic': sum(s['teacher_gen'] != self.teacher_gen for s in replay),
            }
            yield batch, counts

    def _optimize_student_batch(self, batch):
        """Run one checked, normalized, weighted MSE update."""
        if not batch:
            return False
        self.student_net.train()
        try:
            device = model_device(self.student_net)
            states = make_state_batch([s['state'] for s in batch], device=device,
                                      name="training states")
            targets = make_value_batch([s['target'] for s in batch], device=device,
                                       name="training targets")
            weights = make_value_batch(
                [self.TACTICAL_LOSS_WEIGHT if s['tactical'] else 1.0 for s in batch],
                device=device, name="training weights")
            reflect = torch.rand(states.shape[0], device=device) < 0.5
            states[reflect] = mirror_state_batch(states[reflect])
            self.optimizer.zero_grad(set_to_none=True)
            has_policy = getattr(self.student_net, 'supports_policy', False)
            if has_policy:
                logits, predictions = self.student_net.policy_value(states)
            else:
                predictions = self.student_net(states)
            loss = checked_value_loss(self.loss_fn, predictions, targets,
                                      name="student weighted MSE", sample_weights=weights)
            policy_rows = [i for i, sample in enumerate(batch)
                           if sample.get('policy') is not None]
            if has_policy and policy_rows:
                policy = torch.stack([batch[i]['policy'] for i in policy_rows]).to(device)
                if (tuple(policy.shape) != (len(policy_rows), POLICY_SIZE)
                        or not torch.isfinite(policy).all() or (policy < 0).any()
                        or not torch.allclose(policy.sum(1), torch.ones(len(policy_rows), device=device))):
                    raise ValueError('invalid search visit targets')
                policy[reflect[policy_rows]] = mirror_policy_batch(policy[reflect[policy_rows]])
                policy_loss = -(policy * torch.log_softmax(logits[policy_rows], dim=1)).sum(1)
                policy_weights = weights[policy_rows, 0]
                loss = loss + (policy_loss * policy_weights).sum() / policy_weights.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError('non-finite joint policy/value loss')
            with torch.no_grad():
                weighted_errors = weights * self.loss_fn(predictions, targets)
                total_error = weighted_errors.sum().item()
                tactical_error = weighted_errors[weights > 1].sum().item()
                self._last_tactical_loss_share = (
                    tactical_error / total_error if total_error else 0.0)
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.student_net.parameters(), self.GRAD_CLIP_NORM,
                    error_if_nonfinite=True)
            except RuntimeError as error:
                if "non-finite" not in str(error):
                    raise
                for name, parameter in self.student_net.named_parameters():
                    if (parameter.grad is not None
                            and not torch.isfinite(parameter.grad).all().item()):
                        raise FloatingPointError(f"non-finite gradient in {name}")
                raise FloatingPointError("student gradient norm is NaN or infinite")
            self.optimizer.step()
            self.optimizer_steps += 1
            self.last_loss = loss.item()
            return True
        finally:
            self.student_net.eval()

    def _roll_replay_buffers_for_new_teacher(self):
        # Older windows retain every previously accepted game, including wins.
        self.replay_games.setdefault(
            self.teacher_gen, deque(maxlen=self.REPLAY_GAMES_PER_TEACHER))

    def _set_learning_rate(self, requested_lr, reason):
        if not math.isfinite(requested_lr) or requested_lr <= 0:
            raise ValueError('learning rate must be finite and positive')
        old_lr = self.optimizer.param_groups[0]['lr']
        new_lr = requested_lr
        if new_lr == old_lr:
            return False

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(f"LR: {old_lr:.6g} -> {new_lr:.6g} | {reason}")
        return True

    def _reset_lr_search_for_new_teacher(self):
        self.latest_teacher_games_seen = 0
        self.lr_training_games_seen = 0
        self.lr_metric_history.clear()
        self.lr_last_tuned_at_training_game = 0
        self.lr_best_metric = -1.0
        self.lr_best_value = self.optimizer.param_groups[0]['lr']
        self.lr_search_direction = 0
        self._reset_lr_inertia()

    def _adjust_lr_manually(self, factor):
        changed = self._set_learning_rate(
            self.optimizer.param_groups[0]['lr'] * factor, f'manual x{factor:g}')
        if changed:
            self._reset_lr_inertia()
            if self.student_eval_history:
                self.lr_inertia['best_combined_wr'] = (
                    sum(self.student_eval_history) / len(self.student_eval_history))
        self.toast_msg = f"LR: {self.optimizer.param_groups[0]['lr']:.8f}"
        if not changed:
            self.toast_msg += ' (unchanged)'
        self.toast_timer = 180

    def _reduce_lr_on_stagnation(self):
        """Call once after each decided training result updates combined WR."""
        if self.mode != 'TRAIN' or self.is_evaluating or not self.student_eval_history:
            return False
        state = self.lr_inertia
        rate = sum(self.student_eval_history) / len(self.student_eval_history)
        if state['best_combined_wr'] is None or rate > state['best_combined_wr']:
            state.update(best_combined_wr=rate, stagnant_games=0)
            return False
        state['stagnant_games'] += 1
        if state['stagnant_games'] <= 100:
            return False
        changed = self._set_learning_rate(
            self.optimizer.param_groups[0]['lr'] * 0.1,
            'combined WR has not improved for over 100 games')
        # Give the new LR a fresh interval; do not stack an inertia cut on it.
        self._reset_lr_inertia()
        self.lr_inertia.update(best_combined_wr=rate,
                              status='stagnation reduced' if changed else 'LR unchanged')
        return True

    def _reset_lr_inertia(self):
        self.lr_inertia = {
            'version': 1, 'teacher_gen': self.teacher_gen,
            'train_depth': self.train_depth, 'wall_policy': 'all_legal_v1',
            'seats': [[], []], 'previous': None, 'slow': None,
            'velocity': 0.0, 'declines': 0, 'cooldown': 0,
            'last_step': self.optimizer_steps, 'observed_step': self.optimizer_steps,
            'blocks': 0,
            'best_combined_wr': None, 'stagnant_games': 0,
            'status': 'warming up',
        }

    def _restore_lr_inertia(self, saved):
        """Resume only evidence collected under this teacher/search policy."""
        self._reset_lr_inertia()
        if not isinstance(saved, dict) or any(
            saved.get(key) != self.lr_inertia[key]
            for key in ('version', 'teacher_gen', 'train_depth', 'wall_policy')
        ):
            return
        seats = saved.get('seats')
        valid = (isinstance(seats, list) and len(seats) == 2
                 and all(isinstance(seat, list) and len(seat) <= self.LR_BLOCK_PER_SEAT
                         and all(value in (0, 1) for value in seat) for seat in seats))
        for key in ('previous', 'slow'):
            values = saved.get(key)
            valid = valid and (values is None or (
                isinstance(values, list) and len(values) == 2
                and all(isinstance(v, (int, float)) and math.isfinite(v)
                        and 0 <= v <= 1 for v in values)))
        valid = valid and ((saved.get('previous') is None) == (saved.get('slow') is None))
        for key in ('declines', 'cooldown', 'last_step', 'observed_step', 'blocks'):
            valid = valid and isinstance(saved.get(key), int) and saved[key] >= 0
        velocity = saved.get('velocity')
        valid = valid and isinstance(velocity, (int, float)) and math.isfinite(velocity)
        best = saved.get('best_combined_wr')
        stagnant = saved.get('stagnant_games', 0)
        valid = valid and (best is None or (
            isinstance(best, (int, float)) and math.isfinite(best) and 0 <= best <= 1))
        valid = valid and isinstance(stagnant, int) and 0 <= stagnant <= 100
        if not valid or max(saved['last_step'], saved['observed_step']) > self.optimizer_steps:
            return
        self.lr_inertia.update(copy.deepcopy(saved))

    def _record_lr_result(self, winner_id):
        """Never mix seats, teachers, cutoffs, or frozen tests into LR evidence."""
        if (self.mode != 'TRAIN' or self.is_evaluating or winner_id not in (1, 2)
                or self.current_opponent_gen != self.teacher_gen):
            return
        if (self.lr_inertia['teacher_gen'] != self.teacher_gen
                or self.lr_inertia['train_depth'] != self.train_depth):
            self._reset_lr_inertia()
        seat = self.lr_inertia['seats'][self.student_id - 1]
        seat.append(int(winner_id == self.student_id))
        # Bound a seat that is overrepresented by adaptive teacher scheduling.
        del seat[:-self.LR_BLOCK_PER_SEAT]

    def _tune_learning_rate_from_pool(self, force=False):
        """Patient live backoff, not a claim of frozen evaluation improvement.

        Flat or rebounding performance keeps its LR. Only repeated declines
        on both seats can lower it; force never bypasses evidence or cooldown.
        """
        state = self.lr_inertia
        if any(len(seat) < self.LR_BLOCK_PER_SEAT for seat in state['seats']):
            return
        if self.optimizer_steps <= state['observed_step']:
            return
        state['observed_step'] = self.optimizer_steps
        seats = state['seats']
        state['seats'] = [[], []]  # Each result votes in exactly one block.
        rates = [sum(seat) / len(seat) for seat in seats]
        state['blocks'] += 1
        previous, slow = state['previous'], state['slow']
        state['previous'] = rates
        if previous is None:
            state['slow'] = rates
            state['status'] = 'baseline collected'
            return
        delta = sum(r - p for r, p in zip(rates, previous)) / 2
        state['velocity'] = 0.7 * state['velocity'] + 0.3 * delta
        # The slow reference supplies inertia; it does not justify a cut once
        # performance has flattened or begun recovering from a trough.
        state['slow'] = [0.8 * old + 0.2 * new for old, new in zip(slow, rates)]
        half = self.LR_BLOCK_PER_SEAT // 2
        rebound = (delta > 0 or any(r > p + 0.05 for r, p in zip(rates, previous))
                   or any((sum(seat[half:]) - sum(seat[:half])) / half >= 0.25
                          for seat in seats))
        if state['cooldown']:
            state['cooldown'] -= 1
            state['declines'] = 0
            state['status'] = 'cooldown'
            return
        declining = (not rebound and delta < -self.LR_REGRESSION_EPS
                     and state['velocity'] < -self.LR_PLATEAU_EPS
                     and all(rate < reference - 0.10 for rate, reference in zip(rates, slow)))
        state['declines'] = state['declines'] + 1 if declining else 0
        state['status'] = ('recovery protected' if rebound else
                           f"decline {state['declines']}/{self.LR_DECLINE_BLOCKS}" if declining
                           else 'holding')
        if (state['declines'] < self.LR_DECLINE_BLOCKS
                or self.optimizer_steps - state['last_step'] < self.LR_MIN_UPDATE_STEPS):
            return
        changed = self._set_learning_rate(
            self.optimizer.param_groups[0]['lr'] * self.LR_DOWN_FACTOR,
            f"{self.LR_DECLINE_BLOCKS} fresh balanced declines; inertia={state['velocity']:+.3f}")
        state.update(declines=0, cooldown=self.LR_COOLDOWN_BLOCKS,
                     last_step=self.optimizer_steps, slow=rates, velocity=0.0,
                     status='reduced; cooldown' if changed else 'LR unchanged')

    def _load_net_for_gen(self, gen):
        if gen == self.teacher_gen: return self.teacher_net
        archived = self._archived_official.get(gen)
        if archived is not None:
            architecture, state_dict = archived
            try:
                net = ArchivedValueNet(architecture, state_dict)
                net.eval()
                return net
            except Exception as e:
                print(f"Failed to decode archived pool model {gen}: {e}")
                return None
        path = f'quorinet_teacher_gen_{gen}.pth'
        if not os.path.exists(path): return None
        try:
            weights = torch.load(path, weights_only=True)
            net = network_from_weights(weights)
            net.load_state_dict(weights)
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
        print(f"Walls: all legal | LR: inertia ({self.lr_inertia['status']})")
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
        self.episode_policies = []
        self._pending_search_policy = None

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
            self.p1_gen_str = self.p2_gen_str = f"Gen #{self.teacher_gen} Official"

        self.p1_draw_x, self.p1_draw_y = self._get_center_pixels(self.state['p1_pos'])
        self.p2_draw_x, self.p2_draw_y = self._get_center_pixels(self.state['p2_pos'])

    def _load_network_checkpoint(self, path, destination, label):
        """Load strictly through a temporary model so a bad file cannot half-load."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        candidate = network_from_weights(state_dict)
        candidate.load_state_dict(state_dict, strict=True)
        validate_model_contract(candidate, f"{label} checkpoint")
        candidate.eval()
        return candidate

    def _find_manual_snapshots(self):
        return sorted(
            (generation, iteration, name)
            for (generation, iteration), (name, _, _) in self._archived_manual.items()
        )

    def save_manual_student(self):
        """Atomically preserve the current student without changing official files."""
        validate_model_contract(self.student_net, "student before manual save")
        candidate_gen = self.teacher_gen + 1
        iterations = [
            iteration for generation, iteration in self._archived_manual
            if generation == candidate_gen
        ]
        iteration = max(iterations, default=0) + 1
        name = f"gen_{candidate_gen}_manual_{iteration}"
        architecture = capture_model_architecture(self.student_net)
        group = self._archive_groups.setdefault(architecture["id"], {
            "architecture": architecture,
            "official": {}, "manual": {}, "paths": set(),
        })
        group["manual"][name] = {
            "generation": candidate_gen,
            "iteration": iteration,
            "weights": _cpu_state_dict(self.student_net.state_dict()),
        }
        self._write_model_archives()
        self._rebuild_archive_indexes()
        self.manual_snapshots = self._find_manual_snapshots()
        self.manual_snapshot_index = self.manual_snapshots.index(
            (candidate_gen, iteration, name)
        )
        self.toast_msg = f"Saved Gen #{candidate_gen} Manual #{iteration}"
        self.toast_timer = 180
        print(f"[Manual Student Saved] {name}")
        return name

    def _checkpoint_path(self, filename, directory=None):
        return os.path.join(directory or self._checkpoint_dir, filename)

    def _locate_checkpoint_source(self):
        if os.path.isfile(TRAINING_STATE_FILE):
            return "."
        active = []
        candidates = []
        for group in self._archive_groups.values():
            for directory in group.get("paths", set()):
                if not os.path.isfile(self._checkpoint_path(
                        TRAINING_STATE_FILE, directory)):
                    continue
                candidates.append(directory)
                if group.get("active_checkpoint", False):
                    active.append(directory)
        if len(active) > 1:
            raise RuntimeError("multiple architecture folders claim the active checkpoint")
        if active:
            return active[0]
        if len(candidates) > 1:
            raise RuntimeError("multiple unmarked architecture checkpoints need review")
        if candidates:
            return candidates[0]
        if any(os.path.isfile(filename) for filename in ACTIVE_CHECKPOINT_FILES[1:]):
            return "."
        return "."

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.digest()

    def _migrate_loaded_checkpoint(self):
        source = self._checkpoint_source_dir
        if source is None or os.path.abspath(source) == os.path.abspath(self._checkpoint_dir):
            return
        if self._loaded_student_architecture_id != self._active_architecture_id:
            if os.path.abspath(source) == os.path.abspath("."):
                source_group = self._archive_groups.get(
                    self._loaded_student_architecture_id
                )
                if source_group is None or "path" not in source_group:
                    raise RuntimeError(
                        "source checkpoint architecture has no destination folder"
                    )
                self._copy_checkpoint_files(source, source_group["path"])
            self._checkpoint_needs_initial_save = True
            return

        self._copy_checkpoint_files(source, self._checkpoint_dir)

    def _copy_checkpoint_files(self, source, destination_dir):
        if os.path.abspath(source) == os.path.abspath(destination_dir):
            return

        copied = []
        for filename in ACTIVE_CHECKPOINT_FILES:
            source_path = self._checkpoint_path(filename, source)
            if not os.path.isfile(source_path):
                continue
            destination = self._checkpoint_path(filename, destination_dir)
            temporary = destination + ".migrate.tmp"
            shutil.copy2(source_path, temporary)
            if self._file_sha256(source_path) != self._file_sha256(temporary):
                os.remove(temporary)
                raise RuntimeError(f"checkpoint migration verification failed: {filename}")
            os.replace(temporary, destination)
            copied.append((source_path, destination))

        if not any(os.path.basename(path) == TRAINING_STATE_FILE for path, _ in copied):
            return
        for source_path, destination in copied:
            if self._file_sha256(source_path) != self._file_sha256(destination):
                raise RuntimeError(
                    f"checkpoint migration changed after installation: {source_path}"
                )
        for source_path, _ in copied:
            os.remove(source_path)

    def load_checkpoint_if_exists(self):
        # One atomic bundle is authoritative over all legacy sidecar files.
        bundle = None
        self._checkpoint_source_dir = self._locate_checkpoint_source()
        bundle_path = self._checkpoint_path(
            TRAINING_STATE_FILE, self._checkpoint_source_dir
        )
        if os.path.exists(bundle_path):
            bundle = torch.load(bundle_path, map_location='cpu',
                                weights_only=True)
            if bundle.get('format_version') not in (1, 2):
                raise ValueError('unsupported training checkpoint format')
            current_architecture = capture_model_architecture(self.student_net)
            teacher_gen = int(bundle.get('meta', {}).get('teacher_gen', 1))
            teacher_architecture = bundle.get('teacher_architecture')
            student_architecture = bundle.get('student_architecture')
            for label, architecture, weights in (
                ('teacher', teacher_architecture, bundle['teacher']),
                ('student', student_architecture, bundle['student']),
            ):
                if architecture is not None:
                    checked = self._validate_model_archive_payload({
                        'format': MODEL_ARCHIVE_FORMAT,
                        'format_version': MODEL_ARCHIVE_VERSION,
                        'architecture': architecture,
                        'official': {0: weights},
                        'manual': {},
                    })
                    if label == 'teacher':
                        teacher_architecture = checked['architecture']
                    else:
                        student_architecture = checked['architecture']
            self._loaded_student_architecture = copy.deepcopy(student_architecture)
            self._loaded_student_architecture_id = (
                student_architecture.get('id')
                if student_architecture is not None else None
            )
            archived_teacher = self._archived_official.get(teacher_gen)
            if teacher_architecture is None and archived_teacher is not None:
                candidate, _ = archived_teacher
                if candidate['state_schema'] == _state_dict_schema(bundle['teacher']):
                    teacher_architecture = candidate
            if student_architecture is None and teacher_architecture is not None:
                if teacher_architecture['state_schema'] == _state_dict_schema(bundle['student']):
                    student_architecture = teacher_architecture

            teacher_matches_current = (
                teacher_architecture is not None
                and teacher_architecture['id'] == current_architecture['id']
                and _state_dict_schema(bundle['teacher']) == current_architecture['state_schema']
            )
            student_matches_current = (
                student_architecture is not None
                and student_architecture['id'] == current_architecture['id']
                and _state_dict_schema(bundle['student']) == current_architecture['state_schema']
            )
            same_architecture = teacher_matches_current and student_matches_current
            if teacher_matches_current:
                self.teacher_net.load_state_dict(bundle['teacher'], strict=True)
            else:
                if teacher_architecture is None:
                    raise RuntimeError(
                        'checkpoint architecture differs and its official teacher '
                        'has no verified architecture archive'
                    )
                if _state_dict_schema(bundle['teacher']) != teacher_architecture['state_schema']:
                    raise RuntimeError('checkpoint teacher tensor schema is not archived')
                self.teacher_net = ArchivedValueNet(
                    teacher_architecture, bundle['teacher']
                )
                self.teacher_net.eval()

            if student_matches_current:
                self.student_net.load_state_dict(bundle['student'], strict=True)
                self.optimizer.load_state_dict(bundle['optimizer'])
            else:
                self.architecture_transition_status = 'fresh_student_and_optimizer'

            if not teacher_matches_current:
                if student_matches_current:
                    self.architecture_transition_status = 'restored_student_and_optimizer'
                self.architecture_transition_status += '; archived_official_teacher'
            if not same_architecture:
                print(
                    'Architecture transition: preserved the official teacher and '
                    + self.architecture_transition_status
                    + ' student/optimizer state.'
                )

            replay_compatible = (
                student_architecture is not None
                and student_architecture['input_encoder'] == current_architecture['input_encoder']
            )
            replay_version = bundle.get('replay_format_version')
            if replay_version not in (None, REPLAY_FORMAT_VERSION):
                raise ValueError('unsupported replay format')
            if replay_version == REPLAY_FORMAT_VERSION and (same_architecture or replay_compatible):
                self.replay_games = {
                    int(gen): deque(games, maxlen=self.REPLAY_GAMES_PER_TEACHER)
                    for gen, games in bundle['replay_games'].items()
                }
                self.next_replay_game_id = bundle['next_replay_game_id']
            elif replay_version is None:
                print('Replay policy migration: clearing old replay; preserving model and training state.')
            else:
                self.architecture_transition_status += '; incompatible_replay_preserved_in_source_bundle'
                print('Architecture transition: old-encoder replay remains only in the source bundle.')
            self.last_loss = bundle['last_loss']
            self._restored_rng = (bundle['python_rng'], bundle['torch_rng'], bundle.get('cuda_rng'))
        elif os.path.exists(self._checkpoint_path(
                META_FILE, self._checkpoint_source_dir)):
            print('Legacy checkpoint: model/Adam files are authoritative; '
                  'JSON counters may be newer and replay is unavailable.')
        teacher_loaded = False
        teacher_path = self._checkpoint_path(
            TEACHER_WEIGHTS_FILE, self._checkpoint_source_dir
        )
        if bundle is None and os.path.exists(teacher_path):
            try:
                self.teacher_net = self._load_network_checkpoint(
                    teacher_path,
                    self.teacher_net,
                    'teacher network'
                )
                teacher_loaded = True
            except Exception as exc:
                raise RuntimeError(f"Teacher checkpoint rejected: {exc}") from exc

        student_path = self._checkpoint_path(
            STUDENT_WEIGHTS_FILE, self._checkpoint_source_dir
        )
        if bundle is None and os.path.exists(student_path):
            try:
                self.student_net = self._load_network_checkpoint(
                    student_path,
                    self.student_net,
                    'student network'
                )
            except Exception as exc:
                raise RuntimeError(f"Student checkpoint rejected: {exc}") from exc
        elif teacher_loaded:
            self.student_net.load_state_dict(self.teacher_net.state_dict(), strict=False)

        if bundle is None:
            self.optimizer = optim.Adam(self.student_net.parameters(), lr=0.0001)

        optimizer_path = self._checkpoint_path(
            OPTIMIZER_FILE, self._checkpoint_source_dir
        )
        if bundle is None and os.path.exists(optimizer_path):
            try:
                candidate_optimizer = optim.Adam(
                    self.student_net.parameters(),
                    lr=self.optimizer.param_groups[0]['lr']
                )
                restore_student_optimizer(candidate_optimizer,
                    torch.load(
                        optimizer_path,
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
        meta_path = self._checkpoint_path(META_FILE, self._checkpoint_source_dir)
        if bundle is not None or os.path.exists(meta_path):
            try:
                source = (io.StringIO(json.dumps(bundle['meta'])) if bundle is not None
                          else open(meta_path, 'r'))
                with source as f:
                    meta = json.load(f)
                    self.games_played = meta.get('games_played', 0)
                    self.teacher_gen = meta.get('teacher_gen', 1)
                    self.rematch_teacher_gen = meta.get('rematch_teacher_gen')
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
                    # Legacy checkpoints retained only 30 per-teacher results.
                    # The separate latest-teacher history still has the true
                    # preceding outcomes, so recover them once for Gen #teacher.
                    latest_log = self.checkpoint_log.get(self.teacher_gen)
                    if (meta.get('gate_log_window_size', 30) < self.LOG_WINDOW_SIZE
                            and latest_log is not None
                            and len(latest_log) == 30
                            and len(self.latest_teacher_history) >= self.LOG_WINDOW_SIZE
                            and list(self.latest_teacher_history)[-30:] == list(latest_log)):
                        self.checkpoint_log[self.teacher_gen] = deque(
                            list(self.latest_teacher_history)[-self.LOG_WINDOW_SIZE:],
                            maxlen=self.LOG_WINDOW_SIZE
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
                    self.replay_samples_tactical = meta.get('replay_samples_tactical', 0)

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

        # Per-game JSON can be newer than the saved weights. Resume controller
        # evidence only with the matching atomic model/Adam snapshot.
        self._restore_lr_inertia(
            bundle['meta'].get('lr_inertia')
            if bundle is not None and student_matches_current else None)

        if bundle is not None and not student_matches_current:
            self._reset_student_training_progress()
            self._checkpoint_needs_initial_save = True

        if bundle is not None and bundle.get('replay_format_version') is None:
            # Persist the migration immediately, without reserializing live
            # metadata/RNG or rewriting weights/Adam sidecars. All other bundle
            # fields are preserved exactly, including architecture archives.
            migrated = dict(bundle)
            for key in ('latest_replay', 'historic_replay', 'tactical_replay'):
                migrated.pop(key, None)
            migrated.update(replay_format_version=REPLAY_FORMAT_VERSION,
                            replay_games={}, next_replay_game_id=1)
            self._write_training_bundle(migrated, self._checkpoint_source_dir)

        if bundle is None:
            self._loaded_student_architecture = capture_model_architecture(
                self.student_net
            )
            self._loaded_student_architecture_id = (
                self._loaded_student_architecture["id"]
            )

    def _reset_student_training_progress(self):
        """A new architecture must earn every gate with its own training results."""
        self.optimizer = optim.Adam(self.student_net.parameters(), lr=0.0001)
        self.games_played = self.optimizer_steps = 0
        self.replay_games = {}
        self.next_replay_game_id = 1
        self.replay_samples_latest = self.replay_samples_historic = self.replay_samples_tactical = 0
        self.episode_trajectory = []
        self.episode_policies = []
        self._pending_search_policy = None
        self.last_game_batches = []
        self.last_loss = 0.0
        self.student_eval_history.clear()
        self.latest_teacher_history.clear()
        self.checkpoint_log.clear()
        self.escalation_state.clear()
        self.wawr_history.clear()
        self.match_history.clear()
        self.student_win_rate_str = "Pending live WR"
        self.rematch_teacher_gen = None
        self.curriculum_latest_credit = 1.0
        self.lr_last_reduced_at_gen = 0
        self.heuristic_lr_floor = self.heuristic_weight * 0.0001
        self._reset_lr_search_for_new_teacher()
        self.is_evaluating = False
        self.eval_vs_gen = self.eval_game_curr = self.eval_game_total = 0
        self._manual_promotion_test_requested = False
        self._save_requested = False
        if hasattr(self, '_restored_rng'):
            del self._restored_rng

    def _write_training_bundle(self, bundle, directory=None):
        target = self._checkpoint_path(TRAINING_STATE_FILE, directory)
        temporary = target + '.tmp'
        with open(temporary, 'wb') as f:
            torch.save(bundle, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)

    def _checkpoint_metadata(self):
        return {
                'games_played': self.games_played,
                'teacher_gen': self.teacher_gen,
                'rematch_teacher_gen': self.rematch_teacher_gen,
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
                'lr_inertia': copy.deepcopy(self.lr_inertia),
                'curriculum_latest_credit': self.curriculum_latest_credit,
                'heuristic_lr_floor': self.heuristic_lr_floor,
                'optimizer_steps': self.optimizer_steps,
                'replay_samples_latest': self.replay_samples_latest,
                'replay_samples_historic': self.replay_samples_historic,
                'replay_samples_tactical': self.replay_samples_tactical,
                'gate_log_window_size': self.LOG_WINDOW_SIZE
            }

    def save_meta(self):
        # Observational sidecar only; restoration uses the bundle's metadata.
        target = self._checkpoint_path(META_FILE)
        temporary = target + '.tmp'
        with open(temporary, 'w') as f:
            json.dump(self._checkpoint_metadata(), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)

    def save_checkpoint(self, manual=False):
        # A training episode owns deferred targets and updates. Save only after
        # it commits; the input thread requests a save instead of racing Adam.
        if self.episode_trajectory or (
                not self.game_over and getattr(self, 'mode', None) == 'TRAIN'):
            self._save_requested = True
            print('Checkpoint requested; saving after this game finishes.')
            return
        validate_model_contract(self.teacher_net, "teacher network before save")
        validate_model_contract(self.student_net, "student network before save")
        validate_optimizer_state_shapes(self.optimizer, "optimizer before save")
        bundle = {
            'format_version': 2,
            'teacher': _model_state_dict(self.teacher_net),
            'student': self.student_net.state_dict(),
            'teacher_architecture': _model_architecture(self.teacher_net),
            'student_architecture': capture_model_architecture(self.student_net),
            'optimizer': self.optimizer.state_dict(),
            'meta': self._checkpoint_metadata(),
            'replay_format_version': REPLAY_FORMAT_VERSION,
            'replay_games': {gen: list(games) for gen, games in self.replay_games.items()},
            'next_replay_game_id': self.next_replay_game_id,
            'last_loss': self.last_loss,
            'python_rng': random.getstate(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        self._write_training_bundle(bundle)
        self._save_requested = False
        # Compatibility exports. Interrupted exports cannot supersede the bundle.
        torch.save(_model_state_dict(self.teacher_net), self._checkpoint_path(
            f'quorinet_teacher_gen_{self.teacher_gen}.pth'))
        torch.save(_model_state_dict(self.teacher_net), self._checkpoint_path(
            TEACHER_WEIGHTS_FILE))
        torch.save(self.student_net.state_dict(), self._checkpoint_path(
            STUDENT_WEIGHTS_FILE))
        torch.save(self.optimizer.state_dict(), self._checkpoint_path(
            OPTIMIZER_FILE))
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
                if hasattr(self, '_trainer_commands'):
                    self._publish_trainer_view()
                    while not self._trainer_allowed.wait(0.05):
                        if self._trainer_stop.is_set():
                            raise InterruptedError('Trainer closed during evaluation')
                    if self._trainer_stop.is_set():
                        raise InterruptedError('Trainer closed during evaluation')
                elif not self.headless:
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

    def request_manual_promotion_test(self):
        if self._manual_promotion_test_requested:
            return
        self._manual_promotion_test_requested = True
        self.toast_msg = "Promotion test queued after this game"
        self.toast_timer = 180
        print("Manual promotion test queued after the current training game.")

    def _install_promoted_student(self):
        # Preserve the teacher that was actually trained against before the
        # live teacher network is overwritten.
        torch.save(
            _model_state_dict(self.teacher_net),
            self._checkpoint_path(f'quorinet_teacher_gen_{self.teacher_gen}.pth')
        )

        self.update_escalation_state()
        self.teacher_gen += 1
        self.teacher_net = copy.deepcopy(self.student_net)
        self.teacher_net.eval()

        self._roll_replay_buffers_for_new_teacher()
        self.latest_teacher_history.clear()
        self.checkpoint_log.clear()
        self._reset_lr_search_for_new_teacher()
        self.curriculum_latest_credit = 1.0

        self._sync_model_archives()
        self.update_opponent_pool()
        self.student_eval_history.clear()
        self.student_win_rate_str = "Pending live WR"
        self.save_checkpoint(manual=True)

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

        # Keep the student fixed throughout the game. Each state receives one
        # target only after the result is known.
        self.episode_trajectory.append((
            state_tensor.detach().cpu().clone(),
            turn_pid,
            next_state_tensor.detach().cpu().clone(),
            next_turn_pid
        ))
        if not hasattr(self, 'episode_policies'):
            self.episode_policies = []
        policy = getattr(self, '_pending_search_policy', None)
        self._pending_search_policy = None
        self.episode_policies.append(policy.detach().cpu().clone() if policy is not None else None)
        if not done:
            return

        self.last_game_batches = []
        opponent_gen = self.current_opponent_gen
        accepted = winner_id in (1, 2) and (
            opponent_gen == self.teacher_gen or winner_id != self.student_id)
        if not accepted:
            reason = 'unfinished' if winner_id == 0 else 'historical student win'
            print(f"Gen #{opponent_gen} {reason}: discarded {len(self.episode_trajectory)} plies.")
            self.episode_trajectory.clear()
            self.episode_policies.clear()
            return

        tactical = self._tactical_replay_indices(winner_id)
        game_id = self.next_replay_game_id
        student_outcome = 1 if winner_id == self.student_id else -1
        fresh = [
            {'state': state, 'target': 1.0 if pid == winner_id else -1.0,
             'game_id': game_id, 'teacher_gen': opponent_gen,
             'student_outcome': student_outcome, 'ply_index': index,
             'tactical': index in tactical}
            for index, (state, pid, _, _) in enumerate(self.episode_trajectory)
        ]
        for sample, policy in zip(fresh, self.episode_policies):
            if policy is not None:
                sample['policy'] = policy
        for batch, counts in self._training_batches(fresh, opponent_gen):
            if not self._optimize_student_batch(batch):
                raise RuntimeError('accepted game optimizer update did not complete')
            self.replay_samples_latest += counts['replay_latest']
            self.replay_samples_historic += counts['replay_historic']
            self.replay_samples_tactical += (
                counts['same_teacher_tactical'] + counts['general_tactical'])
            counts['tactical'] = sum(sample['tactical'] for sample in batch)
            counts['weighted_tactical_loss_share'] = self._last_tactical_loss_share
            counts['loss'] = self.last_loss
            self.last_game_batches.append(counts)
            print(f"Batch {len(self.last_game_batches)}: size={len(batch)} "
                  f"fresh/same/general={counts['fresh']}/{counts['same_teacher']}/{counts['general']} "
                  f"tactical={counts['tactical']} "
                  f"weighted tactical loss={self._last_tactical_loss_share:.2%}")

        # Commit only after all fresh positions have trained. This game cannot
        # fill its own replay slots; eviction always removes an entire game.
        self.replay_games.setdefault(
            opponent_gen, deque(maxlen=self.REPLAY_GAMES_PER_TEACHER)).append({
                'game_id': game_id, 'teacher_gen': opponent_gen,
                'student_outcome': student_outcome, 'positions': fresh,
            })
        self.next_replay_game_id += 1
        self.episode_trajectory.clear()
        self.episode_policies.clear()

    def log_match_result(self, winner_id):
        duration = round(time.time() - self.game_start_time, 1)
        if winner_id == 0: winner_str = "Unfinished"
        elif getattr(self, 'mode', 'TRAIN') == "TRAIN": winner_str = "Student" if winner_id == self.student_id else "Teacher"
        else: winner_str = "Human" if winner_id == 1 else "Teacher"
            
        self.match_history.appendleft({
            'game': self.games_played + 1, 'winner': winner_str, 'winner_id': winner_id,
            'turns': self.state['turns_played'], 'time': duration,
            'p1_w': self.state['p1_walls'], 'p2_w': self.state['p2_walls']
        })

    def _get_center_pixels(self, pos):
        offset_x, offset_y = LEFT_SIDEBAR_WIDTH + PADDING, 85 + PADDING
        unit = CELL_SIZE + WALL_THICKNESS
        return (
            offset_x + pos[0] * unit + CELL_SIZE / 2,
            offset_y + pos[1] * unit + CELL_SIZE / 2,
        )

    def _blit_text(self, text, x, y, color=TEXT_COLOR, font=None):
        surface = (font or self.tiny_font).render(str(text), True, color)
        self.virtual_screen.blit(surface, (x, y))
        return surface

    def draw_menu(self):
        if self.headless:
            return
        self.virtual_screen.fill(BG_COLOR)
        title = self.title_font.render("QuoriNet Trainer", True, TEXT_COLOR)
        self.virtual_screen.blit(title, title.get_rect(center=(WIDTH // 2, 110)))

        self.rect_train = pygame.Rect(WIDTH // 2 - 160, 220, 320, 55)
        self.rect_save_manual = pygame.Rect(WIDTH // 2 - 160, 290, 320, 55)
        pygame.draw.rect(self.virtual_screen, P2_COLOR, self.rect_train)
        pygame.draw.rect(self.virtual_screen, MUTED_TEXT, self.rect_save_manual)
        for rect, label in (
            (self.rect_train, "TRAIN BOT (WATCH)"),
            (self.rect_save_manual, "SAVE CURRENT STUDENT"),
        ):
            text = self.font.render(label, True, (255, 255, 255))
            self.virtual_screen.blit(text, text.get_rect(center=rect.center))

        self._blit_text("[F11/F: fullscreen, C: save checkpoint]",
                        WIDTH // 2 - 145, 370, MUTED_TEXT)
        if self.toast_timer > 0:
            self.toast_timer -= 1
            toast = self.small_font.render(self.toast_msg, True, (46, 125, 50))
            self.virtual_screen.blit(toast, toast.get_rect(center=(WIDTH // 2, 410)))
        self.render_and_flip()

    def draw_left_sidebar(self):
        pygame.draw.rect(
            self.virtual_screen, SIDEBAR_BG,
            pygame.Rect(0, 0, LEFT_SIDEBAR_WIDTH, HEIGHT)
        )
        self._blit_text("CURRENT TRAINING WIN RATES", 12, 14, font=self.small_font)
        self._blit_text("Generation", 12, 44, MUTED_TEXT)
        self._blit_text("Win Rate", 120, 44, MUTED_TEXT)
        self._blit_text("Sampling", 235, 44, MUTED_TEXT)

        sampling_by_gen = {}
        for (_, pool_gen), weight in zip(self.opponent_pool, self.opponent_weights):
            sampling_by_gen[pool_gen] = sampling_by_gen.get(pool_gen, 0.0) + weight

        y = 66
        for gen in reversed(self._training_generations()):
            if y > HEIGHT - 20:
                break
            win_rate = self._generation_win_rate(gen)
            value = "Pending" if win_rate is None else f"{win_rate * 100:.1f}%"
            color = P1_COLOR if gen == self.current_opponent_gen else TEXT_COLOR
            self._blit_text(f"Gen #{gen}", 12, y, color)
            self._blit_text(value, 120, y)
            self._blit_text(f"{sampling_by_gen.get(gen, 0.0) * 100:.1f}%", 235, y)
            y += 18

    def draw_right_sidebar(self):
        x = LEFT_SIDEBAR_WIDTH + BOARD_PIXEL_SIZE
        pygame.draw.rect(
            self.virtual_screen, SIDEBAR_BG,
            pygame.Rect(x, 0, SIDEBAR_WIDTH, HEIGHT)
        )
        self._blit_text("LIVE METRICS & ANALYTICS", x + 12, 14, font=self.small_font)

        if hasattr(self, '_display_eval'):
            eval_score = self._display_eval
            current_lr = self._display_lr
        else:
            eval_score = round(self.bot1.evaluate(self.state, self.state['turn']), 1)
            current_lr = self.optimizer.param_groups[0]['lr']
        final_test = (
            f"Vs Gen #{self.eval_vs_gen} ({self.eval_game_curr}/{self.eval_game_total})"
            if self.is_evaluating else "Waiting on live gates"
        )
        metrics = [
            ("Red Player (P1)", self.p1_gen_str),
            ("Blue Player (P2)", self.p2_gen_str),
            ("Latest Teacher Gen", f"Gen #{self.teacher_gen}"),
            ("Candidate Student Gen", f"Gen #{self.teacher_gen + 1}"),
            ("Wall candidates", "All legal"),
            ("Student Win Rate", self.student_win_rate_str),
            ("Final Test", final_test),
            ("AI Board Evaluation", f"{eval_score}"),
            ("Learning Rate", f"{current_lr:.8f}"),
            ("Student Training Loss", f"{self.last_loss:.5f}"),
        ]
        y = 46
        for label, value in metrics:
            self._blit_text(label, x + 12, y, MUTED_TEXT)
            rendered = self.tiny_font.render(str(value), True, TEXT_COLOR)
            self.virtual_screen.blit(rendered, (x + SIDEBAR_WIDTH - rendered.get_width() - 12, y))
            y += 20

        self.rect_lr_up = pygame.Rect(x + 12, 240, 156, 26)
        self.rect_lr_down = pygame.Rect(x + 180, 240, 168, 26)
        for rect, label in ((self.rect_lr_up, 'LR x10'), (self.rect_lr_down, 'LR x0.1')):
            pygame.draw.rect(self.virtual_screen, MUTED_TEXT, rect)
            text = self.small_font.render(label, True, (255, 255, 255))
            self.virtual_screen.blit(text, text.get_rect(center=rect.center))

        self._blit_text("MATCH LOG HISTORY", x + 12, 270, font=self.small_font)
        for label, dx in (("Game", 12), ("Winner", 72), ("Turns", 180), ("Time", 245)):
            self._blit_text(label, x + dx, 300, MUTED_TEXT)
        y = 324
        for log in reversed(list(self.match_history)[:4]):
            self._blit_text(f"#{log['game']}", x + 12, y)
            self._blit_text(log['winner'], x + 72, y)
            self._blit_text(log['turns'], x + 180, y)
            self._blit_text(f"{log['time']}s", x + 245, y)
            y += 20

    def draw_board(self, mouse_pos=(-1, -1)):
        if self.headless:
            return 'none', -1, -1, self.wall_orientation
        self.virtual_screen.fill(BG_COLOR)
        self.draw_left_sidebar()
        self.draw_right_sidebar()

        offset_x, offset_y = LEFT_SIDEBAR_WIDTH + PADDING, 85 + PADDING
        unit = CELL_SIZE + WALL_THICKNESS
        pygame.draw.rect(
            self.virtual_screen, BOARD_BG,
            pygame.Rect(LEFT_SIDEBAR_WIDTH, 85, BOARD_PIXEL_SIZE, BOARD_PIXEL_SIZE)
        )
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pygame.draw.rect(
                    self.virtual_screen, CELL_COLOR,
                    pygame.Rect(offset_x + col * unit, offset_y + row * unit,
                                CELL_SIZE, CELL_SIZE)
                )

        for col, row in self.state['walls_h']:
            color = ORANGE_COLOR if (col, row, 'H') in self.state.get('random_walls', set()) else WALL_COLOR
            pygame.draw.rect(
                self.virtual_screen, color,
                pygame.Rect(offset_x + col * unit, offset_y + row * unit + CELL_SIZE,
                            CELL_SIZE * 2 + WALL_THICKNESS, WALL_THICKNESS)
            )
        for col, row in self.state['walls_v']:
            color = ORANGE_COLOR if (col, row, 'V') in self.state.get('random_walls', set()) else WALL_COLOR
            pygame.draw.rect(
                self.virtual_screen, color,
                pygame.Rect(offset_x + col * unit + CELL_SIZE, offset_y + row * unit,
                            WALL_THICKNESS, CELL_SIZE * 2 + WALL_THICKNESS)
            )

        p1_x, p1_y = self._get_center_pixels(self.state['p1_pos'])
        p2_x, p2_y = self._get_center_pixels(self.state['p2_pos'])
        p1_color = ORANGE_COLOR if self.state.get('p1_last_random') else P1_COLOR
        p2_color = ORANGE_COLOR if self.state.get('p2_last_random') else P2_COLOR
        pygame.draw.circle(self.virtual_screen, p1_color, (int(p1_x), int(p1_y)), 14)
        pygame.draw.circle(self.virtual_screen, p2_color, (int(p2_x), int(p2_y)), 14)

        self._blit_text(
            f"Red ({self.p1_gen_str}) · {self.state['p1_walls']}",
            LEFT_SIDEBAR_WIDTH + 16, 22, font=self.small_font
        )
        blue = self.small_font.render(
            f"{self.state['p2_walls']} · Blue ({self.p2_gen_str})", True, TEXT_COLOR
        )
        self.virtual_screen.blit(
            blue, (LEFT_SIDEBAR_WIDTH + BOARD_PIXEL_SIZE - blue.get_width() - 16, 22)
        )
        status = (
            "Unfinished" if self.winner == "Unfinished" else
            f"{self.winner} won!" if self.game_over else
            "Red's turn" if self.state['turn'] == 1 else "Blue's turn"
        )
        status_surface = self.small_font.render(status, True, TEXT_COLOR)
        self.virtual_screen.blit(
            status_surface,
            status_surface.get_rect(center=(LEFT_SIDEBAR_WIDTH + BOARD_PIXEL_SIZE // 2, 50))
        )
        if self.is_evaluating:
            remaining = self.eval_game_total - self.eval_game_curr
            training_status = (
                f"EVAL: vs Gen #{self.eval_vs_gen} | "
                f"{self.eval_game_curr}/{self.eval_game_total} ({remaining} left)"
            )
        else:
            training_status = (
                "TRAINING PAUSED" if not self.is_training_active
                else f"Games: {self.games_played} | Turns: {self.state['turns_played']}"
            )
        training_surface = self.small_font.render(training_status, True, P2_COLOR)
        self.virtual_screen.blit(
            training_surface,
            training_surface.get_rect(
                center=(LEFT_SIDEBAR_WIDTH + BOARD_PIXEL_SIZE // 2, 70)
            )
        )

        button_width = (BOARD_PIXEL_SIZE - 36) // 2
        left = LEFT_SIDEBAR_WIDTH + 12
        right = left + button_width + 12
        self.rect_train_toggle = pygame.Rect(left, HEIGHT - 78, button_width, 30)
        self.rect_test_promotion = pygame.Rect(right, HEIGHT - 78, button_width, 30)
        self.rect_new_game = pygame.Rect(left, HEIGHT - 42, button_width, 30)
        self.rect_menu = pygame.Rect(right, HEIGHT - 42, button_width, 30)
        controls = [
            (self.rect_train_toggle,
             "Stop Training" if self.is_training_active else "Continue Training",
             P1_COLOR if self.is_training_active else (46, 125, 50)),
            (self.rect_test_promotion,
             "Promotion Test Queued" if self._manual_promotion_test_requested else "Test Promotion",
             MUTED_TEXT if self._manual_promotion_test_requested else ORANGE_COLOR),
            (self.rect_new_game, "New Game", MUTED_TEXT),
            (self.rect_menu, "Menu", MUTED_TEXT),
        ]
        for rect, label, color in controls:
            pygame.draw.rect(self.virtual_screen, color, rect)
            text = self.small_font.render(label, True, (255, 255, 255))
            self.virtual_screen.blit(text, text.get_rect(center=rect.center))

        if self.toast_timer > 0:
            self.toast_timer -= 1
            toast = self.small_font.render(self.toast_msg, True, (46, 125, 50))
            self.virtual_screen.blit(
                toast,
                toast.get_rect(center=(LEFT_SIDEBAR_WIDTH + BOARD_PIXEL_SIZE // 2, 8))
            )
        self.render_and_flip()
        return 'none', -1, -1, self.wall_orientation
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
        bot = self.bot1 if turn_pid == 1 else self.bot2
        self._pending_search_policy = getattr(bot, 'last_policy', None)
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
                self.winner = "Unfinished"
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

                if winner_id == 0:
                    # A cutoff has no outcome for learning or win-rate gates.
                    self.save_meta()
                    if self.games_played % 250 == 0 or self._save_requested:
                        self.save_checkpoint(manual=self._save_requested)
                    return

                student_won = 1 if winner_id == self.student_id else 0
                self.student_eval_history.append(student_won)

                gen = self.current_opponent_gen
                if not student_won:
                    self.rematch_teacher_gen = gen
                elif self.rematch_teacher_gen == gen:
                    self.rematch_teacher_gen = None
                if self.rematch_teacher_gen is not None:
                    print(f"Rematch Gen #{self.rematch_teacher_gen} until the student wins.")
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

                self._record_lr_result(winner_id)
                if not self._reduce_lr_on_stagnation():
                    self._tune_learning_rate_from_pool(force=False)
                print(f"Learning: {self.optimizer_steps - self._game_start_optimizer_steps} "
                      f"steps this game / {self.optimizer_steps} total | "
                      f"LR={self.optimizer.param_groups[0]['lr']:.8f} "
                      f"({self.lr_inertia['status']}) | walls=all legal | "
                      f"replay samples latest/historic="
                      f"{self.replay_samples_latest}/{self.replay_samples_historic}")
                print(f"Current training WRs: {self._format_training_win_rates()}")
                self.save_meta()

                manual_test = self._manual_promotion_test_requested
                if self.rematch_teacher_gen is None and (
                    manual_test or self._all_training_gates_met()
                ):
                    self._manual_promotion_test_requested = False
                    if manual_test:
                        print(
                            f"Starting manually requested frozen promotion test "
                            f"for Candidate Gen #{self.teacher_gen + 1}."
                        )
                    else:
                        print(
                            f"Candidate Gen #{self.teacher_gen + 1} cleared every "
                            f"live training gate. Starting final frozen test."
                        )
                    if self.run_final_promotion_test():
                        print(
                            f"Candidate Gen #{self.teacher_gen + 1} passed every "
                            f"final gate. Installing new teacher."
                        )
                        self._install_promoted_student()
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
        if not self.current_opponent_is_latest or is_teacher:
            return 0.0
        return 0.30 * (0.85 ** self.state['turns_played'])

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
            
            # Explore only when the student is facing the latest champion.
            eps = self._training_epsilon(active_bot)
            
            action, _ = active_bot.get_move_and_eval(self.state, epsilon=eps)
            if action: self.state = active_bot.apply_action(self.state, pre_action_pid, action, is_random=getattr(active_bot, 'was_last_random', False))
            else: self.state['turns_played'] = 200
                
            self._commit_training_action(pre_action_state_tensor, pre_action_pid)
                
        print(f"Headless training complete! Total games played: {self.games_played}")
        self.save_checkpoint(manual=True)
        pygame.quit()
        sys.exit()

    def _publish_trainer_view(self):
        """Publish detached display data; only the worker touches models/state."""
        names = (
            'state', 'game_over', 'winner', 'games_played', 'teacher_gen',
            'current_opponent_gen', 'p1_gen_str', 'p2_gen_str',
            'student_win_rate_str', 'last_loss', 'is_evaluating',
            'eval_vs_gen', 'eval_game_curr', 'eval_game_total',
            '_manual_promotion_test_requested', 'match_history',
            'checkpoint_log', 'opponent_weights', 'toast_msg', 'toast_timer',
        )
        snapshot = {name: copy.deepcopy(getattr(self, name)) for name in names}
        snapshot['opponent_pool'] = [(None, gen) for _, gen in self.opponent_pool]
        # A separate cache prevents the display evaluation from changing search
        # caches, and prevents old values surviving an optimizer update.
        display_bot = copy.copy(self.bot1)
        display_bot._value_cache = {}
        snapshot['_display_eval'] = round(
            display_bot.evaluate(self.state, self.state['turn']), 1)
        snapshot['_display_lr'] = self.optimizer.param_groups[0]['lr']
        self._trainer_snapshot = snapshot
        self.toast_timer = 0

    def _apply_trainer_command(self, command, value=None):
        if command == 'start':
            self.mode, self.is_training_active = 'TRAIN', True
            self.reset_game()
        elif command == 'menu':
            self.mode = 'MENU'
        elif command == 'active':
            self.is_training_active = value
            if not value:
                self.save_checkpoint(manual=True)
        elif command == 'new':
            self.reset_game()
        elif command == 'lr':
            self._adjust_lr_manually(value)
        elif command == 'save':
            self.save_checkpoint(manual=True)
        elif command == 'snapshot':
            self.save_manual_student()
        elif command == 'promotion':
            self.request_manual_promotion_test()

    def _trainer_work(self):
        try:
            self._publish_trainer_view()
            while not self._trainer_stop.is_set() or not self._trainer_commands.empty():
                changed = False
                while not self._trainer_commands.empty():
                    self._apply_trainer_command(*self._trainer_commands.get_nowait())
                    changed = True
                if changed:
                    self._publish_trainer_view()
                    changed = False
                if (not self._trainer_stop.is_set()
                        and self._trainer_allowed.is_set() and self.mode == 'TRAIN'
                        and self.is_training_active):
                    if self.game_over:
                        self.reset_game()
                    else:
                        active_bot = self.bot1 if self.state['turn'] == 1 else self.bot2
                        state_tensor = encode_state(self.state, self.state['turn'])
                        pid = self.state['turn']
                        action, _ = active_bot.get_move_and_eval(
                            self.state, epsilon=self._training_epsilon(active_bot))
                        if action:
                            self.state = active_bot.apply_action(
                                self.state, pid, action,
                                is_random=getattr(active_bot, 'was_last_random', False))
                        else:
                            self.state['turns_played'] = 200
                        self._commit_training_action(state_tensor, pid)
                    changed = True
                if changed:
                    self._publish_trainer_view()
                else:
                    self._trainer_stop.wait(0.01)
        except InterruptedError:
            if not self._trainer_stop.is_set():
                self._trainer_error = sys.exc_info()[1]
        except Exception as error:
            self._trainer_error = error

    def run_trainer(self):
        # Pygame stays on the main thread. One worker preserves the existing
        # move/update/curriculum sequence; the view only reads published copies.
        view = copy.copy(self)
        view._display_eval = 'Pending'
        view._display_lr = self.optimizer.param_groups[0]['lr']
        view.opponent_pool = [(None, gen) for _, gen in self.opponent_pool]
        view.opponent_weights = list(self.opponent_weights)
        view.checkpoint_log = copy.deepcopy(self.checkpoint_log)
        view.match_history = copy.deepcopy(self.match_history)
        view.state = copy.deepcopy(self.state)
        self._trainer_commands = queue.SimpleQueue()
        self._trainer_allowed = threading.Event()
        self._trainer_stop = threading.Event()
        self._trainer_snapshot = None
        self._trainer_error = None
        worker = threading.Thread(target=self._trainer_work, name='quoridor-trainer')
        worker.start()
        last_snapshot = None

        def request(command, value=None):
            self._trainer_commands.put((command, value))
            view.toast_msg = 'Request queued; finishing current operation'
            view.toast_timer = 180

        clock = pygame.time.Clock()
        running = True
        try:
            while running or worker.is_alive():
                snapshot = self._trainer_snapshot
                if snapshot is not None and snapshot is not last_snapshot:
                    # Mode/pause belong to the UI so an older snapshot cannot undo
                    # a click while a move or frozen evaluation is finishing.
                    view.__dict__.update(snapshot)
                    last_snapshot = snapshot
                if self._trainer_error is not None:
                    self._trainer_allowed.clear()
                    view.is_training_active = False
                    view.toast_msg = f'Training stopped: {self._trainer_error}'
                    view.toast_timer = 180
                mouse_pos = view.get_virtual_mouse_pos()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        self._trainer_stop.set()
                        self._trainer_allowed.set()
                        view.toast_msg = 'Closing after the current operation finishes'
                        view.toast_timer = 180
                    elif not running:
                        continue
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_F11, pygame.K_f):
                            view.toggle_fullscreen()
                        elif event.key == pygame.K_c:
                            request('save')
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if view.mode == "MENU":
                            if view.rect_train.collidepoint(mouse_pos):
                                view.mode, view.is_training_active = "TRAIN", True
                                request('start')
                                self._trainer_allowed.set()
                            elif view.rect_save_manual.collidepoint(mouse_pos):
                                request('snapshot')
                        elif view.mode == "TRAIN":
                            if view.rect_lr_up.collidepoint(mouse_pos):
                                request('lr', 10.0)
                            elif view.rect_lr_down.collidepoint(mouse_pos):
                                request('lr', 0.1)
                            elif view.rect_menu.collidepoint(mouse_pos):
                                view.mode = "MENU"
                                self._trainer_allowed.clear()
                                request('menu')
                            elif view.rect_new_game.collidepoint(mouse_pos):
                                request('new')
                            elif view.rect_test_promotion.collidepoint(mouse_pos):
                                request('promotion')
                                view._manual_promotion_test_requested = True
                            elif view.rect_train_toggle.collidepoint(mouse_pos):
                                view.is_training_active = not view.is_training_active
                                request('active', view.is_training_active)
                                if view.is_training_active:
                                    self._trainer_allowed.set()
                                else:
                                    self._trainer_allowed.clear()

                if view.mode == "MENU":
                    view.draw_menu()
                else:
                    view.draw_board()
                clock.tick(60)
        finally:
            self._trainer_stop.set()
            self._trainer_allowed.set()
            worker.join()
            pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for a fresh run (resumes restore saved RNG state)')
    parser.add_argument('--threads', type=int, default=min(4, os.cpu_count() or 1),
                        help='PyTorch CPU threads; small boards often benefit from 1-4')
    parser.add_argument('--train-depth', type=int, default=2,
                        help='legacy search depth / MCTS budget in units of 64 simulations (default: 2)')
    parser.add_argument('--play-depth', type=int, default=4,
                        help='legacy search depth / MCTS budget in units of 64 simulations (default: 4)')
    parser.add_argument('--train-updates', type=int, default=None,
                        help='deprecated and ignored; fresh positions each receive one pass')
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

    if min(args.threads, args.train_depth, args.play_depth) < 1:
        parser.error('threads and search depths must be positive')
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

    game = Quoridor(
        headless=headless_mode,
        train_depth=args.train_depth,
        play_depth=args.play_depth,
        train_updates=args.train_updates
    )
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
        game.run_trainer()
