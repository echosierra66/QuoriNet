import pygame
import sys
import os
import json
import random
import time
import threading
import argparse
import copy
import math
import tempfile
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, in_channels=8):
        # Channels: P1 pos, P2 pos, H-wall, V-wall, P1 walls-left, P2 walls-left, P1 path-dist, P2 path-dist
        super().__init__()
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
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.value_head(x)

# Global Helper for encode_state distance calcs
def get_adjacent_global(state, c, r):
    moves = []
    if r > 0 and (c, r - 1) not in state['walls_h'] and (c - 1, r - 1) not in state['walls_h']: moves.append((c, r - 1))
    if r < 8 and (c, r) not in state['walls_h'] and (c - 1, r) not in state['walls_h']: moves.append((c, r + 1))
    if c > 0 and (c - 1, r) not in state['walls_v'] and (c - 1, r - 1) not in state['walls_v']: moves.append((c - 1, r))
    if c < 8 and (c, r) not in state['walls_v'] and (c, r - 1) not in state['walls_v']: moves.append((c + 1, r))
    return moves

def get_shortest_path_dist(state, pid):
    start_pos = state['p1_pos'] if pid == 1 else state['p2_pos']
    target_row = 8 if pid == 2 else 0
    queue = deque([(start_pos, 0)])
    visited = {start_pos}
    while queue:
        pos, dist = queue.popleft()
        if pos[1] == target_row: return dist
        for nxt in get_adjacent_global(state, *pos):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, dist + 1))
    return 999  # Blocked

def encode_state(state, active_pid):
    # 8 Channels exactly as specified
    tensor = torch.zeros((8, 9, 9), dtype=torch.float32)
    
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
    
    return tensor

def get_state_key(state):
    return (state['p1_pos'], state['p2_pos'], tuple(sorted(state['walls_h'])), tuple(sorted(state['walls_v'])))

class QuoridorBot:
    # How many of a player's own most-recent squares to remember, purely to
    # discourage short A<->B (or small-loop) oscillation when two paths are
    # otherwise perfectly tied. This is a soft nudge (see _oscillation_penalty),
    # not a hard rule, so a truly forced backtrack is still allowed.
    RECENT_MEMORY_LEN = 8
    OSCILLATION_PENALTY = 45.0

    def __init__(self, bot_id=2, max_depth=3, nn_model=None,
                 heuristic_weight=1.0, wall_search_breadth=12,
                 wall_exploration=0.05):
        self.bot_id = bot_id
        self.max_depth = max_depth
        self.nn_model = nn_model
        self.heuristic_weight = max(0.0, min(1.0, float(heuristic_weight)))
        self.wall_search_breadth = max(0, int(wall_search_breadth))
        self.wall_exploration = max(0.0, min(1.0, float(wall_exploration)))
        self.was_last_random = False
        
    def get_move_and_eval(self, state, epsilon=0.0):
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
        start_pos = state['p1_pos'] if pid == 1 else state['p2_pos']
        target_row = self.get_target_row(pid)
        queue = deque([(start_pos, [start_pos])])
        visited = {start_pos}

        while queue:
            pos, path = queue.popleft()
            if pos[1] == target_row: return path
            for nxt in self.get_adjacent(state, *pos):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None

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
        if orient == 'H':
            if (c, r) in state['walls_h'] or (c-1, r) in state['walls_h'] or (c+1, r) in state['walls_h']: return False
            if (c, r) in state['walls_v']: return False
            state['walls_h'].add((c, r))
        else:
            if (c, r) in state['walls_v'] or (c, r-1) in state['walls_v'] or (c, r+1) in state['walls_v']: return False
            if (c, r) in state['walls_h']: return False
            state['walls_v'].add((c, r))
            
        try:
            p1_safe = self.get_shortest_path(state, 1) is not None
            p2_safe = self.get_shortest_path(state, 2) is not None
        finally:
            if orient == 'H': state['walls_h'].remove((c, r))
            else: state['walls_v'].remove((c, r))
        
        return p1_safe and p2_safe

    def generate_actions(self, state, pid):
        actions = []
        valid_moves = self.get_valid_moves(state, pid)
        
        def get_path_dist(m):
            sim_state = state.copy()
            if pid == 1: sim_state['p1_pos'] = m
            else: sim_state['p2_pos'] = m
            path = self.get_shortest_path(sim_state, pid)
            return len(path) if path else 999

        valid_moves.sort(key=get_path_dist)
        for move in valid_moves: actions.append(('move', move))
            
        walls_left = state['p1_walls'] if pid == 1 else state['p2_walls']
        if walls_left <= 0: return actions

        opponent_id = self.get_opponent(pid)
        opp_path = self.get_shortest_path(state, opponent_id)
        my_path = self.get_shortest_path(state, pid)
        if not opp_path or not my_path: return actions

        base_my_dist, base_opp_dist = len(my_path), len(opp_path)
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
        for c, r, orient in candidate_walls:
            if self.is_valid_wall(state, c, r, orient):
                try:
                    if orient == 'H': state['walls_h'].add((c, r))
                    else: state['walls_v'].add((c, r))
                    new_opp_path = self.get_shortest_path(state, opponent_id)
                    new_my_path = self.get_shortest_path(state, pid)
                finally:
                    if orient == 'H': state['walls_h'].remove((c, r))
                    else: state['walls_v'].remove((c, r))
                
                if new_opp_path is not None and new_my_path is not None:
                    delay = (len(new_opp_path) - base_opp_dist) - (len(new_my_path) - base_my_dist)
                    scored_walls.append((delay, ('wall', (c, r, orient))))

        scored_walls.sort(key=lambda x: x[0], reverse=True)
        quota = self.wall_search_breadth
        
        for i, (delay, wall_action) in enumerate(scored_walls):
            if i < quota:
                if delay > 0 or random.random() < self.wall_exploration:
                    actions.append(wall_action)
            else: break
        return actions

    def rank_and_prune(self, state, actions, pid, top_k=5):
        if not actions: return []
        actions_copy = list(actions)
        random.shuffle(actions_copy)

        scored_actions = []
        for action in actions_copy:
            next_state = self.apply_action(state, pid, action)
            val = self.evaluate(next_state, pid)
            scored_actions.append((val, action))

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
        if state['p1_pos'][1] == 0: return 10000 if eval_pid == 1 else -10000
        if state['p2_pos'][1] == 8: return -10000 if eval_pid == 1 else 10000
        
        is_repeated = get_state_key(state) in state.get('history', set())
        osc_penalty = self._oscillation_penalty(state, eval_pid)

        p1_path = self.get_shortest_path(state, 1)
        p2_path = self.get_shortest_path(state, 2)
        
        if not p1_path: return 10000 if eval_pid == 2 else -10000
        if not p2_path: return -10000 if eval_pid == 2 else 10000

        d1, d2 = len(p1_path) - 1, len(p2_path) - 1
        p1_progress, p2_progress = 8 - state['p1_pos'][1], state['p2_pos'][1]

        if self.nn_model is not None:
            with torch.no_grad():
                nn_val = self.nn_model(encode_state(state, eval_pid).unsqueeze(0)).item()

            # Both inputs are normalized to [-1, 1] before blending.  A weight
            # of 1.0 is a heuristic policy; decay toward 0.0 increases genuine
            # neural reliance without changing the wall-search action space.
            path_advantage = (d2 - d1) if eval_pid == 1 else (d1 - d2)
            progress_advantage = (
                (p1_progress - p2_progress) if eval_pid == 1
                else (p2_progress - p1_progress)
            )
            heur_val = math.tanh(
                (path_advantage + 0.25 * progress_advantage) / 6.0
            )
            blended = (
                (1.0 - self.heuristic_weight) * nn_val
                + self.heuristic_weight * heur_val
            )
            score = blended * 100.0
            if is_repeated: score -= 8000
            score -= osc_penalty
            return score

        if eval_pid == 1:
            score = (d2 - d1) * 25.0 + (p1_progress - p2_progress) * 5.0
        else:
            score = (d1 - d2) * 25.0 + (p2_progress - p1_progress) * 5.0

        if is_repeated: score -= 8000
        score -= osc_penalty
        return score

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
        
        actions = self.rank_and_prune(state, raw_actions, pid, top_k=5)
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
BOARD_SIZE = 9
PADDING = int(25 * SCALE)
BOARD_PIXEL_SIZE = (BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * WALL_THICKNESS + (PADDING * 2))

SIDEBAR_WIDTH = int(380 * SCALE)
LEFT_SIDEBAR_WIDTH = int(370 * SCALE)  # Extended transparency UI space
WIDTH = BOARD_PIXEL_SIZE + SIDEBAR_WIDTH + LEFT_SIDEBAR_WIDTH
HEIGHT = BOARD_PIXEL_SIZE + int(120 * SCALE)

class Quoridor:
    def __init__(self, headless=False):
        pygame.init()
        self.headless = headless
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

        # Central policy and promotion thresholds.  These names are also saved
        # in evaluation reports so a result is interpretable after restart.
        self.RECENT_WINDOW_SIZE = 5
        self.RECENT_DECAY_RATIO = 2.0
        self.TRAIN_METRIC_WINDOW = 60
        self.MIN_SAMPLES_TRUSTED = 15
        self.GATE_MIN_GAMES = 40
        self.GATE_WIN_RATE_THRESHOLD = 0.57
        self.EVAL_GAMES_PER_MODEL = 20
        self.EVAL_RETRY_AFTER_LATEST_GAMES = 20
        self.EVAL_SEED = 73517
        self.EVAL_LATEST_THRESHOLD = 0.50
        self.EVAL_RECENT_THRESHOLD = 0.52
        self.EVAL_HISTORIC_THRESHOLD = 0.55
        self.EVAL_HEURISTIC_BLEND = 0.35
        self.EVAL_WALL_SEARCH_BREADTH = 12

        # Adaptive opponent allocation.  Difficulty and confidence affect the
        # support half; the current teacher always keeps a plurality.
        self.LATEST_TEACHER_LOG_WINDOW = 60
        self.LATEST_MIN_SHARE = 0.52
        self.OPPONENT_MAX_SHARE = 0.72
        self.RECOVERY_MIN_SHARE = 0.08
        self.RECOVERY_MAX_SHARE = 0.30
        self.DIVERSITY_MIN_MASS = 0.10
        self.HARD_TARGET_THRESHOLD = 0.50
        self.RECOVERY_CLEAR_THRESHOLD = 0.60
        self.WEIGHT_REFRESH_EVIDENCE_INTERVAL = 5

        # The action-search policy is separate from neural/heuristic blending.
        self.wall_search_breadth = 12
        self.wall_exploration = 0.05
        self.HEURISTIC_DECAY = 0.85
        self.HEURISTIC_MIN = 0.05

        # Replay is split so old easy generations can never swamp the current
        # target. Latest-teacher data gets a majority of every train batch once
        # enough of it exists.
        self.REPLAY_LATEST_CAPACITY = 140000
        self.REPLAY_HISTORIC_CAPACITY = 60000
        self.REPLAY_LATEST_MIN_SHARE = 0.70
        self.REPLAY_WARMUP = 256
        self.REPLAY_BATCH_SIZE = 128

        # Sparse terminal outcomes are backfilled for every opponent, including
        # historical recovery games.
        self.OUTCOME_BACKFILL_MIX = 0.70
        self.OUTCOME_BACKFILL_DISCOUNT = 0.995

        # Periodic LR scheduler: fixed, non-overlapping latest-teacher blocks;
        # exponentially weighted cumulative score; patience and cooldown.
        self.LR_MIN = 0.00001
        self.LR_MAX = 0.0008
        self.LR_MEASUREMENT_GAMES = 20
        self.LR_MIN_BLOCKS = 3
        self.LR_PATIENCE = 2
        self.LR_COOLDOWN_GAMES = 40
        self.LR_TREND_EPS = 0.02
        self.LR_DOWN_FACTOR = 0.80
        self.LR_PLATEAU_FACTOR = 0.90
        self.LR_RECENCY_DECAY = 0.72
        self.GRAD_CLIP_NORM = 1.0

        # State vars
        self.student_net = SpatialQuoriNetV2(in_channels=8)
        self.teacher_net = SpatialQuoriNetV2(in_channels=8)
        self.student_net.eval()
        self.teacher_net.eval()
        self.optimizer = optim.Adam(self.student_net.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()

        self.games_played = 0
        self.teacher_gen = 1
        self.heuristic_weight = 1.0

        # Three separate metric families. Outcomes are 1.0 win, 0.5 draw, 0.0
        # loss; counts are retained independently so draws remain visible.
        self.training_metrics = {}
        self.evaluation_metrics = {}
        self.latest_teacher_history = deque(maxlen=self.LATEST_TEACHER_LOG_WINDOW)
        self.student_win_rate_str = "Latest gate: no samples"
        self.student_id = 2

        self.p1_gen_str = "Gen #1"
        self.p2_gen_str = "Gen #1"

        self.latest_replay_buffer = deque(maxlen=self.REPLAY_LATEST_CAPACITY)
        self.historic_replay_buffer = deque(maxlen=self.REPLAY_HISTORIC_CAPACITY)
        self.episode_trajectory = []
        self.last_loss = 0.0

        self.match_history = deque(maxlen=6)
        self.opponent_pool = []
        self.opponent_weights = []
        self.opponent_support_base_weights = []

        self.recovery_targets = {}
        self.last_evaluation = None
        self.last_eval_latest_games_seen = -self.EVAL_RETRY_AFTER_LATEST_GAMES
        self.last_promotion_status = "Waiting for latest-teacher gate"
        self.last_weight_refresh_game = 0
        self.opponent_last_update = {}
        self.generation_policy = {}
        self.current_opponent_gen = 1
        self.current_opponent_is_latest = True
        self.curriculum_latest_credit = 1.0

        self.latest_teacher_games_seen = 0
        self.lr_pending_outcomes = []
        self.lr_measurements = []
        self.lr_no_improvement_count = 0
        self.lr_next_eligible_game = 0
        self.lr_adjustment_log = []

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

    # ==========================================
    # SECTION 4: OPPONENT POOL
    # ==========================================
    def update_opponent_pool(self):
        self.opponent_pool = []
        self.opponent_pool.append((self.teacher_net, self.teacher_gen))
        for gen in range(self.teacher_gen - 1, 0, -1):
            net = self._load_net_for_gen(gen)
            if net is not None:
                self.opponent_pool.append((net, gen))

        self._refresh_opponent_weights()
        print(
            f"Opponent pool updated. Latest target: Gen #{self.teacher_gen} "
            f"({self._probability_for_gen(self.teacher_gen)*100:.1f}%) "
            f"| Recovery: {sorted(self.recovery_targets)} "
            f"| Support models: {len(self.opponent_pool)-1} "
            f"| Total weight: {sum(self.opponent_weights):.4f}"
        )

    def _latest_teacher_win_rate(self):
        if not self.latest_teacher_history:
            return None
        return sum(self.latest_teacher_history) / len(self.latest_teacher_history)

    def _latest_teacher_share(self):
        return self._probability_for_gen(self.teacher_gen) or 1.0

    @staticmethod
    def _outcome_counts(values):
        return {
            "wins": sum(1 for v in values if v >= 0.75),
            "draws": sum(1 for v in values if 0.25 < v < 0.75),
            "losses": sum(1 for v in values if v <= 0.25),
        }

    @staticmethod
    def _smoothed_rate(values):
        # Beta(1, 1) smoothing prevents one result from becoming 0% or 100%.
        values = list(values)
        return (sum(values) + 1.0) / (len(values) + 2.0)

    def _training_log(self, gen):
        if gen not in self.training_metrics:
            self.training_metrics[gen] = deque(maxlen=self.TRAIN_METRIC_WINDOW)
        return self.training_metrics[gen]

    def _performance(self, gen):
        log = self.training_metrics.get(gen, ())
        n = len(log)
        rate = self._smoothed_rate(log)
        confidence = min(1.0, n / float(self.MIN_SAMPLES_TRUSTED * 2))
        deficit = max(0.0, self.RECOVERY_CLEAR_THRESHOLD - rate)
        return rate, n, confidence, deficit

    def _role_for_gen(self, gen):
        if gen == self.teacher_gen:
            return "Latest"
        if gen in self.recovery_targets:
            return "Recovery Target"
        if gen >= max(1, self.teacher_gen - self.RECENT_WINDOW_SIZE + 1):
            return "Recent"
        return "Historic"

    def _recovery_multiplier(self, gen):
        if gen not in self.recovery_targets:
            return None
        _, _, confidence, deficit = self._performance(gen)
        eval_deficit = float(self.recovery_targets[gen].get("eval_deficit", 0.0))
        return min(3.0, 1.0 + 4.0 * max(deficit * confidence, eval_deficit))

    def _update_recovery_from_training(self, gen):
        if gen == self.teacher_gen:
            return
        rate, n, confidence, _ = self._performance(gen)
        if n < self.MIN_SAMPLES_TRUSTED:
            return
        if rate < self.HARD_TARGET_THRESHOLD and confidence >= 0.5:
            self.recovery_targets.setdefault(gen, {})
            self.recovery_targets[gen].update({
                "reason": "reliable training underperformance",
                "since_game": self.games_played,
                "last_rate": rate,
            })
        elif rate >= self.RECOVERY_CLEAR_THRESHOLD:
            target = self.recovery_targets.get(gen)
            if target and target.get("reason") == "reliable training underperformance":
                del self.recovery_targets[gen]

    @staticmethod
    def _apply_escalation_cap(weights, total_mass, max_share):
        """Water-fill without ever returning capped entries to redistribution."""
        if not weights or total_mass <= 0:
            return {g: 0.0 for g in weights}
        if max_share <= 0:
            return {g: 0.0 for g in weights}
        if len(weights) * max_share + 1e-12 < total_mass:
            # The requested mass is mathematically impossible under the cap.
            total_mass = len(weights) * max_share
        remaining = set(weights)
        result = {g: 0.0 for g in weights}
        mass_left = total_mass
        while remaining and mass_left > 1e-15:
            raw_total = sum(max(0.0, weights[g]) for g in remaining)
            proposed = {
                g: (mass_left / len(remaining) if raw_total <= 0 else
                    mass_left * max(0.0, weights[g]) / raw_total)
                for g in remaining
            }
            newly_capped = {g for g, value in proposed.items()
                            if value > max_share + 1e-15}
            if not newly_capped:
                for g, value in proposed.items():
                    result[g] = value
                mass_left = 0.0
                break
            for g in newly_capped:
                result[g] = max_share
                mass_left -= max_share
                remaining.remove(g)
        return result

    def _refresh_opponent_weights(self):
        if not self.opponent_pool:
            self.opponent_weights = []
            return
        gens = [gen for _, gen in self.opponent_pool]
        if len(gens) == 1:
            self.opponent_weights = [1.0]
            return

        supports = [g for g in gens if g != self.teacher_gen]
        latest_share = self.LATEST_MIN_SHARE
        support_mass = 1.0 - latest_share
        priorities = {}
        for gen in supports:
            rate, n, confidence, deficit = self._performance(gen)
            recency = 1.0 / (1.0 + max(0, self.teacher_gen - gen - 1))
            recovery = self._recovery_multiplier(gen) or 1.0
            priorities[gen] = (0.25 + recency + 5.0 * deficit * confidence) * recovery

        # Start recovery targets with a meaningful floor, bounded so many hard
        # opponents cannot crowd out the latest teacher or diversity.
        allocated = {g: 0.0 for g in supports}
        active_recovery = [g for g in supports if g in self.recovery_targets]
        if active_recovery:
            floor = min(
                self.RECOVERY_MIN_SHARE,
                max(0.0, support_mass - self.DIVERSITY_MIN_MASS) / len(active_recovery)
            )
            for gen in active_recovery:
                allocated[gen] = floor

        remaining = support_mass - sum(allocated.values())
        caps = {
            g: (self.RECOVERY_MAX_SHARE if g in active_recovery
                else self.OPPONENT_MAX_SHARE)
            for g in supports
        }
        residual_priorities = dict(priorities)
        while remaining > 1e-12 and residual_priorities:
            capacities = {g: max(0.0, caps[g] - allocated[g])
                          for g in residual_priorities}
            candidates = {g: w for g, w in residual_priorities.items()
                          if capacities[g] > 1e-12}
            if not candidates:
                break
            proposed = self._apply_escalation_cap(
                candidates, remaining, max(capacities.values())
            )
            progress = 0.0
            for gen, value in proposed.items():
                add = min(value, capacities[gen])
                allocated[gen] += add
                progress += add
            remaining -= progress
            residual_priorities = {g: priorities[g] for g in candidates
                                   if caps[g] - allocated[g] > 1e-12}
            if progress <= 1e-15:
                break

        if remaining > 1e-10:
            # Only possible when configured caps cannot fill the bucket. Give
            # the unfillable mass back to latest, still respecting its cap.
            latest_share += min(remaining, self.OPPONENT_MAX_SHARE - latest_share)
        weights_by_gen = {self.teacher_gen: latest_share, **allocated}
        total = sum(weights_by_gen.values())
        if total <= 0:
            weights_by_gen = {self.teacher_gen: 1.0}
            total = 1.0
        self.opponent_weights = [weights_by_gen.get(g, 0.0) / total for g in gens]

    def _probability_for_gen(self, gen):
        for (_, pool_gen), weight in zip(self.opponent_pool, self.opponent_weights):
            if pool_gen == gen:
                return weight
        return 0.0

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

        # Deficit/credit scheduling gives the requested smooth distribution but
        # avoids pathological single-game alternation. As latest share rises,
        # naturally longer back-to-back latest-teacher streaks appear.
        latest_share = self.opponent_weights[latest_idx]
        self.curriculum_latest_credit += latest_share

        if self.curriculum_latest_credit >= 1.0:
            self.curriculum_latest_credit -= 1.0
            return self.opponent_pool[latest_idx]

        support_pool = [self.opponent_pool[i] for i in support_indices]
        support_weights = [self.opponent_weights[i] for i in support_indices]
        return random.choices(support_pool, weights=support_weights, k=1)[0]

    def _append_replay_sample(self, state_tensor, target, opponent_gen):
        item = (state_tensor, float(max(-1.0, min(1.0, target))))
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

        latest_share = max(
            self.REPLAY_LATEST_MIN_SHARE,
            self._latest_teacher_share()
        )
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
        return batch

    def _roll_replay_buffers_for_new_teacher(self):
        # The old "latest" data is still useful for anti-forgetting, but after
        # promotion it must no longer occupy the protected current-target bucket.
        while self.latest_replay_buffer:
            self.historic_replay_buffer.append(self.latest_replay_buffer.popleft())

    def _set_learning_rate(self, requested_lr, reason, metric, sample_window):
        old_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(self.LR_MIN, min(self.LR_MAX, requested_lr))
        changed = abs(new_lr - old_lr) >= 1e-12
        if changed:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        self.lr_next_eligible_game = (
            self.latest_teacher_games_seen + self.LR_COOLDOWN_GAMES
        )
        entry = {
            "latest_game": self.latest_teacher_games_seen,
            "metric": metric,
            "old_lr": old_lr,
            "new_lr": new_lr,
            "reason": reason,
            "sample_window": sample_window,
            "next_eligible": self.lr_next_eligible_game,
        }
        self.lr_adjustment_log.append(entry)
        self.lr_adjustment_log = self.lr_adjustment_log[-50:]
        print(
            f"LR scheduler: {old_lr:.6f} -> {new_lr:.6f} | {reason} | "
            f"weighted metric={metric:.3f}, samples={sample_window}, "
            f"next eligible={self.lr_next_eligible_game}"
        )
        return changed

    def _reset_lr_scheduler_for_new_teacher(self):
        self.latest_teacher_games_seen = 0
        self.lr_pending_outcomes = []
        self.lr_measurements = []
        self.lr_no_improvement_count = 0
        self.lr_next_eligible_game = 0

    def _cumulative_weighted_lr_metric(self):
        if not self.lr_measurements:
            return None
        weights = [
            self.LR_RECENCY_DECAY ** (len(self.lr_measurements) - 1 - i)
            for i in range(len(self.lr_measurements))
        ]
        return sum(m["score"] * w for m, w in zip(self.lr_measurements, weights)) / sum(weights)

    def _update_lr_scheduler(self, outcome):
        self.lr_pending_outcomes.append(float(outcome))
        if len(self.lr_pending_outcomes) < self.LR_MEASUREMENT_GAMES:
            return
        block = self.lr_pending_outcomes[:self.LR_MEASUREMENT_GAMES]
        del self.lr_pending_outcomes[:self.LR_MEASUREMENT_GAMES]
        self.lr_measurements.append({
            "end_game": self.latest_teacher_games_seen,
            "samples": len(block),
            "score": sum(block) / len(block),
        })
        self.lr_measurements = self.lr_measurements[-12:]
        if len(self.lr_measurements) < self.LR_MIN_BLOCKS:
            return
        metric = self._cumulative_weighted_lr_metric()
        if self.latest_teacher_games_seen < self.lr_next_eligible_game:
            return
        recent = self.lr_measurements[-1]["score"]
        previous = self.lr_measurements[-2]["score"]
        trend = recent - previous
        if trend > self.LR_TREND_EPS:
            self.lr_no_improvement_count = 0
            reason = "acquisition improving; hold"
            requested = self.optimizer.param_groups[0]['lr']
        else:
            self.lr_no_improvement_count += 1
            if self.lr_no_improvement_count < self.LR_PATIENCE:
                return
            current_lr = self.optimizer.param_groups[0]['lr']
            if trend < -self.LR_TREND_EPS:
                requested = current_lr * self.LR_DOWN_FACTOR
                reason = "latest-teacher acquisition regressed; stabilize"
            else:
                requested = current_lr * self.LR_PLATEAU_FACTOR
                reason = "fixed-block acquisition plateau; gentle decay"
            self.lr_no_improvement_count = 0
        self._set_learning_rate(
            requested, reason, metric,
            sum(m["samples"] for m in self.lr_measurements)
        )

    def _load_net_for_gen(self, gen):
        if gen == self.teacher_gen: return self.teacher_net
        path = f'quorinet_teacher_gen_{gen}.pth'
        if not os.path.exists(path): return None
        try:
            net = SpatialQuoriNetV2(in_channels=8)
            net.load_state_dict(torch.load(path, weights_only=True))
            net.eval()
            return net
        except Exception as e:
            print(f"Failed to load pool model {gen}: {e}")
            return None

    def _policy_for_gen(self, gen):
        saved = self.generation_policy.get(gen, {})
        return {
            "heuristic_weight": float(saved.get("heuristic_weight", 1.0)),
            "wall_search_breadth": int(saved.get(
                "wall_search_breadth", self.wall_search_breadth
            )),
            "wall_exploration": float(saved.get("wall_exploration", 0.0)),
        }

    def _make_bot(self, bot_id, net, policy=None, depth=2):
        policy = policy or {
            "heuristic_weight": self.heuristic_weight,
            "wall_search_breadth": self.wall_search_breadth,
            "wall_exploration": self.wall_exploration,
        }
        return QuoridorBot(
            bot_id=bot_id, max_depth=depth, nn_model=net,
            heuristic_weight=policy["heuristic_weight"],
            wall_search_breadth=policy["wall_search_breadth"],
            wall_exploration=policy.get("wall_exploration", 0.0),
        )


    def print_headless_metrics(self):
        if not self.headless: return
        eval_score = round(self.bot1.evaluate(self.state, self.state['turn']), 1)
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"\n--- MATCH #{self.games_played} TERMINAL METRICS ---")
        print(f"Red Player (P1):  {self.p1_gen_str}")
        print(f"Blue Player (P2): {self.p2_gen_str}")
        print(f"Latest Teacher:   Gen #{self.teacher_gen}")
        print(f"Latest Share:     {self._latest_teacher_share()*100:.1f}%")
        print(f"Heuristic Blend: {self.heuristic_weight:.3f} (1=heuristic)")
        print(f"Latest Gate WR:  {self.student_win_rate_str}")
        print(f"Promotion Gate:  {self.last_promotion_status}")
        print(f"Recovery Targets:{sorted(self.recovery_targets)}")
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
            teacher_policy = self._policy_for_gen(pool_gen)
            student_policy = {
                "heuristic_weight": self.heuristic_weight,
                "wall_search_breadth": self.wall_search_breadth,
                "wall_exploration": self.wall_exploration,
            }

            if self.games_played % 2 == 0:
                self.bot1 = self._make_bot(1, pool_opponent, teacher_policy)
                self.bot2 = self._make_bot(2, self.student_net, student_policy)
                self.student_id = 2
                self.p1_gen_str, self.p2_gen_str = f"Gen #{pool_gen} (Teacher)", f"Gen #{self.teacher_gen} (Student)"
            else:
                self.bot1 = self._make_bot(1, self.student_net, student_policy)
                self.bot2 = self._make_bot(2, pool_opponent, teacher_policy)
                self.student_id = 1
                self.p1_gen_str, self.p2_gen_str = f"Gen #{self.teacher_gen} (Student)", f"Gen #{pool_gen} (Teacher)"
        else:
            teacher_policy = self._policy_for_gen(self.teacher_gen)
            self.bot1 = self._make_bot(1, self.teacher_net, teacher_policy, depth=4)
            self.bot2 = self._make_bot(2, self.teacher_net, teacher_policy, depth=4)
            self.p1_gen_str, self.p2_gen_str = "Human", f"Gen #{self.teacher_gen} (Teacher)"

        self.p1_draw_x, self.p1_draw_y = self._get_center_pixels(self.state['p1_pos'])
        self.p2_draw_x, self.p2_draw_y = self._get_center_pixels(self.state['p2_pos'])

    def load_checkpoint_if_exists(self):
        teacher_loaded = False
        if os.path.exists('quorinet_teacher_weights.pth'):
            try:
                self.teacher_net.load_state_dict(torch.load('quorinet_teacher_weights.pth', weights_only=True))
                teacher_loaded = True
            except Exception as exc:
                print(f"Checkpoint warning: teacher weights are incompatible: {exc}")

        if os.path.exists('quorinet_student_weights.pth'):
            try:
                self.student_net.load_state_dict(torch.load('quorinet_student_weights.pth', weights_only=True))
            except Exception as exc:
                print(f"Checkpoint warning: student weights are incompatible: {exc}")
        elif teacher_loaded:
            self.student_net.load_state_dict(self.teacher_net.state_dict())

        if os.path.exists('quorinet_optimizer.pth'):
            try:
                self.optimizer.load_state_dict(
                    torch.load('quorinet_optimizer.pth', weights_only=True)
                )
            except Exception as exc:
                print(f"Checkpoint warning: optimizer state was not loaded: {exc}")

        self.heuristic_weight = 1.0
        if os.path.exists('quorinet_meta.json'):
            try:
                with open('quorinet_meta.json', 'r') as f:
                    meta = json.load(f)
                    self.games_played = meta.get('games_played', 0)
                    self.teacher_gen = meta.get('teacher_gen', 1)
                    self.heuristic_weight = meta.get('heuristic_weight', 1.0)
                    self.wall_search_breadth = int(meta.get(
                        'wall_search_breadth', self.wall_search_breadth
                    ))
                    self.wall_exploration = float(meta.get(
                        'wall_exploration', self.wall_exploration
                    ))

                    raw_training = meta.get('training_metrics')
                    if raw_training is None:
                        raw_training = meta.get('checkpoint_log', {})
                        if raw_training:
                            print("Metadata migration: checkpoint_log -> training_metrics")
                    self.training_metrics = {
                        int(k): deque((float(x) for x in v),
                                      maxlen=self.TRAIN_METRIC_WINDOW)
                        for k, v in raw_training.items()
                    }
                    self.evaluation_metrics = {
                        int(k): dict(v)
                        for k, v in meta.get('evaluation_metrics', {}).items()
                    }
                    self.recovery_targets = {
                        int(k): dict(v)
                        for k, v in meta.get('recovery_targets', {}).items()
                    }
                    self.last_evaluation = meta.get('last_evaluation')
                    self.last_eval_latest_games_seen = int(meta.get(
                        'last_eval_latest_games_seen',
                        -self.EVAL_RETRY_AFTER_LATEST_GAMES
                    ))
                    self.last_promotion_status = meta.get(
                        'last_promotion_status', self.last_promotion_status
                    )
                    self.opponent_last_update = {
                        int(k): int(v)
                        for k, v in meta.get('opponent_last_update', {}).items()
                    }
                    self.generation_policy = {
                        int(k): dict(v)
                        for k, v in meta.get('generation_policy', {}).items()
                    }

                    latest_hist = meta.get('latest_teacher_history')
                    if latest_hist is None:
                        latest_hist = list(self.training_metrics.get(self.teacher_gen, []))
                    self.latest_teacher_history = deque(
                        (float(x) for x in latest_hist),
                        maxlen=self.LATEST_TEACHER_LOG_WINDOW
                    )

                    self.latest_teacher_games_seen = meta.get(
                        'latest_teacher_games_seen',
                        len(self.latest_teacher_history)
                    )
                    self.lr_pending_outcomes = [
                        float(x) for x in meta.get('lr_pending_outcomes', [])
                    ]
                    self.lr_measurements = list(meta.get('lr_measurements', []))
                    self.lr_no_improvement_count = int(
                        meta.get('lr_no_improvement_count', 0)
                    )
                    self.lr_next_eligible_game = int(
                        meta.get('lr_next_eligible_game', 0)
                    )
                    self.lr_adjustment_log = list(
                        meta.get('lr_adjustment_log', [])
                    )[-50:]
                    self.curriculum_latest_credit = meta.get(
                        'curriculum_latest_credit',
                        1.0
                    )

                    saved_lr = meta.get('current_lr')
                    if saved_lr is not None:
                        saved_lr = max(self.LR_MIN, min(self.LR_MAX, float(saved_lr)))
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = saved_lr

                    latest_wr = self._latest_teacher_win_rate()
                    if latest_wr is not None:
                        counts = self._outcome_counts(self.latest_teacher_history)
                        games = len(self.latest_teacher_history)
                        self.student_win_rate_str = (
                            f"Latest gate {latest_wr * 100:.1f}% "
                            f"({counts['wins']}W/{counts['draws']}D/{counts['losses']}L, n={games})"
                        )
            except Exception as e:
                print(f"Metadata warning: corrupted or incompatible file: {e}")

        self.generation_policy.setdefault(self.teacher_gen, {
            "heuristic_weight": self.heuristic_weight,
            "wall_search_breadth": self.wall_search_breadth,
            "wall_exploration": 0.0,
        })

    def save_meta(self):
        with open('quorinet_meta.json', 'w') as f:
            json.dump({
                'metadata_version': 2,
                'games_played': self.games_played,
                'teacher_gen': self.teacher_gen,
                'heuristic_weight': self.heuristic_weight,
                'wall_search_breadth': self.wall_search_breadth,
                'wall_exploration': self.wall_exploration,
                'latest_teacher_history': list(self.latest_teacher_history),
                'training_metrics': {k: list(v) for k, v in self.training_metrics.items()},
                'evaluation_metrics': self.evaluation_metrics,
                'recovery_targets': self.recovery_targets,
                'last_evaluation': self.last_evaluation,
                'last_eval_latest_games_seen': self.last_eval_latest_games_seen,
                'last_promotion_status': self.last_promotion_status,
                'opponent_last_update': self.opponent_last_update,
                'generation_policy': self.generation_policy,
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'latest_teacher_games_seen': self.latest_teacher_games_seen,
                'lr_pending_outcomes': self.lr_pending_outcomes,
                'lr_measurements': self.lr_measurements,
                'lr_no_improvement_count': self.lr_no_improvement_count,
                'lr_next_eligible_game': self.lr_next_eligible_game,
                'lr_adjustment_log': self.lr_adjustment_log,
                'curriculum_latest_credit': self.curriculum_latest_credit
            }, f, indent=2, sort_keys=True)

    def save_checkpoint(self, manual=False):
        self.generation_policy.setdefault(self.teacher_gen, {
            "heuristic_weight": self.heuristic_weight,
            "wall_search_breadth": self.wall_search_breadth,
            "wall_exploration": 0.0,
        })
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
    # SECTION 6: FROZEN PROMOTION EVALUATION
    # ==========================================
    def _evaluation_set(self):
        selected = {self.teacher_gen}
        recent_start = max(1, self.teacher_gen - self.RECENT_WINDOW_SIZE + 1)
        selected.update(range(recent_start, self.teacher_gen))
        selected.update(g for g in self.recovery_targets if g < self.teacher_gen)
        for gen in range(1, recent_start):
            _, n, confidence, deficit = self._performance(gen)
            if n >= self.MIN_SAMPLES_TRUSTED and confidence >= 0.5 and deficit > 0:
                selected.add(gen)
        return sorted(selected, reverse=True)

    def _evaluation_threshold(self, gen):
        role = self._role_for_gen(gen)
        if role == "Latest":
            return self.EVAL_LATEST_THRESHOLD
        if role in ("Recent", "Recovery Target"):
            return self.EVAL_RECENT_THRESHOLD
        return self.EVAL_HISTORIC_THRESHOLD

    def run_promotion_evaluation(self, games_per_model=None, seed=None):
        games_per_model = int(games_per_model or self.EVAL_GAMES_PER_MODEL)
        if games_per_model < 2:
            games_per_model = 2
        if games_per_model % 2:
            games_per_model += 1
        seed = int(self.EVAL_SEED + self.teacher_gen * 1000 if seed is None else seed)
        eval_gens = self._evaluation_set()
        result = {
            "overall_pass": True,
            "candidate_gen": self.teacher_gen + 1,
            "seed": seed,
            "games_per_model": games_per_model,
            "policy": {
                "heuristic_weight": self.EVAL_HEURISTIC_BLEND,
                "wall_search_breadth": self.EVAL_WALL_SEARCH_BREADTH,
                "wall_exploration": 0.0,
            },
            "per_generation": [],
            "failure_reasons": [],
            "sample_counts": {"completed": 0, "planned": len(eval_gens) * games_per_model},
            "skipped": [],
            "at_game": self.games_played,
        }
        print(
            f"\n--- Frozen Promotion Evaluation: candidate Gen "
            f"#{self.teacher_gen + 1}, seed={seed} ---"
        )
        random_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        student_mode = self.student_net.training
        teacher_mode = self.teacher_net.training
        lr_before = [group['lr'] for group in self.optimizer.param_groups]
        replay_sizes = (len(self.latest_replay_buffer), len(self.historic_replay_buffer))
        self.student_net.eval()
        self.teacher_net.eval()
        random.seed(seed)
        torch.manual_seed(seed)
        self.is_evaluating = True
        completed = 0
        eval_policy = result["policy"]
        try:
            for gen in eval_gens:
                if gen == self.teacher_gen:
                    opponent_net = self.teacher_net
                else:
                    opponent_net = self._load_net_for_gen(gen)
                if opponent_net is None:
                    missing = {"generation": gen, "reason": "checkpoint missing or incompatible"}
                    result["skipped"].append(missing)
                    result["failure_reasons"].append(
                        f"Gen #{gen} could not be evaluated: {missing['reason']}"
                    )
                    result["overall_pass"] = False
                    continue

                counts = {"wins": 0, "draws": 0, "losses": 0}
                for game_index in range(games_per_model):
                    completed += 1
                    if game_index % 2 == 0:
                        bot1 = self._make_bot(1, self.student_net, eval_policy)
                        bot2 = self._make_bot(2, opponent_net, eval_policy)
                        winner = self.play_eval_game(
                            bot1, bot2, completed, result["sample_counts"]["planned"], gen
                        )
                        outcome = 1.0 if winner == 1 else (0.5 if winner == 0 else 0.0)
                    else:
                        bot1 = self._make_bot(1, opponent_net, eval_policy)
                        bot2 = self._make_bot(2, self.student_net, eval_policy)
                        winner = self.play_eval_game(
                            bot1, bot2, completed, result["sample_counts"]["planned"], gen
                        )
                        outcome = 1.0 if winner == 2 else (0.5 if winner == 0 else 0.0)
                    if outcome == 1.0:
                        counts["wins"] += 1
                    elif outcome == 0.5:
                        counts["draws"] += 1
                    else:
                        counts["losses"] += 1

                samples = sum(counts.values())
                raw_score = (counts["wins"] + 0.5 * counts["draws"]) / samples
                smoothed = (counts["wins"] + 0.5 * counts["draws"] + 1.0) / (samples + 2.0)
                threshold = self._evaluation_threshold(gen)
                passed = smoothed >= threshold
                row = {
                    "generation": gen,
                    "role": self._role_for_gen(gen),
                    **counts,
                    "samples": samples,
                    "score": raw_score,
                    "smoothed_score": smoothed,
                    "threshold": threshold,
                    "passed": passed,
                    "at_game": self.games_played,
                }
                result["per_generation"].append(row)
                self.evaluation_metrics[gen] = dict(row)
                if not passed:
                    result["overall_pass"] = False
                    result["failure_reasons"].append(
                        f"Gen #{gen} {row['role']} score {smoothed:.3f} < {threshold:.3f}"
                    )
                print(
                    f"Vs Gen #{gen} [{row['role']}]: {counts['wins']}W/"
                    f"{counts['draws']}D/{counts['losses']}L; smoothed={smoothed:.3f}, "
                    f"required={threshold:.3f} [{'PASS' if passed else 'FAIL'}]"
                )
        finally:
            result["sample_counts"]["completed"] = completed
            self.is_evaluating = False
            random.setstate(random_state)
            torch.random.set_rng_state(torch_state)
            self.student_net.train(student_mode)
            self.teacher_net.train(teacher_mode)
            # Defensive assertions make accidental training or scheduler coupling
            # during evaluation immediately visible.
            if lr_before != [group['lr'] for group in self.optimizer.param_groups]:
                raise RuntimeError("Frozen evaluation changed the learning rate")
            if replay_sizes != (len(self.latest_replay_buffer), len(self.historic_replay_buffer)):
                raise RuntimeError("Frozen evaluation changed replay buffers")

        self.last_evaluation = result
        if not result["overall_pass"]:
            for row in result["per_generation"]:
                if not row["passed"]:
                    self.recovery_targets.setdefault(row["generation"], {})
                    self.recovery_targets[row["generation"]].update({
                        "reason": "failed promotion evaluation",
                        "since_game": self.games_played,
                        "eval_deficit": row["threshold"] - row["smoothed_score"],
                    })
            self.last_promotion_status = "Regression evaluation failed; candidate retained"
            self._refresh_opponent_weights()
        else:
            self.last_promotion_status = "Regression evaluation passed"
        return result

    def _promotion_gate_status(self):
        n = len(self.latest_teacher_history)
        rate = self._smoothed_rate(self.latest_teacher_history)
        passed = n >= self.GATE_MIN_GAMES and rate >= self.GATE_WIN_RATE_THRESHOLD
        return {
            "passed": passed,
            "samples": n,
            "smoothed_score": rate,
            "threshold": self.GATE_WIN_RATE_THRESHOLD,
            **self._outcome_counts(self.latest_teacher_history),
        }

    def _complete_successful_promotion(self, evaluation_result):
        if not evaluation_result.get("overall_pass"):
            raise ValueError("promotion requires a passing evaluation result")
        old_gen = self.teacher_gen
        candidate_policy = {
            "heuristic_weight": self.heuristic_weight,
            "wall_search_breadth": self.wall_search_breadth,
            "wall_exploration": 0.0,
        }
        # The previous teacher is preserved before the live object is replaced.
        torch.save(self.teacher_net.state_dict(), f'quorinet_teacher_gen_{old_gen}.pth')
        self.teacher_gen = old_gen + 1
        self.teacher_net.load_state_dict(self.student_net.state_dict())
        self.teacher_net.eval()
        self.generation_policy[self.teacher_gen] = candidate_policy

        self._roll_replay_buffers_for_new_teacher()
        self.latest_teacher_history.clear()
        self._reset_lr_scheduler_for_new_teacher()
        self.last_eval_latest_games_seen = -self.EVAL_RETRY_AFTER_LATEST_GAMES
        self.curriculum_latest_credit = 1.0
        for row in evaluation_result.get("per_generation", []):
            if row.get("passed"):
                self.recovery_targets.pop(int(row["generation"]), None)

        old_blend = self.heuristic_weight
        self.heuristic_weight = max(
            self.HEURISTIC_MIN, self.heuristic_weight * self.HEURISTIC_DECAY
        )
        self.last_promotion_status = (
            f"Promoted exactly once: Gen #{old_gen} -> Gen #{self.teacher_gen}; "
            f"heuristic blend {old_blend:.3f} -> {self.heuristic_weight:.3f}"
        )
        print(self.last_promotion_status)
        self.student_win_rate_str = "Latest gate: no samples"
        self.update_opponent_pool()
        self.save_checkpoint(manual=False)

    def _attempt_promotion(self):
        gate = self._promotion_gate_status()
        if not gate["passed"]:
            return False
        if (
            self.latest_teacher_games_seen - self.last_eval_latest_games_seen
            < self.EVAL_RETRY_AFTER_LATEST_GAMES
        ):
            return False
        print(
            f"Stage 1 passed vs latest Gen #{self.teacher_gen}: "
            f"smoothed={gate['smoothed_score']:.3f}, n={gate['samples']}. "
            "Starting frozen regression gate."
        )
        self.last_eval_latest_games_seen = self.latest_teacher_games_seen
        evaluation = self.run_promotion_evaluation()
        if not evaluation["overall_pass"]:
            print("Promotion blocked: " + "; ".join(evaluation["failure_reasons"]))
            self.update_opponent_pool()
            self.save_meta()
            return False
        self._complete_successful_promotion(evaluation)
        return True

    def train_step(self, state_tensor, turn_pid, next_state_tensor, next_turn_pid, done, winner_id):
        gamma = 0.99
        if done:
            target = 0.0 if winner_id == 0 else (1.0 if turn_pid == winner_id else -1.0)
        else:
            with torch.no_grad():
                next_val = self.teacher_net(next_state_tensor.unsqueeze(0)).item()
            target = gamma * next_val if next_turn_pid == turn_pid else gamma * (-next_val)

        opponent_gen = self.current_opponent_gen
        was_latest = self.current_opponent_is_latest

        # Keep the original stable one-step teacher target.
        self._append_replay_sample(state_tensor, target, opponent_gen)

        # Retain the episode long enough to propagate the terminal result for
        # latest, recent, and historical games alike.
        self.episode_trajectory.append(
            (state_tensor, turn_pid, target, opponent_gen, was_latest)
        )

        if done:
            if self.episode_trajectory:
                final_index = len(self.episode_trajectory) - 1
                for i, (
                    episode_state,
                    episode_pid,
                    bootstrap_target,
                    episode_gen,
                    episode_was_latest
                ) in enumerate(self.episode_trajectory):
                    if winner_id == 0:
                        outcome = 0.0
                    else:
                        outcome = 1.0 if episode_pid == winner_id else -1.0

                    horizon = final_index - i
                    discounted_outcome = (
                        outcome * (self.OUTCOME_BACKFILL_DISCOUNT ** horizon)
                    )
                    mixed_target = (
                        self.OUTCOME_BACKFILL_MIX * discounted_outcome
                        + (1.0 - self.OUTCOME_BACKFILL_MIX) * bootstrap_target
                    )
                    self._append_replay_sample(
                        episode_state,
                        mixed_target,
                        episode_gen
                    )

            self.episode_trajectory.clear()

        batch = self._sample_replay_batch()
        if batch is not None:
            self.student_net.train()
            states = torch.stack([item[0] for item in batch])
            targets = torch.tensor(
                [[item[1]] for item in batch],
                dtype=torch.float32
            )
            self.optimizer.zero_grad()
            predictions = self.student_net(states)
            loss = self.loss_fn(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.student_net.parameters(),
                self.GRAD_CLIP_NORM
            )
            self.optimizer.step()
            self.last_loss = loss.item()
            self.student_net.eval()

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
        card = pygame.Rect(padding, int(15 * SCALE), card_w, HEIGHT - int(30 * SCALE))
        pygame.draw.rect(self.virtual_screen, CARD_BG, card, border_radius=int(10 * SCALE))
        title = self.small_font.render("OPPONENT PERFORMANCE", True, TEXT_COLOR)
        self.virtual_screen.blit(title, (card.x + int(10 * SCALE), card.y + int(8 * SCALE)))
        pygame.draw.line(self.virtual_screen, (235, 238, 242),
                         (card.x + int(10 * SCALE), card.y + int(28 * SCALE)),
                         (card.right - int(10 * SCALE), card.y + int(28 * SCALE)))
        header = "Gen  Role       Train/n     Eval/n      Next"
        self.virtual_screen.blit(
            self.tiny_font.render(header, True, MUTED_TEXT),
            (card.x + int(8 * SCALE), card.y + int(33 * SCALE))
        )
        y_offset = card.y + int(52 * SCALE)
        rows = sorted(
            ((gen, weight) for (_, gen), weight in
             zip(self.opponent_pool, self.opponent_weights)),
            reverse=True
        )
        for gen, weight in rows:
            if y_offset > card.bottom - int(35 * SCALE):
                break
            train_log = self.training_metrics.get(gen, ())
            train_rate = self._smoothed_rate(train_log) * 100 if train_log else None
            evaluation = self.evaluation_metrics.get(gen)
            eval_rate = evaluation.get("smoothed_score", 0.0) * 100 if evaluation else None
            role = self._role_for_gen(gen)
            short_role = {
                "Latest": "Latest", "Recent": "Recent",
                "Historic": "Historic", "Recovery Target": "Recovery"
            }[role]
            train_text = "--" if train_rate is None else f"{train_rate:4.1f}/{len(train_log)}"
            eval_text = "--" if eval_rate is None else f"{eval_rate:4.1f}/{evaluation['samples']}"
            first = f"#{gen:<3} {short_role:<10} {train_text:<11} {eval_text:<11} {weight*100:4.1f}%"
            color = P1_COLOR if gen == self.current_opponent_gen else TEXT_COLOR
            self.virtual_screen.blit(
                self.tiny_font.render(first, True, color),
                (card.x + int(8 * SCALE), y_offset)
            )
            trusted = "trusted" if len(train_log) >= self.MIN_SAMPLES_TRUSTED else "low trust"
            status = "--" if not evaluation else ("PASS" if evaluation.get("passed") else "FAIL")
            multiplier = self._recovery_multiplier(gen)
            mult_text = "" if multiplier is None else f" recovery {multiplier:.2f}x"
            updated = self.opponent_last_update.get(gen)
            if evaluation:
                updated = max(updated or 0, int(evaluation.get("at_game", 0)))
            age = "never" if updated is None else f"{max(0, self.games_played-updated)}g ago"
            second = f"     {trusted} | {status}{mult_text} | updated {age}"
            self.virtual_screen.blit(
                self.tiny_font.render(second, True, MUTED_TEXT),
                (card.x + int(8 * SCALE), y_offset + int(14 * SCALE))
            )
            y_offset += int(32 * SCALE)

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
        gate = self._promotion_gate_status()
        eval_counter_str = (
            f"{'PASS' if gate['passed'] else 'WAIT'} "
            f"{gate['smoothed_score']*100:.1f}% n={gate['samples']}/{self.GATE_MIN_GAMES}"
        )
        current_lr = self.optimizer.param_groups[0]['lr']
        eval_phase_str = f"Vs Gen #{self.eval_vs_gen} ({self.eval_game_curr}/{self.eval_game_total})" if self.is_evaluating else "Idle"
        if self.last_evaluation:
            failed_gens = [
                str(row["generation"])
                for row in self.last_evaluation.get("per_generation", [])
                if not row.get("passed")
            ]
            regression_status = (
                "PASS" if self.last_evaluation.get("overall_pass")
                else "FAIL Gen " + ",".join(failed_gens)
            )
        else:
            regression_status = "Not run"

        metrics = [
            ("Red Player (P1)", self.p1_gen_str),
            ("Blue Player (P2)", self.p2_gen_str),
            ("Latest Teacher Gen", f"Gen #{self.teacher_gen}"),
            ("Heuristic Blend", f"{self.heuristic_weight:.3f} (1=heuristic)"),
            ("Latest WR (rolling)", self.student_win_rate_str),
            ("Promotion Gate", eval_counter_str),
            ("Regression Gate", regression_status),
            ("Recovery Targets", str(sorted(self.recovery_targets)) or "None"),
            ("Frozen Eval", eval_phase_str),
            ("AI Board Evaluation", f"{eval_score}"),
            ("Scheduled Learning Rate", f"{current_lr:.6f}"),
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
    def check_win(self):
        winner_id = 0
        if self.state['p1_pos'][1] == 0:
            self.game_over, winner_id = True, 1
        elif self.state['p2_pos'][1] == 8:
            self.game_over, winner_id = True, 2
        elif self.state['turns_played'] >= 200:
            self.game_over, winner_id = True, 0

        if self.game_over:
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
                if self.games_played % 250 == 0:
                    self.save_checkpoint(manual=False)

                gen = self.current_opponent_gen
                outcome = (
                    0.5 if winner_id == 0 else
                    (1.0 if winner_id == self.student_id else 0.0)
                )
                log = self._training_log(gen)
                targets_before = set(self.recovery_targets)
                log.append(outcome)
                self.opponent_last_update[gen] = self.games_played
                self._update_recovery_from_training(gen)

                if self.current_opponent_is_latest:
                    self.latest_teacher_history.append(outcome)
                    self.latest_teacher_games_seen += 1
                    self._update_lr_scheduler(outcome)

                if self.latest_teacher_history:
                    counts = self._outcome_counts(self.latest_teacher_history)
                    latest_wr = sum(self.latest_teacher_history) / len(self.latest_teacher_history)
                    self.student_win_rate_str = (
                        f"{latest_wr*100:.1f}% "
                        f"({counts['wins']}W/{counts['draws']}D/{counts['losses']}L, "
                        f"n={len(self.latest_teacher_history)})"
                    )

                evidence_due = (
                    self.games_played - self.last_weight_refresh_game
                    >= self.WEIGHT_REFRESH_EVIDENCE_INTERVAL
                )
                if evidence_due or targets_before != set(self.recovery_targets):
                    self._refresh_opponent_weights()
                    self.last_weight_refresh_game = self.games_played

                self.save_meta()
                if self.current_opponent_is_latest:
                    self._attempt_promotion()

    def run_headless_training(self, num_games):
        print(f"Starting Headless Turbo Training for {num_games} games...")
        self.mode = "TRAIN"
        self.is_training_active = True
        self.reset_game()
        
        def command_listener():
            while True:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'p': self.is_training_active = not self.is_training_active
                elif cmd == 's': self.save_checkpoint(manual=True)
                elif cmd == 'q': os._exit(0)
                    
        threading.Thread(target=command_listener, daemon=True).start()
        target_games, last_print = self.games_played + num_games, self.games_played
        
        while self.games_played < target_games:
            if not self.is_training_active:
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
            
            # Resolution logic: Disable epsilon entirely for Teacher, enable for Student
            is_teacher = (active_bot.nn_model != self.student_net)
            eps = 0.0 if is_teacher else (0.30 * (0.85 ** self.state['turns_played']))
            
            action, _ = active_bot.get_move_and_eval(self.state, epsilon=eps)
            if action: self.state = active_bot.apply_action(self.state, pre_action_pid, action, is_random=getattr(active_bot, 'was_last_random', False))
            else: self.state['turns_played'] = 200
                
            self.check_win()
            next_state_tensor = encode_state(self.state, self.state['turn'])
            next_pid = self.state['turn']
            winner_id = getattr(self, 'winner_id', 0) if self.game_over else 0
            
            self.train_step(pre_action_state_tensor, pre_action_pid, next_state_tensor, next_pid, self.game_over, winner_id)
                
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

                    is_teacher = (active_bot.nn_model != self.student_net)
                    eps = 0.0 if is_teacher else (0.30 * (0.85 ** self.state['turns_played']))
                    
                    action, _ = active_bot.get_move_and_eval(self.state, epsilon=eps)
                    if action: self.state = active_bot.apply_action(self.state, pre_action_pid, action, is_random=getattr(active_bot, 'was_last_random', False))
                    else: self.state['turns_played'] = 200

                    self.check_win()
                    next_state_tensor = encode_state(self.state, self.state['turn'])
                    next_pid = self.state['turn']
                    winner_id = getattr(self, 'winner_id', 0) if self.game_over else 0
                    self.train_step(pre_action_state_tensor, pre_action_pid, next_state_tensor, next_pid, self.game_over, winner_id)

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


def run_internal_checks():
    """Deterministic focused checks for curriculum invariants."""
    checks = []

    def verify(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
        print(f"[PASS] {name}")

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="quoridor_checks_") as temp_dir:
        os.chdir(temp_dir)
        try:
            game = Quoridor(headless=True)
            game.teacher_gen = 3
            for gen in (1, 2):
                torch.save(game.teacher_net.state_dict(), f"quorinet_teacher_gen_{gen}.pth")
            game.update_opponent_pool()

            before_probability = game._probability_for_gen(2)
            before_generation = game.teacher_gen
            before_latest = list(game.latest_teacher_history)
            game._training_log(2).append(0.0)
            game._update_recovery_from_training(2)
            game._refresh_opponent_weights()
            verify(
                "one older loss is recorded without reset or extreme reweighting",
                len(game.training_metrics[2]) == 1
                and 2 not in game.recovery_targets
                and game.teacher_gen == before_generation
                and list(game.latest_teacher_history) == before_latest
                and abs(game._probability_for_gen(2) - before_probability) < 0.05,
            )

            game.training_metrics[2].clear()
            game.training_metrics[2].extend([0.0] * game.MIN_SAMPLES_TRUSTED)
            game._update_recovery_from_training(2)
            game._refresh_opponent_weights()
            verify(
                "reliable underperformance creates a recovery target",
                2 in game.recovery_targets,
            )
            verify(
                "latest and failing recent teachers have meaningful shares",
                game._probability_for_gen(3) >= game.LATEST_MIN_SHARE - 1e-12
                and game._probability_for_gen(2) >= game.RECOVERY_MIN_SHARE - 1e-12,
            )
            verify(
                "opponent probabilities normalize exactly",
                math.isclose(sum(game.opponent_weights), 1.0, abs_tol=1e-12),
            )
            verify(
                "UI roles follow allocation logic",
                game._role_for_gen(3) == "Latest"
                and game._role_for_gen(2) == "Recovery Target"
                and game._role_for_gen(1) == "Recent",
            )

            capped = game._apply_escalation_cap(
                {1: 100.0, 2: 2.0, 3: 1.0}, total_mass=0.6, max_share=0.2
            )
            verify(
                "water-fill caps survive redistribution",
                math.isclose(sum(capped.values()), 0.6, abs_tol=1e-12)
                and all(value <= 0.2 + 1e-12 for value in capped.values()),
            )

            parameter_before = [p.detach().clone() for p in game.teacher_net.parameters()]
            lr_before = game.optimizer.param_groups[0]['lr']
            replay_before = (len(game.latest_replay_buffer), len(game.historic_replay_buffer))
            latest_before = list(game.latest_teacher_history)
            training_before = {g: list(v) for g, v in game.training_metrics.items()}
            heuristic_before = game.heuristic_weight
            game.play_eval_game = lambda *args, **kwargs: 2
            failed = game.run_promotion_evaluation(games_per_model=2, seed=123)
            verify(
                "failed frozen evaluation preserves promotion lifecycle",
                not failed["overall_pass"]
                and game.teacher_gen == 3
                and game.heuristic_weight == heuristic_before
                and list(game.latest_teacher_history) == latest_before
                and replay_before == (len(game.latest_replay_buffer), len(game.historic_replay_buffer))
                and all(torch.equal(a, b) for a, b in zip(parameter_before, game.teacher_net.parameters())),
            )
            verify(
                "evaluation does not train or change LR",
                game.optimizer.param_groups[0]['lr'] == lr_before,
            )
            verify(
                "training and evaluation metrics stay separate",
                training_before == {g: list(v) for g, v in game.training_metrics.items()}
                and bool(game.evaluation_metrics),
            )

            # Exercise the public two-stage control flow: a passing evaluation
            # can promote once; the cleared stage-one window prevents a repeat.
            game.latest_teacher_history.extend([1.0] * game.GATE_MIN_GAMES)
            game.latest_teacher_games_seen = game.GATE_MIN_GAMES
            passing = {
                "overall_pass": True,
                "per_generation": [
                    {"generation": 3, "passed": True},
                    {"generation": 2, "passed": True},
                ],
                "failure_reasons": [],
            }
            game.run_promotion_evaluation = lambda *args, **kwargs: copy.deepcopy(passing)
            game.save_checkpoint = lambda manual=False: None
            old_gen = game.teacher_gen
            old_blend = game.heuristic_weight
            first = game._attempt_promotion()
            second = game._attempt_promotion()
            verify(
                "successful evaluation promotes and decays exactly once",
                first and not second
                and game.teacher_gen == old_gen + 1
                and math.isclose(
                    game.heuristic_weight,
                    max(game.HEURISTIC_MIN, old_blend * game.HEURISTIC_DECAY),
                    abs_tol=1e-12,
                ),
            )

            old_meta = {
                "games_played": 42,
                "teacher_gen": 2,
                "heuristic_weight": 0.7,
                "student_eval_history": [1, 0, 1],
                "checkpoint_log": {"1": [1, 0, 0], "2": [1, 1, 0]},
                "latest_teacher_history": [1, 0, 1],
                "wawr_history": [0.5],
                "lr_search_direction": 1,
            }
            with open("quorinet_meta.json", "w") as meta_file:
                json.dump(old_meta, meta_file)
            legacy = Quoridor(headless=True)
            verify(
                "legacy metadata migrates without discarding checkpoints",
                legacy.games_played == 42
                and list(legacy.training_metrics[1]) == [1.0, 0.0, 0.0]
                and legacy._load_net_for_gen(1) is not None,
            )

            smoke = QuoridorBot(1, max_depth=1, nn_model=legacy.student_net,
                                heuristic_weight=legacy.heuristic_weight)
            action = smoke.get_move(legacy.state)
            verify(
                "short headless policy smoke test produces a legal action",
                action in smoke.generate_actions(legacy.state, 1),
            )
        finally:
            os.chdir(original_cwd)
            pygame.quit()
    print(f"Internal checks complete: {len(checks)}/{len(checks)} passed.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_internal_checks()
        sys.exit(0)

    if args.headless > 0:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    game = Quoridor(headless=(args.headless > 0))
    if args.headless > 0: game.run_headless_training(args.headless)
    else: game.play()
