import pygame
import sys
import os
import json
import random
import time
import math
import threading
import argparse
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
# NEURAL NETWORK ARCHITECTURE
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

        scored_actions = []
        for action in actions_copy:
            next_state = self.apply_action(state, pid, action)
            val = self.evaluate(next_state, pid)
            scored_actions.append((val, action))

        scored_actions.sort(key=lambda x: x[0], reverse=True)
        return [act for _, act in scored_actions[:top_k]]

    def _oscillation_penalty(self, state, pid):
        my_pos = state['p1_pos'] if pid == 1 else state['p2_pos']
        recent = state.get('p1_recent', ()) if pid == 1 else state.get('p2_recent', ())
        if not recent:
            return 0.0
        penalty = 0.0
        n = len(recent)
        for i, pos in enumerate(recent):
            if pos == my_pos:
                recency_weight = (i + 1) / n
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

        if state['p1_walls'] == 0 and state['p2_walls'] == 0:
            score = (d2 - d1) * 50.0 + (p1_progress - p2_progress) * 10.0
            if eval_pid == 2: score = -score
            if is_repeated: score -= 8000
            score -= osc_penalty
            return score

        if self.nn_model is not None:
            with torch.no_grad():
                nn_val = self.nn_model(encode_state(state, eval_pid).unsqueeze(0)).item()
            
            nn_score = nn_val * 100.0
            heur_score = (d2 - d1) * 5.0
            if eval_pid == 2: heur_score = -heur_score
            
            score = nn_score + heur_score
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
BOARD_SIZE = 9
PADDING = int(25 * SCALE)
BOARD_PIXEL_SIZE = (BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * WALL_THICKNESS + (PADDING * 2))

SIDEBAR_WIDTH = int(380 * SCALE)
LEFT_SIDEBAR_WIDTH = int(370 * SCALE)
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

        self.RECENT_WINDOW_SIZE = 5
        self.RECENT_BUCKET_MASS = 0.60
        self.HISTORIC_BUCKET_MASS = 0.40
        self.RECENT_DECAY_RATIO = 2.0
        self.LOG_WINDOW_SIZE = 30
        self.MIN_SAMPLES_TRUSTED = 15
        self.GATE_MIN_GAMES = 30
        self.GATE_WIN_RATE_THRESHOLD = 0.57
        self.FINAL_TEST_GAMES_PER_MODEL = 10
        self.ESCALATION_LOSS_THRESHOLD = 0.50
        self.ESCALATION_RECOVER_THRESHOLD = 0.65
        self.ESCALATION_STEP_MULTIPLIER = 1.5
        self.ESCALATION_MAX_MULTIPLIER = 8.0
        self.ESCALATION_STALL_LIMIT = 5
        self.ESCALATION_MAX_SHARE_OF_HISTORIC_BUCKET = 0.15

        self.LATEST_TEACHER_LOG_WINDOW = 60
        self.CURRICULUM_UNIFORM_MASS = 0.25

        self.REPLAY_LATEST_CAPACITY = 140000
        self.REPLAY_HISTORIC_CAPACITY = 60000
        self.REPLAY_WARMUP = 256
        self.REPLAY_BATCH_SIZE = 128

        self.OUTCOME_BACKFILL_MIX = 0.70
        self.OUTCOME_BACKFILL_DISCOUNT = 0.995

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

        self.student_net = SpatialQuoriNetV2(in_channels=8)
        self.teacher_net = SpatialQuoriNetV2(in_channels=8)
        self.student_net.eval()
        self.teacher_net.eval()
        self.optimizer = optim.Adam(self.student_net.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()

        self.games_played = 0
        self.teacher_gen = 1
        self.heuristic_weight = 1.0
        self.heuristic_lr_floor = self.optimizer.param_groups[0]['lr']

        self.student_eval_history = deque(maxlen=100)
        self.latest_teacher_history = deque(maxlen=self.LATEST_TEACHER_LOG_WINDOW)
        self.student_win_rate_str = "Pending live WR"
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

        self.checkpoint_log = {}
        self.escalation_state = {}
        self.wawr_history = []
        self.lr_last_reduced_at_gen = 0
        self.current_opponent_gen = 1
        self.current_opponent_is_latest = True
        self.curriculum_latest_credit = 1.0

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
        self.manual_save_requested = False
        self.toast_msg = ""
        self.toast_timer = 0
        
        self.rect_play, self.rect_train, self.rect_new_game, self.rect_menu, self.rect_train_toggle = [pygame.Rect(0, 0, 0, 0) for _ in range(5)]
        self.game_start_time = time.time()
        self.reset_game()

    def update_opponent_pool(self):
        self.opponent_pool = []
        self.opponent_support_base_weights = []

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
        return sorted({gen for _, gen in self.opponent_pool})

    def _required_win_rate(self, gen):
        """Win rate target modeled by f(x, N) = 95 - 38 * e^(-0.15 * (N - x)).
        Where N is self.teacher_gen and x is gen.
        Returns target threshold as a fraction (0.57 to 0.95).
        """
        if gen >= self.teacher_gen:
            return self.GATE_WIN_RATE_THRESHOLD
        
        N = self.teacher_gen
        x = gen
        req_pct = 95.0 - 38.0 * math.exp(-0.15 * (N - x))
        return req_pct / 100.0

    def _generation_win_rate(self, gen):
        log = self.checkpoint_log.get(gen)
        if not log:
            return None
        return sum(log) / len(log)

    def _passes_generation_gate(self, gen, win_rate):
        required = self._required_win_rate(gen)
        if gen == self.teacher_gen:
            return win_rate > required
        return win_rate >= required

    def _all_training_gates_met(self):
        generations = self._training_generations()
        if not generations:
            return False
        for gen in generations:
            log = self.checkpoint_log.get(gen)
            if not log or len(log) < self.GATE_MIN_GAMES:
                return False
            if not self._passes_generation_gate(gen, sum(log) / len(log)):
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
                parts.append(
                    f"Gen #{gen}: WR {sum(log) / len(log) * 100:.1f}% "
                    f"| Sample {sampling:.1f}%"
                )
            else:
                parts.append(
                    f"Gen #{gen}: WR pending | Sample {sampling:.1f}%"
                )
        return " | ".join(parts) if parts else "No training opponents"

    def _weighted_pool_metric_and_trend(self):
        weight_by_gen = {}
        for (_, gen), weight in zip(self.opponent_pool, self.opponent_weights):
            weight_by_gen[gen] = weight_by_gen.get(gen, 0.0) + weight

        weighted_metric = 0.0
        weighted_target = 0.0
        weighted_trend = 0.0
        total_weight = 0.0
        total_samples = 0

        for gen in self._training_generations():
            history = list(self.checkpoint_log.get(gen, ()))
            if not history:
                continue

            weight = weight_by_gen.get(gen, 0.0)
            if weight <= 0.0:
                weight = 1.0

            metric = sum(history) / len(history)
            half = min(15, len(history) // 2)
            if half:
                older = history[-2 * half:-half]
                newer = history[-half:]
                trend = (sum(newer) / len(newer)) - (sum(older) / len(older))
            else:
                trend = 0.0

            confidence = min(1.0, len(history) / self.LR_MIN_SAMPLES)
            effective_weight = weight * confidence
            weighted_metric += metric * effective_weight
            weighted_target += self._required_win_rate(gen) * effective_weight
            weighted_trend += trend * effective_weight
            total_weight += effective_weight
            total_samples += len(history)

        if total_weight <= 0.0:
            return None
        return (
            weighted_metric / total_weight,
            weighted_target / total_weight,
            weighted_trend / total_weight,
            total_samples
        )

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

        # The support weights encode the intended recent/historic curriculum.
        # Normalize them across support opponents, while keeping the latest
        # teacher on the same average scale, then use them as deficit priors.
        if len(self.opponent_support_base_weights) == count:
            support_indices = [
                i for i, (_, gen) in enumerate(self.opponent_pool)
                if gen != self.teacher_gen
            ]
            support_total = sum(
                self.opponent_support_base_weights[i] for i in support_indices
            )
            if support_indices and support_total > 0.0:
                support_scale = len(support_indices) / support_total
                for i in support_indices:
                    deficit_scores[i] *= (
                        self.opponent_support_base_weights[i] * support_scale
                    )

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
        return batch

    def _roll_replay_buffers_for_new_teacher(self):
        while self.latest_replay_buffer:
            self.historic_replay_buffer.append(self.latest_replay_buffer.popleft())

    def _set_learning_rate(self, requested_lr, reason):
        old_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(self.LR_MIN, min(self.LR_MAX, requested_lr))
        if abs(new_lr - old_lr) < 1e-12:
            return False

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(f"AutoLR: {old_lr:.6f} -> {new_lr:.6f} | {reason}")
        previous_floor = getattr(self, 'heuristic_lr_floor', old_lr)
        if new_lr < previous_floor:
            old_hw = self.heuristic_weight
            self.heuristic_weight = max(
                0.1,
                self.heuristic_weight * (new_lr / previous_floor)
            )
            self.heuristic_lr_floor = new_lr
            if self.heuristic_weight < old_hw:
                print(
                    f"AutoLR: Heuristic dependency {old_hw:.3f} -> "
                    f"{self.heuristic_weight:.3f} (new LR floor)"
                )
        return True

    def _reset_lr_search_for_new_teacher(self):
        self.latest_teacher_games_seen = 0
        self.lr_training_games_seen = 0
        self.lr_metric_history.clear()
        self.lr_last_tuned_at_training_game = 0
        self.lr_best_metric = -1.0
        self.lr_best_value = self.optimizer.param_groups[0]['lr']
        self.lr_search_direction = 0

    def _get_dynamic_lr_up_factor(self):
        """Scales LR exploration search margin inversely to generation age N (self.teacher_gen)
        using the required WR exponential decay factor e^(-0.15 * (N - 1)).
        """
        N = self.teacher_gen
        margin_decay = math.exp(-0.15 * (N - 1))
        base_margin = self.LR_UP_FACTOR - 1.0
        dynamic_margin = max(0.02, base_margin * margin_decay)
        return 1.0 + dynamic_margin

    def _tune_learning_rate_from_pool(self, force=False):
        pool_metric = self._weighted_pool_metric_and_trend()
        if pool_metric is None:
            return
        metric, target, trend, samples = pool_metric
        if samples < self.LR_MIN_SAMPLES:
            return

        if (
            not force
            and self.lr_training_games_seen - self.lr_last_tuned_at_training_game
            < self.LR_TUNE_INTERVAL_LATEST_GAMES
        ):
            return

        previous_metric = self.lr_metric_history[-1] if self.lr_metric_history else None
        current_lr = self.optimizer.param_groups[0]['lr']
        dynamic_up_factor = self._get_dynamic_lr_up_factor()

        if metric > self.lr_best_metric + self.LR_PLATEAU_EPS:
            self.lr_best_metric = metric
            self.lr_best_value = current_lr

        reason = None
        candidate_lr = current_lr

        if previous_metric is None:
            if metric < target and trend <= self.LR_PLATEAU_EPS:
                candidate_lr = current_lr * dynamic_up_factor
                self.lr_search_direction = 1
                reason = (
                    f"initial pool probe; weighted WR={metric:.3f}, "
                    f"target={target:.3f}, trend={trend:+.3f}"
                )
            elif metric >= target:
                candidate_lr = current_lr * self.LR_CONVERGE_FACTOR
                self.lr_search_direction = -1
                reason = (
                    f"pool already competitive; weighted WR={metric:.3f}, "
                    f"target={target:.3f}"
                )
            else:
                self.lr_search_direction = 0
                reason = (
                    f"hold; weighted pool WR improving ({metric:.3f}, "
                    f"trend={trend:+.3f})"
                )
        else:
            delta = metric - previous_metric

            if metric < target:
                if delta > self.LR_PLATEAU_EPS or trend > self.LR_PLATEAU_EPS:
                    self.lr_search_direction = 0
                    reason = (
                        f"hold; weighted pool WR improving ({metric:.3f}, "
                        f"delta={delta:+.3f}, trend={trend:+.3f})"
                    )
                elif (
                    delta < -self.LR_REGRESSION_EPS
                    or trend < -self.LR_REGRESSION_EPS
                ):
                    if self.lr_best_metric >= 0.0 and abs(self.lr_best_value - current_lr) > 1e-12:
                        candidate_lr = (current_lr * self.lr_best_value) ** 0.5
                        reason = (
                            f"weighted-pool regression; returning toward best LR "
                            f"{self.lr_best_value:.6f} (best WR={self.lr_best_metric:.3f})"
                        )
                    else:
                        candidate_lr = current_lr * self.LR_DOWN_FACTOR
                        reason = (
                            f"weighted-pool regression; damping LR at WR={metric:.3f}"
                        )
                    self.lr_search_direction = -1
                else:
                    candidate_lr = current_lr * dynamic_up_factor
                    self.lr_search_direction = 1
                    reason = (
                        f"weighted-pool plateau below target; exploration increase "
                        f"(WR={metric:.3f}, target={target:.3f}, trend={trend:+.3f})"
                    )
            else:
                if (
                    delta < -self.LR_REGRESSION_EPS
                    or trend < -self.LR_REGRESSION_EPS
                ):
                    candidate_lr = current_lr * self.LR_DOWN_FACTOR
                    reason = (
                        f"weighted pool competitive but regressing; stabilize "
                        f"(WR={metric:.3f})"
                    )
                else:
                    candidate_lr = current_lr * self.LR_CONVERGE_FACTOR
                    reason = (
                        f"weighted pool competitive; convergence decay "
                        f"(WR={metric:.3f}, trend={trend:+.3f})"
                    )
                self.lr_search_direction = -1

        if reason is not None and candidate_lr != current_lr:
            self._set_learning_rate(candidate_lr, reason)
        elif reason is not None:
            print(f"AutoLR: {reason}; LR held at {current_lr:.6f}")

        self.lr_metric_history.append(metric)
        self.lr_last_tuned_at_training_game = self.lr_training_games_seen

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
        print(f"Heuristic Dep:   {self.heuristic_weight:.3f}")
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
                self.bot1 = QuoridorBot(bot_id=1, max_depth=2, nn_model=pool_opponent, heuristic_weight=self.heuristic_weight)
                self.bot2 = QuoridorBot(bot_id=2, max_depth=2, nn_model=self.student_net, heuristic_weight=self.heuristic_weight)
                self.student_id = 2
                self.p1_gen_str, self.p2_gen_str = f"Gen #{pool_gen} (Teacher)", f"Gen #{self.teacher_gen + 1} (Student)"
            else:
                self.bot1 = QuoridorBot(bot_id=1, max_depth=2, nn_model=self.student_net, heuristic_weight=self.heuristic_weight) 
                self.bot2 = QuoridorBot(bot_id=2, max_depth=2, nn_model=pool_opponent, heuristic_weight=self.heuristic_weight) 
                self.student_id = 1
                self.p1_gen_str, self.p2_gen_str = f"Gen #{self.teacher_gen + 1} (Student)", f"Gen #{pool_gen} (Teacher)"
        else:
            self.bot1 = QuoridorBot(bot_id=1, max_depth=4, nn_model=self.teacher_net, heuristic_weight=self.heuristic_weight)
            self.bot2 = QuoridorBot(bot_id=2, max_depth=4, nn_model=self.teacher_net, heuristic_weight=self.heuristic_weight)
            self.p1_gen_str, self.p2_gen_str = "Human", f"Gen #{self.teacher_gen} (Teacher)"

        self.p1_draw_x, self.p1_draw_y = self._get_center_pixels(self.state['p1_pos'])
        self.p2_draw_x, self.p2_draw_y = self._get_center_pixels(self.state['p2_pos'])

    def load_checkpoint_if_exists(self):
        teacher_loaded = False
        if os.path.exists('quorinet_teacher_weights.pth'):
            try:
                self.teacher_net.load_state_dict(torch.load('quorinet_teacher_weights.pth', weights_only=True))
                teacher_loaded = True
            except Exception as e:
                raise RuntimeError(
                    "Teacher checkpoint exists but could not be loaded; "
                    "refusing to continue with random teacher weights."
                ) from e

        if os.path.exists('quorinet_student_weights.pth'):
            try:
                self.student_net.load_state_dict(torch.load('quorinet_student_weights.pth', weights_only=True))
            except Exception as e:
                print(f"Student checkpoint load warning: {e}")
                if teacher_loaded:
                    self.student_net.load_state_dict(self.teacher_net.state_dict())
        elif teacher_loaded:
            self.student_net.load_state_dict(self.teacher_net.state_dict())

        if os.path.exists('quorinet_optimizer.pth'):
            try:
                self.optimizer.load_state_dict(
                    torch.load('quorinet_optimizer.pth', weights_only=True)
                )
            except Exception as e:
                print(f"Optimizer checkpoint load warning: {e}")

        self.heuristic_weight = 1.0
        if os.path.exists('quorinet_meta.json'):
            try:
                with open('quorinet_meta.json', 'r') as f:
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

                    saved_lr = meta.get('current_lr')
                    if saved_lr is not None:
                        saved_lr = max(self.LR_MIN, min(self.LR_MAX, float(saved_lr)))
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = saved_lr

                    current_lr = self.optimizer.param_groups[0]['lr']
                    if 'heuristic_lr_floor' in meta:
                        self.heuristic_lr_floor = min(
                            current_lr,
                            float(meta['heuristic_lr_floor'])
                        )
                    else:
                        repaired_weight = max(
                            self.heuristic_weight,
                            max(0.1, min(1.0, current_lr / 0.0001))
                        )
                        if repaired_weight > self.heuristic_weight:
                            print(
                                f"Metadata migration: repaired heuristic dependency "
                                f"{self.heuristic_weight:.3f} -> {repaired_weight:.3f}"
                            )
                            self.heuristic_weight = repaired_weight
                        self.heuristic_lr_floor = current_lr

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
                print(f"Metadata load warning: {e}")

    def save_meta(self):
        with open('quorinet_meta.json', 'w') as f:
            json.dump({
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
                'heuristic_lr_floor': self.heuristic_lr_floor
            }, f)

    def save_checkpoint(self, manual=False):
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
                f"{'>' if gen == self.teacher_gen else '>='} "
                f"{self._required_win_rate(gen) * 100:.1f}% "
                f"[{'PASS' if passed else 'FAIL'}]"
            )

        if not all_passed:
            print("Final test failed. Candidate returns to training unchanged.")

        return all_passed

    def train_step(self, state_tensor, turn_pid, next_state_tensor, next_turn_pid, done, winner_id):
        gamma = 0.99
        if done:
            target = 0.0 if winner_id == 0 else (1.0 if turn_pid == winner_id else -1.0)
        else:
            with torch.no_grad():
                next_val = self.teacher_net(next_state_tensor.unsqueeze(0)).item()
            target = gamma * next_val if next_turn_pid == turn_pid else gamma * (-next_val)

        opponent_gen = self.current_opponent_gen
        self._append_replay_sample(state_tensor, target, opponent_gen)

        self.episode_trajectory.append(
            (state_tensor, turn_pid, target, opponent_gen)
        )

        if done:
            if self.episode_trajectory:
                final_index = len(self.episode_trajectory) - 1
                for i, (
                    episode_state,
                    episode_pid,
                    bootstrap_target,
                    episode_gen
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
            ("Heuristic Dep.", f"{self.heuristic_weight:.3f}"),
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

    def finalize_training_game(self, winner_id):
        """Record, evaluate, and persist a game after its terminal train step."""
        self.games_played += 1
        auto_save_due = self.games_played % 250 == 0

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
        print(f"Current training WRs: {self._format_training_win_rates()}")

        promoted = False
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
                self.student_eval_history.clear()
                self.student_win_rate_str = "Pending live WR"

                self.update_opponent_pool()
                self.save_checkpoint(manual=True)
                promoted = True
            else:
                print(f"Current training WRs: {self._format_training_win_rates()}")

        if not promoted:
            if auto_save_due:
                self.save_checkpoint(manual=False)
            else:
                self.save_meta()

    def run_headless_training(self, num_games):
        print(f"Starting Headless Turbo Training for {num_games} games...")
        self.mode = "TRAIN"
        self.is_training_active = True
        self.reset_game()
        
        def command_listener():
            while True:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'p': self.is_training_active = not self.is_training_active
                elif cmd == 's': self.manual_save_requested = True
                elif cmd == 'q': os._exit(0)
                    
        threading.Thread(target=command_listener, daemon=True).start()
        target_games, last_print = self.games_played + num_games, self.games_played
        
        while self.games_played < target_games:
            if self.manual_save_requested:
                self.manual_save_requested = False
                self.save_checkpoint(manual=True)

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
            if self.game_over:
                self.finalize_training_game(winner_id)
                
        print(f"Headless training complete! Total games played: {self.games_played}")
        self.save_checkpoint(manual=True)
        pygame.quit()
        sys.exit()

    def play(self):
        clock = pygame.time.Clock()
        while True:
            mouse_pos = self.get_virtual_mouse_pos()
            
            if self.mode == "MENU":
                self.draw_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_F11, pygame.K_f): self.toggle_fullscreen()
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.rect_play.collidepoint(mouse_pos):
                            self.mode = "PLAY"
                            self.reset_game()
                        elif self.rect_train.collidepoint(mouse_pos):
                            self.mode = "TRAIN"
                            self.reset_game()

            elif self.mode in ("PLAY", "TRAIN"):
                hover_type, hover_c, hover_r, active_orient = self.draw_board(mouse_pos)
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_F11, pygame.K_f): self.toggle_fullscreen()
                        elif event.key == pygame.K_r: self.wall_orientation = 'V' if self.wall_orientation == 'H' else 'H'
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 3: self.wall_orientation = 'V' if self.wall_orientation == 'H' else 'H'
                        elif event.button == 1:
                            if self.rect_menu.collidepoint(mouse_pos): self.mode = "MENU"
                            elif self.rect_new_game.collidepoint(mouse_pos): self.reset_game()
                            elif self.mode == "TRAIN" and self.rect_train_toggle.collidepoint(mouse_pos):
                                self.is_training_active = not self.is_training_active
                            elif self.mode == "PLAY" and not self.game_over and self.state['turn'] == 1:
                                if hover_type == 'cell' and (hover_c, hover_r) in self.bot2.get_valid_moves(self.state, 1):
                                    self.state = self.bot2.apply_action(self.state, 1, ('move', (hover_c, hover_r)))
                                    self.check_win()
                                elif hover_type == 'wall' and self.bot2.is_valid_wall(self.state, hover_c, hover_r, active_orient) and self.state['p1_walls'] > 0:
                                    self.state = self.bot2.apply_action(self.state, 1, ('wall', (hover_c, hover_r, active_orient)))
                                    self.check_win()

                if not self.game_over:
                    if self.mode == "TRAIN":
                        if self.is_training_active:
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
                            if self.game_over:
                                self.finalize_training_game(winner_id)
                                self.reset_game()
                    elif self.mode == "PLAY" and self.state['turn'] == 2:
                        action, _ = self.bot2.get_move_and_eval(self.state, epsilon=0.0)
                        if action: self.state = self.bot2.apply_action(self.state, 2, action, is_random=getattr(self.bot2, 'was_last_random', False))
                        else: self.state['turns_played'] = 200
                        self.check_win()

            clock.tick(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quoridor AI Engine with QuoriNet Deep RL")
    parser.add_argument("--headless", action="store_true", help="Run in headless turbo training mode")
    parser.add_argument("--games", type=int, default=1000, help="Number of games to run in headless mode")
    args = parser.parse_args()

    game = Quoridor(headless=args.headless)
    if args.headless:
        game.run_headless_training(args.games)
    else:
        game.play()
