import pygame
import sys
import os
import json
import random
import time
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

class SpatialQuoriNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

def encode_state(state, active_pid):
    tensor = torch.zeros((5, 9, 9), dtype=torch.float32)
    
    mx, my = state['p1_pos'] if active_pid == 1 else state['p2_pos']
    ox, oy = state['p2_pos'] if active_pid == 1 else state['p1_pos']
    
    if active_pid == 2:
        mx, my = 8 - mx, 8 - my
        ox, oy = 8 - ox, 8 - oy
        
    tensor[0, my, mx] = 1.0
    tensor[1, oy, ox] = 1.0
    
    for c, r in state['walls_h']:
        if active_pid == 2: c, r = 7 - c, 7 - r
        tensor[2, r, c] = 1.0
            
    for c, r in state['walls_v']:
        if active_pid == 2: c, r = 7 - c, 7 - r
        tensor[3, r, c] = 1.0
            
    my_walls = state['p1_walls'] if active_pid == 1 else state['p2_walls']
    opp_walls = state['p2_walls'] if active_pid == 1 else state['p1_walls']
    
    tensor[4, 0:4, :] = my_walls / 10.0
    tensor[4, 5:9, :] = opp_walls / 10.0
    
    return tensor

def get_state_key(state):
    return (
        state['p1_pos'], 
        state['p2_pos'], 
        tuple(sorted(state['walls_h'])), 
        tuple(sorted(state['walls_v']))
    )

class QuoridorBot:
    def __init__(self, bot_id=2, max_depth=3, nn_model=None, heuristic_weight=1.0):
        self.bot_id = bot_id
        self.max_depth = max_depth
        self.nn_model = nn_model
        self.heuristic_weight = heuristic_weight
        
    def get_move_and_eval(self, state, epsilon=0.0):
        if epsilon > 0.0 and random.random() < epsilon:
            raw_actions = self.generate_actions(state, self.bot_id)
            if raw_actions:
                chosen_action = random.choice(raw_actions)
                next_state = self.apply_action(state, self.bot_id, chosen_action)
                return chosen_action, self.evaluate(next_state, self.bot_id)

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
            if pos[1] == target_row:
                return path
                
            for nxt in self.get_adjacent(state, *pos):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None

    def get_adjacent(self, state, c, r):
        moves = []
        if r > 0 and (c, r - 1) not in state['walls_h'] and (c - 1, r - 1) not in state['walls_h']:
            moves.append((c, r - 1))
        if r < 8 and (c, r) not in state['walls_h'] and (c - 1, r) not in state['walls_h']:
            moves.append((c, r + 1))
        if c > 0 and (c - 1, r) not in state['walls_v'] and (c - 1, r - 1) not in state['walls_v']:
            moves.append((c - 1, r))
        if c < 8 and (c, r) not in state['walls_v'] and (c, r - 1) not in state['walls_v']:
            moves.append((c + 1, r))
        return moves

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
        for move in valid_moves:
            actions.append(('move', move))
            
        walls_left = state['p1_walls'] if pid == 1 else state['p2_walls']
        if walls_left <= 0: return actions

        opponent_id = self.get_opponent(pid)
        opp_path = self.get_shortest_path(state, opponent_id)
        my_path = self.get_shortest_path(state, pid)
        if not opp_path or not my_path: return actions

        base_my_dist = len(my_path)
        base_opp_dist = len(opp_path)

        candidate_walls = set()
        
        for i in range(len(opp_path) - 1):
            c1, r1 = opp_path[i]
            c2, r2 = opp_path[i+1]
            if c1 == c2: 
                r_min = min(r1, r2)
                candidate_walls.update([(c1, r_min, 'H'), (c1-1, r_min, 'H')])
            else:
                c_min = min(c1, c2)
                candidate_walls.update([(c_min, r1, 'V'), (c_min, r1-1, 'V')])
                
        for i in range(len(my_path) - 1):
            c1, r1 = my_path[i]
            c2, r2 = my_path[i+1]
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
            else:
                break
            
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

    def evaluate(self, state, eval_pid):
        if state['p1_pos'][1] == 0: return 10000 if eval_pid == 1 else -10000
        if state['p2_pos'][1] == 8: return -10000 if eval_pid == 1 else 10000
        
        is_repeated = get_state_key(state) in state.get('history', set())

        if self.nn_model is not None:
            with torch.no_grad():
                nn_val = self.nn_model(encode_state(state, eval_pid).unsqueeze(0)).item()
            score = nn_val * 100.0
            if is_repeated: score -= 8000
            return score

        p1_path = self.get_shortest_path(state, 1)
        p2_path = self.get_shortest_path(state, 2)
        
        if not p1_path: return 10000 if eval_pid == 2 else -10000
        if not p2_path: return -10000 if eval_pid == 2 else 10000

        d1 = len(p1_path) - 1
        d2 = len(p2_path) - 1
        p1_progress = 8 - state['p1_pos'][1]
        p2_progress = state['p2_pos'][1]

        if state['p1_walls'] == 0 and state['p2_walls'] == 0:
            score = (d2 - d1) * 50.0 + (p1_progress - p2_progress) * 10.0
            if eval_pid == 2: score = -score
            if is_repeated: score -= 8000
            return score

        if eval_pid == 1:
            score = (d2 - d1) * 25.0 + (p1_progress - p2_progress) * 5.0
        else:
            score = (d1 - d2) * 25.0 + (p2_progress - p1_progress) * 5.0

        if is_repeated: score -= 8000
        return score

    def apply_action(self, state, pid, action):
        new_state = {
            'p1_pos': state['p1_pos'], 'p2_pos': state['p2_pos'],
            'p1_walls': state['p1_walls'], 'p2_walls': state['p2_walls'],
            'walls_h': state['walls_h'].copy(), 'walls_v': state['walls_v'].copy(),
            'turn': self.get_opponent(pid),
            'turns_played': state.get('turns_played', 0) + 1,
            'history': state.get('history', set()).copy()
        }
        new_state['history'].add(get_state_key(state))

        if action[0] == 'move':
            if pid == 1: new_state['p1_pos'] = action[1]
            else: new_state['p2_pos'] = action[1]
        else:
            c, r, orient = action[1]
            if orient == 'H': new_state['walls_h'].add((c, r))
            else: new_state['walls_v'].add((c, r))
            if pid == 1: new_state['p1_walls'] -= 1
            else: new_state['p2_walls'] -= 1

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

CELL_SIZE = int(55 * SCALE)
WALL_THICKNESS = int(14 * SCALE)
BOARD_SIZE = 9
PADDING = int(25 * SCALE)
BOARD_PIXEL_SIZE = (BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * WALL_THICKNESS + (PADDING * 2))

SIDEBAR_WIDTH = int(380 * SCALE)
WIDTH = BOARD_PIXEL_SIZE + SIDEBAR_WIDTH
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
        
        self.student_net = SpatialQuoriNet()
        self.teacher_net = SpatialQuoriNet()
        self.student_net.eval()
        self.teacher_net.eval()
        
        self.optimizer = optim.Adam(self.student_net.parameters(), lr=0.005)
        self.loss_fn = nn.MSELoss()
        
        self.games_played = 0
        self.teacher_gen = 1
        self.heuristic_weight = 1.0
        
        self.student_eval_history = deque(maxlen=100) 
        self.student_win_rate_str = "Pending Eval"
        self.student_id = 2
        
        self.p1_gen_str = "Gen #1"
        self.p2_gen_str = "Gen #1"

        self.replay_buffer = deque(maxlen=200000) 
        self.last_loss = 0.0
        
        self.match_history = deque(maxlen=6)
        self.opponent_pool = []
        self.opponent_weights = []

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
        
        self.rect_play = pygame.Rect(0, 0, 0, 0)
        self.rect_train = pygame.Rect(0, 0, 0, 0)
        self.rect_new_game = pygame.Rect(0, 0, 0, 0)
        self.rect_menu = pygame.Rect(0, 0, 0, 0)
        self.rect_train_toggle = pygame.Rect(0, 0, 0, 0)
        
        self.game_start_time = time.time()
        self.reset_game()

    def update_opponent_pool(self):
        self.opponent_pool = []
        self.opponent_weights = []
        
        target_gens = set()
        curr = self.teacher_gen
        step = 1
        while curr >= 1:
            target_gens.add(curr)
            curr -= step
            step *= 2
        target_gens.add(1)
        
        for i in sorted(list(target_gens)):
            if i == self.teacher_gen:
                self.opponent_pool.append((self.teacher_net, i))
                self.opponent_weights.append(1.0)
            else:
                file_path = f'quorinet_teacher_gen_{i}.pth'
                if os.path.exists(file_path):
                    try:
                        net = SpatialQuoriNet()
                        net.load_state_dict(torch.load(file_path, weights_only=True))
                        net.eval()
                        self.opponent_pool.append((net, i))
                        self.opponent_weights.append(1.0)
                    except Exception as e:
                        print(f"Failed to load pool model {i}: {e}")
        if not self.opponent_pool:
            self.opponent_pool = [(self.teacher_net, self.teacher_gen)]
            self.opponent_weights = [1.0]
        print(f"Opponent pool updated. Size: {len(self.opponent_pool)}")

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
        print(f"Eval Progress:   {len(self.student_eval_history)}/30 min games")
        print(f"AI Board Eval:   {eval_score}")
        print(f"Learning Rate:   {current_lr:.6f}")
        print(f"Training Loss:   {self.last_loss:.5f}")
        print("----------------------------------------\n")

    def get_virtual_mouse_pos(self):
        mx, my = pygame.mouse.get_pos()
        sw, sh = self.screen.get_size()
        if sw <= 0 or sh <= 0: return mx, my
        vx = int(mx * (WIDTH / sw))
        vy = int(my * (HEIGHT / sh))
        return vx, vy

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

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
            'turn': 1,
            'turns_played': 0,
            'history': set()
        }
        self.wall_orientation = 'H'
        self.game_over = False
        self.winner = None
        self.game_start_time = time.time()
        
        if getattr(self, 'mode', 'MENU') == 'TRAIN':
            if hasattr(self, 'opponent_pool') and self.opponent_pool:
                pool_opponent, pool_gen = random.choices(self.opponent_pool, weights=self.opponent_weights, k=1)[0]
            else:
                pool_opponent, pool_gen = self.teacher_net, self.teacher_gen
            
            if self.games_played % 2 == 0:
                self.bot1 = QuoridorBot(bot_id=1, max_depth=2, nn_model=pool_opponent, heuristic_weight=self.heuristic_weight)
                self.bot2 = QuoridorBot(bot_id=2, max_depth=2, nn_model=self.student_net, heuristic_weight=self.heuristic_weight)
                self.student_id = 2
                self.p1_gen_str = f"Gen #{pool_gen} (Teacher)"
                self.p2_gen_str = f"Gen #{self.teacher_gen} (Student)"
            else:
                self.bot1 = QuoridorBot(bot_id=1, max_depth=2, nn_model=self.student_net, heuristic_weight=self.heuristic_weight) 
                self.bot2 = QuoridorBot(bot_id=2, max_depth=2, nn_model=pool_opponent, heuristic_weight=self.heuristic_weight) 
                self.student_id = 1
                self.p1_gen_str = f"Gen #{self.teacher_gen} (Student)"
                self.p2_gen_str = f"Gen #{pool_gen} (Teacher)"
        else:
            self.bot1 = QuoridorBot(bot_id=1, max_depth=4, nn_model=self.teacher_net, heuristic_weight=self.heuristic_weight)
            self.bot2 = QuoridorBot(bot_id=2, max_depth=4, nn_model=self.teacher_net, heuristic_weight=self.heuristic_weight)
            self.p1_gen_str = "Human"
            self.p2_gen_str = f"Gen #{self.teacher_gen} (Teacher)"

        self.p1_draw_x, self.p1_draw_y = self._get_center_pixels(self.state['p1_pos'])
        self.p2_draw_x, self.p2_draw_y = self._get_center_pixels(self.state['p2_pos'])

    def load_checkpoint_if_exists(self):
        teacher_loaded = False
        if os.path.exists('quorinet_teacher_weights.pth'):
            try:
                self.teacher_net.load_state_dict(torch.load('quorinet_teacher_weights.pth', weights_only=True))
                print("Loaded Teacher Checkpoint.")
                teacher_loaded = True
            except Exception as e:
                print(f"Failed teacher checkpoint: {e}")
                
        if os.path.exists('quorinet_student_weights.pth'):
            try:
                self.student_net.load_state_dict(torch.load('quorinet_student_weights.pth', weights_only=True))
                print("Loaded Student Checkpoint.")
            except Exception as e:
                print(f"Failed student checkpoint: {e}")
        elif teacher_loaded:
            self.student_net.load_state_dict(self.teacher_net.state_dict())

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
                    if self.student_eval_history:
                        wins = sum(self.student_eval_history)
                        games = len(self.student_eval_history)
                        self.student_win_rate_str = f"{wins/games * 100:.1f}% ({wins}/{games})"
            except Exception:
                pass
        print(f"Total Games Played: {self.games_played} | Teacher Gen: #{self.teacher_gen} | Heuristic W: {self.heuristic_weight:.3f}")

    def save_meta(self):
        with open('quorinet_meta.json', 'w') as f:
            json.dump({
                'games_played': self.games_played,
                'teacher_gen': self.teacher_gen,
                'heuristic_weight': self.heuristic_weight,
                'student_eval_history': list(self.student_eval_history)
            }, f)

    def save_checkpoint(self, manual=False):
        torch.save(self.teacher_net.state_dict(), f'quorinet_teacher_gen_{self.teacher_gen}.pth')
        torch.save(self.teacher_net.state_dict(), 'quorinet_teacher_weights.pth')
        torch.save(self.student_net.state_dict(), 'quorinet_student_weights.pth')
        
        self.save_meta()
            
        tag = "Manual" if manual else "Auto"
        print(f"[{tag} Saved] Gen #{self.teacher_gen} saved ({self.games_played} games).")
        
        self.toast_msg = f"Saved Checkpoint! (Gen #{self.teacher_gen})"
        self.toast_timer = 120

    def play_eval_game(self, bot1, bot2, game_curr, game_total, vs_gen):
        eval_state = {
            'p1_pos': (4, 8), 'p2_pos': (4, 0),
            'p1_walls': 10, 'p2_walls': 10,
            'walls_h': set(), 'walls_v': set(),
            'turn': 1,
            'turns_played': 0,
            'history': set()
        }
        
        self.eval_game_curr = game_curr
        self.eval_game_total = game_total
        self.eval_vs_gen = vs_gen

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
                        if event.type == pygame.KEYDOWN and event.key in (pygame.K_F11, pygame.K_f):
                            self.toggle_fullscreen()

                    self.p1_draw_x, self.p1_draw_y = self._get_center_pixels(eval_state['p1_pos'])
                    self.p2_draw_x, self.p2_draw_y = self._get_center_pixels(eval_state['p2_pos'])
                    self.draw_board((-1, -1))
                    pygame.time.delay(10)

                if eval_state['p1_pos'][1] == 0:
                    result = 1
                    break
                if eval_state['p2_pos'][1] == 8:
                    result = 2
                    break
                
                pid = eval_state['turn']
                active_bot = bot1 if pid == 1 else bot2
                action, _ = active_bot.get_move_and_eval(eval_state, epsilon=0.0)
                if not action:
                    result = 2 if pid == 1 else 1
                    break
                eval_state = active_bot.apply_action(eval_state, pid, action)
                self.state = eval_state
        finally:
            self.state = saved_state

        return result

    def evaluate_and_tune_lr(self, games_per_model=10):
        if self.teacher_gen <= 1:
            print("No previous historical models available to evaluate against.")
            return 1.0
            
        eval_gens_set = set()
        curr = self.teacher_gen
        step = 1
        while curr >= 1:
            eval_gens_set.add(curr)
            curr -= step
            step *= 2
        eval_gens_set.add(1)
        eval_gens = sorted([g for g in eval_gens_set if g < self.teacher_gen])
        
        if not eval_gens:
            print("No previous historical models available to evaluate against.")
            return 1.0
            
        print(f"\n--- Starting Historical Pool Evaluation (Candidate Gen #{self.teacher_gen + 1}) ---")
        print(f"Evaluating candidate against sampled generations: {eval_gens}")

        self.is_evaluating = True
        total_eval_games = len(eval_gens) * games_per_model
        completed_eval_games = 0

        weighted_wins = 0.0
        weighted_total = 0.0
        passed_all_tiers = True
        
        try:
            for gen_i in eval_gens:
                file_path = f'quorinet_teacher_gen_{gen_i}.pth'
                if not os.path.exists(file_path):
                    continue
                    
                past_net = SpatialQuoriNet()
                try:
                    past_net.load_state_dict(torch.load(file_path, weights_only=True))
                    past_net.eval()
                except Exception as e:
                    continue
                    
                gen_wins = 0
                for g in range(games_per_model):
                    completed_eval_games += 1
                    if g % 2 == 0:
                        eval_bot1 = QuoridorBot(bot_id=1, max_depth=2, nn_model=self.student_net, heuristic_weight=self.heuristic_weight)
                        eval_bot2 = QuoridorBot(bot_id=2, max_depth=2, nn_model=past_net, heuristic_weight=self.heuristic_weight)
                        res = self.play_eval_game(eval_bot1, eval_bot2, completed_eval_games, total_eval_games, gen_i)
                        if res == 1: gen_wins += 1
                    else:
                        eval_bot1 = QuoridorBot(bot_id=1, max_depth=2, nn_model=past_net, heuristic_weight=self.heuristic_weight)
                        eval_bot2 = QuoridorBot(bot_id=2, max_depth=2, nn_model=self.student_net, heuristic_weight=self.heuristic_weight)
                        res = self.play_eval_game(eval_bot1, eval_bot2, completed_eval_games, total_eval_games, gen_i)
                        if res == 2: gen_wins += 1
                
                gen_win_rate = gen_wins / games_per_model
                age = self.teacher_gen - gen_i
                weight = 1.0
                
                weighted_wins += gen_win_rate * weight
                weighted_total += weight
                
                required_threshold = min(0.55 + (0.05 * age), 0.85)
                tier_status = "PASS" if gen_win_rate >= required_threshold else "FAIL"
                if tier_status == "FAIL": passed_all_tiers = False
                    
                print(f"Vs Gen #{gen_i} (Age {age}): {gen_wins}/{games_per_model} wins | Req: {required_threshold * 100:.1f}% [{tier_status}]")
        finally:
            self.is_evaluating = False

        if weighted_total == 0:
            return 1.0
            
        final_weighted_win_rate = weighted_wins / weighted_total
        print(f"Overall Weighted Pool Win Rate: {final_weighted_win_rate * 100:.1f}%")
        
        lr_scale = max(0.5, 1.0 - (0.3 * final_weighted_win_rate))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= lr_scale
            print(f"Learning Rate adjusted to: {param_group['lr']:.6f} (Scale factor: {lr_scale:.3f} based on win rate)")
                
        return final_weighted_win_rate

    def train_step(self, state_tensor, turn_pid, next_state_tensor, next_turn_pid, done, winner_id):
        gamma = 0.99
        
        if done:
            if winner_id == 0: 
                target = 0.0
            else:
                target = 1.0 if turn_pid == winner_id else -1.0
        else:
            with torch.no_grad():
                next_val = self.teacher_net(next_state_tensor.unsqueeze(0)).item()
            
            if next_turn_pid == turn_pid:
                target = gamma * next_val
            else:
                target = gamma * (-next_val)
                
        self.replay_buffer.append((state_tensor, target))
        
        if len(self.replay_buffer) >= 256:
            self.student_net.train()
            batch = random.sample(self.replay_buffer, 128)
            states = torch.stack([item[0] for item in batch])
            targets = torch.tensor([[item[1]] for item in batch], dtype=torch.float32)

            self.optimizer.zero_grad()
            predictions = self.student_net(states)
            loss = self.loss_fn(predictions, targets)
            loss.backward()
            self.optimizer.step()
            
            self.last_loss = loss.item()
            self.student_net.eval()

    def log_match_result(self, winner_id):
        duration = round(time.time() - self.game_start_time, 1)
        
        if winner_id == 1:
            winner_str = "Red"
        elif winner_id == 2:
            winner_str = "Blue"
        else:
            winner_str = "Draw"
            
        self.match_history.appendleft({
            'game': self.games_played + 1,
            'winner': winner_str,
            'winner_id': winner_id,
            'turns': self.state['turns_played'],
            'time': duration,
            'p1_w': self.state['p1_walls'],
            'p2_w': self.state['p2_walls']
        })

    def _get_center_pixels(self, pos):
        offset_x, offset_y = PADDING, int(85 * SCALE) + PADDING
        x = offset_x + pos[0] * (CELL_SIZE + WALL_THICKNESS) + CELL_SIZE / 2
        y = offset_y + pos[1] * (CELL_SIZE + WALL_THICKNESS) + CELL_SIZE / 2
        return x, y

    def draw_menu(self):
        if self.headless: return
        self.virtual_screen.fill(BG_COLOR)
        title = self.title_font.render("QuoriNet AI", True, TEXT_COLOR)
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

    def draw_sidebar(self):
        sb_x = BOARD_PIXEL_SIZE
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
        eval_counter_str = f"{len(self.student_eval_history)}/30 min games"
        current_lr = self.optimizer.param_groups[0]['lr']

        if self.is_evaluating:
            eval_phase_str = f"Vs Gen #{self.eval_vs_gen} ({self.eval_game_curr}/{self.eval_game_total})"
        else:
            eval_phase_str = "Idle"

        metrics = [
            ("Red Player (P1)", self.p1_gen_str),
            ("Blue Player (P2)", self.p2_gen_str),
            ("Latest Teacher Gen", f"Gen #{self.teacher_gen}"),
            ("Heuristic Dep.", f"{self.heuristic_weight:.3f}"),
            ("Student Win Rate", self.student_win_rate_str),
            ("Eval Cycle Progress", eval_counter_str),
            ("Eval Phase Active", eval_phase_str),
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
            h_surf = self.tiny_font.render(h_text, True, MUTED_TEXT)
            self.virtual_screen.blit(h_surf, (card2_rect.x + int(h_pos * SCALE), card2_rect.y + int(31 * SCALE)))

        hy_offset = card2_rect.y + int(48 * SCALE)
        for log in reversed(list(self.match_history)[:4]):
            w_color = P1_COLOR if log.get('winner_id') == 1 else (P2_COLOR if log.get('winner_id') == 2 else MUTED_TEXT)
            
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
            if ty_offset + int(13 * SCALE) > card3_rect.bottom - int(5 * SCALE):
                break
            line_str = line[:38] + ".." if len(line) > 40 else line
            log_surf = self.console_font.render(line_str, True, TERMINAL_TEXT)
            self.virtual_screen.blit(log_surf, (card3_rect.x + int(10 * SCALE), ty_offset))
            ty_offset += int(14 * SCALE)

    def draw_board(self, mouse_pos):
        if self.headless: return 'none', -1, -1, self.wall_orientation
        self.virtual_screen.fill(BG_COLOR)
        
        p1_txt = self.small_font.render(f"Red ({self.p1_gen_str}) · {self.state['p1_walls']}", True, TEXT_COLOR)
        p2_txt = self.small_font.render(f"{self.state['p2_walls']} · Blue ({self.p2_gen_str})", True, TEXT_COLOR)
        self.virtual_screen.blit(p1_txt, (int(20 * SCALE), int(35 * SCALE)))
        self.virtual_screen.blit(p2_txt, (BOARD_PIXEL_SIZE - p2_txt.get_width() - int(20 * SCALE), int(35 * SCALE)))

        if self.game_over:
            status_text = "Draw" if self.winner == "Draw" else f"{self.winner} wins!"
            status_color = MUTED_TEXT if self.winner == "Draw" else (P1_COLOR if self.winner == "Red" else P2_COLOR)
        else:
            status_text = "Red's turn" if self.state['turn'] == 1 else "Blue's turn"
            status_color = P1_COLOR if self.state['turn'] == 1 else P2_COLOR

        status_surf = self.small_font.render(status_text, True, status_color)
        self.virtual_screen.blit(status_surf, status_surf.get_rect(center=(BOARD_PIXEL_SIZE // 2, int(35 * SCALE))))
        
        if self.mode == "TRAIN":
            if self.is_evaluating:
                rem_games = self.eval_game_total - self.eval_game_curr
                status_lbl = f"EVAL: vs Gen #{self.eval_vs_gen} | Game {self.eval_game_curr}/{self.eval_game_total} ({rem_games} left)"
                games_txt = self.small_font.render(status_lbl, True, (230, 110, 20))
            else:
                status_lbl = "TRAINING PAUSED" if not self.is_training_active else f"Games: {self.games_played} | Turns: {self.state['turns_played']}"
                games_txt = self.small_font.render(status_lbl, True, (180, 50, 50) if not self.is_training_active else P2_COLOR)
            self.virtual_screen.blit(games_txt, (BOARD_PIXEL_SIZE // 2 - games_txt.get_width() // 2, int(58 * SCALE)))

        board_rect = pygame.Rect(int(12 * SCALE), int(78 * SCALE), BOARD_PIXEL_SIZE - int(24 * SCALE), BOARD_PIXEL_SIZE - int(12 * SCALE))
        pygame.draw.rect(self.virtual_screen, (224, 227, 232), board_rect.move(0, int(4 * SCALE)), border_radius=int(18 * SCALE))
        pygame.draw.rect(self.virtual_screen, BOARD_BG, board_rect, border_radius=int(18 * SCALE))

        offset_x, offset_y = PADDING, int(85 * SCALE) + PADDING
        valid_moves = []
        if not self.game_over and self.mode == "PLAY":
            valid_moves = self.bot2.get_valid_moves(self.state, self.state['turn'])

        mx, my = mouse_pos
        local_x, local_y = mx - offset_x, my - offset_y
        unit = CELL_SIZE + WALL_THICKNESS
        hover_type, hover_c, hover_r = 'none', -1, -1
        active_orient = self.wall_orientation

        board_size_px = BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * WALL_THICKNESS
        if 0 <= local_x <= board_size_px and 0 <= local_y <= board_size_px and not self.game_over:
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
            pygame.draw.rect(self.virtual_screen, WALL_COLOR, rct, border_radius=int(6 * SCALE))

        for (c, r) in self.state['walls_v']:
            x, y = offset_x + c * unit + CELL_SIZE, offset_y + r * unit
            rct = pygame.Rect(x, y, WALL_THICKNESS, CELL_SIZE * 2 + WALL_THICKNESS)
            pygame.draw.rect(self.virtual_screen, WALL_SHADOW, rct.move(int(2 * SCALE), 0), border_radius=int(6 * SCALE))
            pygame.draw.rect(self.virtual_screen, WALL_COLOR, rct, border_radius=int(6 * SCALE))

        if hover_type == 'wall' and not self.game_over and self.mode == "PLAY" and self.state['turn'] == 1:
            if self.bot2.is_valid_wall(self.state, hover_c, hover_r, active_orient) and self.state['p1_walls'] > 0:
                hx, hy = offset_x + hover_c * unit, offset_y + hover_r * unit
                overlay = pygame.Surface((BOARD_PIXEL_SIZE, HEIGHT), pygame.SRCALPHA)
                
                if active_orient == 'H':
                    pygame.draw.rect(overlay, (82, 88, 98, 115), 
                                     (hx, hy + CELL_SIZE, CELL_SIZE * 2 + WALL_THICKNESS, WALL_THICKNESS), border_radius=int(6 * SCALE))
                else:
                    pygame.draw.rect(overlay, (82, 88, 98, 115), 
                                     (hx + CELL_SIZE, hy, WALL_THICKNESS, CELL_SIZE * 2 + WALL_THICKNESS), border_radius=int(6 * SCALE))
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
        pygame.draw.circle(self.virtual_screen, P1_COLOR, (int(self.p1_draw_x), int(self.p1_draw_y)), radius)
        pygame.draw.circle(self.virtual_screen, (218, 221, 226), (int(self.p2_draw_x), int(self.p2_draw_y + int(3 * SCALE))), radius)
        pygame.draw.circle(self.virtual_screen, P2_COLOR, (int(self.p2_draw_x), int(self.p2_draw_y)), radius)

        btn_y = HEIGHT - int(46 * SCALE)
        btn_h = int(34 * SCALE)
        btn_w = int(110 * SCALE)
        
        self.rect_menu = pygame.Rect(BOARD_PIXEL_SIZE - int(125 * SCALE), btn_y, btn_w, btn_h)
        m_hover = self.rect_menu.collidepoint(mouse_pos)
        pygame.draw.rect(self.virtual_screen, (200, 205, 212) if m_hover else (220, 224, 230), self.rect_menu, border_radius=int(8 * SCALE))
        menu_txt = self.small_font.render("Menu", True, TEXT_COLOR)
        self.virtual_screen.blit(menu_txt, menu_txt.get_rect(center=self.rect_menu.center))

        self.rect_new_game = pygame.Rect(BOARD_PIXEL_SIZE - int(245 * SCALE), btn_y, btn_w, btn_h)
        ng_hover = self.rect_new_game.collidepoint(mouse_pos)
        pygame.draw.rect(self.virtual_screen, (200, 205, 212) if ng_hover else (220, 224, 230), self.rect_new_game, border_radius=int(8 * SCALE))
        ng_txt = self.small_font.render("New Game", True, TEXT_COLOR)
        self.virtual_screen.blit(ng_txt, ng_txt.get_rect(center=self.rect_new_game.center))

        if self.mode == "TRAIN":
            self.rect_train_toggle = pygame.Rect(int(15 * SCALE), btn_y, int(160 * SCALE), btn_h)
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
            toast_rect = pygame.Rect(BOARD_PIXEL_SIZE // 2 - t_w // 2, int(8 * SCALE), t_w, t_h)
            pygame.draw.rect(self.virtual_screen, (46, 125, 50), toast_rect, border_radius=int(15 * SCALE))
            self.virtual_screen.blit(toast_surf, toast_surf.get_rect(center=toast_rect.center))

        self.draw_sidebar()
        self.render_and_flip()
        return hover_type, hover_c, hover_r, active_orient

    def check_win(self):
        if self.state['p1_pos'][1] == 0:
            self.game_over = True
            self.winner = "Red"
            print(f"Game #{self.games_played + 1} Winner: RED [{self.p1_gen_str}] vs BLUE [{self.p2_gen_str}] ({self.state['turns_played']} turns)")
            self.log_match_result(1)
        elif self.state['p2_pos'][1] == 8:
            self.game_over = True
            self.winner = "Blue"
            print(f"Game #{self.games_played + 1} Winner: BLUE [{self.p2_gen_str}] vs RED [{self.p1_gen_str}] ({self.state['turns_played']} turns)")
            self.log_match_result(2)
        elif self.state['turns_played'] >= 200:
            self.game_over = True
            self.winner = "Draw"
            print(f"Game #{self.games_played + 1} Ended in DRAW: RED [{self.p1_gen_str}] vs BLUE [{self.p2_gen_str}] (200 turn limit)")
            self.log_match_result(0)
            
        if self.game_over and self.mode == "TRAIN":
            self.games_played += 1
            if self.games_played % 250 == 0:
                self.save_checkpoint(manual=False)

            student_won = 1 if (self.winner == "Red" and self.student_id == 1) or (self.winner == "Blue" and self.student_id == 2) else 0
            self.student_eval_history.append(student_won)
            wins = sum(self.student_eval_history)
            games = len(self.student_eval_history)
            self.student_win_rate_str = f"{wins/games * 100:.1f}% ({wins}/{games})"
            
            self.save_meta()
            
            if games >= 30:
                if (wins / games) >= 0.57:
                    print(f"Student beat Teacher! Upgrading Teacher to Gen #{self.teacher_gen + 1}...")
                    self.evaluate_and_tune_lr(games_per_model=10)
                    self.teacher_gen += 1
                    self.teacher_net.load_state_dict(self.student_net.state_dict())
                    self.update_opponent_pool()
                    
                    old_hw = self.heuristic_weight
                    self.heuristic_weight = self.heuristic_weight * 0.85
                    print(f"Curriculum Update: Heuristic Dependency decayed from {old_hw:.3f} to {self.heuristic_weight:.3f}")

                    self.save_checkpoint(manual=True)
                    self.student_eval_history.clear()
                    self.save_meta()
                else:
                    print(f"Evaluation threshold not met ({wins}/{games}). Continuing cycle...")

    def run_headless_training(self, num_games):
        print(f"Starting Headless Turbo Training for {num_games} games...")
        print("=========================================================")
        print("Headless Controls (Type and press Enter):")
        print("  'p' -> Pause / Resume training")
        print("  's' -> Save checkpoint")
        print("  'q' -> Quit gracefully")
        print("=========================================================")
        
        self.mode = "TRAIN"
        self.is_training_active = True
        self.reset_game()
        
        def command_listener():
            while True:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'p':
                    self.is_training_active = not self.is_training_active
                    state = "RESUMED" if self.is_training_active else "PAUSED"
                    print(f"\n[Command] Training {state}!")
                elif cmd == 's':
                    self.save_checkpoint(manual=True)
                elif cmd == 'q':
                    print("\n[Command] Quitting...")
                    os._exit(0)
                    
        threading.Thread(target=command_listener, daemon=True).start()
        
        target_games = self.games_played + num_games
        last_print = self.games_played
        
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
            
            eps = 0.15 * (0.95 ** self.state['turns_played'])
            action, _ = active_bot.get_move_and_eval(self.state, epsilon=eps)
            
            if action:
                self.state = active_bot.apply_action(self.state, pre_action_pid, action)
            else:
                self.state['turns_played'] = 200
                
            self.check_win()
            
            next_state_tensor = encode_state(self.state, self.state['turn'])
            next_pid = self.state['turn']
            winner_id = 0
            
            if self.game_over:
                if getattr(self, 'winner', None) == "Red": winner_id = 1
                elif getattr(self, 'winner', None) == "Blue": winner_id = 2
                
            self.train_step(pre_action_state_tensor, pre_action_pid, next_state_tensor, next_pid, self.game_over, winner_id)
                
        print(f"Headless training complete! Total games played: {self.games_played}")
        self.save_checkpoint(manual=True)
        pygame.quit()
        sys.exit()

    def play(self):
        clock = pygame.time.Clock()
        running = True
        bot_thinking = False
        bot_result = None

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
            finally:
                bot_thinking = False

        while running:
            virtual_mouse_pos = self.get_virtual_mouse_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_F11, pygame.K_f):
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_c:
                        self.save_checkpoint(manual=True)

                if self.mode == "MENU":
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.rect_play.collidepoint(virtual_mouse_pos):
                            self.mode = "PLAY"
                            self.reset_game()
                        elif self.rect_train.collidepoint(virtual_mouse_pos):
                            self.mode = "TRAIN"
                            self.is_training_active = True
                            self.reset_game()

                elif self.mode in ("PLAY", "TRAIN"):
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.rect_menu.collidepoint(virtual_mouse_pos):
                            self.mode = "MENU"
                            continue
                        elif self.rect_new_game.collidepoint(virtual_mouse_pos):
                            self.reset_game()
                            continue
                        elif self.mode == "TRAIN" and self.rect_train_toggle.collidepoint(virtual_mouse_pos):
                            if self.is_training_active:
                                self.is_training_active = False
                                self.save_checkpoint(manual=True)
                            else:
                                self.is_training_active = True
                            continue

                    if self.mode == "PLAY" and not self.game_over and not bot_thinking and self.state['turn'] == 1:
                        if event.type == pygame.KEYDOWN and event.key in (pygame.K_r, pygame.K_SPACE):
                            self.wall_orientation = 'V' if self.wall_orientation == 'H' else 'H'
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            if event.button == 3:
                                self.wall_orientation = 'V' if self.wall_orientation == 'H' else 'H'
                            elif event.button == 1:
                                hover_type, hover_c, hover_r, active_orient = self.draw_board(virtual_mouse_pos)
                                if hover_type == 'cell':
                                    valid_human_moves = self.bot2.get_valid_moves(self.state, 1)
                                    if (hover_c, hover_r) in valid_human_moves:
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

                    eps = 0.15 * (0.95 ** self.state['turns_played'])
                    action, _ = active_bot.get_move_and_eval(self.state, epsilon=eps)
                    
                    if action:
                        self.state = active_bot.apply_action(self.state, pre_action_pid, action)
                    else:
                        self.state['turns_played'] = 200

                    self.check_win()
                    
                    next_state_tensor = encode_state(self.state, self.state['turn'])
                    next_pid = self.state['turn']
                    winner_id = 0
                    
                    if self.game_over:
                        if getattr(self, 'winner', None) == "Red": winner_id = 1
                        elif getattr(self, 'winner', None) == "Blue": winner_id = 2
                        
                    self.train_step(pre_action_state_tensor, pre_action_pid, next_state_tensor, next_pid, self.game_over, winner_id)

            elif self.mode == "PLAY" and self.state['turn'] == 2 and not self.game_over:
                if not bot_thinking and bot_result is None:
                    bot_thinking = True
                    bot_state = {
                        'p1_pos': self.state['p1_pos'], 'p2_pos': self.state['p2_pos'],
                        'p1_walls': self.state['p1_walls'], 'p2_walls': self.state['p2_walls'],
                        'walls_h': self.state['walls_h'].copy(), 'walls_v': self.state['walls_v'].copy(),
                        'turn': 2,
                        'turns_played': self.state['turns_played'],
                        'history': self.state['history'].copy()
                    }
                    threading.Thread(target=think_bot, args=(bot_state,), daemon=True).start()
                
                if not bot_thinking and bot_result is not None:
                    self.state = self.bot2.apply_action(self.state, 2, bot_result)
                    bot_result = None
                    self.check_win()

            if self.mode == "MENU":
                self.draw_menu()
            else:
                self.draw_board(virtual_mouse_pos if self.mode == "PLAY" else (-1, -1))
                
            clock.tick(60 if self.mode == "PLAY" else 1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", type=int, default=0)
    args = parser.parse_args()

    if args.headless > 0:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    game = Quoridor(headless=(args.headless > 0))
    
    if args.headless > 0:
        game.run_headless_training(args.headless)
    else:
        game.play()
