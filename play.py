"""Polished human-vs-QuoriNet interface.

All rules, search, neural-network, checkpoint, and curriculum code lives in
trainer.py.  This file contains only presentation and human-play orchestration.
"""

import argparse
import os
import random
import threading

import pygame
import torch

from trainer import ArchivedValueNet, Quoridor as Trainer, QuoridorBot


SCALE = 0.92
BG_COLOR, BOARD_BG = (242, 244, 247), (231, 234, 238)
SIDEBAR_BG, CARD_BG = (235, 238, 242), (255, 255, 255)
CELL_COLOR, CELL_HOVER = (255, 255, 255), (246, 248, 252)
P1_COLOR, P2_COLOR = (238, 92, 105), (82, 143, 219)
WALL_COLOR, WALL_SHADOW = (82, 88, 98), (210, 214, 220)
VALID_MOVE_DOT = (205, 211, 220)
TEXT_COLOR, MUTED_TEXT = (78, 84, 94), (145, 151, 160)
ORANGE_COLOR = (255, 140, 0)

BOARD_SIZE = 9
CELL_SIZE = int(55 * SCALE)
WALL_THICKNESS = int(14 * SCALE)
PADDING = int(25 * SCALE)
BOARD_PIXEL_SIZE = (
    BOARD_SIZE * CELL_SIZE
    + (BOARD_SIZE - 1) * WALL_THICKNESS
    + PADDING * 2
)
SIDEBAR_WIDTH = int(380 * SCALE)
LEFT_SIDEBAR_WIDTH = int(370 * SCALE)
WIDTH = BOARD_PIXEL_SIZE + SIDEBAR_WIDTH + LEFT_SIDEBAR_WIDTH
HEIGHT = BOARD_PIXEL_SIZE + int(120 * SCALE)


class Quoridor(Trainer):
    def __init__(self, play_depth=4, train_depth=2, train_updates=16):
        super().__init__(
            headless=False,
            train_depth=train_depth,
            play_depth=play_depth,
            train_updates=train_updates,
        )
        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT), pygame.RESIZABLE | pygame.SRCALPHA
        )
        pygame.display.set_caption(
            "Quoridor - QuoriNet AI Engine [Press F11 for Fullscreen]"
        )
        self.virtual_screen = pygame.Surface((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont("sans-serif", int(24 * SCALE), bold=True)
        self.small_font = pygame.font.SysFont("sans-serif", int(16 * SCALE), bold=True)
        self.tiny_font = pygame.font.SysFont("sans-serif", int(13 * SCALE), bold=False)
        self.title_font = pygame.font.SysFont("sans-serif", int(38 * SCALE), bold=True)
        (
            self.rect_play, self.rect_save_manual, self.rect_play_manual,
            self.rect_manual_prev, self.rect_manual_next, self.rect_new_game,
            self.rect_menu,
        ) = [pygame.Rect(0, 0, 0, 0) for _ in range(7)]
        self.play_opponent_net = self.teacher_net
        self.play_opponent_label = f"Gen #{self.teacher_gen} Official"
        self.mode = "MENU"
        self.reset_game()

    def get_virtual_mouse_pos(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        screen_width, screen_height = self.screen.get_size()
        if screen_width <= 0 or screen_height <= 0:
            return mouse_x, mouse_y
        return (
            int(mouse_x * WIDTH / screen_width),
            int(mouse_y * HEIGHT / screen_height),
        )

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        flags = pygame.FULLSCREEN if self.is_fullscreen else pygame.RESIZABLE
        size = (0, 0) if self.is_fullscreen else (WIDTH, HEIGHT)
        self.screen = pygame.display.set_mode(size, flags)

    def render_and_flip(self):
        scaled = pygame.transform.smoothscale(
            self.virtual_screen, self.screen.get_size()
        )
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def reset_game(self):
        super().reset_game()
        network = getattr(self, "play_opponent_net", self.teacher_net)
        label = getattr(
            self, "play_opponent_label", f"Gen #{self.teacher_gen} Official"
        )
        self.bot1 = QuoridorBot(
            bot_id=1, max_depth=self.play_depth, nn_model=network,
            heuristic_weight=self.heuristic_weight
        )
        self.bot2 = QuoridorBot(
            bot_id=2, max_depth=self.play_depth, nn_model=network,
            heuristic_weight=self.heuristic_weight
        )
        self.p1_gen_str, self.p2_gen_str = "Human", label

    def select_manual_play_opponent(self):
        if not self.manual_snapshots:
            return False
        generation, iteration, _ = self.manual_snapshots[self.manual_snapshot_index]
        _, architecture, state_dict = self._archived_manual[(generation, iteration)]
        candidate = ArchivedValueNet(architecture, state_dict)
        candidate.eval()
        self.play_opponent_net = candidate
        self.play_opponent_label = f"Gen #{generation} Manual #{iteration}"
        return True

    def _get_center_pixels(self, pos):
        offset_x = PADDING + LEFT_SIDEBAR_WIDTH
        offset_y = int(85 * SCALE) + PADDING
        unit = CELL_SIZE + WALL_THICKNESS
        return (
            offset_x + pos[0] * unit + CELL_SIZE / 2,
            offset_y + pos[1] * unit + CELL_SIZE / 2,
        )

    def draw_menu(self):
        self.virtual_screen.fill(BG_COLOR)
        title = self.title_font.render("QuoriNet AI Engine", True, TEXT_COLOR)
        self.virtual_screen.blit(
            title, (WIDTH // 2 - title.get_width() // 2, int(80 * SCALE))
        )

        button_width, button_height = int(320 * SCALE), int(55 * SCALE)
        button_x = WIDTH // 2 - button_width // 2
        self.rect_play = pygame.Rect(
            button_x, int(190 * SCALE), button_width, button_height
        )
        pygame.draw.rect(
            self.virtual_screen, P1_COLOR, self.rect_play,
            border_radius=int(10 * SCALE)
        )
        label = self.font.render(
            f"PLAY OFFICIAL GEN #{self.teacher_gen}", True, (255, 255, 255)
        )
        self.virtual_screen.blit(label, label.get_rect(center=self.rect_play.center))

        selector_y, arrow_width = int(265 * SCALE), int(45 * SCALE)
        self.rect_manual_prev = pygame.Rect(
            button_x, selector_y, arrow_width, button_height
        )
        self.rect_manual_next = pygame.Rect(
            button_x + button_width - arrow_width,
            selector_y, arrow_width, button_height
        )
        for rect, arrow in (
            (self.rect_manual_prev, "<"), (self.rect_manual_next, ">")
        ):
            pygame.draw.rect(
                self.virtual_screen, (220, 224, 230), rect,
                border_radius=int(10 * SCALE)
            )
            text = self.font.render(arrow, True, TEXT_COLOR)
            self.virtual_screen.blit(text, text.get_rect(center=rect.center))

        if self.manual_snapshots:
            generation, iteration, _ = self.manual_snapshots[
                self.manual_snapshot_index
            ]
            manual_label = f"Gen #{generation} Manual #{iteration}"
        else:
            manual_label = "No manual snapshots"
        text = self.small_font.render(manual_label, True, TEXT_COLOR)
        self.virtual_screen.blit(
            text, text.get_rect(center=(WIDTH // 2, selector_y + button_height // 2))
        )

        self.rect_play_manual = pygame.Rect(
            button_x, int(335 * SCALE), button_width, button_height
        )
        pygame.draw.rect(
            self.virtual_screen,
            P2_COLOR if self.manual_snapshots else (180, 184, 190),
            self.rect_play_manual,
            border_radius=int(10 * SCALE),
        )
        text = self.font.render("PLAY SELECTED MANUAL", True, (255, 255, 255))
        self.virtual_screen.blit(
            text, text.get_rect(center=self.rect_play_manual.center)
        )

        self.rect_save_manual = pygame.Rect(
            button_x, int(405 * SCALE), button_width, button_height
        )
        pygame.draw.rect(
            self.virtual_screen, (92, 102, 116), self.rect_save_manual,
            border_radius=int(10 * SCALE)
        )
        text = self.font.render("SAVE CURRENT STUDENT", True, (255, 255, 255))
        self.virtual_screen.blit(
            text, text.get_rect(center=self.rect_save_manual.center)
        )

        hint = self.tiny_font.render(
            "[F11/F: fullscreen, R/Space/right-click: rotate wall]",
            True, MUTED_TEXT
        )
        self.virtual_screen.blit(
            hint, (WIDTH // 2 - hint.get_width() // 2, int(490 * SCALE))
        )
        if self.toast_timer > 0:
            self.toast_timer -= 1
            toast = self.small_font.render(self.toast_msg, True, (46, 125, 50))
            self.virtual_screen.blit(
                toast, toast.get_rect(center=(WIDTH // 2, int(535 * SCALE)))
            )
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

        speed = 0.3
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

    def play(self):
        clock = pygame.time.Clock()
        running, bot_thinking, bot_result = True, False, None

        def think_bot(current_state):
            nonlocal bot_result, bot_thinking
            try:
                bot_result = self.bot2.get_move(current_state)
                if bot_result is None:
                    moves = self.bot2.get_valid_moves(current_state, 2)
                    if moves:
                        bot_result = ('move', moves[0])
            except Exception:
                moves = self.bot2.get_valid_moves(current_state, 2)
                if moves:
                    bot_result = ('move', moves[0])
            finally:
                bot_thinking = False

        while running:
            mouse_pos = self.get_virtual_mouse_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_F11, pygame.K_f):
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_c:
                        self.save_checkpoint(manual=True)

                if (
                    self.mode == "MENU"
                    and event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                ):
                    if self.rect_play.collidepoint(mouse_pos):
                        self.play_opponent_net = self.teacher_net
                        self.play_opponent_label = (
                            f"Gen #{self.teacher_gen} Official"
                        )
                        self.mode, self.is_training_active = "PLAY", False
                        self.reset_game()
                    elif (
                        self.rect_manual_prev.collidepoint(mouse_pos)
                        and self.manual_snapshots
                    ):
                        self.manual_snapshot_index = (
                            self.manual_snapshot_index - 1
                        ) % len(self.manual_snapshots)
                    elif (
                        self.rect_manual_next.collidepoint(mouse_pos)
                        and self.manual_snapshots
                    ):
                        self.manual_snapshot_index = (
                            self.manual_snapshot_index + 1
                        ) % len(self.manual_snapshots)
                    elif (
                        self.rect_play_manual.collidepoint(mouse_pos)
                        and self.select_manual_play_opponent()
                    ):
                        self.mode, self.is_training_active = "PLAY", False
                        self.reset_game()
                    elif self.rect_save_manual.collidepoint(mouse_pos):
                        self.save_manual_student()

                elif (
                    self.mode == "PLAY"
                    and event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                ):
                    if self.rect_menu.collidepoint(mouse_pos):
                        self.mode = "MENU"
                    elif self.rect_new_game.collidepoint(mouse_pos):
                        self.reset_game()

                if (
                    self.mode == "PLAY"
                    and not self.game_over
                    and not bot_thinking
                    and self.state['turn'] == 1
                ):
                    if (
                        event.type == pygame.KEYDOWN
                        and event.key in (pygame.K_r, pygame.K_SPACE)
                    ):
                        self.wall_orientation = (
                            'V' if self.wall_orientation == 'H' else 'H'
                        )
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 3:
                            self.wall_orientation = (
                                'V' if self.wall_orientation == 'H' else 'H'
                            )
                        elif event.button == 1:
                            hover_type, col, row, orientation = self.draw_board(
                                mouse_pos
                            )
                            if (
                                hover_type == 'cell'
                                and (col, row)
                                in self.bot2.get_valid_moves(self.state, 1)
                            ):
                                self.state = self.bot2.apply_action(
                                    self.state, 1, ('move', (col, row))
                                )
                                self.check_win()
                            elif (
                                hover_type == 'wall'
                                and self.state['p1_walls'] > 0
                                and self.bot2.is_valid_wall(
                                    self.state, col, row, orientation
                                )
                            ):
                                self.state = self.bot2.apply_action(
                                    self.state, 1,
                                    ('wall', (col, row, orientation))
                                )
                                self.check_win()

            if self.mode == "PLAY" and self.state['turn'] == 2 and not self.game_over:
                if not bot_thinking and bot_result is None:
                    bot_thinking = True
                    bot_state = {
                        'p1_pos': self.state['p1_pos'],
                        'p2_pos': self.state['p2_pos'],
                        'p1_walls': self.state['p1_walls'],
                        'p2_walls': self.state['p2_walls'],
                        'walls_h': self.state['walls_h'].copy(),
                        'walls_v': self.state['walls_v'].copy(),
                        'turn': 2,
                        'turns_played': self.state['turns_played'],
                        'history': self.state['history'].copy(),
                        'p1_recent': self.state.get('p1_recent', ()),
                        'p2_recent': self.state.get('p2_recent', ()),
                    }
                    threading.Thread(
                        target=think_bot, args=(bot_state,), daemon=True
                    ).start()

                if not bot_thinking and bot_result is not None:
                    self.state = self.bot2.apply_action(
                        self.state, 2, bot_result,
                        is_random=getattr(self.bot2, 'was_last_random', False)
                    )
                    bot_result = None
                    self.check_win()

            if self.mode == "MENU":
                self.draw_menu()
            else:
                self.draw_board(mouse_pos)
            clock.tick(60)
        pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--threads', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--train-depth', type=int, default=2)
    parser.add_argument('--play-depth', type=int, default=4)
    parser.add_argument('--train-updates', type=int, default=16)
    parser.add_argument('--checkpoint-dir', default=None)
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
    Quoridor(
        play_depth=args.play_depth,
        train_depth=args.train_depth,
        train_updates=args.train_updates,
    ).play()


if __name__ == "__main__":
    main()
