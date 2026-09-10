import unittest
from collections import deque

import torch

import placebot


def initial_state():
    return {
        'p1_pos': (4, 8), 'p2_pos': (4, 0),
        'p1_walls': 10, 'p2_walls': 10,
        'walls_h': set(), 'walls_v': set(),
        'turn': 1, 'turns_played': 0, 'history': set(),
        'p1_recent': (), 'p2_recent': (),
        'random_walls': set(), 'p1_last_random': False, 'p2_last_random': False
    }


class NeuralShapeContractTests(unittest.TestCase):
    def test_encode_state_is_exact_float32_shape(self):
        encoded = placebot.encode_state(initial_state(), 1)
        self.assertEqual(encoded.shape, torch.Size([8, 9, 9]))
        self.assertEqual(encoded.dtype, torch.float32)
        self.assertTrue(torch.isfinite(encoded).all().item())

    def test_model_preserves_batch_and_returns_one_value(self):
        model = placebot.SpatialQuoriNetV2()
        batch = torch.zeros((4, 8, 9, 9), dtype=torch.float32)
        output = model(batch)
        self.assertEqual(output.shape, torch.Size([4, 1]))

    def test_model_rejects_unbatched_state(self):
        model = placebot.SpatialQuoriNetV2()
        with self.assertRaises(ValueError):
            model(torch.zeros((8, 9, 9), dtype=torch.float32))

    def test_model_rejects_wrong_channel_count(self):
        model = placebot.SpatialQuoriNetV2()
        with self.assertRaises(ValueError):
            model(torch.zeros((2, 7, 9, 9), dtype=torch.float32))

    def test_value_batch_is_column_vector(self):
        targets = placebot.make_value_batch([1.0, 0.0, -1.0], device="cpu")
        self.assertEqual(targets.shape, torch.Size([3, 1]))
        self.assertEqual(targets.dtype, torch.float32)

    def test_loss_rejects_broadcastable_but_wrong_shape(self):
        predictions = torch.zeros((3, 1), dtype=torch.float32)
        wrong_targets = torch.zeros(3, dtype=torch.float32)
        with self.assertRaises(ValueError):
            placebot.checked_value_loss(
                torch.nn.MSELoss(),
                predictions,
                wrong_targets
            )

    def test_state_batch_rejects_one_bad_sample(self):
        good = torch.zeros((8, 9, 9), dtype=torch.float32)
        bad = torch.zeros((8, 8, 9), dtype=torch.float32)
        with self.assertRaises(ValueError):
            placebot.make_state_batch([good, bad], device="cpu")


class HistoricGreedyPolicyTests(unittest.TestCase):
    def make_training_shell(self, is_latest):
        game = placebot.Quoridor.__new__(placebot.Quoridor)
        game.current_opponent_is_latest = is_latest
        game.student_net = object()
        game.state = {'turns_played': 2}
        return game

    def test_historic_matchup_disables_student_epsilon(self):
        game = self.make_training_shell(is_latest=False)
        student = placebot.QuoridorBot(nn_model=game.student_net)
        self.assertEqual(game._training_epsilon(student), 0.0)

    def test_latest_champion_matchup_keeps_student_epsilon(self):
        game = self.make_training_shell(is_latest=True)
        student = placebot.QuoridorBot(nn_model=game.student_net)
        self.assertAlmostEqual(game._training_epsilon(student), 0.30 * 0.85 ** 2)

    def test_teacher_never_uses_epsilon(self):
        game = self.make_training_shell(is_latest=True)
        teacher = placebot.QuoridorBot(nn_model=object())
        self.assertEqual(game._training_epsilon(teacher), 0.0)


class OutcomeGatedTrainingTests(unittest.TestCase):
    def make_training_shell(self, is_latest):
        game = placebot.Quoridor.__new__(placebot.Quoridor)
        game.current_opponent_gen = 3
        game.current_opponent_is_latest = is_latest
        game.student_id = 2
        game.episode_trajectory = []
        game.OUTCOME_BACKFILL_MIX = 0.70
        game.OUTCOME_BACKFILL_DISCOUNT = 0.995
        game.teacher_net = placebot.SpatialQuoriNetV2()
        game.teacher_net.eval()
        game.committed_samples = []
        game.optimizer_step_calls = 0

        def record_sample(state, target, opponent_gen):
            game.committed_samples.append((state, target, opponent_gen))

        def record_optimizer_step():
            game.optimizer_step_calls += 1
            return True

        game._append_replay_sample = record_sample
        game._optimize_student_from_replay = record_optimizer_step
        return game

    def test_historical_win_discards_episode_without_optimizer_step(self):
        game = self.make_training_shell(is_latest=False)
        state = torch.zeros((8, 9, 9), dtype=torch.float32)
        game.train_step(state, 2, state, 1, True, winner_id=2)
        self.assertEqual(game.committed_samples, [])
        self.assertEqual(game.optimizer_step_calls, 0)
        self.assertEqual(game.episode_trajectory, [])

    def test_historical_draw_discards_episode_without_optimizer_step(self):
        game = self.make_training_shell(is_latest=False)
        state = torch.zeros((8, 9, 9), dtype=torch.float32)
        game.train_step(state, 2, state, 1, True, winner_id=0)
        self.assertEqual(game.committed_samples, [])
        self.assertEqual(game.optimizer_step_calls, 0)

    def test_historical_loss_commits_and_replays_deferred_steps(self):
        game = self.make_training_shell(is_latest=False)
        state = torch.zeros((8, 9, 9), dtype=torch.float32)
        game.train_step(state, 2, state, 1, False, winner_id=0)
        self.assertEqual(game.committed_samples, [])
        self.assertEqual(game.optimizer_step_calls, 0)

        game.train_step(state, 1, state, 2, True, winner_id=1)
        # Two bootstrap samples plus two terminal-outcome backfill samples.
        self.assertEqual(len(game.committed_samples), 4)
        self.assertEqual(game.optimizer_step_calls, 2)
        self.assertEqual(game.episode_trajectory, [])

    def test_latest_win_updates_immediately_and_keeps_terminal_backfill(self):
        game = self.make_training_shell(is_latest=True)
        state = torch.zeros((8, 9, 9), dtype=torch.float32)
        game.train_step(state, 2, state, 1, True, winner_id=2)
        self.assertEqual(len(game.committed_samples), 2)
        self.assertEqual(game.optimizer_step_calls, 1)


class DeficitSamplingTests(unittest.TestCase):
    def make_sampling_shell(self, win_rates):
        game = placebot.Quoridor.__new__(placebot.Quoridor)
        game.teacher_gen = len(win_rates)
        game.GATE_WIN_RATE_THRESHOLD = 0.57
        game.GATE_MIN_GAMES = 30
        game.CURRICULUM_UNIFORM_MASS = 0.25
        game.opponent_pool = [
            (object(), gen)
            for gen in range(1, len(win_rates) + 1)
        ]
        game.checkpoint_log = {
            gen: deque(
                [1] * wins + [0] * (30 - wins),
                maxlen=30
            )
            for gen, wins in enumerate(win_rates, start=1)
        }
        return game

    def test_every_generation_uses_the_same_required_win_rate(self):
        game = self.make_sampling_shell([17, 14])
        self.assertEqual(game._required_win_rate(1), 0.57)
        self.assertEqual(game._required_win_rate(2), 0.57)

    def test_lower_win_rate_receives_more_deficit_sampling(self):
        # 17/30 is 56.7%; 14/30 is 46.7%. Gen #2 must receive more weight.
        game = self.make_sampling_shell([17, 14])
        game._refresh_opponent_weights()
        self.assertGreater(game.opponent_weights[1], game.opponent_weights[0])

    def test_equal_win_rates_receive_equal_sampling_weight(self):
        game = self.make_sampling_shell([14, 14, 14])
        game._refresh_opponent_weights()
        self.assertEqual(game.opponent_weights, [1.0 / 3.0] * 3)

    def test_exact_threshold_passes_for_every_generation(self):
        game = self.make_sampling_shell([17, 14])
        for gen in (1, 2):
            self.assertTrue(game._passes_generation_gate(gen, 0.57))


if __name__ == "__main__":
    unittest.main()
