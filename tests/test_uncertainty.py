import copy
import unittest
from unittest.mock import patch

import bench
from uncertainty import interval, paired


class UncertaintyTest(unittest.TestCase):
    def test_deterministic_and_small_samples(self):
        self.assertEqual(interval([1] * 4)['status'], 'insufficient_samples')
        self.assertEqual(interval([1] * 5)['lower'], 1)
        self.assertEqual(interval(list(range(10))), interval(list(range(10))))

    def test_pairs_by_nonce_and_refuses_invalid_pairs(self):
        left = [{'request_nonce': str(i), 'decode_tokens_per_second': 10 + i,
                 'completion_validation': {'valid': True}} for i in range(8)]
        right = copy.deepcopy(left)
        for row in right:
            row['decode_tokens_per_second'] *= 1.1
        self.assertTrue(paired(left, right[::-1])['resolvable'])
        self.assertAlmostEqual(paired(left, right)['lower'], 10)
        self.assertFalse(paired(left, left)['resolvable'])
        with self.assertRaises(ValueError):
            paired(left, right[:-1])
        with self.assertRaises(ValueError):
            paired(left, right + right[:1])
        right[0]['completion_validation']['valid'] = False
        with self.assertRaises(ValueError):
            paired(left, right)

    def test_warmups_excluded_and_measured_nonces_unchanged(self):
        class Client:
            def stream(self, path, payload):
                return {'decode_tokens_per_second': 10, 'completion_tokens': 11,
                        'decode_seconds': 1, 'ttft_seconds': .1, 'wall_seconds': 1.1,
                        'time_to_last_output_seconds': 1.1, 'output': 'ok',
                        'finish_reason': 'stop', 'stream_done_marker': True}
        prompts = {'system': 'system', 'workloads': {'prose': 'prompt'}}
        with patch('builtins.print'):
            old = bench.run_decode(Client(), 'm', prompts, 5, 16, 1, {}, 'id')
            new = bench.run_decode(Client(), 'm', prompts, 5, 16, 1, {}, 'id', 2, True)
        self.assertEqual(old['prose']['runs'], new['prose']['runs'])
        self.assertEqual(2, len(new['prose']['warmup_runs']))
        self.assertEqual(5, new['prose']['completion_gate']['total'])
        self.assertTrue(new['prose']['warmup_runs'][0]['excluded'])
