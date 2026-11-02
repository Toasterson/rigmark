import copy
import io
import hashlib
import itertools
from pathlib import Path
import json
import unittest
from unittest.mock import patch

import agentic
import bench


class AgenticTest(unittest.TestCase):
    def test_arrivals_repeat_with_seed(self):
        offsets = agentic.arrivals(5, .5, 1)
        self.assertEqual(offsets, agentic.arrivals(5, .5, 1))
        self.assertEqual(0, offsets[0])
        self.assertEqual(sorted(offsets), offsets)
        self.assertNotEqual(offsets, agentic.arrivals(5, .5, 2))

    def test_tool_only_stream_and_fragmented_arguments(self):
        events = [
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call1',
                         'function': {'name': 'read_config', 'arguments': '{"path":'}}]}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0,
                         'function': {'arguments': '"config.json"}'}}]}, 'finish_reason': 'tool_calls'}],
             'usage': {'prompt_tokens': 40, 'completion_tokens': 10}},
        ]
        response = io.BytesIO((''.join('data: ' + json.dumps(e) + '\n\n' for e in events)
                               + 'data: [DONE]\n\n').encode())
        with patch('bench.urllib.request.urlopen', return_value=response):
            row = bench.Client('http://example.test', '', 1).stream('/v1/chat/completions', {})
        self.assertTrue(agentic.turn_valid(0, row))
        self.assertEqual('', row['output'])
        self.assertEqual(2, row['measured_sse_events'])

    def test_session_grows_history_and_failures_are_not_completed(self):
        class Client:
            def __init__(self, valid=True):
                self.requests = []
                self.valid = valid

            def stream(self, path, payload):
                self.requests.append(copy.deepcopy(payload))
                turn = len(self.requests)
                return {'stream_done_marker': True,
                        'finish_reason': 'tool_calls' if turn == 1 else 'stop',
                        'output': '' if turn == 1 else json.dumps(agentic.CONFIG) if turn == 2 else 'READY',
                        'tool_calls': [{'id': 'call', 'function': {'name': 'read_config',
                                        'arguments': '{"path":"config.json"}' if self.valid else '{}'}}]}
        settings = {'max_tokens': 128, 'seed': 1, 'tool_latency_seconds': .25}
        good = Client()
        with patch('agentic.time.sleep') as sleep:
            result = agentic.run_session(good, 'm', 1, 0, 0, settings)
        self.assertTrue(result['completed'])
        self.assertEqual([2, 5, 7], [len(p['messages']) for p in good.requests])
        self.assertEqual('tool', good.requests[1]['messages'][3]['role'])
        self.assertIn(unittest.mock.call(.25), sleep.call_args_list)
        with patch('agentic.time.sleep'):
            failed = agentic.run_session(Client(False), 'm', 1, 0, 0, settings)
        self.assertFalse(failed['completed'])
        summary = agentic.summarise([failed])
        self.assertEqual(0, summary['completed_turns'])
        self.assertIsNone(summary['p99_turn_seconds'])

    def test_receipt_recomputes_requests_gates_and_summaries(self):
        template = json.loads(next(Path('results/reference').glob('*.json')).read_text())['decode']['code']['runs'][0]

        class Client:
            calls = 0

            def stream(self, path, payload):
                self.calls += 1
                row = copy.deepcopy(template)
                output = '' if self.calls == 1 else json.dumps(agentic.CONFIG) if self.calls == 2 else 'READY'
                row.update(output=output, output_characters=len(output),
                           output_sha256=hashlib.sha256(output.encode()).hexdigest(),
                           ttft_seconds=.1, time_to_first_visible_seconds=None,
                           time_to_last_output_seconds=1.1, decode_seconds=1,
                           completion_tokens=11, decode_tokens_per_second=10,
                           wall_seconds=1.1, response_tail_seconds=0,
                           measured_sse_events=10, stream_done_marker=True,
                           finish_reason='tool_calls' if self.calls == 1 else 'stop')
                if self.calls == 1:
                    row['tool_calls'] = [{'id': 'call', 'type': 'function', 'function': {
                        'name': 'read_config', 'arguments': '{"path":"config.json"}'}}]
                return row

        settings = {'max_tokens': 128, 'seed': 1, 'tool_latency_seconds': .25,
                    'sessions': 1, 'arrival_rate': .5, 'workers': 1}
        with patch('agentic.time.sleep'), patch('agentic.time.monotonic', side_effect=itertools.count(0, 2)):
            session = agentic.run_session(Client(), 'm', 1, 0, 0, settings)
        result = {'schema': 1, 'profile': agentic.PROFILE, 'settings': settings,
                  'run': {'model': 'm'}, 'agentic': {'sessions': [session],
                                                    'summary': agentic.summarise([session])}}
        self.assertEqual([], agentic.validate_result(result))
        bad = copy.deepcopy(result)
        bad['agentic']['summary']['completed_turns'] = 99
        self.assertTrue(agentic.validate_result(bad))
        bad = copy.deepcopy(result)
        bad['agentic']['sessions'][0]['turns'][1]['request']['messages'][3]['content'] = '{}'
        self.assertTrue(agentic.validate_result(bad))

    def test_tail_uses_completed_turns_with_failures_disclosed(self):
        sessions = [{'completed': False, 'wall_seconds': 103, 'turns': [
            {'completed': True, 'turn_seconds': 1},
            {'completed': True, 'turn_seconds': 2},
            {'completed': False, 'turn_seconds': 100}]}]
        summary = agentic.summarise(sessions)
        self.assertEqual(51.5, summary['time_per_completed_turn_seconds'])
        self.assertEqual(1.5, summary['mean_completed_turn_seconds'])
        self.assertEqual(2, summary['p95_turn_seconds'])
        self.assertEqual(3, summary['attempted_turns'])
