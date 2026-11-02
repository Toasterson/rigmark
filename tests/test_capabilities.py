import unittest
import json
from pathlib import Path

from receipt import validate_result
import urllib.error
from unittest.mock import patch

import bench


class CapabilitiesTest(unittest.TestCase):
    def probe(self, counts):
        def once(*args):
            return {'cached_prompt_tokens': next(counts)}
        with patch('bench.exact_token_ids', return_value=[1] * 64), \
             patch('bench.prefill_once', side_effect=once):
            return bench.probe_prefill(None, 'm')

    def test_functional_isolation_and_missing_evidence(self):
        self.assertEqual('supported', self.probe(iter([0, 64, 0]))['status'])
        self.assertEqual('unsupported', self.probe(iter([0, 64, 64]))['status'])
        self.assertEqual('unknown', self.probe(iter([None, None, None]))['status'])
        self.assertEqual('unknown', self.probe(iter([0, 0, 0]))['status'])

    def test_rejections_are_distinct_from_auth_or_outage(self):
        for code in (400, 404, 422, 401, 403, 429, 500):
            error = urllib.error.HTTPError('http://example.test', code, 'error', {}, None)
            with patch('bench.exact_token_ids', side_effect=error):
                if code in (400, 404, 422):
                    self.assertEqual('unsupported', bench.probe_prefill(None, 'm')['status'])
                else:
                    with self.assertRaises(urllib.error.HTTPError):
                        bench.probe_prefill(None, 'm')

    def test_bad_tokenizer_does_not_claim_support(self):
        with patch('bench.exact_token_ids', side_effect=RuntimeError('empty tokenizer')):
            self.assertEqual('unsupported', bench.probe_prefill(None, 'm')['status'])

    def test_forged_support_or_malformed_probes_are_refused(self):
        result = json.loads(next(Path('results/reference').glob('*.json')).read_text())
        result['settings']['capability_detection'] = True
        result['capabilities'] = {'prefill': {'status': 'supported', 'reason': 'claimed', 'probe_runs': []}}
        self.assertTrue(any('support lacks' in e for e in validate_result(result)))
        result['settings']['prefill_depths'] = []
        result['capabilities']['prefill'].update(status='unknown', probe_runs=None)
        self.assertTrue(any('probe_runs is malformed' in e for e in validate_result(result)))

    def test_detection_omits_only_prefill_and_keeps_chat_phases(self):
        argv = ['bench.py', '--base-url', 'http://localhost:1', '--model', 'm',
                '--label', 'test', '--comparison-id', 'id', '--metadata', 'unused',
                '--detect-capabilities']
        with patch('sys.argv', argv), \
             patch('bench.load_metadata', return_value={'context_limit': 4096}), \
             patch('bench.probe_prefill', return_value={'status': 'unsupported', 'reason': 'tokenize: HTTP 400', 'probe_runs': []}), \
             patch('bench.run_decode', return_value={}) as decode, \
             patch('bench.run_prefill') as prefill, \
             patch('bench.run_concurrency', return_value={}) as concurrency, \
             patch('bench.write_result') as write, \
             patch('release_check.check_release', return_value={'status': 'identical'}), \
             patch('release_check.format_check', return_value='current'), \
             patch('report.save_report', return_value=Path('unused.card.txt')), \
             patch('report.print_report'), patch('builtins.print'):
            bench.main()
        decode.assert_called_once()
        concurrency.assert_called_once()
        prefill.assert_not_called()
        result = write.call_args.args[0]
        self.assertEqual([], result['settings']['prefill_depths'])
        self.assertEqual('unsupported', result['capabilities']['prefill']['status'])
        self.assertEqual(3, len(result['capabilities']['prefill']['unsupported_depths']))
