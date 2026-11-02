import copy
import json
import unittest
from pathlib import Path

from bench import cached_token_usage
from receipt import validate_result
from report import render


class CacheEvidenceTest(unittest.TestCase):
    def test_explicit_counts_and_unknown(self):
        self.assertEqual((None, None), cached_token_usage({}, 10))
        self.assertEqual((7, 'usage.prompt_cache_hit_tokens'),
                         cached_token_usage({'prompt_cache_hit_tokens': 7}, 10))
        self.assertEqual((0, 'usage.prompt_tokens_details.cached_tokens'),
                         cached_token_usage({'prompt_tokens_details': {'cached_tokens': 0}}, 10))
        for invalid in (-1, True, '0', 11):
            with self.assertRaises(RuntimeError):
                cached_token_usage({'prompt_cache_hit_tokens': invalid}, 10)
        with self.assertRaisesRegex(RuntimeError, 'conflicting'):
            cached_token_usage({'prompt_cache_hit_tokens': 1,
                                'prompt_tokens_details': {'cached_tokens': 0}}, 10)

    def test_old_receipts_flag_missing_counts_and_evidence_is_validated(self):
        for path in Path('results/reference').glob('*.json'):
            result = json.loads(path.read_text())
            self.assertEqual([], validate_result(result))
            card = render(result, 'a' * 64)
            self.assertIn('CACHE EVIDENCE', card)
            if any(row.get('cached_prompt_tokens') is None
                   for row in result['decode']['code']['runs']):
                self.assertIn('CACHE UNKNOWN', card)
            bad = copy.deepcopy(result)
            bad['decode']['code']['runs'][0]['cache_evidence'] = {'status': 'reported', 'source': None}
            self.assertTrue(any('cache_evidence' in e for e in validate_result(bad)))
