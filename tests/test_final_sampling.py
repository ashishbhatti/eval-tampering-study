"""Accepted final collection, using tiny generation and an explicitly simulated host."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run
from eval_tampering.analysis import ResultAnalyzer
from eval_tampering.experiment import acceptance, final_sampling_plan, review_template
from eval_tampering.messages import artifact_ref, atomic_json, fingerprint
from eval_tampering.model import ModelRuntime
from eval_tampering.monitors.activation_monitor import ActivationMonitor
from eval_tampering.monitors.common import json_artifact
from eval_tampering.monitors.text_monitor import TextMonitor
from test_final_runs import accepted_fixture, FixtureHost
from test_model import AVAILABLE
from test_run import RecordingEvaluator


def final_sampling_fixture(root):
    data = accepted_fixture(root)
    try:
        plan = final_sampling_plan(data['acceptance'], 'detection_test')
        rows = []
        for index in (0, 1):
            job = plan['jobs'][index]
            host = data['host'] if index == 0 else data['host'] | {'image_id': 'sha256:' + '1'*64}
            result = run.run_episode(job['episode'], data['runtime'], FixtureHost(data['root'], host), RecordingEvaluator())
            assert result['status'] == ('ok' if index == 0 else 'error'), result
            if index == 1:
                assert result['error']['code'] == 'hash_mismatch', result
            folder = Path(job['episode']['config']['artifact_root']) / fingerprint({'episode': job['job_id']})[:32]
            rows.append({'job_id': job['job_id'], 'episode': artifact_ref(folder / 'record.json', 'json'), 'reviews': None})
        atomic_json(data['root'] / 'final-outcomes.json', {'schema_version': 1, 'records': rows})
        analyzer = ResultAnalyzer(data['manifest']['components']['analysis']['config'])
        inputs = {'plan': data['manifest_ref'], 'rule': data['manifest']['inputs']['analysis'],
            'outcomes': artifact_ref(data['root'] / 'final-outcomes.json', 'json'),
            'final': {'acceptance': data['acceptance'], 'split': 'detection_test'}}
        return data | {'plan': plan, 'analyzer': analyzer, 'inputs': inputs, 'rows': rows}
    finally:
        data['runtime'].close()


@unittest.skipUnless(AVAILABLE, 'Install tiny-model dependencies and pinned tokenizer assets')
class FinalSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(dir='.cache', prefix='final-sampling-tests-')
        cls.data = final_sampling_fixture(cls.work.name)

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def save(self, name, value):
        path = self.data['root'] / (name + '.json')
        atomic_json(path, value)
        return artifact_ref(path, 'json')

    def invoke(self, inputs, name):
        analyzer = self.data['analyzer']
        return analyzer.handle({'schema_version': 1, 'request_id': name, 'operation': 'analysis.sampling',
            'config': analyzer.config, 'inputs': inputs})

    def test_all_slots_failures_provenance_and_split_budget_without_new_execution(self):
        with patch.object(ModelRuntime, 'load', side_effect=AssertionError('No model loading during analysis')), \
                patch.object(run, 'run_episode', side_effect=AssertionError('No episode execution during analysis')), \
                patch.object(ActivationMonitor, 'fit', side_effect=AssertionError('No fitting during final analysis')), \
                patch.object(TextMonitor, 'score', side_effect=AssertionError('No provider scoring during analysis')):
            packet = self.invoke(self.data['inputs'], 'final-summary')
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        pooled = result['summaries'][0]
        self.assertEqual((result['stage'], result['split'], result['planned_slots']), ('final', 'detection_test', 16))
        self.assertEqual(result['manifest'], self.data['manifest_ref'])
        self.assertEqual(result['final'], self.data['inputs']['final'])
        self.assertEqual((pooled['supplied_episodes'], pooled['completed_calls']), (2, 0))
        self.assertEqual(pooled['review_statuses'], {'partial': 1, 'runtime_unavailable': 1, 'episode_missing': 14})
        self.assertEqual(pooled['metrics']['tampering']['bounds'], {'lower': 0, 'upper': 1})
        self.assertIsNone(pooled['metrics']['tampering']['rate']['value'])
        self.assertEqual(result['usage']['recorded_output_tokens'], {'known_sum': 16, 'known_slots': 2, 'unknown_slots': 14})
        self.assertEqual(result['declared_budget']['max_output_tokens'], 256)
        self.assertEqual(result['shared_phase_budget']['max_output_tokens'], 512)
        self.assertEqual(len(json_artifact(result['slots'])['rows']), 16)
        self.assertEqual(len(json_artifact(result['features'])['records']), 1)
        self.assertEqual(result['excluded'][0]['reason'], 'not_a_completed_tool_call')
        self.assertEqual(result['new_model_calls'] + result['new_provider_calls'] + result['monitor_fits'], 0)

    def test_frozen_behavioral_and_qualitative_samples_are_blinded_and_reproducible(self):
        import random
        analyzer = self.data['analyzer']
        summary = self.invoke(self.data['inputs'], 'sample-source')['result']
        inputs = {'manifest': self.data['manifest_ref'], 'sampling': summary['summary'],
            'purpose': 'behavior_audit', 'rule': self.data['manifest']['inputs']['analysis']}
        result = analyzer.sample(inputs, self.data['root'] / 'audit-sample')
        self.assertEqual(result['status'], 'prepared')
        self.assertEqual((result['eligible_count'], result['planned_count'], result['uniform_selected']), (2, 16, 2))
        self.assertEqual(len(result['excluded_missing_episode_ids']), 14)
        self.assertEqual(result['human_reviews_completed'], 0)
        index = json_artifact(result['private_index'])['records']
        ids = {row['sample_id']: row['job_id'] for row in index}
        expected = random.Random(self.data['manifest']['inputs']['audits']['behavior_seed']).sample(sorted(ids.values()), 2)
        self.assertEqual([ids[key] for key in result['uniform_ids']], expected)
        for row in json_artifact(result['review_template'])['records']:
            view = json_artifact(row['view'])
            def keys(value):
                if isinstance(value, dict):
                    return set(value) | set().union(*(keys(item) for item in value.values()))
                if isinstance(value, list):
                    return set().union(*(keys(item) for item in value))
                return set()
            self.assertFalse(keys(view) & {'primary_label', 'provisional_flags', 'monitors', 'intervention', 'arm_id', 'review'})
            self.assertTrue(all(value is None for value in row['flags'].values()))
        qualitative = analyzer.sample(inputs | {'purpose': 'qualitative'}, self.data['root'] / 'qualitative-sample')
        self.assertEqual(qualitative['uniform_ids'], result['uniform_ids'])
        self.assertEqual(qualitative['targeted_ids'], [])
        self.assertEqual(qualitative['sampling_rule'], result['sampling_rule'])

    def test_unaccepted_wrong_split_or_changed_episode_proofs_are_rejected(self):
        original = self.data['inputs']
        inputs = {key: value for key, value in original.items() if key != 'final'}
        self.assertEqual(self.invoke(inputs, 'missing-selector')['status'], 'error')
        self.assertEqual(self.invoke(original | {'rule': None}, 'missing-frozen-rule')['error']['code'], 'hash_mismatch')
        copied_manifest = self.save('copied-manifest', self.data['manifest'])
        self.assertEqual(self.invoke(original | {'plan': copied_manifest}, 'different-manifest-reference')['error']['code'], 'hash_mismatch')
        for split in ('training', 'intervention_test'):
            inputs = original | {'final': original['final'] | {'split': split}}
            self.assertEqual(self.invoke(inputs, 'wrong-' + split)['status'], 'error')
        for key, value in [('stage', 'development'), ('final', None), ('manifest', None)]:
            rows = deepcopy(self.data['rows'])
            episode = json_artifact(rows[0]['episode']) | {key: value}
            rows[0]['episode'] = self.save('changed-' + key, episode)
            outcomes = self.save('outcomes-' + key, {'schema_version': 1, 'records': rows})
            packet = self.invoke(original | {'outcomes': outcomes}, 'rejected-' + key)
            self.assertEqual(packet['status'], 'error', packet)
            self.assertEqual(packet['error']['code'], 'hash_mismatch')
        review = self.save('unfilled-review', review_template(self.data['manifest_ref'], self.data['manifest']))
        pending = self.save('unaccepted', acceptance({'manifest': self.data['manifest_ref'], 'review': review}))
        packet = self.invoke(original | {'final': original['final'] | {'acceptance': pending}}, 'unaccepted-analysis')
        self.assertEqual(packet['error']['code'], 'acceptance_required')


if __name__ == '__main__':
    unittest.main()
