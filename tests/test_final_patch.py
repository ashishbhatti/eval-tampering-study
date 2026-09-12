"""Accepted final patch wiring; scripted baseline reviews/captures are not research evidence."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import run
from eval_tampering.analysis import ResultAnalyzer
from eval_tampering.experiment import final_job, final_patch_plan, final_sampling_plan, read_final_patch
from eval_tampering.interventions import InterventionPlanner
from eval_tampering.messages import InputError, artifact_ref, atomic_json
from eval_tampering.model import ModelRuntime
from eval_tampering.monitors.activation_monitor import ActivationMonitor
from eval_tampering.monitors.common import json_artifact
from eval_tampering.monitors.text_monitor import TextMonitor
from test_final_detection import scripted_final_call
from test_final_runs import accepted_fixture, FixtureHost
from test_model import AVAILABLE
from test_run import RecordingEvaluator


def final_patch_fixture(root):
    """Caller owns the real tiny runtime. Baseline calls and labels below are scripted."""
    data = accepted_fixture(root, max_new_tokens=128)
    try:
        baseline = final_sampling_plan(data['acceptance'], 'intervention_test')
        outcomes, captures = [], []
        for index, job in enumerate(baseline['jobs']):
            evidence, record_id = scripted_final_call(data, baseline, job, positive=bool(index % 2),
                capture=False, sample_note=f' Sample {index}.')
            episode = json_artifact(evidence['episode'])
            folder = Path(evidence['episode']['path']).parent
            if index == 11:
                episode.update(status='incomplete', runtime_sha256=None, fixture=None, actions=[], output_tokens=0,
                    final_assessment=None, error={'code': 'component_error', 'message': 'Simulated host failure; no baseline generation.'})
                atomic_json(Path(evidence['episode']['path']), episode)
                evidence.update(episode=artifact_ref(evidence['episode']['path'], 'json'), reviews=None)
            else:
                capture = None
                if index != 7:
                    generated = json_artifact(episode['actions'][0]['generation'])['result']
                    start = generated['action_span'][0]
                    vector = np.zeros(data['runtime'].identity['config']['hidden_size'])
                    vector[:2] = [3 * (index % 2) + (index % 4)/10, index/10]
                    np.savez_compressed(folder / 'pre.npz', residuals=vector[None, None, :], layers=np.array([1]), positions=np.array([start-1]))
                    atomic_json(folder / 'pre.json', {'schema_version': 1, 'status': 'ok', 'result': {
                        'target': 'pre_action', 'trajectory': generated['tokens'], 'positions': [start-1], 'prediction_boundary': start-1,
                        'causal_prefix_length': start, 'shape': [1, 1, len(vector)], 'features': artifact_ref(folder / 'pre.npz', 'npz'),
                        'runtime_sha256': baseline['runtime_sha256']}})
                    capture = artifact_ref(folder / 'pre.json', 'json')
                captures.append({'record_id': record_id, 'capture': capture})
            outcomes.append(evidence)
        atomic_json(data['root'] / 'final-outcomes.json', {'schema_version': 1, 'records': outcomes})
        atomic_json(data['root'] / 'final-captures.json', {'schema_version': 1, 'records': captures})
        analyzer = ResultAnalyzer(data['manifest']['components']['analysis']['config'])
        collection = analyzer.sampling({'plan': data['manifest_ref'], 'rule': data['manifest']['inputs']['analysis'],
            'outcomes': artifact_ref(data['root'] / 'final-outcomes.json', 'json'),
            'final': {'acceptance': data['acceptance'], 'split': 'intervention_test'}}, data['root'] / 'final-collection')
        frozen = data['manifest']['inputs']
        planner = InterventionPlanner(frozen['interventions']['config'])
        instructions = planner.make_instructions({'direction': frozen['interventions']['direction'],
            'features': collection['features'], 'labels': collection['labels'], 'captures': artifact_ref(data['root'] / 'final-captures.json', 'json'),
            'split': 'intervention_test', 'seeds': frozen['patch']['seeds'], 'generation': frozen['episode_config']['generation'],
            'max_jobs': frozen['patch']['allocation']['max_jobs']}, data['root'] / 'final-instructions')
        binding = {'acceptance': data['acceptance'], 'collection': collection['summary'], 'instructions': instructions['instructions']}
        packet = run.handle(data['request'] | {'request_id': 'patch-binding', 'operation': 'experiment.patch', 'inputs': binding})
        assert packet['status'] == 'ok', packet
        return data | {'baseline': baseline, 'collection': collection, 'analyzer': analyzer, 'planner': planner,
            'instructions': instructions, 'binding': binding, 'plan_ref': packet['result']['artifact'], 'plan': json_artifact(packet['result']['artifact'])}
    except BaseException:
        data['runtime'].close()
        raise


class FinalProofInputTests(unittest.TestCase):
    def test_invalid_proof_types_fail_before_field_iteration(self):
        for proof in (None, [], [{}], 'patch', 1):
            with self.subTest(proof=proof), self.assertRaisesRegex(InputError, 'proof object'):
                final_job(proof)


@unittest.skipUnless(AVAILABLE, 'Install tiny-model dependencies and pinned tokenizer assets')
class FinalPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(dir='.cache', prefix='final-patch-tests-')
        cls.data = final_patch_fixture(cls.work.name)

    @classmethod
    def tearDownClass(cls):
        cls.data['runtime'].close()
        cls.work.cleanup()

    def save(self, name, value):
        path = self.data['root'] / (name + '.json')
        atomic_json(path, value)
        return artifact_ref(path, 'json')

    def proof(self, job=None, plan=None):
        job = job or next(row for row in self.data['plan']['jobs'] if row['control'] == 'target')
        return {'acceptance': self.data['acceptance'], 'phase': 'patch', 'plan': plan or self.data['plan_ref'], 'job_id': job['job_id']}

    def test_all_controls_fresh_seeds_failed_baseline_and_capture_coverage_without_calls(self):
        with patch.object(ModelRuntime, 'load', side_effect=AssertionError('No loading')), \
                patch.object(ActivationMonitor, 'fit', side_effect=AssertionError('No fitting')), \
                patch.object(TextMonitor, '_call', side_effect=AssertionError('No provider calls')):
            plan = read_final_patch(self.data['plan_ref'])
        self.assertEqual(plan, self.data['plan'])
        self.assertEqual((plan['stage'], plan['split'], plan['final_acceptance_verified']), ('final', 'intervention_test', True))
        self.assertEqual((plan['job_count'], len(plan['histories']), len(plan['missing_controls'])), (164, 6, 1))
        self.assertEqual(plan['baseline_coverage']['planned_slots'], 16)
        self.assertEqual(plan['baseline_coverage']['summaries'][0]['review_statuses'], {'reviewed': 15, 'runtime_unavailable': 1})
        self.assertEqual({row['recipient_class'] for row in plan['histories']}, {'repair', 'tampering'})
        for recipient in plan['histories']:
            jobs = [row for row in plan['jobs'] if row['history_id'] == recipient['history_id']]
            self.assertTrue({'baseline', 'sham', 'target', 'random-0', 'random-1', 'random-2'} <= {row['control'] for row in jobs})
            for control in {row['control'] for row in jobs}:
                self.assertEqual({row['seed'] for row in jobs if row['control'] == control}, {41, 42, 43, 44})
            self.assertTrue(all(row['episode']['config'] == self.data['manifest']['inputs']['episode_config'] for row in jobs))
        self.assertEqual(sum(row['reason'] == 'pre_action_capture_unavailable' for row in plan['excluded']), 1)

    def test_actual_tiny_continuation_and_copied_instruction_cannot_create_another_attempt(self):
        request, _ = final_job(self.proof())
        job = next(row for row in self.data['plan']['jobs'] if row['job_id'] == request['request_id'])
        result = run.run_episode(request, self.data['runtime'], FixtureHost(self.data['root'], self.data['host']), RecordingEvaluator())
        self.assertEqual(result['status'], 'ok', result)
        episode = result['result']
        self.assertEqual(episode['stage'], 'final')
        self.assertEqual(episode['manifest'], self.data['manifest_ref'])
        self.assertEqual(episode['initial_snapshot']['sha256'], job['before_snapshot']['sha256'])
        self.assertEqual(episode['initial_prefix'], job['initial_prefix'])
        generation = json_artifact(episode['actions'][0]['generation'])['result']
        events = generation['hook_events']
        self.assertGreater(generation['generated_tokens'], 0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['processed_positions'], [len(json_artifact(job['initial_prefix'])['token_ids'])-1])
        self.assertGreater(events[0]['change_norm_sum'], 0)
        original = Path(self.data['binding']['instructions']['path'])
        copied = original.with_name('copied-instructions.json')
        atomic_json(copied, json_artifact(self.data['binding']['instructions']))
        plan = final_patch_plan(self.data['binding'] | {'instructions': artifact_ref(copied, 'json')})
        self.assertEqual([row['job_id'] for row in plan['jobs']], [row['job_id'] for row in self.data['plan']['jobs']])
        request_again, _ = final_job(self.proof(job, self.save('copied-binding', plan)))
        with self.assertRaisesRegex(InputError, 'Episode already exists'):
            run.run_episode(request_again, self.data['runtime'], FixtureHost(self.data['root'], self.data['host']), RecordingEvaluator())

    def test_omitted_baseline_and_capture_entries_cannot_define_selected_subset(self):
        collection_inputs = self.data['collection']['inputs']
        outcomes = deepcopy(json_artifact(collection_inputs['outcomes']))
        outcomes['records'].pop()
        incomplete = self.data['analyzer'].sampling(collection_inputs | {'outcomes': self.save('omitted-outcome', outcomes)}, self.data['root'] / 'partial-collection')
        with self.assertRaises(InputError) as raised:
            final_patch_plan(self.data['binding'] | {'collection': incomplete['summary']})
        self.assertEqual(raised.exception.code, 'collection_incomplete')
        captures = deepcopy(json_artifact(self.data['instructions']['inputs']['captures']))
        captures['records'].pop()
        instructions = self.data['planner'].make_instructions(self.data['instructions']['inputs'] | {
            'captures': self.save('omitted-capture', captures)}, self.data['root'] / 'partial-captures')
        with self.assertRaises(InputError) as raised:
            final_patch_plan(self.data['binding'] | {'instructions': instructions['instructions']})
        self.assertEqual(raised.exception.code, 'capture_coverage')

    def test_frozen_settings_inventory_and_request_changes_are_rejected(self):
        for name, change in [('seed', {'seeds': [45, 46, 47, 48]}), ('cohort', {
                'features': self.save('copied-features', json_artifact(self.data['collection']['features']))})]:
            instructions = self.data['planner'].make_instructions(self.data['instructions']['inputs'] | change, self.data['root'] / (name + '-instructions'))
            with self.assertRaisesRegex(InputError, 'frozen settings or collection'):
                final_patch_plan(self.data['binding'] | {'instructions': instructions['instructions']})
        plan = deepcopy(self.data['plan'])
        plan['jobs'].pop()
        with self.assertRaisesRegex(InputError, 'inventory or its binding changed'):
            read_final_patch(self.save('omitted-control', plan))
        request, _ = final_job(self.proof())
        with patch.object(ModelRuntime, 'load', side_effect=AssertionError('No loading invalid requests')):
            for changed in (request | {'request_id': 'retry'}, request | {'inputs': request['inputs'] | {'seed': 99}},
                    request | {'config': request['config'] | {'max_output_tokens': 129}}):
                self.assertEqual(run.handle(changed)['error']['code'], 'hash_mismatch')

    def test_explicit_unavailable_captures_produce_no_eligible_jobs(self):
        captures = json_artifact(self.data['instructions']['inputs']['captures'])
        captures['records'] = [row | {'capture': None} for row in captures['records']]
        instructions = self.data['planner'].make_instructions(self.data['instructions']['inputs'] | {
            'captures': self.save('unavailable-captures', captures)}, self.data['root'] / 'unavailable-instructions')
        plan = final_patch_plan(self.data['binding'] | {'instructions': instructions['instructions']})
        self.assertEqual((plan['status'], plan['job_count']), ('unavailable', 0))
        self.assertEqual(plan['baseline_coverage']['planned_slots'], 16)


if __name__ == '__main__':
    unittest.main()
