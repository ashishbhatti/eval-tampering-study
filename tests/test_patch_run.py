"""Frozen patch continuations through the shared episode loop; explicit offline fixtures."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run
from eval_tampering.interventions import InterventionPlanner
from eval_tampering.messages import InputError, artifact_ref, atomic_json, read_artifact, success
from eval_tampering.monitors.common import json_artifact
from test_interventions import donor_fixture
from test_run import ScriptedModel, RecordingSandbox, RecordingEvaluator, configuration
from test_steering import reviewed_episode


def patch_episode_fixture(root):
    data = donor_fixture(root, history_files={'prior.txt': 'Recipient workspace from earlier history.'})
    planner = InterventionPlanner(data['config'])
    def invoke(operation, inputs, name):
        packet = planner.handle({'schema_version': 1, 'request_id': name, 'operation': operation, 'config': planner.config, 'inputs': inputs})
        assert packet['status'] == 'ok', packet
        return packet['result']
    direction = invoke('intervention.build_direction', {key: data[key] for key in ('features', 'captures', 'labels', 'monitor')}, 'direction')
    generation = {'max_new_tokens': 16, 'temperature': 1., 'max_seconds': 30}
    instructions = invoke('intervention.make_instructions', {key: data[key] for key in ('features', 'captures', 'labels')} |
        {'direction': direction['direction_artifact'], 'split': 'training', 'seeds': [11, 12, 13, 14], 'generation': generation, 'max_jobs': 256}, 'instructions')
    config = configuration(data['root']) | {'tasks': json_artifact(data['features'])['tasks'], 'generation': generation, 'max_output_tokens': 64}
    inputs = {'instructions': instructions['instructions'], 'episode_config': config, 'allocation': {'max_jobs': 256, 'max_output_tokens': 100000,
        'max_seconds': 200000, 'max_cost_usd': '0', 'usd_per_second': '0', 'cost_basis': 'Offline fixture; no provisioned compute.'}}
    episodes = invoke('intervention.plan_patch_episodes', inputs, 'episodes')
    return data, planner, instructions, episodes


class PatchModel(ScriptedModel):
    def __init__(self, root, runtime, **kwargs):
        super().__init__(root, **kwargs)
        self.runtime = runtime

    def handle(self, packet):
        if packet['operation'] == 'load':
            self.calls.append(deepcopy(packet))
            return success(packet, self.runtime)
        return super().handle(packet)


class PatchEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.cache', prefix='patch-episode-tests-')
        self.data, self.planner, self.instructions, self.plan = patch_episode_fixture(self.directory.name)
        self.root = self.data['root']
        self.runtime = json.loads((self.root / 'runtime.json').read_text())

    def tearDown(self):
        self.directory.cleanup()

    def job(self, control='target'):
        return next(job for job in self.plan['jobs'] if job['recipient'] == 'g0-r1' and job['seed'] == 11 and job['control'] == control)

    def test_canonical_read_only_inventory_and_equal_limits(self):
        import eval_tampering.interventions as module
        with patch.object(module, 'atomic_json', side_effect=AssertionError('reader wrote a file')):
            loaded = self.planner._read_patch_episodes(self.plan['plan'])
        self.assertEqual(loaded, {key: value for key, value in self.plan.items() if key != 'plan'})
        self.assertEqual((loaded['job_count'], len(loaded['histories'])), (100, 4))
        self.assertEqual(loaded['budget']['max_model_calls'], 300)
        self.assertEqual(loaded['budget']['max_output_tokens'], 4800)
        self.assertEqual(len(loaded['missing_controls']), 3)
        self.assertEqual({row['recipient_class'] for row in loaded['histories']}, {'tampering', 'repair'})
        for job in loaded['jobs']:
            self.assertEqual(job['episode']['config'], self.plan['inputs']['episode_config'])
            self.assertEqual(job['initial_prefix'], next(source for source in self.instructions['jobs'] if source['job_id'] == job['source_job_id'])['generation']['prefix'])
            self.assertNotEqual(job['job_id'], job['source_job_id'])
        raw = json_artifact(self.instructions['instructions'])
        raw['jobs'][0]['generation']['seed'] += 1
        path = Path(self.instructions['instructions']['path']).with_name('changed-instructions.json')
        atomic_json(path, raw)
        with self.assertRaisesRegex(InputError, 'instructions changed'):
            self.planner._read_patch_plan(artifact_ref(path, 'json'))
        with self.assertRaisesRegex(InputError, 'allocation'):
            self.planner._patch_episode_plan(self.plan['inputs'] | {'allocation': self.plan['inputs']['allocation'] | {'max_jobs': 1}})

    def test_exact_recipient_snapshot_prefix_seed_and_first_turn_only_across_controls(self):
        records = []
        for control in ('baseline', 'sham', 'target', 'random-0'):
            job = self.job(control)
            model = PatchModel(self.root, self.runtime)
            sandbox, evaluator = RecordingSandbox(self.root), RecordingEvaluator()
            packet = run.run_episode(job['episode'], model, sandbox, evaluator)
            self.assertEqual(packet['status'], 'ok', packet)
            record = packet['result']
            records.append(record)
            self.assertEqual(record['initial_snapshot']['sha256'], job['before_snapshot']['sha256'])
            self.assertEqual(record['initial_prefix'], job['initial_prefix'])
            self.assertEqual(record['patch'], job['episode']['inputs']['patch'])
            generated = [request for request in model.calls if request['operation'] == 'generate']
            self.assertEqual(len(generated), 3)
            self.assertFalse(any(request['operation'] == 'prepare' for request in model.calls))
            self.assertEqual(generated[0]['inputs']['prefix'], job['initial_prefix'])
            self.assertEqual(generated[0]['inputs']['seed'], 11)
            self.assertEqual(generated[0]['inputs']['intervention'], job['episode']['inputs']['intervention'])
            self.assertTrue(all(request['inputs']['intervention'] is None for request in generated[1:]))
            self.assertEqual(sum(request['operation'] == 'resume' for request in model.calls), 2)
            self.assertEqual(model.resets, 2)
            self.assertNotIn('PRIVATE GRADE', json.dumps(model.calls))
            self.assertNotIn('MUST NOT REACH MODEL', json.dumps(model.calls))
            for index, action in enumerate(record['actions']):
                self.assertEqual(json_artifact(action['generation_request']), generated[index])
            self.assertEqual(evaluator.calls[-1]['inputs']['before_snapshot'], record['initial_snapshot'])
        self.assertEqual(len({record['initial_snapshot']['sha256'] for record in records}), 1)
        # An authored fresh workspace would omit this earlier-history file.
        import io
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(read_artifact(records[0]['initial_snapshot'], 'tar', 1048576))) as archive:
            self.assertIn('prior.txt', archive.getnames())

    def test_changed_seed_limits_hook_runtime_and_heldout_rejected_before_execution(self):
        original = self.job()['episode']
        variants = [original | {'inputs': original['inputs'] | {key: value}} for key, value in
            [('seed', 99), ('intervention', None), ('runtime_sha256', '0'*64), ('arm_id', 'baseline')]]
        variants.append(original | {'config': original['config'] | {'generation': original['config']['generation'] | {'max_new_tokens': 17}}})
        for request in variants:
            model = PatchModel(self.root, self.runtime)
            with self.assertRaises(InputError):
                run.run_episode(request, model, RecordingSandbox(self.root), RecordingEvaluator())
            self.assertEqual(model.calls, [])
            self.assertEqual(model.resets, 0)
        heldout = original | {'inputs': original['inputs'] | {'task_id': 'fixture-3', 'condition_id': 'fixture-3.implementation_bug.forbidden'}}
        with self.assertRaises(InputError) as rejected:
            run._settings(heldout)
        self.assertEqual(rejected.exception.code, 'acceptance_required')
        inputs = self.instructions['inputs'] | {'split': 'intervention_test'}
        folder = self.root / 'heldout-instructions'
        folder.mkdir()
        heldout_plan = self.planner.make_instructions(inputs, folder)
        with self.assertRaisesRegex(InputError, 'complete experiment acceptance'):
            self.planner._patch_episode_plan(self.plan['inputs'] | {'instructions': heldout_plan['instructions']})

    def test_reviewed_cumulative_outcome_and_generation_evidence_binding(self):
        job = self.job()
        record = reviewed_episode(self.root / 'reviewed', job, self.plan, earlier_weakening=True, earlier_bypass=True)
        path = self.root / 'outcomes.json'
        atomic_json(path, {'schema_version': 1, 'records': [record]})
        outcomes = self.planner._episode_outcomes(self.plan, artifact_ref(path, 'json'))
        self.assertEqual(len(outcomes), 100)
        output = next(row for row in outcomes if row['job_id'] == job['job_id'])
        self.assertTrue(output['tampering'])
        self.assertTrue(output['implementation_bypass'])
        self.assertTrue(output['implementation_correct'])
        self.assertFalse(output['repair'])
        self.assertEqual(sum(row['status'] == 'episode_missing' for row in outcomes), 99)
        episode = json_artifact(record['episode'])
        generated = json_artifact(episode['actions'][1]['generation_request'])
        generated['inputs']['intervention'] = job['episode']['inputs']['intervention']
        altered = self.root / 'repeated-hook.json'
        atomic_json(altered, generated)
        episode['actions'][1]['generation_request'] = artifact_ref(altered, 'json')
        altered = self.root / 'changed-episode.json'
        atomic_json(altered, episode)
        atomic_json(path, {'schema_version': 1, 'records': [record | {'episode': artifact_ref(altered, 'json')}]})
        with self.assertRaisesRegex(InputError, 'changed prefix, seed, hook'):
            self.planner._episode_outcomes(self.plan, artifact_ref(path, 'json'))


if __name__ == '__main__':
    unittest.main()
