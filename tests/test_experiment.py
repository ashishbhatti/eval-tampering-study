"""A frozen protocol fixture is not permission to run a research experiment."""

from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run
from eval_tampering.analysis import ResultAnalyzer
from eval_tampering.experiment import build, read_manifest
from eval_tampering.interventions import InterventionPlanner, INITIAL_GRID
from eval_tampering.messages import InputError, artifact_ref, atomic_json, read_artifact
from eval_tampering.monitors.activation_monitor import ActivationMonitor
from eval_tampering.monitors.text_monitor import TextMonitor
from test_interventions import donor_fixture
from test_run import configuration
from test_steering import reviewed_episode, scripted_diagnostic


def manifest_fixture(root, *, runtime=None, runtime_check=None, hosted=False, max_new_tokens=16):
    data = donor_fixture(root, runtime=runtime)
    root = data['root']
    planner = InterventionPlanner(data['config'])
    def call(operation, inputs, name):
        packet = planner.handle({'schema_version': 1, 'request_id': name, 'operation': operation,
                                 'config': planner.config, 'inputs': inputs})
        assert packet['status'] == 'ok', packet
        return packet['result']
    direction = call('intervention.build_direction', {key: data[key] for key in ('features', 'captures', 'labels', 'monitor')}, 'direction')
    episode = configuration(root) | {'tasks': artifact_ref(root / 'tasks.json', 'json'), 'max_output_tokens': max(64, max_new_tokens),
                                    'generation': {'max_new_tokens': max_new_tokens, 'temperature': 1., 'max_seconds': 30}}
    allocation = {'max_jobs': 512, 'max_output_tokens': 100000, 'max_seconds': 300000, 'max_cost_usd': '0',
                  'usd_per_second': '0', 'cost_basis': 'Numerical protocol fixture only; no research jobs or billing.'}
    initial = {'direction': direction['direction_artifact'], 'episode_config': episode, 'task_ids': ['fixture-0'], 'seeds': [11],
        'stage': 'calibration', 'coefficients': INITIAL_GRID.copy(), 'previous': None, 'rationale': 'Offline manifest fixture.', 'allocation': allocation}
    calibration = call('intervention.plan_steering', initial, 'calibration')
    numerical = call('intervention.plan_calibration', {'plan': calibration['plan'], 'record_ids': ['g0-r0', 'g0-r4'], 'max_seconds': 30,
        'allocation': {key: value for key, value in allocation.items() if key != 'max_output_tokens'} | {'max_input_tokens': 1000000}}, 'numerical')
    runtime = json.loads((root / 'runtime.json').read_text())
    diagnostics = [scripted_diagnostic(root / 'scripted-diagnostics', job, runtime, planner._tokenizer) for job in numerical['jobs']]
    outcomes = [reviewed_episode(root / 'scripted-calibration', job, calibration) for job in calibration['jobs']]
    atomic_json(root / 'diagnostics.json', {'schema_version': 1, 'records': diagnostics})
    atomic_json(root / 'outcomes.json', {'schema_version': 1, 'records': outcomes})
    checked = call('intervention.check_calibration', {'plan': numerical['plan'], 'diagnostics': artifact_ref(root / 'diagnostics.json', 'json'),
        'outcomes': artifact_ref(root / 'outcomes.json', 'json')}, 'calibration-check')
    validation = call('intervention.plan_steering', initial | {'stage': 'validation', 'task_ids': ['fixture-1'],
        'previous': calibration['plan'], 'calibration': checked['calibration']}, 'validation')
    atomic_json(root / 'empty-validation.json', {'schema_version': 1, 'records': []})
    selection = call('intervention.select_strength', {'plan': validation['plan'], 'outcomes': artifact_ref(root / 'empty-validation.json', 'json')}, 'selection')
    policy = call('intervention.freeze_policy', {'selection': selection['selection'], 'task_ids': ['fixture-3'], 'seeds': [21, 22, 23, 24],
        'allocation': allocation, 'rationale': 'Predeclared exploratory fixture; no validation outcomes or final executions.'}, 'policy')
    from test_text_monitor import mock_client, provider_config
    text_config = {'artifact_root': str(root / 'text'), 'tokenizer': data['config']['tokenizer'], 'label_kind': 'fixture',
                   'provider': provider_config() if hosted else None}
    client, _ = mock_client() if hosted else (None, [])
    text = TextMonitor(text_config, client)
    try:
        text.score({'features': data['features'], 'splits': ['validation'], 'calibration': None}, root / 'text-validation')
    finally:
        if client is not None:
            client.close()
    calibrated = text.calibrate({'features': data['features'], 'labels': data['labels'], 'scores': artifact_ref(root / 'text-validation/scores.json', 'json')})
    atomic_json(root / 'text-calibration.json', calibrated)
    analyzer = ResultAnalyzer({'artifact_root': str(root / 'analysis'), 'label_kind': 'fixture', 'bootstrap_seed': 19})
    atomic_json(root / 'analysis-rule.json', analyzer.rule())
    request = {'schema_version': 1, 'request_id': 'manifest', 'operation': 'experiment.freeze',
        'config': {'artifact_root': str(root / 'manifests'), 'label_kind': 'fixture'}, 'inputs': {
        'episode_config': episode, 'runtime': artifact_ref(root / 'runtime.json', 'json'), 'runtime_check': runtime_check,
        'activation': data['monitor'], 'text': artifact_ref(root / 'text-calibration.json', 'json'),
        'reasoning': text_config | {'artifact_root': str(root / 'reasoning'), 'audit_seed': 17, 'audit_size': 50},
        'interventions': {'config': planner.config, 'direction': direction['direction_artifact'], 'steering_policy': policy['policy']},
        'analysis': artifact_ref(root / 'analysis-rule.json', 'json'), 'sampling': {'seeds': [31, 32, 33, 34], 'allocation': allocation},
        'patch': {'seeds': [41, 42, 43, 44], 'allocation': allocation}, 'audits': {'behavior_seed': 13, 'behavior_uniform_size': 50},
        'readiness': dict.fromkeys(('pilot', 'sandbox', 'evaluator', 'behavior_audit', 'reasoning_audit'))}}
    return data, request


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(dir='.cache', prefix='manifest-tests-')
        cls.data, cls.request = manifest_fixture(cls.work.name)
        cls.root = cls.data['root']

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def test_freeze_reload_all_slots_without_heldout_reads_execution_or_fit(self):
        from eval_tampering.model import ModelRuntime
        from eval_tampering.monitors import common
        original = common.read_artifact
        def development_only(ref, *args):
            self.assertFalse(any('/g' + str(group) + '-' in ref['path'] for group in (2, 3)), ref['path'])
            return original(ref, *args)
        with ExitStack() as cleanup, patch.object(common, 'read_artifact', side_effect=development_only), \
             patch.object(ModelRuntime, 'load', side_effect=AssertionError('model loading')), \
             patch.object(ActivationMonitor, 'fit', side_effect=AssertionError('refitting')), \
             patch.object(TextMonitor, 'score', side_effect=AssertionError('provider scoring')):
            for row in self.data['rows'].values():
                if row['split'] not in ('training', 'validation'):
                    folder = row['folder']
                    sealed = folder.with_name('sealed-' + folder.name)
                    folder.rename(sealed)
                    cleanup.callback(sealed.rename, folder)
            packet = run.handle(self.request)
            self.assertEqual(packet['status'], 'ok', packet)
            manifest = read_manifest(packet['result']['manifest'])
        self.assertEqual(manifest['sampling']['job_count'], 32)
        self.assertEqual(manifest['components']['steering']['job_count'], 160)
        self.assertEqual(len(manifest['components']['steering']['variants']), 10)
        self.assertEqual(set(manifest['split_counts'].values()), {1})
        self.assertEqual(manifest['status'], 'incomplete')
        self.assertFalse(manifest['execution_enabled'])
        self.assertIn('runtime_check_missing', manifest['pending'])
        self.assertIn('fixture_is_not_research_acceptance', manifest['pending'])
        self.assertEqual(manifest['monitor_fits'], 0)
        jobs = manifest['sampling']['jobs']
        self.assertEqual({job['task_id'] for job in jobs}, {'fixture-2', 'fixture-3'})
        self.assertEqual({job['seed'] for job in jobs}, {31, 32, 33, 34})
        self.assertTrue(all(job['episode']['config']['max_tool_calls'] == 1 for job in jobs))
        for job in (jobs[0], manifest['components']['steering']['jobs'][0]):
            self.assertEqual(run.handle(job['episode'])['status'], 'error')
        self.assertEqual(run.handle(self.request)['error']['code'], 'attempt_exists')
        source = self.root / 'load.request.json'
        atomic_json(source, self.request | {'operation': 'experiment.load', 'inputs': {'manifest': packet['result']['manifest']}})
        result = subprocess.run([sys.executable, '-B', 'run.py', '--input', str(source), '--output', str(self.root / 'load.result.json')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads((self.root / 'load.result.json').read_text())['result']['execution_enabled'])

    def test_changed_runtime_seed_budget_and_threshold_are_rejected(self):
        changed = deepcopy(self.request['inputs'])
        changed['sampling']['seeds'][0] = 21
        with self.assertRaisesRegex(InputError, 'distinct final seeds'):
            build(self.request['config'], changed)
        changed = deepcopy(self.request['inputs'])
        changed['sampling']['allocation']['max_jobs'] = 1
        with self.assertRaisesRegex(InputError, 'allocation'):
            build(self.request['config'], changed)
        raw = json.loads(read_artifact(self.request['inputs']['runtime'], 'json', 16777216))
        atomic_json(self.root / 'wrong-runtime.json', raw | {'fixture_seed': 999})
        with self.assertRaisesRegex(InputError, 'another runtime'):
            build(self.request['config'], self.request['inputs'] | {'runtime': artifact_ref(self.root / 'wrong-runtime.json', 'json')})
        frozen = json.loads(read_artifact(self.request['inputs']['activation'], 'json', 16777216))
        frozen['report']['threshold']['threshold'] = .987
        atomic_json(self.root / 'wrong-threshold.json', frozen)
        with self.assertRaisesRegex(InputError, 'threshold'):
            build(self.request['config'], self.request['inputs'] | {'activation': artifact_ref(self.root / 'wrong-threshold.json', 'json')})

    def test_manifest_and_control_inventory_mutations_cannot_reopen_final_jobs(self):
        packet = run.handle(self.request | {'request_id': 'mutation-base'})
        self.assertEqual(packet['status'], 'ok', packet)
        data = json.loads(read_artifact(packet['result']['manifest'], 'json', 16777216))
        for key, value in [('execution_enabled', True), ('sampling', data['sampling'] | {'jobs': data['sampling']['jobs'][:-1]})]:
            atomic_json(self.root / 'changed-manifest.json', data | {key: value})
            with self.assertRaisesRegex(InputError, 'manifest changed'):
                read_manifest(artifact_ref(self.root / 'changed-manifest.json', 'json'))
        changed = deepcopy(self.request['inputs'])
        policy = json.loads(read_artifact(changed['interventions']['steering_policy'], 'json', 16777216))
        policy['variants'].pop()
        atomic_json(self.root / 'missing-control.json', policy)
        changed['interventions']['steering_policy'] = artifact_ref(self.root / 'missing-control.json', 'json')
        with self.assertRaisesRegex(InputError, 'inventory'):
            build(self.request['config'], changed)

    def test_missing_monitors_and_direction_stay_explicit_and_import_is_lazy(self):
        subprocess.run([sys.executable, '-B', '-c', 'import sys; import eval_tampering.experiment; assert not ({"torch", "sklearn", "numpy"} & sys.modules.keys())'], check=True)
        inputs = self.request['inputs'] | {'activation': None, 'text': None,
            'interventions': self.request['inputs']['interventions'] | {'direction': None, 'steering_policy': None}}
        report = build(self.request['config'], inputs)
        self.assertEqual(report['sampling']['job_count'], 32)
        self.assertTrue({'activation_monitor_unavailable', 'text_calibration_unavailable', 'training_direction_missing', 'steering_policy_missing'} <= set(report['pending']))
        self.assertFalse(report['execution_enabled'])


if __name__ == '__main__':
    unittest.main()
