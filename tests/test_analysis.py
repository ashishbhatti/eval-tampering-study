"""Independent detection metrics and explicit fixture/artifact analysis checks."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eval_tampering.analysis import OUTCOMES, ResultAnalyzer, _negative_class, _weighted_metrics, detection_summary, permission_summary, steering_summary
from eval_tampering.messages import InputError, artifact_ref, atomic_json, read_artifact
from eval_tampering.monitors.activation_monitor import ActivationMonitor
from eval_tampering.monitors.text_monitor import TextMonitor
from test_interventions import donor_fixture
from test_text_monitor import mock_client, provider_config


def analysis_fixture(root, *, hosted=False):
    data = donor_fixture(root)
    root = data['root']
    analyzer = ResultAnalyzer({'artifact_root': str(root / 'analysis'), 'label_kind': 'fixture', 'bootstrap_seed': 19})
    frozen = analyzer.handle({'schema_version': 1, 'request_id': 'freeze', 'operation': 'analysis.freeze', 'config': analyzer.config, 'inputs': {}})
    assert frozen['status'] == 'ok', frozen
    raw = json.loads(read_artifact(data['monitor'], 'json', 16777216))
    monitor = ActivationMonitor(raw['config'])
    monitor.load(data['monitor'])
    scored = monitor.score({'features': data['features'], 'splits': ['detection_test']})
    atomic_json(root / 'activation-scores.json', {'schema_version': 1, 'status': 'ok', 'result': scored})
    config = {'artifact_root': str(root / 'text'), 'tokenizer': data['config']['tokenizer'], 'label_kind': 'fixture',
        'provider': provider_config() if hosted else None}
    client, requests = mock_client() if hosted else (None, [])
    try:
        text = TextMonitor(config, client)
        text.score({'features': data['features'], 'splits': ['validation'], 'calibration': None}, root / 'text-validation')
        calibrated = text.calibrate({'features': data['features'], 'labels': data['labels'],
            'scores': artifact_ref(root / 'text-validation' / 'scores.json', 'json')})
        atomic_json(root / 'text-calibration.json', calibrated)
        text.score({'features': data['features'], 'splits': ['detection_test'], 'calibration': artifact_ref(root / 'text-calibration.json', 'json')}, root / 'text-test')
    finally:
        if client is not None:
            client.close()
    return data, analyzer, {'features': data['features'], 'labels': data['labels'], 'split': 'detection_test',
        'activation': {'monitor': data['monitor'], 'scores': artifact_ref(root / 'activation-scores.json', 'json')},
        'text': {'calibration': artifact_ref(root / 'text-calibration.json', 'json'), 'scores': artifact_ref(root / 'text-test' / 'scores.json', 'json')},
        'rule': frozen['result']['rule']}, requests


class MetricTests(unittest.TestCase):
    def test_permission_pairs_and_negative_subclasses_match_hand_cases(self):
        slots = []
        for task, values in [('A', (False, True)), ('B', (None, False))]:
            for permission, value in zip(('forbidden', 'conditional'), values):
                slots.append({'task_id': task, 'clone_group_id': task, 'seed': 7, 'problem': 'test_conflict',
                    'permission': permission, 'job_id': task + permission,
                    **dict.fromkeys(('test_edits', 'valid_correction', 'tampering', 'conflict_report'), value)})
        result = permission_summary(slots, ['A', 'B'], 3)
        metric = result['metrics']['test_edits']
        self.assertEqual(metric['paired']['scorable_count'], 1)
        self.assertEqual(metric['paired']['difference']['value'], 1)
        self.assertEqual(metric['paired']['bounds'], {'lower': 0., 'upper': .5})
        self.assertEqual(len(result['per_task']), 2)
        self.assertEqual(result['pairs'][0]['forbidden_job_id'], 'Aforbidden')
        row = {'problem': 'implementation_bug'}
        for flags, expected in [({'authorization_violation': True, 'implementation_correct': True}, 'authorization_edge'),
                                ({'valid_test_repair': True}, 'legitimate_test_repair'),
                                ({'implementation_correct': True}, 'implementation_fix'),
                                ({'implementation_correct': False}, 'ordinary_error'), ({}, 'other_negative')]:
            self.assertEqual(_negative_class(row, {'label': 0, 'flags': flags}), expected)
            self.assertIsNone(_negative_class(row, {'label': None, 'flags': flags}))

    def test_paired_counts_bounds_and_grouped_bootstrap_match_hand_table(self):
        rows = []
        for i, (group, base, arm) in enumerate([('A', True, False), ('A', False, True), ('B', False, True), ('B', None, False)]):
            rows.append({'clone_group_id': group, 'problem': 'implementation_bug', 'permission': 'forbidden',
                'baseline': {'job_id': f'b{i}', **{name: base for name in OUTCOMES}},
                'arm': {'job_id': f'a{i}', **{name: arm for name in OUTCOMES}}})
        result = steering_summary(rows, ['A', 'B'], 7)
        metric = result['metrics']['repair']
        self.assertEqual((metric['baseline']['event_count'], metric['baseline']['known_count'], metric['baseline']['unknown_count']), (1, 3, 1))
        self.assertEqual(metric['baseline']['bounds'], {'lower': .25, 'upper': .5})
        self.assertEqual(metric['arm']['rate']['value'], .5)
        self.assertEqual(metric['paired']['scorable_count'], 3)
        self.assertEqual(metric['paired']['difference']['value'], 1/3)
        self.assertEqual(metric['paired']['retention']['value'], 2.)
        self.assertEqual(metric['paired']['bounds'], {'lower': 0., 'upper': .25})
        draws = np.random.default_rng(7).integers(0, 2, size=(2000, 2))
        effects, ratios = [], []
        for draw in draws:
            selected = [row for index in draw for row in rows if row['clone_group_id'] == ['A', 'B'][index]]
            paired = [row for row in selected if row['baseline']['repair'] is not None and row['arm']['repair'] is not None]
            base, arm = [sum(row[side]['repair'] for row in paired) for side in ('baseline', 'arm')]
            effects.append((arm-base)/len(paired))
            if base:
                ratios.append(arm/base)
        self.assertEqual(metric['paired']['difference']['bootstrap']['interval95'], np.quantile(effects, [.025, .975]).tolist())
        self.assertEqual(metric['paired']['retention']['bootstrap']['valid'], len(ratios))
        self.assertEqual(metric['paired']['retention']['bootstrap']['invalid'], 2000-len(ratios))
        self.assertEqual(metric['paired']['retention']['bootstrap']['interval95'], np.quantile(ratios, [.025, .975]).tolist())
        self.assertEqual(result['metrics']['conflict_handled']['applicable_count'], 0)
        self.assertIsNone(result['metrics']['conflict_handled']['paired']['difference']['value'])
        self.assertEqual(result['metrics']['conflict_handled']['paired']['difference']['bootstrap']['invalid'], 2000)
        for row in rows:
            row['baseline']['repair'] = False
        zero = steering_summary(rows, ['A', 'B'], 7)['metrics']['repair']
        self.assertIsNone(zero['paired']['retention']['value'])
        self.assertEqual(zero['paired']['retention']['bootstrap']['invalid'], 2000)
        empty = steering_summary([], [], 7)['metrics']['tampering']
        self.assertIsNone(empty['arm']['rate']['value'])
        self.assertEqual(empty['paired']['bounds'], {'lower': None, 'upper': None})

    def test_weighted_ties_match_independent_sklearn_and_hand_counts(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        rows = [{'label': label, 'score': score, 'positive': score >= .5, 'valid_test_repair': label == 0, 'clone_group_id': str(i//2)}
                for i, (label, score) in enumerate([(1, .8), (0, .8), (1, .4), (0, .1)])]
        weights = np.array([[1, 1, 1, 1], [2, 1, 3, 2]], dtype=np.float64)
        values, counts = _weighted_metrics(rows, weights)
        np.testing.assert_array_equal(counts['recall'][0], [1, 2])
        np.testing.assert_array_equal(counts['recall'][1], [2, 5])
        for i, weight in enumerate(weights):
            y, scores = [row['label'] for row in rows], [row['score'] for row in rows]
            self.assertAlmostEqual(values['average_precision'][i], average_precision_score(y, scores, sample_weight=weight))
            self.assertAlmostEqual(values['auroc'][i], roc_auc_score(y, scores, sample_weight=weight))
        self.assertAlmostEqual(values['average_precision'][0], 7/12)
        self.assertAlmostEqual(values['auroc'][0], 5/8)

    def test_bootstrap_keeps_groups_together_and_counts_invalid_draws(self):
        rows = [{'label': 1, 'score': score, 'positive': score >= .5, 'valid_test_repair': False, 'clone_group_id': 'A'} for score in (.9, .3)]
        rows.append({'label': 0, 'score': .2, 'positive': False, 'valid_test_repair': True, 'clone_group_id': 'B'})
        summary = detection_summary(rows, ['A', 'B'], 7)
        draws = np.random.default_rng(7).integers(0, 2, size=(2000, 2))
        expected = sum(len(set(draw)) == 2 for draw in draws)
        self.assertEqual(summary['metrics']['auroc']['bootstrap']['valid'], expected)
        self.assertEqual(summary['metrics']['auroc']['bootstrap']['invalid'], 2000-expected)
        self.assertEqual(summary['metrics']['recall']['bootstrap']['interval95'], [.5, .5])
        missing = rows + [{'label': 0, 'score': None, 'positive': None, 'valid_test_repair': True, 'clone_group_id': 'B'}]
        values, counts = _weighted_metrics(missing, np.ones((1, 4)))
        self.assertEqual(counts['false_positive_rate'][1][0], 1)
        self.assertEqual(values['false_positive_rate'][0], 0)
        for population in ([], [{'label': 0, 'score': 0., 'positive': False, 'valid_test_repair': True, 'clone_group_id': 'B'}]):
            result = detection_summary(population, ['B'], 7)
            self.assertIsNone(result['metrics']['auroc']['value'])
            self.assertIsNone(result['metrics']['average_precision']['value'])
            self.assertEqual(result['metrics']['auroc']['bootstrap']['invalid'], 2000)
            self.assertIsNone(result['metrics']['auroc']['bootstrap']['interval95'])


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.cache', prefix='analysis-tests-')
        self.data, self.analyzer, self.inputs, _ = analysis_fixture(self.directory.name)
        self.root = self.data['root']

    def tearDown(self):
        self.directory.cleanup()

    def invoke(self, inputs=None, name='summary'):
        return self.analyzer.handle({'schema_version': 1, 'request_id': name, 'operation': 'analysis.summarize',
            'config': self.analyzer.config, 'inputs': inputs or self.inputs})

    def test_frozen_scores_join_csv_metrics_and_no_fitting_or_provider_calls(self):
        with patch.object(ActivationMonitor, 'fit', side_effect=AssertionError('fitting')), patch.object(TextMonitor, '_call', side_effect=AssertionError('provider')):
            packet = self.invoke()
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        self.assertEqual(result['coverage']['declared_feature_records'], 8)
        self.assertEqual(result['coverage']['completed_calls'], 8)
        self.assertEqual(result['coverage']['reviewed_calls'], 8)
        self.assertEqual(result['coverage']['joint_activation_text_scores']['weakening_marker'], 8)
        pooled = {row['method']: row for row in result['summaries'] if row['scope'] == 'pooled'}
        self.assertEqual(pooled['activation']['metrics']['auroc']['value'], 1)
        self.assertEqual(pooled['weakening_marker']['metrics']['average_precision']['value'], .5)
        self.assertEqual(pooled['weakening_marker']['metrics']['auroc']['value'], .5)
        self.assertEqual(pooled['hosted']['score_count'], 0)
        common = next(row for row in result['paired_comparisons'] if row['baseline'] == 'weakening_marker')
        self.assertEqual(len(common['record_ids']), 8)
        self.assertEqual(common['methods']['activation']['joint_scorable_count'], common['methods']['weakening_marker']['joint_scorable_count'])
        self.assertIsNone(pooled['hosted']['metrics']['recall']['value'])
        with Path(result['table']['path']).open(newline='') as stream:
            flat = list(csv.DictReader(stream))
        self.assertEqual(len(flat), 125)
        self.assertEqual(result['monitor_fits'], 0)
        rows = json.loads(read_artifact(result['rows'], 'json', 16777216))['rows']
        self.assertEqual({row['record_id'] for row in rows}, {f'g2-r{i}' for i in range(8)})
        self.assertTrue(all(row['review'] is not None for row in rows))

    def test_changed_score_rule_and_view_are_rejected_and_partial_text_is_missing(self):
        activation = json.loads(read_artifact(self.inputs['activation']['scores'], 'json', 16777216))
        activation['result']['scores'][0]['score'] = .99
        path = self.root / 'changed.json'
        atomic_json(path, activation)
        result = self.invoke(self.inputs | {'activation': self.inputs['activation'] | {'scores': artifact_ref(path, 'json')}}, 'changed-activation')
        self.assertEqual(result['status'], 'error')
        self.assertIn('disagree', result['error']['message'])
        result = self.invoke(self.inputs | {'rule': None}, 'missing-rule')
        self.assertEqual(result['error']['code'], 'rule_required')
        scored = json.loads(read_artifact(self.inputs['text']['scores'], 'json', 16777216))
        scored['scores'][0]['view'] = scored['scores'][1]['view']
        atomic_json(path, scored)
        result = self.invoke(self.inputs | {'text': self.inputs['text'] | {'scores': artifact_ref(path, 'json')}}, 'changed-view')
        self.assertEqual(result['status'], 'error')
        self.assertIn('prefix/action', result['error']['message'])
        scored = json.loads(read_artifact(self.inputs['text']['scores'], 'json', 16777216))
        scored.update(status='incomplete', scores=scored['scores'][:3])
        atomic_json(path, scored)
        result = self.invoke(self.inputs | {'text': self.inputs['text'] | {'scores': artifact_ref(path, 'json')}}, 'partial-text')['result']
        pooled = next(row for row in result['summaries'] if row['scope'] == 'pooled' and row['method'] == 'weakening_marker')
        self.assertEqual(pooled['record_count'], 8)
        self.assertEqual(pooled['score_count'], 3)
        common = next(row for row in result['paired_comparisons'] if row['baseline'] == 'weakening_marker')
        self.assertEqual(len(common['record_ids']), 3)
        scored['status'] = 'scored'
        atomic_json(path, scored)
        result = self.invoke(self.inputs | {'text': self.inputs['text'] | {'scores': artifact_ref(path, 'json')}}, 'false-complete')
        self.assertEqual(result['status'], 'error')
        self.assertIn('omitted eligible', result['error']['message'])
        calibration = json.loads(read_artifact(self.inputs['text']['calibration'], 'json', 16777216))
        calibration['tasks_sha256'] = '0' * 64
        atomic_json(self.root / 'wrong-task-calibration.json', calibration)
        reference = artifact_ref(self.root / 'wrong-task-calibration.json', 'json')
        scored = json.loads(read_artifact(self.inputs['text']['scores'], 'json', 16777216))
        scored['tasks_sha256'] = calibration['tasks_sha256']
        scored['inputs']['calibration'] = reference
        atomic_json(path, scored)
        result = self.invoke(self.inputs | {'activation': None, 'text': {'calibration': reference, 'scores': artifact_ref(path, 'json')}}, 'wrong-task-cohort')
        self.assertEqual(result['status'], 'error')
        self.assertIn('analyzed cohort', result['error']['message'])

    def test_cross_host_score_rounding_preserves_saved_values_and_rejects_corruption(self):
        original = json.loads(read_artifact(self.inputs['activation']['scores'], 'json', 16777216))
        saved = json.loads(json.dumps(original))
        saved['result']['scores'][0]['score'] += 1e-13
        path = self.root / 'rounded-scores.json'
        atomic_json(path, saved)
        inputs = self.inputs | {'activation': self.inputs['activation'] | {'scores': artifact_ref(path, 'json')}}
        packet = self.invoke(inputs, 'rounded')
        self.assertEqual(packet['status'], 'ok', packet)
        rows = json.loads(read_artifact(packet['result']['rows'], 'json', 16777216))['rows']
        row = next(row for row in rows if row['record_id'] == saved['result']['scores'][0]['record_id'])
        self.assertEqual(row['scores']['activation']['score'], saved['result']['scores'][0]['score'])
        for key, value in [('score', .123456789), ('positive', not original['result']['scores'][0]['positive']), ('task_id', 'changed')]:
            with self.subTest(key=key):
                changed = json.loads(json.dumps(original))
                changed['result']['scores'][0][key] = value
                atomic_json(path, changed)
                inputs['activation']['scores'] = artifact_ref(path, 'json')
                result = self.invoke(inputs, 'corrupt-' + key)
                self.assertEqual(result['status'], 'error', result)
                self.assertEqual(result['error']['code'], 'hash_mismatch')

    def test_uncertain_label_is_missing_and_cli_matches_object(self):
        labels = json.loads(read_artifact(self.inputs['labels'], 'json', 16777216))
        row = next(row for row in labels['records'] if row['record_id'] == 'g2-r0')
        review = json.loads(read_artifact(row['review'], 'json', 16777216))
        review['uncertain'] = True
        atomic_json(self.root / 'uncertain.json', review)
        row['review'] = artifact_ref(self.root / 'uncertain.json', 'json')
        atomic_json(self.root / 'labels-uncertain.json', labels)
        result = self.invoke(self.inputs | {'labels': artifact_ref(self.root / 'labels-uncertain.json', 'json')})
        self.assertEqual(result['status'], 'ok', result)
        self.assertEqual(result['result']['coverage']['reviewed_calls'], 7)
        sensitivity = [row for row in result['result']['uncertain_label_sensitivity'] if row['scope'] == 'pooled' and row['method'] == 'activation']
        self.assertEqual([row['reviewed_count'] for row in sensitivity], [8, 8])
        self.assertEqual(sensitivity[1]['positives'] - sensitivity[0]['positives'], 1)
        self.assertEqual(sum(row['record_count'] for row in result['result']['negative_subclasses'] if row['method'] == 'activation'), 3)
        self.assertTrue(any(row['record_id'] == 'g2-r0' for row in result['result']['label_exclusions']))
        request = {'schema_version': 1, 'request_id': 'cli', 'operation': 'analysis.summarize', 'config': self.analyzer.config,
            'inputs': self.inputs | {'labels': artifact_ref(self.root / 'labels-uncertain.json', 'json')}}
        atomic_json(self.root / 'request.json', request)
        subprocess.run([sys.executable, '-B', '-m', 'eval_tampering', 'analysis', '--input', str(self.root / 'request.json'),
            '--output', str(self.root / 'response.json')], check=True)
        self.assertEqual(json.loads((self.root / 'response.json').read_text())['result']['summaries'], result['result']['summaries'])
        subprocess.run([sys.executable, '-B', '-c', 'import sys; import eval_tampering.analysis; assert not ({"numpy", "torch", "sklearn", "openai"} & sys.modules.keys())'], check=True)

    def test_saved_hosted_request_response_are_verified_without_network(self):
        data, analyzer, inputs, requests = analysis_fixture(self.root / 'hosted', hosted=True)
        self.assertEqual(len(requests), 32)
        with patch.object(TextMonitor, '_call', side_effect=AssertionError('network')):
            packet = analyzer.handle({'schema_version': 1, 'request_id': 'offline-check', 'operation': 'analysis.summarize', 'config': analyzer.config, 'inputs': inputs})
        self.assertEqual(packet['status'], 'ok', packet)
        pooled = next(row for row in packet['result']['summaries'] if row['scope'] == 'pooled' and row['method'] == 'hosted')
        self.assertEqual(pooled['score_count'], 8)
        scored = json.loads(read_artifact(inputs['text']['scores'], 'json', 16777216))
        scored['scores'][0]['provider']['request'] = scored['scores'][1]['provider']['request']
        atomic_json(data['root'] / 'wrong-hosted.json', scored)
        result = analyzer.handle({'schema_version': 1, 'request_id': 'wrong-hosted', 'operation': 'analysis.summarize', 'config': analyzer.config,
            'inputs': inputs | {'text': inputs['text'] | {'scores': artifact_ref(data['root'] / 'wrong-hosted.json', 'json')}}})
        self.assertEqual(result['status'], 'error')
        self.assertIn('another view', result['error']['message'])


def steering_analysis_fixture(root):
    """Two-repeat development inventory with scripted reviews and explicit gaps."""
    from eval_tampering.interventions import InterventionPlanner, INITIAL_GRID
    from test_run import configuration
    from test_steering import reviewed_episode
    data = donor_fixture(root)
    root = data['root']
    planner = InterventionPlanner(data['config'])
    def invoke(operation, inputs, name):
        packet = planner.handle({'schema_version': 1, 'request_id': name, 'operation': operation, 'config': planner.config, 'inputs': inputs})
        assert packet['status'] == 'ok', packet
        return packet['result']
    direction = invoke('intervention.build_direction', {key: data[key] for key in ('features', 'captures', 'labels', 'monitor')}, 'direction')
    config = configuration(root) | {'tasks': json.loads((root / 'features.json').read_text())['tasks'],
        'max_output_tokens': 64, 'generation': {'max_new_tokens': 16, 'temperature': 1., 'max_seconds': 30}}
    plan = invoke('intervention.plan_steering', {'direction': direction['direction_artifact'], 'episode_config': config,
        'task_ids': ['fixture-0'], 'seeds': [11, 12], 'stage': 'calibration', 'coefficients': INITIAL_GRID.copy(), 'previous': None,
        'rationale': 'Scripted analysis fixture; no episodes execute.', 'allocation': {'max_jobs': 512, 'max_output_tokens': 100000,
            'max_seconds': 300000, 'max_cost_usd': '0', 'usd_per_second': '0', 'cost_basis': 'Offline fixture; no billed compute.'}}, 'plan')
    supplied = []
    for job in plan['jobs']:
        arm = job['episode']['inputs']['arm_id']
        cell = job['problem'], job['permission']
        if arm in ('baseline', 'S1-target--0.5'):
            if arm != 'baseline' and cell == ('implementation_bug', 'forbidden') and job['seed'] == 12:
                continue
            earlier = arm == 'baseline' and cell == ('implementation_bug', 'forbidden') and job['seed'] == 11
            correct = not (job['seed'] == 12 and (cell == ('test_conflict', 'conditional') if arm == 'baseline' else cell == ('implementation_bug', 'conditional')))
            supplied.append(reviewed_episode(root / 'episodes', job, plan, correct=correct, earlier_weakening=earlier, earlier_bypass=earlier))
        elif cell == ('implementation_bug', 'forbidden') and job['seed'] == 11 and arm in ('prompt-reminder', 'S2-target--0.5'):
            evidence = reviewed_episode(root / 'episodes', job, plan)
            if arm == 'prompt-reminder':
                episode = json.loads(read_artifact(evidence['episode'], 'json', 16777216))
                episode.update(status='incomplete', runtime_sha256=None, stop_reason='runtime_unavailable', actions=[],
                    initial_snapshot=None, final_snapshot=None, final_assessment=None)
                atomic_json(Path(evidence['episode']['path']), episode)
                evidence['episode'] = artifact_ref(evidence['episode']['path'], 'json')
            evidence['reviews'] = None
            supplied.append(evidence)
    atomic_json(root / 'outcomes.json', {'schema_version': 1, 'records': supplied})
    analyzer = ResultAnalyzer({'artifact_root': str(root / 'analysis'), 'label_kind': 'fixture', 'bootstrap_seed': 19})
    frozen = analyzer.handle({'schema_version': 1, 'request_id': 'freeze', 'operation': 'analysis.freeze', 'config': analyzer.config, 'inputs': {}})
    assert frozen['status'] == 'ok', frozen
    return data, analyzer, plan, {'plan': plan['plan'], 'outcomes': artifact_ref(root / 'outcomes.json', 'json'), 'rule': frozen['result']['rule']}


class SteeringAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.cache', prefix='steering-analysis-tests-')
        self.data, self.analyzer, self.plan, self.inputs = steering_analysis_fixture(self.directory.name)
        self.root = self.data['root']

    def tearDown(self):
        self.directory.cleanup()

    def invoke(self, inputs=None, name='summary'):
        return self.analyzer.handle({'schema_version': 1, 'request_id': name, 'operation': 'analysis.steering',
            'config': self.analyzer.config, 'inputs': self.inputs if inputs is None else inputs})

    def test_all_slots_pairing_earlier_bypass_missingness_csv_and_cli(self):
        import run
        from eval_tampering.model import ModelRuntime
        from eval_tampering.monitors import provider
        from sklearn.linear_model import LogisticRegression
        with patch.object(run, 'run_episode', side_effect=AssertionError('episode execution')), \
             patch.object(ModelRuntime, 'load', side_effect=AssertionError('model load')), \
             patch.object(provider, 'call_json', side_effect=AssertionError('provider request')), \
             patch.object(LogisticRegression, 'fit', side_effect=AssertionError('fit')):
            packet = self.invoke()
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        self.assertEqual(result['planned_episodes'], 272)
        self.assertEqual(sum(row['planned'] for row in result['coverage']), 272)
        self.assertEqual(sum(row['supplied_episodes'] for row in result['coverage']), 17)
        self.assertEqual(len(result['summaries']), 170)
        self.assertEqual((result['new_model_calls'], result['new_provider_calls'], result['monitor_fits']), (0, 0, 0))
        pooled = next(row for row in result['summaries'] if row['arm_id'] == 'S1-target--0.5' and row['scope'] == 'pooled')
        tampering, repair = pooled['metrics']['tampering'], pooled['metrics']['repair']
        self.assertEqual(tampering['baseline']['rate']['value'], 1/8)
        self.assertEqual(tampering['arm']['known_count'], 7)
        self.assertEqual(tampering['paired']['difference']['value'], -1/7)
        self.assertEqual(tampering['paired']['bounds'], {'lower': -1/8, 'upper': 0.})
        self.assertEqual((repair['applicable_count'], repair['paired']['scorable_count']), (6, 5))
        self.assertEqual(repair['paired']['difference']['value'], .2)
        self.assertEqual(repair['paired']['retention']['value'], 4/3)
        rows = json.loads(read_artifact(result['outcomes'], 'json', 16777216))['rows']
        earlier = next(row for row in rows if row['arm_id'] == 'baseline' and (row['problem'], row['permission']) == ('implementation_bug', 'forbidden') and row['seed'] == 11)
        self.assertTrue(earlier['implementation_bypass'])
        self.assertTrue(earlier['tampering'])
        self.assertTrue(earlier['implementation_correct'])
        self.assertFalse(earlier['repair'])
        self.assertEqual(sum(row['status'] == 'runtime_unavailable' for row in rows), 1)
        unknown = next(row for row in rows if row['status'] == 'runtime_unavailable')
        self.assertTrue(all(unknown[name] is None for name in OUTCOMES))
        self.assertIn('evidence', unknown)
        partial = next(row for row in rows if row['status'] == 'partial')
        self.assertIsNone(partial['tampering'])
        self.assertTrue(partial['missing_review_ids'])
        self.assertEqual(len(rows), 272)
        identities = {row['job_id']: row for row in rows}
        pairs = json.loads(read_artifact(result['pairs'], 'json', 16777216))['rows']
        for pair in pairs:
            left, right = identities[pair['baseline_job_id']], identities[pair['arm_job_id']]
            self.assertEqual((left['history_id'], left['seed']), (right['history_id'], right['seed']))
            self.assertEqual(left['arm_id'], 'baseline')
        task_rows = json.loads(read_artifact(result['task_counts'], 'json', 16777216))['rows']
        self.assertEqual(len(task_rows), 136)
        table = list(csv.DictReader(read_artifact(result['table'], 'csv', 16777216).decode().splitlines()))
        self.assertEqual(len(table), 3230)
        csv_effect = next(row for row in table if row['scope'] == 'pooled' and row['arm_id'] == 'S1-target--0.5' and row['outcome'] == 'tampering' and row['estimate'] == 'difference')
        self.assertEqual((int(csv_effect['numerator']), int(csv_effect['denominator']), int(csv_effect['unknown_count'])), (-1, 7, 1))
        self.assertEqual(float(csv_effect['value']), -1/7)
        request = {'schema_version': 1, 'request_id': 'cli', 'operation': 'analysis.steering', 'config': self.analyzer.config, 'inputs': self.inputs}
        atomic_json(self.root / 'cli.json', request)
        completed = subprocess.run([sys.executable, '-B', '-m', 'eval_tampering', 'analysis', '--input', str(self.root / 'cli.json'), '--output', str(self.root / 'cli-result.json')], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        cli = json.loads((self.root / 'cli-result.json').read_text())['result']
        self.assertEqual(cli['summaries'], result['summaries'])
        self.assertEqual(cli['coverage'], result['coverage'])

    def test_changed_plan_review_and_rule_rejected_unknown_bypass_preserved(self):
        def ref(name, data):
            atomic_json(self.root / name, data)
            return artifact_ref(self.root / name, 'json')
        raw = json.loads(read_artifact(self.inputs['plan'], 'json', 67108864))
        raw['jobs'][0]['seed'] = 99
        packet = self.invoke(self.inputs | {'plan': ref('wrong-plan.json', raw)}, 'wrong-plan')
        self.assertEqual(packet['status'], 'error')
        self.assertEqual(packet['error']['code'], 'hash_mismatch')
        raw['split'] = 'intervention_test'
        packet = self.invoke(self.inputs | {'plan': ref('heldout.json', raw)}, 'heldout')
        self.assertEqual(packet['error']['code'], 'split_leakage')
        rule = self.analyzer.rule() | {'intervention_comparison': 'unmatched'}
        packet = self.invoke(self.inputs | {'rule': ref('wrong-rule.json', rule)}, 'wrong-rule')
        self.assertEqual(packet['error']['code'], 'hash_mismatch')
        manifest = json.loads(read_artifact(self.inputs['outcomes'], 'json', 16777216))
        first = manifest['records'][0]
        reviews = json.loads(read_artifact(first['reviews'], 'json', 16777216))
        # Make a coherent primary-negative review claim a bypass; the reader must reject it.
        final = json.loads(read_artifact(reviews['records'][-1]['review'], 'json', 16777216))
        final['flags']['implementation_bypass'] = True
        reviews['records'][-1]['review'] = ref('wrong-bypass.json', final)
        first['reviews'] = ref('wrong-reviews.json', reviews)
        packet = self.invoke(self.inputs | {'outcomes': ref('wrong-outcomes.json', manifest)}, 'wrong-bypass')
        self.assertEqual(packet['status'], 'error')
        self.assertIn('bypass implies weakening', packet['error']['message'])
        # Missing auxiliary bypass flags remain unknown despite a known negative primary label.
        manifest = json.loads(read_artifact(self.inputs['outcomes'], 'json', 16777216))
        row = manifest['records'][1]
        reviews = json.loads(read_artifact(row['reviews'], 'json', 16777216))
        for i, review in enumerate(reviews['records']):
            saved = json.loads(read_artifact(review['review'], 'json', 16777216))
            saved['flags']['implementation_bypass'] = None
            review['review'] = ref(f'unknown-bypass-{i}.json', saved)
        row['reviews'] = ref('unknown-reviews.json', reviews)
        from eval_tampering.interventions import InterventionPlanner
        planner = InterventionPlanner(self.data['config'])
        checked = planner._episode_outcomes(self.plan, ref('unknown-outcomes.json', manifest))
        output = next(item for item in checked if item['job_id'] == row['job_id'])
        self.assertFalse(output['tampering'])
        self.assertIsNone(output['implementation_bypass'])


if __name__ == '__main__':
    unittest.main()
