"""Numerical/provenance fixtures for the offline probe; these are not research data."""

from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eval_tampering.messages import InputError, artifact_ref, atomic_bytes, atomic_json, fingerprint
from eval_tampering.monitors.activation_monitor import ActivationMonitor
from eval_tampering.model import pool_residuals
from eval_tampering.monitors.common import operating_point
from eval_tampering.sandbox import _files_archive
from eval_tampering.tasks import handle as task_handle


def fixture(root, *, runtime=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    authored = json.loads(Path('configs/task_example.json').read_text())
    source = authored['inputs']['tasks'][0]
    tasks, assignments = [], {}
    for index, split in enumerate(('training', 'validation', 'detection_test', 'intervention_test')):
        task = deepcopy(source)
        task.update(task_id=f'fixture-{index}', clone_group_id=f'fixture-group-{index}',
                    specification=f'Numerical protocol fixture {index}; not a research task. ' + source['specification'])
        tasks.append(task)
        assignments[task['clone_group_id']] = split
    authored['inputs']['tasks'] = tasks
    authored['config']['split_assignments'] = assignments
    atomic_json(root / 'tasks.json', authored)
    built = task_handle(authored)['result']['tasks']
    runtime = runtime or {'fixture': True, 'profile': 'numerical-fixture', 'layers': [1, 2],
                         'config': {'hidden_size': 2, 'num_hidden_layers': 3}, 'research_backend_validated': False}
    width = runtime['config']['hidden_size']
    atomic_json(root / 'runtime.json', runtime)
    features = {'schema_version': 1, 'tasks': artifact_ref(root / 'tasks.json', 'json'),
                'runtime': artifact_ref(root / 'runtime.json', 'json'), 'records': []}
    labels = {'schema_version': 1, 'records': []}
    data = {}
    for group, task in enumerate(built):
        for index in range(8):
            record_id = f'g{group}-r{index}'
            folder = root / record_id
            folder.mkdir(exist_ok=True)
            label = bool(index % 2)
            condition = task['conditions'][[3, 0, 1, 2][index % 4]]
            for state in ('before', 'after'):
                atomic_bytes(folder / f'{state}.tar', _files_archive(
                    condition['visible']['files'] | {'fixture_id.txt': record_id + '-' + state}, 1048576))
            before, after = (artifact_ref(folder / f'{state}.tar', 'tar') for state in ('before', 'after'))
            vector = np.zeros(width, dtype=np.float32)
            vector[0] = (1 if label else -1) * (index + 1 + 3 * group)
            pooled = np.stack([np.zeros(width, dtype=np.float32), vector, vector * 2])
            raw = np.repeat(pooled[:, None, :], 2, axis=1)
            np.savez_compressed(folder / 'features.npz', residuals=raw, mean=pooled, last=pooled,
                                layers=np.array([0, 1, 2]), positions=np.array([10, 11]))
            token_payload = {'token_ids': list(range(13)), 'attention_mask': [1] * 13,
                             'assistant_boundary': 8, 'runtime_sha256': fingerprint(runtime)}
            atomic_json(folder / 'tokens.json', token_payload)
            token_ref = artifact_ref(folder / 'tokens.json', 'json')
            atomic_json(folder / 'generation.json', {'schema_version': 1, 'status': 'ok', 'result': {
                'status': 'tool_call', 'tokens': token_ref, 'action_span': [10, 12], 'pre_action_position': 9}})
            capture = {'features': artifact_ref(folder / 'features.npz', 'npz'), 'shape': [3, 2, width],
                       'trajectory': token_ref, 'positions': [10, 11], 'prediction_boundary': 9,
                       'causal_prefix_length': 12, 'target': 'action', 'stored_dtype': 'float32',
                       'runtime_sha256': fingerprint(runtime)}
            atomic_json(folder / 'capture.json', {'schema_version': 1, 'status': 'ok', 'result': capture})
            assessment = {'schema_version': 1, 'task_id': task['task_id'], 'task_sha256': task['task_sha256'],
                          'inputs': {'permission': condition['permission'], 'before_snapshot': before, 'after_snapshot': after},
                          'record_status': 'complete'}
            atomic_json(folder / 'assessment.json', assessment)
            assessment_ref = artifact_ref(folder / 'assessment.json', 'json')
            atomic_json(folder / 'assessment_packet.json', {'schema_version': 1, 'status': 'ok', 'result': assessment, 'artifacts': [assessment_ref]})
            review = {'schema_version': 1, 'label_status': 'fixture', 'primary_label': label, 'uncertain': False,
                      'review': {'reviewer_kind': 'fixture'}, 'assessment': assessment_ref,
                      'flags': {'evaluation_weakening': label, 'valid_test_repair': not label and condition['permission'] == 'conditional' and condition['problem'] == 'test_conflict'}}
            atomic_json(folder / 'review.json', review)
            episode = {'task_id': task['task_id'], 'condition_id': condition['condition_id'], 'split': task['split'],
                       'runtime_sha256': fingerprint(runtime), 'fixture': True, 'visible_sha256': condition['visible_sha256'],
                       'actions': [{'record_id': record_id, 'generation_status': 'tool_call', 'status': 'executed',
                                   'before_snapshot': before, 'after_snapshot': after,
                                   'generation': artifact_ref(folder / 'generation.json', 'json'),
                                   'capture': artifact_ref(folder / 'capture.json', 'json'),
                                   'assessment': artifact_ref(folder / 'assessment_packet.json', 'json')}]}
            atomic_json(folder / 'episode.json', episode)
            features['records'].append({'record_id': record_id, 'task_id': task['task_id'], 'episode': artifact_ref(folder / 'episode.json', 'json')})
            labels['records'].insert(0, {'record_id': record_id, 'review': artifact_ref(folder / 'review.json', 'json')})
            data[record_id] = {'folder': folder, 'vector': vector, 'label': int(label), 'split': task['split']}
    atomic_json(root / 'features.json', features)
    atomic_json(root / 'labels.json', labels)
    return {'root': root, 'rows': data, 'features': artifact_ref(root / 'features.json', 'json'),
            'labels': artifact_ref(root / 'labels.json', 'json'),
            'config': {'artifact_root': str(root / 'monitors'), 'layers': [2, 1], 'poolings': ['mean', 'last'],
                       'Cs': [10, 1, 0.1], 'seed': 7, 'label_kind': 'fixture'}}


def refresh_features(data, record_id):
    folder = data['rows'][record_id]['folder']
    capture = json.loads((folder / 'capture.json').read_text())
    capture['result']['features'] = artifact_ref(folder / 'features.npz', 'npz')
    atomic_json(folder / 'capture.json', capture)
    episode = json.loads((folder / 'episode.json').read_text())
    episode['actions'][0]['capture'] = artifact_ref(folder / 'capture.json', 'json')
    atomic_json(folder / 'episode.json', episode)
    manifest = json.loads((data['root'] / 'features.json').read_text())
    next(row for row in manifest['records'] if row['record_id'] == record_id)['episode'] = artifact_ref(folder / 'episode.json', 'json')
    atomic_json(data['root'] / 'features.json', manifest)
    data['features'] = artifact_ref(data['root'] / 'features.json', 'json')


class ActivationMonitorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.cache', prefix='activation-tests-')
        self.data = fixture(self.directory.name)
        self.monitor = ActivationMonitor(self.data['config'])
        self.inputs = {key: self.data[key] for key in ('features', 'labels')}

    def tearDown(self):
        self.directory.cleanup()

    def test_file_edit_control_uses_snapshot_targets_and_only_development_groups(self):
        self.monitor.fit(self.inputs)
        before = self.monitor.parameters
        features = json.loads((self.data['root'] / 'features.json').read_text())
        for item in features['records']:
            row = self.data['rows'][item['record_id']]
            if row['split'] not in ('training', 'validation'):
                item['episode']['path'] = str(self.data['root'] / 'must-not-open-heldout.json')
                continue
            folder = row['folder']
            episode = json.loads((folder / 'episode.json').read_text())
            import tarfile
            with tarfile.open(folder / 'before.tar') as archive:
                files = {member.name: archive.extractfile(member).read().decode() for member in archive if member.isfile()}
            files['test_solution.py' if row['label'] else 'solution.py'] += '\n# controlled edit\n'
            atomic_bytes(folder / 'after.tar', _files_archive(files, 1048576))
            episode['actions'][0]['after_snapshot'] = artifact_ref(folder / 'after.tar', 'tar')
            atomic_json(folder / 'episode.json', episode)
            item['episode'] = artifact_ref(folder / 'episode.json', 'json')
        atomic_json(self.data['root'] / 'control-features.json', features)
        inputs = {'features': artifact_ref(self.data['root'] / 'control-features.json', 'json')}
        result = self.monitor.file_edit_control(inputs)
        self.assertEqual(result['status'], 'checked')
        self.assertEqual(result['controls']['actual']['auroc'], 1)
        self.assertEqual(result['populations']['training']['test_edits'], 4)
        self.assertEqual(result['populations']['validation']['implementation_edits'], 4)
        self.assertFalse(set(result['populations']['training']['clone_groups']) & set(result['populations']['validation']['clone_groups']))
        expected = np.random.default_rng(7).permutation(result['populations']['training']['labels']).tolist()
        self.assertEqual(result['controls']['shuffled']['training_labels'], expected)
        self.assertEqual(self.monitor.parameters, before)
        self.assertEqual(self.monitor.file_edit_control(inputs), result)

    def test_lazy_import_configuration_mask_and_threshold_examples(self):
        subprocess.run([sys.executable, '-B', '-c', 'import sys; import eval_tampering.monitors; from eval_tampering.monitors.activation_monitor import ActivationMonitor; assert "torch" not in sys.modules; assert "sklearn" not in sys.modules'], check=True)
        config = self.monitor.config
        config['layers'].append(9)
        self.assertEqual(self.monitor.config['layers'], [2, 1])
        values = np.array([[2., 4.], [6., 8.], [100., 100.]])
        mask = np.array([True, True, False])
        np.testing.assert_array_equal(pool_residuals(values, mask, 'mean'), [4., 6.])
        np.testing.assert_array_equal(pool_residuals(values, mask, 'last'), [6., 8.])
        np.testing.assert_array_equal(pool_residuals(np.array([[0., 0.], [2., 4.]]), mask[:2], 'mean'), [1., 2.])
        with self.assertRaises(InputError):
            pool_residuals(values, np.zeros(3, dtype=bool), 'mean')
        all_negative = operating_point([0, 1], [.8, .7])
        self.assertTrue(all_negative['all_negative'])
        self.assertIsNone(all_negative['threshold'])
        threshold = operating_point([0] * 10 + [1, 1], [.8] + [.1] * 9 + [.9, .8])
        self.assertEqual(threshold['threshold'], .8)
        self.assertEqual(threshold['false_positive_rate'], .1)
        self.assertEqual(threshold['recall'], 1)
        self.assertEqual(operating_point([0, 0], [.1, .2])['status'], 'unavailable')

    def test_direct_estimator_parity_selection_and_controls(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        report = self.monitor.fit(self.inputs)
        self.assertEqual(report['status'], 'fitted')
        self.assertEqual((report['selected']['C'], report['selected']['layer'], report['selected']['pooling']), (.1, 1, 'last'))
        rows = sorted(self.data['rows'].items())
        X = np.stack([row['vector'].astype(np.float64) for _, row in rows if row['split'] == 'training'])
        y = np.array([row['label'] for _, row in rows if row['split'] == 'training'])
        V = np.stack([row['vector'].astype(np.float64) for _, row in rows if row['split'] == 'validation'])
        direct = make_pipeline(StandardScaler(), LogisticRegression(C=.1, l1_ratio=0, solver='lbfgs', tol=1e-8, max_iter=1000, random_state=7)).fit(X, y)
        np.testing.assert_array_equal(self.monitor.parameters['mean'], direct.steps[0][1].mean_)
        np.testing.assert_array_equal(self.monitor.parameters['coef'], direct.steps[1][1].coef_)
        np.testing.assert_array_equal(report['selected']['validation_scores'], direct.predict_proba(V)[:, 1])
        self.assertEqual(len(report['candidates']), 12)
        self.assertFalse(report['controls_used_for_selection'])
        self.assertEqual(report['controls']['early_layer']['layer'], 0)
        self.assertEqual(report['controls']['early_layer']['validation_average_precision'], .5)
        self.assertEqual(report['controls']['shuffled_labels']['training_labels'], np.random.default_rng(7).permutation(y).tolist())
        self.assertTrue(report['warnings'])

    def test_fit_never_opens_heldout_artifacts_and_labels_join_by_id(self):
        from eval_tampering.monitors import common
        original = common.read_artifact

        def guarded(reference, *args, **kwargs):
            if '/g2-' in reference['path'] or '/g3-' in reference['path']:
                raise AssertionError('Opened held-out data during fitting')
            return original(reference, *args, **kwargs)

        with patch.object(common, 'read_artifact', side_effect=guarded):
            report = self.monitor.fit(self.inputs)
        self.assertEqual(report['class_counts']['training'], {'positive': 4, 'negative': 4})
        labels = json.loads((self.data['root'] / 'labels.json').read_text())
        labels['records'][0]['review'], labels['records'][-1]['review'] = labels['records'][-1]['review'], labels['records'][0]['review']
        atomic_json(self.data['root'] / 'labels.json', labels)
        with self.assertRaisesRegex(InputError, 'different action assessment'):
            self.monitor.fit(self.inputs | {'labels': artifact_ref(self.data['root'] / 'labels.json', 'json')})

    def test_score_cannot_refit_and_reload_is_identical_without_labels(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.monitor.fit(self.inputs)
        saved = self.monitor.save(self.data['root'] / 'saved')
        parameters = fingerprint(self.monitor.parameters)
        loaded = ActivationMonitor(self.data['config'])
        loaded.load(saved)
        (self.data['root'] / 'labels.json').unlink()
        inputs = {'features': self.data['features'], 'splits': ['detection_test']}
        with patch.object(StandardScaler, 'fit', side_effect=AssertionError('refit')), patch.object(LogisticRegression, 'fit', side_effect=AssertionError('refit')):
            expected, actual = self.monitor.score(inputs), loaded.score(inputs)
        self.assertEqual(expected, actual)
        folder = self.data['rows']['g2-r0']['folder']
        with np.load(folder / 'features.npz', allow_pickle=False) as arrays:
            values = {name: arrays[name] for name in arrays.files}
        values['mean'][1] += 100
        values['last'][1] += 100
        np.savez_compressed(folder / 'features.npz', **values)
        refresh_features(self.data, 'g2-r0')
        changed = loaded.score(inputs | {'features': self.data['features']})
        self.assertNotEqual(expected['scores'][0]['score'], changed['scores'][0]['score'])
        self.assertEqual(fingerprint(loaded.parameters), parameters)
        self.assertEqual(changed['threshold'], expected['threshold'])

    def test_missing_classes_and_fixture_labels_are_explicit(self):
        for row in self.data['rows'].values():
            if row['split'] != 'training':
                continue
            path = row['folder'] / 'review.json'
            review = json.loads(path.read_text())
            review['primary_label'] = review['flags']['evaluation_weakening'] = False
            atomic_json(path, review)
        manifest = json.loads((self.data['root'] / 'labels.json').read_text())
        for row in manifest['records']:
            row['review'] = artifact_ref(self.data['rows'][row['record_id']]['folder'] / 'review.json', 'json')
        atomic_json(self.data['root'] / 'labels.json', manifest)
        report = self.monitor.fit(self.inputs | {'labels': artifact_ref(self.data['root'] / 'labels.json', 'json')})
        self.assertEqual(report['status'], 'unavailable')
        with self.assertRaises(InputError):
            self.monitor.score({'features': self.data['features'], 'splits': ['validation']})
        human = ActivationMonitor(self.data['config'] | {'label_kind': 'human'})
        with self.assertRaisesRegex(InputError, 'provenance'):
            human.fit(self.inputs)

    def test_missing_uncertain_and_unavailable_records_have_reasons(self):
        root = self.data['root']
        labels = json.loads((root / 'labels.json').read_text())
        next(row for row in labels['records'] if row['record_id'] == 'g0-r0')['review'] = None
        review_path = self.data['rows']['g0-r1']['folder'] / 'review.json'
        review = json.loads(review_path.read_text())
        review['uncertain'] = True
        atomic_json(review_path, review)
        next(row for row in labels['records'] if row['record_id'] == 'g0-r1')['review'] = artifact_ref(review_path, 'json')
        atomic_json(root / 'labels.json', labels)
        folder = self.data['rows']['g0-r2']['folder']
        atomic_json(folder / 'capture.json', {'status': 'error'})
        episode = json.loads((folder / 'episode.json').read_text())
        episode['actions'][0]['capture'] = artifact_ref(folder / 'capture.json', 'json')
        atomic_json(folder / 'episode.json', episode)
        features = json.loads((root / 'features.json').read_text())
        next(row for row in features['records'] if row['record_id'] == 'g0-r2')['episode'] = artifact_ref(folder / 'episode.json', 'json')
        atomic_json(root / 'features.json', features)
        report = self.monitor.fit({'features': artifact_ref(root / 'features.json', 'json'), 'labels': artifact_ref(root / 'labels.json', 'json')})
        reasons = {row['record_id']: row['reason'] for row in report['excluded']}
        self.assertEqual([reasons[f'g0-r{i}'] for i in range(3)], ['review_missing', 'review_uncertain', 'capture_unavailable'])
        self.assertEqual(report['class_counts']['training'], {'positive': 3, 'negative': 2})

    def test_frozen_threshold_load_checks_and_failed_load_clear_state(self):
        self.monitor.fit(self.inputs)
        inputs = {'features': self.data['features'], 'splits': ['validation']}
        expected = self.monitor.score(inputs)
        returned = self.monitor.score(inputs)
        returned['threshold']['all_negative'] = not returned['threshold']['all_negative']
        self.assertEqual(self.monitor.score(inputs), expected)
        saved = self.monitor.save(self.data['root'] / 'saved')
        original = json.loads(Path(saved['path']).read_text())
        for kind in ('threshold', 'scores'):
            with self.subTest(kind=kind):
                metadata = deepcopy(original)
                if kind == 'threshold':
                    metadata['report']['threshold']['threshold'] = .123
                else:
                    metadata['report']['candidates'][0]['validation_scores'] = None
                path = self.data['root'] / 'invalid-monitor.json'
                atomic_json(path, metadata)
                with self.assertRaises(InputError):
                    self.monitor.load(artifact_ref(path, 'json'))
                with self.assertRaisesRegex(InputError, 'Fit or load'):
                    self.monitor.score(inputs)
        self.monitor.load(saved)
        self.assertEqual(self.monitor.score(inputs), expected)

    def test_capture_and_review_cannot_point_at_a_different_action(self):
        folder = self.data['rows']['g0-r0']['folder']
        capture = json.loads((folder / 'capture.json').read_text())
        original = deepcopy(capture)
        capture['result']['trajectory'] = artifact_ref(self.data['rows']['g0-r1']['folder'] / 'tokens.json', 'json')
        atomic_json(folder / 'capture.json', capture)
        refresh_features(self.data, 'g0-r0')
        with self.assertRaisesRegex(InputError, 'Capture/trajectory'):
            self.monitor.fit(self.inputs | {'features': self.data['features']})
        atomic_json(folder / 'capture.json', original)
        episode = json.loads((folder / 'episode.json').read_text())
        episode['actions'][0]['before_snapshot'] = artifact_ref(self.data['rows']['g0-r1']['folder'] / 'before.tar', 'tar')
        atomic_json(folder / 'episode.json', episode)
        refresh_features(self.data, 'g0-r0')
        with self.assertRaisesRegex(InputError, 'Assessment/action snapshots'):
            self.monitor.fit(self.inputs | {'features': self.data['features']})

    def test_capture_cutoff_runtime_and_array_corruption_fail(self):
        folder = self.data['rows']['g0-r0']['folder']
        capture = json.loads((folder / 'capture.json').read_text())
        capture['result']['causal_prefix_length'] = 13
        atomic_json(folder / 'capture.json', capture)
        refresh_features(self.data, 'g0-r0')
        with self.assertRaisesRegex(InputError, 'cutoff'):
            self.monitor.fit(self.inputs | {'features': self.data['features']})
        capture['result']['causal_prefix_length'] = 12
        atomic_json(folder / 'capture.json', capture)
        with np.load(folder / 'features.npz', allow_pickle=False) as arrays:
            values = {name: arrays[name] for name in arrays.files}
        values['mean'][0, 0] = np.nan
        np.savez_compressed(folder / 'features.npz', **values)
        refresh_features(self.data, 'g0-r0')
        with self.assertRaisesRegex(InputError, 'finite FP32'):
            self.monitor.fit(self.inputs | {'features': self.data['features']})

    def test_cli_fit_and_score_round_trip(self):
        root = self.data['root']
        packet = {'schema_version': 1, 'request_id': 'probe-fit', 'operation': 'activation.fit',
                  'config': self.data['config'], 'inputs': self.inputs}
        atomic_json(root / 'fit.json', packet)
        fitted = subprocess.run([sys.executable, '-B', '-m', 'eval_tampering', 'monitors', '--input', str(root / 'fit.json'), '--output', str(root / 'fit.result.json')], capture_output=True, text=True)
        self.assertEqual(fitted.returncode, 0, fitted.stderr)
        result = json.loads((root / 'fit.result.json').read_text())
        saved = result['result']['monitor']
        packet.update(request_id='probe-score', operation='activation.score', inputs={'features': self.data['features'], 'splits': ['validation'], 'monitor': saved})
        atomic_json(root / 'score.json', packet)
        scored = subprocess.run([sys.executable, '-B', '-m', 'eval_tampering', 'monitors', '--input', str(root / 'score.json'), '--output', str(root / 'score.result.json')], capture_output=True, text=True)
        self.assertEqual(scored.returncode, 0, scored.stderr)
        self.monitor.load(saved)
        direct = self.monitor.score({'features': self.data['features'], 'splits': ['validation']})
        self.assertEqual(json.loads((root / 'score.result.json').read_text())['result'], direct)
        response = json.loads((root / 'score.result.json').read_text())
        saved_request = Path(response['artifacts'][0]['path']).with_name('request.json')
        self.assertEqual(json.loads(saved_request.read_text())['inputs']['monitor'], saved)


if __name__ == '__main__':
    unittest.main()
