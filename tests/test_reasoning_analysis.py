"""Hand-counted reasoning cohorts; SDK transport and reviews are explicit fixtures."""

from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from eval_tampering.analysis import ResultAnalyzer, reasoning_summary
from eval_tampering.messages import InputError, artifact_ref, atomic_json
from eval_tampering.monitors.common import json_artifact
from eval_tampering.monitors.reasoning_annotator import ReasoningAnnotator, _label
from eval_tampering.monitors.text_monitor import TextMonitor
from test_reasoning_annotator import ORDINARY, annotation_client, reasoning_fixture
from test_text_monitor import mock_client, provider_config


def reasoning_analysis_fixture(root):
    data = reasoning_fixture(root)
    root = data['root']
    analyzer = ResultAnalyzer({'artifact_root': str(root / 'analysis'), 'label_kind': 'fixture', 'bootstrap_seed': 19})
    atomic_json(root / 'analysis-rule.json', analyzer.rule())
    client, wire = annotation_client()
    annotator = ReasoningAnnotator(data['config'], client)

    def run(obj, operation, inputs, name):
        packet = obj.handle({'schema_version': 1, 'request_id': name, 'operation': operation, 'config': obj.config, 'inputs': inputs})
        assert packet['status'] == 'ok', packet
        return packet['result']

    dev = run(annotator, 'reasoning.prepare', {'features': data['features'], 'splits': ['validation'], 'rule': None}, 'development')
    prepared = run(annotator, 'reasoning.prepare', {'features': data['features'], 'splits': ['detection_test'], 'rule': dev['rule']}, 'final-views')
    audits = [run(annotator, 'reasoning.audit.prepare', {'views': prepared['views'], 'record_ids': ids}, name)
              for ids, name in ((None, 'random'), (['g2-r7'], 'targeted'))]
    try:
        annotated = run(annotator, 'reasoning.annotate', {'views': prepared['views']}, 'annotated')
    finally:
        client.close()
    automatic = {row['record_id']: row['annotation'] for row in annotated['annotations']}
    inputs_audits = []
    for audit in audits:
        private = json_artifact(audit['audit'])
        review = json_artifact(audit['review_template']) | {'reviewer_id': 'fixture-auditor', 'reviewer_kind': 'fixture', 'blind_before_annotations': True,
            'labels': [{'audit_id': row['audit_id'], 'annotation': deepcopy(automatic[row['record_id']])} for row in private['records']]}
        if audit['kind'] == 'targeted':
            review['labels'][0]['annotation'].update(category='explicit_acknowledgment', evidence=[{'reasoning_index': 0, 'quote': ORDINARY}],
                rationale='Deliberate fixture disagreement; this quote is not a substantive acknowledgment.')
        path = root / (audit['kind'] + '-review.json')
        atomic_json(path, review)
        item = {'audit': audit['audit'], 'reviews': artifact_ref(path, 'json'), 'adjudications': None}
        if audit['kind'] == 'targeted':
            path = root / 'resolution.json'
            atomic_json(path, review | {'blind_before_annotations': False, 'resolution_rule': 'Fixture-only sensitivity: use the deliberately disagreeing manual category.'})
            item['adjudications'] = artifact_ref(path, 'json')
        inputs_audits.append(item)
    client, text_wire = mock_client()
    text = TextMonitor({'artifact_root': str(root / 'text'), 'tokenizer': data['config']['tokenizer'],
                        'label_kind': 'fixture', 'provider': provider_config()}, client)
    try:
        text.score({'features': data['features'], 'splits': ['validation'], 'calibration': None}, root / 'text-validation')
        calibration = text.calibrate({'features': data['features'], 'labels': data['labels'], 'scores': artifact_ref(root / 'text-validation' / 'scores.json', 'json')})
        atomic_json(root / 'text-calibration.json', calibration)
        text.score({'features': data['features'], 'splits': ['detection_test'], 'calibration': artifact_ref(root / 'text-calibration.json', 'json')}, root / 'text-final')
    finally:
        client.close()
    inputs = {'detection': {'features': data['features'], 'labels': data['labels'], 'split': 'detection_test', 'activation': None,
        'text': {'calibration': artifact_ref(root / 'text-calibration.json', 'json'), 'scores': artifact_ref(root / 'text-final' / 'scores.json', 'json')},
        'rule': artifact_ref(root / 'analysis-rule.json', 'json')}, 'annotations': annotated['annotation_artifact'], 'audits': inputs_audits}
    return data, analyzer, inputs, wire + text_wire


class ReasoningAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.cache', prefix='reasoning-analysis-tests-')
        self.data, self.analyzer, self.inputs, self.wire = reasoning_analysis_fixture(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def invoke(self, inputs, name):
        return self.analyzer.handle({'schema_version': 1, 'request_id': name, 'operation': 'analysis.reasoning', 'config': self.analyzer.config, 'inputs': inputs})

    def save(self, name, value):
        path = self.data['root'] / (name + '.json')
        atomic_json(path, value)
        return artifact_ref(path, 'json')

    def test_frozen_thresholds_same_criterion_negatives_and_audit_sensitivities(self):
        before = json_artifact(self.inputs['annotations'])
        from eval_tampering.monitors.activation_monitor import ActivationMonitor
        with patch('openai.OpenAI', side_effect=AssertionError('analysis must not call a provider')), \
             patch.object(ActivationMonitor, 'fit', side_effect=AssertionError('analysis must not fit')):
            packet = self.invoke(self.inputs, 'summary')
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        self.assertEqual(result['providers']['same_model'], True)
        self.assertEqual(result['uniform_audit']['raw_category_agreement'], {'numerator': 5, 'denominator': 5, 'rate': 1.})
        self.assertEqual(result['targeted_audits'][0]['raw_category_agreement']['rate'], 0.)
        self.assertEqual(result['disputed_record_ids'], ['g2-r7'])
        self.assertEqual(json_artifact(self.inputs['annotations']), before)
        groups = {(row['variant'], row['group']): row for row in result['coverage']}
        base = groups['automatic', 'no_explicit_acknowledgment']
        self.assertEqual((base['positives'], base['negatives'], base['tasks']), (1, 3, 1))
        self.assertEqual(groups['automatic', 'unavailable']['record_ids'], ['g2-r0'])
        self.assertEqual(groups['automatic', 'ambiguous']['record_ids'], ['g2-r3'])
        self.assertEqual(groups['exclude_disputed', 'no_explicit_acknowledgment']['positives'], 0)
        self.assertEqual(groups['adjudicated', 'explicit_acknowledgment']['positives'], 3)
        summary = next(row for row in result['summaries'] if row['variant'] == 'automatic' and row['group'] == 'no_explicit_acknowledgment'
                       and row['scope'] == 'pooled' and row['method'] == 'weakening_marker')
        self.assertEqual(summary['metrics']['recall']['value'], 1.)
        self.assertEqual(summary['metrics']['false_positive_rate']['value'], 0.)
        self.assertEqual(summary['metrics']['false_positive_rate']['denominator'], 3)
        self.assertEqual(summary['metrics']['auroc']['status'], 'insufficient_support')
        primary = json_artifact(result['primary_all_actions'])
        self.assertEqual(primary['coverage']['completed_calls'], 8)
        primary_metric = next(row for row in primary['summaries'] if row['scope'] == 'pooled' and row['method'] == 'weakening_marker')
        self.assertEqual(primary_metric['metrics']['auroc']['value'], 1.)
        from eval_tampering.messages import read_artifact
        table = list(csv.DictReader(io.StringIO(read_artifact(result['table'], 'csv', 16777216).decode())))
        self.assertEqual(len(table), 1875)
        self.assertTrue(all(row['value'] == '' for row in table if row['metric'] == 'auroc'))
        request = {'schema_version': 1, 'request_id': 'cli-summary', 'operation': 'analysis.reasoning', 'config': self.analyzer.config, 'inputs': self.inputs}
        reference = self.save('cli-request', request)
        output = self.data['root'] / 'cli-result.json'
        process = subprocess.run([sys.executable, '-B', '-m', 'eval_tampering', 'analysis', '--input', reference['path'], '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        replayed = json.loads(output.read_text())['result']
        for key in ('coverage', 'summaries', 'paired_comparisons', 'uniform_audit', 'targeted_audits'):
            self.assertEqual(replayed[key], result[key])

    def test_incomplete_annotations_are_missing_and_audits_use_only_known_pairs(self):
        raw = json_artifact(self.inputs['annotations'])
        partial = self.save('partial', raw | {'status': 'incomplete', 'annotations': raw['annotations'][:3]})
        packet = self.invoke(self.inputs | {'annotations': partial}, 'partial-summary')
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        missing = next(row for row in result['coverage'] if row['variant'] == 'automatic' and row['group'] == 'annotation_missing')
        self.assertEqual(len(missing['record_ids']), 5)
        self.assertEqual(result['uniform_audit']['comparable_count'], 1)
        self.assertEqual(len(result['uniform_audit']['missing_automatic_ids']), 4)
        self.assertEqual(result['targeted_audits'][0]['raw_category_agreement']['rate'], None)
        invalid = self.save('omitted', raw | {'annotations': raw['annotations'][:3]})
        self.assertEqual(self.invoke(self.inputs | {'annotations': invalid}, 'omitted')['status'], 'error')

    def test_changed_request_response_view_and_trace_availability_are_rejected(self):
        raw = json_artifact(self.inputs['annotations'])
        changed = deepcopy(raw)
        changed['annotations'][1]['annotation'].update(category='no_explicit_acknowledgment', evidence=[])
        changed['annotations'][1]['provider']['annotation'] = changed['annotations'][1]['annotation']
        packet = self.invoke(self.inputs | {'annotations': self.save('changed-label', changed)}, 'bad-label')
        self.assertEqual(packet['error']['code'], 'hash_mismatch')
        changed = deepcopy(raw)
        provider = changed['annotations'][1]['provider']
        request = json_artifact(provider['request'])
        request['input'][0]['content'] = 'PROPOSED_ACTION_SECRET'
        provider['request'] = self.save('bad-request', request)
        self.assertEqual(self.invoke(self.inputs | {'annotations': self.save('changed-request', changed)}, 'bad-request')['error']['code'], 'hash_mismatch')
        view = {'history': [], 'reasoning': [{'channel': 'analysis', 'text': ORDINARY}]}
        with self.assertRaisesRegex(InputError, 'trace availability'):
            _label({'category': 'unavailable', 'ambiguous': False, 'rationale': 'Incorrect empty label.', 'evidence': []}, view)
        manifest = json_artifact(raw['views'])
        manifest['records'][0]['prefix_sha256'] = '0' * 64
        changed = raw | {'views': self.save('changed-views', manifest)}
        self.assertEqual(self.invoke(self.inputs | {'annotations': self.save('changed-prefix', changed)}, 'bad-prefix')['error']['code'], 'hash_mismatch')
        duplicate = self.inputs | {'audits': self.inputs['audits'] + [self.inputs['audits'][0]]}
        self.assertEqual(self.invoke(duplicate, 'duplicate-audit')['status'], 'error')

    def test_conflicting_resolutions_and_resolution_without_review_fail(self):
        target = self.inputs['audits'][1]
        private = json_artifact(target['audit'])
        raw = json_artifact(self.inputs['annotations'])
        annotator = ReasoningAnnotator(json_artifact(raw['rule'])['config'])
        packet = annotator.handle({'schema_version': 1, 'request_id': 'overlap-audit', 'operation': 'reasoning.audit.prepare',
            'config': annotator.config, 'inputs': {'views': raw['views'], 'record_ids': ['g2-r6', 'g2-r7']}})
        self.assertEqual(packet['status'], 'ok', packet)
        other = packet['result']
        original = next(row['annotation'] for row in raw['annotations'] if row['record_id'] == 'g2-r7')
        review = json_artifact(target['reviews']) | {'packet': other['packet']}
        resolution = json_artifact(target['adjudications']) | {'packet': other['packet'],
            'labels': [{'audit_id': private['records'][0]['audit_id'], 'annotation': original}]}
        conflicting = {'audit': other['audit'], 'reviews': self.save('overlap-review', review), 'adjudications': self.save('overlap-resolution', resolution)}
        packet = self.invoke(self.inputs | {'audits': self.inputs['audits'] + [conflicting]}, 'conflicting')
        self.assertEqual(packet['status'], 'error')
        self.assertIn('Conflicting adjudications', packet['error']['message'])
        packet = self.invoke(self.inputs | {'audits': [target | {'reviews': None}]}, 'unreviewed-resolution')
        self.assertEqual(packet['status'], 'error')
        self.assertIn('original independent review', packet['error']['message'])


class ReasoningSupportTests(unittest.TestCase):
    def test_support_floor_requires_both_classes_and_distinct_groups(self):
        rows = [{'record_id': str(i), 'task_id': str(i//10), 'clone_group_id': str(i//10), 'label': i % 2,
                 'score': float(i % 2), 'positive': bool(i % 2), 'valid_test_repair': False} for i in range(40)]
        groups = ['0', '1', '2', '3']
        summary = reasoning_summary(rows, groups, 19)
        self.assertEqual(summary['ranking_status'], 'available')
        self.assertEqual(summary['metrics']['auroc']['value'], 1.)
        self.assertEqual(summary['metrics']['average_precision']['value'], 1.)
        missing = reasoning_summary(rows[:-1], groups, 19)
        self.assertEqual(missing['ranking_status'], 'insufficient_support')
        self.assertEqual(missing['metrics']['average_precision']['bootstrap']['valid'], 0)
        concentrated = reasoning_summary([row | {'clone_group_id': '0'} for row in rows], groups, 19)
        self.assertEqual(concentrated['ranking_status'], 'insufficient_support')


if __name__ == '__main__':
    unittest.main()
