"""Trace rendered values to saved evidence; retain missingness and every control."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from eval_tampering.evidence import selected
from eval_tampering.figures import ROLES
from eval_tampering.messages import artifact_ref, atomic_json, read_artifact
from eval_tampering.monitors.common import json_artifact
from test_analysis import analysis_fixture, steering_analysis_fixture
from test_evidence import evidence_fixture
from test_patch_analysis import patch_analysis_fixture
from test_sampling import sampling_analysis_fixture
from test_reasoning_analysis import reasoning_analysis_fixture


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.cache', prefix='figure-tests-')
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def save(self, name, value):
        path = self.root / (name + '.json')
        atomic_json(path, value)
        return artifact_ref(path, 'json')

    def request(self, analyzer, reports, request_id='figures'):
        return {'schema_version': 1, 'request_id': request_id, 'operation': 'analysis.figures', 'config': analyzer.config,
                'inputs': {'reports': {role: reports.get(role) for role in ROLES}, 'rule': self.save('rule', analyzer.rule())}}

    def verify_exports(self, result):
        from PIL import Image
        for figure in result['figures']:
            self.assertEqual(ET.fromstring(read_artifact(figure['svg'], 'svg', 16777216)).tag, '{http://www.w3.org/2000/svg}svg')
            with Image.open(figure['png']['path']) as picture:
                self.assertGreaterEqual(picture.width, 1200)
                self.assertGreaterEqual(picture.height, 600)
            payload = json_artifact(figure['data'])
            self.assertEqual(payload['source'], figure['source'])
            self.assertIn('FIXTURE', payload['scope'])
        self.assertEqual(result['new_model_calls'], 0)
        self.assertEqual(result['new_provider_calls'], 0)
        self.assertEqual(result['monitor_fits'], 0)

    def test_sampling_rates_missing_slots_and_separate_cost_views(self):
        _, _, _, analyzer, inputs = sampling_analysis_fixture(self.root / 'sampling')
        sampling = analyzer.sampling(inputs, self.root / 'sampling-report')
        _, cost_analyzer, inputs, wire = evidence_fixture(self.root / 'cost')
        cost = cost_analyzer.costs(inputs, self.root / 'cost-report')
        request = self.request(analyzer, {'sampling': sampling['summary'], 'cost': cost['summary']})
        with patch('openai.OpenAI', side_effect=AssertionError('figures must not call providers')):
            packet = analyzer.handle(request)
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        self.verify_exports(result)
        self.assertEqual(result['missing_reports'], ['detection', 'patch', 'steering', 'reasoning'])
        self.assertEqual(len(result['figures']), 4)
        data = json_artifact(result['figures'][0]['data'])
        unavailable = 0
        for panel in data['data']:
            for row in panel['rows']:
                metric = selected({'artifact': sampling['summary'], 'pointer': row['source_pointer']})
                self.assertEqual([row[key] for key in ('value', 'numerator', 'denominator', 'unknown', 'bounds', 'interval95')],
                                 [metric['rate']['value'], metric['event_count'], metric['known_count'], metric['unknown_count'], metric['bounds'], metric['rate']['bootstrap']['interval95']])
                unavailable += row['value'] is None
        self.assertGreater(unavailable, 0)
        self.assertEqual(sum(row['planned'] for row in json_artifact(result['figures'][1]['data'])['data']), 64)
        cost_data = json_artifact(next(row['data'] for row in result['figures'] if row['role'] == 'cost'))
        self.assertIn('not additive', cost_data['caption'])
        self.assertIn('USD 9.00', cost_data['caption'])
        self.assertEqual(len(wire), 6)

    def test_reasoning_groups_and_disputes_use_saved_frozen_metrics(self):
        _, analyzer, inputs, _ = reasoning_analysis_fixture(self.root / 'reasoning')
        (self.root / 'reasoning-report').mkdir()
        report = analyzer.reasoning(inputs, self.root / 'reasoning-report')
        packet = analyzer.handle(self.request(analyzer, {'reasoning': report['summary']}))
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        self.verify_exports(result)
        self.assertEqual(len(result['figures']), 15)
        for figure in result['figures']:
            payload = json_artifact(figure['data'])
            for panel in payload['data']:
                for row in panel['rows']:
                    metric = selected({'artifact': report['summary'], 'pointer': row['source_pointer']})
                    self.assertEqual(row['value'], metric['value'])
                    self.assertEqual(row['interval95'], metric['bootstrap']['interval95'])
                    if panel['metric'] in ('average_precision', 'auroc'):
                        self.assertIsNone(row['value'])
        changed = json_artifact(report['summary'])
        primary = json_artifact(changed['primary_all_actions'])
        primary['coverage']['completed_calls'] += 1
        changed['primary_all_actions'] = self.save('altered-primary', primary)
        packet = analyzer.handle(self.request(analyzer, {'reasoning': self.save('altered-reasoning', changed)}, 'altered-reasoning'))
        self.assertEqual(packet['error']['code'], 'hash_mismatch')

    def test_detection_bins_metrics_unavailable_hosted_and_changed_table(self):
        data, analyzer, inputs, _ = analysis_fixture(self.root / 'detection', hosted=False)
        summary = analyzer.summarize(inputs, self.root / 'detection-report')
        request = self.request(analyzer, {'detection': summary['summary']})
        packet = analyzer.handle(request)
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        self.verify_exports(result)
        self.assertEqual(len(result['figures']), 11)
        for figure in result['figures'][:5]:
            payload = json_artifact(figure['data'])
            self.assertTrue(all(panel['bins'] == payload['data'][0]['bins'] for panel in payload['data']))
            self.assertEqual([payload['data'][0]['bins'][0], payload['data'][0]['bins'][-1]], [0., 1.])
            self.assertIn('orange dashed', payload['caption'])
            self.assertEqual(sum(len(panel['observations']) for panel in payload['data']), 8)
            for panel in payload['data']:
                for series in panel['series']:
                    self.assertEqual(sum(series['counts']), sum(row['label'] == series['label'] and row['score'] is not None for row in panel['observations']))
        hosted = json_artifact(next(figure['data'] for figure in result['figures'] if figure['figure_id'] == 'scores-hosted'))
        self.assertEqual(sum(panel['missing'] for panel in hosted['data']), 8)
        for figure in result['figures'][5:]:
            for panel in json_artifact(figure['data'])['data']:
                for row in panel['rows']:
                    metric = selected({'artifact': summary['summary'], 'pointer': row['source_pointer']})
                    self.assertEqual(row['value'], metric['value'])
                    self.assertEqual(row['interval95'], metric['bootstrap']['interval95'])
        changed = json_artifact(summary['summary'])
        path = self.root / 'wrong-table.csv'
        path.write_text('scope,value\npooled,0\n')
        changed['table'] = artifact_ref(path, 'csv')
        bad = self.request(analyzer, {'detection': self.save('changed-table', changed)}, 'changed-table')
        self.assertEqual(analyzer.handle(bad)['error']['code'], 'hash_mismatch')

    def test_patch_recipient_classes_steering_controls_and_paired_counts(self):
        _, _, _, patch_analyzer, inputs = patch_analysis_fixture(self.root / 'patch')
        patch_summary = patch_analyzer.patch(inputs, self.root / 'patch-report')
        _, analyzer, _, inputs = steering_analysis_fixture(self.root / 'steering')
        steering = analyzer.steering(inputs, self.root / 'steering-report')
        packet = analyzer.handle(self.request(analyzer, {'patch': patch_summary['summary'], 'steering': steering['summary']}))
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        self.verify_exports(result)
        for role, summary in [('patch', patch_summary), ('steering', steering)]:
            points = []
            for figure in result['figures']:
                if figure['role'] != role or figure['figure_id'].endswith('coverage'):
                    continue
                payload = json_artifact(figure['data'])
                for panel in payload['data']:
                    for row in panel['rows']:
                        metric = selected({'artifact': summary['summary'], 'pointer': row['source_pointer']})
                        self.assertEqual(row['value'], metric['difference']['value'])
                        self.assertEqual(row['denominator'], metric['scorable_count'])
                        self.assertEqual(row['unknown'], metric['missing_count'])
                        self.assertEqual(row['bounds'], metric['bounds'])
                        points.append(row['source_pointer'])
            self.assertEqual(len(points), len(summary['summaries'])*2)
            self.assertEqual(len(points), len(set(points)))
        self.assertTrue(any('original tampering recipients' in figure['title'] for figure in result['figures']))
        self.assertTrue(any('original repair recipients' in figure['title'] for figure in result['figures']))

    def test_cli_missing_reports_stale_rule_and_wrong_provenance(self):
        _, analyzer, inputs, _ = evidence_fixture(self.root / 'cost')
        summary = analyzer.costs(inputs, self.root / 'cost-report')
        request = self.request(analyzer, {'cost': summary['summary']}, 'cli-figures')
        source, output = self.save('cli-request', request), self.root / 'cli-output.json'
        process = subprocess.run([sys.executable, '-B', '-m', 'eval_tampering', 'analysis', '--input', source['path'], '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(output.read_text())['result']
        self.assertEqual(result['missing_reports'], ['sampling', 'detection', 'patch', 'steering', 'reasoning'])
        for kind in ('fixture', 'rule', 'value'):
            changed = json_artifact(summary['summary'])
            if kind == 'fixture':
                changed['fixture'] = False
            elif kind == 'rule':
                changed['rule']['sources']['figures.py'] = '0'*64
            else:
                changed['declared_invoice_total_usd'] = '1000'
            packet = analyzer.handle(self.request(analyzer, {'cost': self.save('bad-' + kind, changed)}, 'bad-' + kind))
            self.assertEqual(packet['status'], 'error', packet)
        empty = self.request(analyzer, {}, 'empty')
        self.assertEqual(analyzer.handle(empty)['status'], 'error')
        inventory = self.save('empty-inventory', {'schema_version': 1, 'fixture': True, 'records': [], 'invoices': []})
        empty_cost = analyzer.costs(inputs | {'inventory': inventory}, self.root / 'empty-cost-report')
        packet = analyzer.handle(self.request(analyzer, {'cost': empty_cost['summary']}, 'empty-cost'))
        self.assertEqual(packet['status'], 'ok', packet)
        self.assertIn('No declared work', read_artifact(packet['result']['figures'][0]['svg'], 'svg', 16777216).decode())


if __name__ == '__main__':
    unittest.main()
