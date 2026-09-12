"""Hand-counted cost and local evidence checks; no billing or research claims."""

from copy import deepcopy
import csv
from decimal import Decimal
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from eval_tampering.analysis import ResultAnalyzer
from eval_tampering.evidence import claim_index, selected, verify_links
from eval_tampering.messages import InputError, artifact_ref, atomic_bytes, atomic_json, fingerprint, read_artifact
from eval_tampering.monitors.common import json_artifact
from eval_tampering.monitors.text_monitor import TextMonitor
from test_run import configuration, request as episode_request
from test_text_monitor import text_fixture, mock_client, provider_config


def evidence_fixture(root):
    data = text_fixture(root)
    root = data['root']
    client, wire = mock_client(('good', 'malformed', 'timeout'))
    config = data['config'] | {'provider': provider_config()}
    try:
        report = TextMonitor(config, client).score({'features': data['features'], 'splits': ['validation'], 'calibration': None}, root / 'provider-batch')
    finally:
        client.close()
    atomic_json(root / 'provider-config.json', config['provider'])
    atomic_bytes(root / 'rate.txt', b'Fictional fixture compute rate: USD 0.02 per second. Not a purchase or a bill.\n')
    rate = {'usd_per_second': '0.02', 'evidence': artifact_ref(root / 'rate.txt', 'bytes')}
    inventory = {'schema_version': 1, 'fixture': True, 'records': [], 'invoices': []}

    def add(work_id, kind, category, raw=None, request=None, parent=None, compute_rate=None):
        folder = root / 'work' / work_id
        folder.mkdir(parents=True)
        if request is not None:
            atomic_json(folder / 'request.json', request)
        if raw is not None:
            atomic_json(folder / 'record.json', raw)
        row = {'work_id': work_id, 'kind': kind, 'category': category, 'parent_work_id': parent,
            'request': artifact_ref(folder / 'request.json', 'json') if request is not None else None,
            'record': artifact_ref(folder / 'record.json', 'json') if raw is not None else None,
            'provider_config': None, 'compute_rate': compute_rate}
        inventory['records'].append(row)
        return row

    request = episode_request(configuration(root), 'collection-fixture')
    add('collection', 'episode', 'collection', {'schema_version': 1, 'status': 'complete', 'fixture': True,
        'episode_id': request['request_id'], 'request_sha256': fingerprint(request), 'elapsed_seconds': 100., 'output_tokens': 20}, request, compute_rate=rate)
    request = {'schema_version': 1, 'request_id': 'capture-fixture', 'operation': 'capture', 'config': configuration(root)['model'], 'inputs': {}}
    add('capture', 'model', 'activation_replay', {'status': 'complete', 'request_id': request['request_id'], 'operation': 'capture',
        'operation_seconds': 25., 'load_seconds': 999., 'result': {}}, request, parent='collection', compute_rate=rate)
    request = {'schema_version': 1, 'request_id': 'probe-fixture', 'operation': 'activation.score', 'config': {'label_kind': 'fixture'}, 'inputs': {}}
    add('probe', 'probe', 'probe_scoring', {'status': 'complete', 'operation': 'activation.score', 'elapsed_seconds': .5, 'result': {}}, request, compute_rate=rate)
    for work_id, category, seconds, parent in [('labeling', 'labeling', 300., None), ('hook-overhead', 'hook_pooling_overhead', 2., 'capture')]:
        add(work_id, 'measurement', category, {'schema_version': 1, 'fixture': True, 'work_id': work_id, 'status': 'recorded',
            'elapsed_seconds': seconds, 'method': 'Explicit fictional timing for arithmetic tests; no human activity or profiled overhead is claimed.',
            'evidence': [rate['evidence']]}, parent=parent, compute_rate=rate if parent else None)
    add('transfer', 'measurement', 'transfer')
    for index, scored in enumerate(report['scores']):
        provider = scored['provider']
        inventory['records'].append({'work_id': f'provider-{index}', 'kind': 'provider', 'category': 'text_monitor', 'parent_work_id': None,
            'request': provider.get('request'), 'record': provider.get('record'),
            'provider_config': {'artifact': artifact_ref(root / 'provider-config.json', 'json'), 'pointer': ''}, 'compute_rate': None})
    atomic_bytes(root / 'invoice.txt', b'Explicit fictional invoice transcription fixture: USD 9.00. No actual bill.\n')
    inventory['invoices'].append({'invoice_id': 'fixture-invoice', 'amount_usd': '9.00', 'evidence': artifact_ref(root / 'invoice.txt', 'bytes'),
                                  'work_ids': ['collection', 'capture', 'probe']})
    atomic_json(root / 'inventory.json', inventory)
    analyzer = ResultAnalyzer({'artifact_root': str(root / 'analysis'), 'label_kind': 'fixture', 'bootstrap_seed': 19})
    atomic_json(root / 'analysis-rule.json', analyzer.rule())
    return data, analyzer, {'inventory': artifact_ref(root / 'inventory.json', 'json'), 'rule': artifact_ref(root / 'analysis-rule.json', 'json')}, wire


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.cache', prefix='evidence-tests-')
        self.data, self.analyzer, self.inputs, self.wire = evidence_fixture(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def save(self, name, value):
        path = self.data['root'] / (name + '.json')
        atomic_json(path, value)
        return artifact_ref(path, 'json')

    def invoke(self, operation, inputs, name):
        return self.analyzer.handle({'schema_version': 1, 'request_id': name, 'operation': operation, 'config': self.analyzer.config, 'inputs': inputs})

    def test_decimal_usage_unknown_charges_nested_time_and_invoice_views(self):
        with patch('openai.OpenAI', side_effect=AssertionError('accounting must not call a provider')):
            packet = self.invoke('analysis.costs', self.inputs, 'costs')
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        groups = {row['category']: row for row in result['categories']}
        self.assertEqual(groups['collection']['known_root_seconds'], 100.)
        self.assertEqual(Decimal(groups['collection']['known_compute_estimate_usd']), Decimal('2'))
        self.assertEqual(groups['activation_replay']['nested_work'], 1)
        self.assertEqual(groups['activation_replay']['known_root_seconds'], 0)
        self.assertEqual(Decimal(groups['probe_scoring']['known_compute_estimate_usd']), Decimal('.01'))
        self.assertEqual(Decimal(groups['text_monitor']['usage_calculated_usd']), Decimal('.00068'))
        self.assertEqual(Decimal(groups['text_monitor']['unknown_charge_reservations_usd']), Decimal('.001224'))
        self.assertEqual(groups['text_monitor']['unknown_charge_attempts'], 1)
        self.assertEqual(groups['text_monitor']['unresolved_provider_work'], 6)
        self.assertEqual(result['declared_invoice_total_usd'], '9.00')
        self.assertNotIn('total_cost_usd', result)
        rows = json_artifact(result['rows'])['rows']
        self.assertEqual(len(rows), 14)
        self.assertEqual(next(row for row in rows if row['work_id'] == 'capture')['seconds'], 25.)
        known = [row for row in rows if row['charge_status'] == 'known']
        self.assertEqual([(row['input_tokens'], row['cached_input_tokens'], row['output_tokens']) for row in known], [(100, 20, 20)] * 2)
        self.assertEqual(sum(row['status'] == 'missing' for row in rows), 6)
        table = list(csv.DictReader(io.StringIO(read_artifact(result['table'], 'csv', 16777216).decode())))
        self.assertEqual(len(table), 14)
        self.assertEqual(next(row for row in table if row['work_id'] == 'transfer')['seconds'], '')
        self.assertGreater(result['verified_links']['reference_count'], 10)
        self.assertEqual(len(self.wire), 6)

    def test_duplicate_costs_cycles_changed_usage_and_evidence_fail(self):
        original = json_artifact(self.inputs['inventory'])
        changed = deepcopy(original)
        changed['records'].append(changed['records'][0] | {'work_id': 'duplicate'})
        self.assertEqual(self.invoke('analysis.costs', self.inputs | {'inventory': self.save('duplicate', changed)}, 'duplicate')['status'], 'error')
        changed = deepcopy(original)
        changed['records'][0]['parent_work_id'] = 'hook-overhead'
        self.assertEqual(self.invoke('analysis.costs', self.inputs | {'inventory': self.save('cycle', changed)}, 'cycle')['status'], 'error')
        for name in ('usage', 'cost'):
            changed = deepcopy(original)
            provider = changed['records'][6]
            raw = json_artifact(provider['record'])
            if name == 'usage':
                raw['usage']['input_tokens'] += 1
            else:
                raw['cost_usd'] = '0'
            provider['record'] = self.save('changed-provider-' + name, raw)
            packet = self.invoke('analysis.costs', self.inputs | {'inventory': self.save(name, changed)}, 'bad-' + name)
            self.assertEqual(packet['error']['code'], 'hash_mismatch')
        changed = deepcopy(original)
        raw = json_artifact(changed['records'][6]['record'])
        changed['records'].append(changed['records'][6] | {'work_id': 'same-charge', 'record': self.save('copied-charge', raw)})
        self.assertEqual(self.invoke('analysis.costs', self.inputs | {'inventory': self.save('same-charge', changed)}, 'same-charge')['status'], 'error')
        raw['request_id'] = 'changed-request-id'
        changed['records'][-1]['record'] = self.save('changed-charge-id', raw)
        self.assertEqual(self.invoke('analysis.costs', self.inputs | {'inventory': self.save('same-request', changed)}, 'same-request')['status'], 'error')
        changed = deepcopy(original)
        changed['invoices'][0]['evidence']['sha256'] = '0' * 64
        self.assertEqual(self.invoke('analysis.costs', self.inputs | {'inventory': self.save('changed-invoice', changed)}, 'bad-invoice')['error']['code'], 'hash_mismatch')

    def test_claim_values_partial_links_scope_and_cli_roundtrip(self):
        result = self.invoke('analysis.costs', self.inputs, 'costs')['result']
        exclusions = self.save('exclusions', {'schema_version': 1, 'fixture': True, 'excluded': []})
        claim = {'claim_id': 'fixture-invoice', 'text': 'The declared fictional invoice amount is USD 9.00.', 'scope': 'fixture',
            'manifest': self.inputs['inventory'], 'records': [result['rows']],
            'calculation': {'artifact': result['summary'], 'pointer': '/declared_invoice_total_usd', 'expected': '9.00'},
            'figures': [], 'exclusions': exclusions, 'audits': []}
        inputs = {'claims': self.save('claims', {'schema_version': 1, 'records': [claim]}), 'rule': self.inputs['rule']}
        packet = self.invoke('analysis.index', inputs, 'index')
        self.assertEqual(packet['status'], 'ok', packet)
        indexed = packet['result']
        self.assertEqual(indexed['partial_link_sets'], 1)
        self.assertEqual(indexed['claims'][0]['missing_evidence'], ['figures', 'audits'])
        self.assertIn(str(self.data['root'].resolve()), read_artifact(indexed['index'], 'markdown', 16777216).decode())
        changed = deepcopy(claim)
        changed['calculation']['expected'] = '10.00'
        with self.assertRaisesRegex(InputError, 'Cited value'):
            claim_index({'claims': self.save('wrong-value', {'schema_version': 1, 'records': [changed]})}, self.data['root'], True)
        for name, value in [('manifest', 'malformed'), ('records', [{'path': 'incomplete'}])]:
            with self.subTest(name=name), self.assertRaises(InputError):
                claim_index({'claims': self.save('bad-' + name, {'schema_version': 1, 'records': [claim | {name: value}]})}, self.data['root'], True)
        changed = claim | {'scope': 'development'}
        with self.assertRaisesRegex(InputError, 'provenance'):
            claim_index({'claims': self.save('false-research', {'schema_version': 1, 'records': [changed]})}, self.data['root'], False)
        calculation = self.save('development', {'schema_version': 1, 'fixture': False, 'stage': 'development', 'split': 'validation', 'count': 1})
        changed = claim | {'scope': 'detection_test', 'calculation': {'artifact': calculation, 'pointer': '/count', 'expected': 1}}
        with self.assertRaisesRegex(InputError, 'held-out'):
            claim_index({'claims': self.save('false-heldout', {'schema_version': 1, 'records': [changed]})}, self.data['root'], False)
        request = {'schema_version': 1, 'request_id': 'cli-index', 'operation': 'analysis.index', 'config': self.analyzer.config, 'inputs': inputs}
        source, output = self.save('cli-request', request), self.data['root'] / 'cli-result.json'
        process = subprocess.run([sys.executable, '-B', '-m', 'eval_tampering', 'analysis', '--input', source['path'], '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(output.read_text())['result']['claims'], indexed['claims'])

    def test_pointer_escapes_bounds_and_broken_nested_links(self):
        reference = self.save('pointer', {'a/b': {'~key': [3, None]}})
        self.assertEqual(selected({'artifact': reference, 'pointer': '/a~1b/~0key/0'}), 3)
        self.assertIsNone(selected({'artifact': reference, 'pointer': '/a~1b/~0key/1'}))
        for pointer in ('/a~2b', '/a~', '/a~1b/~0key/01', '/a~1b/~0key/2', '/a~1b/~0key/-'):
            with self.subTest(pointer=pointer), self.assertRaises(InputError):
                selected({'artifact': reference, 'pointer': pointer})
        nested = self.save('nested', {'child': reference | {'sha256': '0' * 64}})
        with self.assertRaisesRegex(InputError, 'hash mismatch'):
            verify_links(nested)


if __name__ == '__main__':
    unittest.main()
