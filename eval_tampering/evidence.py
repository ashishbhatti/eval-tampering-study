"""Saved-work accounting and checked claim links; no execution, fitting or billing API."""

from collections import Counter
import csv
from decimal import Decimal
import html
import io
import math

from .messages import artifact_ref, atomic_bytes, atomic_json, decode_json, fields, fingerprint, identifier, local_path, read_artifact, require, validate_request
from .monitors.common import json_artifact
from .monitors.provider import _decimal, validate_provider

CATEGORIES = ('collection', 'intervention', 'calibration', 'activation_replay', 'probe_training', 'probe_scoring', 'probe_setup',
              'text_monitor', 'reasoning_annotation', 'labeling', 'instrumentation', 'hook_pooling_overhead', 'startup_idle', 'transfer')


def selected(reference):
    """Resolve a bounded RFC 6901 pointer without a query language or dependency."""
    fields(reference, {'artifact', 'pointer'}, 'JSON value reference')
    pointer = reference['pointer']
    require(type(pointer) is str and len(pointer) <= 2000 and (pointer == '' or pointer.startswith('/')), 'Invalid JSON pointer')
    value = json_artifact(reference['artifact'])
    for token in pointer.split('/')[1:]:
        require(all(part and part[0] in '01' for part in token.split('~')[1:]), 'Invalid JSON pointer escape')
        token = token.replace('~1', '/').replace('~0', '~')
        if type(value) is list:
            require(token.isascii() and token.isdecimal() and (token == '0' or not token.startswith('0')) and int(token) < len(value), 'JSON array pointer is out of range')
            value = value[int(token)]
        else:
            require(type(value) is dict and token in value, 'JSON pointer does not identify a value')
            value = value[token]
    return value


def verify_links(value):
    """Verify the declared local graph, including nested artifact references."""
    queue, checked, total = [], {}, 0
    def visit(item):
        if type(item) is dict:
            if set(item) == {'path', 'sha256', 'format'}:
                queue.append(item)
            else:
                for child in item.values():
                    visit(child)
        elif type(item) is list:
            for child in item:
                visit(child)
    visit(value)
    while queue:
        ref = queue.pop()
        fields(ref, {'path', 'sha256', 'format'}, 'evidence reference')
        require(type(ref['format']) is str, 'Invalid evidence format')
        key = str(local_path(ref['path'])), ref['sha256'], ref['format']
        require(type(ref['sha256']) is str, 'Invalid evidence hash')
        if key in checked:
            continue
        require(len(checked) < 8192, 'Evidence graph exceeds 8192 distinct references')
        data = read_artifact(ref, ref['format'], 134217728)
        total += len(data)
        require(total <= 1073741824, 'Evidence graph exceeds 1 GiB of declared local artifacts')
        checked[key] = ref
        if ref['format'] == 'json':
            visit(decode_json(data))
    return {'references': list(checked.values()), 'reference_count': len(checked), 'bytes_checked': total,
            'scope': 'content hashes and local access only; no remote access or semantic claim validation'}


def _number(value, name, integer=False):
    require(value is None or (type(value) in ((int,) if integer else (int, float)) and math.isfinite(value) and value >= 0), 'Invalid ' + name)
    return value


def _provider_cost(item, raw, seen):
    config = selected(item['provider_config'])
    validate_provider(config)
    require(config is not None and raw.get('request') == item['request'], 'Provider cost/request mismatch')
    request = json_artifact(item['request'])
    require(type(request) is dict and request.get('model') == config['model'], 'Provider cost/model mismatch')
    request_id = raw.get('request_id')
    require(request_id is None or type(request_id) is str and bool(request_id), 'Invalid provider request ID')
    keys = {('provider_request_artifact', str(local_path(item['request']['path'])))}
    if request_id is not None:
        keys.add(('provider_request_id', request_id))
    require(not keys & seen, 'Duplicate provider charge/request in cost inventory')
    seen.update(keys)
    reserved = _decimal(raw.get('reservation_usd'), 'reservation_usd')
    charge = raw.get('charge_status')
    require(charge in (None, 'known', 'unknown'), 'Invalid provider charge status')
    result = {'charge_status': charge or 'generation_not_reserved', 'usage_calculated_usd': None, 'unknown_reservation_usd': None,
              'input_tokens': None, 'cached_input_tokens': None, 'output_tokens': None}
    if charge == 'known':
        response = json_artifact(raw.get('response'))
        usage = raw.get('usage')
        require(type(response) is dict and response.get('model') == config['model'] and response.get('service_tier') == 'default' and
                type(usage) is dict and usage == response.get('usage'), 'Provider cost/raw usage mismatch', 'hash_mismatch')
        for name in ('input_tokens', 'output_tokens', 'total_tokens'):
            require(type(usage.get(name)) is int and 0 <= usage[name] <= 262144, 'Invalid provider token accounting')
        details = usage.get('input_tokens_details')
        require(type(details) is dict and type(details.get('cached_tokens')) is int and 0 <= details['cached_tokens'] <= usage['input_tokens'] and
                details.get('cache_write_tokens', 0) == 0 and usage['total_tokens'] == usage['input_tokens'] + usage['output_tokens'] and
                type(raw.get('counted_input_tokens')) is int and usage['input_tokens'] <= raw['counted_input_tokens'] <= config['max_input_tokens'] and
                usage['output_tokens'] <= config['max_output_tokens'], 'Invalid provider billing/token limits')
        prices = {name: _decimal(config['rates'][name], name) for name in ('input', 'cached_input', 'output')}
        cached = details['cached_tokens']
        cost = ((usage['input_tokens']-cached)*prices['input'] + cached*prices['cached_input'] + usage['output_tokens']*prices['output']) / 1000000
        require(cost == _decimal(raw.get('cost_usd'), 'cost_usd') and cost <= reserved, 'Provider recorded cost disagrees with raw usage/rates', 'hash_mismatch')
        result.update(usage_calculated_usd=str(cost), input_tokens=usage['input_tokens'], cached_input_tokens=cached, output_tokens=usage['output_tokens'])
    else:
        require(raw.get('cost_usd') is None, 'Unknown provider charge cannot have a known cost')
        if charge == 'unknown':
            result['unknown_reservation_usd'] = str(reserved)
        else:
            require(reserved == 0 and raw.get('response') is None and raw.get('usage') is None, 'Unsent generation has a reservation or response')
            result['usage_calculated_usd'] = '0'
    return result


def cost_report(inputs, directory, fixture):
    fields(inputs, {'inventory'}, 'cost report inputs')
    inventory = json_artifact(inputs['inventory'])
    fields(inventory, {'schema_version', 'fixture', 'records', 'invoices'}, 'cost inventory')
    require(type(inventory['schema_version']) is int and inventory['schema_version'] == 1 and type(inventory['fixture']) is bool and inventory['fixture'] == fixture, 'Cost inventory provenance mismatch')
    require(type(inventory['records']) is list and len(inventory['records']) <= 4096 and type(inventory['invoices']) is list and len(inventory['invoices']) <= 256, 'Invalid cost inventory size')
    rows, work, sources, charges = [], {}, set(), set()
    for item in inventory['records']:
        fields(item, {'work_id', 'kind', 'category', 'parent_work_id', 'request', 'record', 'provider_config', 'compute_rate'}, 'cost work item')
        identifier(item['work_id'], 'work_id')
        require(item['work_id'] not in work and type(item['category']) is str and item['category'] in CATEGORIES and
                item['kind'] in ('episode', 'model', 'probe', 'provider', 'measurement'), 'Invalid or duplicate cost work')
        work[item['work_id']] = item
        for name in ('request', 'record'):
            if item[name] is not None:
                fields(item[name], {'path', 'sha256', 'format'}, 'cost ' + name + ' reference')
        require((item['provider_config'] is not None) == (item['kind'] == 'provider'), 'Provider settings belong only to provider work')
        if item['kind'] == 'provider':
            config = selected(item['provider_config'])
            require(config is not None, 'Provider configuration is required')
            validate_provider(config)
        row = {key: item[key] for key in ('work_id', 'kind', 'category', 'parent_work_id', 'request', 'record')}
        row.update(status='missing', seconds=None, output_tokens=None, input_tokens=None, cached_input_tokens=None,
                   usage_calculated_usd=None, unknown_reservation_usd=None, compute_estimate_usd=None, charge_status=None)
        if item['record'] is not None:
            fields(item['record'], {'path', 'sha256', 'format'}, 'cost record reference')
            key = str(local_path(item['record']['path']))
            require(key not in sources, 'One saved work record appears more than once')
            sources.add(key)
            raw = json_artifact(item['record'])
            require(type(raw) is dict and type(raw.get('status')) is str, 'Invalid saved work record')
            row['status'] = raw['status']
            if item['kind'] == 'provider':
                require(item['category'] in ('text_monitor', 'reasoning_annotation'), 'Provider work category mismatch')
                row.update(_provider_cost(item, raw, charges))
            elif item['kind'] == 'measurement':
                fields(raw, {'schema_version', 'fixture', 'work_id', 'status', 'elapsed_seconds', 'method', 'evidence'}, 'manual measurement')
                require(type(raw['schema_version']) is int and raw['schema_version'] == 1 and type(raw['fixture']) is bool and raw['fixture'] == fixture and raw['work_id'] == item['work_id'] and
                        type(raw['method']) is str and 0 < len(raw['method'].strip()) <= 2000 and type(raw['evidence']) is list,
                        'Manual measurement identity/method mismatch')
                row['seconds'] = _number(raw['elapsed_seconds'], 'measured seconds')
            else:
                request = json_artifact(item['request'])
                operations = {'episode'} if item['kind'] == 'episode' else {'activation.fit', 'activation.score', 'activation.load', 'activation.save'} if item['kind'] == 'probe' else {'load', 'prepare', 'resume', 'generate', 'capture', 'diagnose', 'check'}
                validate_request(request, operations)
                if item['kind'] == 'episode':
                    require(raw.get('request_sha256') == fingerprint(request) and raw.get('episode_id') == request['request_id'] and
                            (raw.get('fixture') is None or type(raw['fixture']) is bool and raw['fixture'] == fixture) and
                            item['category'] in ('collection', 'intervention', 'calibration'), 'Episode cost/request mismatch', 'hash_mismatch')
                    row.update(seconds=_number(raw.get('elapsed_seconds'), 'episode seconds'), output_tokens=_number(raw.get('output_tokens'), 'episode tokens', True))
                else:
                    require(local_path(item['record']['path']).parent == local_path(item['request']['path']).parent and raw.get('operation') == request['operation'],
                            'Operation cost/request mismatch', 'hash_mismatch')
                    if item['kind'] == 'probe':
                        require(request['config'].get('label_kind') == ('fixture' if fixture else 'human') and item['category'] == (
                            'probe_training' if request['operation'] == 'activation.fit' else 'probe_scoring' if request['operation'] == 'activation.score' else 'probe_setup'), 'Probe cost provenance/category mismatch')
                        row['seconds'] = _number(raw.get('elapsed_seconds'), 'probe seconds')
                    else:
                        require(raw.get('request_id') == request['request_id'] and request['config'].get('profile') in ('tiny-gpt-oss-cpu', 'gpt-oss-20b-mxfp4') and
                                (request['config']['profile'] == 'tiny-gpt-oss-cpu') == fixture, 'Model cost provenance mismatch')
                        require(request['operation'] != 'capture' or item['category'] == 'activation_replay', 'Capture is a separately costed model replay')
                        require(request['operation'] not in ('diagnose', 'check') or item['category'] == 'instrumentation', 'Diagnostics belong to instrumentation costs')
                        result = raw.get('result')
                        require(result is None or type(result) is dict, 'Invalid saved model result')
                        row.update(seconds=_number(raw.get('operation_seconds'), 'model operation seconds'),
                                   output_tokens=_number((result or {}).get('generated_tokens'), 'model output tokens', True))
        if item['kind'] == 'measurement':
            require(item['request'] is None, 'Manual measurements have no component request')
        if item['compute_rate'] is not None:
            fields(item['compute_rate'], {'usd_per_second', 'evidence'}, 'compute rate')
            rate = _decimal(item['compute_rate']['usd_per_second'], 'usd_per_second')
            require(item['compute_rate']['evidence'] is not None and item['kind'] != 'provider', 'Compute rate requires evidence and non-provider work')
            fields(item['compute_rate']['evidence'], {'path', 'sha256', 'format'}, 'compute rate evidence')
            if row['seconds'] is not None:
                row['compute_estimate_usd'] = str(Decimal(str(row['seconds'])) * rate)
        rows.append(row)
    for item in work.values():
        path, parent = {item['work_id']}, item['parent_work_id']
        while parent is not None:
            require(type(parent) is str and parent in work and parent not in path and len(path) < 32, 'Invalid/cyclic cost parent relationship')
            path.add(parent)
            parent = work[parent]['parent_work_id']
    invoices, invoice_ids, invoice_sources = [], set(), set()
    for invoice in inventory['invoices']:
        fields(invoice, {'invoice_id', 'amount_usd', 'evidence', 'work_ids'}, 'declared invoice')
        identifier(invoice['invoice_id'], 'invoice_id')
        fields(invoice['evidence'], {'path', 'sha256', 'format'}, 'invoice evidence reference')
        source = str(local_path(invoice['evidence']['path']))
        require(invoice['invoice_id'] not in invoice_ids and source not in invoice_sources and type(invoice['work_ids']) is list and
                all(type(key) is str and key in work for key in invoice['work_ids']) and len(set(invoice['work_ids'])) == len(invoice['work_ids']), 'Duplicate invoice or invalid work scope')
        invoice_ids.add(invoice['invoice_id'])
        invoice_sources.add(source)
        invoices.append(invoice | {'amount_usd': str(_decimal(invoice['amount_usd'], 'invoice amount')), 'verification': 'declared_transcription; receipt bytes linked, amount not independently extracted'})
    categories = []
    for category in CATEGORIES:
        chosen = [row for row in rows if row['category'] == category]
        roots = [row for row in chosen if row['parent_work_id'] is None]
        categories.append({'category': category, 'declared_work': len(chosen), 'status_counts': dict(Counter(row['status'] for row in chosen)),
            'root_work': len(roots), 'nested_work': len(chosen)-len(roots),
            'known_root_seconds': sum(row['seconds'] for row in roots if row['seconds'] is not None),
            'unknown_root_seconds': sum(row['seconds'] is None for row in roots),
            'known_compute_estimate_usd': str(sum((Decimal(row['compute_estimate_usd']) for row in roots if row['compute_estimate_usd'] is not None), Decimal(0))),
            'unknown_root_compute_estimates': sum(row['kind'] != 'provider' and row['compute_estimate_usd'] is None for row in roots),
            'usage_calculated_usd': str(sum((Decimal(row['usage_calculated_usd']) for row in chosen if row['usage_calculated_usd'] is not None), Decimal(0))),
            'unresolved_provider_work': sum(row['kind'] == 'provider' and row['usage_calculated_usd'] is None for row in chosen),
            'unknown_charge_attempts': sum(row['charge_status'] == 'unknown' for row in chosen),
            'unknown_charge_reservations_usd': str(sum((Decimal(row['unknown_reservation_usd']) for row in chosen if row['unknown_reservation_usd'] is not None), Decimal(0)))})
    links = verify_links(inputs['inventory'])
    atomic_json(directory / 'cost-rows.json', {'schema_version': 1, 'fixture': fixture, 'rows': rows})
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['work_id', 'kind', 'category', 'parent_work_id', 'status', 'seconds', 'input_tokens', 'cached_input_tokens', 'output_tokens',
                                              'usage_calculated_usd', 'unknown_reservation_usd', 'compute_estimate_usd', 'charge_status'])
    writer.writeheader()
    writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    atomic_bytes(directory / 'costs.csv', stream.getvalue().encode())
    return {'schema_version': 1, 'status': 'accounted', 'fixture': fixture, 'categories': categories, 'invoices': invoices,
        'declared_invoice_total_usd': str(sum((Decimal(row['amount_usd']) for row in invoices), Decimal(0))) if invoices else None,
        'rows': artifact_ref(directory / 'cost-rows.json', 'json'), 'table': artifact_ref(directory / 'costs.csv', 'csv'), 'verified_links': links,
        'limitations': ['Accounting covers only the declared inventory; orphaned or undeclared work is not proven absent',
            'Nested operation times are shown but not added again to root estimates; overlaps between roots remain unverified',
            'Sums of operation durations are not wall-clock or active research hours; inspect categories separately',
            'Usage charges, unknown reservations, compute estimates and declared invoices are separate non-additive views',
            'Unknown reservations are not charge upper bounds; token-count endpoint billing and account totals require invoice reconciliation',
            'Missing records, times and rates remain unknown; capture replay is not free monitoring and repeated load_seconds are not added to operation time']}


def claim_index(inputs, directory, fixture):
    fields(inputs, {'claims'}, 'claim index inputs')
    manifest = json_artifact(inputs['claims'])
    fields(manifest, {'schema_version', 'records'}, 'claim manifest')
    require(type(manifest['schema_version']) is int and manifest['schema_version'] == 1 and type(manifest['records']) is list and len(manifest['records']) <= 256, 'Invalid claim manifest')
    rows, seen = [], set()
    markdown = ['# Claim evidence index', '', 'Checked local references and cited values; claim interpretation and public access require review.', '']
    for claim in manifest['records']:
        fields(claim, {'claim_id', 'text', 'scope', 'manifest', 'records', 'calculation', 'figures', 'exclusions', 'audits'}, 'claim')
        identifier(claim['claim_id'], 'claim_id')
        require(claim['claim_id'] not in seen and type(claim['text']) is str and 0 < len(claim['text'].strip()) <= 4000 and
                claim['scope'] in ('fixture', 'development', 'detection_test', 'intervention_test') and (claim['scope'] == 'fixture') == fixture, 'Claim identity/scope mismatch')
        seen.add(claim['claim_id'])
        for name in ('records', 'figures', 'audits'):
            require(type(claim[name]) is list and len(claim[name]) <= 4096, 'Invalid claim evidence list')
        for ref in claim['figures']:
            fields(ref, {'path', 'sha256', 'format'}, 'claim figure')
            require(ref['format'] in ('png', 'svg', 'pdf'), 'Expected an exported figure artifact')
        value = None
        if claim['calculation'] is not None:
            fields(claim['calculation'], {'artifact', 'pointer', 'expected'}, 'claim calculation')
            expected = claim['calculation']['expected']
            require(expected is None or type(expected) in (str, bool, int, float), 'Cite one scalar calculation value')
            value = selected({key: claim['calculation'][key] for key in ('artifact', 'pointer')})
            require(type(value) is type(expected) and value == expected, 'Cited value differs from the saved calculation', 'hash_mismatch')
            calculation = json_artifact(claim['calculation']['artifact'])
            require(type(calculation) is dict and type(calculation.get('fixture')) is bool and calculation['fixture'] == fixture, 'Calculation fixture/research provenance mismatch')
            if claim['scope'] in ('detection_test', 'intervention_test'):
                parameters = calculation.get('inputs', {})
                require(type(parameters) is dict and type(parameters.get('detection', {})) is dict, 'Invalid calculation split metadata')
                split = calculation.get('split') or parameters.get('split') or parameters.get('detection', {}).get('split')
                require(split == claim['scope'] and calculation.get('stage') != 'development', 'Development calculation cannot support a held-out claim', 'split_leakage')
        missing = [name for name in ('manifest', 'records', 'calculation', 'figures', 'exclusions', 'audits') if not claim[name]]
        rows.append(claim | {'cited_value': value, 'missing_evidence': missing, 'link_status': 'complete' if not missing else 'partial'})
        markdown += [f"## {claim['claim_id']}", '', html.escape(claim['text']), '', 'Scope: ' + claim['scope'] + '. Missing evidence: ' + (', '.join(missing) or 'none') + '.', '']
        for name in ('manifest', 'records', 'calculation', 'figures', 'exclusions', 'audits'):
            refs = claim[name] if type(claim[name]) is list else [claim[name]['artifact']] if name == 'calculation' and claim[name] is not None else [claim[name]]
            for ref in refs:
                if ref is not None:
                    fields(ref, {'path', 'sha256', 'format'}, 'claim ' + name + ' reference')
                    markdown.append(f"- [{name}](<{local_path(ref['path'])}>)")
        markdown.append('')
    links = verify_links(inputs['claims'])
    atomic_bytes(directory / 'evidence-index.md', '\n'.join(markdown).encode())
    return {'schema_version': 1, 'status': 'indexed', 'fixture': fixture, 'claims': rows, 'verified_links': links,
        'complete_link_sets': sum(row['link_status'] == 'complete' for row in rows), 'partial_link_sets': sum(row['link_status'] == 'partial' for row in rows),
        'index': artifact_ref(directory / 'evidence-index.md', 'markdown'),
        'limitations': ['Checked values and local links do not establish scientific validity, correct claim wording, human review or public accessibility',
                       'Missing evidence remains explicit; an index is not experiment acceptance or permission to publish']}
