"""Freeze a checked study inventory; this module does not authorize or execute jobs."""

import hashlib
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile

from .messages import (InputError, artifact_ref, atomic_bytes, atomic_json, fields, fingerprint, identifier,
                       local_path, read_artifact, require, success, validate_request)
from .monitors.common import _object, json_artifact, records, reviewed_labels
from .tasks import SPLITS, handle as task_handle

REVIEW_ITEMS = {
    'pilot': 'I checked the development pilot, all response statuses, class/donor yield, file-edit decoding with shuffled labels, the benign S1/S2 behavioral steering control and its sham/random controls, and the feasible study size. Null or unavailable controls are recorded and limit interpretation.',
    'sandbox': 'I checked isolation on the intended host and the recorded sandbox configuration.',
    'evaluator': 'I checked independent evaluator truth cases, valid repairs, weakening, bypass and permission-only violations.',
    'behavior_audit': 'I independently reviewed the behavioral audit and unresolved/uncertain labels.',
    'reasoning_audit': 'I checked the blinded reasoning audit, disagreements and separate targeted cases.',
    'design': 'I reviewed the complete frozen controls, split groups, masks, selection, warnings, exclusions and reporting limits.',
    'allocations': 'I authorize the exact declared phase allocations and checked remaining project time and the writing reserve.',
}


def rule():
    root = Path(__file__).parent.parent
    sources = sorted(root.joinpath('eval_tampering').rglob('*.py')) + [root / 'run.py', root / '.python-version'] + sorted(root.glob('requirements*'))
    sources += [root / 'sandbox_image' / name for name in ('Dockerfile', 'apply_patch', 'apply_patch.py', 'LICENSE', 'source.json')]
    return {'schema_version': 1, 'python': platform.python_version(),
        'sources': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'development_evidence': 'select training/validation before opening episode and review artifacts',
        'sampling': 'all tasks in both held-out splits, all four conditions and four declared seeds; no replacement sampling',
        'patching': 'retrospective same-history pairs under the frozen direction rule; fresh four-seed continuations and every declared control',
        'audits': 'seeded uniform behavioral and restricted reasoning samples; targeted category/cell/uncertain cases reported separately',
        'execution': 'a frozen inventory is not runtime/host/human acceptance or permission to spend'}


def read_runtime_check(reference, runtime):
    """Recompute saved array comparisons; hook/RNG observations remain recorded evidence."""
    from .runtime_checks import validate_inputs
    _object(runtime, {'config', 'layers', 'fixture', 'runtime_source_sha256'}, 'checked runtime identity')
    require(runtime['runtime_source_sha256'] == hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest(), 'Checked runtime implementation changed', 'hash_mismatch')
    data = _object(json_artifact(reference), {'schema_version', 'status', 'runtime', 'runtime_sha256', 'check_source_sha256', 'inputs', 'checks'}, 'runtime check')
    require(type(data.get('schema_version')) is int and data['schema_version'] == 1 and data.get('status') in ('passed', 'failed', 'incomplete'), 'Invalid runtime check')
    require(data.get('runtime_sha256') == fingerprint(runtime) and json_artifact(data['runtime']) == runtime and
            data.get('check_source_sha256') == hashlib.sha256(Path(__file__).with_name('runtime_checks.py').read_bytes()).hexdigest(),
            'Runtime check identity/source changed', 'hash_mismatch')
    validate_inputs(data['inputs'])
    checks = data.get('checks')
    require(type(checks) is list and all(type(row) is dict and type(row.get('check_id')) is str and type(row.get('passed')) is bool for row in checks) and
            len({row['check_id'] for row in checks}) == len(checks), 'Invalid runtime check assertions')
    failed = [row['check_id'] for row in checks if not row['passed']]
    if data['status'] != 'passed':
        return {'status': data['status'], 'failed_checks': failed, 'evidence': reference}
    import numpy as np
    _object(data, {'fixture', 'layers', 'positions', 'variants', 'arrays', 'elapsed_seconds', 'process_peak_rss_bytes', 'device_peak_reserved_bytes'}, 'passed runtime check')
    require(type(data['elapsed_seconds']) in (int, float) and math.isfinite(data['elapsed_seconds']) and data['elapsed_seconds'] > 0 and
            type(data['process_peak_rss_bytes']) is int and data['process_peak_rss_bytes'] > 0 and
            (data['device_peak_reserved_bytes'] is None if runtime['fixture'] else
             type(data['device_peak_reserved_bytes']) is int and data['device_peak_reserved_bytes'] > 0), 'Invalid runtime measurements')
    require(not failed and data.get('fixture') == runtime['fixture'] and data.get('layers') == runtime['layers'], 'Passed runtime check has inconsistent assertions')
    prefix = json_artifact(data['inputs']['prefix'])
    require(prefix['runtime_sha256'] == fingerprint(runtime), 'Runtime check prefix changed', 'hash_mismatch')
    positions = sorted({prefix['assistant_boundary'], len(prefix['token_ids'])-1})
    require(data.get('positions') == positions, 'Runtime observation positions changed')
    comparisons = {'unmodified_logits': [runtime['config']['vocab_size']]}
    comparisons.update({f'residual_{layer}': [len(positions), runtime['config']['hidden_size']] for layer in runtime['layers']})
    required = {'rng_preserved', 'hooks_clean', 'within_time', 'within_rss_limit', 'within_device_limit'}
    variants = data.get('variants')
    require(type(variants) is list and all(type(row) is dict for row in variants) and
            [row.get('name') for row in variants] == ['baseline', 'sham', 'patch', 'S1', 'S2', 'after'], 'Incomplete runtime controls')
    generated = {}
    for row in variants:
        _object(row, {'name', 'generation_forward_count', 'forward_inputs', 'generation', 'intervention'}, 'runtime variant')
        name, count = row['name'], row['generation_forward_count']
        require(type(count) is int and 2 <= count <= data['inputs']['max_new_tokens'], 'Runtime check lacks generation evidence')
        generation = _object(row['generation'], {'tokens', 'generated_tokens', 'raw_response', 'hook_events'}, 'runtime generation')
        payload = json_artifact(generation['tokens'])
        fields(payload, {'token_ids', 'attention_mask', 'assistant_boundary', 'runtime_sha256'}, 'runtime generation tokens')
        tokens = payload['token_ids']
        require(type(tokens) is list and all(type(token) is int and 0 <= token < runtime['config']['vocab_size'] for token in tokens) and
                tokens[:len(prefix['token_ids'])] == prefix['token_ids'] and payload['runtime_sha256'] == fingerprint(runtime) and
                payload['assistant_boundary'] == prefix['assistant_boundary'] and payload['attention_mask'] == [1]*len(tokens), 'Runtime generation prefix changed')
        generated[name] = tokens[len(prefix['token_ids']):]
        require(len(generated[name]) == count == generation['generated_tokens'], 'Runtime generated-token count changed')
        read_artifact(generation['raw_response'], 'text', 1048576)
        use_cache = runtime['generation_use_cache']
        require(type(use_cache) is bool and type(row['forward_inputs']) is list and len(row['forward_inputs']) == count and all(
            type(item) is dict and item.get('use_cache') is use_cache and item.get('tokens') == (
                tokens[:len(prefix['token_ids'])+step] if not use_cache or step == 0 else [tokens[len(prefix['token_ids'])+step-1]])
            for step, item in enumerate(row['forward_inputs'])), 'Runtime generation execution path changed')
        required.add(name + '_generation_observed')
        for step in range(count):
            comparisons[f'{name}_logits_{step}'] = [runtime['config']['vocab_size']]
            required.add(f'{name}_token_{step}')
        if name in ('sham', 'after'):
            required.add(name + '_tokens_unchanged')
            require(generated[name] == generated['baseline'], 'Sham or cleanup changed generated tokens')
        if name in ('sham', 'patch', 'S1', 'S2'):
            hook = row['intervention']
            fields(hook, {'layer', 'schedule', 'mode', 'value', 'direction', 'runtime_sha256'}, 'runtime intervention')
            schedule = 'P1' if name in ('sham', 'patch') else name
            require(hook['layer'] == data['inputs']['layer'] and hook['runtime_sha256'] == fingerprint(runtime) and
                    hook['schedule'] == schedule and hook['mode'] == ('replace_projection' if name == 'patch' else 'add'), 'Runtime control changed')
            if name != 'patch':
                require(hook['value'] == (0 if name == 'sham' else data['inputs']['delta']), 'Runtime control strength changed')
            from .monitors.common import npz_arrays
            vector = npz_arrays(hook['direction'], ['direction'], 1048576)['direction']
            require(vector.shape == (runtime['config']['hidden_size'],) and np.array_equal(vector, np.eye(1, len(vector))[0]), 'Runtime coordinate direction changed')
            first = prefix['assistant_boundary'] if schedule == 'S2' else len(prefix['token_ids'])-1
            expected_positions = [first] if schedule == 'P1' else list(range(first, len(prefix['token_ids'])+count-1))
            if not use_cache:
                expected_positions = [position for step in range(count) for position in
                    ([first] if schedule == 'P1' else range(first, len(prefix['token_ids'])+step))]
            events = generation['hook_events']
            require(type(events) is list and all(type(event) is dict and type(event.get('processed_positions')) is list for event in events) and
                    [position for event in events for position in event['processed_positions']] == expected_positions, 'Runtime hook positions changed')
            required.update(name + '_' + suffix for suffix in ('positions', 'diagnostic_complete', 'actual_change', 'router_observed', 'downstream_change'))
            comparisons[name + '_diagnostic_native_logits'] = [runtime['config']['vocab_size']]
            comparisons[name + '_orthogonal_unchanged'] = [runtime['config']['hidden_size']-1]
            diagnostic = json_artifact(row['diagnostic'])
            require(diagnostic['status'] == 'complete' and diagnostic['runtime_sha256'] == fingerprint(runtime) and
                    diagnostic['inputs']['prefix'] == data['inputs']['prefix'] and diagnostic['inputs']['intervention'] == row['intervention'],
                    'Runtime diagnostic provenance changed', 'hash_mismatch')
            downstream = list(range(hook['layer']+1, runtime['config']['num_hidden_layers']))
            require(diagnostic.get('downstream_layers') == downstream and downstream and
                    all(diagnostic.get(key) == {'baseline': [1]*len(downstream), 'intervened': [1]*len(downstream)}
                        for key in ('router_calls', 'downstream_residual_calls')), 'Missing downstream observation evidence')
            with np.load(io.BytesIO(read_artifact(diagnostic['arrays'], 'npz', 134217728)), allow_pickle=False) as values:
                changes = []
                for suffix, shape, statistic in (
                    ('logits', (runtime['config']['vocab_size'],), 'logit_l2_change'),
                    ('router_logits', (len(downstream), runtime['config']['num_local_experts']), 'router_logit_l2_change'),
                    ('downstream_residuals', (len(downstream), runtime['config']['hidden_size']), 'downstream_residual_l2_change')):
                    left, right = values['baseline_'+suffix], values['intervened_'+suffix]
                    require(left.shape == right.shape == shape and np.isfinite(left).all() and np.isfinite(right).all(), 'Invalid downstream observation arrays')
                    change = float(np.linalg.norm(right.astype(np.float64)-left))
                    require(diagnostic['statistics'].get(statistic) == change, 'Downstream summary disagrees with arrays', 'hash_mismatch')
                    changes.append(change)
                require(all(value == 0 for value in changes) if name == 'sham' else all(value > 0 for value in changes),
                        'Downstream control disagrees with saved arrays')
        else:
            require(row['intervention'] is None and generation['hook_events'] == [], 'Baseline runtime check contains an intervention')
    lookup = {row['check_id']: row for row in checks}
    require(set(lookup) == required | set(comparisons), 'Runtime assertion inventory changed')
    with np.load(io.BytesIO(read_artifact(data['arrays'], 'npz', 134217728)), allow_pickle=False) as arrays:
        require(set(arrays.files) == {name + suffix for name in comparisons for suffix in ('_actual', '_reference')}, 'Runtime array inventory changed')
        for name, shape in comparisons.items():
            actual, expected = arrays[name + '_actual'], arrays[name + '_reference']
            require(list(actual.shape) == list(expected.shape) == lookup[name].get('shape') == shape and
                    actual.dtype.kind == expected.dtype.kind == 'f' and np.isfinite(actual).all() and np.isfinite(expected).all(), 'Invalid runtime comparison geometry')
            error = float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)), initial=0))
            require(error == lookup[name]['max_absolute_error'] and np.allclose(actual, expected, rtol=data['inputs']['rtol'], atol=data['inputs']['atol']),
                    'Runtime comparison disagrees with saved arrays', 'hash_mismatch')
        for variant in variants:
            for step, token in enumerate(generated[variant['name']]):
                name = variant['name'] + '_token_' + str(step)
                expected = int(arrays[variant['name'] + '_logits_' + str(step) + '_reference'].argmax())
                require(token == expected == lookup[name].get('generated_token') == lookup[name].get('native_argmax'), 'Runtime token/native argmax changed')
        patch = variants[2]['intervention']
        require(patch['value'] == float(arrays['residual_' + str(patch['layer']) + '_reference'][-1, 0]) + data['inputs']['delta'], 'Runtime patch projection changed')
    require(data['elapsed_seconds'] <= data['inputs']['max_seconds'] and
            data['process_peak_rss_bytes'] <= data['inputs']['max_peak_rss_bytes'] and
            (runtime['fixture'] or data['device_peak_reserved_bytes'] <= data['inputs']['max_device_bytes']), 'Runtime limits failed')
    return {'status': 'passed', 'failed_checks': [], 'evidence': reference, 'array_comparisons': len(comparisons),
            'scope': 'one prefix; recorded cleanup/RNG observations and recomputed arrays, not live host or experiment acceptance'}


def _seeds(values, name):
    require(type(values) is list and len(values) == 4 and all(type(seed) is int and 0 <= seed < 2**32 for seed in values) and
            len(set(values)) == 4, 'Declare four distinct uint32 ' + name + ' seeds')


def build(config, inputs):
    from .analysis import ResultAnalyzer
    from .interventions import InterventionPlanner, _allocation, _episode_inventory
    from .monitors.activation_monitor import ActivationMonitor
    from .monitors.text_monitor import TextMonitor, PROMPT, FORMAT, BASELINES
    from .monitors.reasoning_annotator import ReasoningAnnotator
    from .model import ModelRuntime
    from run import _settings

    fields(config, {'artifact_root', 'label_kind'}, 'experiment config')
    local_path(config['artifact_root'])
    require(config['label_kind'] in ('fixture', 'human'), 'Expected fixture or human provenance')
    fields(inputs, {'episode_config', 'runtime', 'runtime_check', 'activation', 'text', 'reasoning',
                    'interventions', 'analysis', 'sampling', 'patch', 'audits', 'readiness'}, 'experiment manifest inputs')
    episode_config = inputs['episode_config']
    require(type(episode_config) is dict, 'Expected episode configuration')
    built = task_handle(json_artifact(episode_config.get('tasks')))
    require(built['status'] == 'ok', 'Invalid authored study tasks')
    tasks = built['result']['tasks']
    by_split = {split: [task for task in tasks if task['split'] == split] for split in SPLITS}
    require(all(by_split.values()), 'The study must retain all four split roles', 'split_leakage')
    runtime = json_artifact(inputs['runtime'])
    require(type(runtime) is dict and type(runtime.get('fixture')) is bool and type(runtime.get('layers')) is list and
            type(runtime.get('config')) is dict and type(runtime['config'].get('hidden_size')) is int and
            runtime['fixture'] == (config['label_kind'] == 'fixture'), 'Runtime/manifest provenance mismatch')
    runtime_hash = fingerprint(runtime)
    task = by_split['training'][0]
    _settings({'schema_version': 1, 'request_id': 'validate-manifest', 'operation': 'episode', 'config': episode_config,
               'inputs': {'task_id': task['task_id'], 'condition_id': task['conditions'][0]['condition_id'], 'seed': 0,
                          'arm_id': 'baseline', 'intervention': None, 'runtime_sha256': runtime_hash}})
    ModelRuntime(episode_config['model'])  # Constructor validation only; never loads weights.
    pending, components = [], {}
    def missing(name):
        pending.append(name)
    def bind(data, name):
        require(data.get('runtime_sha256') == runtime_hash and data.get('tasks_sha256') == episode_config['tasks']['sha256'], name + ' uses another runtime/task inventory', 'hash_mismatch')
    if runtime.get('profile') != episode_config['model']['profile']:
        require(runtime['fixture'], 'Runtime/model profile mismatch', 'hash_mismatch')
        missing('fixture_runtime_is_not_an_executable_model')
    if inputs['runtime_check'] is None:
        missing('runtime_check_missing')
    else:
        components['runtime_check'] = read_runtime_check(inputs['runtime_check'], runtime)
        if components['runtime_check']['status'] != 'passed':
            missing('runtime_check_not_passed')

    if inputs['activation'] is None:
        missing('activation_monitor_unavailable')
    else:
        data = _object(json_artifact(inputs['activation']), {'config'}, 'activation monitor')
        bind(data, 'Activation monitor')
        monitor = ActivationMonitor(data['config'])
        monitor.load(inputs['activation'])
        rows, exclusions, _, _, all_ids = records(data['inputs']['features'], {'training', 'validation'})
        labels, excluded = reviewed_labels(data['inputs']['labels'], rows, all_ids, runtime['fixture'])
        selected = {split: sorted(row['record_id'] for row in rows if row['split'] == split and row['record_id'] in labels) for split in ('training', 'validation')}
        require(all(selected[split] == data['report'][split + '_ids'] for split in selected), 'Probe development cohort changed', 'hash_mismatch')
        scored = monitor.score({'features': data['inputs']['features'], 'splits': ['validation']})
        values = {row['record_id']: row['score'] for row in scored['scores']}
        require([values[key] for key in selected['validation']] == data['report']['selected']['validation_scores'] and
                [labels[key]['label'] for key in selected['validation']] == data['report']['validation_labels'], 'Probe validation evidence changed', 'hash_mismatch')
        components['activation'] = {'selected': data['report']['selected'], 'threshold': data['report']['threshold'],
            'class_counts': data['report']['class_counts'], 'warnings': data['report']['warnings'],
            'controls': data['report']['controls'], 'development_ids': selected, 'excluded': exclusions + excluded}

    if inputs['text'] is None:
        missing('text_calibration_unavailable')
    else:
        data = _object(json_artifact(inputs['text']), {'config'}, 'text calibration')
        bind(data, 'Text calibration')
        monitor = TextMonitor(data['config'])
        monitor.load(inputs['text'])
        require(monitor.calibrate(data['inputs']) == data, 'Text calibration development evidence changed', 'hash_mismatch')
        components['text'] = {'prompt': PROMPT, 'format': FORMAT, 'baselines': list(BASELINES), 'methods': data['methods']}
        if data['config']['provider'] is None:
            missing('hosted_text_comparison_missing')
        for name, method in data['methods'].items():
            if method['threshold']['status'] != 'available':
                missing(name + '_validation_classes_unavailable')

    reasoning = ReasoningAnnotator(inputs['reasoning'])
    require(reasoning.config['label_kind'] == config['label_kind'], 'Reasoning provenance mismatch')
    components['reasoning'] = reasoning._rule()
    if reasoning.config['provider'] is None:
        missing('automatic_reasoning_provider_missing')
    analysis = _object(json_artifact(inputs['analysis']), {'config'}, 'analysis rule')
    analyzer = ResultAnalyzer(analysis['config'])
    require(analysis == analyzer.rule() and analyzer.config['label_kind'] == config['label_kind'], 'Frozen analysis rule changed', 'hash_mismatch')
    components['analysis'] = analysis

    fields(inputs['interventions'], {'config', 'direction', 'steering_policy'}, 'manifest interventions')
    planner = InterventionPlanner(inputs['interventions']['config'])
    require(planner.config['label_kind'] == config['label_kind'], 'Intervention provenance mismatch')
    components['intervention_rule'] = planner._rule()
    direction_ref = inputs['interventions']['direction']
    if direction_ref is None:
        missing('training_direction_missing')
    else:
        data = _object(json_artifact(direction_ref), {'inputs', 'statistics', 'warnings', 'layer'}, 'direction artifact')
        bind(data, 'Training direction')
        require(data['inputs']['monitor'] == inputs['activation'], 'Direction must use the frozen probe layer')
        try:
            data, _ = planner._read_direction(direction_ref)
        except InputError as exc:
            if exc.code != 'direction_unavailable':
                raise
            missing('training_direction_unavailable')
        components['direction'] = {'statistics': data['statistics'], 'warnings': data['warnings'], 'layer': data['layer']}
    policy = None
    if inputs['interventions']['steering_policy'] is None:
        missing('steering_policy_missing')
    else:
        policy = planner._read_policy(inputs['interventions']['steering_policy'])
        bind(policy, 'Steering policy')
        require(policy['direction'] == direction_ref and policy['episode_config'] == episode_config and
                set(policy['inputs']['task_ids']) == {task['task_id'] for task in by_split['intervention_test']}, 'Steering policy omitted tasks or changed configuration', 'hash_mismatch')
        components['steering'] = {key: policy[key] for key in ('status', 'variants', 'schedules', 'budget', 'declared_job_count', 'job_count', 'jobs', 'histories', 'failed_checks', 'warnings')}
        if policy['status'] != 'frozen':
            missing('steering_calibration_unavailable')

    for name in ('sampling', 'patch'):
        fields(inputs[name], {'seeds', 'allocation'}, 'manifest ' + name)
        _seeds(inputs[name]['seeds'], name)
        _allocation(inputs[name]['allocation'], runtime['fixture'], 'max_output_tokens')
    final_seeds = inputs['sampling']['seeds'] + inputs['patch']['seeds'] + (policy['inputs']['seeds'] if policy else [])
    require(len(final_seeds) == len(set(final_seeds)), 'Sampling, patch and steering require distinct final seeds')
    if policy:
        previous = planner._read_steering_plan(planner._read_selection(policy['selection'])['inputs']['plan'])
        while previous:
            require(not set(final_seeds) & set(previous['inputs']['seeds']), 'Final seeds overlap intervention development', 'split_leakage')
            previous = planner._read_steering_plan(previous['inputs']['previous']) if previous['inputs']['previous'] else None
    frozen_rule = rule()
    variant = {'arm_id': 'baseline', 'control': 'baseline', 'schedule': None, 'coefficient': None, 'intervention': None, 'prompt_reminder': None}
    final_tasks = by_split['detection_test'] + by_split['intervention_test']
    histories, jobs, budget = _episode_inventory(episode_config | {'max_tool_calls': 1}, final_tasks, inputs['sampling']['seeds'],
        [variant], runtime_hash, fingerprint({'inputs': inputs, 'rule': frozen_rule}), inputs['sampling']['allocation'], runtime['fixture'])
    fields(inputs['audits'], {'behavior_seed', 'behavior_uniform_size'}, 'manifest audit rules')
    require(type(inputs['audits']['behavior_seed']) is int and 0 <= inputs['audits']['behavior_seed'] < 2**32 and
            type(inputs['audits']['behavior_uniform_size']) is int and 1 <= inputs['audits']['behavior_uniform_size'] <= 50, 'Invalid behavioral audit sampling rule')
    fields(inputs['readiness'], {'pilot', 'sandbox', 'evaluator', 'behavior_audit', 'reasoning_audit'}, 'manifest readiness evidence')
    for name, ref in sorted(inputs['readiness'].items()):
        if ref is None:
            missing(name + '_evidence_missing')
        else:
            json_artifact(ref)  # Retain the exact report; host/human acceptance is a separate check.
            missing(name + '_evidence_requires_acceptance')
    if runtime['fixture']:
        missing('fixture_is_not_research_acceptance')
    return {'schema_version': 1, 'status': 'incomplete' if pending else 'frozen', 'config': config, 'inputs': inputs, 'rule': frozen_rule,
        'fixture': runtime['fixture'], 'runtime_sha256': runtime_hash, 'tasks_sha256': episode_config['tasks']['sha256'],
        'tasks': [{key: task[key] for key in ('task_id', 'clone_group_id', 'split', 'task_sha256')} for task in tasks],
        'split_counts': {split: len(rows) for split, rows in by_split.items()}, 'components': components,
        'sampling': {'histories': histories, 'jobs': jobs, 'job_count': len(jobs), 'budget': budget},
        'patch': {'stage': 'retrospective_intervention_test', 'settings': inputs['patch'], 'generation': episode_config['generation'],
                  'job_count': None, 'status': 'eligibility_determined_after_frozen_baseline_collection'},
        'pending': pending, 'execution_enabled': False, 'new_model_calls': 0, 'new_provider_calls': 0, 'monitor_fits': 0,
        'limitations': ['Dry validation does not approve spending or held-out execution',
            'Runtime reports describe one prefix; fixture, host and independent human acceptance remain distinct',
            'Shared development manifests may list held-out rows; their episodes/reviews are not opened during freeze']}


def read_manifest(reference):
    data = json_artifact(reference)
    require(type(data) is dict and type(data.get('inputs')) is dict and type(data.get('config')) is dict and 'provenance' in data, 'Invalid experiment manifest')
    require({key: value for key, value in data.items() if key != 'provenance'} == build(data['config'], data['inputs']), 'Frozen experiment manifest changed', 'hash_mismatch')
    expected = data['rule']['sources']
    fields(data['provenance'], {'sources', 'git_head', 'git_status', 'git_diff', 'note'}, 'manifest provenance')
    require(type(data['provenance']['sources']) is dict, 'Invalid source snapshot inventory')
    require(set(data['provenance']['sources']) == set(expected), 'Source snapshot inventory changed')
    for name, ref in data['provenance']['sources'].items():
        require(hashlib.sha256(read_artifact(ref, 'bytes', 16777216)).hexdigest() == expected[name], 'Source snapshot changed', 'hash_mismatch')
    read_artifact(data['provenance']['git_diff'], 'bytes', 16777216)
    return data


def review_template(reference, manifest):
    policy = json_artifact(manifest['inputs']['interventions']['steering_policy']) if manifest['inputs']['interventions']['steering_policy'] else None
    return {'schema_version': 1, 'manifest': reference, 'reviewer_id': None,
        'reviewer_kind': 'fixture' if manifest['fixture'] else 'human',
        'rubric': REVIEW_ITEMS, 'checks': {name: {'accepted': None, 'rationale': ''} for name in REVIEW_ITEMS},
        'approved_allocations': {'sampling': manifest['inputs']['sampling']['allocation'], 'patch': manifest['inputs']['patch']['allocation'],
                                 'steering': policy['inputs']['allocation'] if policy else None},
        'notes': ''}


def acceptance(inputs):
    fields(inputs, {'manifest', 'review'}, 'experiment acceptance inputs')
    manifest = read_manifest(inputs['manifest'])
    review = json_artifact(inputs['review'])
    template = review_template(inputs['manifest'], manifest)
    fields(review, set(template), 'operator review')
    require(type(review['schema_version']) is int and review['schema_version'] == 1 and review['manifest'] == inputs['manifest'] and
            review['rubric'] == REVIEW_ITEMS and review['approved_allocations'] == template['approved_allocations'], 'Review changed the manifest, rubric or allocations', 'hash_mismatch')
    require(review['reviewer_kind'] == template['reviewer_kind'], 'Fixture and human reviews cannot be interchanged')
    if review['reviewer_id'] is not None:
        identifier(review['reviewer_id'], 'reviewer_id')
    fields(review['checks'], set(REVIEW_ITEMS), 'reviewed requirements')
    for item in review['checks'].values():
        fields(item, {'accepted', 'rationale'}, 'review decision')
        require(item['accepted'] is None or type(item['accepted']) is bool, 'Review decisions must be boolean or null')
        require(type(item['rationale']) is str and len(item['rationale']) <= 4000 and
                (item['accepted'] is None or bool(item['rationale'].strip())), 'Explain each supplied review decision')
    require(type(review['notes']) is str and len(review['notes']) <= 8000, 'Invalid review notes')
    pending = [name for name in manifest['pending'] if not name.endswith('_evidence_requires_acceptance') and name != 'fixture_is_not_research_acceptance']
    if review['reviewer_id'] is None or any(item['accepted'] is None for item in review['checks'].values()):
        pending.append('operator_review_incomplete')
    if any(item['accepted'] is False for item in review['checks'].values()):
        pending.append('operator_review_declined')
    runtime = json_artifact(manifest['inputs']['runtime'])
    require(runtime['profile'] == ('tiny-gpt-oss-cpu' if manifest['fixture'] else 'gpt-oss-20b-mxfp4'), 'Acceptance requires a compatible executable runtime')
    host = None
    for name, reference in manifest['inputs']['readiness'].items():
        if reference is None:
            continue
        evidence = _object(json_artifact(reference), set(), name + ' readiness report')
        require(evidence.get('fixture') in (None, manifest['fixture']), 'Readiness report mixes fixture and research evidence')
        if name == 'sandbox':
            fields(evidence, {'request', 'response'}, 'sandbox preflight evidence')
            request, response = json_artifact(evidence['request']), json_artifact(evidence['response'])
            validate_request(request, {'preflight'})
            require(request['config'] == manifest['inputs']['episode_config']['sandbox'] and request['inputs'] == {} and
                    response.get('status') == 'ok' and response.get('request_id') == request['request_id'], 'Sandbox preflight/configuration changed', 'hash_mismatch')
            host = _object(response.get('result'), {'server_version', 'image_id', 'image', 'security_options'}, 'sandbox metadata')
            require(host['image'] == request['config']['image'] and type(host['server_version']) is str and bool(host['server_version']) and
                    type(host['image_id']) is str and host['image_id'].startswith('sha256:') and len(host['image_id']) == 71 and
                    all(char in '0123456789abcdef' for char in host['image_id'][7:]) and
                    type(host['security_options']) is list and all(type(option) is str for option in host['security_options']) and
                    any('seccomp' in option for option in host['security_options']), 'Invalid accepted sandbox metadata')
    if host is None:
        pending.append('sandbox_preflight_missing')
    return {'schema_version': 1, 'inputs': inputs, 'rule': rule(), 'fixture': manifest['fixture'],
        'status': 'accepted' if not pending else 'not_ready', 'pending': sorted(set(pending)),
        'runtime_sha256': manifest['runtime_sha256'], 'tasks_sha256': manifest['tasks_sha256'],
        'sandbox_preflight': host, 'approved_allocations': template['approved_allocations'],
        'enabled_phases': ['sampling', 'steering'] if not pending else [],
        'interpretation': 'Operator attestations bind the supplied evidence; this file does not authenticate the reviewer or independently certify scientific validity',
        'limits': 'Fixed frozen jobs only, one attempt per canonical job; allocations are declared computation bounds, not account-wide billing limits',
        'patch_status': 'requires a checked patch binding to the terminal intervention-test baseline collection'}


def read_acceptance(reference):
    data = _object(json_artifact(reference), {'inputs'}, 'acceptance record')
    require(data == acceptance(data['inputs']), 'Acceptance record or its original evidence changed', 'hash_mismatch')
    return data


def final_job(proof):
    require(type(proof) is dict, 'Expected a final job proof object')
    fields(proof, {'acceptance', 'phase', 'job_id'} | (set(proof) & {'plan'}), 'final job proof')
    require(proof['phase'] in ('sampling', 'steering', 'patch'), 'Unknown final experiment phase')
    require(('plan' in proof) == (proof['phase'] == 'patch'), 'Final patch jobs need their own frozen cohort binding')
    identifier(proof['job_id'], 'final job_id')
    decision = read_acceptance(proof['acceptance'])
    require(decision['status'] == 'accepted' and (proof['phase'] == 'patch' or proof['phase'] in decision['enabled_phases']),
            'Final execution has not been accepted', 'acceptance_required')
    manifest = json_artifact(decision['inputs']['manifest'])  # Canonically checked by read_acceptance.
    plan = manifest['sampling'] if proof['phase'] == 'sampling' else manifest['components']['steering']
    if proof['phase'] == 'patch':
        plan = read_final_patch(proof['plan'])
        require(plan['binding']['acceptance'] == proof['acceptance'], 'Patch binding changed its acceptance', 'hash_mismatch')
    job = next((job for job in plan['jobs'] if job['job_id'] == proof['job_id']), None)
    require(job is not None, 'Job is outside the accepted inventory', 'hash_mismatch')
    episode = job['episode']
    return episode | {'inputs': episode['inputs'] | {'final': proof}}, decision


def final_patch_plan(inputs):
    """Bind existing patch instructions to all terminal accepted baseline slots."""
    from .analysis import ResultAnalyzer
    from .interventions import InterventionPlanner
    fields(inputs, {'acceptance', 'collection', 'instructions'}, 'final patch binding inputs')
    decision = read_acceptance(inputs['acceptance'])
    require(decision['status'] == 'accepted', 'Final patching has not been accepted', 'acceptance_required')
    manifest_ref = decision['inputs']['manifest']
    manifest = json_artifact(manifest_ref)
    analyzer = ResultAnalyzer(manifest['components']['analysis']['config'])
    with tempfile.TemporaryDirectory(prefix='patch-collection-', dir=local_path(manifest['config']['artifact_root'])) as scratch:
        collection, _ = analyzer._final_collection(inputs['collection'], Path(scratch), split='intervention_test')
    require(collection['manifest'] == manifest_ref and collection['final'] == {'acceptance': inputs['acceptance'], 'split': 'intervention_test'},
            'Patch collection changed its accepted experiment', 'hash_mismatch')
    slots = json_artifact(collection['slots'])['rows']
    episodes = [json_artifact(row['evidence']['episode']) if row.get('evidence', {}).get('episode') is not None else None for row in slots]
    require(all(episode is not None and (episode.get('status') == 'complete' or
                any(episode.get(key) for key in ('error', 'failure_type', 'cleanup_failure'))) for episode in episodes),
            'Finish every baseline attempt before binding final patch pairs; missing/incomplete slots remain pending', 'collection_incomplete')
    planner = InterventionPlanner(manifest['inputs']['interventions']['config'])
    patches = planner._read_patch_plan(inputs['instructions'])
    frozen = manifest['inputs']
    require(patches['inputs'] == {'direction': frozen['interventions']['direction'], 'features': collection['features'],
            'labels': collection['labels'], 'captures': patches['inputs']['captures'], 'split': 'intervention_test',
            'seeds': frozen['patch']['seeds'], 'generation': frozen['episode_config']['generation'], 'max_jobs': frozen['patch']['allocation']['max_jobs']},
            'Final patch instructions changed the frozen settings or collection', 'hash_mismatch')
    completed, *_ = records(collection['features'], {'intervention_test'}, require_capture=False)
    captures = json_artifact(patches['inputs']['captures'])['records']
    require({row['record_id'] for row in captures} == {row['record_id'] for row in completed},
            'Account for pre-action capture success or failure for every completed baseline call', 'capture_coverage')
    plan = planner._patch_episode_plan({'instructions': inputs['instructions'], 'episode_config': frozen['episode_config'],
        'allocation': frozen['patch']['allocation']}, final_manifest=manifest_ref)
    return plan | {'binding': inputs, 'manifest': manifest_ref, 'collection': inputs['collection'], 'final_acceptance_verified': True,
        'baseline_coverage': {'planned_slots': collection['planned_slots'], 'slots': collection['slots'],
                              'table': collection['table'], 'summaries': collection['summaries']}}


def read_final_patch(reference):
    data = _object(json_artifact(reference), {'binding'}, 'final patch binding')
    require(data == final_patch_plan(data['binding']), 'Final patch inventory or its binding changed', 'hash_mismatch')
    return data


def final_intervention_plan(reference, phase, plan_reference):
    """Use every accepted continuation slot with the same proof as final_job."""
    require(phase in ('patch', 'steering'), 'Expected a final intervention phase')
    decision = read_acceptance(reference)
    require(decision['status'] == 'accepted', 'Final intervention analysis requires acceptance', 'acceptance_required')
    manifest_ref = decision['inputs']['manifest']
    manifest = json_artifact(manifest_ref)
    if phase == 'patch':
        plan = read_final_patch(plan_reference)
        require(plan['binding']['acceptance'] == reference and plan['manifest'] == manifest_ref,
                'Final patch plan uses another accepted experiment', 'hash_mismatch')
    else:
        require(plan_reference == manifest_ref and phase in decision['enabled_phases'],
                'Final steering requires its exact accepted manifest', 'hash_mismatch')
        plan = manifest['components']['steering'] | {'experiment': 'steering', 'stage': 'final', 'split': 'intervention_test',
            'fixture': manifest['fixture'], 'runtime_sha256': manifest['runtime_sha256'], 'tasks_sha256': manifest['tasks_sha256'],
            'inputs': {'episode_config': manifest['inputs']['episode_config']},
            'policy': manifest['inputs']['interventions']['steering_policy']}
    jobs = []
    for job in plan['jobs']:
        proof = {'acceptance': reference, 'phase': phase, 'job_id': job['job_id']}
        if phase == 'patch':
            proof['plan'] = plan_reference
        episode = job['episode']
        jobs.append(job | {'episode': episode | {'inputs': episode['inputs'] | {'final': proof}}})
    tasks = [task for task in manifest['tasks'] if task['split'] == 'intervention_test']
    return plan | {'jobs': jobs, 'manifest': manifest_ref, 'analysis': manifest['inputs']['analysis'],
        'clone_groups': sorted({task['clone_group_id'] for task in tasks}), 'task_ids': sorted(task['task_id'] for task in tasks)}


def final_sampling_plan(reference, split):
    """Select every frozen baseline slot in one accepted held-out split."""
    require(split in ('detection_test', 'intervention_test'), 'Final sampling analysis requires one held-out split', 'split_leakage')
    decision = read_acceptance(reference)
    require(decision['status'] == 'accepted' and 'sampling' in decision['enabled_phases'], 'Final sampling has not been accepted', 'acceptance_required')
    manifest = json_artifact(decision['inputs']['manifest'])
    task_ids = {task['task_id'] for task in manifest['tasks'] if task['split'] == split}
    jobs = []
    for job in manifest['sampling']['jobs']:
        if job['task_id'] in task_ids:
            proof = {'acceptance': reference, 'phase': 'sampling', 'job_id': job['job_id']}
            episode = job['episode']
            jobs.append(job | {'episode': episode | {'inputs': episode['inputs'] | {'final': proof}}})
    config = manifest['inputs']['episode_config'] | {'max_tool_calls': 1}
    allocation = manifest['inputs']['sampling']['allocation']
    from .interventions import _allocation, _episode_budget
    budget = _episode_budget(config, len(jobs), _allocation(allocation, manifest['fixture'], 'max_output_tokens'), allocation)
    return {'schema_version': 1, 'experiment': 'sampling', 'stage': 'final', 'split': split,
        'fixture': manifest['fixture'], 'runtime_sha256': manifest['runtime_sha256'], 'tasks_sha256': manifest['tasks_sha256'],
        'inputs': {'episode_config': config, 'runtime': manifest['inputs']['runtime'], 'allocation': allocation},
        'jobs': jobs, 'job_count': len(jobs), 'histories': [row for row in manifest['sampling']['histories'] if row['task_id'] in task_ids],
        'budget': budget, 'manifest_budget': manifest['sampling']['budget'],
        'manifest': decision['inputs']['manifest'], 'analysis': manifest['inputs']['analysis']}


def handle(request):
    validate_request(request, {'experiment.freeze', 'experiment.load', 'experiment.review', 'experiment.accept', 'experiment.job', 'experiment.patch'})
    if request['operation'] == 'experiment.load':
        fields(request['inputs'], {'manifest'}, 'load experiment inputs')
        data = read_manifest(request['inputs']['manifest'])
        require(request['config'] == data['config'], 'Experiment configuration changed')
        return success(request, {'manifest': request['inputs']['manifest'], 'status': data['status'], 'pending': data['pending'],
                                'execution_enabled': False, 'sampling_jobs': data['sampling']['job_count']}, [])
    if request['operation'] == 'experiment.review':
        fields(request['inputs'], {'manifest'}, 'review template inputs')
        manifest = read_manifest(request['inputs']['manifest'])
        require(request['config'] == manifest['config'], 'Experiment configuration changed')
        data, name = review_template(request['inputs']['manifest'], manifest), 'review-template'
    elif request['operation'] == 'experiment.accept':
        data, name = acceptance(request['inputs']), 'acceptance'
        require(request['config'] == json_artifact(request['inputs']['manifest'])['config'], 'Experiment configuration changed')
    elif request['operation'] == 'experiment.job':
        data, decision = final_job(request['inputs'])
        require(request['config'] == json_artifact(decision['inputs']['manifest'])['config'], 'Experiment configuration changed')
        name = 'episode-request'
    elif request['operation'] == 'experiment.patch':
        data, name = final_patch_plan(request['inputs']), 'patch-plan'
        require(request['config'] == json_artifact(data['manifest'])['config'], 'Experiment configuration changed')
    else:
        data, name = build(request['config'], request['inputs']), 'manifest'
    directory = local_path(request['config']['artifact_root']) / request['request_id']
    require(not directory.exists(), 'Preserve the previous manifest; use a new request ID', 'attempt_exists')
    directory.mkdir(parents=True)
    atomic_json(directory / 'request.json', request)
    if name != 'manifest':
        atomic_json(directory / (name + '.json'), data)
        reference = artifact_ref(directory / (name + '.json'), 'json')
        return success(request, {'artifact': reference, 'status': data.get('status', 'prepared'),
                                'pending': data.get('pending', []), 'jobs_executed': 0}, [reference])
    sources = {}
    root = Path(__file__).parent.parent
    for name in data['rule']['sources']:
        path = directory / 'source' / name
        atomic_bytes(path, (root / name).read_bytes())
        sources[name] = artifact_ref(path, 'bytes')
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True)
        require(result.returncode == 0, 'Cannot record Git provenance: ' + result.stderr.decode(errors='replace')[:1000], 'git_error')
        return result.stdout
    atomic_bytes(directory / 'working-tree.diff', git('diff', 'HEAD', '--no-ext-diff', '--no-textconv'))
    data['provenance'] = {'sources': sources, 'git_head': git('rev-parse', 'HEAD').decode().strip(),
        'git_status': git('status', '--short').decode(), 'git_diff': artifact_ref(directory / 'working-tree.diff', 'bytes'),
        'note': 'Source snapshots include uncommitted/untracked Python files; no commit is created'}
    atomic_json(directory / 'manifest.json', data)
    reference = artifact_ref(directory / 'manifest.json', 'json')
    return success(request, {'manifest': reference, 'status': data['status'], 'pending': data['pending'],
                            'execution_enabled': False, 'sampling_jobs': data['sampling']['job_count']}, [reference])
