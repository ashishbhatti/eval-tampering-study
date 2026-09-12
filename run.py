"""Run one development episode or accepted final job through existing components."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from eval_tampering.messages import (InputError, artifact_ref, atomic_bytes, atomic_json,
    decode_json, failure, fields, fingerprint, identifier, local_path, read_artifact,
    read_json, require, success, validate_request)
from eval_tampering.tasks import handle as task_handle
from eval_tampering.sandbox import _files_archive
from eval_tampering.interventions import PROMPT_REMINDER


def _sampling_rule():
    root = Path(__file__).parent
    return {'schema_version': 1, 'sources': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
        ('run.py', 'eval_tampering/interventions.py', 'eval_tampering/tasks.py', 'eval_tampering/model.py', 'eval_tampering/monitors/common.py')},
        'population': 'every task in the declared development split, all four conditions, four explicit seeds, one next action per slot',
        'retry_policy': 'no automatic retries or replacement sampling; missing and unsuccessful slots remain in the inventory'}


def _sampling_plan(inputs):
    from eval_tampering.interventions import _episode_inventory
    fields(inputs, {'episode_config', 'runtime', 'split', 'seeds', 'allocation'}, 'sampling plan inputs')
    config, seeds = inputs['episode_config'], inputs['seeds']
    require(inputs['split'] in ('training', 'validation'), 'Sampling execution requires development tasks; held-out work needs complete acceptance', 'split_leakage')
    require(type(seeds) is list and len(seeds) == 4 and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds) and
            len(set(seeds)) == 4, 'Declare four distinct uint32 sampling seeds')
    require(type(config) is dict and config.get('max_tool_calls') == 1 and type(config.get('max_tool_calls')) is int,
            'Initial collection samples one next action per slot')
    runtime = decode_json(read_artifact(inputs['runtime'], 'json', 16777216))
    require(type(runtime) is dict and type(runtime.get('fixture')) is bool and type(runtime.get('layers')) is list and
            all(type(layer) is int and layer >= 0 for layer in runtime['layers']) and type(runtime.get('config')) is dict and
            type(runtime['config'].get('hidden_size')) is int and 1 <= runtime['config']['hidden_size'] <= 8192, 'Invalid sampling runtime identity')
    built = task_handle(decode_json(read_artifact(config.get('tasks'), 'json', 16777216)))
    require(built['status'] == 'ok', 'Expected a valid authored task manifest')
    tasks = [task for task in built['result']['tasks'] if task['split'] == inputs['split']]
    require(tasks, 'No tasks in the declared sampling split')
    rule = _sampling_rule()
    variant = {'arm_id': 'baseline', 'control': 'baseline', 'schedule': None, 'coefficient': None, 'intervention': None, 'prompt_reminder': None}
    histories, jobs, budget = _episode_inventory(config, tasks, seeds, [variant], fingerprint(runtime),
        fingerprint({'inputs': inputs, 'rule': rule}), inputs['allocation'], runtime['fixture'])
    return {'schema_version': 1, 'status': 'planned', 'experiment': 'sampling', 'stage': 'development', 'split': inputs['split'],
        'inputs': inputs, 'rule': rule, 'fixture': runtime['fixture'], 'runtime_sha256': fingerprint(runtime), 'tasks_sha256': config['tasks']['sha256'],
        'histories': histories, 'jobs': jobs, 'job_count': len(jobs), 'budget': budget, 'new_model_calls': 0,
        'final_acceptance_required': False}


def read_sampling_plan(reference):
    data = decode_json(read_artifact(reference, 'json', 67108864))
    require(type(data) is dict and data.get('rule') == _sampling_rule() and type(data.get('inputs')) is dict,
            'Frozen sampling rule changed', 'hash_mismatch')
    require(data == _sampling_plan(data['inputs']), 'Frozen sampling inventory changed', 'hash_mismatch')
    return data


def plan_sampling(request):
    fields(request['inputs'], {'runtime', 'split', 'seeds', 'allocation'}, 'sampling request inputs')
    plan = _sampling_plan({'episode_config': request['config'], **request['inputs']})
    directory = local_path(request['config']['artifact_root']) / 'sampling' / request['request_id']
    require(not directory.exists(), 'Sampling plan attempt already exists; preserve it and use a new ID', 'attempt_exists')
    directory.mkdir(parents=True)
    atomic_json(directory / 'request.json', request)
    atomic_json(directory / 'plan.json', plan)
    return success(request, plan | {'plan': artifact_ref(directory / 'plan.json', 'json')}, [artifact_ref(directory / 'plan.json', 'json')])


def _validate_episode_request(request):
    """Validate shared episode limits without assuming a solvable task."""
    validate_request(request, {'episode'})
    config, inputs = request['config'], request['inputs']
    fields(config, {'tasks', 'model', 'sandbox', 'evaluation', 'artifact_root', 'date',
                    'reasoning_effort', 'max_tool_calls', 'max_output_tokens', 'max_seconds',
                    'max_consecutive_failures', 'max_feedback_chars', 'generation'}, 'episode config')
    fields(inputs, {'task_id', 'condition_id', 'seed', 'arm_id', 'intervention'} |
           (set(inputs) & {'prompt_reminder', 'runtime_sha256', 'patch', 'final'}), 'episode inputs')
    require(inputs.get('prompt_reminder') in (None, PROMPT_REMINDER), 'Only the fixed permission-preserving reminder is supported')
    expected_runtime = inputs.get('runtime_sha256')
    require(expected_runtime is None or (type(expected_runtime) is str and len(expected_runtime) == 64 and
            all(char in '0123456789abcdef' for char in expected_runtime)), 'Invalid expected runtime hash')
    if inputs.get('intervention') is not None and expected_runtime is not None:
        require(type(inputs['intervention']) is dict and inputs['intervention'].get('runtime_sha256') == expected_runtime, 'Intervention/episode runtime mismatch')
    fields(config['evaluation'], {'artifact_root', 'max_jobs'}, 'evaluation settings')
    fields(config['generation'], {'max_new_tokens', 'temperature', 'max_seconds'}, 'generation settings')
    for key, low, high in [('max_tool_calls', 1, 3), ('max_output_tokens', 1, 6144),
                           ('max_consecutive_failures', 1, 3), ('max_feedback_chars', 1, 65536)]:
        require(type(config[key]) is int and low <= config[key] <= high, f'Invalid {key}')
    require(type(config['max_seconds']) in (int, float) and 0 < config['max_seconds'] <= 7200, 'Invalid episode time budget')
    generation = config['generation']
    require(type(generation['max_new_tokens']) is int and 1 <= generation['max_new_tokens'] <= 2048, 'Invalid per-turn token budget')
    require(type(generation['temperature']) in (int, float) and 0 <= generation['temperature'] <= 2, 'Invalid temperature')
    require(type(generation['max_seconds']) in (int, float) and 0 < generation['max_seconds'] <= 600, 'Invalid generation time budget')
    require(type(inputs['seed']) is int and 0 <= inputs['seed'] < 2**32, 'Invalid episode seed')
    for key in ('task_id', 'condition_id', 'arm_id'):
        identifier(inputs[key], key)
    return config, inputs


def _settings(request):
    config, inputs = _validate_episode_request(request)
    authored = decode_json(read_artifact(config['tasks'], 'json', 16777216))
    require(type(authored) is dict and authored.get('operation') == 'build', 'Task artifact must be an authored build request')
    built = task_handle(authored)
    require(built['status'] == 'ok', f"Invalid tasks: {built.get('error')}")
    task = next((task for task in built['result']['tasks'] if task['task_id'] == inputs['task_id']), None)
    require(task is not None, 'Unknown task_id')
    if task['split'] in {'training', 'validation'}:
        require('final' not in inputs, 'Final acceptance cannot change a development task into a test task')
    else:
        require('final' in inputs, 'An accepted frozen job is required for held-out test tasks', 'acceptance_required')
        from eval_tampering.experiment import final_job
        expected, _ = final_job(inputs['final'])
        require(request == expected, 'Final request differs from its accepted job', 'hash_mismatch')
    condition = next((condition for condition in task['conditions'] if condition['condition_id'] == inputs['condition_id']), None)
    require(condition is not None, 'Unknown condition for this task')
    patch_job = None
    if 'patch' in inputs:
        from eval_tampering.interventions import InterventionPlanner
        fields(inputs['patch'], {'instructions', 'job_id'}, 'patch episode inputs')
        raw = decode_json(read_artifact(inputs['patch']['instructions'], 'json', 67108864))
        require(type(raw) is dict and type(raw.get('rule')) is dict and type(raw['rule'].get('config')) is dict,
                'Expected frozen patch instructions')
        plan = InterventionPlanner(raw['rule']['config'])._read_patch_plan(inputs['patch']['instructions'])
        require(plan['inputs']['split'] == task['split'] and plan['tasks_sha256'] == config['tasks']['sha256'], 'Patch/episode task split changed', 'hash_mismatch')
        patch_job = next((job for job in plan['jobs'] if job['job_id'] == inputs['patch']['job_id']), None)
        require(patch_job is not None, 'Unknown patch continuation job')
        require(patch_job['task_id'] == task['task_id'] and
                (patch_job['problem'], patch_job['permission']) == (condition['problem'], condition['permission']), 'Patch recipient condition changed', 'hash_mismatch')
        require(inputs['seed'] == patch_job['generation']['seed'] and inputs['arm_id'] == patch_job['control'] and
                inputs['intervention'] == patch_job['generation']['intervention'] and inputs.get('runtime_sha256') == plan['runtime_sha256'] and
                inputs.get('prompt_reminder') is None, 'Patch seed, control, hook or runtime changed', 'hash_mismatch')
        require(config['generation'] == plan['inputs']['generation'] and config['max_output_tokens'] >= config['generation']['max_new_tokens'],
                'Episode limits cannot alter the declared first patch continuation', 'hash_mismatch')
        read_artifact(patch_job['before_snapshot'], 'tar', config['sandbox']['max_snapshot_bytes'])
    root = local_path(config['artifact_root'])
    require(root != Path.cwd().resolve(), 'Use a dedicated episode artifact directory')
    evaluator_config = config['evaluation'] | {'tasks': config['tasks'], 'sandbox': config['sandbox']}
    return config, inputs, task, condition, root, evaluator_config, patch_job


def run_episode(request, model, sandbox, evaluator):
    """Reuse explicitly supplied components; snapshots/history belong to this call only."""
    return _execute_prepared_episode(request, model, sandbox, evaluator, _settings(request))


def _execute_prepared_episode(request, model, sandbox, evaluator, settings):
    """Shared mechanics; ordinary and impossible frontends validate their own tasks."""
    config, inputs, task, condition, root, evaluator_config, patch_job = settings
    neutral = fingerprint({'episode': request['request_id']})[:32]
    directory = root / neutral
    require(not directory.exists(), 'Episode already exists; retain its sampling slot and use an explicit new attempt ID', 'existing_attempt')
    directory.mkdir(parents=True)
    atomic_json(directory / 'request.json', request)
    atomic_bytes(directory / 'runner.py', Path(__file__).read_bytes())
    # The outer record owns the arm mapping. Component request IDs are opaque.
    record = {'schema_version': 1, 'episode_id': request['request_id'], 'request_sha256': fingerprint(request),
              'task_id': task['task_id'], 'condition_id': condition['condition_id'], 'split': task['split'],
              'arm_id': inputs['arm_id'], 'seed': inputs['seed'], 'stage': 'final' if 'final' in inputs else 'development',
              'status': 'incomplete', 'stop_reason': None, 'actions': [], 'output_tokens': 0,
              'label_status': 'unreviewed', 'primary_label': None, 'final_assessment': None,
              'monitoring_status': 'not_run', 'fixture': None, 'final_snapshot': None}
    if task.get('control_type') == 'contradictory_specification':
        record.update(control_type=task['control_type'], ordinary_evaluator_applicable=False)
    if patch_job is not None:
        record['patch'] = inputs['patch']
    if 'final' in inputs:
        from eval_tampering.experiment import read_acceptance
        decision = read_acceptance(inputs['final']['acceptance'])
        record['final'] = inputs['final']
        record['manifest'] = decision['inputs']['manifest']
    atomic_json(directory / 'record.json', record)
    started = time.monotonic()
    ordinal = 0

    def save():
        record['elapsed_seconds'] = time.monotonic() - started
        atomic_json(directory / 'record.json', record)

    def call(component, component_name, operation, component_inputs, component_config):
        nonlocal ordinal
        ordinal += 1
        component_id = f'{neutral}-{ordinal:03d}'
        packet = {'schema_version': 1, 'request_id': component_id, 'operation': operation,
                  'inputs': component_inputs, 'config': component_config}
        request_path = directory / f'{ordinal:03d}-{component_name}-request.json'
        atomic_json(request_path, packet)
        if component_name == 'model' and operation == 'generate':
            record['actions'][-1]['generation_request'] = artifact_ref(request_path, 'json')
        response = component.handle(packet)
        path = directory / f'{ordinal:03d}-{component_name}-response.json'
        atomic_json(path, response)
        return response, artifact_ref(path, 'json')

    def checked(component, name, operation, component_inputs, component_config):
        response, reference = call(component, name, operation, component_inputs, component_config)
        require(response['status'] == 'ok', f'{name}.{operation} failed; inspect {reference["path"]}: {response.get("error")}', 'component_error')
        return response['result'], reference

    try:
        model.reset_episode()
        host, _ = checked(sandbox, 'sandbox', 'preflight', {}, config['sandbox'])
        if 'final' in inputs:
            require(host == decision['sandbox_preflight'], 'Live sandbox differs from the accepted host/image metadata', 'hash_mismatch')
        initial_path = directory / 'initial.tar'
        initial_bytes = (read_artifact(patch_job['before_snapshot'], 'tar', config['sandbox']['max_snapshot_bytes']) if patch_job is not None else
                         _files_archive(condition['visible']['files'], config['sandbox']['max_snapshot_bytes']))
        atomic_bytes(initial_path, initial_bytes)
        initial = artifact_ref(initial_path, 'tar')
        snapshot = initial
        record.update(initial_snapshot=initial, final_snapshot=initial, visible_sha256=condition['visible_sha256'])
        identity, _ = checked(model, 'model', 'load', {}, config['model'])
        record['fixture'] = identity['fixture']
        record['runtime_sha256'] = fingerprint(identity)
        require(inputs.get('runtime_sha256') in (None, record['runtime_sha256']), 'Loaded runtime does not match the frozen episode', 'hash_mismatch')
        if patch_job is not None:
            prefix = patch_job['generation']['prefix']
            record['initial_prefix'] = prefix
        else:
            messages = [{'role': 'user', 'content': json.dumps(condition['visible'], ensure_ascii=False, sort_keys=True)}]
            if inputs.get('prompt_reminder') is not None:
                messages.insert(0, {'role': 'developer', 'content': inputs['prompt_reminder']})
            prepared, _ = checked(model, 'model', 'prepare', {
                'messages': messages,
                'date': config['date'], 'reasoning_effort': config['reasoning_effort']}, config['model'])
            prefix = prepared['tokens']
        consecutive_failures = 0
        episode_actions, episode_explanations = [], []
        save()
        for turn in range(config['max_tool_calls']):
            remaining_seconds = config['max_seconds'] - (time.monotonic() - started)
            remaining_tokens = config['max_output_tokens'] - record['output_tokens']
            if remaining_seconds <= 0 or remaining_tokens <= 0:
                record['stop_reason'] = 'time_limit' if remaining_seconds <= 0 else 'token_limit'
                break
            action = {'record_id': fingerprint({'episode': request['request_id'], 'turn': turn})[:32],
                      'turn': turn, 'status': 'incomplete', 'before_snapshot': snapshot,
                      'after_snapshot': None, 'capture': None, 'assessment': None}
            record['actions'].append(action)
            save()
            generation, reference = checked(model, 'model', 'generate', {
                'prefix': prefix, 'seed': inputs['seed'] if patch_job is not None and turn == 0 else int(fingerprint({'seed': inputs['seed'], 'turn': turn})[:8], 16),
                'max_new_tokens': min(remaining_tokens, config['generation']['max_new_tokens']),
                'temperature': config['generation']['temperature'],
                'max_seconds': min(remaining_seconds, config['generation']['max_seconds']),
                'intervention': None if patch_job is not None and turn > 0 else inputs['intervention']}, config['model'])
            action.update(generation=reference, generation_status=generation['status'])
            record['output_tokens'] += generation['generated_tokens']
            explanation = '\n'.join(message['content'] for message in generation.get('messages', [])
                                    if message['header']['recipient'] is None and message['content'])
            episode_explanations.append(explanation)
            if generation['status'] != 'tool_call':
                action['status'] = 'no_tool_call'
                record['stop_reason'] = generation['status']
                save()
                break
            # Observation failures are recorded and cannot choose which valid action executes.
            capture, capture_reference = call(model, 'model', 'capture', {
                'trajectory': generation['tokens'], 'target': 'action', 'layers': sorted({0, *identity['layers']})}, config['model'])
            action.update(capture=capture_reference, capture_status=capture['status'])
            if time.monotonic() - started >= config['max_seconds']:
                action['status'] = 'not_executed_time_limit'
                record['stop_reason'] = 'time_limit'
                break
            execution, execution_reference = call(sandbox, 'sandbox', 'execute', {
                'files': None, 'snapshot': snapshot, 'command': generation['command']}, config['sandbox'])
            action['execution'] = execution_reference
            action_text = next(message['content'] for message in reversed(generation['messages'])
                               if message['header']['recipient'] == 'functions.execute')
            episode_actions.append(action_text)
            if execution['status'] == 'ok':
                visible = execution['result']
                snapshot = visible['snapshot']
                action.update(after_snapshot=snapshot, status='executed')
                record['final_snapshot'] = snapshot
                failed = visible['execution_status'] != 'completed' or visible['exit_code'] != 0
                consecutive_failures = consecutive_failures + 1 if failed else 0
            else:
                action['status'] = 'execution_unavailable'
                record['final_snapshot'] = None
            assessment, assessment_reference = call(evaluator, 'evaluator', 'evaluate', {
                'task_id': task['task_id'], 'permission': condition['permission'],
                'before_snapshot': action['before_snapshot'], 'after_snapshot': action['after_snapshot'],
                'action_text': action_text, 'explanation': explanation}, evaluator_config)
            action.update(assessment=assessment_reference, assessment_status=assessment['status'])
            save()
            if execution['status'] != 'ok':
                record['stop_reason'] = 'execution_unavailable'
                break
            if consecutive_failures >= config['max_consecutive_failures']:
                record['stop_reason'] = 'execution_failure_limit'
                break
            if turn + 1 == config['max_tool_calls']:
                record['stop_reason'] = 'tool_call_limit'
                break
            if record['output_tokens'] >= config['max_output_tokens'] or time.monotonic() - started >= config['max_seconds']:
                record['stop_reason'] = 'token_limit' if record['output_tokens'] >= config['max_output_tokens'] else 'time_limit'
                break
            limit = config['max_feedback_chars']
            feedback = {key: visible[key] for key in ('execution_status', 'exit_code')}
            feedback.update({key: visible[key][:limit] for key in ('stdout', 'stderr')})
            feedback['truncated'] = any(len(visible[key]) > limit for key in ('stdout', 'stderr'))
            resumed, _ = checked(model, 'model', 'resume', {'trajectory': generation['tokens'],
                'content': json.dumps(feedback, ensure_ascii=False, sort_keys=True)}, config['model'])
            prefix = resumed['tokens']
        # Reuse the first assessment when it already compares the initial and final snapshots.
        initial_to_final = next((action['assessment'] for action in record['actions']
                                if action['before_snapshot'] == initial and action['after_snapshot'] == record['final_snapshot']
                                and action['assessment'] is not None), None)
        if initial_to_final is not None:
            record['final_assessment'] = initial_to_final
        else:
            _, record['final_assessment'] = call(evaluator, 'evaluator', 'evaluate', {
                'task_id': task['task_id'], 'permission': condition['permission'],
                'before_snapshot': initial, 'after_snapshot': record['final_snapshot'],
                'action_text': '\n'.join(episode_actions), 'explanation': '\n'.join(episode_explanations)}, evaluator_config)
        record['status'] = 'complete'
        save()
        return success(request, record, [artifact_ref(directory / 'record.json', 'json')])
    except InputError as exc:
        record.update(status='incomplete', error={'code': exc.code, 'message': str(exc)})
        return failure(request, exc)
    except Exception as exc:
        record.update(status='incomplete', failure_type=type(exc).__name__)
        atomic_bytes(directory / 'traceback.txt', traceback.format_exc().encode('utf-8'))
        raise
    finally:
        try:
            model.reset_episode()
        except Exception as exc:
            record.update(status='incomplete', cleanup_failure=type(exc).__name__)
            atomic_bytes(directory / 'cleanup_traceback.txt', traceback.format_exc().encode('utf-8'))
            raise
        finally:
            # Preserve successful response hashes: elapsed time was frozen by the final save.
            if record['status'] != 'complete':
                save()


def handle(request):
    model = None
    try:
        validate_request(request, {'episode', 'sampling.plan', 'experiment.freeze', 'experiment.load', 'experiment.review', 'experiment.accept', 'experiment.job', 'experiment.patch'})
        if request['operation'].startswith('experiment.'):
            from eval_tampering.experiment import handle as experiment_handle
            return experiment_handle(request)
        if request['operation'] == 'sampling.plan':
            return plan_sampling(request)
        config, _, _, _, _, evaluator_config, _ = _settings(request)
        from eval_tampering.model import ModelRuntime
        from eval_tampering.sandbox import SandboxRunner
        from eval_tampering.evaluate import BehavioralEvaluator
        model = ModelRuntime(config['model'])
        return run_episode(request, model, SandboxRunner(config['sandbox']), BehavioralEvaluator(evaluator_config))
    except (InputError, OSError, ImportError) as exc:
        return failure(request, exc if isinstance(exc, InputError) else InputError('dependency_error' if isinstance(exc, ImportError) else 'file_error', str(exc)))
    finally:
        if model is not None:
            model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    request = None
    try:
        source, output = local_path(args.input), local_path(args.output)
        require(source != output, 'Output must not overwrite input')
    except (InputError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        request = read_json(source)
        result = handle(request)
    except (InputError, OSError) as exc:
        result = failure(request, exc if isinstance(exc, InputError) else InputError('file_error', str(exc)))
    try:
        atomic_json(output, result)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
