"""Bounded native-HF/bridge evidence on one prefix; not experiment acceptance."""

import hashlib
import io
import math
import os
from pathlib import Path
import resource
import time

from .messages import artifact_ref, atomic_bytes, atomic_json, decode_json, fields, fingerprint, read_artifact, require


def validate_benign_inputs(inputs):
    from datetime import date
    fields(inputs, {'layer', 'strength', 'seed', 'max_new_tokens', 'max_seconds', 'date'}, 'benign control inputs')
    require(type(inputs['layer']) is int and 0 <= inputs['layer'] < 24, 'Invalid benign control layer')
    require(type(inputs['seed']) is int and 0 <= inputs['seed'] < 2**32, 'Invalid benign control seed')
    require(type(inputs['max_new_tokens']) is int and 1 <= inputs['max_new_tokens'] <= 128, 'Bound benign generation to 1–128 tokens')
    for key, maximum in [('strength', 10), ('max_seconds', 600)]:
        require(type(inputs[key]) in (int, float) and math.isfinite(inputs[key]) and 0 < inputs[key] <= maximum, 'Invalid benign ' + key)
    require(type(inputs['date']) is str and len(inputs['date']) == 10, 'Declare the benign control template date')
    try:
        valid = date.fromisoformat(inputs['date']).isoformat() == inputs['date']
    except ValueError:
        valid = False
    require(valid, 'Invalid benign control template date')


def benign_control(runtime, inputs, directory):
    """Development-only letter-case control through the actual S1/S2 tool hooks."""
    import numpy as np
    import torch
    from .messages import InputError
    validate_benign_inputs(inputs)
    require(inputs['layer'] in runtime.identity['layers'], 'Use a nominated benign control layer')
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {'schema_version': 1, 'status': 'incomplete', 'fixture': runtime.identity['fixture'],
        'runtime_sha256': fingerprint(runtime.identity), 'inputs': inputs, 'training_words': ['alpha', 'bravo'],
        'validation_words': ['cedar', 'delta'], 'training': [], 'generations': [], 'known_effective': False,
        'planned_generations': 20, 'executed_commands': 0,
        'method': 'mean uppercase-minus-lowercase pre-action residual on training words; baseline/sham/positive/negative/matched-random on new validation words in S1/S2',
        'limitations': ['Small development instrumentation attempt, not tampering mitigation or independent research tasks',
            'A changed residual or malformed output is not a successful letter-case change; null behavior is retained',
            'Generated commands are inspected as data and never executed']}

    def remaining():
        left = inputs['max_seconds'] - (time.monotonic() - started)
        require(left > 0, 'Benign control time budget reached', 'time_limit')
        return left

    def prefix(word, case, name):
        remaining()
        payload = runtime.prepare({'messages': [{'role': 'user', 'content':
            f'Make exactly one functions.execute call with command ["printf", "{word}"], converting the word to {case}. No extra arguments.'}],
            'date': inputs['date'], 'reasoning_effort': 'low'})
        payload['token_ids'] += runtime.tokenizer.encode(' to=functions.execute<|channel|>commentary json<|message|>', add_special_tokens=False)
        payload['attention_mask'] = [1] * len(payload['token_ids'])
        runtime._tokens(payload)
        require(len(payload['token_ids']) + inputs['max_new_tokens'] <= runtime.config['max_context_tokens'], 'Benign control exceeds context budget')
        atomic_json(directory / (name + '.json'), payload)
        return payload, artifact_ref(directory / (name + '.json'), 'json')

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(directory / 'benign-control.json', report)

    save()
    try:
        vectors = []
        for word in report['training_words']:
            pair = []
            for case in ('lowercase', 'uppercase'):
                payload, reference = prefix(word, case, word + '-' + case)
                tensor = torch.tensor([payload['token_ids']], device=runtime.identity['device'])
                name = f"blocks.{inputs['layer']}.hook_resid_post"
                with torch.inference_mode():
                    _, cache = runtime.bridge.run_with_cache(tensor, attention_mask=torch.ones_like(tensor), prepend_bos=False,
                        names_filter=[name], return_cache_object=False, use_cache=False, logits_to_keep=1)
                pair.append(cache[name][0, -1].float().detach().cpu().numpy().copy())
                report['training'].append({'word': word, 'case': case, 'prefix': reference})
                del cache
                runtime.reset_episode()
            vectors.append(pair)
        raw = np.asarray(vectors, dtype=np.float32)
        contrast = (raw[:, 1].astype(np.float64) - raw[:, 0]).mean(axis=0)
        norm = float(np.linalg.norm(contrast))
        require(math.isfinite(norm) and norm > 1e-12, 'Benign training contrast is zero or nonfinite', 'control_unavailable')
        direction = (contrast / norm).astype(np.float32)
        random = np.random.default_rng(inputs['seed']).normal(size=len(direction))
        random = (random / np.linalg.norm(random)).astype(np.float32)
        for name, values in [('direction', {'direction': direction}), ('random', {'direction': random}), ('training', {'residuals': raw})]:
            stream = io.BytesIO()
            np.savez_compressed(stream, **values)
            atomic_bytes(directory / (name + '.npz'), stream.getvalue())
            report[name + '_arrays'] = artifact_ref(directory / (name + '.npz'), 'npz')
        report['contrast_norm'] = norm
        for word in report['validation_words']:
            _, reference = prefix(word, 'lowercase', word + '-validation')
            for schedule in ('S1', 'S2'):
                baseline = None
                for arm, value in [('baseline', None), ('sham', 0.), ('positive', inputs['strength']), ('negative', -inputs['strength']), ('random', inputs['strength'])]:
                    folder = directory / (word + '-' + schedule + '-' + arm)
                    folder.mkdir()
                    intervention = None if value is None else {'layer': inputs['layer'], 'schedule': schedule, 'mode': 'add',
                        'direction': report['random_arrays' if arm == 'random' else 'direction_arrays'], 'value': value,
                        'runtime_sha256': report['runtime_sha256']}
                    request = {'prefix': reference, 'seed': inputs['seed'], 'max_new_tokens': inputs['max_new_tokens'],
                        'temperature': 0., 'max_seconds': remaining(), 'intervention': intervention}
                    atomic_json(folder / 'request.json', request)
                    generated = runtime.generate(request, folder)
                    atomic_json(folder / 'generation.json', generated)
                    command = generated.get('command')
                    category = 'lowercase' if command == ['printf', word] else 'uppercase' if command == ['printf', word.upper()] else 'other_or_incomplete'
                    tokens = decode_json(read_artifact(generated['tokens'], 'json', 4194304))['token_ids']
                    if arm == 'baseline':
                        baseline = tokens
                    report['generations'].append({'word': word, 'schedule': schedule, 'arm': arm, 'category': category,
                        'request': artifact_ref(folder / 'request.json', 'json'), 'generation': artifact_ref(folder / 'generation.json', 'json'),
                        'sham_matches_baseline': tokens == baseline if arm == 'sham' else None,
                        'observed_nonzero': any(event['max_change_norm'] > 0 for event in generated['hook_events'])})
                    save()
        checks = report['generations']
        report['known_effective'] = all(row['category'] == ('uppercase' if row['arm'] == 'positive' else 'lowercase') and row['observed_nonzero']
            for row in checks if row['arm'] in ('positive', 'negative')) and all(row['sham_matches_baseline'] for row in checks if row['arm'] == 'sham')
        report['status'] = 'checked'
    except InputError as exc:
        if exc.code not in ('time_limit', 'control_unavailable'):
            raise
        report.update(status='unavailable' if exc.code == 'control_unavailable' else 'incomplete', reason=exc.code)
    finally:
        runtime.reset_episode()
        save()
    return report | {'control': artifact_ref(directory / 'benign-control.json', 'json')}


def validate_inputs(inputs):
    fields(inputs, {'prefix', 'layer', 'delta', 'max_new_tokens', 'max_seconds', 'rtol', 'atol',
                    'max_peak_rss_bytes', 'max_device_bytes'}, 'runtime check inputs')
    fields(inputs['prefix'], {'path', 'sha256', 'format'}, 'runtime check prefix')
    require(inputs['prefix']['format'] == 'json', 'Expected a JSON token prefix')
    fields(decode_json(read_artifact(inputs['prefix'], 'json', 4194304)),
           {'token_ids', 'attention_mask', 'assistant_boundary', 'runtime_sha256'}, 'runtime check token payload')
    require(type(inputs['layer']) is int and 0 <= inputs['layer'] < 24, 'Invalid check layer')
    require(type(inputs['delta']) in (int, float) and math.isfinite(inputs['delta']) and 0 < abs(inputs['delta']) <= 10, 'Declare a finite nonzero check delta')
    require(type(inputs['max_new_tokens']) is int and 2 <= inputs['max_new_tokens'] <= 4, 'Check 2–4 greedy tokens')
    for name, maximum in (('max_seconds', 600), ('rtol', .1), ('atol', .1)):
        require(type(inputs[name]) in (int, float) and math.isfinite(inputs[name]) and 0 <= inputs[name] <= maximum and
                (name != 'max_seconds' or inputs[name] > 0), 'Invalid ' + name)
    for name in ('max_peak_rss_bytes', 'max_device_bytes'):
        require((name == 'max_device_bytes' and inputs[name] is None) or type(inputs[name]) is int and 0 < inputs[name] <= 2**50, 'Invalid ' + name)


def check(runtime, inputs, directory):
    import numpy as np
    import torch
    from .model import _action
    validate_inputs(inputs)
    payload = decode_json(read_artifact(inputs['prefix'], 'json', 4194304))
    ids, boundary, messages = runtime._tokens(payload)
    require(_action(messages[-1]) and messages[-1]['ending'] is None and messages[-1]['content_start'] == len(ids),
            'Runtime checks require an exact pre-action prefix', 'ineligible')
    require(inputs['layer'] in runtime.identity['layers'] and inputs['layer'] < runtime.identity['config']['num_hidden_layers']-1,
            'Choose a nominated residual layer with downstream layers')
    require(len(ids) + inputs['max_new_tokens'] <= runtime.config['max_context_tokens'], 'Check exceeds context budget')
    fixture = runtime.identity['fixture']
    require((inputs['max_device_bytes'] is None) == fixture, 'Declare a device-memory limit for a GPU check; CPU checks use null')
    directory.mkdir(parents=True, exist_ok=True)
    width, layers = runtime.identity['config']['hidden_size'], runtime.identity['layers']
    positions = sorted({boundary, len(ids)-1})
    vector = np.zeros(width, dtype=np.float32)
    vector[0] = 1
    stream = io.BytesIO()
    np.savez_compressed(stream, direction=vector)
    atomic_bytes(directory / 'coordinate.npz', stream.getvalue())
    direction = artifact_ref(directory / 'coordinate.npz', 'npz')
    atomic_json(directory / 'runtime.json', runtime.identity)
    report = {'schema_version': 1, 'status': 'incomplete', 'fixture': fixture, 'inputs': inputs,
        'runtime': artifact_ref(directory / 'runtime.json', 'json'), 'runtime_sha256': fingerprint(runtime.identity),
        'check_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference': 'Direct native Hugging Face calls to the same loaded weights, with adapter hooks cleared; not a second checkpoint instance',
        'positions': positions, 'layers': layers, 'checks': [], 'variants': [], 'experiment_accepted': False,
        'limitations': ['One declared pre-action prefix and 2–4 greedy tokens; not general model behavior or study acceptance',
            'Coordinate controls test backend mechanics; training-derived directions need their separate calibration',
            'The deadline is cooperative between synchronous forwards and generation steps; a single kernel cannot be preempted',
            'Check time excludes model loading; the outer operation record includes loading. RSS is the process lifetime high-water mark']}
    arrays, handles = {}, []
    model = runtime.bridge.original_model
    started = time.monotonic()
    before_rng = torch.get_rng_state().clone()
    before_cuda_rng = torch.cuda.get_rng_state(0).clone() if not fixture else None
    original_hooks = [(module, tuple(module._forward_hooks), tuple(module._forward_pre_hooks)) for module in model.modules()]
    if not fixture:
        torch.cuda.reset_peak_memory_stats(0)

    def remaining():
        if not fixture:
            torch.cuda.synchronize(0)
        left = inputs['max_seconds'] - (time.monotonic()-started)
        require(left > 0, 'Runtime check deadline reached', 'time_limit')
        return left

    def compare(name, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(), 'Invalid check array geometry/values', 'numerical_error')
        arrays[name + '_actual'], arrays[name + '_reference'] = actual.copy(), expected.copy()
        passed = bool(np.allclose(actual, expected, rtol=inputs['rtol'], atol=inputs['atol']))
        report['checks'].append({'check_id': name, 'passed': passed, 'shape': list(actual.shape),
            'max_absolute_error': float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)), initial=0))})

    def boolean(name, value, **evidence):
        report['checks'].append({'check_id': name, 'passed': bool(value), **evidence})

    def save():
        if arrays:
            stream = io.BytesIO()
            np.savez_compressed(stream, **arrays)
            atomic_bytes(directory / 'check-arrays.npz', stream.getvalue())
            report['arrays'] = artifact_ref(directory / 'check-arrays.npz', 'npz')
        report['elapsed_seconds'] = time.monotonic()-started
        if report['status'] == 'passed' and report['elapsed_seconds'] > inputs['max_seconds']:
            next(row for row in report['checks'] if row['check_id'] == 'within_time')['passed'] = False
            report['status'] = 'failed'
        atomic_json(directory / 'check.json', report)

    def native(tokens, intervention=None, collect=False):
        remaining()
        runtime.reset_episode()
        captured, local = {}, []
        if intervention is not None:
            def coordinate(module, args, output):
                changed = output.clone()
                first = boundary if intervention['schedule'] == 'S2' else len(ids)-1
                stop = first+1 if intervention['schedule'] == 'P1' else len(tokens)
                if intervention['mode'] == 'add':
                    changed[:, first:stop, 0] = (output[:, first:stop, 0].float() + intervention['value']).to(output.dtype)
                else:
                    changed[:, first:stop, 0] = intervention['value']
                return changed
            local.append(model.model.layers[inputs['layer']].register_forward_hook(coordinate))
        if collect:
            for layer in layers:
                def observe(module, args, output, layer=layer):
                    captured[layer] = output[0, positions, :].float().detach().cpu().numpy().copy()
                local.append(model.model.layers[layer].register_forward_hook(observe))
        try:
            tensor = torch.tensor([tokens], device=runtime.identity['device'])
            with torch.inference_mode():
                output = model(tensor, attention_mask=torch.ones_like(tensor), use_cache=False, logits_to_keep=1).logits
            return output[0, -1].float().detach().cpu().numpy().copy(), captured
        finally:
            for handle in local:
                handle.remove()

    completed = False
    save()
    try:
        native_logits, native_residuals = native(ids, collect=True)
        remaining()
        tensor = torch.tensor([ids], device=runtime.identity['device'])
        names = [f'blocks.{layer}.hook_resid_post' for layer in layers]
        with torch.inference_mode():
            logits, cache = runtime.bridge.run_with_cache(tensor, attention_mask=torch.ones_like(tensor), prepend_bos=False,
                names_filter=names, return_cache_object=False, use_cache=False, logits_to_keep=1)
        compare('unmodified_logits', logits[0, -1].float().detach().cpu().numpy(), native_logits)
        for layer, name in zip(layers, names):
            compare('residual_' + str(layer), cache[name][0, positions, :].float().detach().cpu().numpy(), native_residuals[layer])
        del cache, logits
        runtime.reset_episode()
        # The first native/bridge forwards install upstream capture hooks lazily.
        # Measure episode cleanup against that initialized, unmodified baseline.
        original_hooks = [(module, tuple(module._forward_hooks), tuple(module._forward_pre_hooks)) for module in model.modules()]
        report['hook_baseline'] = 'After unmodified native and observation-only bridge forwards initialize upstream capture hooks'
        patch_value = float(native_residuals[inputs['layer']][-1, 0]) + inputs['delta']
        variants = [('baseline', None, None, None), ('sham', 'P1', 'add', 0.),
                    ('patch', 'P1', 'replace_projection', patch_value), ('S1', 'S1', 'add', inputs['delta']),
                    ('S2', 'S2', 'add', inputs['delta']), ('after', None, None, None)]
        baseline_tokens = None
        for name, schedule, mode, value in variants:
            intervention = None if schedule is None else {'layer': inputs['layer'], 'schedule': schedule, 'mode': mode,
                'value': value, 'direction': direction, 'runtime_sha256': fingerprint(runtime.identity)}
            folder = directory / name
            folder.mkdir()
            cached, forward_inputs = [], []
            def observe_inputs(module, args, kwargs):
                value = kwargs.get('input_ids', args[0] if args else None)
                forward_inputs.append({'tokens': value[0].tolist(), 'use_cache': kwargs.get('use_cache')})
            input_handle = model.register_forward_pre_hook(observe_inputs, with_kwargs=True)
            handles.append(input_handle)
            def observe_logits(module, args, output):
                cached.append(output[0, -1].float().detach().cpu().numpy().copy())
            handle = model.lm_head.register_forward_hook(observe_logits)
            handles.append(handle)
            try:
                result = runtime.generate({'prefix': inputs['prefix'], 'seed': 21, 'max_new_tokens': inputs['max_new_tokens'],
                    'temperature': 0., 'max_seconds': min(600, remaining()), 'intervention': intervention}, folder)
            finally:
                handle.remove()
                handles.remove(handle)
                input_handle.remove()
                handles.remove(input_handle)
            tokens = decode_json(read_artifact(result['tokens'], 'json', 4194304))['token_ids']
            suffix = tokens[len(ids):]
            row = {'name': name, 'intervention': intervention, 'generation': result, 'generation_forward_count': len(cached),
                   'forward_inputs': forward_inputs}
            report['variants'].append(row)
            use_cache = runtime.identity['generation_use_cache']
            valid_inputs = len(forward_inputs) == len(cached) and all(
                item['use_cache'] is use_cache and item['tokens'] == (
                    tokens[:len(ids)+step] if not use_cache or step == 0 else [tokens[len(ids)+step-1]])
                for step, item in enumerate(forward_inputs))
            boolean(name + '_generation_observed', len(cached) == len(suffix) and len(cached) >= 2 and valid_inputs,
                    forward_count=len(cached), generated_tokens=len(suffix), use_cache=use_cache)
            for step, cached_logits in enumerate(cached):
                require(step < len(suffix), 'Cached forward has no corresponding generated token', 'runtime_error')
                reference, _ = native(tokens[:len(ids)+step], intervention)
                compare(f'{name}_logits_{step}', cached_logits, reference)
                boolean(f'{name}_token_{step}', suffix[step] == int(reference.argmax()), generated_token=suffix[step], native_argmax=int(reference.argmax()))
            if name == 'baseline':
                baseline_tokens = tokens
            if name in ('sham', 'after'):
                boolean(name + '_tokens_unchanged', tokens == baseline_tokens)
            if intervention is not None:
                observed = [p for event in result['hook_events'] for p in event['processed_positions']]
                first = boundary if schedule == 'S2' else len(ids)-1
                expected = ([first] if schedule == 'P1' else list(range(first, len(ids)+len(cached)-1))) if use_cache else [
                    position for step in range(len(cached)) for position in
                    ([first] if schedule == 'P1' else range(first, len(ids)+step))]
                boolean(name + '_positions', observed == expected, observed=observed, expected=expected)
                diagnostic = runtime.diagnose({'prefix': inputs['prefix'], 'intervention': intervention, 'max_seconds': min(600, remaining())}, folder / 'diagnostic')
                row['diagnostic'] = diagnostic['diagnostic']
                stats = diagnostic['statistics']
                boolean(name + '_diagnostic_complete', diagnostic['status'] == 'complete')
                boolean(name + '_actual_change', stats['observed_nonzero'] == (name != 'sham'), statistics=stats)
                router_observed = stats['router_observation'] == 'observed'
                boolean(name + '_router_observed', router_observed)
                residuals_observed = stats['downstream_residual_observation'] == 'observed'
                boolean(name + '_downstream_change',
                        stats['logit_l2_change'] == 0 and residuals_observed and stats['downstream_residual_l2_change'] == 0
                        if name == 'sham' else stats['logit_l2_change'] > 0 and router_observed and
                        stats['router_logit_l2_change'] > 0 and residuals_observed and stats['downstream_residual_l2_change'] > 0)
                with np.load(io.BytesIO(read_artifact(diagnostic['arrays'], 'npz', 16777216)), allow_pickle=False) as data:
                    independent, _ = native(ids, intervention)
                    compare(name + '_diagnostic_native_logits', data['intervened_logits'], independent)
                    compare(name + '_orthogonal_unchanged', data['boundary_after'][1:], data['boundary_before'][1:])
            save()
        completed = True
    finally:
        for handle in handles:
            handle.remove()
        runtime.reset_episode()
        if not fixture:
            torch.cuda.synchronize(0)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname == 'Darwin' else 1024)
        report['process_peak_rss_bytes'] = rss
        report['device_peak_allocated_bytes'] = None if fixture else torch.cuda.max_memory_allocated(0)
        report['device_peak_reserved_bytes'] = None if fixture else torch.cuda.max_memory_reserved(0)
        boolean('rng_preserved', torch.equal(before_rng, torch.get_rng_state()) and
                (fixture or torch.equal(before_cuda_rng, torch.cuda.get_rng_state(0))))
        boolean('hooks_clean', all(tuple(module._forward_hooks) == forward and tuple(module._forward_pre_hooks) == before
                                  for module, forward, before in original_hooks) and all(not hook.fwd_hooks for hook in runtime.bridge.hook_dict.values()))
        boolean('within_time', time.monotonic()-started <= inputs['max_seconds'])
        boolean('within_rss_limit', rss <= inputs['max_peak_rss_bytes'])
        boolean('within_device_limit', fixture or report['device_peak_reserved_bytes'] <= inputs['max_device_bytes'])
        if completed:
            report['status'] = 'passed' if all(row['passed'] for row in report['checks']) else 'failed'
        save()
    return report | {'check': artifact_ref(directory / 'check.json', 'json')}
