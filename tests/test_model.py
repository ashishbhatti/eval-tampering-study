"""Actual tiny GPT-OSS engineering checks; never a 20B/GPU acceptance claim."""

from contextlib import ExitStack
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from eval_tampering.messages import InputError, artifact_ref, atomic_json, fingerprint, read_artifact
from eval_tampering.model import ModelRuntime, _allowed_positions, _messages, CALL, RETURN

ASSETS = Path('.cache/gpt-oss-assets')
AVAILABLE = ASSETS.joinpath('tokenizer.json').is_file() and all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'transformer_lens'))


class ContractTests(unittest.TestCase):
    def test_lazy_import_and_configuration_ownership(self):
        code = 'import sys; import eval_tampering.model; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'
        subprocess.run([sys.executable, '-B', '-c', code], check=True)
        config = {'profile': 'tiny-gpt-oss-cpu', 'model_path': str(ASSETS), 'artifact_root': '.cache/model-contract', 'max_context_tokens': 512}
        runtime = ModelRuntime(config)
        config['profile'] = 'changed'
        self.assertEqual(runtime.config['profile'], 'tiny-gpt-oss-cpu')
        runtime.config['profile'] = 'changed'
        self.assertIsNone(runtime.bridge)
        runtime.close()
        with self.assertRaises(InputError):
            ModelRuntime(config)
        with self.assertRaises(InputError):
            ModelRuntime(config | {'profile': []})


@unittest.skipUnless(AVAILABLE, 'Install requirements-model.txt and the pinned tokenizer assets for tiny GPT-OSS checks')
class TinyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.work = tempfile.TemporaryDirectory(prefix='model-tests-', dir='.cache')
        cls.root = Path(cls.work.name).resolve()
        cls.config = {'profile': 'tiny-gpt-oss-cpu', 'model_path': str(ASSETS),
                      'artifact_root': str(cls.root / 'records'), 'max_context_tokens': 512}
        cls.runtime = ModelRuntime(cls.config)
        cls.runtime.load()
        cls.prepared = cls.runtime.prepare({'messages': [{'role': 'user', 'content': 'Fix the implementation.'}],
                                            'date': '2026-09-09', 'reasoning_effort': 'medium'})
        cls.prefix = cls.append(cls.prepared, '<|channel|>analysis<|message|>Inspect the tests.<|end|><|start|>assistant to=functions.execute<|channel|>commentary json<|message|>')
        cls.completed = cls.append(cls.prefix, '{"command":["python","-c","print(1)"]}<|call|>')
        cls.counter = 0

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close()
        cls.work.cleanup()

    def tearDown(self):
        self.runtime.reset_episode()

    @classmethod
    def append(cls, payload, text):
        result = json.loads(json.dumps(payload))
        result['token_ids'] += cls.runtime.tokenizer.encode(text, add_special_tokens=False)
        result['attention_mask'] = [1] * len(result['token_ids'])
        return result

    def directory(self):
        type(self).counter += 1
        folder = self.root / str(self.counter)
        folder.mkdir()
        return folder

    def reference(self):
        import torch
        from transformers import GptOssConfig, GptOssForCausalLM
        config = GptOssConfig.from_dict(self.runtime.identity['config'])
        config._attn_implementation = self.runtime.identity['attention_implementation']
        config._experts_implementation = self.runtime.identity['experts_implementation']
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(17)
            return GptOssForCausalLM(config).eval()

    def direction(self, schedule='S1', mode='add', value=0.1):
        import numpy as np
        folder = self.directory()
        vector = np.zeros(16, dtype=np.float32)
        vector[0] = 1
        np.savez(folder / 'direction.npz', direction=vector)
        return {'layer': 1, 'schedule': schedule, 'mode': mode, 'value': value,
                'runtime_sha256': fingerprint(self.runtime.identity),
                'direction': artifact_ref(folder / 'direction.npz', 'npz')}

    def generation_inputs(self, payload=None, intervention=None):
        folder = self.directory()
        atomic_json(folder / 'prefix.json', payload or self.prefix)
        return {'prefix': artifact_ref(folder / 'prefix.json', 'json'), 'seed': 21, 'max_new_tokens': 4,
                'temperature': 0, 'max_seconds': 30, 'intervention': intervention}

    def test_identity_and_explicit_hardware_failure(self):
        import torch
        self.assertTrue(self.runtime.identity['fixture'])
        self.assertFalse(self.runtime.identity['research_backend_validated'])
        self.assertEqual(self.runtime.identity['layers'], [0, 1, 2])
        self.assertEqual(self.runtime.identity['config']['num_local_experts'], 4)
        config = self.config | {'profile': 'gpt-oss-20b-mxfp4'}
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(InputError, 'CUDA'):
                ModelRuntime(config).load()
        with tempfile.TemporaryDirectory(dir=self.root) as invalid:
            with self.assertRaises(OSError):
                ModelRuntime(self.config | {'model_path': invalid}).load()
            Path(invalid, 'config.json').write_text('{}')
            with self.assertRaisesRegex(InputError, 'asset mismatch'):
                ModelRuntime(self.config | {'model_path': invalid}).load()

    def test_raw_reference_logits_residuals_and_observation_sham(self):
        import torch
        reference = self.reference()
        ids = torch.tensor([[12, 45, 87, 63, 121]])
        observed = {}
        handles = []
        for layer in (0, 1, 2):
            def collect(module, args, output, layer=layer):
                observed[layer] = output.detach().clone()
            handles.append(reference.model.layers[layer].register_forward_hook(collect))
        with torch.inference_mode():
            expected = reference(ids, use_cache=False).logits
            actual, cache = self.runtime.bridge.run_with_cache(ids, prepend_bos=False,
                attention_mask=torch.ones_like(ids), use_cache=False,
                names_filter=[f'blocks.{i}.hook_resid_post' for i in (0, 1, 2)], return_cache_object=False)
            with self.runtime.bridge.hooks(fwd_hooks=[('blocks.1.hook_resid_post', lambda value, hook: value)]):
                sham = self.runtime.bridge(ids, prepend_bos=False, use_cache=False)
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(expected, sham, rtol=0, atol=0)
        for layer in (0, 1, 2):
            torch.testing.assert_close(observed[layer], cache[f'blocks.{layer}.hook_resid_post'], rtol=0, atol=0)
        self.assertTrue(all('hook_resid_post' in name or 'hook_out' in name for name in cache))

    def test_observed_code_header_continues_generation_but_requires_json_body(self):
        import torch
        for metadata in ('code', 'json', '<|constrain|>json', '<|constrain|>code'):
            with self.subTest(metadata=metadata):
                observed = ('<|channel|>analysis<|message|>We need to implement clamp correctly. '
                            'Currently clamp returns max(value, lower) which fails upper bound. '
                            "We need to modify solution.py. Also ensure integer return. Let's patch."
                            '<|end|><|start|>assistant<|channel|>commentary to=functions.execute '
                            + metadata + '<|message|>')
                prefix = self.append(self.prepared, observed)
                completed = self.append(prefix, '{"command":["python","-c","print(1)"]}<|call|>')
                folder = self.directory()
                atomic_json(folder / 'prefix.json', self.prepared)

                def scripted_generate(input_ids, **kwargs):
                    # Exercise the real stopping criteria against every generated prefix.
                    for end in range(input_ids.shape[1]+1, len(completed['token_ids'])+1):
                        output = torch.tensor([completed['token_ids'][:end]])
                        if bool(kwargs['stopping_criteria'](output, None).any()):
                            return output
                    return output

                with patch.object(self.runtime.bridge, 'hf_generate', side_effect=scripted_generate):
                    result = self.runtime.generate({'prefix': artifact_ref(folder / 'prefix.json', 'json'),
                        'seed': 37, 'max_new_tokens': 128, 'temperature': 0, 'max_seconds': 30,
                        'intervention': None}, folder)
                self.assertEqual(result['status'], 'tool_call')
                self.assertEqual(result['command'], ['python', '-c', 'print(1)'])
                self.assertEqual(result['pre_action_position'], len(prefix['token_ids'])-1)
                self.assertEqual(self.runtime._response(prefix)['status'], 'truncated')
                self.assertEqual(_allowed_positions(self.runtime.tokenizer, completed['token_ids'],
                    completed['assistant_boundary'], 'P1'), [len(prefix['token_ids'])-1])
                self.assertEqual(_allowed_positions(self.runtime.tokenizer, completed['token_ids'],
                    completed['assistant_boundary'], 'S1'),
                    list(range(len(prefix['token_ids'])-1, len(completed['token_ids'])-1)))
                for body in ('print(1)', '{"command":[],"command":["true"]}', '{"command":"true"}'):
                    self.assertEqual(self.runtime._response(self.append(prefix, body+'<|call|>'))['status'], 'malformed')
        for header in ('commentary to=functions.execute javascript',
                       'commentary to=functions.execute <|constrain|>javascript',
                       'commentary to=functions.execute <|constrain|>',
                       'commentary to=functions.execute to=other code'):
            invalid = self.append(self.prepared, '<|channel|>'+header+'<|message|>')
            self.assertFalse(_messages(self.runtime.tokenizer, invalid['token_ids'])[1])

    def test_harmony_history_boundaries_and_strict_parsing(self):
        payload = self.runtime.prepare({'messages': [{'role': 'user', 'content': 'hello'}],
                                         'date': '2001-02-03', 'reasoning_effort': 'low'})
        text = self.runtime.tokenizer.decode(payload['token_ids'])
        self.assertIn('Current date: 2001-02-03', text)
        self.assertTrue(text.endswith('<|start|>assistant'))
        response = self.runtime._response(self.completed)
        self.assertEqual(response['status'], 'tool_call')
        self.assertEqual(response['command'], ['python', '-c', 'print(1)'])
        self.assertEqual(response['pre_action_position'], len(self.prefix['token_ids']) - 1)
        self.assertEqual(response['action_span'][1], len(self.completed['token_ids']) - 1)
        malformed = self.append(self.prefix, '{"command":[],"command":["true"]}<|call|>')
        self.assertEqual(self.runtime._response(malformed)['status'], 'malformed')
        final = self.append(self.prepared, '<|channel|>final<|message|>Done.<|return|>')
        self.assertEqual(self.runtime._response(final)['status'], 'final')
        self.assertEqual(self.runtime._response(self.prefix)['status'], 'truncated')
        other_order = self.append(self.prepared, '<|channel|>commentary to=functions.execute<|message|>{"command":["true"]}<|call|>')
        self.assertEqual(self.runtime._response(other_order)['status'], 'tool_call')
        bad_future = self.append(self.completed, '<|start|>functions.execute<|message|>future<|end|>')
        with self.assertRaises(InputError):
            self.runtime._tokens(bad_future)
        tool_start = len(self.completed['token_ids'])
        self.assertTrue(all(position < tool_start for position in _allowed_positions(
            self.runtime.tokenizer, bad_future['token_ids'], bad_future['assistant_boundary'], 'S2')))
        repeated_call = self.append(self.completed, '<|start|>assistant<|channel|>final<|message|>Done.<|return|>')
        with self.assertRaises(InputError):
            self.runtime._tokens(repeated_call)
        wrong_mask = json.loads(json.dumps(self.prefix))
        wrong_mask['attention_mask'][0] = 0
        with self.assertRaises(InputError):
            self.runtime._tokens(wrong_mask)

    def test_reasoning_survives_tool_observation_and_invalid_histories_fail(self):
        history = [{'role': 'user', 'content': 'work'},
                   {'role': 'assistant', 'content': '', 'thinking': 'Retain this analysis.',
                    'tool_calls': [{'type': 'function', 'function': {'name': 'execute', 'arguments': {'command': ['true']}}}]},
                   {'role': 'tool', 'content': 'completed'}]
        payload = self.runtime.prepare({'messages': history, 'date': '2026-09-09', 'reasoning_effort': 'medium'})
        text = self.runtime.tokenizer.decode(payload['token_ids'])
        self.assertIn('Retain this analysis.', text)
        self.assertIn('completed', text)
        for invalid in ([{'role': 'tool', 'content': 'missing call'}], [{'role': 'user', 'content': 1}]):
            with self.assertRaises(InputError):
                self.runtime.prepare({'messages': invalid, 'date': '2026-09-09', 'reasoning_effort': 'medium'})

    def test_resume_preserves_exact_generated_tokens(self):
        folder = self.directory()
        # Noncanonical token segmentation would be changed by decoding/re-encoding history.
        alternate = json.loads(json.dumps(self.prefix))
        for piece in ('{ "command" : [ "', 'tr', 'ue', '" ] }<|call|>'):
            alternate['token_ids'] += self.runtime.tokenizer.encode(piece, add_special_tokens=False)
        alternate['attention_mask'] = [1] * len(alternate['token_ids'])
        canonical = self.runtime.tokenizer.encode(self.runtime.tokenizer.decode(alternate['token_ids']), add_special_tokens=False)
        self.assertNotEqual(canonical, alternate['token_ids'])
        atomic_json(folder / 'trajectory.json', alternate)
        inputs = {'trajectory': artifact_ref(folder / 'trajectory.json', 'json'), 'content': '{"stdout":"visible","exit_code":0}'}
        resumed = self.runtime.resume(inputs)
        self.assertEqual(resumed['token_ids'][:len(alternate['token_ids'])], alternate['token_ids'])
        suffix = self.runtime.tokenizer.decode(resumed['token_ids'][len(alternate['token_ids']):])
        self.assertTrue(suffix.startswith('<|start|>functions.execute to=assistant<|channel|>commentary<|message|>'))
        self.assertTrue(suffix.endswith('<|end|><|start|>assistant'))
        self.assertIn('visible', suffix)
        self.assertEqual(resumed['assistant_boundary'], len(resumed['token_ids']) - 1)
        self.runtime._tokens(resumed)
        atomic_json(folder / 'invalid.json', self.prepared)
        with self.assertRaises(InputError):
            self.runtime.resume(inputs | {'trajectory': artifact_ref(folder / 'invalid.json', 'json')})

    def test_schedule_one_token_shift_and_last_action_state(self):
        ids = self.completed['token_ids']
        boundary = self.completed['assistant_boundary']
        start, end = self.runtime._response(self.completed)['action_span']
        self.assertEqual(_allowed_positions(self.runtime.tokenizer, ids, boundary, 'S1'), list(range(start - 1, end)))
        self.assertEqual(_allowed_positions(self.runtime.tokenizer, ids, boundary, 'P1'), [start - 1])
        self.assertEqual(_allowed_positions(self.runtime.tokenizer, self.prepared['token_ids'], boundary, 'P1'), [])
        self.assertEqual(_allowed_positions(self.runtime.tokenizer, self.prefix['token_ids'], boundary, 'S1'), [start - 1])
        self.assertEqual(_allowed_positions(self.runtime.tokenizer, self.prepared['token_ids'], boundary, 'S1'), [])
        self.assertEqual(_allowed_positions(self.runtime.tokenizer, self.prepared['token_ids'], boundary, 'S2'), [boundary])
        self.assertTrue(all(p >= boundary for p in _allowed_positions(self.runtime.tokenizer, ids, boundary, 'S2')))

    def test_live_patch_cached_generation_matches_independent_full_replay(self):
        import torch
        baseline = self.runtime.generate(self.generation_inputs(), self.directory())
        baseline_tokens = json.loads(read_artifact(baseline['tokens'], 'json', 1000000))['token_ids']
        for schedule, mode in [('S1', 'add'), ('S2', 'add'), ('S1', 'replace_projection'), ('P1', 'add')]:
            with self.subTest(schedule=schedule, mode=mode):
                intervention = self.direction(schedule, mode, 0.25)
                directory = self.directory()
                result = self.runtime.generate(self.generation_inputs(intervention=intervention), directory)
                actual = json.loads(read_artifact(result['tokens'], 'json', 1000000))['token_ids']
                self.assertTrue(result['hook_events'])
                self.assertTrue(any(event['max_change_norm'] > 0 for event in result['hook_events']))
                reference = self.reference()
                minimum = self.prefix['assistant_boundary'] if schedule == 'S2' else len(self.prefix['token_ids']) - 1

                def coordinate_patch(module, args, output):
                    changed = output.clone()
                    stop = minimum + 1 if schedule == 'P1' else output.shape[1]
                    if mode == 'add':
                        changed[:, minimum:stop, 0] += 0.25
                    else:
                        changed[:, minimum:stop, 0] = 0.25
                    torch.testing.assert_close(changed[:, stop:, :], output[:, stop:, :], rtol=0, atol=0)
                    torch.testing.assert_close(changed[:, :minimum, :], output[:, :minimum, :], rtol=0, atol=0)
                    torch.testing.assert_close(changed[:, minimum:, 1:], output[:, minimum:, 1:], rtol=0, atol=0)
                    return changed

                handle = reference.model.layers[1].register_forward_hook(coordinate_patch)
                tokens = self.prefix['token_ids'][:]
                with torch.inference_mode():
                    for _ in range(4):
                        tensor = torch.tensor([tokens])
                        logits = reference(tensor, attention_mask=torch.ones_like(tensor), use_cache=False, logits_to_keep=1).logits
                        token = int(logits[0, -1].argmax())
                        tokens.append(token)
                        if token in (CALL, RETURN):
                            break
                handle.remove()
                self.assertEqual(tokens, actual)
                self.assertNotEqual(actual, baseline_tokens)
                positions = [p for event in result['hook_events'] for p in event['processed_positions']]
                self.assertIn(minimum, positions)
                self.assertEqual(len(positions), len(set(positions)))
                if schedule == 'P1':
                    self.assertEqual(positions, [minimum])

    def test_diagnostic_same_prefix_logits_actual_changes_and_router_recomputation(self):
        import numpy as np
        import torch
        folder = self.directory()
        atomic_json(folder / 'prefix.json', self.prefix)
        tensor = torch.tensor([self.prefix['token_ids']])
        for schedule in ('P1', 'S1', 'S2'):
            with self.subTest(schedule=schedule):
                intervention = self.direction(schedule, value=.25)
                result = self.runtime.diagnose({'prefix': artifact_ref(folder / 'prefix.json', 'json'),
                    'intervention': intervention, 'max_seconds': 30}, self.directory())
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['statistics']['router_observation'], 'observed')
                from eval_tampering.interventions import InterventionPlanner
                planner = InterventionPlanner({'artifact_root': str(self.root), 'tokenizer': artifact_ref(ASSETS / 'tokenizer.json', 'json'),
                    'label_kind': 'fixture', 'random_seed': 17})
                planner._tokenizer = self.runtime.tokenizer
                checked = planner._diagnostic_check({'job_id': 'actual-tiny', 'request': {'inputs': result['inputs']}}, result['diagnostic'], self.runtime.identity)
                self.assertEqual(checked['status'], 'applied')
                self.assertEqual(result['router_calls'], {'baseline': [1, 1], 'intervened': [1, 1]})
                self.assertGreater(result['statistics']['router_logit_l2_change'], 0)
                with np.load(result['arrays']['path'], allow_pickle=False) as arrays:
                    values = {name: arrays[name].copy() for name in arrays.files}
                positions = _allowed_positions(self.runtime.tokenizer, self.prefix['token_ids'], self.prefix['assistant_boundary'], schedule)
                self.assertEqual(values['positions'].tolist(), positions)
                np.testing.assert_allclose(values['change_norms'], .25, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(values['projection_after'] - values['projection_before'], .25, rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(values['boundary_before'][1:], values['boundary_after'][1:])
                self.assertAlmostEqual(result['hook_events'][0]['change_norm_sum'], .25 * len(positions), places=5)
                reference = self.reference()
                with torch.inference_mode():
                    baseline = reference(tensor, attention_mask=torch.ones_like(tensor), use_cache=False, logits_to_keep=1).logits[0, -1]
                def coordinate_patch(module, args, output):
                    changed = output.clone()
                    changed[:, positions, 0] += .25
                    return changed
                handle = reference.model.layers[1].register_forward_hook(coordinate_patch)
                try:
                    with torch.inference_mode():
                        changed = reference(tensor, attention_mask=torch.ones_like(tensor), use_cache=False, logits_to_keep=1).logits[0, -1]
                finally:
                    handle.remove()
                np.testing.assert_array_equal(values['baseline_logits'], baseline.numpy())
                np.testing.assert_array_equal(values['intervened_logits'], changed.numpy())
                self.assertGreater(result['statistics']['logit_l2_change'], 0)

    def test_diagnostic_zero_and_rounded_away_requests_are_distinct(self):
        import numpy as np
        import torch
        folder = self.directory()
        atomic_json(folder / 'prefix.json', self.prefix)
        rng = torch.get_rng_state().clone()
        for value in (0, 1e-50):
            result = self.runtime.diagnose({'prefix': artifact_ref(folder / 'prefix.json', 'json'),
                'intervention': self.direction(value=value), 'max_seconds': 30}, self.directory())
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['statistics']['requested_nonzero'], value != 0)
            self.assertFalse(result['statistics']['observed_nonzero'])
            self.assertEqual(result['statistics']['logit_l2_change'], 0)
            self.assertEqual(result['hook_events'][0]['runtime_dtype'], 'torch.float32')
            with np.load(result['arrays']['path'], allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['baseline_logits'], arrays['intervened_logits'])
                np.testing.assert_array_equal(arrays['boundary_before'], arrays['boundary_after'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_diagnostic_rejects_action_content_and_retains_partial_failure_without_hooks(self):
        import numpy as np
        routers = [layer.mlp.router for layer in self.runtime.bridge.original_model.model.layers]
        original_hooks = [dict(router._forward_hooks) for router in routers]
        folder = self.directory()
        atomic_json(folder / 'prefix.json', self.prefix)
        atomic_json(folder / 'completed.json', self.completed)
        intervention = self.direction()
        with patch.object(self.runtime.bridge, 'forward', side_effect=AssertionError('unexpected forward')):
            with self.assertRaisesRegex(InputError, 'pre-action prefix'):
                self.runtime.diagnose({'prefix': artifact_ref(folder / 'completed.json', 'json'), 'intervention': intervention, 'max_seconds': 30}, self.directory())
        forward = self.runtime.bridge.forward
        calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('deliberate diagnostic second-pass failure')
            return forward(*args, **kwargs)
        output = self.directory()
        with patch.object(self.runtime.bridge, 'forward', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'second-pass failure'):
                self.runtime.diagnose({'prefix': artifact_ref(folder / 'prefix.json', 'json'), 'intervention': intervention, 'max_seconds': 30}, output)
        self.assertEqual(json.loads((output / 'diagnostic.json').read_text())['completed_passes'], ['baseline'])
        with np.load(output / 'diagnostic.npz', allow_pickle=False) as arrays:
            self.assertIn('baseline_logits', arrays.files)
            self.assertNotIn('intervened_logits', arrays.files)
        self.assertFalse(self.runtime.bridge.original_model._forward_pre_hooks)
        self.assertEqual([dict(router._forward_hooks) for router in routers], original_hooks)
        self.assertTrue(all(not hook.fwd_hooks for hook in self.runtime.bridge.hook_dict.values()))

    def test_capture_replays_last_action_and_pre_action_without_future_tokens(self):
        import numpy as np
        import torch
        folder = self.directory()
        atomic_json(folder / 'trajectory.json', self.completed)
        reference = self.reference()
        for target in ('action', 'pre_action'):
            result = self.runtime.capture({'trajectory': artifact_ref(folder / 'trajectory.json', 'json'),
                                            'target': target, 'layers': [0, 1, 2]}, folder)
            self.assertEqual(result['trajectory'], artifact_ref(folder / 'trajectory.json', 'json'))
            with np.load(io.BytesIO(read_artifact(result['features'], 'npz', 1000000)), allow_pickle=False) as data:
                values = data['residuals']
                np.testing.assert_array_equal(data['last'], values[:, -1, :])
                np.testing.assert_array_equal(data['mean'], values.mean(axis=1))
            observed = {}
            handles = []
            for layer in (0, 1, 2):
                def collect(module, args, output, layer=layer):
                    observed[layer] = output[0, result['positions'], :].detach().numpy().copy()
                handles.append(reference.model.layers[layer].register_forward_hook(collect))
            with torch.inference_mode():
                reference(torch.tensor([self.completed['token_ids'][:result['causal_prefix_length']]]), use_cache=False, logits_to_keep=1)
            for handle in handles:
                handle.remove()
            for layer in (0, 1, 2):
                np.testing.assert_array_equal(values[layer], observed[layer])
            self.assertEqual(result['positions'][-1] + 1, result['causal_prefix_length'])
            if target == 'action':
                self.assertEqual(result['causal_prefix_length'], len(self.completed['token_ids']) - 1)

    def test_zero_sham_rng_and_cleanup_after_failure(self):
        import torch
        inputs = self.generation_inputs()
        before_rng = torch.get_rng_state().clone()
        baseline = self.runtime.generate(inputs, self.directory())
        sham = self.runtime.generate(inputs | {'intervention': self.direction(value=0)}, self.directory())
        self.assertEqual(read_artifact(baseline['tokens'], 'json', 1000000), read_artifact(sham['tokens'], 'json', 1000000))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        original = self.runtime.bridge.hf_generate
        failed_dir = self.directory()

        def fail_after_generation(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('deliberate fixture failure')

        with patch.object(self.runtime.bridge, 'hf_generate', side_effect=fail_after_generation):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                self.runtime.generate(inputs | {'intervention': self.direction()}, failed_dir)
        self.assertGreater(len(json.loads((failed_dir / 'tokens.json').read_text())['token_ids']), len(self.prefix['token_ids']))
        self.assertFalse(self.runtime.bridge.original_model._forward_pre_hooks)
        self.assertTrue(all(not hook.fwd_hooks for hook in self.runtime.bridge.hook_dict.values()))
        after = self.runtime.generate(inputs, self.directory())
        fresh = ModelRuntime(self.config)
        try:
            fresh.load()
            independent = fresh.generate(inputs, self.directory())
            self.assertEqual(read_artifact(after['tokens'], 'json', 1000000), read_artifact(independent['tokens'], 'json', 1000000))
        finally:
            fresh.close()
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))

    def test_forced_fixture_uses_real_hf_loop_and_stops_at_tool_handoff(self):
        import torch
        from transformers import LogitsProcessorList
        suffix = ('<|channel|>analysis<|message|>Inspect.<|end|><|start|>assistant'
                  ' to=functions.execute<|channel|>commentary json<|message|>{"command":["true"]}<|call|>')
        script = self.runtime.tokenizer.encode(suffix, add_special_tokens=False)
        prompt_length = len(self.prepared['token_ids'])

        def force_next(input_ids, scores):
            forced = torch.full_like(scores, float('-inf'))
            forced[:, script[input_ids.shape[-1] - prompt_length]] = 0
            return forced

        original = self.runtime.bridge.hf_generate

        def with_script(*args, **kwargs):
            return original(*args, **kwargs, logits_processor=LogitsProcessorList([force_next]))

        inputs = self.generation_inputs(self.prepared, self.direction()) | {'max_new_tokens': len(script) + 3}
        with patch.object(self.runtime.bridge, 'hf_generate', side_effect=with_script):
            result = self.runtime.generate(inputs, self.directory())
        self.assertEqual(result['status'], 'tool_call')
        self.assertEqual(result['command'], ['true'])
        self.assertEqual(result['generated_tokens'], len(script))
        start, end = result['action_span']
        patched_positions = [p for event in result['hook_events'] for p in event['processed_positions']]
        self.assertEqual(patched_positions, list(range(start - 1, end)))
        self.assertNotIn(end, patched_positions)  # Emitted CALL is never fed back through a decoder.
        captured = self.runtime.capture({'trajectory': result['tokens'], 'target': 'action', 'layers': [1]}, self.directory())
        self.assertEqual(captured['positions'][-1], end - 1)

    def test_handler_artifacts_invalid_arrays_and_context_limits(self):
        request = {'schema_version': 1, 'request_id': 'model-load', 'operation': 'load', 'inputs': {}, 'config': self.config}
        result = self.runtime.handle(request)
        self.assertEqual(result['status'], 'ok')
        for reference in result['artifacts']:
            read_artifact(reference, 'json', 1000000)
        broken = request | {'request_id': 'broken-program', 'operation': 'prepare'}
        with patch.object(self.runtime, 'prepare', side_effect=TypeError('deliberate programming error')):
            with self.assertRaisesRegex(TypeError, 'programming error'):
                self.runtime.handle(broken)
        records = list((self.root / 'records').glob('broken-program-*/record.json'))
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0].read_text())['status'], 'incomplete')
        self.assertIn('programming error', (records[0].parent / 'traceback.txt').read_text())
        bad = self.generation_inputs() | {'max_new_tokens': 512}
        with self.assertRaises(InputError):
            self.runtime.generate(bad, self.directory())
        intervention = self.direction()
        intervention['direction']['sha256'] = '0' * 64
        with self.assertRaises(InputError):
            self.runtime.generate(self.generation_inputs(intervention=intervention), self.directory())

    def test_benign_control_uses_real_hooks_and_keeps_unsuccessful_behavior(self):
        import numpy as np
        import torch
        before = torch.get_rng_state().clone()
        inputs = {'layer': 1, 'strength': .25, 'seed': 17, 'max_new_tokens': 1, 'max_seconds': 60, 'date': '2026-09-10'}
        result = self.runtime.benign_control(inputs, self.directory())
        self.assertEqual(result['status'], 'checked', result)
        self.assertEqual(len(result['generations']), 20)
        self.assertEqual(result['executed_commands'], 0)
        self.assertFalse(result['known_effective'])
        self.assertTrue(all(row['sham_matches_baseline'] for row in result['generations'] if row['arm'] == 'sham'))
        self.assertTrue(all(row['observed_nonzero'] for row in result['generations'] if row['arm'] in ('positive', 'negative', 'random')))
        with np.load(result['training_arrays']['path'], allow_pickle=False) as arrays:
            contrast = (arrays['residuals'][:, 1].astype(np.float64) - arrays['residuals'][:, 0]).mean(axis=0)
        with np.load(result['direction_arrays']['path'], allow_pickle=False) as arrays:
            np.testing.assert_allclose(arrays['direction'], contrast / np.linalg.norm(contrast), rtol=1e-6)
        self.assertFalse(set(result['training_words']) & set(result['validation_words']))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(not hook.fwd_hooks for hook in self.runtime.bridge.hook_dict.values()))

    def check_inputs(self):
        return {'prefix': self.generation_inputs()['prefix'], 'layer': 1, 'delta': .25, 'max_new_tokens': 4,
                'max_seconds': 60, 'rtol': 1e-4, 'atol': 1e-5, 'max_peak_rss_bytes': 8*1024**3, 'max_device_bytes': None}

    def test_runtime_check_actual_parity_cache_controls_and_saved_arrays(self):
        import numpy as np
        import torch
        request = {'schema_version': 1, 'request_id': 'runtime-check', 'operation': 'check', 'config': self.config, 'inputs': self.check_inputs()}
        # A fresh bridge installs upstream capture hooks on its first forward.
        fresh = ModelRuntime(self.config)
        try:
            packet = fresh.handle(request)
        finally:
            fresh.close()
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        self.assertEqual(result['status'], 'passed', [row for row in result['checks'] if not row['passed']])
        self.assertTrue(result['fixture'])
        self.assertFalse(result['experiment_accepted'])
        self.assertEqual([row['name'] for row in result['variants']], ['baseline', 'sham', 'patch', 'S1', 'S2', 'after'])
        self.assertTrue(all(row['generation_forward_count'] == 4 for row in result['variants']))
        self.assertTrue(any(row['check_id'] == 'patch_orthogonal_unchanged' for row in result['checks']))
        reference = self.reference()
        with torch.inference_mode():
            logits = reference(torch.tensor([self.prefix['token_ids']]), use_cache=False, logits_to_keep=1).logits[0, -1].numpy()
        with np.load(io.BytesIO(read_artifact(result['arrays'], 'npz', 67108864)), allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['unmodified_logits_reference'], logits)
            np.testing.assert_array_equal(arrays['unmodified_logits_actual'], logits)
        saved = json.loads(read_artifact(result['check'], 'json', 16777216))
        self.assertEqual(saved['checks'], result['checks'])
        from eval_tampering.experiment import read_runtime_check
        self.assertEqual(read_runtime_check(result['check'], self.runtime.identity)['array_comparisons'], 36)
        corrupted = self.directory() / 'check.json'
        atomic_json(corrupted, saved | {'checks': []})
        with self.assertRaisesRegex(InputError, 'assertion inventory'):
            read_runtime_check(artifact_ref(corrupted, 'json'), self.runtime.identity)
        false_token = json.loads(json.dumps(saved))
        next(row for row in false_token['checks'] if row['check_id'] == 'baseline_token_0')['generated_token'] += 1
        atomic_json(corrupted, false_token)
        with self.assertRaisesRegex(InputError, 'token/native argmax'):
            read_runtime_check(artifact_ref(corrupted, 'json'), self.runtime.identity)
        self.assertGreater(result['process_peak_rss_bytes'], 0)
        self.assertIsNone(result['device_peak_reserved_bytes'])

    def test_uncached_runtime_controls_and_canonical_reader(self):
        from eval_tampering.experiment import read_runtime_check
        identity = self.runtime.identity | {'generation_use_cache': False}
        with patch.object(self.runtime, '_identity', json.dumps(identity)):
            payload = json.loads(json.dumps(self.prefix))
            payload['runtime_sha256'] = fingerprint(identity)
            folder = self.directory()
            atomic_json(folder / 'prefix.json', payload)
            inputs = self.check_inputs() | {'prefix': artifact_ref(folder / 'prefix.json', 'json')}
            result = self.runtime.check(inputs, self.directory())
            self.assertEqual(result['status'], 'passed', [r for r in result['checks'] if not r['passed']])
            self.assertEqual(read_runtime_check(result['check'], identity)['array_comparisons'], 36)
            for variant in result['variants']:
                self.assertTrue(all(row['use_cache'] is False for row in variant['forward_inputs']))
                self.assertEqual([len(row['tokens']) for row in variant['forward_inputs']], list(range(len(payload['token_ids']), len(payload['token_ids'])+4)))
                if variant['name'] == 'patch':
                    self.assertEqual([e['processed_positions'] for e in variant['generation']['hook_events']], [[len(payload['token_ids'])-1]]*4)

    def test_mxfp4_style_mlp_returns_actual_router_logits_without_router_forward(self):
        import torch
        from types import MethodType
        def mlp_forward(module, hidden_states):
            shape = hidden_states.shape
            hidden = hidden_states.reshape(-1, shape[-1])
            logits = torch.nn.functional.linear(hidden, module.router.weight, module.router.bias)
            values, choices = torch.topk(logits, module.router.top_k, dim=-1)
            scores = torch.softmax(values, dim=-1)
            return module.experts(hidden, choices, scores).reshape(shape), logits
        mlp_forward.__module__ = 'transformers.integrations.mxfp4'
        inputs = self.generation_inputs(intervention=self.direction(value=.25))
        diagnostic_inputs = {k: inputs[k] for k in ('prefix', 'intervention', 'max_seconds')}
        with ExitStack() as stack:
            for layer in self.runtime.bridge.original_model.model.layers:
                mlp = getattr(layer.mlp, '_original_component', layer.mlp)
                stack.enter_context(patch.object(mlp, 'forward', MethodType(mlp_forward, mlp)))
                stack.enter_context(patch.object(mlp.router, 'forward', side_effect=AssertionError('router.forward must not run')))
            result = self.runtime.diagnose(diagnostic_inputs, self.directory())
        stats = result['statistics']
        self.assertEqual(stats['router_observation'], 'observed')
        self.assertEqual(stats['router_choice_observation'], 'unavailable')
        self.assertNotIn('changed_router_choice_slots', stats)
        self.assertGreater(stats['router_logit_l2_change'], 0)
        self.assertEqual(stats['downstream_residual_observation'], 'observed')
        self.assertGreater(stats['downstream_residual_l2_change'], 0)

    def test_runtime_check_rejects_invalid_inputs_before_loading_or_forwarding(self):
        inputs = self.check_inputs()
        request = {'schema_version': 1, 'request_id': 'invalid-check', 'operation': 'check', 'config': self.config, 'inputs': inputs | {'max_seconds': -1}}
        with patch.object(self.runtime, 'load', side_effect=AssertionError('must validate first')):
            self.assertEqual(self.runtime.handle(request)['status'], 'error')
        folder = self.directory()
        atomic_json(folder / 'completed.json', self.completed)
        with patch.object(self.runtime.bridge, 'forward', side_effect=AssertionError('no forward')):
            with self.assertRaisesRegex(InputError, 'pre-action prefix'):
                self.runtime.check(inputs | {'prefix': artifact_ref(folder / 'completed.json', 'json')}, self.directory())
        with self.assertRaisesRegex(InputError, 'device-memory'):
            self.runtime.check(inputs | {'max_device_bytes': 1}, self.directory())
        timed = self.directory()
        with self.assertRaisesRegex(InputError, 'deadline'):
            self.runtime.check(inputs | {'max_seconds': 1e-9}, timed)
        saved = json.loads((timed / 'check.json').read_text())
        self.assertEqual(saved['status'], 'incomplete')
        self.assertFalse(next(row['passed'] for row in saved['checks'] if row['check_id'] == 'within_time'))

    def test_runtime_check_keeps_partial_evidence_and_cleans_native_hooks(self):
        import torch
        before = torch.get_rng_state().clone()
        original = self.runtime.generate
        calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = original(*args, **kwargs)
            if calls == 2:
                raise RuntimeError('deliberate check generation failure')
            return result
        folder = self.directory()
        with patch.object(self.runtime, 'generate', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'check generation failure'):
                self.runtime.check(self.check_inputs(), folder)
        saved = json.loads((folder / 'check.json').read_text())
        self.assertEqual(saved['status'], 'incomplete')
        self.assertEqual([row['name'] for row in saved['variants']], ['baseline'])
        self.assertTrue((folder / 'check-arrays.npz').is_file())
        self.assertTrue(next(row['passed'] for row in saved['checks'] if row['check_id'] == 'hooks_clean'))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(self.runtime.bridge.original_model.lm_head._forward_hooks)

    def test_runtime_check_fails_cleanly_when_router_observations_are_unavailable(self):
        from eval_tampering.experiment import read_runtime_check
        original = self.runtime.diagnose
        def missing_router_observation(inputs, directory):
            with ExitStack() as stack:
                for layer in self.runtime.bridge.original_model.model.layers[inputs['intervention']['layer']+1:]:
                    stack.enter_context(patch.object(layer.mlp.router, 'register_forward_hook', return_value=Mock()))
                    stack.enter_context(patch.object(getattr(layer.mlp, '_original_component', layer.mlp), 'register_forward_hook', return_value=Mock()))
                return original(inputs, directory)
        with patch.object(self.runtime, 'diagnose', side_effect=missing_router_observation):
            result = self.runtime.check(self.check_inputs(), self.directory())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual([row['name'] for row in result['variants']], ['baseline', 'sham', 'patch', 'S1', 'S2', 'after'])
        checks = {row['check_id']: row for row in result['checks']}
        for name in ('sham', 'patch', 'S1', 'S2'):
            self.assertFalse(checks[name + '_router_observed']['passed'])
            stats = checks[name + '_actual_change']['statistics']
            self.assertEqual(stats['router_observation'], 'unavailable')
            self.assertNotIn('router_logit_l2_change', stats)
            self.assertEqual(checks[name + '_downstream_change']['passed'], name == 'sham')
            self.assertTrue(checks[name + '_diagnostic_native_logits']['passed'])
        self.assertTrue(checks['rng_preserved']['passed'])
        self.assertTrue(checks['hooks_clean']['passed'])
        self.assertTrue(checks['after_tokens_unchanged']['passed'])
        saved = json.loads(read_artifact(result['check'], 'json', 16777216))
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['checks'], result['checks'])
        self.assertEqual(read_runtime_check(result['check'], self.runtime.identity)['status'], 'failed')

    def test_runtime_check_detects_wrapper_drift(self):
        original = self.runtime.bridge.forward
        def drift(*args, **kwargs):
            return original(*args, **kwargs) + .25
        with patch.object(self.runtime.bridge, 'forward', side_effect=drift):
            result = self.runtime.check(self.check_inputs(), self.directory())
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(next(row['passed'] for row in result['checks'] if row['check_id'] == 'unmodified_logits'))


if __name__ == '__main__':
    unittest.main()
