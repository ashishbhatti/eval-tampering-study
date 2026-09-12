"""Acceptance wiring with a real tiny runtime and explicitly scripted operator/host evidence."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run
from eval_tampering.experiment import acceptance, final_job, read_acceptance, read_manifest, review_template
from eval_tampering.messages import InputError, artifact_ref, atomic_json, success
from eval_tampering.model import ModelRuntime
from eval_tampering.monitors.common import json_artifact
from test_experiment import manifest_fixture
from test_model import AVAILABLE
from test_run import RecordingSandbox, RecordingEvaluator


def accepted_fixture(root, *, max_new_tokens=16):
    """Returns an owned tiny runtime; caller must close it. No human approval is supplied."""
    import torch
    torch.set_num_threads(2)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model_config = {'profile': 'tiny-gpt-oss-cpu', 'model_path': '.cache/gpt-oss-assets',
                    'artifact_root': str(root / 'study/model'), 'max_context_tokens': 4096}
    runtime = ModelRuntime(model_config)
    runtime.load()
    prefix = runtime.prepare({'messages': [{'role': 'user', 'content': 'Fix the implementation.'}],
                              'date': '2026-09-09', 'reasoning_effort': 'medium'})
    prefix['token_ids'] += runtime.tokenizer.encode('<|channel|>analysis<|message|>Inspect the tests.<|end|><|start|>assistant to=functions.execute<|channel|>commentary json<|message|>', add_special_tokens=False)
    prefix['attention_mask'] = [1] * len(prefix['token_ids'])
    atomic_json(root / 'prefix.json', prefix)
    checked = runtime.handle({'schema_version': 1, 'request_id': 'acceptance-runtime', 'operation': 'check', 'config': model_config,
        'inputs': {'prefix': artifact_ref(root / 'prefix.json', 'json'), 'layer': 1, 'delta': .25, 'max_new_tokens': 4,
                   'max_seconds': 60, 'rtol': 1e-4, 'atol': 1e-5, 'max_peak_rss_bytes': 8*1024**3, 'max_device_bytes': None}})
    assert checked['status'] == 'ok' and checked['result']['status'] == 'passed', checked
    data, request = manifest_fixture(root / 'study', runtime=runtime.identity, runtime_check=checked['result']['check'], hosted=True, max_new_tokens=max_new_tokens)
    assert request['inputs']['episode_config']['model'] == model_config
    for role in request['inputs']['readiness']:
        if role != 'sandbox':
            path = root / (role + '.json')
            atomic_json(path, {'fixture': True, 'status': 'scripted_evidence',
                'description': 'Operator-attestation schema fixture; not a real pilot, audit or evaluator acceptance result.'})
            request['inputs']['readiness'][role] = artifact_ref(path, 'json')
    preflight = {'schema_version': 1, 'request_id': 'fixture-host', 'operation': 'preflight',
                 'config': request['inputs']['episode_config']['sandbox'], 'inputs': {}}
    host = {'server_version': 'fixture-daemon', 'image_id': 'sha256:' + '0'*64,
            'image': preflight['config']['image'], 'security_options': ['name=seccomp,profile=fixture']}
    atomic_json(root / 'host.request.json', preflight)
    atomic_json(root / 'host.response.json', success(preflight, host))
    atomic_json(root / 'host.json', {'request': artifact_ref(root / 'host.request.json', 'json'),
                                   'response': artifact_ref(root / 'host.response.json', 'json')})
    request['inputs']['readiness']['sandbox'] = artifact_ref(root / 'host.json', 'json')
    frozen = run.handle(request)
    assert frozen['status'] == 'ok', frozen
    manifest_ref = frozen['result']['manifest']
    manifest = read_manifest(manifest_ref)
    review = review_template(manifest_ref, manifest) | {'reviewer_id': 'fixture-operator',
        'notes': 'Automated fixture attestation. No actual human review, permission to spend or research approval.'}
    for item in review['checks'].values():
        item.update(accepted=True, rationale='Explicitly scripted fixture decision used to exercise the guard.')
    atomic_json(root / 'review.json', review)
    acceptance_inputs = {'manifest': manifest_ref, 'review': artifact_ref(root / 'review.json', 'json')}
    accepted = run.handle(request | {'request_id': 'accepted-fixture', 'operation': 'experiment.accept', 'inputs': acceptance_inputs})
    assert accepted['status'] == 'ok' and accepted['result']['status'] == 'accepted', accepted
    return {'root': root, 'runtime': runtime, 'request': request, 'manifest': manifest, 'manifest_ref': manifest_ref,
            'acceptance': accepted['result']['artifact'], 'acceptance_inputs': acceptance_inputs, 'host': host, 'review': review}


class FixtureHost(RecordingSandbox):
    def __init__(self, root, host):
        super().__init__(root)
        self.host = host

    def handle(self, packet):
        if packet['operation'] == 'preflight':
            self.calls.append(deepcopy(packet))
            return success(packet, self.host)
        return super().handle(packet)


@unittest.skipUnless(AVAILABLE, 'Install tiny-model dependencies and pinned tokenizer assets')
class FinalJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(dir='.cache', prefix='final-job-tests-')
        cls.data = accepted_fixture(cls.work.name)

    @classmethod
    def tearDownClass(cls):
        cls.data['runtime'].close()
        cls.work.cleanup()

    def proof(self, phase='sampling', index=0):
        manifest = self.data['manifest']
        plan = manifest['sampling'] if phase == 'sampling' else manifest['components']['steering']
        return {'acceptance': self.data['acceptance'], 'phase': phase, 'job_id': plan['jobs'][index]['job_id']}

    def test_accepted_fixture_runs_actual_tiny_model_and_records_final_stage_once(self):
        expected, decision = final_job(self.proof())
        self.assertTrue(decision['fixture'])
        self.assertEqual(decision['enabled_phases'], ['sampling', 'steering'])
        host = FixtureHost(self.data['root'], self.data['host'])
        result = run.run_episode(expected, self.data['runtime'], host, RecordingEvaluator())
        self.assertEqual(result['status'], 'ok', result)
        self.assertEqual(result['result']['stage'], 'final')
        self.assertEqual(result['result']['manifest'], self.data['manifest_ref'])
        self.assertTrue(result['result']['fixture'])
        self.assertGreater(result['result']['output_tokens'], 0)
        with self.assertRaisesRegex(InputError, 'Episode already exists'):
            run.run_episode(expected, self.data['runtime'], host, RecordingEvaluator())
        packet = run.handle(self.data['request'] | {'request_id': 'prepared-steering', 'operation': 'experiment.job', 'inputs': self.proof('steering')})
        self.assertEqual(packet['status'], 'ok', packet)
        self.assertEqual(json_artifact(packet['result']['artifact']), final_job(self.proof('steering'))[0])
        self.assertEqual(packet['result']['jobs_executed'], 0)

    def test_mutated_job_and_changed_live_host_fail_before_generation(self):
        expected, _ = final_job(self.proof(index=1))
        mutations = [expected | {'request_id': 'another-attempt'},
            expected | {'config': expected['config'] | {'max_seconds': 700}},
            expected | {'inputs': expected['inputs'] | {'seed': 999}},
            expected | {'inputs': expected['inputs'] | {'arm_id': 'another-arm'}}]
        with patch.object(ModelRuntime, 'load', side_effect=AssertionError('must validate before loading')):
            for request in mutations:
                self.assertEqual(run.handle(request)['status'], 'error')
        host = FixtureHost(self.data['root'], self.data['host'] | {'image_id': 'sha256:' + '1'*64})
        with patch.object(self.data['runtime'], 'handle', side_effect=AssertionError('changed host must stop before model operations')):
            result = run.run_episode(expected, self.data['runtime'], host, RecordingEvaluator())
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error']['code'], 'hash_mismatch')
        with self.assertRaisesRegex(InputError, 'own frozen cohort'):
            final_job(self.proof() | {'phase': 'patch'})

    def test_declined_missing_or_relabelled_reviews_and_budget_changes_cannot_accept(self):
        for kind in ('declined', 'incomplete', 'budget', 'human'):
            review = deepcopy(self.data['review'])
            if kind in ('declined', 'incomplete'):
                review['checks']['design']['accepted'] = False if kind == 'declined' else None
            elif kind == 'budget':
                review['approved_allocations']['sampling']['max_jobs'] += 1
            else:
                review['reviewer_kind'] = 'human'
            path = self.data['root'] / (kind + '-review.json')
            atomic_json(path, review)
            inputs = self.data['acceptance_inputs'] | {'review': artifact_ref(path, 'json')}
            if kind in ('budget', 'human'):
                with self.assertRaises(InputError):
                    acceptance(inputs)
            else:
                result = acceptance(inputs)
                self.assertEqual(result['status'], 'not_ready')
                self.assertEqual(result['enabled_phases'], [])
        changed = json_artifact(self.data['acceptance']) | {'enabled_phases': ['sampling', 'steering', 'patch']}
        path = self.data['root'] / 'changed-acceptance.json'
        atomic_json(path, changed)
        with self.assertRaisesRegex(InputError, 'Acceptance record'):
            read_acceptance(artifact_ref(path, 'json'))

        # Filled operator decisions cannot replace a missing technical prerequisite.
        request = deepcopy(self.data['request'])
        request['request_id'] = 'missing-runtime-check'
        request['inputs']['runtime_check'] = None
        frozen = run.handle(request)
        self.assertEqual(frozen['status'], 'ok', frozen)
        reference = frozen['result']['manifest']
        review = deepcopy(self.data['review'])
        review['manifest'] = reference
        path = self.data['root'] / 'missing-runtime-review.json'
        atomic_json(path, review)
        result = acceptance({'manifest': reference, 'review': artifact_ref(path, 'json')})
        self.assertEqual(result['status'], 'not_ready')
        self.assertIn('runtime_check_missing', result['pending'])
        self.assertEqual(result['enabled_phases'], [])


if __name__ == '__main__':
    unittest.main()
