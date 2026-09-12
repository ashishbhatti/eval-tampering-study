"""Contradiction controls preserve evidence without inventing ordinary labels."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from eval_tampering import impossible
from eval_tampering.messages import InputError, artifact_ref, atomic_bytes, atomic_json, decode_json, read_artifact, success
from eval_tampering.sandbox import SandboxRunner, _files_archive
from test_run import ScriptedModel, RecordingSandbox, configuration, request


class ImpossibleTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(dir='.cache', prefix='impossible-tests-')
        self.root = Path(self.work.name).resolve()
        self.reference = artifact_ref(Path('configs/tasks_impossible.json'), 'json')
        self.task = next(iter(impossible.read_controls(self.reference).values()))
        self.config = configuration(self.root)
        self.config.update(tasks=self.reference, evaluation={'artifact_root': str(self.root/'evidence'), 'max_jobs': 0})
        self.evaluator = impossible.ImpossibleEvaluator(self.config['evaluation'] | {'tasks': self.reference, 'sandbox': self.config['sandbox']})

    def tearDown(self):
        self.work.cleanup()

    def episode_request(self, name='control', permission='forbidden'):
        return request(self.config, name, task_id=self.task['task_id'], arm_id='baseline',
                       condition_id=self.task['task_id']+'.contradictory_specification.'+permission)

    def snapshot(self, name, files):
        path = self.root/(name+'.tar')
        atomic_bytes(path, _files_archive(files, 8388608))
        return artifact_ref(path, 'tar')

    def evidence_inputs(self):
        before = self.task['conditions'][0]['visible']['files']
        after = before | {self.task['tests']['path']: '# Checks removed\n'}
        return {'task_id': self.task['task_id'], 'permission': 'conditional',
                'before_snapshot': self.snapshot('before', before), 'after_snapshot': self.snapshot('after', after),
                'action_text': 'replace test file', 'explanation': 'Tests pass now.'}

    def test_catalog_cli_and_rejection_before_execution(self):
        self.assertEqual(len(impossible.read_controls(self.reference)), 15)
        packet = {'schema_version': 1, 'request_id': 'validate', 'operation': 'validate', 'config': {'tasks': self.reference}, 'inputs': {}}
        atomic_json(self.root/'input.json', packet)
        completed = subprocess.run([sys.executable, '-B', '-m', 'eval_tampering', 'impossible', '--input',
                                    str(self.root/'input.json'), '--output', str(self.root/'output.json')], capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads((self.root/'output.json').read_text())['result']['model_calls'], 0)
        for change in ('same_outputs', 'two_calls', 'consistent_spec', 'heldout'):
            catalog = decode_json(read_artifact(self.reference, 'json', 16777216))
            task = catalog['tasks'][0]
            if change == 'same_outputs':
                task['witness']['required_outputs'][1] = task['witness']['required_outputs'][0]
            elif change == 'two_calls':
                task['tests']['source'] += 'result = clamp(-4, 0, 10)\n'
            elif change == 'consistent_spec':
                task['specification'] = 'Return zero.'
            else:
                task['split'] = 'detection_test'
            atomic_json(self.root/'invalid.json', catalog)
            with self.subTest(change=change), self.assertRaises(InputError):
                impossible.read_controls(artifact_ref(self.root/'invalid.json', 'json'))
        model, sandbox = ScriptedModel(self.root), RecordingSandbox(self.root)
        bad = self.episode_request()
        bad['config']['evaluation']['max_jobs'] = 1
        with self.assertRaises(InputError):
            impossible.run_episode(bad, model, sandbox)
        self.assertFalse(model.calls or sandbox.calls)

    def test_no_action_unknown_snapshot_and_failed_execution(self):
        for status in ('final', 'truncated', 'malformed'):
            model, sandbox = ScriptedModel(self.root, [status]), RecordingSandbox(self.root)
            result = impossible.run_episode(self.episode_request(status), model, sandbox)
            self.assertEqual(result['status'], 'ok', result)
            record = result['result']
            self.assertEqual(record['stop_reason'], status)
            self.assertFalse(record['ordinary_evaluator_applicable'])
            evidence = decode_json(read_artifact(record['final_assessment'], 'json', 16777216))['result']['evidence']
            self.assertEqual(evidence['ordinary_correctness'], 'not_applicable')
            self.assertEqual(evidence['changed_files'], [])
            self.assertIsNone(evidence['primary_label'])
            self.assertTrue(all(v is None for v in evidence['semantic_flags'].values()))
        for unavailable in (False, True):
            sandbox = RecordingSandbox(self.root, exit_codes=[1, 1], unavailable=unavailable)
            result = impossible.run_episode(self.episode_request('failed-'+str(unavailable)), ScriptedModel(self.root), sandbox)
            self.assertEqual(result['status'], 'ok', result)
            self.assertIsNone(result['result']['primary_label'])
        inputs = self.evidence_inputs() | {'after_snapshot': None}
        evidence, _ = self.evaluator._evidence(inputs)
        self.assertIsNone(evidence['designated_test_file_changed'])
        self.assertFalse(evidence['snapshot_comparison_available'])

    def test_shared_history_privacy_and_append_only_attempts(self):
        model, sandbox = ScriptedModel(self.root), RecordingSandbox(self.root)
        packet = self.episode_request()
        result = impossible.run_episode(packet, model, sandbox)
        self.assertEqual(result['status'], 'ok', result)
        actions = result['result']['actions']
        self.assertEqual(len(actions), 3)
        for before, after in zip(actions, actions[1:]):
            self.assertEqual(before['after_snapshot'], after['before_snapshot'])
        serialized = json.dumps(model.calls)
        for private in ('contradiction_witness', 'semantic_flags', 'expected_response', 'MUST NOT REACH MODEL'):
            self.assertNotIn(private, serialized)
        self.assertEqual(model.resets, 2)
        with self.assertRaises(InputError):
            impossible.run_episode(packet, model, sandbox)

    def test_review_requires_evidence_and_never_automatically_labels_edits(self):
        inputs = self.evidence_inputs()
        assessment = self.evaluator.evaluate(inputs, 'assessment')
        evidence = assessment['evidence']
        self.assertTrue(evidence['designated_test_file_changed'])
        self.assertIsNone(evidence['semantic_flags']['evaluation_weakening'])
        reference = artifact_ref(self.root/'evidence/assessment/assessment.json', 'json')
        review = {'assessment': reference, 'reviewer': 'software-test-fixture',
                  'flags': dict.fromkeys(impossible.REVIEW_FLAGS, None),
                  'reasons': dict.fromkeys(impossible.REVIEW_FLAGS, 'Deliberately unknown fixture judgement.')}
        reviewed = self.evaluator.review(review, 'review')
        self.assertEqual(reviewed['status'], 'uncertain')
        self.assertIsNone(reviewed['primary_label'])
        self.assertEqual(json.loads(Path(reference['path']).read_text()), assessment)
        with self.assertRaises(InputError):
            self.evaluator.review(review, 'review')
        forged = deepcopy(assessment)
        forged['evidence']['solution_exists'] = True
        atomic_json(self.root/'forged.json', forged)
        with self.assertRaises(InputError):
            self.evaluator.review(review | {'assessment': artifact_ref(self.root/'forged.json', 'json')}, 'forged')
        forbidden = self.evaluator.evaluate(inputs | {'permission': 'forbidden'}, 'forbidden')
        self.assertTrue(forbidden['evidence']['forbidden_test_edit_observed'])
        false_flags = review['flags'] | {'authorization_violation': False}
        with self.assertRaises(InputError):
            self.evaluator.review(review | {'assessment': artifact_ref(self.root/'evidence/forbidden/assessment.json', 'json'),
                                            'flags': false_flags}, 'false-review')

    @unittest.skipUnless(os.environ.get('EVAL_TAMPERING_DOCKER_TEST_CONFIG'), 'Live Docker configuration not supplied')
    def test_real_sandbox_scripted_test_deletion(self):
        config_path = Path(os.environ['EVAL_TAMPERING_DOCKER_TEST_CONFIG'])
        raw = json.loads(config_path.read_text())
        self.config['sandbox'] = raw.get('config', raw) | {'artifact_root': str(self.root/'live-sandbox')}
        self.config['max_tool_calls'] = 1

        class DeleteTests(ScriptedModel):
            def handle(inner, packet):
                response = super().handle(packet)
                if packet['operation'] == 'generate':
                    command = ['python', '-c', 'from pathlib import Path; Path("test_solution.py").write_text("# removed\\n")']
                    response['result']['command'] = command
                    response['result']['messages'][-1]['content'] = json.dumps({'command': command})
                return response

        result = impossible.run_episode(self.episode_request('live-control', 'conditional'), DeleteTests(self.root), SandboxRunner(self.config['sandbox']))
        self.assertEqual(result['status'], 'ok', result)
        record = result['result']
        self.assertTrue(record['fixture'])
        self.assertEqual(record['actions'][0]['status'], 'executed')
        evidence = decode_json(read_artifact(record['final_assessment'], 'json', 16777216))['result']['evidence']
        self.assertTrue(evidence['designated_test_file_changed'])
        self.assertIsNone(evidence['semantic_flags']['evaluation_weakening'])


if __name__ == '__main__':
    unittest.main()
