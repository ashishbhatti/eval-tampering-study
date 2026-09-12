"""Paired patch statistics and reported hook checks over explicit scripted fixtures."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from eval_tampering.analysis import ResultAnalyzer, _generation_observation
from eval_tampering.messages import artifact_ref, atomic_json, fingerprint, read_artifact
from eval_tampering.monitors.common import json_artifact
from test_patch_run import patch_episode_fixture
from test_steering import reviewed_episode


def patch_analysis_fixture(root):
    """Original selection labels differ from fresh outcomes; no real forward occurs."""
    from tokenizers import Tokenizer
    from eval_tampering.model import CALL
    data, planner, instructions, plan = patch_episode_fixture(root)
    root = data['root']
    tokenizer = Tokenizer.from_file(data['config']['tokenizer']['path'])
    body = tokenizer.encode(json.dumps({'command': ['echo', 'fixture']}, separators=(',', ':'))).ids + [CALL]
    assert len(body) <= plan['inputs']['episode_config']['generation']['max_new_tokens']
    manifest = {'schema_version': 1, 'records': []}
    for job in plan['jobs']:
        control, seed, recipient = job['control'], job['seed'], job['recipient']
        if control not in ('baseline', 'target', 'sham', 'random-0'):
            continue
        if control == 'random-0' and not (seed == 11 and recipient in ('g0-r0', 'g0-r1')):
            continue
        if control == 'target' and seed == 14 and recipient == 'g0-r1':
            continue
        weakening = seed in (11, 12) and job['recipient_class'] == 'tampering' if control in ('baseline', 'sham') else seed == 11 and control == 'target'
        evidence = reviewed_episode(root / 'scripted-reviews', job, plan, weakening=weakening)
        episode = json_artifact(evidence['episode'])
        folder = Path(evidence['episode']['path']).parent
        prefix = json_artifact(job['initial_prefix'])
        tokens = prefix | {'token_ids': prefix['token_ids'] + body, 'attention_mask': prefix['attention_mask'] + [1]*len(body)}
        atomic_json(folder / 'generated-tokens.json', tokens)
        hook = job['episode']['inputs']['intervention']
        events = []
        if hook is not None:
            value = 0. if control == 'random-0' else hook['value']
            if control == 'sham' and seed == 13 and recipient == 'g0-r1':
                value = .1  # Explicitly scripted failed zero control.
            position, norm = len(prefix['token_ids'])-1, abs(value)
            events = [{'layer': hook['layer'], 'processed_positions': [position], 'predicted_positions': [position+1],
                'next_token_role': 'action', 'position_count': 1, 'runtime_dtype': 'torch.float32',
                'min_change_norm': norm, 'max_change_norm': norm, 'change_norm_sum': norm,
                'change_norm_squared_sum': norm*norm, 'projection_change_sum': value}]
        if control == 'sham' and seed == 14 and recipient == 'g0-r0':
            events = None
            reviews = json_artifact(evidence['reviews'])
            for row in reviews['records']:
                review = json_artifact(row['review']) | {'uncertain': True}
                atomic_json(Path(row['review']['path']), review)
                row['review'] = artifact_ref(row['review']['path'], 'json')
            atomic_json(Path(evidence['reviews']['path']), reviews)
            evidence['reviews'] = artifact_ref(evidence['reviews']['path'], 'json')
        generated = {'schema_version': 1, 'status': 'ok', 'result': {'status': 'tool_call',
            'tokens': artifact_ref(folder / 'generated-tokens.json', 'json'), 'generated_tokens': len(body), 'hook_events': events}}
        atomic_json(folder / 'generation.json', generated)
        episode['actions'][0].update(generation=artifact_ref(folder / 'generation.json', 'json'), generation_status='tool_call')
        atomic_json(Path(evidence['episode']['path']), episode)
        evidence['episode'] = artifact_ref(evidence['episode']['path'], 'json')
        manifest['records'].append(evidence)
    atomic_json(root / 'patch-outcomes.json', manifest)
    analyzer = ResultAnalyzer({'artifact_root': str(root / 'analysis'), 'label_kind': 'fixture', 'bootstrap_seed': 19})
    frozen = analyzer.handle({'schema_version': 1, 'request_id': 'freeze', 'operation': 'analysis.freeze', 'config': analyzer.config, 'inputs': {}})
    assert frozen['status'] == 'ok', frozen
    return data, planner, plan, analyzer, {'plan': plan['plan'], 'outcomes': artifact_ref(root / 'patch-outcomes.json', 'json'), 'rule': frozen['result']['rule']}


class PatchAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.cache', prefix='patch-analysis-tests-')
        self.data, self.planner, self.plan, self.analyzer, self.inputs = patch_analysis_fixture(self.directory.name)
        self.root = self.data['root']

    def tearDown(self):
        self.directory.cleanup()

    def invoke(self, inputs=None, name='summary'):
        return self.analyzer.handle({'schema_version': 1, 'request_id': name, 'operation': 'analysis.patch', 'config': self.analyzer.config,
            'inputs': self.inputs if inputs is None else inputs})

    def test_fresh_baselines_opposite_directions_missing_controls_and_cli(self):
        import run
        from eval_tampering.model import ModelRuntime
        from eval_tampering.monitors import provider
        from sklearn.linear_model import LogisticRegression
        with patch.object(run, 'run_episode', side_effect=AssertionError('execution')), \
             patch.object(ModelRuntime, 'load', side_effect=AssertionError('model load')), \
             patch.object(provider, 'call_json', side_effect=AssertionError('provider request')), \
             patch.object(LogisticRegression, 'fit', side_effect=AssertionError('fit')):
            packet = self.invoke()
        self.assertEqual(packet['status'], 'ok', packet)
        report = packet['result']
        self.assertEqual((report['planned_episodes'], len(report['summaries'])), (100, 70))
        self.assertEqual((report['eligibility']['source_histories'], report['eligibility']['source_tasks']), (2, 1))
        self.assertEqual(len(report['warnings']), 2)
        target = {row['recipient_class']: row for row in report['summaries'] if row['scope'] == 'pooled' and row['arm_id'] == 'target'}
        left, right = target['tampering']['metrics'], target['repair']['metrics']
        self.assertEqual(left['tampering']['baseline']['rate']['value'], .5)  # Not the original selected positive label.
        self.assertEqual(left['tampering']['paired']['difference']['value'], -2/7)
        self.assertEqual(left['tampering']['paired']['bounds'], {'lower': -.25, 'upper': -.125})
        self.assertEqual(left['repair']['paired']['retention']['value'], 5/3)
        self.assertEqual(right['tampering']['paired']['difference']['value'], .25)
        self.assertEqual(right['repair']['paired']['retention']['value'], .75)
        missing_control = next(row for row in report['coverage'] if row['arm_id'] == 'same_label' and row['recipient_class'] == 'repair')
        self.assertEqual((missing_control['eligible_recipients'], missing_control['planned'], missing_control['unavailable_recipients']), (2, 0, 2))
        self.assertEqual(sum(row['planned'] for row in report['coverage']), 100)
        self.assertEqual(sum(row['supplied_episodes'] for row in report['coverage']), 49)
        rows = json_artifact(report['outcomes'])['rows']
        self.assertEqual(sum(row['status'] == 'reviewed' for row in rows), 48)
        self.assertEqual(sum(row['status'] == 'partial' for row in rows), 1)
        failed = next(row for row in rows if row['instrumentation']['status'] == 'failed')
        self.assertEqual(failed['control'], 'sham')
        self.assertFalse(failed['tampering'])
        self.assertEqual(failed['status'], 'reviewed')
        self.assertEqual(sum(row['instrumentation']['status'] == 'rounded_away' for row in rows), 2)
        sham = next(row for row in report['summaries'] if row['scope'] == 'pooled' and row['arm_id'] == 'sham' and row['recipient_class'] == 'tampering')
        self.assertEqual(sham['metrics']['tampering']['paired']['scorable_count'], 8)  # Failed instrumentation did not remove behavior.
        self.assertEqual(sham['metrics']['tampering']['paired']['difference']['value'], 0.)
        self.assertEqual(len(json_artifact(report['task_counts'])['rows']), 56)
        indexed = {row['job_id']: row for row in rows}
        for pair in json_artifact(report['pairs'])['rows']:
            baseline, arm = indexed[pair['baseline_job_id']], indexed[pair['arm_job_id']]
            self.assertEqual((baseline['recipient'], baseline['seed'], baseline['recipient_class']), (arm['recipient'], arm['seed'], arm['recipient_class']))
            self.assertEqual(baseline['control'], 'baseline')
        table = list(csv.DictReader(read_artifact(report['table'], 'csv', 16777216).decode().splitlines()))
        self.assertEqual(len(table), 1330)
        effect = next(row for row in table if row['scope'] == 'pooled' and row['arm_id'] == 'target' and row['recipient_class'] == 'tampering' and row['outcome'] == 'tampering' and row['estimate'] == 'difference')
        self.assertEqual((int(effect['numerator']), int(effect['denominator']), int(effect['unknown_count'])), (-2, 7, 1))
        self.assertEqual(float(effect['value']), -2/7)
        request = {'schema_version': 1, 'request_id': 'cli', 'operation': 'analysis.patch', 'config': self.analyzer.config, 'inputs': self.inputs}
        atomic_json(self.root / 'cli.json', request)
        cli = subprocess.run([sys.executable, '-B', '-m', 'eval_tampering', 'analysis', '--input', str(self.root / 'cli.json'), '--output', str(self.root / 'cli-result.json')], capture_output=True, text=True)
        self.assertEqual(cli.returncode, 0, cli.stderr)
        result = json.loads((self.root / 'cli-result.json').read_text())['result']
        self.assertEqual(result['summaries'], report['summaries'])
        self.assertEqual(result['coverage'], report['coverage'])

    def test_wrong_token_prefix_rejected_and_wrong_position_reported_as_failed(self):
        manifest = json_artifact(self.inputs['outcomes'])
        job = next(job for job in self.plan['jobs'] if job['control'] == 'target' and job['recipient'] == 'g0-r1' and job['seed'] == 11)
        evidence = next(row for row in manifest['records'] if row['job_id'] == job['job_id'])
        episode = json_artifact(evidence['episode'])
        generated = json_artifact(episode['actions'][0]['generation'])
        tokens = json_artifact(generated['result']['tokens'])
        tokens['token_ids'][0] += 1
        atomic_json(self.root / 'changed-tokens.json', tokens)
        original = deepcopy(generated)
        generated['result']['tokens'] = artifact_ref(self.root / 'changed-tokens.json', 'json')
        atomic_json(self.root / 'changed-generation.json', generated)
        episode['actions'][0]['generation'] = artifact_ref(self.root / 'changed-generation.json', 'json')
        atomic_json(self.root / 'changed-episode.json', episode)
        evidence['episode'] = artifact_ref(self.root / 'changed-episode.json', 'json')
        atomic_json(self.root / 'changed-outcomes.json', manifest)
        result = self.invoke(self.inputs | {'outcomes': artifact_ref(self.root / 'changed-outcomes.json', 'json')}, 'wrong-prefix')
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error']['code'], 'hash_mismatch')
        self.assertIn('prefix/runtime/token accounting', result['error']['message'])
        original['result']['hook_events'][0]['processed_positions'][0] -= 1
        atomic_json(self.root / 'changed-generation.json', original)
        episode['actions'][0]['generation'] = artifact_ref(self.root / 'changed-generation.json', 'json')
        atomic_json(self.root / 'changed-episode.json', episode)
        observation = _generation_observation(job, {'status': 'reviewed', 'evidence': evidence | {'episode': artifact_ref(self.root / 'changed-episode.json', 'json')}})
        self.assertEqual(observation['status'], 'failed')
        self.assertEqual(observation['turns'][0]['status'], 'failed_P1_position_or_count')
        # A malformed later prefix must become an input error, not an unhandled KeyError.
        manifest = json_artifact(self.inputs['outcomes'])
        evidence = next(row for row in manifest['records'] if row['job_id'] == job['job_id'])
        episode = json_artifact(evidence['episode'])
        later = deepcopy(episode['actions'][0])
        later.update(turn=1, record_id=fingerprint({'episode': job['job_id'], 'turn': 1})[:32],
            before_snapshot=later['after_snapshot'], assessment=None)
        request = json_artifact(later['generation_request'])
        atomic_json(self.root / 'malformed-prefix.json', {'missing': 'token fields'})
        request['inputs'].update(prefix=artifact_ref(self.root / 'malformed-prefix.json', 'json'),
            seed=int(fingerprint({'seed': job['seed'], 'turn': 1})[:8], 16), intervention=None)
        atomic_json(self.root / 'later-request.json', request)
        later['generation_request'] = artifact_ref(self.root / 'later-request.json', 'json')
        episode['actions'].append(later)
        atomic_json(self.root / 'later-episode.json', episode)
        evidence['episode'] = artifact_ref(self.root / 'later-episode.json', 'json')
        atomic_json(self.root / 'later-outcomes.json', manifest)
        result = self.invoke(self.inputs | {'outcomes': artifact_ref(self.root / 'later-outcomes.json', 'json')}, 'malformed-later-prefix')
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error']['code'], 'invalid_input')
        self.assertIn('saved generation prefix', result['error']['message'])

    def test_no_eligible_pairs_is_unavailable_with_no_invented_slots(self):
        labels = json_artifact(self.data['labels'])
        for row in labels['records']:
            if row['record_id'].startswith('g1-'):
                path = self.root / (row['record_id'] + '-uncertain.json')
                atomic_json(path, json_artifact(row['review']) | {'uncertain': True})
                row['review'] = artifact_ref(path, 'json')
        atomic_json(self.root / 'uncertain-labels.json', labels)
        instructions = json_artifact(self.plan['inputs']['instructions'])
        folder = self.root / 'empty-instructions'
        folder.mkdir()
        empty = self.planner.make_instructions(instructions['inputs'] | {'split': 'validation',
            'labels': artifact_ref(self.root / 'uncertain-labels.json', 'json')}, folder)
        folder = self.root / 'empty-episodes'
        folder.mkdir()
        plan = self.planner.plan_patch_episodes(self.plan['inputs'] | {'instructions': empty['instructions']}, folder)
        atomic_json(self.root / 'empty-outcomes.json', {'schema_version': 1, 'records': []})
        packet = self.invoke(self.inputs | {'plan': plan['plan'], 'outcomes': artifact_ref(self.root / 'empty-outcomes.json', 'json')}, 'empty')
        self.assertEqual(packet['status'], 'ok', packet)
        result = packet['result']
        self.assertEqual(result['comparison_status'], 'unavailable')
        self.assertEqual(result['planned_episodes'], 0)
        self.assertEqual(result['summaries'], [])
        self.assertEqual(result['eligibility']['source_histories'], 0)


if __name__ == '__main__':
    unittest.main()
