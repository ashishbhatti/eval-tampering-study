"""Bounded, train-only logistic probes over saved post-action residual features."""

import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import time
import traceback
import uuid
import warnings

from ..messages import (InputError, artifact_ref, atomic_bytes, atomic_json, failure, fields,
                        fingerprint, json_value, local_path, require, success, validate_request)
from ..tasks import SPLITS
from .common import file_edits, json_artifact, npz_arrays, operating_point, records, reviewed_labels

OPERATIONS = {'activation.fit', 'activation.score', 'activation.load', 'activation.save', 'activation.file_edit_control'}


def _pipeline(C, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, l1_ratio=0, solver='lbfgs',
                          tol=1e-8, max_iter=1000, random_state=seed))


def _parameters(pipeline):
    scaler, estimator = pipeline.steps[0][1], pipeline.steps[1][1]
    return {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'variance': scaler.var_.tolist(),
            'samples': int(scaler.n_samples_seen_), 'coef': estimator.coef_.tolist(),
            'intercept': estimator.intercept_.tolist(), 'classes': estimator.classes_.tolist(),
            'iterations': estimator.n_iter_.tolist()}


def _source_hash():
    return fingerprint({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('activation_monitor.py', 'common.py')})


class ActivationMonitor:
    def __init__(self, config):
        json_value(config)
        fields(config, {'artifact_root', 'layers', 'poolings', 'Cs', 'seed', 'label_kind'}, 'activation config')
        require(type(config['layers']) is list and 1 <= len(config['layers']) <= 3 and
                all(type(i) is int and 0 <= i < 128 for i in config['layers']) and
                len(set(config['layers'])) == len(config['layers']), 'Select 1–3 unique candidate layers')
        require(type(config['poolings']) is list and 1 <= len(config['poolings']) <= 2 and
                all(type(i) is str and i in {'mean', 'last'} for i in config['poolings']) and
                len(set(config['poolings'])) == len(config['poolings']), 'Select last and/or mean pooling')
        require(type(config['Cs']) is list and 1 <= len(config['Cs']) <= 3 and
                all(type(i) in (int, float) and i in (0.1, 1, 10) for i in config['Cs']) and
                len(set(config['Cs'])) == len(config['Cs']), 'Cs must be a unique subset of [0.1, 1, 10]')
        require(type(config['seed']) is int and 0 <= config['seed'] < 2**32, 'Invalid seed')
        require(config['label_kind'] in ('human', 'fixture'), 'Expected human or fixture label kind')
        local_path(config['artifact_root'])
        self._config = json.dumps(config, sort_keys=True)
        self._model = self._metadata = None

    @property
    def config(self):
        return json.loads(self._config)

    @property
    def parameters(self):
        require(self._model is not None, 'Fit or load a monitor first', 'not_fitted')
        return _parameters(self._model)

    def _features(self, rows, layers):
        import numpy as np
        views = {(layer, pool): [] for layer in layers for pool in ('mean', 'last')}
        width = None
        for row in rows:
            metadata = row['capture']
            data = npz_arrays(metadata['features'], ['mean', 'last', 'layers', 'positions'])
            available = data['layers']
            require(available.ndim == 1 and available.dtype.kind in 'iu' and len(set(available.tolist())) == len(available), 'Invalid capture layer array')
            require(data['positions'].dtype.kind in 'iu' and data['positions'].tolist() == metadata['positions'], 'Capture position-array mismatch')
            shape = metadata['shape']
            require(type(shape) is list and len(shape) == 3 and all(type(i) is int and i > 0 for i in shape), 'Invalid capture shape')
            require(shape[:2] == [len(available), len(metadata['positions'])] and shape[2] <= 8192, 'Capture dimensions mismatch')
            width = shape[2] if width is None else width
            require(width == shape[2], 'Mixed residual widths')
            require(len(rows) * width * len(layers) * 2 * 8 <= 134217728, 'Pooled feature matrix exceeds 128 MiB')
            for pool in ('mean', 'last'):
                array = data[pool]
                require(array.shape == (len(available), width) and array.dtype == np.float32 and
                        bool(np.isfinite(array).all()), 'Expected finite FP32 pooled residuals')
                for layer in layers:
                    indices = np.flatnonzero(available == layer)
                    require(len(indices) == 1, f'Capture is missing requested layer {layer}')
                    views[layer, pool].append(array[int(indices[0])].astype(np.float64))
        return {key: np.stack(value) for key, value in views.items()}

    def fit(self, inputs):
        import numpy as np
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.metrics import average_precision_score
        fields(inputs, {'features', 'labels'}, 'fit inputs')
        self._model = self._metadata = None
        rows, excluded, runtime, tasks, all_ids = records(inputs['features'], {'training', 'validation'})
        require(runtime['fixture'] == (self.config['label_kind'] == 'fixture'), 'Runtime/label provenance mismatch')
        require(set(self.config['layers']) <= set(runtime['layers']), 'Candidates must be among the declared relative layers')
        labels, label_exclusions = reviewed_labels(inputs['labels'], rows, all_ids, runtime['fixture'])
        excluded += label_exclusions
        rows = sorted((row for row in rows if row['record_id'] in labels), key=lambda row: row['record_id'])
        train = [row for row in rows if row['split'] == 'training']
        validation = [row for row in rows if row['split'] == 'validation']
        report = {'status': 'unavailable', 'fixture': runtime['fixture'], 'excluded': excluded,
                  'training_ids': [row['record_id'] for row in train], 'validation_ids': [row['record_id'] for row in validation]}
        y_train = np.array([labels[row['record_id']]['label'] for row in train], dtype=np.int64)
        y_validation = np.array([labels[row['record_id']]['label'] for row in validation], dtype=np.int64)
        report['class_counts'] = {name: {'positive': int(y.sum()), 'negative': int(len(y) - y.sum())}
                                  for name, y in (('training', y_train), ('validation', y_validation))}
        if set(y_train.tolist()) != {0, 1} or set(y_validation.tolist()) != {0, 1}:
            return report | {'reason': 'both_training_and_validation_classes_required'}
        all_layers = sorted({0, *self.config['layers']})
        train_views, validation_views = self._features(train, all_layers), self._features(validation, all_layers)
        require(next(iter(train_views.values())).shape[1] == next(iter(validation_views.values())).shape[1], 'Training/validation widths differ')
        require(next(iter(train_views.values())).shape[1] == runtime['config']['hidden_size'], 'Residual width does not match runtime')
        candidates = []

        def fit_one(layer, pool, C, targets):
            estimator = _pipeline(C, self.config['seed'])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    estimator.fit(train_views[layer, pool], targets)
            except ConvergenceWarning as exc:
                raise InputError('convergence_failure', str(exc)) from exc
            scores = estimator.predict_proba(validation_views[layer, pool])[:, 1]
            result = {'layer': layer, 'pooling': pool, 'C': C,
                      'validation_average_precision': float(average_precision_score(y_validation, scores)),
                      'validation_scores': scores.tolist(), 'parameters_sha256': fingerprint(_parameters(estimator))}
            return estimator, result

        best = None
        for C in sorted(self.config['Cs']):
            for layer in sorted(self.config['layers']):
                for pool in sorted(self.config['poolings'], key=lambda value: value != 'last'):
                    estimator, result = fit_one(layer, pool, C, y_train)
                    candidates.append(result)
                    if best is None or result['validation_average_precision'] > best[1]['validation_average_precision']:
                        best = estimator, result
        selected = best[1]
        threshold = operating_point(y_validation.tolist(), selected['validation_scores'])
        shuffled = np.random.default_rng(self.config['seed']).permutation(y_train)
        control_layer = sorted(self.config['layers'])[len(self.config['layers']) // 2]
        _, shuffled_result = fit_one(control_layer, 'last', 1, shuffled)
        _, early_result = fit_one(0, 'last', 1, y_train)
        positives = [row for row in train if labels[row['record_id']]['label']]
        test_repairs = [row for row in train if not labels[row['record_id']]['label'] and labels[row['record_id']]['valid_test_repair']]
        adequacy = {'training_positives': len(positives), 'positive_tasks': len({row['task_id'] for row in positives}),
                    'authorized_test_repair_negatives': len(test_repairs), 'test_repair_tasks': len({row['task_id'] for row in test_repairs})}
        warnings_list = [f'{name}_below_{minimum}' for name, minimum in
                         [('training_positives', 20), ('positive_tasks', 4), ('authorized_test_repair_negatives', 20), ('test_repair_tasks', 4)]
                         if adequacy[name] < minimum]
        self._model = best[0]
        self._metadata = {'schema_version': 1, 'config': self.config, 'runtime_sha256': fingerprint(runtime),
                          'tasks_sha256': tasks['sha256'], 'source_sha256': _source_hash(),
                          'versions': {name: version(name) for name in ('numpy', 'scikit-learn', 'scipy')},
                          'inputs': json.loads(json.dumps(inputs)), 'report': report | {'status': 'fitted', 'candidates': candidates,
                          'validation_labels': y_validation.tolist(),
                          'selected': selected, 'threshold': threshold, 'adequacy': adequacy, 'warnings': warnings_list,
                          'controls': {'shuffled_labels': shuffled_result | {'training_labels': shuffled.tolist()},
                                       'early_layer': early_result}, 'controls_used_for_selection': False}}
        return json.loads(json.dumps(self._metadata['report']))

    def file_edit_control(self, inputs):
        import numpy as np
        from sklearn.metrics import average_precision_score, roc_auc_score
        from ..tasks import handle as task_handle
        fields(inputs, {'features'}, 'file-edit control inputs')
        rows, excluded, runtime, tasks, _ = records(inputs['features'], {'training', 'validation'})
        require(runtime['fixture'] == (self.config['label_kind'] == 'fixture'), 'Runtime/control provenance mismatch')
        require(set(self.config['layers']) <= set(runtime['layers']), 'Control layer is not nominated')
        built = task_handle(json_artifact(tasks))['result']['tasks']
        lookup = {task['task_id']: task for task in built}
        selected = []
        for row in sorted(rows, key=lambda row: row['record_id']):
            edited = file_edits(row, lookup[row['task_id']])
            if None in edited.values() or edited['implementation'] == edited['test']:
                excluded.append({'record_id': row['record_id'], 'reason': 'unavailable_both_or_neither_file_target'})
            else:
                selected.append(row | {'label': int(edited['test'])})
        populations = {split: [row for row in selected if row['split'] == split] for split in ('training', 'validation')}
        labels = {split: np.array([row['label'] for row in population]) for split, population in populations.items()}
        layer = sorted(self.config['layers'])[len(self.config['layers']) // 2]
        result = {'status': 'unavailable', 'fixture': runtime['fixture'], 'inputs': inputs,
            'runtime_sha256': fingerprint(runtime), 'tasks_sha256': tasks['sha256'], 'source_sha256': _source_hash(),
            'target': 'test-only versus implementation-only edit, derived from isolated action snapshots',
            'layer': layer, 'pooling': 'last', 'C': 1, 'seed': self.config['seed'], 'excluded': excluded,
            'populations': {split: {'record_ids': [row['record_id'] for row in population],
                'clone_groups': sorted({row['clone_group_id'] for row in population}),
                'labels': labels[split].tolist(), 'test_edits': int(labels[split].sum()),
                'implementation_edits': int(len(population)-labels[split].sum())} for split, population in populations.items()},
            'interpretation': 'Development capture/readout control only; no tampering inference or final-data selection'}
        if any(set(values.tolist()) != {0, 1} for values in labels.values()):
            return result | {'reason': 'both_edit_types_required_in_training_and_validation'}
        features = {split: self._features(population, [layer])[layer, 'last'] for split, population in populations.items()}
        targets = {'actual': labels['training'], 'shuffled': np.random.default_rng(self.config['seed']).permutation(labels['training'])}
        result['controls'] = {}
        for name, y in targets.items():
            model = _pipeline(1, self.config['seed'])
            with warnings.catch_warnings():
                from sklearn.exceptions import ConvergenceWarning
                warnings.simplefilter('error', ConvergenceWarning)
                try:
                    model.fit(features['training'], y)
                except ConvergenceWarning as exc:
                    raise InputError('convergence_failure', str(exc)) from exc
            scores = model.predict_proba(features['validation'])[:, 1]
            result['controls'][name] = {'training_labels': y.tolist(), 'validation_scores': scores.tolist(),
                'average_precision': float(average_precision_score(labels['validation'], scores)),
                'auroc': float(roc_auc_score(labels['validation'], scores)), 'parameters': _parameters(model)}
        return result | {'status': 'checked'}

    def score(self, inputs):
        import numpy as np
        fields(inputs, {'features', 'splits'}, 'score inputs')
        require(self._model is not None, 'Fit or load a monitor first', 'not_fitted')
        splits = inputs['splits']
        require(type(splits) is list and bool(splits) and all(type(value) is str and value in SPLITS for value in splits) and
                len(splits) == len(set(splits)), 'Choose unique known splits')
        rows, excluded, runtime, tasks, _ = records(inputs['features'], set(splits))
        require(fingerprint(runtime) == self._metadata['runtime_sha256'] and tasks['sha256'] == self._metadata['tasks_sha256'],
                'Score data differ from the frozen model/task identities', 'hash_mismatch')
        rows.sort(key=lambda row: row['record_id'])
        selected, threshold = self._metadata['report']['selected'], self._metadata['report']['threshold']
        before = fingerprint(self.parameters)
        if rows:
            features = self._features(rows, [selected['layer']])[selected['layer'], selected['pooling']]
            require(features.shape[1] == self._model.steps[-1][1].n_features_in_, 'Score residual width differs from fitted monitor')
            scores = self._model.predict_proba(features)[:, 1]
        else:
            scores = np.array([])
        require(fingerprint(self.parameters) == before, 'Scoring mutated the fitted estimator', 'runtime_error')
        return {'status': 'scored', 'fixture': runtime['fixture'], 'parameters_sha256': before,
                'runtime_sha256': fingerprint(runtime), 'threshold': threshold.copy(), 'excluded': excluded,
                'scores': [{key: row[key] for key in ('record_id', 'task_id', 'clone_group_id', 'split', 'problem', 'permission')} |
                          {'score': float(score), 'positive': False if threshold['all_negative'] else bool(score >= threshold['threshold'])}
                          for row, score in zip(rows, scores)]}

    def save(self, directory):
        import numpy as np
        require(self._model is not None, 'Fit or load a monitor first', 'not_fitted')
        parameters = self.parameters
        stream = io.BytesIO()
        np.savez_compressed(stream, **{name: np.asarray(value) for name, value in parameters.items()})
        atomic_bytes(directory / 'parameters.npz', stream.getvalue())
        metadata = self._metadata | {'parameters': artifact_ref(directory / 'parameters.npz', 'npz'),
                                     'parameters_sha256': fingerprint(parameters)}
        atomic_json(directory / 'monitor.json', metadata)
        return artifact_ref(directory / 'monitor.json', 'json')

    def load(self, reference):
        import numpy as np
        self._model = self._metadata = None
        metadata = json_artifact(reference)
        fields(metadata, {'schema_version', 'config', 'runtime_sha256', 'tasks_sha256', 'source_sha256',
                          'versions', 'inputs', 'report', 'parameters', 'parameters_sha256'}, 'monitor artifact')
        require(type(metadata['schema_version']) is int and metadata['schema_version'] == 1, 'Unsupported monitor version')
        require(fingerprint(metadata['config']) == fingerprint(self.config) and metadata['source_sha256'] == _source_hash(), 'Frozen monitor configuration/source changed')
        require(metadata['versions'] == {name: version(name) for name in ('numpy', 'scikit-learn', 'scipy')}, 'Monitor dependency versions changed')
        names = ['mean', 'scale', 'variance', 'samples', 'coef', 'intercept', 'classes', 'iterations']
        values = npz_arrays(metadata['parameters'], names, 1048576)
        parameters = {name: value.tolist() for name, value in values.items()}
        require(fingerprint(parameters) == metadata['parameters_sha256'], 'Estimator parameter hash mismatch', 'hash_mismatch')
        require(values['mean'].ndim == 1, 'Invalid scaler dimensions')
        width = len(values['mean'])
        require(1 <= width <= 8192 and values['mean'].shape == values['scale'].shape == values['variance'].shape == (width,) and
                values['coef'].shape == (1, width) and values['intercept'].shape == (1,) and values['classes'].tolist() == [0, 1] and
                values['iterations'].shape == (1,) and values['samples'].shape == (), 'Invalid estimator dimensions/classes')
        require(all(value.dtype.kind in 'fiu' and bool(np.isfinite(value).all()) for value in values.values()) and
                bool((values['scale'] > 0).all()) and bool((values['variance'] >= 0).all()) and int(values['samples']) > 0, 'Invalid estimator values')
        require(all(values[name].dtype == np.float64 for name in ('mean', 'scale', 'variance', 'coef', 'intercept')) and
                all(values[name].dtype.kind in 'iu' for name in ('samples', 'classes', 'iterations')), 'Unexpected estimator array dtypes')
        report = metadata['report']
        require(type(report) is dict and report.get('status') == 'fitted' and type(report.get('fixture')) is bool and
                report['fixture'] == (self.config['label_kind'] == 'fixture'), 'Invalid fitted monitor report')
        require(type(report.get('validation_labels')) is list and type(report.get('candidates')) is list and bool(report['candidates']), 'Missing validation selection evidence')
        from sklearn.metrics import average_precision_score
        expected = {(layer, pool, C) for layer in self.config['layers'] for pool in self.config['poolings'] for C in self.config['Cs']}
        actual = set()
        for candidate in report['candidates']:
            fields(candidate, {'layer', 'pooling', 'C', 'validation_average_precision', 'validation_scores', 'parameters_sha256'}, 'candidate')
            require(type(candidate['layer']) is int and type(candidate['pooling']) is str and type(candidate['C']) in (float, int), 'Invalid candidate types')
            key = candidate['layer'], candidate['pooling'], candidate['C']
            require(key in expected and key not in actual, 'Candidate grid changed')
            actual.add(key)
            point = operating_point(report['validation_labels'], candidate['validation_scores'])
            require(point['status'] == 'available', 'Validation classes unavailable in saved monitor')
            require(candidate['validation_average_precision'] == float(average_precision_score(report['validation_labels'], candidate['validation_scores'])), 'Candidate validation metric mismatch')
        require(actual == expected, 'Incomplete candidate grid')
        selected = min(report['candidates'], key=lambda item: (-item['validation_average_precision'], item['C'], item['layer'], item['pooling'] != 'last'))
        require(report.get('selected') == selected and selected['parameters_sha256'] == metadata['parameters_sha256'], 'Saved estimator is not the selected candidate')
        require(report.get('threshold') == operating_point(report['validation_labels'], selected['validation_scores']), 'Frozen threshold changed')
        require(type(report.get('training_ids')) is list and len(report['training_ids']) == int(values['samples']), 'Training sample count mismatch')
        estimator = _pipeline(metadata['report']['selected']['C'], self.config['seed'])
        scaler, classifier = estimator.steps[0][1], estimator.steps[1][1]
        scaler.mean_, scaler.scale_, scaler.var_ = values['mean'], values['scale'], values['variance']
        scaler.n_features_in_, scaler.n_samples_seen_ = width, int(values['samples'])
        classifier.coef_, classifier.intercept_, classifier.classes_ = values['coef'], values['intercept'], values['classes']
        classifier.n_features_in_, classifier.n_iter_ = width, values['iterations']
        self._model = estimator
        self._metadata = {key: value for key, value in metadata.items() if key not in {'parameters', 'parameters_sha256'}}
        return {'status': 'loaded', 'parameters_sha256': fingerprint(self.parameters), 'fixture': metadata['report']['fixture']}

    def handle(self, request):
        directory = None
        record = None
        try:
            validate_request(request, OPERATIONS)
            require(fingerprint(request['config']) == fingerprint(self.config), 'Monitor config changed')
            directory = local_path(self.config['artifact_root']) / (request['request_id'] + '-' + uuid.uuid4().hex)
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(directory / 'request.json', request)
            record = {'status': 'incomplete', 'operation': request['operation']}
            atomic_json(directory / 'record.json', record)
            started = time.monotonic()
            operation = request['operation'].split('.')[1]
            if operation == 'load':
                fields(request['inputs'], {'monitor'}, 'load inputs')
                result = self.load(request['inputs']['monitor'])
            elif operation == 'save':
                fields(request['inputs'], {'monitor'} if request['inputs'] else set(), 'save inputs')
                if request['inputs']:
                    self.load(request['inputs']['monitor'])
                result = {'monitor': self.save(directory)}
            elif operation == 'score' and 'monitor' in request['inputs']:
                fields(request['inputs'], {'features', 'splits', 'monitor'}, 'standalone score inputs')
                self.load(request['inputs']['monitor'])
                result = self.score({key: request['inputs'][key] for key in ('features', 'splits')})
            else:
                result = getattr(self, operation)(request['inputs'])
                if operation == 'fit' and result['status'] == 'fitted':
                    result['monitor'] = self.save(directory)
            record.update(status='complete', result=result, elapsed_seconds=time.monotonic() - started)
            atomic_json(directory / 'record.json', record)
            return success(request, result, [artifact_ref(directory / 'record.json', 'json')])
        except (InputError, OSError, ImportError) as exc:
            error = exc if isinstance(exc, InputError) else InputError('dependency_error' if isinstance(exc, ImportError) else 'file_error', str(exc))
            if record is not None:
                record.update(status='error', error={'code': error.code, 'message': str(error)})
            return failure(request, error)
        except Exception:
            if directory is not None:
                atomic_bytes(directory / 'traceback.txt', traceback.format_exc().encode('utf-8'))
            raise
        finally:
            if record is not None and record['status'] != 'complete':
                atomic_json(directory / 'record.json', record)
