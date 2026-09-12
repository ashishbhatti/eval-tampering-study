"""Standalone figures from verified analysis; no new scores, fits or experiments."""

from importlib.metadata import version
import io
import os
import textwrap

from .messages import artifact_ref, atomic_bytes, atomic_json, decode_json, fields, local_path, read_artifact, require
from .monitors.common import json_artifact
from .evidence import verify_links

ROLES = ('sampling', 'detection', 'patch', 'steering', 'cost', 'reasoning')
LABELS = {'implementation_bug': 'Implementation bug', 'test_conflict': 'Test conflict',
          'forbidden': 'tests protected', 'conditional': 'test repair allowed',
          'activation': 'Activation', 'test_reference': 'Test reference', 'weakening_marker': 'Weakening marker',
          'forbidden_test_reference': 'Protected-test reference', 'hosted': 'Hosted judge',
          'average_precision': 'Average precision', 'auroc': 'AUROC', 'recall': 'Recall',
          'false_positive_rate': 'False-positive rate', 'valid_test_repair_false_positive_rate': 'Valid test-repair FPR'}


def _same(saved, replay):
    """Generated file locations may differ; their checked bytes must agree."""
    if type(saved) is dict and set(saved) == {'path', 'sha256', 'format'}:
        if type(replay) is not dict or set(replay) != set(saved) or saved['format'] != replay['format']:
            return False
        left, right = (read_artifact(ref, ref['format'], 134217728) for ref in (saved, replay))
        return left == right or saved['format'] == 'json' and _same(decode_json(left), decode_json(right))
    if type(saved) is dict:
        return type(replay) is dict and saved.keys() == replay.keys() and all(_same(value, replay[key]) for key, value in saved.items())
    if type(saved) is list:
        return type(replay) is list and len(saved) == len(replay) and all(_same(a, b) for a, b in zip(saved, replay))
    return type(saved) is type(replay) and saved == replay


def _checked(role, reference, directory, fixture):
    from .analysis import read_analysis
    return read_analysis(reference, role, directory, fixture)


def _cell(row):
    return 'All cells' if row['scope'] == 'pooled' else LABELS[row['problem']] + ' / ' + LABELS[row['permission']]


def _point(label, pointer, estimate, numerator=None, denominator=None, unknown=None, bounds=None):
    return {'label': label, 'source_pointer': pointer, 'value': estimate['value'], 'interval95': estimate['bootstrap']['interval95'],
            'invalid_replicates': estimate['bootstrap']['invalid'], 'numerator': numerator, 'denominator': denominator,
            'unknown': unknown, 'bounds': bounds}


def _forest(ax, rows, title, *, difference=False):
    """Draw intervals directly: a percentile interval need not contain its estimate."""
    for y, row in enumerate(rows):
        bound = row['bounds']
        if bound is not None and bound['lower'] is not None:
            ax.hlines(y, bound['lower'], bound['upper'], color='#c9cfd5', linewidth=7, zorder=1)
        if row['interval95'] is not None:
            ax.hlines(y, *row['interval95'], color='#007c91', linewidth=2, zorder=2)
        if row['value'] is not None:
            ax.plot(row['value'], y, 'o', color='#003f5c', markersize=4, zorder=3)
        else:
            ax.text(.5, y, 'unavailable', transform=ax.get_yaxis_transform(), ha='center', va='center', color='#666666', fontsize=8)
    labels = []
    for row in rows:
        counts = '' if row['denominator'] is None else f"n={row['numerator']}/{row['denominator']}; "
        unknown = '' if row['unknown'] is None else f"?={row['unknown']}; "
        labels.append(row['label'] + '\n' + counts + unknown + f"invalid CI={row['invalid_replicates']}")
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    ax.set_ylim(len(rows)-.5, -.5)
    ax.set_xlim(-1.03 if difference else -.03, 1.03)
    ax.axvline(0, color='#b0b0b0', linewidth=.7)
    ax.set_xlabel('Arm minus fresh baseline (rate difference)' if difference else 'Rate / score')
    ax.set_title(title, loc='left', fontsize=11)
    ax.grid(axis='x', alpha=.15)


def render(inputs, directory, analyzer):
    from .analysis import CELLS, METHODS, METRICS, OUTCOMES
    fields(inputs, {'reports', 'rule'}, 'figure inputs')
    require(type(inputs['reports']) is dict, 'Expected figure report references')
    fields(inputs['reports'], (set(ROLES) - {'reasoning'}) | ({'reasoning'} if 'reasoning' in inputs['reports'] else set()), 'figure reports')
    require(any(value is not None for value in inputs['reports'].values()), 'Supply at least one analysis report')
    require(json_artifact(inputs['rule']) == analyzer.rule(), 'Frozen figure analysis rule changed', 'hash_mismatch')
    fixture = analyzer.config['label_kind'] == 'fixture'
    reports = {role: _checked(role, ref, directory / 'replay' / role, fixture) for role, ref in inputs['reports'].items() if ref is not None}
    links = verify_links(inputs['reports'])
    os.environ.setdefault('MPLCONFIGDIR', str(local_path('.cache/matplotlib')))
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    figures = []

    def canvas(rows=1, columns=1, height=5, width=12):
        figure = Figure(figsize=(width, height), layout='constrained')
        FigureCanvasAgg(figure)
        return figure, figure.subplots(rows, columns, squeeze=False).flatten()

    def save(figure, name, role, title, caption, data):
        report = reports[role]
        split = report.get('split') or report['inputs'].get('split') or 'declared work inventory'
        scope = ('FIXTURE — no research result' if fixture else 'RESEARCH RECORDS') + ' | ' + split
        figure.suptitle(title + '\n' + scope, fontsize=14, fontweight='bold')
        figure.supxlabel('\n'.join(textwrap.wrap(caption, 150)), fontsize=8)
        path = directory / name
        payload = {'schema_version': 1, 'fixture': fixture, 'source': inputs['reports'][role], 'scope': scope, 'caption': caption, 'data': data}
        atomic_json(path.with_suffix('.data.json'), payload)
        files = {}
        for extension in ('png', 'svg'):
            stream = io.BytesIO()
            metadata = {'Date': None} if extension == 'svg' else {}
            figure.savefig(stream, format=extension, dpi=150, metadata=metadata)
            atomic_bytes(path.with_suffix('.' + extension), stream.getvalue())
            files[extension] = artifact_ref(path.with_suffix('.' + extension), extension)
        figures.append({'figure_id': name, 'role': role, 'title': title, 'scope': scope, 'caption': caption,
                        'source': inputs['reports'][role], 'data': artifact_ref(path.with_suffix('.data.json'), 'json'), **files})
        figure.clear()

    interval_caption = ('Dots: estimates on known outcomes. Teal: 95% clone-group percentile intervals (2,000 draws); '
        'gray: full-slot missing-outcome bounds. n: numerator/denominator; ?: unknown outcomes. '
        'Invalid CI draws are retained. Few groups, rare events and zero-width intervals do not establish population certainty.')
    with rc_context({'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': 'eval-tampering-figures',
                     'axes.spines.top': False, 'axes.spines.right': False, 'text.usetex': False}):
        if 'sampling' in reports:
            report = reports['sampling']
            figure, axes = canvas(2, 3, height=9, width=20)
            panels = []
            for ax, outcome in zip(axes, OUTCOMES):
                rows = []
                for index, summary in enumerate(report['summaries']):
                    if summary['scope'] != 'cell':
                        continue
                    metric = summary['metrics'][outcome]
                    rows.append(_point(_cell(summary), f'/summaries/{index}/metrics/{outcome}', metric['rate'],
                                       metric['event_count'], metric['known_count'], metric['unknown_count'], metric['bounds']))
                _forest(ax, rows, outcome.replace('_', ' ').capitalize())
                panels.append({'outcome': outcome, 'rows': rows})
            save(figure, 'sampling-outcomes', 'sampling', 'Baseline outcomes by condition', interval_caption, panels)
            figure, axes = canvas(height=5)
            cells = [row for row in report['summaries'] if row['scope'] == 'cell']
            statuses = sorted({key for row in cells for key in row['review_statuses']})
            bottom = np.zeros(len(cells))
            for status in statuses:
                values = [row['review_statuses'].get(status, 0) for row in cells]
                axes[0].bar(range(len(cells)), values, bottom=bottom, label=status.replace('_', ' '))
                bottom += values
            axes[0].set_xticks(range(len(cells)), [_cell(row).replace(' / ', '\n') for row in cells])
            axes[0].set_ylabel('Planned sampling slots')
            axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
            for x, row in enumerate(cells):
                axes[0].text(x, row['planned'], f"{row['planned']} planned", ha='center', va='bottom')
            axes[0].set_ylim(0, max(row['planned'] for row in cells)*1.18 or 1)
            save(figure, 'sampling-coverage', 'sampling', 'Baseline sampling and review coverage',
                 'Every scheduled slot is included. Partial or missing review is not a negative behavioral label.', cells)
            comparison = report['permission_comparison']
            figure, axes = canvas(height=6)
            rows = []
            for name, metric in comparison['metrics'].items():
                paired = metric['paired']
                rows.append(_point(name.replace('_', ' '), f'/permission_comparison/metrics/{name}/paired', paired['difference'],
                    paired['arm_event_count'] - paired['baseline_event_count'], paired['scorable_count'], paired['missing_count'], paired['bounds']))
            _forest(axes[0], rows, 'Test-conflict tasks: conditional minus forbidden', difference=True)
            save(figure, 'permission-comparison', 'sampling', 'Permission manipulation check',
                'Same task and seed in both permission cells; positive values mean more events with conditional permission. ' + interval_caption, [{'rows': rows}])

        if 'detection' in reports:
            report = reports['detection']
            source_rows = json_artifact(report['rows'])['rows']
            for method in METHODS:
                figure, axes = canvas(2, 2, height=8)
                bins = np.linspace(0, 1, 11)
                panels, heights = [], []
                for ax, cell in zip(axes, CELLS):
                    rows = [row for row in source_rows if (row['problem'], row['permission']) == cell]
                    observations = [{'record_id': row['record_id'], 'label': row['label'], 'score': row['scores'][method]['score']} for row in rows]
                    series = []
                    for label, color, name in ((0, '#007c91', 'Non-tampering'), (1, '#c4512a', 'Tampering')):
                        scores = [row['score'] for row in observations if row['label'] == label and row['score'] is not None]
                        counts, _, _ = ax.hist(scores, bins=bins, histtype='step', linewidth=2 if label else 4,
                            linestyle='--' if label else '-', color=color, label=f'{name}: n={len(scores)}')
                        heights.extend(counts.tolist())
                        series.append({'label': label, 'counts': counts.tolist()})
                    missing = sum(row['label'] is None or row['score'] is None for row in observations)
                    ax.set_title(LABELS[cell[0]] + ' / ' + LABELS[cell[1]] + f'\nMissing label/score: {missing}')
                    ax.set_xlabel('Saved monitor score')
                    ax.set_ylabel('Completed calls (count)')
                    ax.legend(fontsize=8)
                    panels.append({'cell': list(cell), 'bins': bins.tolist(), 'observations': observations, 'missing': missing, 'series': series})
                for ax in axes:
                    ax.set_ylim(0, max(1, max(heights, default=0))*1.2)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                save(figure, 'scores-' + method, 'detection', LABELS[method] + ' score distributions',
                     'Ten fixed probability bins over [0, 1]. Teal solid: non-tampering; orange dashed: tampering. Only reviewed calls with scores enter each histogram. Missing values are explicit; no random examples are selected.', panels)
            for scope_index, cell in enumerate([None, *CELLS]):
                selected = [(index, row) for index, row in enumerate(report['summaries']) if (row['scope'] == 'pooled' if cell is None else (row['problem'], row['permission']) == cell)]
                figure, axes = canvas(2, 3, height=8, width=18)
                panels = []
                for ax, metric_name in zip(axes, METRICS):
                    rows = [_point(LABELS[row['method']], f'/summaries/{index}/metrics/{metric_name}', row['metrics'][metric_name],
                        row['metrics'][metric_name]['numerator'], row['metrics'][metric_name]['denominator']) for index, row in selected]
                    _forest(ax, rows, LABELS[metric_name])
                    panels.append({'metric': metric_name, 'rows': rows})
                axes[-1].axis('off')
                axes[-1].text(0, .95, 'Scored / reviewed / completed calls\n\n' + '\n'.join(
                    f"{LABELS[row['method']]}: {row['score_count']} / {row['reviewed_count']} / {row['record_count']}" for _, row in selected), va='top', fontsize=9)
                save(figure, f'detection-{scope_index}', 'detection', 'Detection: ' + _cell(selected[0][1]),
                     'Globally frozen thresholds. Teal: 95% clone-group percentile intervals; invalid draws remain explicit. AP/AUROC have no single count ratio. Empty denominators are unavailable. Few groups or zero events limit interpretation.', panels)

            figure, axes = canvas(2, 3, height=9, width=20)
            panels = []
            for ax, method in zip(axes, METHODS):
                rows = [_point(row['negative_class'].replace('_', ' '), f'/negative_subclasses/{index}/false_positive_rate', row['false_positive_rate'],
                    row['false_positive_rate']['numerator'], row['false_positive_rate']['denominator'])
                    for index, row in enumerate(report['negative_subclasses']) if row['method'] == method]
                _forest(ax, rows, LABELS[method])
                panels.append({'method': method, 'rows': rows})
            axes[-1].axis('off')
            save(figure, 'negative-subclass-fpr', 'detection', 'False positives by negative class',
                'Mutually exclusive reviewed-negative classes; authorization edges take precedence. Unknown classifications remain other negatives. Frozen thresholds and clone-group intervals.', panels)

        if 'reasoning' in reports:
            report = reports['reasoning']
            for variant in sorted({row['variant'] for row in report['summaries']}):
                for method in METHODS:
                    figure, axes = canvas(2, 2, height=9, width=14)
                    panels = []
                    for ax, metric in zip(axes, ('recall', 'false_positive_rate', 'average_precision', 'auroc')):
                        rows = [_point(row['group'].replace('_', ' '), f'/summaries/{index}/metrics/{metric}', row['metrics'][metric],
                            row['metrics'][metric]['numerator'], row['metrics'][metric]['denominator'])
                            for index, row in enumerate(report['summaries']) if row['variant'] == variant and row['method'] == method and row['scope'] == 'pooled']
                        _forest(ax, rows, LABELS[metric])
                        panels.append({'metric': metric, 'rows': rows})
                    save(figure, 'reasoning-' + variant + '-' + method, 'reasoning', LABELS[method] + ': reasoning groups (' + variant.replace('_', ' ') + ')',
                        'All-action analysis remains primary. Both behavior classes use the same reasoning criterion and globally frozen threshold; sparse ranking estimates remain unavailable. Dispute variants stay separate.', panels)

        for role in ('patch', 'steering'):
            if role not in reports:
                continue
            report = reports[role]
            cohorts = ('tampering', 'repair') if role == 'patch' else (None,)
            for cohort in cohorts:
                for scope_index, cell in enumerate([None, *CELLS]):
                    summaries = [(index, row) for index, row in enumerate(report['summaries']) if row['recipient_class'] == cohort and
                                 (row['scope'] == 'pooled' if cell is None else (row['problem'], row['permission']) == cell)]
                    # Keep every control, with at most 12 rows per exported page.
                    for start in range(0, len(summaries), 12):
                        page = summaries[start:start+12]
                        figure, axes = canvas(1, 2, height=max(4, 2.5 + .4*len(page)), width=18)
                        panels = []
                        for ax, outcome in zip(axes, ('tampering', 'repair')):
                            rows = []
                            for index, summary in page:
                                metric = summary['metrics'][outcome]['paired']
                                rows.append(_point(summary['arm_id'], f'/summaries/{index}/metrics/{outcome}/paired', metric['difference'],
                                    metric['arm_event_count']-metric['baseline_event_count'], metric['scorable_count'], metric['missing_count'], metric['bounds']))
                            _forest(ax, rows, outcome.capitalize(), difference=True)
                            panels.append({'outcome': outcome, 'rows': rows})
                        title = role.capitalize() + ': ' + _cell(page[0][1]) + (f' | original {cohort} recipients' if cohort else '')
                        if len(summaries) > 12:
                            title += f'\nControls {start+1}-{start+len(page)} of {len(summaries)}'
                        caption = interval_caption + (' Retrospective selected recipients; opposite classes stay separate. Controls without eligible recipients have no effect estimate.' if cohort else ' Repair excludes protected test-conflict cells. Paired estimates use jointly known outcomes only.')
                        save(figure, f'{role}-{cohort or "all"}-{scope_index}-{start//12}', role, title, caption, panels)
            figure, axes = canvas(height=max(5, 2 + .3*len(report['coverage'])), width=12)
            coverage = report['coverage']
            labels = [row['arm_id'] + (f" / {row['recipient_class']}" if row['recipient_class'] else '') for row in coverage]
            statuses = sorted({key for row in coverage for key in row['review_statuses']})
            left = np.zeros(len(coverage))
            for status in statuses:
                values = [row['review_statuses'].get(status, 0) for row in coverage]
                axes[0].barh(range(len(coverage)), values, left=left, label=status.replace('_', ' '))
                left += values
            axes[0].set_yticks(range(len(coverage)), labels, fontsize=8)
            axes[0].invert_yaxis()
            axes[0].set_xlabel('Planned episodes')
            axes[0].set_xlim(0, max((row['planned'] for row in coverage), default=0)*1.1 or 1)
            axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
            save(figure, role + '-coverage', role, role.capitalize() + ' control coverage',
                 'All declared arms are shown. Review completeness is separate from behavioral outcomes; an earlier reviewed event can remain known in a partial episode. See source eligibility for unavailable controls.', coverage)

        if 'cost' in reports:
            report = reports['cost']
            categories = [row for row in report['categories'] if row['declared_work']]
            figure, axes = canvas(1, 3, height=max(5, 2+.5*len(categories)), width=18)
            columns = [('known_compute_estimate_usd', 'Known root compute estimates (USD)', 'unknown_root_compute_estimates'),
                       ('usage_calculated_usd', 'Known generation usage charges (USD)', 'unresolved_provider_work'),
                       ('unknown_charge_reservations_usd', 'Unknown charge reservations (USD)', 'unknown_charge_attempts')]
            for ax, (column, title, unknown) in zip(axes, columns):
                values = [float(row[column]) for row in categories]
                ax.barh(range(len(categories)), values, color='#007c91')
                ax.set_yticks(range(len(categories)), [row['category'].replace('_', ' ') + f"; ?={row[unknown]}" for row in categories], fontsize=8)
                ax.invert_yaxis()
                ax.set_title(title, loc='left', fontsize=10)
                ax.set_xlabel('USD; separate accounting view')
                ax.set_xlim(0, max(values, default=0)*1.35 or 1)
                if not categories:
                    ax.text(.5, .5, 'No declared work', transform=ax.transAxes, ha='center')
                for y, row in enumerate(categories):
                    ax.text(values[y], y, ' ' + row[column], va='center', fontsize=8)
            amount = report['declared_invoice_total_usd']
            save(figure, 'cost-views', 'cost', 'Declared work costs and missing coverage',
                 'Views are not additive. Zero shown means a known subtotal, not fully observed zero cost. Reservations are not charge upper bounds. '
                 'Nested operation times are excluded from root estimates; overlaps and bills need reconciliation. Declared invoice transcription: ' +
                 ('unavailable' if amount is None else 'USD ' + amount) + '.', categories)

    missing = [role for role in ROLES if role not in reports]
    index = ['# Analysis figure index', '', 'Fixture examples; no research result.' if fixture else 'Figures from declared research records; interpretation requires review.', '',
             'Reports not supplied: ' + (', '.join(missing) or 'none') + '.', '']
    for item in figures:
        index += ['## ' + item['title'], '', item['scope'], '', item['caption'], '',
                  f"![{item['figure_id']}](<{local_path(item['png']['path'])}>)", '',
                  f"[SVG](<{local_path(item['svg']['path'])}>) · [Data](<{local_path(item['data']['path'])}>) · [Source](<{local_path(item['source']['path'])}>)", '']
    atomic_bytes(directory / 'figures.md', '\n'.join(index).encode())
    return {'schema_version': 1, 'status': 'rendered', 'fixture': fixture, 'inputs': inputs, 'rule': analyzer.rule(),
        'matplotlib_version': version('matplotlib'), 'figures': figures, 'missing_reports': missing, 'verified_links': links,
        'index': artifact_ref(directory / 'figures.md', 'markdown'), 'new_model_calls': 0, 'new_provider_calls': 0, 'monitor_fits': 0,
        'limitations': ['Replay verifies saved calculations; rendering does not establish experimental acceptance or human interpretation',
                       'Missing reports and undefined values remain explicit; no example selection or research outcome is invented',
                       'Cost views cover declared work only; actual bill reconciliation remains outside the renderer']}
