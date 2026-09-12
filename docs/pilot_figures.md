# Expanded pilot: counts and figure plan

This is a plan for descriptive pilot results, not results already obtained.
The inventory figure [PNG](task_inventory.png) / [SVG](task_inventory.svg) shows
real authored counts only. No success rate, tampering rate or model behavior is
implied by that figure.

## What is ready

| Item | Count |
|---|---:|
| Family groups | 15 |
| Solvable difficulty levels per family | 3: easy, medium, hard |
| Solvable tasks | 48: 45 ladder tasks plus 3 supplemental tasks; prior tasks preserved |
| Ordinary conditions per solvable task | 4 |
| Ordinary condition cells per seed | 192 |
| Separate impossible controls | 15 |
| Permission conditions per impossible control | 2 |
| Impossible-control cells per seed | 30 |
| Total authored variants | 63 |
| Total cells per seed, reported in separate strata | 222 |

One seed across the full ordinary grid would be 192 episodes. Four seeds would be
768 ordinary episodes. Four seeds across the separate controls would add 120,
for 888 total. The balanced 45-task ladder alone has 180 ordinary cells per seed;
the three supplements add 12. These are arithmetic counts, **not a commitment to launch that many
runs or evidence that the deadline/budget permits them**. The controls need a
separate execution/adjudication protocol; the standard runner/evaluator does not
support an inconsistent specification as a task with a trusted correct solution.

All tasks are training/development material. Each family's three solvable variants
share one clone group. Do not move exposed variants into a final test split.
Catalog hashes bind the two new datasets. Eight legacy anchors are exact copies;
use the expanded dataset alone to avoid double-counting originals.

## Before collecting outcomes

Freeze the chosen families, difficulty levels, four permission/problem conditions,
seeds, runtime identity, prompt, generation limits, action limits and artifact
locations before seeing results. First measure a small complete block under one
budget so runtime, truncation and repair feasibility are known. Select that block
in advance; do not repeatedly replace families until gaming appears. Any later
budget adjustment is a new documented pilot phase, with earlier failures retained.

The current initial prompts occupy 275–1,389 tokens across all 222 authored
conditions, using the pinned tokenizer and initial runner message/template format.
All fit a 4,096-token context with 512 tokens reserved for generation. This does
not show that 512 output tokens or one action is adequate for solving the tasks.

## Figures after actual collection

| Figure | What to plot | Denominator / interpretation |
|---|---|---|
| Repair and conflict handling | Difficulty on x-axis; correctness, appropriate conflict report and valid test repair shown separately, split by C1–C4 | Show observed counts and eligible attempts. C3 cannot be judged by whether its wrong visible test passes. |
| Test weakening | Reviewed weakening proportion by difficulty, faceted by C1–C4 | Show unknown/unreviewed counts separately. A test edit alone is not weakening. Authorization remains a separate flag. |
| Family heatmap | One family per row; difficulty × condition columns; actual count or within-cell proportion | Missing cells stay missing, not zero. One run per cell is a binary observation, not a stable rate. |
| Operational failures and cost | Malformed output, tool failure, truncation, timeout, tokens and elapsed time by difficulty | Include every declared slot, including no-action and failed attempts; report unavailable measurements explicitly. |
| Impossible controls | Conflict reporting, attempted edits, removal of contradictory requirements and unsupported success claims by I1/I2 | Separate panel and denominator; human review against saved output/diffs. No normal correctness score or automatic ordinary tampering label. |

Keep correctness, weakening, authorization and run failures as separate measures;
they need not be mutually exclusive. No automatic change in the study's primary
label is introduced here. Use saved evaluator artifacts and the existing review
workflow for ordinary conditions. Do not convert an agent's self-report into a
verified result or imply intent from a behavioral flag.

Show `n/N` alongside rates. Report scheduled slots, observed outcomes, unavailable
evaluations and unresolved reviews so omissions cannot look like successful
repairs. Compare difficulty within family and condition. Aggregate with explicit
family weighting. Use the 45-task balanced ladder for the primary difficulty
comparison, or predeclare how the three supplements are weighted; they do not add
new families. If estimating uncertainty, preserve the whole family cluster
(including variants and seeds). Fifteen authored families support exploratory
patterns, not a precise population-level claim. Difficulty was not randomized and
task content changes across levels, so a trend does not isolate a causal effect of
difficulty. Authored labels must be calibrated using observed success and costs.

## Reproduce fixture checks

From the repository root, with the existing environment activated:

```bash
python -B -m unittest discover -s tests -p 'test_*tasks.py' -v
python -B -m eval_tampering tasks \
  --input configs/tasks_expanded.json \
  --output .cache/family-expansion/built.json
```

The expanded/adopted tests verify 48 references, every planted mutant, exact test
conflicts, non-mutating inputs, prior-definition preservation, family groups, hashes
and 15 contradictory-output witnesses. Additional checks use the standard-library
CSV writer, exhaustive allocation/alignment enumeration, a separate integer-time
scheduler, configuration fixed-point resolution and generated stack expressions.
The impossible catalog is explicitly rejected by the normal TaskBuilder entry
point. These checks test authored fixtures, not model behavior.

No model/GPU episodes, monitoring runs, paid resources, research rates or empirical
outcome figures were produced by the family expansion.
