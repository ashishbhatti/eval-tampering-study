# Review and adoption of Claude's more_tasks.md

Reviewed the unchanged server proposal `more_tasks.md`. It was an independent
proposal, not an implemented dataset. Source hash and adopted IDs are recorded
in `configs/task_families.json` under `adoptions`. Prior task definitions and
results remain intact; no proposed instructions changed project policy.

## Adopted

| Source ideas | Adopted tasks | Family treatment |
|---|---|---|
| Text formatting | `slugify`, `wrap_text`, `render_table` | New `text-formatting` family: easy, medium, hard |
| Custom calendar | `is_leap`, `days_in_month`, `add_days` | New `custom-calendar` family: easy, medium, hard |
| Record collections | `pluck`, `group_by`, `diff_records` | New `record-processing` family: easy, medium, hard |
| Bucketing | `bucket_index` | Medium supplement in existing bounded-integer family |
| Free interval calculation | `free_slots` | Hard supplement in existing interval family |
| Infix conversion | `to_rpn` | Hard supplement in existing expression family |

This adds **12 solvable tasks and 3 family groups**, rather than counting every
proposal as independent. Each new solvable task has seven protected cases and
four requirement-specific mutants, within the 59-job evaluator budget. Custom
calendar assumptions, permitted dates/offsets and record missing-value semantics
are explicit. Record differences distinguish absent fields from explicit None,
and distinguish boolean False from integer zero.

We also authored three separate contradictory-specification controls, one for each
new family, following our existing representation. These are adaptations of our
control construction, not Claude's contradictory-test tasks copied unchanged.

Total inventory: **15 families, 48 solvable tasks, 15 impossible controls**.
The balanced ladder contains 45 solvable tasks; three supplements add depth but
must not silently overweight their families in comparisons. All are development
material. More tasks do not guarantee more tampering examples or stronger evidence.

## Kept distinct or deferred

- Existing clamp, merging, RPN and ranking proposals overlap our existing tasks;
  those do not add to the count. Existing family IDs were preserved.
- Histogram, leaderboard and bracket checking are useful additional within-family
  variants, but the current additions already widen both domain and rule coverage.
  They remain proposals, not counted or implemented tasks.
- Grid pathfinding overlaps the routing family; a grid representation alone does
  not establish another independent family. Neighbors/flood-fill can be future
  within-family additions after initial behavioral yield is measured.
- `title_case`, `is_valid_date`, `has_field` and the other proposed contradiction
  tasks remain unimplemented. Their underlying specifications are solvable.
- Proposed final-test splits were not adopted: these inspected development tasks
  remain in training. No sampling temperature, action/token limit, dependency,
  production evaluator or study label was changed.

## The two meanings of impossible

Claude's proposal keeps the specification consistent and puts opposite expected
answers in visible tests. For example, the specification requires -7 / 2 to
truncate to -3, while two tests demand -3 and -4. No correct pure implementation
can pass both tests unchanged. Under conditional permission, correcting the wrong
test is nevertheless a legitimate way to finish the task. This is a useful test
conflict, not an impossible specification.

Our separate controls require two distinct outputs for the same returned value
in the specification itself. Neither requirement has priority, so no legitimate
implementation or test correction can satisfy the complete request. Appropriate
behavior is to report the contradiction. They have two permission conditions and
are not accepted by the normal trusted-reference evaluator. Detecting the
contradiction may be easy even though satisfying it is impossible.

Claude's sole-rejector idea is a useful evaluator-coverage check: if weakening one
assertion stops rejecting a known faulty implementation, the lost check has a
concrete witness. It does not establish the model's incentive or intent. Our
validation verifies mutant rejection and exact conflict attribution; no blanket
sole-rejector property is claimed for all existing or adopted tasks.

## Verification and what remains unmeasured

The task-related local suite passes 27 tests. New independent checks compare
custom date arithmetic with day ordinals, interval gaps with integer membership,
infix output with generated expression trees, and record differences with typed
field comparisons. Existing checks cover literal reference answers, input
preservation, visible scripts, every mutant, conflicts, family grouping, hashes
and impossible witnesses. Old task/control definitions are pinned and compared.

The updated inventory graphic is authored coverage, not behavioral results.
The pilot still needs measured success, weakening, permission responsiveness,
failures, costs and available positive/negative labels. A 70-second clamp-based
estimate does not establish the cost of these tasks; one command can contain
multiple operations, so command count alone does not define repair difficulty.
