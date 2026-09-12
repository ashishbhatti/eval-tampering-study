"""Independent checks for adaptations of Claude's task proposals."""
from bisect import bisect_right
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import unittest

from test_harder_tasks import function

ROOT = Path(__file__).resolve().parents[1]


class AdoptedTaskTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT / 'configs/task_families.json').read_text())
        self.tasks = json.loads((ROOT / 'configs/tasks_expanded.json').read_text())['inputs']['tasks']
        self.functions = {task['function_name']: function(task['implementation']['correct'], task['function_name'])
                          for task in self.tasks}

    def test_preserves_every_previous_task_and_control(self):
        adopted = self.catalog['adoptions']
        self.assertEqual(len(adopted['solvable_task_ids']), 12)
        for rows, key in [(self.tasks, 'prior_solvable_definitions'),
                          (json.loads((ROOT / 'configs/tasks_impossible.json').read_text())['tasks'],
                           'prior_impossible_definitions')]:
            byid = {row['task_id']: row for row in rows}
            for tid, expected in adopted[key].items():
                self.assertEqual(hashlib.sha256(json.dumps(byid[tid], sort_keys=True).encode()).hexdigest(), expected)
        self.assertEqual(len(adopted['prior_solvable_definitions']), 36)
        self.assertEqual(len(adopted['prior_impossible_definitions']), 12)

    def test_custom_calendar_against_independent_day_ordinals(self):
        def year_start(year):
            y = year - 1
            return 365*y + y//4 - y//100 + y//400

        def ordinal(year, month, day):
            return year_start(year) + 31*(month//2) + 30*((month-1)//2) + day - 1

        def from_ordinal(value):
            low, high = 1, value//365+2
            while low+1 < high:
                middle = (low+high)//2
                if year_start(middle) <= value: low = middle
                else: high = middle
            remaining = value-year_start(low)
            starts = [31*(m//2)+30*((m-1)//2) for m in range(1,13)]
            month = bisect_right(starts, remaining)
            return [low, month, remaining-starts[month-1]+1]

        rng = random.Random(8901)
        for _ in range(300):
            year = rng.randrange(1,2601)
            value = rng.randrange(year_start(year), year_start(year+1))
            date = from_ordinal(value)
            delta = rng.randrange(max(-10000,-value),10001)
            self.assertEqual(self.functions['add_days'](*date,delta),from_ordinal(value+delta))
            self.assertEqual(ordinal(*date),value)
            leap = year_start(year+1)-year_start(year)==366
            self.assertEqual(self.functions['is_leap'](year),leap)
            month = rng.randrange(1,13)
            following = ordinal(year,month+1,1) if month<12 else year_start(year+1)
            self.assertEqual(self.functions['days_in_month'](year,month),following-ordinal(year,month,1))

    def test_buckets_and_free_slots_against_membership_enumeration(self):
        rng = random.Random(8902)
        for _ in range(150):
            edges = sorted(rng.sample(range(-20,21),rng.randrange(2,8)))
            value = rng.randrange(-25,26)
            matches = [i for i in range(len(edges)-1)
                       if edges[i] <= value and (value < edges[i+1] or i==len(edges)-2 and value==edges[-1])]
            self.assertEqual(self.functions['bucket_index'](value,edges), matches[0] if matches else -1)
            busy = [sorted(rng.sample(range(-12,13),2)) for _ in range(rng.randrange(8))]
            lower, upper = sorted(rng.sample(range(-8,9),2)); minimum = rng.randrange(1,6)
            remaining = [x for x in range(lower,upper+1) if not any(a<=x<=b for a,b in busy)]
            runs=[]
            for point in remaining:
                if runs and runs[-1][-1]+1==point:runs[-1].append(point)
                else:runs.append([point])
            expected=[[run[0],run[-1]] for run in runs if len(run)>=minimum]
            self.assertEqual(self.functions['free_slots'](busy,[lower,upper],minimum),expected)

    def test_infix_against_expression_trees_and_typed_record_diffs(self):
        rng = random.Random(8903)
        def expression(depth):
            if depth==0 or rng.random()<.3:
                number=str(rng.randrange(-9,10))
                return [number],[number]
            left, left_rpn=expression(depth-1);right,right_rpn=expression(depth-1)
            op=rng.choice(['+','-','*','/'])
            return ['(']+left+[op]+right+[')'],left_rpn+right_rpn+[op]
        def flatten(records,common):
            return {(row['id'],key):value for row in records if row['id'] in common
                    for key,value in row.items() if key!='id'}
        for _ in range(150):
            tokens,expected=expression(3)
            self.assertEqual(self.functions['to_rpn'](tokens),expected)
            arrays=[]
            for side in range(2):
                rows=[]
                for ident in 'abc':
                    if rng.random()<.7:
                        row={'id':ident}
                        row.update({key:rng.choice([None,False,0,1,'x']) for key in 'xy' if rng.random()<.7})
                        rows.append(row)
                rng.shuffle(rows);arrays.append(rows)
            before,after=arrays
            ids_a={r['id'] for r in before};ids_b={r['id'] for r in after}
            a,b=flatten(before,ids_a&ids_b),flatten(after,ids_a&ids_b)
            changed=[]
            for pair in sorted(a.keys()|b.keys()):
                if pair not in a or pair not in b or json.dumps(a[pair])!=json.dumps(b[pair]):
                    changed.append([*pair,a.get(pair),b.get(pair)])
            expected={'added':sorted(ids_b-ids_a),'removed':sorted(ids_a-ids_b),'changed':changed}
            saved=deepcopy(arrays)
            self.assertEqual(self.functions['diff_records'](before,after,'id'),expected)
            self.assertEqual(arrays,saved)


if __name__ == '__main__':
    unittest.main()
