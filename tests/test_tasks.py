from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from eval_tampering.messages import InputError, atomic_json, fingerprint, read_json
from eval_tampering.tasks import TaskBuilder, handle


ROOT = Path(__file__).resolve().parents[1]


class TaskBuilderTests(unittest.TestCase):
    def setUp(self):
        self.request = read_json(ROOT / "configs/task_example.json")
        self.task = self.request["inputs"]["tasks"][0]
        self.builder = TaskBuilder(self.request["config"])
        (ROOT / ".cache/tests").mkdir(parents=True, exist_ok=True)

    def test_four_conditions_and_protected_information_boundary(self):
        result = self.builder.build([self.task])
        bundle = result["tasks"][0]
        conditions = bundle["conditions"]
        self.assertEqual(len({condition["condition_id"] for condition in conditions}), 4)
        self.assertEqual(
            [(item["problem"], item["permission"]) for item in conditions],
            [("implementation_bug", "forbidden"), ("implementation_bug", "conditional"),
             ("test_conflict", "forbidden"), ("test_conflict", "conditional")],
        )
        self.assertEqual(conditions[0]["visible"]["files"], conditions[1]["visible"]["files"])
        self.assertEqual(conditions[2]["visible"]["files"], conditions[3]["visible"]["files"])
        self.assertEqual(conditions[0]["visible"]["files"]["solution.py"], self.task["implementation"]["buggy"])
        self.assertEqual(conditions[2]["visible"]["files"]["solution.py"], self.task["implementation"]["correct"])
        self.assertIn("must not be changed", conditions[2]["visible"]["permission"])
        self.assertIn("only when it contradicts", conditions[3]["visible"]["permission"])
        for item in conditions:
            self.assertEqual(set(item["visible"]), {"specification", "permission", "files"})
            self.assertEqual(set(item["visible"]["files"]), {"solution.py", "test_solution.py"})
            self.assertEqual(item["visible_sha256"], fingerprint(item["visible"]))
        self.assertEqual(bundle["protected"]["reference_cases"], self.task["reference_cases"])
        self.assertEqual(result["validation"]["semantic_validation"], "not_run")

    def test_trusted_example_has_independent_expected_behavior(self):
        # Execute only this checked-in, test-authored fixture; TaskBuilder never executes input code.
        namespace = {}
        exec(self.task["implementation"]["correct"], namespace)
        for case in self.task["reference_cases"]:
            value = namespace["clamp"](*case["args"], **case["kwargs"])
            self.assertEqual(value, case["expected"])
            self.assertIs(type(value), int)
        for mutant in self.task["mutants"]:
            namespace = {}
            exec(mutant["source"], namespace)
            self.assertTrue(any(namespace["clamp"](*case["args"], **case["kwargs"]) != case["expected"]
                                for case in self.task["reference_cases"]))
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tests") as directory:
            workspace = Path(directory)
            for implementation, tests, should_pass in [
                ("correct", "valid", True), ("buggy", "valid", False), ("correct", "conflicting", False),
            ]:
                (workspace / "solution.py").write_text(self.task["implementation"][implementation])
                (workspace / "test_solution.py").write_text(self.task["tests"][tests])
                completed = subprocess.run([sys.executable, "-B", "test_solution.py"], cwd=workspace,
                                           capture_output=True, timeout=10)
                self.assertEqual(completed.returncode == 0, should_pass, completed.stderr.decode())

    def test_invalid_task_and_message_inputs_fail_explicitly(self):
        mutations = [
            lambda req: req.update(schema_version=True),
            lambda req: req.update(operation="unknown"),
            lambda req: req.update(request_id="../escape"),
            lambda req: req["inputs"]["tasks"].append(deepcopy(req["inputs"]["tasks"][0])),
            lambda req: req["config"]["split_assignments"].update({"integer-clamping": "test"}),
            lambda req: req["inputs"]["tasks"][0]["implementation"].update(path="../solution.py"),
            lambda req: req["inputs"]["tasks"][0]["implementation"].update(correct="this is not python!"),
            lambda req: req["inputs"]["tasks"][0].update(function_name="other"),
            lambda req: req["inputs"]["tasks"][0]["reference_cases"][0].update(expected=float("nan")),
            lambda req: req["inputs"]["tasks"][0]["reference_cases"][0].update(args=(1, 2)),
        ]
        for mutation in mutations:
            request = deepcopy(self.request)
            mutation(request)
            with self.subTest(request=request):
                response = handle(request)
                self.assertEqual(response["status"], "error")
                self.assertNotIn("result", response)
                self.assertFalse(response["error"]["retryable"])

    def test_clone_group_split_and_exact_clone_leakage(self):
        clone = deepcopy(self.task)
        clone["task_id"] = "second-clamp"
        result = self.builder.build([self.task, clone])
        self.assertEqual([task["split"] for task in result["tasks"]], ["training", "training"])
        clone["clone_group_id"] = "disguised-clone"
        builder = TaskBuilder({"split_assignments": {"integer-clamping": "training", "disguised-clone": "detection_test"}})
        with self.assertRaisesRegex(InputError, "Identical task content"):
            builder.build([self.task, clone])

    def test_determinism_config_ownership_and_content_fingerprints(self):
        original = self.builder.handle(self.request)
        self.assertEqual(original, handle(self.request))
        self.request["config"]["split_assignments"]["integer-clamping"] = "validation"
        self.assertEqual(self.builder.handle(self.request)["error"]["code"], "configuration_mismatch")
        second = handle(self.request)
        self.assertNotEqual(original["result"]["manifest"]["config_sha256"], second["result"]["manifest"]["config_sha256"])
        self.task["reference_cases"][0]["expected"] = 99
        changed = handle(self.request)
        self.assertNotEqual(second["result"]["manifest"]["dataset_sha256"], changed["result"]["manifest"]["dataset_sha256"])
        self.assertEqual(original["result"]["tasks"][0]["protected"]["reference_cases"][0]["expected"], 0)

    def test_authoring_does_not_execute_supplied_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tests") as directory:
            marker = Path(directory) / "executed"
            self.task["implementation"]["correct"] += f"\nraise RuntimeError({str(marker)!r})\n"
            self.assertEqual(handle(self.request)["status"], "ok")
            self.assertFalse(marker.exists())

    def test_non_json_cycles_and_large_integer_literals_fail_explicitly(self):
        cycle = []
        cycle.append(cycle)
        self.request["inputs"]["tasks"] = cycle
        self.assertEqual(handle(self.request)["status"], "error")
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tests") as directory:
            path = Path(directory) / "request.json"
            path.write_text('{"number": ' + "9" * 5000 + "}")
            with self.assertRaises(InputError):
                read_json(path)

    def test_cli_round_trip_validate_and_path_rejection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tests") as directory:
            input_path, output_path = Path(directory) / "request.json", Path(directory) / "result.json"
            atomic_json(input_path, self.request)
            command = [sys.executable, "-B", "-m", "eval_tampering", "tasks", "--input", str(input_path), "--output", str(output_path)]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(read_json(output_path), self.builder.handle(self.request))
            self.request["operation"] = "validate"
            atomic_json(input_path, self.request)
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(read_json(output_path)["result"]["validation_scope"], "structure_and_python_syntax")
            completed = subprocess.run(command[:-1] + [str(input_path)], cwd=ROOT, capture_output=True, timeout=10)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(read_json(input_path), self.request)
            output_path.unlink()
            link = Path(directory) / "outside"
            link.symlink_to(ROOT.parent, target_is_directory=True)
            completed = subprocess.run(command[:-1] + [str(link / "should-not-exist.json")], cwd=ROOT,
                                       capture_output=True, timeout=10)
            self.assertEqual(completed.returncode, 2)
            self.assertFalse((ROOT.parent / "should-not-exist.json").exists())

    def test_invalid_json_and_atomic_write_failure_preserve_evidence(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tests") as directory:
            path = Path(directory) / "result.json"
            for text in ('{"key": 1, "key": 2}', '{"key": NaN}', '{"key": 1e999}', '{'):
                path.write_text(text)
                with self.assertRaises(InputError):
                    read_json(path)
            atomic_json(path, {"previous": True})
            with patch("eval_tampering.messages.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    atomic_json(path, {"new": True})
            self.assertEqual(read_json(path), {"previous": True})
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
