from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from eval_tampering.messages import InputError, atomic_json, read_json
from eval_tampering.sandbox import CommandResult, SandboxRunner, _bounded_command, _checked_archive, _files_archive, handle


ROOT = Path(__file__).resolve().parents[1]


def tar_bytes(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, kind, contents, mode in entries:
            member = tarfile.TarInfo(name)
            member.type, member.mode = kind, mode
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = contents.decode()
            else:
                member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents) if member.isfile() else None)
    return buffer.getvalue()


class SandboxTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".cache/tests").mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / ".cache/tests")
        self.addCleanup(self.directory.cleanup)
        self.request = read_json(ROOT / "configs/sandbox_example.json")
        self.request["config"]["artifact_root"] = str(Path(self.directory.name) / "artifacts")
        self.runner = SandboxRunner(self.request["config"])
        self.calls = []
        self.running = False
        self.command_status = "completed"
        self.export = _files_archive({"solution.py": "repaired", "new.py": "new"}, 8388608)
        self.fail_operation = None

    def fake_docker(self, args, **kwargs):
        # Tests only: independent responses exercise the host controller, not Docker isolation.
        self.calls.append((args, kwargs))
        out = b""
        if args[0] == self.fail_operation:
            return CommandResult(1, b"", b"injected Docker failure", "completed", 0.01)
        if args[0] == "info":
            out = json.dumps({"OSType": "linux", "MemoryLimit": True, "PidsLimit": True,
                              "CpuCfsQuota": True, "SwapLimit": True, "ServerVersion": "test-server",
                              "Runtimes": {"runc": {}},
                              "SecurityOptions": ["name=seccomp,profile=builtin"]}).encode()
        elif args[0] == "image":
            out = json.dumps({"Os": "linux", "Config": {"Volumes": None}, "Id": "test-image"}).encode()
        elif args[0] == "start":
            self.running = self.command_status != "completed"
            return CommandResult(1, b'{"grade":"success"}', b"diagnostic", self.command_status, 0.05)
        elif args[0] == "inspect":
            out = json.dumps({"Running": self.running, "ExitCode": 1, "OOMKilled": False}).encode()
        elif args[0] == "kill":
            self.running = False
        elif args[0] == "cp" and args[-1] == "-":
            out = self.export
        return CommandResult(0, out, b"", "completed", 0.01)

    def test_completed_execution_uses_frozen_boundaries_and_independent_snapshot(self):
        with patch.object(self.runner, "_docker", side_effect=self.fake_docker):
            response = self.runner.handle(self.request)
        self.assertEqual(response["status"], "ok", response)
        result = response["result"]
        self.assertEqual(result["record_status"], "complete")
        self.assertEqual(result["exit_code"], 1)
        self.assertNotIn("grade", result)
        self.assertEqual(result["changes"], {"added": ["new.py"], "deleted": ["test_solution.py"], "modified": ["solution.py"]})
        create = next(args for args, _ in self.calls if args[0] == "create")
        for flag, value in (("--network", "none"), ("--cap-drop", "ALL"), ("--user", "65534:65534"),
                            ("--memory", "256m"), ("--memory-swap", "256m"), ("--pids-limit", "32"),
                            ("--cpus", "1"), ("--security-opt", "no-new-privileges=true"), ("--pull", "never")):
            self.assertEqual(create[create.index(flag) + 1], value)
        self.assertIn("--read-only", create)
        self.assertIn("type=volume", create[create.index("--mount") + 1])
        self.assertNotIn("type=bind", " ".join(create))
        self.assertEqual(create[-3:], self.request["inputs"]["command"])
        self.assertEqual([args[0] for args, _ in self.calls][-2:], ["rm", "volume"])
        saved = read_json(Path(self.request["config"]["artifact_root"]) / self.request["request_id"] / "record.json")
        self.assertEqual(saved, result)
        self.assertEqual(len(response["artifacts"]), 6)

    def test_derived_image_requires_pinned_identity_provenance_and_base_layers(self):
        from eval_tampering.sandbox import BASE_IMAGE, PATCH_SHA256
        image_id = 'sha256:' + 'a'*64
        runner = SandboxRunner(self.request['config'] | {'image': image_id})
        good = {'Os': 'linux', 'Id': image_id, 'RootFS': {'Layers': ['base', 'patch']},
                'Config': {'Volumes': None, 'Labels': {'org.mats.base-image': BASE_IMAGE,
                           'org.mats.apply-patch-sha256': PATCH_SHA256}}}
        image = deepcopy(good)
        def docker(args, **kwargs):
            if args[0] == 'image':
                result = {'RootFS': {'Layers': ['base']}} if args[-1] == BASE_IMAGE else image
                return CommandResult(0, json.dumps(result).encode(), b'', 'completed', .01)
            return self.fake_docker(args, **kwargs)
        with patch.object(runner, '_docker', side_effect=docker):
            self.assertEqual(runner.preflight()['image_id'], image_id)
            for key, value in [('Id', 'sha256:'+'b'*64), ('RootFS', {'Layers': ['other', 'patch']}),
                               ('RootFS', {'Layers': ['base']}), ('Config', {'Volumes': None, 'Labels': {}})]:
                image = deepcopy(good) | {key: value}
                with self.assertRaises(InputError):
                    runner.preflight()
        for mutable in ('eval-tampering-sandbox:latest', 'sha256:short'):
            with self.assertRaises(InputError):
                SandboxRunner(self.request['config'] | {'image': mutable})

    def test_attach_failures_are_distinct_from_task_exit_codes(self):
        for index, (returncode, exit_code, status, error, expected) in enumerate([
            (1, 0, "created", "injected start failure", "infrastructure_error"),
            (1, 0, "exited", "", "infrastructure_error"),
            (0, 1, "exited", "", "infrastructure_error"),
            (127, 127, "exited", "injected start failure", "infrastructure_error"),
            (1, 1, "created", "", "infrastructure_error"),
            (1, 1, "dead", "", "infrastructure_error"),
            (0, 0, "exited", "", "completed"),
            (9, 9, "exited", "", "completed"),
        ]):
            with self.subTest(returncode=returncode, exit_code=exit_code, status=status, error=error):
                self.request["request_id"] = f"attach-{index}"
                self.calls.clear()
                def docker(args, **kwargs):
                    result = self.fake_docker(args, **kwargs)
                    if args[0] == "start":
                        return CommandResult(returncode, b"", b"attach diagnostic", "completed", .01)
                    if args[0] == "inspect":
                        state = {"Running": False, "ExitCode": exit_code, "Status": status, "Error": error}
                        return CommandResult(0, json.dumps(state).encode(), b"", "completed", .01)
                    return result
                with patch.object(self.runner, "_docker", side_effect=docker):
                    response = self.runner.handle(self.request)
                self.assertEqual(response["status"], "ok", response)
                result = response["result"]
                self.assertEqual(result["execution_status"], expected)
                self.assertEqual(result["exit_code"], exit_code if expected == "completed" else None)
                self.assertEqual(result["docker_returncode"], returncode)
                self.assertEqual(Path(result["stderr_artifact"]["path"]).read_bytes(), b"attach diagnostic")
                self.assertTrue(Path(result["snapshot"]["path"]).is_file())
                self.assertEqual(result["record_status"], "complete")
                self.assertEqual([args[0] for args, _ in self.calls][-2:], ["rm", "volume"])
                self.assertEqual(result["cleanup_errors"], [])

    def test_time_and_output_limits_stop_the_container_before_export(self):
        for status in ("time_limit", "output_limit"):
            self.request["request_id"] = status
            self.command_status = status
            self.calls.clear()
            with patch.object(self.runner, "_docker", side_effect=self.fake_docker):
                response = self.runner.handle(self.request)
            self.assertEqual(response["status"], "ok", response)
            self.assertEqual(response["result"]["execution_status"], status)
            self.assertIsNone(response["result"]["exit_code"])
            operations = [args[0] for args, _ in self.calls]
            self.assertLess(operations.index("kill"), len(operations) - 1 - operations[::-1].index("cp"))
            self.assertFalse(self.running)

    def test_copy_failure_and_cleanup_failure_remain_failed_records(self):
        for operation in ("cp", "rm"):
            self.request["request_id"] = "failed-" + operation
            self.fail_operation = operation
            self.calls.clear()
            with patch.object(self.runner, "_docker", side_effect=self.fake_docker):
                response = self.runner.handle(self.request)
            self.assertEqual(response["status"], "error")
            self.assertIn("evidence:", response["error"]["message"])
            saved = read_json(Path(self.request["config"]["artifact_root"]) / self.request["request_id"] / "record.json")
            self.assertEqual(saved["record_status"], "failed")
            self.assertEqual([args[0] for args, _ in self.calls][-2:], ["rm", "volume"])
            if operation == "rm":
                self.assertEqual(response["error"]["code"], "cleanup_failed")

    def test_unsafe_output_is_preserved_without_host_extraction(self):
        self.export = tar_bytes([("escape", tarfile.SYMTYPE, b"/etc/passwd", 0o777)])
        with patch.object(self.runner, "_docker", side_effect=self.fake_docker):
            response = self.runner.handle(self.request)
        self.assertEqual(response["error"]["code"], "unsafe_snapshot")
        directory = Path(self.request["config"]["artifact_root"]) / self.request["request_id"]
        self.assertEqual((directory / "workspace-export.tar").read_bytes(), self.export)
        self.assertFalse((directory / "escape").exists())
        self.assertEqual(read_json(directory / "record.json")["execution_status"], "completed")

    def test_previous_snapshot_round_trip_hash_checks_and_unique_attempts(self):
        with patch.object(self.runner, "_docker", side_effect=self.fake_docker):
            first = self.runner.handle(self.request)
            duplicate = self.runner.handle(self.request)
            self.assertEqual(duplicate["error"]["code"], "existing_attempt")
            self.request["request_id"] = "continued"
            self.request["inputs"].update(files=None, snapshot=first["result"]["snapshot"])
            second = self.runner.handle(self.request)
        self.assertEqual(second["status"], "ok", second)
        self.assertEqual(second["result"]["changes"], {"added": [], "deleted": [], "modified": []})
        self.request["inputs"]["snapshot"]["sha256"] = "0" * 64
        with patch.object(self.runner, "_docker") as docker:
            self.assertEqual(self.runner.handle(self.request)["error"]["code"], "hash_mismatch")
            docker.assert_not_called()

    def test_archive_limits_traversal_links_duplicates_and_binary_modes(self):
        data = tar_bytes([("empty", tarfile.DIRTYPE, b"", 0o755), ("binary", tarfile.REGTYPE, b"\xff\x00", 0o700)])
        normalized, index = _checked_archive(data, 65536, 4096)
        self.assertEqual(index["binary"]["mode"], 0o700)
        self.assertEqual(index["binary"]["size"], 2)
        self.assertEqual(_checked_archive(normalized, 65536, 4096)[1], index)
        nested = _files_archive({"nested/code.py": "print('ok')"}, 65536)
        self.assertEqual(_checked_archive(nested, 65536, 4096)[1]["nested"]["kind"], "directory")
        bad_entries = [
            [("../escape", tarfile.REGTYPE, b"x", 0o644)],
            [("/escape", tarfile.REGTYPE, b"x", 0o644)],
            [(".", tarfile.REGTYPE, b"x", 0o644)],
            [("link", tarfile.LNKTYPE, b"inside", 0o644)],
            [("fifo", tarfile.FIFOTYPE, b"", 0o644)],
            [("a", tarfile.REGTYPE, b"a", 0o644), ("a", tarfile.REGTYPE, b"b", 0o644)],
            [("a", tarfile.REGTYPE, b"a", 0o644), ("a/b", tarfile.REGTYPE, b"b", 0o644)],
        ]
        for entries in bad_entries:
            with self.subTest(entries=entries), self.assertRaises(InputError):
                _checked_archive(tar_bytes(entries), 65536, 4096)
        with self.assertRaises(InputError):
            _checked_archive(data[:-1], 65536, 4096)
        with self.assertRaises(InputError):
            _checked_archive(data, 65536, 1)
        with self.assertRaises(InputError):
            _files_archive({"invalid.py": "\ud800"}, 65536)

    def test_malformed_input_and_missing_runtime_never_run_task_on_host(self):
        original = deepcopy(self.request)
        for key, value in (("image", "python:latest"), ("pids_limit", 0), ("timeout_seconds", float("inf")),
                           ("docker_host", "tcp://remote:2375")):
            self.request = deepcopy(original)
            self.request["config"][key] = value
            with patch("eval_tampering.sandbox._bounded_command") as command:
                self.assertEqual(handle(self.request)["status"], "error")
                command.assert_not_called()
        self.request = deepcopy(original)
        self.request["config"]["docker_executable"] = str(Path(self.directory.name) / "missing-docker")
        with patch("eval_tampering.sandbox._bounded_command") as command:
            response = handle(self.request)
            self.assertEqual(response["error"]["code"], "runtime_unavailable")
            command.assert_not_called()
        self.assertFalse(Path(self.request["config"]["artifact_root"]).exists())

    def test_bounded_capture_with_real_trusted_processes(self):
        echo = _bounded_command([sys.executable, "-B", "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"], 5, 100000, b"x" * 90000)
        self.assertEqual((echo.status, echo.stdout), ("completed", b"x" * 90000))
        noisy = _bounded_command([sys.executable, "-B", "-c", "import os; os.write(1,b'a'*20000); os.write(2,b'b'*20000)"], 5, 1024)
        self.assertEqual(noisy.status, "output_limit")
        self.assertEqual(len(noisy.stdout) + len(noisy.stderr), 1024)
        sleeping = _bounded_command([sys.executable, "-B", "-c", "import time; time.sleep(10)"], 0.05, 1024)
        self.assertEqual(sleeping.status, "time_limit")
        self.assertLess(sleeping.elapsed_seconds, 2)

    def test_cli_missing_runtime_returns_structured_error(self):
        self.request["config"]["docker_executable"] = str(Path(self.directory.name) / "missing-docker")
        input_path, output_path = Path(self.directory.name) / "request.json", Path(self.directory.name) / "result.json"
        atomic_json(input_path, self.request)
        completed = subprocess.run([sys.executable, "-B", "-m", "eval_tampering", "sandbox", "--input", str(input_path),
                                    "--output", str(output_path)], cwd=ROOT, capture_output=True, timeout=10)
        self.assertEqual(completed.returncode, 1, completed.stderr.decode())
        self.assertEqual(read_json(output_path)["error"]["code"], "runtime_unavailable")

    def test_unexpected_programming_failure_keeps_traceback_and_cleans_resources(self):
        def broken(args, **kwargs):
            if args[0] == "start":
                raise TypeError("programming bug")
            return self.fake_docker(args, **kwargs)
        with patch.object(self.runner, "_docker", side_effect=broken):
            with self.assertRaisesRegex(TypeError, "programming bug"):
                self.runner.handle(self.request)
        self.assertEqual([args[0] for args, _ in self.calls][-2:], ["rm", "volume"])
        record = read_json(Path(self.request["config"]["artifact_root"]) / self.request["request_id"] / "record.json")
        self.assertEqual(record["record_status"], "incomplete")

    def test_configuration_copy_and_invalid_state_metadata(self):
        self.request["config"]["timeout_seconds"] = 6
        with patch.object(self.runner, "_docker") as docker:
            self.assertEqual(self.runner.handle(self.request)["error"]["code"], "configuration_mismatch")
            docker.assert_not_called()
        with patch.object(self.runner, "_docker", return_value=CommandResult(0, b'{"Running":"false"}', b"", "completed", 0)):
            with self.assertRaises(InputError):
                self.runner._state("example")


@unittest.skipUnless(os.environ.get("EVAL_TAMPERING_DOCKER_TEST_CONFIG"), "Live Docker configuration not supplied")
class LiveDockerTests(unittest.TestCase):
    """Explicit opt-in: run only on a host prepared with the pinned image and Docker."""

    def test_real_memory_process_and_file_limits(self):
        import uuid
        config = read_json(Path(os.environ["EVAL_TAMPERING_DOCKER_TEST_CONFIG"]))["config"]
        config.update(memory_mb=64, pids_limit=8, timeout_seconds=5)
        runner = SandboxRunner(config)
        memory_probe = """import pathlib, subprocess
child = subprocess.run(['python', '-c', 'bytearray(512*1024*1024)'])
events = dict(line.split() for line in pathlib.Path('/sys/fs/cgroup/memory.events').read_text().splitlines())
assert child.returncode == -9, child.returncode
assert int(events['oom_kill']) >= 1, events
print('verified cgroup OOM kill')
"""
        exhausted = runner.execute({"files": {}, "snapshot": None,
                                    "command": ["python", "-c", memory_probe]},
                                   "live-" + uuid.uuid4().hex)
        # Docker's asynchronous OOMKilled flag was false after a real exit 137.
        # Check the kernel event while the parent remains alive in the cgroup.
        self.assertEqual(exhausted["exit_code"], 0, exhausted)
        self.assertEqual(exhausted["stdout"].strip(), "verified cgroup OOM kill")
        probe = f"""import errno, pathlib, subprocess
assert pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip() == str(64*1024*1024)
assert pathlib.Path('/sys/fs/cgroup/pids.max').read_text().strip() == '8'
quota, period = map(int, pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split())
assert abs(quota / period - {config['cpus']!r}) < .0001
children = []
try:
    for _ in range(16):
        try:
            children.append(subprocess.Popen(['sleep', '30']))
        except OSError as exc:
            assert exc.errno == errno.EAGAIN
            break
    else:
        raise AssertionError('process limit was not enforced')
finally:
    for child in children:
        child.kill()
    for child in children:
        child.wait()
try:
    pathlib.Path('bounded.bin').write_bytes(bytes({config['max_file_bytes']} + 1))
except OSError as exc:
    assert exc.errno == errno.EFBIG
else:
    raise AssertionError('file size limit was not enforced')
assert pathlib.Path('bounded.bin').stat().st_size <= {config['max_file_bytes']}
"""
        result = runner.execute({"files": {}, "snapshot": None, "command": ["python", "-c", probe]},
                                "live-" + uuid.uuid4().hex)
        self.assertEqual(result["exit_code"], 0, result)
        self.assertFalse(result["cleanup_errors"])

    def test_real_isolation_and_cumulative_snapshot(self):
        config = read_json(Path(os.environ["EVAL_TAMPERING_DOCKER_TEST_CONFIG"]))["config"]
        runner = SandboxRunner(config)
        probe = (
            "import os,socket,pathlib,resource; assert os.getuid()==65534; "
            "assert not pathlib.Path('/var/run/docker.sock').exists(); "
            "assert not pathlib.Path('/dev/nvidia0').exists(); "
            "assert 'NoNewPrivs:\\t1' in pathlib.Path('/proc/self/status').read_text(); "
            "assert 'Seccomp:\\t2' in pathlib.Path('/proc/self/status').read_text(); "
            "assert 'CapEff:\\t0000000000000000' in pathlib.Path('/proc/self/status').read_text(); "
            f"assert resource.getrlimit(resource.RLIMIT_FSIZE)[0]=={config['max_file_bytes']}; "
            "s=socket.socket(); s.settimeout(.2); assert s.connect_ex(('1.1.1.1',443))!=0; "
            "pathlib.Path('test_solution.py').write_text('assert True\\n')"
        )
        import uuid
        first = runner.execute({"files": {"test_solution.py": "assert False\n"}, "snapshot": None,
                                "command": ["python", "-B", "-c", probe]}, "live-" + uuid.uuid4().hex)
        self.assertEqual(first["exit_code"], 0, first)
        second = runner.execute({"files": None, "snapshot": first["snapshot"],
                                 "command": ["python", "-B", "test_solution.py"]}, "live-" + uuid.uuid4().hex)
        self.assertEqual(second["exit_code"], 0, second)
        self.assertFalse(first["cleanup_errors"])
        self.assertFalse(second["cleanup_errors"])

    def test_real_limits_and_denied_host_access(self):
        import uuid
        config = read_json(Path(os.environ["EVAL_TAMPERING_DOCKER_TEST_CONFIG"]))["config"]
        config["timeout_seconds"] = 1
        config["max_output_bytes"] = 1024
        runner = SandboxRunner(config)
        for command, expected in [
            (["python", "-c", "import time; time.sleep(30)"], "time_limit"),
            (["python", "-c", "print('x'*100000)"], "output_limit"),
        ]:
            result = runner.execute({"files": {}, "snapshot": None, "command": command}, "live-" + uuid.uuid4().hex)
            self.assertEqual(result["execution_status"], expected, result)
            self.assertFalse(result["cleanup_errors"])
        (ROOT / ".cache/tests").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tests") as directory:
            sentinel = Path(directory) / "protected-canary"
            sentinel.write_text("never copied into the sandbox")
            script = f"""import pathlib
assert not pathlib.Path({str(sentinel)!r}).exists()
try:
    pathlib.Path('/outside-workspace').write_text('no')
except OSError:
    pass
else:
    raise AssertionError('root filesystem was writable')
print('GRADE=PASS EXIT_CODE=0')
raise SystemExit(9)
"""
            result = runner.execute({"files": {}, "snapshot": None, "command": ["python", "-c", script]},
                                    "live-" + uuid.uuid4().hex)
            self.assertEqual(result["exit_code"], 9, result)
            self.assertEqual(result["stdout"].strip(), "GRADE=PASS EXIT_CODE=0")
            attempt = "live-" + uuid.uuid4().hex
            with self.assertRaises(InputError) as raised:
                runner.execute({"files": {}, "snapshot": None,
                                "command": ["python", "-c", f"import os; os.symlink({str(sentinel)!r}, 'escape')"]},
                               attempt)
            self.assertEqual(raised.exception.code, "unsafe_snapshot")
            evidence = read_json(Path(config["artifact_root"]) / attempt / "record.json")
            self.assertEqual(evidence["record_status"], "failed")
            self.assertIn("workspace_export", evidence)
            self.assertFalse(evidence["cleanup_errors"])
            self.assertEqual(sentinel.read_text(), "never copied into the sandbox")


if __name__ == "__main__":
    unittest.main()
