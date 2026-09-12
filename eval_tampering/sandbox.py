"""Docker execution and evidence collection; never execute task code on the host."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import PurePosixPath
import re
import selectors
import shutil
import signal
import subprocess
import tarfile
import time
import uuid

from .messages import (InputError, artifact_ref, atomic_bytes, atomic_json, failure, fields,
                       fingerprint, identifier, json_value, local_path, read_artifact,
                       require, success, validate_request)


BASE_IMAGE = 'python:3.14.7-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6'
PATCH_SHA256 = 'cd93563e9e1b2aebc78a418543f687b43696676647dfd739f49f456fa369ac21'


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    status: str
    elapsed_seconds: float


def _bounded_command(argv: list[str], timeout: float, limit: int, data: bytes | None = None) -> CommandResult:
    """Bound both output streams and stdin transfer, including pipe-holding children."""
    started = time.monotonic()
    process = subprocess.Popen(argv, stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    output = {"stdout": bytearray(), "stderr": bytearray()}
    status, sent = "completed", 0
    try:
        with selectors.DefaultSelector() as selector:
            for name in output:
                stream = getattr(process, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            if data is not None:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            while selector.get_map():
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    status = "time_limit"
                    break
                for key, _ in selector.select(remaining):
                    stream = key.fileobj
                    if key.data == "stdin":
                        try:
                            sent += os.write(stream.fileno(), data[sent:sent + 65536])
                        except BrokenPipeError:
                            sent = len(data)
                        if sent == len(data):
                            selector.unregister(stream)
                            stream.close()
                        continue
                    chunk = os.read(stream.fileno(), 65536)
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    available = limit - sum(map(len, output.values()))
                    output[key.data].extend(chunk[:available])
                    if len(chunk) > available:
                        status = "output_limit"
                        break
                if status != "completed":
                    break
        if status == "completed":
            try:
                process.wait(timeout=max(0, timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                status = "time_limit"
    finally:
        # Kill the CLI's process group too; killing an attach client does not kill its container.
        if process.poll() is None or status != "completed":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
    return CommandResult(process.returncode, bytes(output["stdout"]), bytes(output["stderr"]),
                         status, time.monotonic() - started)


def _workspace_path(name: str) -> str:
    require(type(name) is str and 0 < len(name) <= 512 and "\0" not in name and "\\" not in name,
            "Invalid workspace path", "unsafe_snapshot")
    path = PurePosixPath(name)
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts and path.as_posix() == name,
            "Workspace paths must be normalized and relative", "unsafe_snapshot")
    return name


def _files_archive(files: dict[str, str], limit: int) -> bytes:
    require(type(files) is dict, "files must be an object")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, content in sorted(files.items()):
            _workspace_path(name)
            require(type(content) is str, "Initial file contents must be text")
            try:
                data = content.encode("utf-8")
            except UnicodeError as exc:
                raise InputError("invalid_input", "Initial file contents must be valid UTF-8 text") from exc
            require(len(data) <= limit, "Initial file exceeds the declared limit")
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o644
            member.uid = member.gid = 65534
            archive.addfile(member, io.BytesIO(data))
            require(stream.tell() <= limit, "Initial workspace exceeds the snapshot limit")
    require(len(stream.getvalue()) <= limit, "Initial archive exceeds the snapshot limit")
    return stream.getvalue()


def _checked_archive(data: bytes, limit: int, file_limit: int) -> tuple[bytes, dict]:
    """Validate without host extraction; preserve file bytes/modes and empty directories."""
    require(len(data) <= limit and len(data) >= 1024 and len(data) % 512 == 0 and data[-1024:] == bytes(1024),
            "Snapshot is oversized or truncated", "unsafe_snapshot")
    entries, total = {}, 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if name in ("", ".") and member.isdir():
                    continue
                _workspace_path(name)
                require(name not in entries and len(entries) < 1024, "Duplicate or excessive snapshot entries", "unsafe_snapshot")
                require(member.isfile() or member.isdir(), "Snapshot links and special files are unsupported", "unsafe_snapshot")
                require(not member.sparse, "Sparse files are unsupported", "unsafe_snapshot")
                require(0 <= member.size <= file_limit, "Snapshot file exceeds limit", "unsafe_snapshot")
                total += member.size
                require(total <= limit, "Snapshot contents exceed limit", "unsafe_snapshot")
                contents = archive.extractfile(member).read() if member.isfile() else b""
                require(len(contents) == member.size, "Truncated snapshot member", "unsafe_snapshot")
                entries[name] = (member.isdir(), member.mode & 0o777, contents)
        for name in list(entries):
            for parent in PurePosixPath(name).parents:
                if str(parent) == ".":
                    continue
                if str(parent) in entries:
                    require(entries[str(parent)][0], "A file cannot contain other paths", "unsafe_snapshot")
                else:
                    require(len(entries) < 1024, "Excessive implicit directories", "unsafe_snapshot")
                    entries[str(parent)] = (True, 0o755, b"")
    except (tarfile.TarError, OSError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError("unsafe_snapshot", f"Invalid tar snapshot: {exc}") from exc
    normalized, index = io.BytesIO(), {}
    with tarfile.open(fileobj=normalized, mode="w") as archive:
        for name, (is_dir, mode, contents) in sorted(entries.items()):
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE if is_dir else tarfile.REGTYPE
            member.uid = member.gid = 65534
            member.mode, member.size = mode, len(contents)
            archive.addfile(member, io.BytesIO(contents) if not is_dir else None)
            index[name] = {"kind": "directory" if is_dir else "file", "mode": mode,
                           "size": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
    require(len(normalized.getvalue()) <= limit, "Normalized snapshot exceeds limit", "unsafe_snapshot")
    return normalized.getvalue(), index


class SandboxRunner:
    """Own one frozen execution configuration; each execute call gets fresh resources."""

    def __init__(self, config: dict):
        json_value(config)
        fields(config, {"docker_executable", "docker_host", "image", "timeout_seconds", "memory_mb", "cpus",
                        "pids_limit", "max_output_bytes", "max_file_bytes", "max_snapshot_bytes", "artifact_root"}, "config")
        require(type(config["docker_executable"]) is str and bool(config["docker_executable"]) and "\0" not in config["docker_executable"], "Specify the Docker executable")
        require(type(config["docker_host"]) is str and config["docker_host"].startswith("unix:///") and "\0" not in config["docker_host"],
                "Use an explicit Unix Docker socket on the execution host")
        require(type(config["image"]) is str and re.fullmatch(r"(?:(?:docker.io/library/)?python:3\.14\.7-slim@)?sha256:[0-9a-f]{64}", config["image"]) is not None,
                "Pin the official Python image by digest or the derived sandbox by immutable image ID")
        for key, low, high in (("timeout_seconds", 0.1, 300), ("cpus", 0.1, 8)):
            require(type(config[key]) in (int, float) and low <= config[key] <= high, f"Invalid {key}")
        for key, low, high in (("memory_mb", 64, 4096), ("pids_limit", 8, 256),
                              ("max_output_bytes", 1, 1048576), ("max_file_bytes", 1, 67108864),
                              ("max_snapshot_bytes", 10240, 134217728)):
            require(type(config[key]) is int and low <= config[key] <= high, f"Invalid {key}")
        require(config["max_file_bytes"] <= config["max_snapshot_bytes"], "File limit exceeds snapshot limit")
        require(type(config["artifact_root"]) is str and bool(config["artifact_root"]), "Specify artifact_root")
        self._config = deepcopy(config)

    def _docker(self, args: list[str], timeout: float = 15, limit: int = 1048576, data: bytes | None = None) -> CommandResult:
        executable = shutil.which(self._config["docker_executable"])
        require(executable is not None, "Docker CLI is unavailable; no local execution fallback", "runtime_unavailable")
        try:
            return _bounded_command([executable, "--host", self._config["docker_host"], *args], timeout, limit, data)
        except OSError as exc:
            raise InputError("runtime_unavailable", f"Cannot invoke Docker: {exc}") from exc

    def _checked(self, args: list[str], **kwargs) -> bytes:
        outcome = self._docker(args, **kwargs)
        require(outcome.status == "completed" and outcome.returncode == 0,
                f"Docker {args[0]} failed ({outcome.status}): {outcome.stderr.decode('utf-8', 'replace')[:2000]}", "docker_error")
        return outcome.stdout

    def preflight(self) -> dict:
        try:
            info = json.loads(self._checked(["info", "--format", "{{json .}}"]))
            image = json.loads(self._checked(["image", "inspect", "--format", "{{json .}}", self._config["image"]]))
            require(info.get("OSType") == "linux", "The sandbox requires a Linux Docker engine", "unsupported_runtime")
            require(all(info.get(key) is True for key in ("MemoryLimit", "PidsLimit", "CpuCfsQuota", "SwapLimit")),
                    "The Docker host lacks required resource-limit support", "unsupported_runtime")
            require(any("seccomp" in option for option in info.get("SecurityOptions", [])),
                    "Docker's default seccomp profile must be enabled", "unsupported_runtime")
            require("runc" in info.get("Runtimes", {}), "The CPU-only runc runtime is required", "unsupported_runtime")
            require(image.get("Os") == "linux" and not image["Config"].get("Volumes"),
                    "The image must be Linux and must not declare additional volumes", "unsupported_runtime")
            if self._config['image'].startswith('sha256:'):
                labels = image['Config'].get('Labels') or {}
                require(image.get('Id') == self._config['image'] and
                        labels.get('org.mats.base-image') == BASE_IMAGE and
                        labels.get('org.mats.apply-patch-sha256') == PATCH_SHA256,
                        'Derived sandbox identity or provenance changed', 'unsupported_runtime')
                base = json.loads(self._checked(['image', 'inspect', '--format', '{{json .}}', BASE_IMAGE]))
                base_layers, layers = base.get('RootFS', {}).get('Layers'), image.get('RootFS', {}).get('Layers')
                require(type(base_layers) is list and bool(base_layers) and type(layers) is list and
                        len(layers) > len(base_layers) and layers[:len(base_layers)] == base_layers,
                        'Derived sandbox must extend the pinned Python base layers', 'unsupported_runtime')
            return {"server_version": info["ServerVersion"], "image_id": image["Id"],
                    "image": self._config["image"], "security_options": info["SecurityOptions"]}
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            if isinstance(exc, InputError):
                raise
            raise InputError("docker_error", f"Invalid Docker metadata: {exc}") from exc

    def _state(self, name: str) -> dict:
        try:
            state = json.loads(self._checked(["inspect", "--format", "{{json .State}}", name]))
        except ValueError as exc:
            if isinstance(exc, InputError):
                raise
            raise InputError("docker_error", f"Invalid container state: {exc}") from exc
        require(type(state) is dict and type(state.get("Running")) is bool and type(state.get("ExitCode")) is int,
                "Docker returned invalid container state", "docker_error")
        return state

    def execute(self, inputs: dict, request_id: str) -> dict:
        started = time.monotonic()
        json_value(inputs)
        identifier(request_id, "request_id")
        fields(inputs, {"files", "snapshot", "command"}, "inputs")
        command = inputs["command"]
        require(type(command) is list and 0 < len(command) <= 128 and bool(command[0]) and
                all(type(arg) is str and "\0" not in arg for arg in command),
                "command must name an executable and contain string arguments without NUL bytes")
        require(sum(len(arg) for arg in command) <= 131072, "Command is too large")
        require((inputs["files"] is None) != (inputs["snapshot"] is None), "Supply exactly one of files or snapshot")
        config = self._config
        raw = (_files_archive(inputs["files"], config["max_snapshot_bytes"]) if inputs["files"] is not None
               else read_artifact(inputs["snapshot"], "tar", config["max_snapshot_bytes"]))
        initial, before = _checked_archive(raw, config["max_snapshot_bytes"], config["max_file_bytes"])
        root = local_path(config["artifact_root"])
        require(root != local_path("."), "Use a dedicated artifact directory rather than the repository root", "invalid_path")
        runtime = self.preflight()
        root.mkdir(parents=True, exist_ok=True)
        directory = root / request_id
        require(not directory.exists(), "Request ID already has artifacts; use a new attempt ID", "existing_attempt")
        directory.mkdir()
        name = "eval-tampering-" + uuid.uuid4().hex
        volume = name + "-workspace"
        record = {"request_id": request_id, "config_sha256": fingerprint(config), "runtime": runtime,
                  "container": name, "volume": volume, "record_status": "incomplete",
                  "execution_status": "not_started", "command": command.copy()}
        atomic_json(directory / "request.json", {"inputs": inputs, "config": config})
        atomic_bytes(directory / "before.tar", initial)
        record["before_snapshot"] = artifact_ref(directory / "before.tar", "tar")
        atomic_json(directory / "record.json", record)
        error = None
        try:
            memory = f"{config['memory_mb']}m"
            self._checked(["create", "--name", name, "--hostname", "eval-tampering", "--label", "eval_tampering=1", "--pull", "never",
                           "--network", "none", "--read-only", "--cap-drop", "ALL",
                           "--security-opt", "no-new-privileges=true", "--user", "65534:65534",
                           "--init", "--runtime", "runc", "--ipc", "none", "--cgroupns", "private", "--log-driver", "none",
                           "--memory", memory, "--memory-swap", memory, "--cpus", str(config["cpus"]),
                           "--pids-limit", str(config["pids_limit"]), "--ulimit", "nofile=128:128",
                           "--ulimit", "core=0:0", "--ulimit", f"fsize={config['max_file_bytes']}:{config['max_file_bytes']}",
                           "--mount", f"type=volume,source={volume},target=/tmp", "--workdir", "/tmp",
                           "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONHASHSEED=0",
                           "--entrypoint", "", config["image"], *command])
            self._checked(["cp", "--archive", "-", f"{name}:/tmp"], data=initial)
            action = self._docker(["start", "--attach", name], timeout=config["timeout_seconds"], limit=config["max_output_bytes"])
            atomic_bytes(directory / "stdout.bin", action.stdout)
            atomic_bytes(directory / "stderr.bin", action.stderr)
            record["stdout_artifact"] = artifact_ref(directory / "stdout.bin", "bytes")
            record["stderr_artifact"] = artifact_ref(directory / "stderr.bin", "bytes")
            record.update(execution_status=action.status, docker_returncode=action.returncode, elapsed_seconds=action.elapsed_seconds,
                          stdout=action.stdout.decode("utf-8", "replace"), stderr=action.stderr.decode("utf-8", "replace"))
            state = self._state(name)
            if state["Running"]:
                if action.status == "completed":
                    record["execution_status"] = "infrastructure_error"
                self._checked(["kill", name])
                state = self._state(name)
            require(state["Running"] is False, "Container did not stop; refusing to read a changing workspace", "docker_error")
            # Attach propagates task exits too; a nonzero code alone is not a Docker failure.
            if record["execution_status"] == "completed" and (action.returncode != state["ExitCode"] or
                    state.get("Error") or state.get("Status") in {"created", "dead"}):
                record["execution_status"] = "infrastructure_error"
            record["container_state"] = state
            record["exit_code"] = state["ExitCode"] if record["execution_status"] == "completed" else None
            exported = self._docker(["cp", f"{name}:/tmp/.", "-"], limit=config["max_snapshot_bytes"])
            atomic_bytes(directory / "workspace-export.tar", exported.stdout)
            record["workspace_export"] = artifact_ref(directory / "workspace-export.tar", "tar" if exported.status == "completed" else "tar-fragment")
            require(exported.status == "completed" and exported.returncode == 0,
                    "Workspace export failed or exceeded limits; partial bytes were preserved", "snapshot_unavailable")
            normalized, after = _checked_archive(exported.stdout, config["max_snapshot_bytes"], config["max_file_bytes"])
            atomic_bytes(directory / "after.tar", normalized)
            record["snapshot"] = artifact_ref(directory / "after.tar", "tar")
            record["changes"] = {"added": sorted(after.keys() - before.keys()), "deleted": sorted(before.keys() - after.keys()),
                                 "modified": sorted(path for path in before.keys() & after.keys() if before[path] != after[path])}
            record["record_status"] = "complete"
        except (InputError, OSError) as exc:
            error = exc if isinstance(exc, InputError) else InputError("execution_error", str(exc))
            record.update(record_status="failed", error={"code": error.code, "message": str(error)})
        finally:
            cleanup_errors = []
            for args, absent in [(["rm", "--force", name], "No such container"), (["volume", "rm", volume], "no such volume")]:
                try:
                    outcome = self._docker(args)
                    absent_resource = absent.lower() in outcome.stderr.decode("utf-8", "replace").lower()
                    if outcome.status != "completed" or (outcome.returncode != 0 and not absent_resource):
                        cleanup_errors.append(f"{' '.join(args)}: {outcome.stderr.decode('utf-8', 'replace')[:1000]}")
                except InputError as exc:
                    cleanup_errors.append(str(exc))
            record["cleanup_errors"] = cleanup_errors
            if cleanup_errors:
                error = InputError("cleanup_failed", "Container or volume cleanup failed; inspect the saved record")
                record["record_status"] = "failed"
            record["total_elapsed_seconds"] = time.monotonic() - started
            atomic_json(directory / "record.json", record)
        if error is not None:
            raise InputError(error.code, f"{error}; evidence: {directory / 'record.json'}")
        return record

    def handle(self, request: dict) -> dict:
        try:
            validate_request(request, {"preflight", "execute"})
            require(request["config"] == self._config, "Request configuration differs from this SandboxRunner", "configuration_mismatch")
            if request["operation"] == "preflight":
                fields(request["inputs"], set(), "inputs")
                return success(request, self.preflight())
            result = self.execute(request["inputs"], request["request_id"])
            artifacts = [result[key] for key in ("before_snapshot", "snapshot", "workspace_export", "stdout_artifact", "stderr_artifact")]
            artifacts.append(artifact_ref(local_path(self._config["artifact_root"]) / request["request_id"] / "record.json", "json"))
            return success(request, result, artifacts)
        except (InputError, OSError) as exc:
            return failure(request, exc if isinstance(exc, InputError) else InputError("file_error", str(exc)))


def handle(request: dict) -> dict:
    try:
        validate_request(request, {"preflight", "execute"})
        return SandboxRunner(request["config"]).handle(request)
    except InputError as exc:
        return failure(request, exc)
