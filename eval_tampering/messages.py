"""Strict JSON messages, content fingerprints, and atomic result writes."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


class InputError(ValueError):
    """An expected invalid request, distinct from a programming failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def require(condition: bool, message: str, code: str = "invalid_input") -> None:
    if not condition:
        raise InputError(code, message)


def fields(value: object, required: set[str], where: str) -> None:
    require(type(value) is dict, f"{where} must be an object")
    require(set(value) == required, f"{where} requires exactly: {', '.join(sorted(required))}")


def identifier(value: object, where: str) -> None:
    require(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is not None,
        f"{where} must be a 1–128 character identifier starting with a letter or digit",
    )


def json_value(value: object) -> None:
    """Reject Python values that would change type or become invalid JSON."""
    try:
        _json_value(value)
    except RecursionError as exc:
        raise InputError("invalid_input", "JSON must be acyclic and within Python's nesting limit") from exc


def _json_value(value: object) -> None:
    if type(value) is dict:
        require(all(type(key) is str for key in value), "JSON object keys must be strings")
        for item in value.values():
            _json_value(item)
    elif type(value) is list:
        for item in value:
            _json_value(item)
    elif type(value) is float:
        require(math.isfinite(value), "JSON numbers must be finite")
    else:
        require(value is None or type(value) in (str, int, bool), "Value is not a JSON type")


def fingerprint(value: object) -> str:
    json_value(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def decode_json(text: str | bytes) -> object:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}", "invalid_json")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=unique_object)
        json_value(value)
        return value
    except InputError:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InputError("invalid_json", f"Cannot decode JSON: {exc}") from exc


def read_json(path: Path) -> object:
    try:
        return decode_json(path.read_text(encoding="utf-8"))
    except UnicodeError as exc:
        raise InputError("invalid_json", f"Cannot decode JSON: {exc}") from exc


def validate_request(request: object, operations: set[str]) -> None:
    json_value(request)
    fields(request, {"schema_version", "request_id", "operation", "inputs", "config"}, "request")
    require(type(request["schema_version"]) is int and request["schema_version"] == 1,
            "Supported schema_version is 1", "unsupported_version")
    identifier(request["request_id"], "request_id")
    require(type(request["operation"]) is str and request["operation"] in operations,
            "Unsupported operation", "unsupported_operation")
    require(type(request["inputs"]) is dict, "inputs must be an object")
    require(type(request["config"]) is dict, "config must be an object")


def success(request: dict, result: object, artifacts: list | None = None) -> dict:
    return {"schema_version": 1, "request_id": request["request_id"], "status": "ok",
            "result": result, "artifacts": artifacts if artifacts is not None else []}


def failure(request: object, error: InputError) -> dict:
    request_id = request.get("request_id") if type(request) is dict else None
    return {"schema_version": 1, "request_id": request_id if type(request_id) is str else None,
            "status": "error", "error": {"code": error.code, "message": str(error), "retryable": False}}


def atomic_json(path: Path, value: object) -> None:
    """Replace only after serialization and a complete, flushed sibling write."""
    json_value(value)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    atomic_bytes(path, text.encode("utf-8"))


def atomic_bytes(path: Path, data: bytes) -> None:
    """Write a complete artifact before replacing the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def local_path(value: str) -> Path:
    require(type(value) is str and bool(value) and "\0" not in value, "Artifact path must be nonempty")
    path = Path(value).resolve()
    require(path.is_relative_to(Path.cwd().resolve()), "Artifact path leaves the working directory", "invalid_path")
    return path


def artifact_ref(path: Path, format_name: str) -> dict:
    path = local_path(str(path))
    return {"path": path.relative_to(Path.cwd().resolve()).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "format": format_name}


def read_artifact(reference: dict, format_name: str, max_bytes: int) -> bytes:
    fields(reference, {"path", "sha256", "format"}, "artifact reference")
    require(reference["format"] == format_name, f"Expected artifact format {format_name}")
    require(type(reference["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is not None,
            "Artifact sha256 must contain 64 lowercase hexadecimal characters")
    with local_path(reference["path"]).open("rb") as stream:
        data = stream.read(max_bytes + 1)
    require(len(data) <= max_bytes, "Artifact exceeds the declared size limit", "artifact_too_large")
    require(hashlib.sha256(data).hexdigest() == reference["sha256"], "Artifact content hash mismatch", "hash_mismatch")
    return data
