"""One TransformerBridge runtime; local assets, explicit loading, operation-local hooks."""

from contextlib import contextmanager
from datetime import date
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import time
import traceback
import uuid

from .messages import (InputError, artifact_ref, atomic_bytes, atomic_json, decode_json,
                       failure, fields, fingerprint, json_value, local_path,
                       read_artifact, require, success, validate_request)

MODEL_ID = "openai/gpt-oss-20b"
OPERATIONS = {"load", "prepare", "resume", "generate", "capture", "diagnose", "check", "benign_control"}
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
ASSETS = {
    "config.json": "3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce",
    "generation_config.json": "f9970ada892d2d1f72e3ed0a6535ccebadd11897318794ca671d8c7014c957da",
    "tokenizer_config.json": "9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070",
    "chat_template.jinja": "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146",
    "tokenizer.json": "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
}
WEIGHT_INDEX_SHA256 = "0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1"
WEIGHTS = {
    "model-00000-of-00002.safetensors": "16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427",
    "model-00001-of-00002.safetensors": "4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea",
    "model-00002-of-00002.safetensors": "a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303",
}
START, END, MESSAGE, CHANNEL, RETURN, CALL = 200006, 200007, 200008, 200005, 200002, 200012
STOPS = {END, RETURN, CALL}
TOOLS = [{"type": "function", "function": {
    "name": "execute", "description": "Run an argv command in the isolated task workspace.",
    "parameters": {"type": "object", "properties": {"command": {
        "type": "array", "items": {"type": "string"}}},
        "required": ["command"], "additionalProperties": False}}}]


def _integer(value, low, high, name):
    require(type(value) is int and low <= value <= high, f"{name} must be an integer in [{low}, {high}]")


def pool_residuals(values, mask, pooling):
    """Pool only explicitly valid token rows; zero-valued activations remain valid."""
    import numpy as np
    values, mask = np.asarray(values), np.asarray(mask)
    require(values.ndim == 2 and values.dtype.kind == 'f' and mask.dtype.kind == 'b' and
            mask.shape == (len(values),) and bool(mask.any()), 'Expected residual rows and a nonempty boolean mask')
    require(pooling in ('mean', 'last') and bool(np.isfinite(values[mask]).all()), 'Invalid pooling or valid residuals')
    valid = values if bool(mask.all()) else values[mask]
    return valid.mean(axis=0) if pooling == 'mean' else valid[-1].copy()


def _header(tokenizer, tokens):
    """Parse only this study's text/tool header subset; delimiters come from IDs."""
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    parts = text.split("<|channel|>")
    if len(parts) > 2:
        return None
    words = parts[0].split()
    if not words:
        return None
    role, rest = words[0], words[1:]
    channel = None
    if len(parts) == 2:
        channel_words = parts[1].split()
        if not channel_words:
            return None
        channel, more = channel_words[0], channel_words[1:]
        rest += more
    recipients = [word[3:] for word in rest if word.startswith("to=")]
    constraints = {"json", "code", "<|constrain|>json", "<|constrain|>code"}
    if len(recipients) > 1 or any(word not in constraints and not word.startswith("to=") for word in rest):
        return None
    return {"role": role, "channel": channel, "recipient": recipients[0] if recipients else None}


def _messages(tokenizer, ids):
    """Return token spans and validity, including an unfinished final header/body."""
    result, position = [], 0
    while position < len(ids):
        if ids[position] != START:
            return result, False
        start, position = position, position + 1
        header_start = position
        while position < len(ids) and ids[position] not in STOPS | {START, MESSAGE}:
            position += 1
        if position == len(ids):
            result.append({"start": start, "content_start": None, "end": position,
                           "ending": None, "header": _header(tokenizer, ids[header_start:position])})
            return result, True
        if ids[position] != MESSAGE:
            return result, False
        header = _header(tokenizer, ids[header_start:position])
        if header is None:
            return result, False
        content_start = position + 1
        position = content_start
        while position < len(ids) and ids[position] not in STOPS | {START}:
            position += 1
        if position < len(ids) and ids[position] == START:
            return result, False
        ending = ids[position] if position < len(ids) else None
        result.append({"start": start, "content_start": content_start, "end": position,
                       "ending": ending, "header": header})
        if ending is not None:
            position += 1
    return result, True


def _action(message):
    header = message["header"]
    return bool(header and header["role"] == "assistant" and
                header["channel"] == "commentary" and header["recipient"] == "functions.execute" and
                message["content_start"] is not None)


def _allowed_positions(tokenizer, ids, boundary, schedule):
    messages, valid = _messages(tokenizer, ids)
    if not valid:
        return []
    if schedule == "P1":
        return [message["content_start"] - 1 for message in messages
                if message["start"] > boundary - 2 and _action(message)][:1]
    if schedule == "S2":
        # Stop at any generated non-assistant header, before its content can be patched.
        stop = next((m["start"] for m in messages if m["start"] >= boundary and
                     m["header"] is not None and m["header"]["role"] != "assistant"), len(ids))
        return list(range(boundary, stop))
    return [position for message in messages if message["start"] > boundary - 2 and _action(message)
            for position in range(message["content_start"] - 1, message["end"])]


class ModelRuntime:
    """Own one loaded model. No KV cache, hook, capture, or RNG state survives an operation."""

    def __init__(self, config):
        json_value(config)
        fields(config, {"profile", "model_path", "artifact_root", "max_context_tokens"}, "model config")
        require(type(config["profile"]) is str and config["profile"] in {"tiny-gpt-oss-cpu", "gpt-oss-20b-mxfp4"}, "Unsupported model profile",
                "unsupported_backend")
        _integer(config["max_context_tokens"], 8, 8192, "max_context_tokens")
        self._config = json.dumps(config, sort_keys=True)
        self.model_path = local_path(config["model_path"])
        self.artifact_root = local_path(config["artifact_root"])
        self.bridge = self.tokenizer = None
        self._identity = None
        self.load_seconds = None

    @property
    def config(self):
        return json.loads(self._config)

    @property
    def identity(self):
        require(self._identity is not None, "Call load() first", "not_loaded")
        return json.loads(self._identity)

    def load(self):
        if self.bridge is not None:
            return self.identity
        started = time.monotonic()
        try:
            import torch
            from transformers import AutoTokenizer, GptOssConfig, GptOssForCausalLM
            from transformer_lens.model_bridge.bridge import TransformerBridge
        except ImportError as exc:
            raise InputError("missing_dependency", str(exc)) from exc
        fixture = self.config["profile"] == "tiny-gpt-oss-cpu"
        if not fixture:
            require(torch.cuda.is_available(), "MXFP4 pilot requires a CUDA host", "unsupported_backend")
            require(torch.cuda.get_device_capability(0) >= (7, 5), "GPU lacks MXFP4 kernel support",
                    "unsupported_backend")
            require(os.environ.get("HF_HUB_OFFLINE") == "1", "Load requires HF_HUB_OFFLINE=1 and local kernels")
            from transformers import Mxfp4Config
            from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
            quantizer = Mxfp4HfQuantizer(Mxfp4Config(dequantize=False))
            quantizer.pre_quantized = True
            quantizer.validate_environment()
            require(not quantizer.quantization_config.dequantize,
                    "MXFP4 kernel preflight requested dequantization; refusing fallback", "unsupported_backend")
        for name, expected in ASSETS.items():
            path = local_path(str(self.model_path / name))
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    f"Pinned asset mismatch: {name}", "hash_mismatch")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=False)
        for token_id, spelling in ((START, "<|start|>"), (MESSAGE, "<|message|>"), (CALL, "<|call|>"),
                                   (RETURN, "<|return|>"), (END, "<|end|>"), (CHANNEL, "<|channel|>")):
            require(tokenizer.convert_ids_to_tokens(token_id) == spelling, "Unexpected Harmony token IDs")
        weights = {}
        if fixture:
            cfg = GptOssConfig(vocab_size=201088, hidden_size=16, intermediate_size=16,
                               num_hidden_layers=4, num_local_experts=4, num_experts_per_tok=2,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                               max_position_embeddings=8192, sliding_window=16,
                               pad_token_id=199999, eos_token_id=RETURN,
                               attn_implementation="eager", experts_implementation="eager")
            cfg.architectures = ["GptOssForCausalLM"]
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(17)
                model = GptOssForCausalLM(cfg).eval()
            device, dtype = "cpu", torch.float32
        else:
            index_bytes = local_path(str(self.model_path / "model.safetensors.index.json")).read_bytes()
            require(hashlib.sha256(index_bytes).hexdigest() == WEIGHT_INDEX_SHA256,
                    "Pinned weight index mismatch", "hash_mismatch")
            index = decode_json(index_bytes)
            require(set(index["weight_map"].values()) == set(WEIGHTS), "Unexpected checkpoint shards")
            for name in sorted(set(index["weight_map"].values())):
                path = local_path(str(self.model_path / name))
                require(path.is_relative_to(self.model_path) and path.suffix == ".safetensors", "Invalid weight path")
                with path.open("rb") as stream:
                    weights[name] = hashlib.file_digest(stream, "sha256").hexdigest()
                require(weights[name] == WEIGHTS[name], f"Pinned weight mismatch: {name}", "hash_mismatch")
            model = GptOssForCausalLM.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager",
                quantization_config=quantizer.quantization_config).eval()
            cfg = model.config
            require((cfg.model_type, cfg.num_hidden_layers, cfg.hidden_size, cfg.num_local_experts,
                     cfg.num_experts_per_tok) == ("gpt_oss", 24, 2880, 32, 4), "Unexpected checkpoint dimensions")
            require(not model.hf_quantizer.quantization_config.dequantize, "Loaded format changed", "unsupported_backend")
            device, dtype = "cuda:0", torch.bfloat16
        bridge = TransformerBridge.boot_transformers(MODEL_ID, hf_model=model, tokenizer=tokenizer,
                                                      dtype=dtype, device=device).eval()
        layers = sorted({math.ceil(depth * cfg.num_hidden_layers) - 1 for depth in (0.25, 0.5, 0.75)})
        for layer in layers:
            require(bridge.get_hook_point(f"blocks.{layer}.hook_resid_post") is not None, "Missing residual hook")
        identity = {"model_id": MODEL_ID, "revision": REVISION, "assets": ASSETS,
                    "runtime_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "profile": self.config["profile"], "fixture": fixture, "fixture_seed": 17 if fixture else None,
                    "research_backend_validated": False, "weights": weights, "config": cfg.to_dict(),
                    "versions": {name: version(name) for name in
                                 ("torch", "transformers", "transformer-lens", "numpy", "accelerate", "safetensors")},
                    "device": device, "dtype": str(dtype), "layers": layers,
                    "module_types": {name: f"{type(module).__module__}.{type(module).__qualname__}"
                                     for name, module in model.named_modules() if
                                     re.fullmatch(r"model.layers.\d+(\.self_attn|\.mlp|\.mlp.experts)?", name)},
                    "attention_implementation": cfg._attn_implementation,
                    "generation_use_cache": fixture,
                    "experts_implementation": getattr(cfg, "_experts_implementation", None)}
        if not fixture:
            identity["kernel_versions"] = {name: version(name) for name in ("triton", "kernels")}
            identity["gpu_name"] = torch.cuda.get_device_name(0)
            identity["cuda_version"] = torch.version.cuda
        self._identity = json.dumps(identity, sort_keys=True)
        self.bridge, self.tokenizer = bridge, tokenizer
        self.load_seconds = time.monotonic() - started
        return self.identity

    def reset_episode(self):
        if self.bridge is not None:
            self.bridge.reset_hooks(clear_contexts=True, including_permanent=True)

    def close(self):
        self.reset_episode()
        self.bridge = self.tokenizer = self._identity = None

    def _tokens(self, payload):
        fields(payload, {"token_ids", "attention_mask", "assistant_boundary", "runtime_sha256"}, "token payload")
        ids = payload["token_ids"]
        require(type(ids) is list and 2 <= len(ids) <= self.config["max_context_tokens"], "Invalid token count")
        require(all(type(i) is int and 0 <= i < self.identity["config"]["vocab_size"] for i in ids), "Invalid token ID")
        require(type(payload["attention_mask"]) is list and len(payload["attention_mask"]) == len(ids) and
                all(type(i) is int and i == 1 for i in payload["attention_mask"]), "Only unpadded single sequences are supported")
        require(payload["runtime_sha256"] == fingerprint(self.identity), "Token/runtime identity mismatch", "hash_mismatch")
        boundary = payload["assistant_boundary"]
        _integer(boundary, 1, len(ids) - 1, "assistant_boundary")
        assistant = self.tokenizer.encode("assistant", add_special_tokens=False)
        require(ids[boundary - len(assistant):boundary + 1] == [START] + assistant,
                "Boundary must be the template's open assistant header")
        messages, valid = _messages(self.tokenizer, ids)
        require(valid and any(m["start"] == boundary - len(assistant) for m in messages), "Malformed token prefix")
        require(all(m["header"] and m["header"]["role"] == "assistant" for m in messages
                    if m["start"] >= boundary - len(assistant)), "Prefix includes a future non-assistant message")
        current = [m for m in messages if m["start"] >= boundary - len(assistant)]
        require(all(m["ending"] == END and m["header"]["recipient"] is None and
                    m["header"]["channel"] in {"analysis", "commentary"} for m in current[:-1]),
                "Assistant turn contains an unexpected earlier handoff/final message")
        return ids, boundary, messages

    def prepare(self, inputs):
        fields(inputs, {"messages", "date", "reasoning_effort"}, "prepare inputs")
        require(type(inputs["reasoning_effort"]) is str and inputs["reasoning_effort"] in {"low", "medium", "high"}, "Invalid reasoning effort")
        try:
            require(type(inputs["date"]) is str and date.fromisoformat(inputs["date"]).isoformat() == inputs["date"], "Invalid date")
        except ValueError as exc:
            raise InputError("invalid_input", "Invalid ISO date") from exc
        messages = inputs["messages"]
        require(type(messages) is list and 1 <= len(messages) <= 64, "Expected 1–64 history messages")
        for message in messages:
            require(type(message) is dict and type(message.get("role")) is str and
                    message["role"] in {"system", "developer", "user", "assistant", "tool"}, "Invalid history role")
            require(set(message) <= {"role", "content", "thinking", "tool_calls"}, "Unsupported history field")
            require(type(message.get("content")) is str, "History content must be text")
            require("thinking" not in message or type(message["thinking"]) is str, "Thinking must be text")
            if "tool_calls" in message:
                require(message["role"] == "assistant", "Only assistant messages can call tools")
                calls = message["tool_calls"]
                require(type(calls) is list and len(calls) == 1, "Exactly one call per tool message")
                fields(calls[0], {"type", "function"}, "history tool call")
                fields(calls[0]["function"], {"name", "arguments"}, "history function")
                require(calls[0]["type"] == "function" and calls[0]["function"]["name"] == "execute", "Unsupported history tool")
                self._command(calls[0]["function"]["arguments"])
        require(messages[-1]["role"] in {"user", "tool"}, "History must end in a user or tool message")
        require(len(json.dumps(messages)) <= 262144, "History too large")
        from jinja2 import TemplateError
        try:
            ids = self.tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=True, return_dict=False, add_generation_prompt=True,
                        reasoning_effort=inputs["reasoning_effort"], strftime_now=lambda _: inputs["date"])
        except (TemplateError, ValueError, TypeError) as exc:
            raise InputError("invalid_input", f"History cannot be rendered: {exc}") from exc
        payload = {"token_ids": ids, "attention_mask": [1] * len(ids), "assistant_boundary": len(ids) - 1,
                   "runtime_sha256": fingerprint(self.identity)}
        self._tokens(payload)
        return payload

    @staticmethod
    def _command(arguments):
        fields(arguments, {"command"}, "tool arguments")
        command = arguments["command"]
        require(type(command) is list and 1 <= len(command) <= 64 and
                all(type(arg) is str and "\0" not in arg for arg in command) and bool(command[0]), "Invalid argv")
        return command

    def resume(self, inputs):
        """Append a template-rendered tool result without retokenizing earlier generation."""
        fields(inputs, {"trajectory", "content"}, "resume inputs")
        require(type(inputs["content"]) is str and len(inputs["content"]) <= 262144, "Invalid tool result text")
        payload = decode_json(read_artifact(inputs["trajectory"], "json", 4 * 1024 * 1024))
        response = self._response(payload)
        require(response["status"] == "tool_call", "Only a valid tool handoff can receive a result")
        # Only the suffix is used; earlier generated IDs remain byte-for-byte identical.
        rendered = self.prepare({"messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "execute", "arguments": {"command": response["command"]}}}]},
            {"role": "tool", "content": inputs["content"]}], "date": "2000-01-01", "reasoning_effort": "medium"})
        messages, valid = _messages(self.tokenizer, rendered["token_ids"])
        require(valid and messages[-2]["header"]["role"] == "functions.execute", "Unexpected template tool suffix")
        payload["token_ids"] += rendered["token_ids"][messages[-2]["start"]:]
        payload["attention_mask"] = [1] * len(payload["token_ids"])
        payload["assistant_boundary"] = len(payload["token_ids"]) - 1
        self._tokens(payload)
        return payload

    def _response(self, payload):
        ids, boundary, messages = self._tokens(payload)
        generated = [m for m in messages if m["start"] >= boundary - 1]
        for message in generated:
            message["content"] = (self.tokenizer.decode(ids[message["content_start"]:message["end"]], skip_special_tokens=False)
                                  if message["content_start"] is not None else None)
        last = generated[-1]
        result = {"status": "truncated", "messages": generated, "command": None, "action_span": None,
                  "pre_action_position": None}
        if last["ending"] == CALL:
            if not _action(last) or last["content_start"] == last["end"]:
                result["status"] = "malformed"
            else:
                result["action_span"] = [last["content_start"], last["end"]]
                result["pre_action_position"] = last["content_start"] - 1
                try:
                    result["command"] = self._command(decode_json(last["content"]))
                    result["status"] = "tool_call"
                except InputError as exc:
                    result.update(status="malformed", parse_error=str(exc))
        elif last["ending"] == RETURN:
            result["status"] = "final" if last["header"]["channel"] == "final" and last["header"]["recipient"] is None else "malformed"
        return result

    @contextmanager
    def _hooks(self, payload, intervention, events, observations=None):
        import numpy as np
        import torch
        ids, boundary, _ = self._tokens(payload)
        if intervention is None:
            yield
            return
        fields(intervention, {"layer", "schedule", "mode", "direction", "value", "runtime_sha256"}, "intervention")
        _integer(intervention["layer"], 0, self.identity["config"]["num_hidden_layers"] - 1, "layer")
        require(type(intervention["schedule"]) is str and intervention["schedule"] in {"P1", "S1", "S2"} and
                type(intervention["mode"]) is str and intervention["mode"] in {"add", "replace_projection"}, "Invalid intervention")
        require(type(intervention["value"]) in (int, float) and math.isfinite(intervention["value"]), "Invalid intervention value")
        require(intervention["runtime_sha256"] == fingerprint(self.identity), "Direction/runtime identity mismatch", "hash_mismatch")
        try:
            with np.load(io.BytesIO(read_artifact(intervention["direction"], "npz", 1024 * 1024)), allow_pickle=False) as arrays:
                require(arrays.files == ["direction"], "Expected a single direction array")
                direction = arrays["direction"]
                require(direction.shape == (self.identity["config"]["hidden_size"],) and direction.dtype.kind == "f" and
                        bool(np.isfinite(direction).all()) and abs(float(np.linalg.norm(direction)) - 1) < 1e-5, "Direction must be a finite unit vector")
                direction = torch.from_numpy(direction.astype(np.float32, copy=True)).to(self.identity["device"])
        except (ValueError, TypeError) as exc:
            if isinstance(exc, InputError):
                raise
            raise InputError("invalid_input", f"Invalid direction array: {exc}") from exc
        seen, offset = [], 0

        def track(module, args, kwargs):
            nonlocal seen, offset
            tokens = kwargs.get("input_ids", args[0] if args else None)
            mask = kwargs["attention_mask"]
            offset = mask.shape[-1] - tokens.shape[-1]
            current = tokens[0].tolist()
            require(offset == 0 or offset == len(seen), "Unexpected cache positions", "runtime_error")
            seen = current if offset == 0 else seen + current

        def patch(residual, hook):
            positions = [p for p in _allowed_positions(self.tokenizer, seen, boundary, intervention["schedule"])
                         if offset <= p < offset + residual.shape[1]]
            if not positions:
                return residual
            local = [p - offset for p in positions]
            output = residual.clone()
            original = residual[:, local, :].float()
            delta = intervention["value"]
            if intervention["mode"] == "replace_projection":
                delta = delta - (original @ direction).unsqueeze(-1)
            changed = (original + delta * direction).to(residual.dtype)
            require(bool(torch.isfinite(changed).all()), "Intervention produced nonfinite runtime values", "numerical_error")
            output[:, local, :] = changed
            changes = changed.float() - original
            norms = changes.norm(dim=-1)
            projection_before = original @ direction
            projection_after = changed.float() @ direction
            stats = torch.stack((norms.min(), norms.max(), norms.sum(), norms.square().sum(),
                                 (projection_after - projection_before).sum())).detach().cpu().tolist()
            require(all(math.isfinite(value) for value in stats), "Nonfinite intervention diagnostics", "numerical_error")
            events.append({"layer": intervention["layer"], "processed_positions": positions,
                           "predicted_positions": [p + 1 for p in positions],
                           "next_token_role": "action" if intervention["schedule"] in {"P1", "S1"} else "assistant",
                           "runtime_dtype": str(residual.dtype), "position_count": len(positions),
                           "min_change_norm": stats[0], "max_change_norm": stats[1], "change_norm_sum": stats[2],
                           "change_norm_squared_sum": stats[3], "projection_change_sum": stats[4]})
            if observations is not None:
                requested = np.full(len(positions), intervention["value"], dtype=np.float64)
                if intervention["mode"] == "replace_projection":
                    requested -= projection_before[0].detach().cpu().numpy().astype(np.float64)
                observations.append({"positions": np.array(positions, dtype=np.int64),
                    "change_norms": norms[0].detach().cpu().numpy().copy(),
                    "projection_before": projection_before[0].detach().cpu().numpy().copy(),
                    "projection_after": projection_after[0].detach().cpu().numpy().copy(),
                    "requested_deltas": requested,
                    "boundary_before": original[0, -1].detach().cpu().numpy().copy(),
                    "boundary_after": changed[0, -1].float().detach().cpu().numpy().copy(),
                    "direction": direction.detach().cpu().numpy().copy()})
            return output

        handle = self.bridge.original_model.register_forward_pre_hook(track, with_kwargs=True)
        try:
            with self.bridge.hooks(fwd_hooks=[(f"blocks.{intervention['layer']}.hook_resid_post", patch)], clear_contexts=True):
                yield
        finally:
            handle.remove()

    def generate(self, inputs, directory):
        import torch
        from transformers import StoppingCriteriaList
        fields(inputs, {"prefix", "seed", "max_new_tokens", "temperature", "max_seconds", "intervention"}, "generate inputs")
        payload = decode_json(read_artifact(inputs["prefix"], "json", 4 * 1024 * 1024))
        ids, _, messages = self._tokens(payload)
        require(messages[-1]["ending"] is None, "Generation prefix is already complete")
        _integer(inputs["seed"], 0, 2**32 - 1, "seed")
        _integer(inputs["max_new_tokens"], 1, 2048, "max_new_tokens")
        require(len(ids) + inputs["max_new_tokens"] <= self.config["max_context_tokens"], "Generation exceeds context budget")
        require(type(inputs["temperature"]) in (float, int) and 0 <= inputs["temperature"] <= 2, "Temperature must be in [0, 2]")
        require(type(inputs["max_seconds"]) in (float, int) and 0 < inputs["max_seconds"] <= 600, "Invalid generation time budget")
        tensor = torch.tensor([ids], device=self.identity["device"])
        events = []
        partial_ids = ids[:]

        def retain_tokens(input_ids, scores, **kwargs):
            nonlocal partial_ids
            partial_ids = input_ids[0].tolist()
            parsed, valid = _messages(self.tokenizer, partial_ids)
            return not valid or any(m["start"] >= payload["assistant_boundary"] and m["content_start"] is not None and
                m["header"] is not None and m["header"]["role"] != "assistant" for m in parsed)

        started = time.monotonic()
        devices = [] if self.identity["fixture"] else [0]
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode(), self._hooks(payload, inputs["intervention"], events):
                torch.random.default_generator.manual_seed(inputs["seed"])
                if devices:
                    torch.cuda.default_generators[0].manual_seed(inputs["seed"])
                output = self.bridge.hf_generate(tensor, attention_mask=torch.ones_like(tensor),
                    max_new_tokens=inputs["max_new_tokens"], do_sample=inputs["temperature"] > 0,
                    temperature=inputs["temperature"] or 1.0, top_k=0, top_p=1.0,
                    eos_token_id=[RETURN, 199999, CALL], use_past_kv_cache=self.identity['generation_use_cache'],
                    use_cache=self.identity['generation_use_cache'], return_type="tokens",
                    max_time=inputs["max_seconds"], return_dict_in_generate=False, logits_to_keep=1,
                    stopping_criteria=StoppingCriteriaList([retain_tokens]))
            require(output.shape[0] == 1 and output[0, :len(ids)].tolist() == ids, "Generation changed input tokens", "runtime_error")
            payload["token_ids"] = output[0].tolist()
            partial_ids = payload["token_ids"]
            payload["attention_mask"] = [1] * len(payload["token_ids"])
            atomic_json(directory / "tokens.json", payload)
            raw = self.tokenizer.decode(payload["token_ids"][len(ids):], skip_special_tokens=False)
            atomic_bytes(directory / "raw_response.txt", raw.encode("utf-8"))
            try:
                response = self._response(payload)
            except InputError as exc:
                response = {"status": "malformed", "parse_error": str(exc), "command": None}
            response.update(tokens=artifact_ref(directory / "tokens.json", "json"),
                            raw_response=artifact_ref(directory / "raw_response.txt", "text"),
                            generated_tokens=len(payload["token_ids"]) - len(ids),
                            generation_seconds=time.monotonic() - started,
                            time_budget_reached=time.monotonic() - started >= inputs["max_seconds"],
                            hook_events=events)
            return response
        finally:
            payload["token_ids"] = partial_ids
            payload["attention_mask"] = [1] * len(partial_ids)
            atomic_json(directory / "tokens.json", payload)
            atomic_bytes(directory / "raw_response.txt", self.tokenizer.decode(
                partial_ids[len(ids):], skip_special_tokens=False).encode("utf-8"))
            atomic_json(directory / "hook_events.json", events)
            self.reset_episode()

    def diagnose(self, inputs, directory):
        """Two same-prefix forwards; compact actual-dtype and downstream evidence."""
        import numpy as np
        import torch
        fields(inputs, {"prefix", "intervention", "max_seconds"}, "diagnostic inputs")
        payload = decode_json(read_artifact(inputs["prefix"], "json", 4 * 1024 * 1024))
        ids, boundary, messages = self._tokens(payload)
        require(messages and _action(messages[-1]) and messages[-1]["ending"] is None and
                messages[-1]["content_start"] == len(ids), "Diagnostics require an exact pre-action prefix without action content", "ineligible")
        require(type(inputs["max_seconds"]) in (int, float) and math.isfinite(inputs["max_seconds"]) and 0 < inputs["max_seconds"] <= 600, "Invalid diagnostic time budget")
        require(inputs["intervention"] is not None, "An explicit intervention, including a zero sham, is required")
        # Validate all hook inputs before the first potentially expensive model forward.
        with self._hooks(payload, inputs["intervention"], []):
            pass
        layer = inputs["intervention"]["layer"]
        downstream = list(range(layer+1, self.identity["config"]["num_hidden_layers"]))
        arrays, events, observations = {}, [], []
        started = time.monotonic()
        report = {"schema_version": 1, "status": "incomplete", "inputs": inputs,
            "runtime_sha256": fingerprint(self.identity), "fixture": self.identity["fixture"], "completed_passes": [],
            "causal_prefix_length": len(ids), "prediction_boundary": len(ids)-1,
            "positions": _allowed_positions(self.tokenizer, ids, boundary, inputs["intervention"]["schedule"]),
            "downstream_layers": downstream, "router_calls": {}, "downstream_residual_calls": {}, "hook_events": events,
            "method": "same exact prefix; baseline and intervention; fresh full forwards without KV cache"}
        tensor = torch.tensor([ids], device=self.identity["device"])
        devices = [] if self.identity["fixture"] else [0]

        def save():
            report["elapsed_seconds"] = time.monotonic() - started
            if report["status"] == "complete" and report["elapsed_seconds"] >= inputs["max_seconds"]:
                report["status"] = "time_limit"
            if arrays:
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays)
                atomic_bytes(directory / "diagnostic.npz", stream.getvalue())
                report["arrays"] = artifact_ref(directory / "diagnostic.npz", "npz")
            atomic_json(directory / "diagnostic.json", report)

        save()
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                for name, intervention in (("baseline", None), ("intervened", inputs["intervention"])):
                    require(time.monotonic() - started < inputs["max_seconds"], "Diagnostic deadline reached before the next forward", "time_limit")
                    routing, residuals, handles = {}, {}, []
                    calls = dict.fromkeys(downstream, 0)
                    residual_calls = dict.fromkeys(downstream, 0)
                    try:
                        for index in downstream:
                            def observe(module, args, output, index=index):
                                calls[index] += 1
                                require(type(output) in (tuple, list) and len(output) == 3, "Unrecognized upstream router output", "unsupported_backend")
                                logits, scores, choices = output
                                require(logits.ndim == scores.ndim == choices.ndim == 2 and logits.shape[0] == len(ids), "Unexpected router geometry", "unsupported_backend")
                                require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(scores).all()), "Nonfinite router diagnostics", "numerical_error")
                                routing[index] = (logits[-1].float().detach().cpu().numpy().copy(), choices[-1].detach().cpu().numpy().copy())
                            handles.append(self.bridge.original_model.model.layers[index].mlp.router.register_forward_hook(observe))
                            # MXFP4 returns the executed functional-linear router logits
                            # from the MLP, without calling router.forward. Choices are
                            # absent from that output and must not be invented.
                            def observe_mlp(module, args, output, index=index):
                                if index in routing or getattr(module.forward, '__module__', '') != 'transformers.integrations.mxfp4':
                                    return
                                require(type(output) in (tuple, list) and len(output) == 2,
                                        "Unrecognized upstream MLP output", "unsupported_backend")
                                logits = output[1]
                                require(tuple(logits.shape) == (len(ids), self.identity['config']['num_local_experts']) and
                                        bool(torch.isfinite(logits).all()), "Unexpected MLP router geometry", "unsupported_backend")
                                calls[index] += 1
                                routing[index] = (logits[-1].float().detach().cpu().numpy().copy(), None)
                            mlp = self.bridge.original_model.model.layers[index].mlp
                            handles.append(getattr(mlp, '_original_component', mlp).register_forward_hook(observe_mlp))
                            def observe_residual(module, args, output, index=index):
                                require(tuple(output.shape) == (1, len(ids), self.identity['config']['hidden_size']) and
                                        bool(torch.isfinite(output).all()), "Invalid downstream residual geometry", "unsupported_backend")
                                residual_calls[index] += 1
                                residuals[index] = output[0, -1].float().detach().cpu().numpy().copy()
                            handles.append(self.bridge.original_model.model.layers[index].register_forward_hook(observe_residual))
                        with self._hooks(payload, intervention, events, observations if intervention is not None else None):
                            logits = self.bridge(tensor, prepend_bos=False, attention_mask=torch.ones_like(tensor), use_cache=False, logits_to_keep=1)
                        require(tuple(logits.shape) == (1, 1, self.identity["config"]["vocab_size"]) and bool(torch.isfinite(logits).all()), "Invalid diagnostic logits", "numerical_error")
                        arrays[name + "_logits"] = logits[0, -1].float().detach().cpu().numpy().copy()
                        report["router_calls"][name] = [calls[index] for index in downstream]
                        report["downstream_residual_calls"][name] = [residual_calls[index] for index in downstream]
                        if downstream and all(calls[index] == 1 for index in downstream):
                            arrays[name + "_router_logits"] = np.stack([routing[index][0] for index in downstream])
                            if all(routing[index][1] is not None for index in downstream):
                                arrays[name + "_router_choices"] = np.stack([routing[index][1] for index in downstream])
                        if downstream and all(residual_calls[index] == 1 for index in downstream):
                            arrays[name + '_downstream_residuals'] = np.stack([residuals[index] for index in downstream])
                        report["completed_passes"].append(name)
                    finally:
                        for handle in handles:
                            handle.remove()
                        if observations:
                            require(len(observations) == 1, "Expected one diagnostic intervention application")
                            arrays.update(observations[0])
                        save()
            require(len(events) == 1 and events[0]["processed_positions"] == report["positions"], "Diagnostic hook did not cover its declared positions", "runtime_error")
            difference = arrays["intervened_logits"].astype(np.float64) - arrays["baseline_logits"]
            report["statistics"] = {"logit_l2_change": float(np.linalg.norm(difference)), "logit_max_abs_change": float(np.abs(difference).max()),
                "baseline_argmax": int(arrays["baseline_logits"].argmax()), "intervened_argmax": int(arrays["intervened_logits"].argmax()),
                "changed_positions": int(np.count_nonzero(arrays["change_norms"])), "requested_nonzero": bool(np.any(arrays["requested_deltas"] != 0)),
                "observed_nonzero": bool(np.any(arrays["change_norms"] != 0)),
                "router_observation": "not_applicable" if not downstream else ("observed" if all(count == 1 for counts in report["router_calls"].values() for count in counts) else "unavailable")}
            if all(name + "_router_logits" in arrays for name in ("baseline", "intervened")):
                report["statistics"]['router_logit_l2_change'] = float(np.linalg.norm(arrays["intervened_router_logits"].astype(np.float64) - arrays["baseline_router_logits"]))
            choices_observed = all(name + '_router_choices' in arrays for name in ('baseline', 'intervened'))
            report['statistics']['router_choice_observation'] = 'observed' if choices_observed else 'unavailable'
            if choices_observed:
                report['statistics']['changed_router_choice_slots'] = int(np.count_nonzero(arrays['intervened_router_choices'] != arrays['baseline_router_choices']))
            residuals_observed = all(name + '_downstream_residuals' in arrays for name in ('baseline', 'intervened'))
            report['statistics']['downstream_residual_observation'] = 'observed' if residuals_observed else ('not_applicable' if not downstream else 'unavailable')
            if residuals_observed:
                report['statistics']['downstream_residual_l2_change'] = float(np.linalg.norm(arrays['intervened_downstream_residuals'].astype(np.float64) - arrays['baseline_downstream_residuals']))
            report["status"] = "complete" if time.monotonic() - started < inputs["max_seconds"] else "time_limit"
        finally:
            self.reset_episode()
            save()
        return report | {"diagnostic": artifact_ref(directory / "diagnostic.json", "json")}

    def capture(self, inputs, directory):
        import numpy as np
        import torch
        fields(inputs, {"trajectory", "target", "layers"}, "capture inputs")
        payload = decode_json(read_artifact(inputs["trajectory"], "json", 4 * 1024 * 1024))
        ids, _, _ = self._tokens(payload)
        response = self._response(payload)
        require(response.get("action_span") is not None, "No completed action-content span", "ineligible")
        require(type(inputs["target"]) is str and inputs["target"] in {"action", "pre_action"}, "Invalid capture target")
        layers = inputs["layers"]
        require(type(layers) is list and bool(layers) and all(type(i) is int for i in layers) and
                len(layers) == len(set(layers)), "Expected unique integer layers")
        for layer in layers:
            _integer(layer, 0, self.identity["config"]["num_hidden_layers"] - 1, "layer")
        start, stop = response["action_span"]
        positions = list(range(start, stop)) if inputs["target"] == "action" else [start - 1]
        cutoff = max(positions) + 1
        tensor = torch.tensor([ids[:cutoff]], device=self.identity["device"])
        names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
        try:
            with torch.inference_mode():
                _, cache = self.bridge.run_with_cache(tensor, attention_mask=torch.ones_like(tensor),
                    prepend_bos=False, names_filter=names, return_cache_object=False,
                    return_type=None, use_cache=False, logits_to_keep=1)
            raw = [cache[name][0, positions, :].detach() for name in names]
            require(all(tuple(value.shape) == (len(positions), self.identity["config"]["hidden_size"]) for value in raw),
                    "Capture shape mismatch", "runtime_error")
            values = np.stack([value.float().cpu().numpy() for value in raw])
            require(bool(np.isfinite(values).all()), "Nonfinite residuals", "runtime_error")
            stream = io.BytesIO()
            mask = np.ones(len(positions), dtype=bool)
            np.savez_compressed(stream, residuals=values,
                                mean=np.stack([pool_residuals(layer, mask, 'mean') for layer in values]),
                                last=np.stack([pool_residuals(layer, mask, 'last') for layer in values]),
                                positions=np.array(positions, dtype=np.int64), layers=np.array(layers, dtype=np.int64))
            atomic_bytes(directory / "features.npz", stream.getvalue())
            return {"features": artifact_ref(directory / "features.npz", "npz"), "shape": list(values.shape),
                    "trajectory": inputs["trajectory"].copy(),
                    "runtime_dtypes": [str(value.dtype) for value in raw], "stored_dtype": "float32",
                    "positions": positions, "prediction_boundary": start - 1, "causal_prefix_length": cutoff,
                    "target": inputs["target"], "sites": [{"hook": name, "module": f"model.layers.{layer}",
                      "meaning": "decoder output after attention and MoE residual additions; before next block/final norm"}
                     for layer, name in zip(layers, names)], "runtime_sha256": fingerprint(self.identity)}
        finally:
            self.reset_episode()

    def check(self, inputs, directory):
        from .runtime_checks import check
        return check(self, inputs, directory)

    def benign_control(self, inputs, directory):
        from .runtime_checks import benign_control
        return benign_control(self, inputs, directory)

    def handle(self, request):
        directory = None
        record = None
        try:
            validate_request(request, OPERATIONS)
            require(fingerprint(request["config"]) == fingerprint(self.config), "Runtime configuration changed")
            if request['operation'] == 'check':
                from .runtime_checks import validate_inputs
                validate_inputs(request['inputs'])
            elif request['operation'] == 'benign_control':
                from .runtime_checks import validate_benign_inputs
                validate_benign_inputs(request['inputs'])
            directory = self.artifact_root / (request["request_id"] + "-" + uuid.uuid4().hex)
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(directory / "request.json", request)
            record = {"status": "started", "operation": request["operation"], "request_id": request["request_id"]}
            atomic_json(directory / "record.json", record)
            started = time.monotonic()
            self.load()
            atomic_json(directory / "identity.json", self.identity)
            if request["operation"] == "load":
                fields(request["inputs"], set(), "load inputs")
                result = self.identity
            elif request["operation"] in {"prepare", "resume"}:
                payload = getattr(self, request["operation"])(request["inputs"])
                atomic_json(directory / "tokens.json", payload)
                result = {"tokens": artifact_ref(directory / "tokens.json", "json")}
            else:
                result = getattr(self, request["operation"])(request["inputs"], directory)
            record.update(status="complete", result=result, load_seconds=self.load_seconds,
                          operation_seconds=time.monotonic() - started,
                          process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          process_peak_rss_units="bytes" if os.uname().sysname == "Darwin" else "KiB")
            if not self.identity["fixture"]:
                import torch
                record["cuda_process_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                record["cuda_process_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            atomic_json(directory / "record.json", record)
            return success(request, result, [artifact_ref(directory / "identity.json", "json"),
                                             artifact_ref(directory / "record.json", "json")])
        except (InputError, OSError) as exc:
            error = exc if isinstance(exc, InputError) else InputError("file_error", str(exc))
            if record is not None:
                record.update(status="error", error={"code": error.code, "message": str(error)})
            return failure(request, error)
        except Exception as exc:
            if record is not None:
                record.update(status="incomplete", failure_type=type(exc).__name__)
                atomic_bytes(directory / "traceback.txt", traceback.format_exc().encode("utf-8"))
            raise
        finally:
            if record is not None:
                atomic_json(directory / "record.json", record)


def handle(request):
    runtime = None
    try:
        validate_request(request, OPERATIONS)
        runtime = ModelRuntime(request["config"])
        return runtime.handle(request)
    except (InputError, OSError) as exc:
        return failure(request, exc if isinstance(exc, InputError) else InputError("file_error", str(exc)))
    finally:
        if runtime is not None:
            runtime.close()
