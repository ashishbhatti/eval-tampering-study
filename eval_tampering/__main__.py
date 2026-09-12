"""Read one request and write one result. No component loads on import."""

import argparse
from pathlib import Path
import sys

from .messages import InputError, atomic_json, failure, read_json, require


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=["tasks", "sandbox", "evaluate", "model", "monitors", "interventions", "analysis", "impossible"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    request = None
    try:
        root = Path.cwd().resolve()
        input_path, output_path = args.input.resolve(), args.output.resolve()
        require(input_path.is_relative_to(root) and output_path.is_relative_to(root),
                "Input and output must stay under the current working directory", "invalid_path")
        require(input_path != output_path, "Output must not overwrite input", "invalid_path")
    except (InputError, OSError) as exc:
        print(f"Invalid input/output path: {exc}", file=sys.stderr)
        return 2
    try:
        request = read_json(input_path)
        if args.module == "tasks":
            from .tasks import handle
        elif args.module == "sandbox":
            from .sandbox import handle
        elif args.module == "evaluate":
            from .evaluate import handle
        elif args.module == "model":
            from .model import handle
        elif args.module == "interventions":
            from .interventions import handle
        elif args.module == "analysis":
            from .analysis import handle
        elif args.module == "impossible":
            from .impossible import handle
        else:
            from .monitors import handle
        result = handle(request)
    except (InputError, OSError) as exc:
        error = exc if isinstance(exc, InputError) else InputError("file_error", str(exc))
        result = failure(request, error)
    try:
        atomic_json(output_path, result)
    except OSError as exc:
        print(f"Cannot write result: {exc}", file=sys.stderr)
        return 2
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
