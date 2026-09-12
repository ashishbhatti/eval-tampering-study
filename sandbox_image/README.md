# Reproducible editing sandbox

This extends the exact Python base in `Dockerfile` with the official GPT-OSS
patch utility. `source.json` records the upstream revision and hashes; `LICENSE`
is the upstream Apache-2.0 license. `apply_patch.py` is unmodified. The small
`apply_patch` launcher makes parsing/filesystem errors return a nonzero exit.
No model, task, answer or protected evaluator is included in the image.

From the repository root on the Linux server, after the pinned base image is
available locally:

```bash
mkdir -p .cache
docker build --network=none --pull=false --iidfile .cache/sandbox-patch-image.id sandbox_image
.venv/bin/python - <<'PY'
import json, subprocess
from pathlib import Path
image = Path('.cache/sandbox-patch-image.id').read_text().strip()
image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
request = json.loads(Path('configs/sandbox_example.json').read_text())
request['config']['image'] = image_id
Path('.cache/sandbox-patch.json').write_text(json.dumps(request, indent=2)+'\n')
print(image_id)
PY
```

Use that immutable `image` value in new episode/evaluation configurations. The
runner checks its identity, pinned base/tool labels and base-layer ancestry;
mutable tags are rejected. The existing read-only root, UID, network denial,
seccomp, CPU/memory/process/file limits and isolated workspace remain enforced.
Run live checks with `EVAL_TAMPERING_DOCKER_TEST_CONFIG=.cache/sandbox-patch.json`.

Preserve completed requests and results. A changed image requires a new attempt
and matching configuration hashes; do not rewrite historical artifacts. Replaying
a saved command for an engineering check is distinct from generating a new model
episode. Record the build output/image inspection with each server handoff.
