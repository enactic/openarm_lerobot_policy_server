# OpenArm LeRobot Policy Server

[LeRobot](https://github.com/huggingface/lerobot) policy servers for
[OpenArm](https://openarm.dev/) inference with [dora-rs](https://dora-rs.ai/).

They serve a pretrained LeRobot policy over a Unix domain socket to the policy
server nodes used in
[dora-openarm-evaluation](https://github.com/enactic/dora-openarm-evaluation):

| Command | dora node | Description |
| --- | --- | --- |
| `openarm-lerobot-policy-server-local` | [dora-openarm-local-policy-server](https://github.com/enactic/dora-openarm-local-policy-server) | Runs on the host and listens on the socket. |
| `openarm-lerobot-policy-server-docker` | [dora-openarm-docker-policy-server](https://github.com/enactic/dora-openarm-docker-policy-server) | Runs in the Docker image and connects to the socket. |

## Setup

```bash
git clone git@github.com:enactic/openarm_lerobot_policy_server.git
cd openarm_lerobot_policy_server
uv sync
source .venv/bin/activate
# uv pip install --torch-backend=cu128 --reinstall torch torchvision  # for CUDA 12.8
```

## Policy

The policy is
[enactic/act-openarm-2-cell-pick_up_cube_mujoco](https://huggingface.co/enactic/act-openarm-2-cell-pick_up_cube_mujoco)
by default. To use another one, pass a Hugging Face Hub repository ID or a local
directory with `--policy`, or set it to the `OPENARM_LEROBOT_POLICY`
environment variable. The policy must take `observation.state` and the
`observation.images.{ceiling,head_left,head_right,wrist_left,wrist_right}`
camera images as its inputs.

## Local inference

```bash
openarm-lerobot-policy-server-local /dev/shm/policy-server.socket
# openarm-lerobot-policy-server-local /dev/shm/policy-server.socket --policy ${POLICY}
```

Then run `dataflow-local-inference.yaml` in
[dora-openarm-evaluation](https://github.com/enactic/dora-openarm-evaluation)
with the same `SOCKET`.

## Docker inference

Pull a released Docker image for your PyTorch backend:

| Tag | PyTorch backend |
| --- | --- |
| `latest`, `cpu-latest` | CPU |
| `cu130-latest` | CUDA 13.0 |
| `cu128-latest` | CUDA 12.8 |
| `cu126-latest` | CUDA 12.6 |

```bash
docker pull ghcr.io/enactic/openarm_lerobot_policy_server:latest
```

`${BACKEND}-${VERSION}` such as `cu128-1.0.0` is also available for each
release. `main` and `${BACKEND}-main` are built from the `main` branch.

Or build the Docker image:

```bash
docker build -t ghcr.io/enactic/openarm_lerobot_policy_server:latest .
```

The policy and its weights are downloaded at build time, so the container runs
offline. Use `--build-arg POLICY=${POLICY}` to bake another policy into the
image:

```bash
docker build \
  --build-arg POLICY=${POLICY} \
  -t ghcr.io/enactic/openarm_lerobot_policy_server:latest \
  .
```

PyTorch for CPU is installed by default. Use `--build-arg
TORCH_BACKEND=${BACKEND}` to install PyTorch for another backend such as
`cu128` (CUDA 12.8). Make sure to match the CUDA version with your host
machine:

```bash
docker build \
  --build-arg TORCH_BACKEND=cu128 \
  -t ghcr.io/enactic/openarm_lerobot_policy_server:latest \
  .
```

Then run `dataflow-docker-inference.yaml` or
`dataflow-docker-evaluation.yaml` in
[dora-openarm-evaluation](https://github.com/enactic/dora-openarm-evaluation).

## Tests

```bash
uv run pytest
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
