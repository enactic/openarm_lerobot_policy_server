# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ghcr.io/astral-sh/uv:0.13.1 AS uv
FROM python:3.15

SHELL ["/bin/bash", "-c"]
WORKDIR /project

COPY --from=uv /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/project/.venv \
    PATH="/project/.venv/bin:$PATH"

RUN uv venv .venv

# The PyTorch backend passed to uv's --torch-backend such as cpu and
# cu128. Make sure to match the CUDA version with your host machine.
ARG TORCH_BACKEND=cpu

# Install dependencies before copying the source so that changing the
# source doesn't reinstall them.
COPY pyproject.toml ./
RUN uv pip install --torch-backend=${TORCH_BACKEND} -r pyproject.toml
# Download the backbone weights so that the container can run offline.
RUN python -c "import torchvision; torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)"

COPY README.md ./
COPY src/ src/
RUN uv pip install --no-deps .

ARG POLICY=enactic/act-openarm-2-cell-pick_up_cube_mujoco
ENV OPENARM_LEROBOT_POLICY=${POLICY}

# Download the policy so that the container can run offline.
RUN python -c "\
import os; \
from openarm_lerobot_policy_server.policy import POLICY_ENV, Policy; \
Policy(os.environ[POLICY_ENV], device='cpu')"

ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["openarm-lerobot-policy-server-docker"]
