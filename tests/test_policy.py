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

"""Options and observation conversion that don't need the pretrained policy."""

import argparse

import numpy as np
import pyarrow as pa
import pytest

from openarm_lerobot_policy_server.policy import (
    CAMERA_KEY_MAP,
    DEFAULT_POLICY,
    POLICY_ENV,
    add_policy_argument,
    detect_resolution,
    observation_to_batch,
    prepare_image,
)


def parse_policy(*args):
    parser = argparse.ArgumentParser()
    add_policy_argument(parser)
    return parser.parse_args(args).policy


def test_policy_default(monkeypatch):
    monkeypatch.delenv(POLICY_ENV, raising=False)
    assert parse_policy() == DEFAULT_POLICY


def test_policy_environment(monkeypatch):
    monkeypatch.setenv(POLICY_ENV, "user/policy")
    assert parse_policy() == "user/policy"


def test_policy_option(monkeypatch):
    monkeypatch.setenv(POLICY_ENV, "user/policy")
    assert parse_policy("--policy", "/path/to/policy") == "/path/to/policy"


def image(h, w):
    return pa.array([np.zeros(h * w * 3, dtype=np.uint8)], pa.list_(pa.uint8()))[0]


def test_detect_known_resolution():
    assert detect_resolution(600 * 960 * 3) == (600, 960)


def test_detect_ratio_resolution():
    assert detect_resolution(240 * 320 * 3) == (240, 320)


def test_detect_unknown_resolution():
    with pytest.raises(ValueError):
        detect_resolution(11 * 13 * 3)


def test_prepare_image_resizes():
    tensor = prepare_image(image(480, 640), 240, 320)
    assert tuple(tensor.shape) == (3, 240, 320)


def test_observation_to_batch():
    fields = {"position": pa.array([[0.0] * 16], pa.list_(pa.float32()))}
    for arrow_key in CAMERA_KEY_MAP:
        fields[arrow_key] = pa.array(
            [np.zeros(480 * 640 * 3, dtype=np.uint8)], pa.list_(pa.uint8())
        )
    observation = pa.StructArray.from_arrays(
        list(fields.values()), names=list(fields.keys())
    )[0]
    image_sizes = {model_key: (240, 320) for model_key in CAMERA_KEY_MAP.values()}

    batch = observation_to_batch(observation, image_sizes)

    assert tuple(batch["observation.state"].shape) == (16,)
    for model_key in CAMERA_KEY_MAP.values():
        assert tuple(batch[model_key].shape) == (3, 240, 320)
