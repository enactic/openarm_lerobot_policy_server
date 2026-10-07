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

"""Docker-based policy server for OpenArm inference.

It runs in the Docker image and connects to the Unix domain socket opened by
the dora-openarm-docker-policy-server node.
"""

import argparse
import socket

from .policy import Policy, add_policy_argument


def main():
    """Run the Docker-based policy server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("socket_path", help="Unix domain socket path to connect to")
    add_policy_argument(parser)
    args = parser.parse_args()
    socket_path = args.socket_path

    policy = Policy(args.policy)

    print(f"Connecting to {socket_path}...")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(socket_path)
        with sock.makefile("rw") as io:
            policy.serve(io)


if __name__ == "__main__":
    main()
