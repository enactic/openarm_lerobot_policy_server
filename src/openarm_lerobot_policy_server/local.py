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

"""Local policy server for OpenArm inference.

It listens on a Unix domain socket and serves the first
dora-openarm-local-policy-server node that connects to it.
"""

import argparse
import os
import socket

from .policy import Policy, add_policy_argument

DEFAULT_SOCKET = "/dev/shm/policy-server.socket"


def main():
    """Run the local policy server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "socket_path",
        nargs="?",
        default=DEFAULT_SOCKET,
        help=f"Unix domain socket path to listen on (default: {DEFAULT_SOCKET})",
    )
    add_policy_argument(parser)
    args = parser.parse_args()
    socket_path = args.socket_path

    policy = Policy(args.policy)

    if os.path.exists(socket_path):
        os.remove(socket_path)

    print(f"Listening on {socket_path}")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(socket_path)
        sock.listen()
        try:
            with sock.accept()[0] as conn:
                print("Connected")
                with conn.makefile("rw") as io:
                    policy.serve(io)
        finally:
            if os.path.exists(socket_path):
                os.remove(socket_path)


if __name__ == "__main__":
    main()
