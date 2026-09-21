# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check server launch ports and cross-rank startup failure propagation on CPU."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_server_atom.sh"
)
SOURCE = SCRIPT.read_text()


class ServerStartupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atomesh startup ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = {
            "PATH": os.environ["PATH"],
            "ATOMESH_SCRIPT_DIR": str(SCRIPT.parent),
            "RUN_DIR": str(self.root),
            "RUNTIME_LOG_DIR": str(self.root / "logs"),
            "SLURM_JOB_ID": "42",
            "ATOMESH_RUN_TOKEN": "this-run",
            "IPADDRS": "10.19.0.1,10.19.0.2,10.19.0.3",
            "NODE0_ADDR": "10.19.0.1",
            "ROUTER_PORT": "8000",
        }

    def run_shell(self, body, expected_rc=0, **env):
        result = subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + body],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, expected_rc, result.stdout + result.stderr)
        return result

    def publish_failure(self, token="this-run", rc=7):
        (self.root / "rank-workload-2.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    job_id="42",
                    run_token=token,
                    rank=2,
                    num_ranks=3,
                    status="running",
                )
            )
        )
        (self.root / "rank-rc-2").write_text(f"{rc}\n")

    def wait_functions(self):
        return SOURCE[
            SOURCE.index("check_peer_failures()") : SOURCE.index("start_logged_process()")
        ]

    def test_peer_failure_aborts_before_health_request_or_sleep(self):
        self.publish_failure(rc=143)
        for call in (
            "wait_http http://decode:8020/health decode 1800",
            "wait_router_closed",
        ):
            with self.subTest(call=call):
                result = self.run_shell(
                    self.wait_functions()
                    + '\ncurl() { echo unexpected-curl >&2; return 99; }\n'
                    + 'sleep() { echo unexpected-sleep >&2; exit 99; }\n'
                    + call,
                    expected_rc=143,
                )
                self.assertIn("rank 2 exited rc=143", result.stderr)
                self.assertNotIn("unexpected-", result.stderr)

    def test_failure_published_during_wait_aborts_next_poll(self):
        self.publish_failure(rc=0)
        result = self.run_shell(
            self.wait_functions()
            + '\ncurl() { return 1; }\n'
            + 'sleep() { printf "7\\n" > "$RUN_DIR/rank-rc-2"; }\n'
            + 'wait_http http://decode:8020/health decode 1800',
            expected_rc=7,
        )
        self.assertIn("rank 2 exited rc=7", result.stderr)
        self.assertNotIn("not ready after", result.stderr)

    def test_stale_failure_does_not_abort_healthy_service(self):
        self.publish_failure(token="previous-run")
        result = self.run_shell(
            self.wait_functions()
            + '\ncurl() { return 0; }\n'
            + 'wait_http http://decode:8020/health decode 1800'
        )
        self.assertIn("[wait][OK] decode", result.stdout)

    def test_dead_local_worker_reports_worker_and_preserves_code(self):
        result = self.run_shell(
            self.wait_functions()
            + '\ncurl() { return 1; }\n'
            + '(exit 8) &\npid=$!\nwait "$pid" || true\n'
            + 'wait_http http://router:8000/health router 1800 "$pid"',
            expected_rc=8,
        )
        self.assertIn("local worker pid=", result.stderr)
        self.assertIn("while waiting for router rc=8", result.stderr)

    def test_launch_passes_planned_ports_for_all_layouts_and_phases(self):
        # Load the actual port defaults/offsets and launch functions; replace
        # only GPU execution and unrelated cache setup with CPU stubs.
        config = SOURCE[
            SOURCE.index("ATOMESH_PD_WORKER_LAYOUT=") : SOURCE.index("KV_CACHE_DTYPE=")
        ]
        launch = SOURCE[
            SOURCE.index("start_prefill()") : SOURCE.index("start_router()")
        ]
        stubs = r'''
apply_role_env() { :; }
reset_lmcache_disk() { :; }
build_server_cache_env() { :; }
dump_launch_info() { :; }
start_logged_process() {
  shift 2
  # Run env with a shell in place of the GPU server, retaining its assignments.
  local -a command=()
  while [[ "$1" != python3 ]]; do command+=("$1"); shift; done
  "${command[@]}" bash -c 'echo "PORTS=$ATOM_DP_MASTER_PORT/$ATOM_DP_BASE_PORT"'
}
server_common=() prefill_parallel=() decode_parallel=()
prefill_cudagraph_args=() decode_cudagraph_args=()
host_ip=127.0.0.1 host_name=test NODE_RANK=0 HIP_VISIBLE_DEVICES=0
MAX_NUM_SEQS=256 DECODE_MAX_NUM_SEQS=1024 DECODE_MAX_NUM_BATCHED_TOKENS=""
BENCH_MAX_CONCURRENCY=1024 ISL_LIST=8192 OSL=1024
PREFILL_KV_TRANSFER_CONFIG="" DECODE_KV_TRANSFER_CONFIG=""
PREFILL_SERVER_ARGS="" DECODE_SERVER_ARGS=""
start_prefill prefill-rank-0
start_decode decode-rank-2
# Co-located workers retain their per-worker overrides.
start_prefill prefill-worker-1 8011 6302 30100 30200
start_decode decode-worker-1 8021 6303 30300 30400
'''
        for layout in (
            "multi_node", "single_node", "prefill_single_node", "decode_single_node"
        ):
            for phase, offset in (("combined", 0), ("eval", 1000)):
                with self.subTest(layout=layout, phase=phase):
                    result = self.run_shell(
                        config + launch + stubs,
                        ATOMESH_PD_WORKER_LAYOUT=layout,
                        ATOMESH_EXECUTION_PHASE=phase,
                        ATOMESH_SERVICE_PORT_OFFSET=str(offset),
                    )
                    ports = [
                        line for line in result.stdout.splitlines()
                        if line.startswith("PORTS=")
                    ]
                    self.assertEqual(
                        ports,
                        [
                            f"PORTS={29500 + offset}/{29600 + offset}",
                            f"PORTS={29700 + offset}/{29800 + offset}",
                            "PORTS=30100/30200",
                            "PORTS=30300/30400",
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
