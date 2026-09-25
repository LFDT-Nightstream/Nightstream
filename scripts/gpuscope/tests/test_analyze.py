import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
import levers
import parse
import validate


N = 1_000_000


class AnalyzeTests(unittest.TestCase):
    def test_decompose_window_reconciles(self):
        intervals = {
            "kernel": analyze.interval_union([(0, 10 * N)]),
            "copy": analyze.interval_union([(10 * N, 15 * N)]),
            "memset": [],
            "device_busy": analyze.interval_union([(0, 10 * N)]),
            "device": analyze.interval_union([(0, 10 * N), (10 * N, 15 * N)]),
            "sync_api": analyze.interval_union([(15 * N, 18 * N)]),
            "other_api": analyze.interval_union([(18 * N, 20 * N)]),
        }

        row = analyze.decompose_window(0, 25 * N, intervals)
        bucket_sum = (
            row["gpu_busy_ms"]
            + row["transfer_wait_ms"]
            + row["sync_wait_ms"]
            + row["api_ms"]
            + row["host_gap_ms"]
            + row["unattributed_ms"]
        )

        self.assertAlmostEqual(row["wall_ms"], 25.0)
        self.assertAlmostEqual(row["gpu_busy_ms"], 10.0)
        self.assertAlmostEqual(row["transfer_wait_ms"], 5.0)
        self.assertAlmostEqual(row["sync_wait_ms"], 3.0)
        self.assertAlmostEqual(row["api_ms"], 2.0)
        self.assertAlmostEqual(row["host_gap_ms"], 5.0)
        self.assertAlmostEqual(row["wall_ms"], bucket_sum)

    def test_exclusive_segments_choose_innermost_range(self):
        phases = [
            {"label": "fold", "start": 0, "end": 10 * N, "duration": 10 * N},
            {"label": "fold.ingest", "start": 2 * N, "end": 5 * N, "duration": 3 * N},
        ]

        segments = analyze.exclusive_segments(phases)

        self.assertEqual(
            [(s["label"], s["start"] // N, s["end"] // N) for s in segments],
            [("fold", 0, 2), ("fold.ingest", 2, 5), ("fold", 5, 10)],
        )

    def test_terminal_fold_range_uses_last_ingest(self):
        phases = [
            {"label": "fold.ingest", "start": 10 * N, "end": 20 * N, "duration": 10 * N},
            {"label": "fold.ingest", "start": 30 * N, "end": 35 * N, "duration": 5 * N},
            {"label": "fold.superneo", "start": 35 * N, "end": 50 * N, "duration": 15 * N},
        ]

        self.assertEqual(analyze.terminal_fold_range(phases), (30 * N, 50 * N))

    def test_gpuprof_validation_accepts_matching_raw_facts(self):
        phases = [{"label": "fold.ingest", "source": "nvtx", "start": 0, "end": 2 * N, "duration": 2 * N}]
        report = {
            "stages": [
                {
                    "stage_id": "fold.ingest",
                    "kernel_attributed_ms": 1.5,
                    "launches": 3,
                    "h2d_mb": 4.0,
                    "h2d_copies": 2,
                    "api_total_ms": 0.25,
                    "api_calls": 5,
                }
            ]
        }
        oracle = {
            "nvtx_ranges": [{"stage": "fold.ingest", "wall_ms": 2.0}],
            "stages": {
                "fold.ingest": {
                    "gpu_ms": 1.5,
                    "launches": 3,
                    "h2d_mb": 4.0,
                    "h2d_copies": 2,
                    "api_ms": 0.25,
                    "api_calls": 5,
                }
            },
        }

        nvtx = validate.compare_nvtx_ranges(phases, oracle)
        stages = validate.compare_stage_attribution(report, oracle)

        self.assertTrue(nvtx["ok"])
        self.assertTrue(stages["ok"])

    def test_gpuprof_validation_reports_mismatch(self):
        phases = [{"label": "fold.ingest", "source": "nvtx", "start": 0, "end": 3 * N, "duration": 3 * N}]
        oracle = {"nvtx_ranges": [{"stage": "fold.ingest", "wall_ms": 2.0}]}

        result = validate.compare_nvtx_ranges(phases, oracle)

        self.assertFalse(result["ok"])
        self.assertEqual(result["mismatches"][0]["stage_id"], "fold.ingest")

    def test_gpuprof_validation_skips_repeat_stage_rollups(self):
        report = {"stages": [{"stage_id": "fold.ingest", "launches": 3}]}
        oracle = {
            "runs": [{"online": {}}, {"online": {}}],
            "stages": {"fold.ingest": {"launches": 9}},
        }

        result = validate.compare_stage_attribution(report, oracle)

        self.assertTrue(result["ok"])
        self.assertTrue(result["skipped"])

    def test_stderr_phase_alignment_and_chain_tagging(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = Path(tmp) / "stderr.txt"
            stderr.write_text(
                "\n".join(
                    [
                        "optimized_prove: 4. FE sumcheck            2.00ms @10000000",
                        "optimized_prove: 5. NC sumcheck            3.00ms @16000000",
                    ]
                ),
                encoding="utf-8",
            )
            nvtx = [{"text": "fold.ingest", "start": 12 * N, "end": 13 * N}]

            phases, counts = parse.build_phases(stderr, 0, nvtx)

        by_label = {phase["label"]: phase for phase in phases}
        self.assertEqual(by_label["fold.superneo.pi_ccs.sumcheck.fe"]["chain"], "cpu")
        self.assertEqual(by_label["fold.superneo.pi_ccs.sumcheck.nc"]["chain"], "gpu")
        self.assertEqual(by_label["fold.superneo.pi_ccs.sumcheck.nc"]["start"], 13 * N)
        self.assertEqual(counts["stderr_stamped_lines"], 2)

    def test_stderr_validation_compares_gpu_timer_windows(self):
        phases = [
            {
                "label": "fold.superneo.pi_ccs.sumcheck.fe",
                "source": "stderr",
                "chain": "gpu",
                "start": 0,
                "end": 2 * N,
            }
        ]
        oracle = {
            "phase_trace": [
                {
                    "stage": "fold.superneo.pi_ccs.sumcheck.fe",
                    "family": "ccs",
                    "chain": "gpu",
                    "synthetic": False,
                    "wall_ms": 2.0,
                }
            ]
        }

        result = validate.compare_stderr_phases(phases, oracle)

        self.assertTrue(result["ok"])

    def test_levers_rank_by_window_recoverable_time(self):
        stages = [
            {
                "stage_id": "fold.fast",
                "window_wall_ms": 10.0,
                "wall_ms": 4.0,
                "gpu_busy_ms": 8.0,
                "launches": 1,
                "host_gap_ms": 1.0,
                "sync_wait_ms": 0.0,
                "api_ms": 0.0,
                "transfer_wait_ms": 0.0,
                "unattributed_ms": 0.0,
                "top_kernels": [],
            },
            {
                "stage_id": "fold.slow",
                "window_wall_ms": 50.0,
                "wall_ms": 20.0,
                "gpu_busy_ms": 10.0,
                "launches": 2,
                "host_gap_ms": 30.0,
                "sync_wait_ms": 2.0,
                "api_ms": 1.0,
                "transfer_wait_ms": 0.0,
                "unattributed_ms": 0.0,
                "top_kernels": ["slow_kernel"],
            },
            {
                "stage_id": "finalize.terminal_fold",
                "window_wall_ms": 100.0,
                "wall_ms": 100.0,
                "gpu_busy_ms": 1.0,
                "launches": 0,
                "host_gap_ms": 99.0,
                "top_kernels": [],
            },
        ]

        report = levers.build_levers(stages, {"memcpys": []})

        self.assertEqual(report["levers"][0]["stage_id"], "fold.slow")
        self.assertEqual(report["levers"][0]["fix_class"], "device_fs_chain")
        self.assertNotIn("finalize.terminal_fold", [row["stage_id"] for row in report["levers"]])


if __name__ == "__main__":
    unittest.main()
