# PiRLC witness replay

The selected nonzero-running trace passes: Lean independently combines the
17 original witnesses and all 253,011,276 coefficients match Rust, including
the carrier tail. The checked kernel equals the selected `honestResponse`.
Changing the last tail coefficient rejects in its range. The current Rust
producer was rerun at `9cb553c4`; its private and public outputs match the
previously compared files byte-for-byte.

Lean receives the Rust PiCCS sum-check polynomials and final `Eval_K`/`Eval_A`
values. It checks C and derives the mixing challenges. Rust's combined witness
never enters the Lean calculation. Independent C generation remains a later
milestone. This is evidence for the tested inputs, not universal Rust equivalence.

`PIRLC_WITNESS_REPLAY.json` records source hashes, source-opening evidence,
range hashes, measured time/memory and local gates. `PIRLC_WITNESS_REPLAY_ASSET.json`
identifies the archive outside Git. Upload and fresh-download verification are
pending; no remote availability is claimed. The archive contains the original
inputs as regular files and the independently computed Lean ranges. It does
not require the earlier temporary-directory symlinks.

After obtaining the archive, check its SHA-256 against the asset record and
extract it under the existing ignored fixture directory
`crates/neo-fold-clean/tests/nifs/fixtures/stage1_terminal_replay/pirlc`.
Use this commit's normal Rust/Lean dependencies and materialized package LFS
object. From the repository root, set these shell variables:

```bash
nightstream_replay="$PWD/crates/neo-fold-clean/tests/nifs/fixtures/stage1_terminal_replay/pirlc"
nightstream_package="$PWD/formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"
nightstream_run="$(mktemp -d "$nightstream_replay/run.XXXXXX")"
```

Run each command separately and stop if it fails. The existing lean-graph
`pirlc-witness-values` gate registers this same sequence and its completion
checks. Build before execution so the execution cap measures the replay.

```bash
timeout 1500 python3 -B scripts/lean_graph/guard.py --kind lean --cwd formal/nightstream-fprime -- bash scripts/validate.sh build replayPiRLCWitness
timeout 300 python3 -B scripts/lean_graph/guard.py --kind rust --cwd . -- cargo build -p neo-fold-clean --release --features perf-timers --bin generate_pi_ccs_fixture
timeout 300 python3 -B scripts/lean_graph/guard.py --kind rust --cwd . -- cargo run -p neo-fold-clean --release --features perf-timers --bin generate_pi_ccs_fixture -- export-pirlc-replay "$nightstream_package" "$nightstream_replay/inputs/original-sources" "$nightstream_replay/inputs/saved-ccs.json" "$nightstream_run/capture"
timeout 300 cmp "$nightstream_run/capture/sources.jsonl" "$nightstream_replay/inputs/capture/sources.jsonl"
timeout 300 cmp "$nightstream_run/capture/ccs-input.json" "$nightstream_replay/inputs/capture/ccs-input.json"
timeout 300 python3 -B scripts/lean_graph/guard.py --kind rust --cwd . -- cargo run -p neo-fold-clean --release --features perf-timers --bin generate_pi_ccs_fixture -- prove-owned-rlc "$nightstream_package" "$nightstream_replay/inputs/original-sources" "$nightstream_replay/inputs/saved-ccs.json" "$nightstream_run/parent"
timeout 1500 python3 -B scripts/lean_graph/guard.py --kind lean --cwd formal/nightstream-fprime -- bash scripts/validate.sh pi-rlc-witness-replay "$nightstream_run/capture/ccs-input.json" "$nightstream_run/capture/sources.jsonl" "$nightstream_run/prefix.jsonl" 0 74272
timeout 1500 python3 -B scripts/lean_graph/guard.py --kind lean --cwd formal/nightstream-fprime -- bash scripts/validate.sh pi-rlc-witness-replay "$nightstream_run/capture/ccs-input.json" "$nightstream_run/capture/sources.jsonl" "$nightstream_run/suffix.jsonl" 74272 4685394
timeout 300 python3 -B scripts/lean_graph/guard.py --kind rust --cwd . -- cargo run -p neo-fold-clean --release --features perf-timers --bin generate_pi_ccs_fixture -- compare-pirlc-replay "$nightstream_run/parent/parent-witness.json" "$nightstream_run/prefix.jsonl" "$nightstream_run/suffix.jsonl"
```

The split at block 74,272 comes from the measured calibration run. It is not
a protocol parameter. The largest Lean range took 1,085 seconds at about
1.43 GiB peak RSS. The complete comparison and tail mutation took 20.1 seconds
after compilation. The fresh native R producer took 87.9 seconds, with
6,389,456 KiB peak RSS. The fixed profile remains `b = 2`, `k_rho = 16`.

The earlier source-opening check is reused only for the unchanged original
witness hashes recorded in the report. Source export alone does not verify
commitments. Protected-checker acceptance and external delivery remain separate
from these local passes.
