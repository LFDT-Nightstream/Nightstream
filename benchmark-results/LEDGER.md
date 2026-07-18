# CUDA Prover Performance Ledger

## 2026-07-03 — iteration 1: device transcript IO primitive

Hypothesis: the largest current lever is `fold.superneo.pi_ccs.sumcheck.fe`
host orchestration. Slice 2 needs device-driven FE/NC challenges, and the first
required primitive is a Poseidon2 transcript op stream that can absorb
kernel-written coeffs from a device buffer and write challenges into a device
buffer for fold kernels. This should not improve e2e by itself until the FE/NC
loops consume it, but it removes the main implementation blocker for the
~100ms FE recoverable bucket.

Orientation evidence:
- `benchmark-results/gpuprof-recheck-20260703/gpuprof.json`: CPU 2127.23ms,
  CUDA 882.03ms, 2.41x, byte-identical.
- Top lever: `pi_ccs.sumcheck.fe` wall 141.9ms, floor 35.6ms, recoverable
  106.3ms.

Expected outcome for this sub-iteration:
- `parity transcript` proves host/device transcript equality for mixed
  host-buffer and device-buffer ops.
- `parity quick` and `parity e2e_bench` remain byte-identical.
- Full e2e speed may remain within noise until the FE/NC loop rewrite lands.

Outcome:
- Accepted as slice-2 infrastructure, not promoted as a performance baseline.
- Implemented `p2_transcript_io_ops` plus `DeviceTranscript::run_io`.
- `parity transcript`: green; IO stream matched 300 host-returned and 300
  device-written challenges plus final state/cursor.
- `parity quick`: green.
- `parity e2e_bench`: green, byte-identical; direct print CPU 2302.01ms,
  CUDA 905.32ms, 2.54x.
- `gpuprof --repeat 3 --assert-residency`: CPU 2103.5ms, CUDA 888.9ms,
  2.36x, spread 0.3%, byte-identical, residency clean.
- `gpuprof check benchmark-results/gpuprof-recheck-20260703/gpuprof.json
  benchmark-results/gpuprof/20260703-transcript-io.json`: clean, no
  regressions.
- Not counted as a speed win because the FE/NC loops still use host-derived
  challenges; next hypothesis is to expose FE round coeff buffers/challenge
  buffers behind the backend seam and drive one FE row-round path from
  `DeviceTranscript::run_io`.

## 2026-07-03 — iteration 2: FE batch trace attempt rejected

Hypothesis: drive all FE row rounds from the device transcript, then bulk
download coeffs/challenges for host replay. This should remove the per-round
host challenge loop from `pi_ccs.sumcheck.fe`.

Outcome:
- Rejected and disabled as an active prover path.
- The first integration built, but `parity ccs_prove` failed with an ME output
  mismatch after the device-folded FE trace. That means the transcript replay
  was self-consistent but the device table state diverged from the canonical
  oracle state.
- The active backend hook now returns `None`, so the prover falls back to the
  verified per-round CUDA FE backend while keeping the transcript IO primitive
  available for a narrower follow-up gate.

Verification after disabling the active hook:
- `cargo fmt --all`: green, with the existing stable rustfmt
  `imports_granularity` warning.
- `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
- `parity ccs_prove`: green; full Pi_CCS proof identical.
- `parity quick`: green.
- `parity e2e_bench`: green, byte-identical; direct print CPU 2136.66ms,
  CUDA 906.64ms, 2.36x.
- `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
  byte-identical; median CPU 2123.1ms, CUDA 892.1ms, 2.38x, spread 1.5%.

Current gpuprof levers from
`benchmark-results/gpuprof/20260703-recheck-after-transcript-io/gpuprof.json`:
- `pi_ccs.sumcheck.fe`: 142.9ms wall, 35.1ms floor, 107.7ms recoverable.
- `pi_dec.open_children.forms`: 91.0ms wall, 27.9ms floor, 63.1ms recoverable.
- `pi_ccs.oracle.F`: 60.5ms wall, 3.1ms floor, 57.4ms recoverable.
- Kernel lint still flags `nc_col_partials` at about 17% theoretical occupancy
  and `fe_round_partials` at about 33%, with tail-round grid underfill.

## 2026-07-03 — iteration 3: isolate device challenge-fed FE fold

Hypothesis: the rejected FE batch trace diverged either in the new
device-buffer challenge fold primitive or in the higher-level batched FE trace
state machine. Add a narrow parity assertion to the existing `ccs_fe` gate:
folding the same K-table with a scalar host challenge and with the challenge
read from a device buffer must produce identical device output.

Expected outcome:
- No e2e speed change; this is a correctness isolator for slice 2.
- If the narrow fold check fails, fix `table_fold_from_challenge`.
- If it passes, the remaining bug is in the batched FE trace orchestration
  around coeff logging / transcript enqueue / CPU oracle advancement.

Outcome:
- Accepted as a correctness repair, not promoted as a performance baseline.
- The FE device-transcript path now reaches byte-identical proof output again
  after canonicalizing host/device Goldilocks field words at proof/download
  boundaries.
- Added structural Pi_CCS/NIFS parity diagnostics before serde byte assertions
  so future mismatches point at the proof component instead of only reporting
  opaque byte drift.
- `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
- `parity nifs`: green; 3 NIFS.P folds identical.
- `parity quick`: green.
- `parity e2e_bench`: green, byte-identical; direct print CPU 2124.80ms,
  CUDA 913.29ms, 2.33x.
- `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
  byte-identical; median CPU 2121.2ms, CUDA 902.5ms, 2.35x, spread 1.5%,
  residency clean.
- `scripts/gpuprof.sh ncu e2e_bench --top 2 --ncu-launch-count 1
  -- --assert-residency`: Nsight Systems pass green at CPU 2115.63ms,
  CUDA 895.13ms, 2.36x; Nsight Compute attempted `mat_vec_coeff_partials`
  and `nc_col_partials` but hardware counters are still blocked by
  `ERR_NVGPUCTRPERM`.

Current gpuprof levers from
`benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
- `pi_ccs.sumcheck.fe`: 155.7ms wall, 49.5ms floor, 106.2ms recoverable.
- `pi_dec.open_children.forms`: 89.9ms wall, 27.9ms floor, 61.9ms recoverable.
- `pi_ccs.oracle.F`: 59.2ms wall, 3.1ms floor, 56.1ms recoverable.
- `pi_rlc.combine_claims`: 20.0ms pure host recoverable.
- `pi_rlc.challenge_rhos`: 14.0ms pure host recoverable.
- Kernel lint still flags `nc_col_partials` at 17% theoretical occupancy and
  `fe_round_partials` at 33%, with tail-round grid underfill.

## 2026-07-03 — iteration 4: fuse FE final reduction and device transcript

Hypothesis: the active FE device-transcript path is correct but still pays
three tiny launches per row round after the FE partials block reduction:
`sum_partials`, `plane_copy` to the coeff log, and `p2_transcript_io_ops`.
Fuse final coeff reduction, coeff logging, and Poseidon2 challenge derivation
into one single-thread transcript kernel. This keeps the same transcript
prefix/coeff/challenge semantics and should reduce launch count in the top
`pi_ccs.sumcheck.fe` bucket without touching proof logic.

Expected outcome:
- Byte-identical `ccs_fe`, `quick`, and `e2e_bench`.
- Lower FE launch count and lower `pi_ccs.sumcheck.fe` recoverable time.
- If e2e median does not improve by at least 2%, reject or keep only if it is
  clearly neutral infrastructure for the next FE fusion step.

Outcome:
- Rejected and reverted.
- The candidate stayed byte-identical, but failed the stage regression gate.
- Candidate verification while active:
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
  - `parity ccs_fe`: green.
  - `parity quick`: green.
  - `parity e2e_bench`: green, byte-identical.
- Candidate gpuprof snapshot:
  `benchmark-results/gpuprof/20260703-fe-fused-round-challenge.json`.
  Median CPU 2119.4ms, CUDA 923.3ms, 2.29x, CUDA repeats
  `[919.6, 923.3, 930.5]`, spread 1.2%, residency clean.
- Regression check against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`
  failed:
  - `fold.superneo.pi_ccs`: 921.4ms -> 971.0ms.
  - `fold.superneo.pi_ccs.sumcheck`: 281.6ms -> 304.6ms.
  - `fold.superneo.pi_ccs.sumcheck.fe`: 155.7ms -> 176.7ms.
- Cause: FE launch count dropped 384 -> 256 and syncs dropped 100 -> 32,
  but the new single-thread `p2_sumcheck_round_challenge` kernel cost 36.7ms.
  It replaced `p2_transcript_io_ops` at 13.2ms plus `sum_partials` at 2.6ms,
  so the fused shape was slower even though it reduced orchestration noise.
- Revert verification:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
  - `parity ccs_fe`: green.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2131.75ms,
    CUDA 908.37ms, 2.35x.

Do not retry this exact FE single-thread fused reduction shape. The next
performance hypothesis should target an independent measured host island,
preferably `pi_rlc.challenge_rhos` + `pi_rlc.combine_claims` or
`pi_dec.open_children.forms`, before returning to FE kernel internals.

## 2026-07-03 — iteration 5: device χ_r for DEC forms

Hypothesis: `pi_dec.open_children.forms` spends most of its recoverable time
building `χ_r = tensor_point(parent.r)` on the host and uploading the full K
table before the existing CSR form kernels run. Generate `χ_r` on device from
the small challenge vector and feed the device table directly into
`forms_from_bar_csr`.

Expected outcome:
- Byte-identical `dec`, `quick`, and `e2e_bench`.
- Lower `pi_dec.open_children.forms` H2D bytes and host-gap time.
- If e2e median does not improve by at least 2%, reject unless this cleanly
  unlocks a larger resident DEC path.

Outcome:
- Accepted as a residency cleanup, not as a headline performance baseline.
- Implemented `tensor_point_k` in the CSR kernel module and added
  `DeviceBarMatrices::build_forms_from_challenges`, so DEC forms now build
  `χ_r` from the small challenge vector on device instead of uploading the
  full host-built K table.
- Verification:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
  - `parity dec`: green, children identical.
  - `parity quick`: green.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2119.05ms,
    CUDA 868.92ms, 2.44x.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2127.6ms, CUDA 896.3ms, 2.37x, spread 0.9%,
    residency clean.
- Regression check against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
  clean, 32 comparisons, 0 regressions.
- Local effect:
  - `pi_dec.open_children.forms` H2D dropped from about 4MB/fold group to 0.
  - `pi_dec.open_children.forms` wall improved from about 90.0ms to 84.7ms.
  - Online median did not materially improve versus the accepted 895.1ms
    baseline, so this is not enough for the 10x path by itself.

Next hypothesis should return to a large measured wall: either remove the FE
sumcheck host gaps without a single-thread fused transcript kernel, or attack
`pi_ccs.oracle.F` / DEC forms at a larger granularity where the whole row/form
pipeline stays resident instead of just moving `χ_r`.

## 2026-07-03 — iteration 6: pack FE transcript op metadata

Hypothesis: FE device-transcript rounds are correct but still upload one tiny
op buffer per row round. Keep the existing cheap `p2_transcript_io_ops` shape
and upload all FE round op triples once, then launch each round against a
window of that device-resident op stream. This targets the FE host/API gap
without retrying the rejected single-thread coeff-reduction fusion.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower FE H2D copy count / API calls, with no new slow transcript kernel.
- Accept only if the repeat-3 e2e median improves by at least 2%; otherwise
  revert as a too-small orchestration cleanup.

Outcome:
- Rejected and reverted.
- The candidate stayed byte-identical and passed residency/check gates, but it
  did not meet the performance bar.
- Candidate verification while active:
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2123.37ms,
    CUDA 909.12ms, 2.34x.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2124.2ms, CUDA 899.6ms, 2.36x, CUDA repeats
    `[894.7, 899.9, 899.6]`, spread 0.6%, residency clean.
- Regression check against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
  clean, 32 comparisons, 0 regressions, but the median moved from 895.1ms to
  899.6ms instead of improving.
- Local effect:
  - FE H2D copy count dropped 88 -> 28.
  - FE sync count dropped about 100 -> 40.
  - The old `p2_transcript_io_ops` bucket was replaced by
    `p2_transcript_io_ops_at`, still about 13ms.
  - `pi_ccs.sumcheck.fe` stayed the top lever at about 155ms wall and about
    106ms recoverable.
- Cause: this removed tiny metadata uploads, not the dominant FE wall. The
  remaining cost is the round-loop shape itself: many small launches/API gaps
  plus per-round transcript/fold sequencing. Packing op metadata is below the
  noise floor for e2e.
- Revert verification:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
  - `parity quick`: green.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2134.29ms,
    CUDA 911.61ms, 2.34x.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2129.6ms, CUDA 896.5ms, 2.38x, CUDA repeats
    `[895.1, 896.5, 897.4]`, spread 0.3%, residency clean.
  - `gpuprof.py check` against
    `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
    clean, online CUDA 895.1ms -> 896.5ms, 32 comparisons, 0 regressions.

Do not retry FE op-metadata packing by itself. The next FE attempt needs to
remove a real round boundary or move a larger sumcheck slice device-resident;
otherwise use the levers table to attack a different host island.

## 2026-07-03 — iteration 7: use rotation-ring RLC combine for small claim matrices

Hypothesis: `fold.superneo.pi_rlc.combine_claims` spends about 20ms in pure
host algebra. `rlc_combine_claims` uses the rotation-ring multiplication path
only when the mixed matrix has at least 256 columns; the public `X` claim
matrix has few columns, so it falls back to a generic D×D multiply despite
ρ being a strict rotation matrix. Lowering that threshold should make the
claim-side mix use the same value-identical ring path as large witnesses.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `pi_rlc.combine_claims` wall time.
- Accept only if repeat-3 e2e median improves by at least 2%; otherwise
  revert the threshold change.

Outcome:
- Rejected and reverted.
- The candidate stayed byte-identical and residency-clean, but it did not
  improve the online CUDA median or the targeted `pi_rlc.combine_claims`
  bucket.
- Candidate verification while active:
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2110.89ms,
    CUDA 918.59ms, 2.30x.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2105.0ms, CUDA 896.5ms, 2.34x, CUDA repeats
    `[896.5, 902.4, 895.1]`, spread 0.8%, residency clean.
- Regression check against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
  clean, 32 comparisons, 0 regressions, but the online CUDA median moved from
  895.1ms to 896.5ms instead of improving.
- Local effect:
  - `fold.superneo.pi_rlc.combine_claims` stayed about 20.0ms.
  - `fold.superneo.pi_rlc.challenge_rhos` stayed about 13.9ms.
- Cause: the 20ms RLC bucket is not solved by the small-matrix
  left-multiplication threshold. It is likely dominated by surrounding
  host-side claim digest/algebra, commitment mix/validation, taxonomy
  attribution, or y-ring/y-zcol scalar work that this threshold does not touch.
- Revert verification:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
  - `parity quick`: green.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2130.50ms,
    CUDA 912.57ms, 2.33x.

Do not retry threshold-only RLC small-ring specialization. The next useful
attempt should hit a larger measured wall: real device-driven FE/NC sumcheck
rounds, `pi_ccs.oracle.F`, or `pi_dec.open_children.forms`.

## 2026-07-03 — iteration 8: device-transcript NC column-round trace

Hypothesis: FE already has a device-transcript row-round trace path, but NC
column rounds still use the per-round host path: coeffs D2H, host transcript
challenge, then host-scalar device fold. Add the same bulk trace shape for NC:
round coeff kernels write to device, Poseidon2 absorbs them on device, the
challenge stays in device memory for the fold kernels, and coeffs/challenges
download once after the column phase for host transcript replay.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `pi_ccs.sumcheck.nc` D2H/sync/API overhead and fewer host-fed column
  folds.
- Accept only if repeat-3 e2e median improves by at least 2%; otherwise
  revert the NC trace wiring.

Outcome:
- Rejected and reverted.
- The candidate stayed byte-identical, but it made the NC phase and e2e median
  worse. It reduced NC D2H copies, but paid that back with many more launches,
  tiny H2D/op-buffer uploads, transcript kernels, memsets, and syncs.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2106.76ms,
    CUDA 920.79ms, 2.29x.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2107.2ms, CUDA 911.5ms, 2.32x, CUDA repeats
    `[915.1, 907.8, 911.5]`, spread 0.8%, residency clean.
- Regression check against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
  failed. Online CUDA moved 895.1ms -> 911.5ms, and
  `fold.superneo.pi_ccs.sumcheck.nc` regressed 126.0ms -> 137.8ms.
- Local effect from the failed candidate:
  - NC launches increased 384 -> 536.
  - H2D copies increased 8 -> 92.
  - D2H copies decreased 80 -> 12.
  - Memsets increased 20 -> 32.
  - Syncs increased 92 -> 112.
  - New `p2_transcript_io_ops` cost was about 9.7ms.
  - New `nc_fold_strided_from_challenge` cost was about 2.4ms.
  - `nc_col_partials` stayed about the same, 106.0ms -> 107.8ms.
- Cause: this FE-style per-round device-transcript trace is the wrong shape
  for NC. The NC wall is already mostly kernel busy (`nc_col_partials`), not
  host D2H. Moving tiny round challenges through device transcript IO ops
  removes the smaller cost and adds more launch/API traffic.
- Revert verification:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2317.29ms,
    CUDA 917.25ms, 2.53x print noise on CPU.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2122.0ms, CUDA 894.4ms, 2.37x, CUDA repeats
    `[894.2, 894.4, 902.9]`, spread 1.0%, residency clean.
  - `gpuprof.py check` against
    `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
    clean, online CUDA 895.1ms -> 894.4ms, 32 comparisons, 0 regressions.

Do not retry per-round NC transcript IO-op tracing. Future NC work should be
kernel-internal, such as reducing `nc_col_partials` register pressure or
batching/fusing tail rounds. The next broader performance attempt should target
larger recoverable host islands: `pi_ccs.sumcheck.fe`, `pi_ccs.oracle.F`, or
`pi_dec.open_children.forms`.

## 2026-07-03 — iteration 9: contiguous device f-var row tables

Hypothesis: `fold.superneo.pi_ccs.oracle.F` spends most of its wall in host/API
gaps around per-f-var row-table construction. Today each f-var table gets its
own zeroed device buffer, then `DeviceFeOracle` copies those tables into the FE
table arena one at a time. Build all f-var row tables for one MCS into one
contiguous device buffer and copy that block into the FE table arena once.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `pi_ccs.oracle.F` memset count/API wall and lower
  `pi_ccs.oracle.upload` launch count.
- Accept only if repeat-3 e2e median improves by at least 2%; otherwise
  revert this row-table layout change.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity and improved the local counters it
  targeted, but it did not improve the canonical e2e median.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2106.62ms,
    CUDA 910.80ms, 2.31x.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2116.3ms, CUDA 898.4ms, 2.35x, CUDA repeats
    `[898.4, 900.0, 893.3]`, spread 0.7%, residency clean.
- Regression check against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
  clean, online CUDA 895.1ms -> 898.4ms, 32 comparisons, 0 regressions, but
  no improvement.
- Local effect:
  - `fold.superneo.pi_ccs.oracle.F` memsets dropped 32 -> 4.
  - `fold.superneo.pi_ccs.oracle.upload` launches dropped 42 -> 14.
  - `fold.superneo.pi_ccs.oracle.F` wall only moved 59.6ms -> 58.8ms.
  - Online CUDA moved the wrong way, 895.1ms -> 898.4ms.
- Cause: per-table zero/copy cleanup removes some bookkeeping, but the
  measured `oracle.F` wall is still dominated by host gaps around the
  row-table build path, not by the number of table buffers or D2D copies.
- Revert verification:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2128.59ms,
    CUDA 912.72ms, 2.33x.

Do not retry f-var row-table buffer packing by itself. The remaining large
levers still require a bigger architectural move: FE round-loop fusion/device
command shape, DEC form generation residency/fusion, or true kernel-internal
work on the low-occupancy FE/NC kernels.

## 2026-07-03 — iteration 10: uninitialized overwritten form buffers

Hypothesis: `fold.superneo.pi_dec.open_children.forms` and the FE Ajtai
`y_prime` path allocate large zeroed form buffers even though
`forms_from_bar_csr` writes every output word before the buffer is consumed.
Use cuda-oxide's `DeviceBuffer::uninitialized_async` for those fully
overwritten buffers, and for the χ table generated by `tensor_point_k`, to
remove the large form-buffer memsets without changing any arithmetic or proof
surface.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `pi_dec.open_children.forms` memset MB/count and API time.
- Accept only if repeat-3 e2e median improves by at least 2%; otherwise
  revert the uninitialized allocation change.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity and removed the targeted form-buffer
  memset surface, but online CUDA moved the wrong way and failed the >=2%
  improvement bar.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2105.57ms,
    CUDA 916.44ms, 2.30x.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2115.3ms, CUDA 904.8ms, 2.33x, CUDA repeats
    `[907.6, 904.8, 900.3]`, spread 0.8%, residency clean.
- Regression check against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
  clean under the tolerance gate, online CUDA 895.1ms -> 904.8ms, 32
  comparisons, 0 hard regressions, but no improvement.
- Local effect:
  - `fold.superneo.pi_dec.open_children.forms` reported no memset traffic.
  - `fold.superneo.pi_dec.open_children.forms` wall was still 86.1ms.
  - Top levers remained `pi_ccs.sumcheck.fe` 107.6ms recoverable,
    `pi_dec.open_children.forms` 58.4ms recoverable, and `pi_ccs.oracle.F`
    56.4ms recoverable.
- Cause: the zero-init cost is not the limiting factor for this path. The
  remaining forms wall is mostly host/API gap around `forms_from_bar_csr`, so
  removing memsets alone does not improve the serial online wall.

Do not retry overwritten-buffer uninitialized allocation as a standalone
optimization. Future work on `open_children.forms` should fuse/reshape the
forms pipeline or move more of the surrounding host preparation into one
device-owned command path.

## 2026-07-03 — iteration 11: FE coeff reduction writes trace log

Hypothesis: `fold.superneo.pi_ccs.sumcheck.fe` still launches a tiny
device-to-device copy once per row round only to append `coeffs_out` into the
bulk coeff trace used for host transcript replay. Write the final reduced
round coefficients directly into the trace log from the final reduction kernel
instead, removing one launch per FE round while preserving the exact same
coeff bytes, transcript replay, challenges, and folded tables.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `pi_ccs.sumcheck.fe` launch count and API calls.
- No change to FE arithmetic, transcript domain separation, or host replay.
- Accept only if repeat-3 e2e median improves by at least 2%; otherwise
  revert this launch-shape change.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity and removed exactly the intended
  launch surface, but the e2e median still moved the wrong way and failed the
  >=2% improvement bar.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2112.33ms,
    CUDA 907.93ms, 2.33x.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2125.9ms, CUDA 900.5ms, 2.35x, CUDA repeats
    `[900.5, 905.4, 899.7]`, spread 0.6%, residency clean.
- Regression check against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
  clean under tolerance, online CUDA 895.1ms -> 900.5ms, 32 comparisons, 0
  hard regressions, but no improvement.
- Local effect:
  - `fold.superneo.pi_ccs.sumcheck.fe` launches dropped 384 -> 320.
  - `fold.superneo.pi_ccs.sumcheck` launches dropped 768 -> 704.
  - `fold.superneo.pi_ccs.sumcheck.fe` wall stayed flat, 155.7ms -> 156.6ms.
  - Online CUDA moved the wrong way, 895.1ms -> 900.5ms.
- Cause: deleting one tiny D2D logging launch per FE round is too small. The
  FE wall remains dominated by the broader device-transcript command shape,
  final D2H wait/replay, and host gaps, not by the proof-log copy launch.

Do not retry FE coeff-log copy elimination by itself. Future FE work must fuse
larger round phases or attack the low-occupancy `fe_round_partials` kernel
itself; reducing isolated bookkeeping launches is not enough.

## 2026-07-03 — iteration 12: skip RingMatVecScratch reuse memsets

Hypothesis: `RingMatVecScratch` zeroes its large reused `partials` and `sums`
buffers before every Ajtai/DEC/y-eval ring mat-vec, but the two producer
kernels fully overwrite the exact regions consumed by later stages:
`mat_vec_coeff_partials` writes every current `(group, chunk, coeff)` partial,
and `mat_vec_sum_chunks` writes every current `(group, coeff)` sum. Skipping
the reuse memset should remove a large measured memset surface without changing
any arithmetic or proof bytes.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower fold-level memset MB/count, especially in Ajtai commit/y-ring/y-prime
  paths.
- No change to kernel arithmetic or output layout.
- Accept only if repeat-3 e2e median improves by at least 2%; otherwise
  revert the scratch reuse change.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity, but the improvement was only ~0.2%,
  below the required >=2% acceptance bar.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2132.48ms,
    CUDA 909.51ms, 2.34x.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2121.0ms, CUDA 893.3ms, 2.37x, CUDA repeats
    `[901.2, 893.2, 893.3]`, spread 0.9%, residency clean.
- Regression check against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
  clean under tolerance, online CUDA 895.1ms -> 893.3ms, 32 comparisons, 0
  hard regressions, but no material improvement.
- Local effect:
  - Fold-level memset dropped from the candidate path's previous large surface
    only partially; `fold` still reported 1632MB/156 memsets, with
    `pi_ccs.oracle`, `pi_ccs.output.y_prime`, `pi_dec.open_children.forms`,
    and split/emit buffers still dominating.
  - Top levers were unchanged: `pi_ccs.sumcheck.fe` 106.4ms recoverable,
    `pi_dec.open_children.forms` 57.2ms recoverable, and `pi_ccs.oracle.F`
    56.4ms recoverable.
- Cause: even where ring scratch reuse memset is safe to skip, it is not the
  serial bottleneck. The remaining wall is dominated by FE host gaps,
  forms/oracle host/API prep, and kernel-bound NC/y-ring work.

Do not retry RingMatVecScratch reuse-memset removal by itself. Future memset
work must be paired with a larger pipeline change that removes the surrounding
host/API phase wall.

## 2026-07-03 — iteration 13: specialized FE transcript round kernel

Hypothesis: the FE device-transcript path already keeps row-round challenges
on device, but it still prebuilds and uploads one tiny generic IO op-buffer
per row round, then launches the generic `p2_transcript_io_ops` interpreter.
Replace that in the FE loop with a specialized Poseidon2 kernel that absorbs
the fixed sumcheck prefix, absorbs the just-produced round coefficients from
device memory, and writes the two challenge words directly into the device
challenge buffer. This should remove FE's 88 tiny H2D copies and reduce
generic transcript interpreter overhead without changing transcript semantics.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `fold.superneo.pi_ccs.sumcheck.fe` H2D copy count and API surface.
- No change to FE coefficient arithmetic, proof-log bytes, host replay, or
  challenge values.
- Accept only if repeat-3 e2e median improves by at least 2%; otherwise revert
  the specialized transcript kernel.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity and removed the intended tiny H2D
  surface, but it worsened the repeat-3 online median.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2116.17ms,
    CUDA 909.56ms, 2.33x.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2109.7ms, CUDA 902.0ms, 2.35x, CUDA repeats
    `[902.2, 902.0, 898.1]`, spread 0.5%, residency clean.
- Regression check against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
  clean under tolerance, online CUDA 895.1ms -> 902.0ms, 32 comparisons, 0
  hard regressions, but slower and below the improvement bar.
- Local effect:
  - `fold.superneo.pi_ccs.sumcheck.fe` H2D copies dropped 88 -> 20.
  - `fold.superneo.pi_ccs.sumcheck.fe` API calls dropped 836 -> 552.
  - `p2_transcript_io_ops` became `p2_transcript_sumcheck_round`, but total
    transcript kernel time stayed about 13ms.
  - `fold.superneo.pi_ccs.sumcheck.fe` wall stayed flat, 155.7ms -> 156.1ms.
  - Online CUDA worsened, 895.1ms -> 902.0ms.
- Cause: FE's top-line wall is not dominated by the tiny op-buffer upload or
  generic interpreter dispatch. The same number of serialized round kernels
  remains, and the single-thread Poseidon2 challenge kernel cost is unchanged.

Do not retry FE transcript-op specialization by itself. Future FE work needs a
larger fusion or batching step that reduces serialized round launches, or a
kernel-level change to `fe_round_partials`; removing tiny op-buffer copies is
not enough.

## 2026-07-03 — iteration 14: flattened DEC forms launch

Hypothesis: `fold.superneo.pi_dec.open_children.forms` is the highest remaining
non-FE structural lever. The current path keeps the static bar matrices
resident, but it still launches one `forms_from_bar_csr` kernel per SuperNeo
matrix each fold. Flatten the static bar CSR upload once into one device
surface, then build all ring-linear-form rows with one multi-matrix kernel.
This should reduce forms host/API gaps without changing the chi table, matrix
entries, or output layout consumed by DEC.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `fold.superneo.pi_dec.open_children.forms` launch/API surface.
- Accept only if repeat-3 e2e median improves by at least 2% against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`;
  otherwise revert the flattening.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity and reduced the forms launch count, but
  it made the online CUDA median materially worse and failed the gpuprof check.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2109.84ms,
    CUDA 943.82ms, 2.24x.
  - `scripts/gpuprof.sh quick e2e_bench --repeat 3 --assert-residency`: green,
    byte-identical; median CPU 2112.0ms, CUDA 936.4ms, 2.26x, CUDA repeats
    `[936.4, 933.4, 938.2]`, spread 0.5%, residency clean.
- Regression check against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`:
  failed. Online CUDA moved 895.1ms -> 936.4ms. `pi_dec.open_children.forms`
  wall moved 89.9ms -> 114.1ms even though launches dropped 32 -> 8.
- Local effect:
  - `forms_from_bar_csr` 27.6ms disappeared, but the new
    `all_forms_from_bar_csr` cost 22.4ms.
  - Forms launch count improved, but host idle stayed large (`0/1/77`) and
    the fused kernel's average duration was worse.
  - `pi_dec` wall regressed 561.6ms -> 619.2ms.
- Cause: flattening all matrices into one kernel reduced launch count but lost
  useful per-matrix scheduling/shape behavior and did not attack the dominant
  forms host gap. Kernel fusion at this granularity is not enough.

Do not retry one-launch flattened bar-CSR forms by itself. Future forms work
must either move the surrounding host preparation/state ownership onto the
device path or change the arithmetic kernel enough to reduce kernel time, not
only launch count.

## 2026-07-03 — iteration 15: typed Π_RLC claim combine

Hypothesis: `fold.superneo.pi_rlc.challenge_rhos` + `combine_claims` is a
measured pure-host island of about 34ms. The CUDA adapter derives typed
`RotRho` values, then immediately clones them into full matrices and sends
those matrices back through generic rotation-matrix validation before the claim
algebra runs. Add a narrow typed-rho combine entrypoint that consumes
`RotRho::as_mat()` directly for claim algebra, while keeping existing matrix
materialization for the commitment mixer and device witness mix.

Expected outcome:
- Byte-identical `rlc_bench`, `quick`, and `e2e_bench`.
- Lower `pi_rlc.combine_claims` host wall without touching transcript
  derivation, commitment mixing, witness mixing, or proof output.
- Accept only if repeat-3 e2e median improves by at least 2% against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`;
  otherwise revert this helper/wiring as too small.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity and passed the regression/residency
  gates, but did not improve the e2e median.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity rlc_bench`: green, byte-identical; per-prove CPU 90.29ms,
    GPU 54.36ms.
  - `parity quick`: green.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2109.11ms,
    CUDA 908.07ms, 2.32x.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json
    benchmark-results/gpuprof-iteration15-typed-rlc-rejected.json`: green,
    byte-identical; median CPU 2113.4ms, CUDA 896.5ms, 2.35x, CUDA repeats
    `[896.5, 896.3, 901.5]`, spread 0.6%, residency clean.
  - `gpuprof.py check
    benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json
    benchmark-results/gpuprof-iteration15-typed-rlc-rejected.json`: clean,
    online CUDA 895.1ms -> 896.5ms, 32 comparisons, 0 regressions.
- Local effect:
  - `pi_rlc.combine_claims` remained about 19.8ms.
  - `pi_rlc.challenge_rhos` remained about 14.0ms.
  - The candidate avoided duplicate rho validation in one claim-combine helper,
    but still had to materialize rho matrices for the commitment mixer and
    device witness mix.
- Cause: the RLC host island is dominated by actual rho derivation and claim
  algebra, not by the small revalidation boundary. A useful RLC iteration needs
  to move rho derivation/claim algebra to the device transcript path or
  overlap/remove the host island, not just bypass typed-rho revalidation.

Do not retry typed-rho claim combine as a standalone optimization.

## 2026-07-03 — iteration 16: narrower NC column chunks

Hypothesis: `nc_col_partials` is the largest measured kernel-side lead:
about 108ms busy in the accepted baseline, 17% theoretical occupancy, and
repeated grid-underfill warnings for late rounds. Halve
`NC_CHUNK_PAIRS` from 8 to 4 so each NC column round exposes twice as many
independent pair groups to CUDA. This should reduce underfilled
`nc_col_partials` duration if the extra `sum_partials` reduction work stays
below the occupancy gain.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `nc_col_partials` total kernel time and lower
  `fold.superneo.pi_ccs.sumcheck.nc` wall.
- Accept only if repeat-3 e2e median improves by at least 2% against
  `benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json`;
  otherwise revert the chunk-size change.

Outcome:
- Accepted.
- Change: `NC_CHUNK_PAIRS` reduced from 8 to 4 in
  `crates/neo-prover-cuda/src/kernels/pi_ccs_nc.rs`.
- Verification:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity ccs_nc`: green, byte-identical.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2120.14ms,
    CUDA 870.32ms, 2.44x.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json benchmark-results/gpuprof/20260703-nc-chunk4.json`:
    green, byte-identical; median CPU 2116.6ms, CUDA 858.0ms, 2.47x, CUDA
    repeats `[858.0, 855.3, 858.8]`, spread 0.4%, residency clean.
  - `gpuprof.py check
    benchmark-results/gpuprof-e2e_bench-ncu-20260703T070503Z/gpuprof.json
    benchmark-results/gpuprof/20260703-nc-chunk4.json`: clean, online CUDA
    895.1ms -> 858.0ms, 32 comparisons, 0 regressions.
- Local effect:
  - `fold.superneo.pi_ccs.sumcheck.nc` improved from about 114.7ms in the
    accepted baseline to 85.1ms.
  - `nc_col_partials` improved from about 108.3ms in the accepted baseline to
    61.0ms.
  - `sum_partials_blocks` became the next visible NC-side cost at about 5.8ms,
    so smaller chunks paid some extra reduction overhead but the occupancy win
    dominated.

This is the current accepted baseline:
`benchmark-results/gpuprof/20260703-nc-chunk4.json`.

## 2026-07-03 — iteration 17: narrower FE row chunks

Hypothesis: after the accepted NC chunking win, the current top lever is
`fold.superneo.pi_ccs.sumcheck.fe` at 155.5ms wall with about 106.1ms
recoverable. The kernel lint flags `fe_round_partials` as register-limited
to about 33% theoretical occupancy and repeatedly grid-underfilled at the
tail rounds. Halve `EVAL_CHUNK_PAIRS` from 16 to 8 so each FE row round
exposes twice as many independent row-pair groups to CUDA. This mirrors the
accepted NC change while staying inside the same proof semantics.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `fe_round_partials` total kernel time and lower
  `fold.superneo.pi_ccs.sumcheck.fe` wall.
- Accept only if repeat-3 e2e median improves by at least 2% against the
  current accepted baseline
  `benchmark-results/gpuprof/20260703-nc-chunk4.json`; otherwise revert the
  chunk-size change.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity and improved the FE kernel locally,
  but the e2e median improvement was below the 2% acceptance bar.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2122.35ms,
    CUDA 856.00ms, 2.48x.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json benchmark-results/gpuprof/20260703-fe-chunk8.json`:
    green, byte-identical; median CPU 2110.5ms, CUDA 845.0ms, 2.50x, CUDA
    repeats `[843.7, 845.1, 845.0]`, spread 0.2%, residency clean.
  - `gpuprof.py check benchmark-results/gpuprof/20260703-nc-chunk4.json
    benchmark-results/gpuprof/20260703-fe-chunk8.json`: clean, online CUDA
    858.0ms -> 845.0ms, 33 comparisons, 0 regressions.
- Local effect:
  - `fold.superneo.pi_ccs.sumcheck.fe` improved from 155.5ms to 142.5ms.
  - `fe_round_partials` improved from about 31.3ms to 17.0ms.
  - `p2_transcript_io_ops` stayed about 13ms and host/API gaps still dominated
    the FE stage, so the local kernel win did not produce enough e2e gain.

Do not retry `EVAL_CHUNK_PAIRS = 8` as a standalone optimization. A useful FE
iteration now needs to remove round-loop orchestration work, batch/fuse the
transcript/fold path, or combine this chunking with a larger FE-stage change
that clears the e2e threshold.

## 2026-07-03 — iteration 18: ring-structured Π_RLC claim vectors

Hypothesis: `fold.superneo.pi_rlc.combine_claims` remains a pure-host island
of about 20.2ms in the current accepted baseline. The current claim combine
uses the full 54×54 rotation matrix for `y_ring` and `y_zcol`, even though
the same ρ is a rotation/ring scalar and the witness path already uses the
ring product form. Replace those two vector products with a bit-identical
Φ₈₁ ring multiplication from the rho first column. This reduces arithmetic
inside the CUDA path's host island without changing transcript derivation,
commitment mixing, witness mixing, or proof/public data.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `fold.superneo.pi_rlc.combine_claims` wall.
- Accept only if repeat-3 e2e median improves by at least 2% against the
  current accepted baseline
  `benchmark-results/gpuprof/20260703-nc-chunk4.json`; otherwise revert the
  helper/wiring.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity, but it did not clear the 2% acceptance
  rule and was slightly slower than the current accepted baseline.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2139.16ms,
    CUDA 875.75ms, 2.44x.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json benchmark-results/gpuprof/20260703-rlc-ring-vectors.json`:
    green, byte-identical; median CPU 2119.2ms, CUDA 861.9ms, 2.46x, CUDA
    repeats `[862.6, 861.9, 856.5]`, spread 0.7%, residency clean.
  - `gpuprof.py check benchmark-results/gpuprof/20260703-nc-chunk4.json
    benchmark-results/gpuprof/20260703-rlc-ring-vectors.json`: clean, online
    CUDA 858.0ms -> 861.9ms, 33 comparisons, 0 regressions.
- Local effect:
  - `fold.superneo.pi_rlc.combine_claims` stayed about 20.4ms, so the ring-form
    helper did not remove the visible host island.
  - The useful conclusion is negative: this stage needs to move ρ derivation
    and claim algebra onto the device transcript path, not just rewrite the
    local CPU vector multiply.

Current accepted baseline remains:
`benchmark-results/gpuprof/20260703-nc-chunk4.json`.

## 2026-07-03 — iteration 22: narrower FE row chunks, second cut

Hypothesis: FE remains the top recoverable lever at about 155.8ms wall, and
the lint still flags `fe_round_partials` as register-limited / underfilled.
The previous `EVAL_CHUNK_PAIRS = 8` standalone cut improved
`fe_round_partials` and FE wall locally but missed the 2% e2e bar. Try
`EVAL_CHUNK_PAIRS = 4` so late FE rounds expose more independent groups. This
preserves the same round polynomial coefficients and transcript order; only
the partial partition changes.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `fe_round_partials` and possibly
  `fold.superneo.pi_ccs.sumcheck.fe`, with some extra reduction overhead.
- Accept only if repeat-3 e2e median improves by at least 2% against
  `benchmark-results/gpuprof/20260703-nc-chunk2.json`; otherwise restore
  `EVAL_CHUNK_PAIRS = 16`.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity and improved the repeat-3 CUDA median,
  but missed the campaign's >=2% acceptance rule.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2110.00ms,
    CUDA 839.35ms, 2.51x.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json benchmark-results/gpuprof/20260703-fe-chunk4.json`:
    green, byte-identical; median CPU 2111.7ms, CUDA 825.9ms, 2.56x, CUDA
    repeats `[828.0, 825.9, 819.0]`, spread 1.1%, residency clean.
  - `gpuprof.py check benchmark-results/gpuprof/20260703-nc-chunk2.json
    benchmark-results/gpuprof/20260703-fe-chunk4.json`: clean, online CUDA
    840.0ms -> 825.9ms, 32 comparisons, 0 regressions.
- Local effect:
  - `fold.superneo.pi_ccs.sumcheck.fe` improved from about 155.8ms to
    135.4ms.
  - `fe_round_partials` dropped to about 10ms in the profiled run, but FE
    still had about 107.6ms recoverable and the full e2e win was only about
    1.7%.
  - The useful conclusion is negative for standalone chunk tuning: the FE
    chunk size is a local improvement but not enough by itself. It should only
    return as part of a deeper FE loop change that removes host gaps or
    round-boundary launch pressure.

Current accepted baseline remains:
`benchmark-results/gpuprof/20260703-nc-chunk2.json`.

## 2026-07-03 — iteration 26: device-owned Π_RLC rho sampling

Hypothesis: after moving FE/NC transcript pieces to the device, the next
protocol migration target is Π_RLC. Before this slice, Π_RLC sampled ρ on the
host, copied host-derived ρ coefficients into the device witness mixer, and
kept only `Z_mix` on GPU. That is still a CPU-owned protocol island even though
the bulk mix kernel is device-side.

Change:
- Expose `pi_rlc::begin_rho_sampling`, which binds the input CE-claim digest
  on the canonical host transcript and returns the exact sampling-start
  snapshot.
- Add a Poseidon2 CUDA rho sampler for the production Goldilocks profile
  (`D = 54`, alphabet `[-2, -1, 0, 1, 2]`).
- Have `CudaNifsProver` sample Π_RLC rho coefficients with the device
  transcript, restore the host transcript from the device final snapshot, use
  the device rho buffer directly for `Z_mix`, and download only the tiny
  coefficient list needed to assemble today's CPU-shaped `CeClaim`.
- Keep final proof semantics unchanged and byte-identical.

Verification:
- `cargo fmt --all`: green, with the existing stable rustfmt warning.
- `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
- `./target/release/parity quick`: green, byte-identical before the later
  non-cuda cfg cleanup.
- `./target/release/parity e2e_bench`: green, byte-identical; direct print
  CPU 2137.84ms, CUDA 875.09ms, 2.44x.
- `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
  --assert-residency --json
  benchmark-results/gpuprof/20260703-device-rlc-rhos-r3.json`: green,
  byte-identical; median CPU 2125.5ms, CUDA 871.2ms, speedup 2.44x, CUDA
  repeats `[871.2, 866.5, 872.5]`, residency clean.
- `cargo check --workspace --release`: initially exposed a pre-existing
  non-cuda cfg issue in `ring_layout`; fixed locally; rerun green.
- Final `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
  green.
- Final `./target/release/parity rlc`: green, byte-identical.

Outcome:
- Accepted as architecture migration only.
- Not accepted as the performance baseline.
- `gpuprof check benchmark-results/gpuprof/20260703-nc-chunk2.json
  benchmark-results/gpuprof/20260703-device-rlc-rhos-r3.json` failed stage
  regression gates even though overall online CUDA time stayed within the
  loose limit:
  - `fold.superneo.pi_rlc` regressed from 41.3ms to 61.6ms.
  - `fold.superneo.pi_rlc.challenge_rhos` regressed from 14.1ms to 24.2ms.
  - New `p2_transcript_sample_rlc_rhos` cost is about 9.5ms across 4 folds.
  - The current CPU-shaped proof assembly still forces tiny rho coefficient
    D2H plus final transcript snapshot D2H.
- Current accepted performance baseline remains:
  `benchmark-results/gpuprof/20260703-nc-chunk2.json`.

Next architectural step:
- Move Π_RLC claim combination/output onto the device, or more broadly make
  a whole-fold device transcript/session own FE/NC/RLC/DEC control so the host
  downloads final proof material only. Until that happens, device rho sampling
  is correctness-useful but pays extra host/device boundary cost.

## 2026-07-03 — iteration 26: NC device-driven column-round trace

Hypothesis: the prior direction was too focused on polishing the hybrid path.
Move Π_CCS NC column-round Fiat-Shamir ownership onto the device, matching the
FE row-round trace shape: round kernels write coefficients to device buffers,
the device Poseidon2 transcript absorbs them and writes challenges to device,
NC fold kernels read those challenges directly, and the host downloads one
bulk trace afterward only to replay into the canonical transcript.

Outcome:
- Architecturally useful but not a speed win.
- Correctness:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
  - `parity quick`: green, byte-identical; `ccs_prove` now reports
    device FE+NC rounds.
  - `parity e2e_bench`: green, byte-identical.
  - `gpuprof run e2e_bench --repeat 3 --assert-residency --json
    benchmark-results/gpuprof/20260703-nc-device-trace-direct-r3.json`:
    green, byte-identical; median CPU 2100.3ms, CUDA 855.9ms, speedup
    2.45x, CUDA repeats `[855.9, 854.4, 858.8]`, residency clean.
  - `gpuprof check benchmark-results/gpuprof/20260703-nc-chunk2.json
    benchmark-results/gpuprof/20260703-nc-device-trace-direct.json`: failed
    on `fold.superneo.pi_ccs.sumcheck.nc`.
- Local effect:
  - NC D2H copies dropped from about 80 to 12 and syncs dropped from about
    92 to 32, so the CPU-owned per-round challenge boundary was removed.
  - NC wall regressed from about 67.2ms to about 76ms because the new path
    adds per-round tiny device transcript launches and device-challenge fold
    kernels. `p2_transcript_absorb_device_challenge` accounts for about
    22.6ms / 140 launches across FE+NC.
  - A 1-thread launch experiment for transcript kernels preserved byte
    identity but did not improve timing and was reverted.
- Conclusion:
  - Full-GPU ownership is now further along, but performance requires the
    next architectural step: collapse FE/NC round control into a whole
    sumcheck command stream/CUDA graph or a deeper fused scheduler. Simply
    replacing host round-trips with one tiny kernel per round is not enough.

Current accepted performance baseline remains:
`benchmark-results/gpuprof/20260703-nc-chunk2.json`.

## 2026-07-03 — iteration 28: device Π_RLC K surfaces + lean claim shell

Hypothesis: after device-derived Π_RLC `X`, the next valid migration step is
not to recompute NC sidecars from `Z_mix`; it is to combine the actual Π_CCS CE
claim surfaces under the device-sampled rho coefficients. `y_ring` and
`y_zcol` are exactly K-vector RLC surfaces:
`out = Σ_i rot(ρ_i) · input_i`. Moving that algebra to CUDA makes the Π_RLC
output contract device-owned and avoids the previous invalid `Z_mix · χ_s`
shortcut.

Change:
- Add `rlc_combine_k_surfaces` in `kernels/pi_rlc.rs`. It consumes compact
  device rho coefficients, applies the Φ81 rotation convention on device, and
  combines `[input][surface][lane]` K surfaces into combined CE output
  surfaces.
- Add `reduce::rlc::{combine_y_ring, combine_y_zcol}` wrappers and a
  `claim_shell` constructor. The adapter no longer calls full
  `rlc_combine_claims`; it builds only the small shell fields, then fills
  `X`, `y_ring`, `ct`, and `y_zcol` from CUDA-derived outputs before
  `validate_combined`.
- Extend gpuprof taxonomy with `fold.superneo.pi_rlc.output.y_ring` and
  `fold.superneo.pi_rlc.output.y_zcol`.

Outcome:
- Accepted as a migration slice, not as the performance baseline.
- Correctness:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
  - `parity rlc`: green; CPU 3.47ms, GPU 2.60ms.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2139.42ms,
    CUDA 896.69ms, 2.39x.
  - `cargo check --workspace --release`: green.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json benchmark-results/gpuprof/20260703-rlc-device-surfaces-shell-r3.json`:
    green, byte-identical; median CPU 2131.7ms, CUDA 885.7ms, speedup 2.40x,
    CUDA repeats `[885.7, 885.1, 889.9]`, spread 0.5%, residency clean.
- Measured effect:
  - `fold.superneo.pi_rlc.output.y_ring`: about 7.5ms across 4 folds.
  - `fold.superneo.pi_rlc.output.y_zcol`: about 6.4ms across 4 folds.
  - `fold.superneo.pi_rlc.output.X`: about 0.2ms across 4 folds.
  - `fold.superneo.pi_rlc.combine_claims`: about 25.7ms across 4 folds. This
    is lower than the previous duplicate-compute version but still not gone,
    because the stage still owns claim validation, input-claim digest binding,
    rho setup, commitment mixing, and shell assembly.
- Architecture note:
  - This slice still uploads host-materialized Π_CCS CE surfaces into the RLC
    surface kernel. The next real migration step is a resident Π_CCS output
    bundle that feeds Π_RLC directly on device, so `y_ring`/`y_zcol` do not
    round-trip through host proof structs before being combined.

Current accepted performance baseline remains:
`benchmark-results/gpuprof/20260703-nc-chunk2.json`.

## 2026-07-03 — iteration 23: device-driven NC column rounds plus FE chunk cut

Hypothesis: standalone FE chunking improved the e2e median by about 1.7% but
missed the 2% rule because the sumcheck path still carries round-boundary
host work. Add the missing NC counterpart to the existing FE device transcript
trace: compute NC column-round coefficients, absorb them into the device
Poseidon2 transcript, fold at the device-written challenge, and bulk-download
the NC proof material/final column state once. Pair this with
`EVAL_CHUNK_PAIRS = 4` so the whole Π_CCS sumcheck package removes a boundary
and exposes finer FE work.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower NC D2H/sync counts and lower FE kernel time.
- Accept only if repeat-3 e2e median improves by at least 2% against
  `benchmark-results/gpuprof/20260703-nc-chunk2.json`; otherwise revert the
  NC trace and restore `EVAL_CHUNK_PAIRS = 16`.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity but introduced a measured NC
  regression and missed the >=2% e2e acceptance rule.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green after exporting the new trace type.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2116.37ms,
    CUDA 850.69ms, 2.49x.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json benchmark-results/gpuprof/20260703-nc-trace-fe-chunk4.json`:
    green, byte-identical; median CPU 2115.6ms, CUDA 836.3ms, 2.53x, CUDA
    repeats `[836.8, 836.3, 835.0]`, spread 0.2%, residency clean.
  - `gpuprof.py check benchmark-results/gpuprof/20260703-nc-chunk2.json
    benchmark-results/gpuprof/20260703-nc-trace-fe-chunk4.json`: failed.
- Local effect:
  - FE stayed improved locally, but the NC column trace made
    `fold.superneo.pi_ccs.sumcheck.nc` worse: about 67.2ms -> 77.2ms.
  - NC D2H copies dropped from about 80 to 12, but launches increased
    384 -> 536, H2D copies increased 8 -> 92, syncs increased 92 -> 112,
    and `p2_transcript_io_ops` added about 9.5ms.
  - The useful conclusion is negative: NC device transcript per-round op
    streams are not a win. The next sumcheck design must avoid per-round op
    buffer uploads/extra transcript launches entirely, not just move the
    challenge bytes onto the device.

Current accepted baseline remains:
`benchmark-results/gpuprof/20260703-nc-chunk2.json`.

## 2026-07-03 — iteration 20: narrower NC column chunks, second cut

Hypothesis: `nc_col_partials` is still the largest non-matvec CUDA kernel and
the lint still flags it at about 17% theoretical occupancy with underfilled
tail rounds. The accepted `NC_CHUNK_PAIRS = 4` cut improved e2e by 4.1%.
Try one more step to `NC_CHUNK_PAIRS = 2` so the column-round work exposes
more independent groups to CUDA. This keeps the same coefficient algebra and
same transcript order; only the partial partition changes.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `nc_col_partials` total time and possibly lower
  `fold.superneo.pi_ccs.sumcheck.nc` wall, with some extra reduction overhead.
- Accept only if repeat-3 e2e median improves by at least 2% against the
  current accepted baseline
  `benchmark-results/gpuprof/20260703-nc-chunk4.json`; otherwise restore
  `NC_CHUNK_PAIRS = 4`.

Outcome:
- Accepted.
- The candidate preserved byte identity and cleared the 2% acceptance rule
  against the prior accepted baseline.
- Verification:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2114.55ms,
    CUDA 853.32ms, 2.48x.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json benchmark-results/gpuprof/20260703-nc-chunk2.json`:
    green, byte-identical; median CPU 2110.3ms, CUDA 840.0ms, 2.51x, CUDA
    repeats `[843.8, 838.4, 840.0]`, spread 0.6%, residency clean.
  - `gpuprof.py check benchmark-results/gpuprof/20260703-nc-chunk4.json
    benchmark-results/gpuprof/20260703-nc-chunk2.json`: clean, online CUDA
    858.0ms -> 840.0ms, 33 comparisons, 0 regressions.
- Local effect:
  - `fold.superneo.pi_ccs.sumcheck.nc` improved to about 67.2ms.
  - `fold.superneo.pi_ccs.sumcheck.fe` remains the top recoverable lever at
    about 155.8ms wall and 106.3ms recoverable.
  - The useful conclusion is positive: the NC column-round chunk size was
    still too coarse at 4; 2 is now the accepted setting.

Current accepted baseline is now:
`benchmark-results/gpuprof/20260703-nc-chunk2.json`.

## 2026-07-03 — iteration 21: reuse DEC ring-form scratch

Hypothesis: `fold.superneo.pi_dec.open_children.forms` still costs about
84.2ms wall for about 27.5ms of kernel work. The current path allocates a
large device `chi` table and forms output inside the timed phase, and nsys
attributes about 27.6ms to `cuMemFree` there. Keep those buffers as reusable
Π_DEC scratch owned by `DeviceDec`, overwriting them each fold instead of
freeing them in the repeated prover loop.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `fold.superneo.pi_dec.open_children.forms` wall/API time and lower
  per-fold `cuMemFree`.
- Accept only if repeat-3 e2e median improves by at least 2% against the
  current accepted baseline
  `benchmark-results/gpuprof/20260703-nc-chunk2.json`; otherwise revert the
  scratch plumbing.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity and fixed part of the local API
  problem, but did not improve the full online median.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2118.02ms,
    CUDA 863.40ms, 2.45x.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json benchmark-results/gpuprof/20260703-dec-forms-scratch.json`:
    green, byte-identical; median CPU 2109.7ms, CUDA 846.0ms, 2.50x, CUDA
    repeats `[842.3, 850.4, 846.0]`, spread 1.0%, residency clean.
  - `gpuprof.py check benchmark-results/gpuprof/20260703-nc-chunk2.json
    benchmark-results/gpuprof/20260703-dec-forms-scratch.json`: clean by
    regression tolerance, but below this campaign's >=2% improvement rule.
- Local effect:
  - `fold.superneo.pi_dec.open_children.forms` improved from about 84.2ms to
    56.5ms and its API time dropped to about 0.1ms.
  - The win was offset elsewhere in the same DEC path, notably `split`
    becoming about 27.9ms in the captured run, so the full online median
    regressed from 840.0ms to 846.0ms.
  - The useful conclusion is negative: scratch lifetime alone is not enough;
    DEC needs a broader split/forms/open scheduling change or should wait
    behind the larger FE/oracle/RLC levers.

Current accepted baseline remains:
`benchmark-results/gpuprof/20260703-nc-chunk2.json`.

## 2026-07-03 — iteration 19: fused FE round transcript path

Hypothesis: the current accepted top lever is
`fold.superneo.pi_ccs.sumcheck.fe` at about 155.5ms wall with about 106.1ms
recoverable. The FE device transcript path still launches three tiny kernels
after the per-round partial-block reduction: final block-sum reduction,
coefficient-log copy, and Poseidon2 absorb/challenge. Fuse those into one
single-thread transcript kernel that reduces the block sums, writes the
canonical coefficient log, absorbs the same length prefix and coefficient
words, and writes the device challenge consumed by the fold kernel. Pair this
with the previously measured `EVAL_CHUNK_PAIRS = 8` FE occupancy win; do not
retry chunking alone.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower FE launch count and lower `fold.superneo.pi_ccs.sumcheck.fe` wall.
- Accept only if repeat-3 e2e median improves by at least 2% against the
  current accepted baseline
  `benchmark-results/gpuprof/20260703-nc-chunk4.json`; otherwise revert the
  fused kernel and restore `EVAL_CHUNK_PAIRS = 16`.

Outcome:
- Rejected and reverted.
- The candidate preserved byte identity and reduced the FE launch count, but
  the e2e median improvement was far below the 2% acceptance rule.
- Candidate verification while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`:
    green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2103.07ms,
    CUDA 867.22ms, 2.43x.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json benchmark-results/gpuprof/20260703-fe-fused-transcript.json`:
    green, byte-identical; median CPU 2110.7ms, CUDA 853.9ms, 2.47x, CUDA
    repeats `[857.7, 851.6, 853.9]`, spread 0.7%, residency clean.
  - `gpuprof.py check benchmark-results/gpuprof/20260703-nc-chunk4.json
    benchmark-results/gpuprof/20260703-fe-fused-transcript.json`: clean,
    online CUDA 858.0ms -> 853.9ms, 33 comparisons, 0 regressions.
- Local effect:
  - `fold.superneo.pi_ccs.sumcheck.fe` improved from about 155.5ms to
    149.4ms.
  - FE launches dropped from about 384 to 256.
  - `fe_round_partials` improved to about 17ms, but the fused
    single-thread transcript/reduction kernel became about 23ms, so the
    local win was too small for e2e.
  - The useful conclusion is negative: launch fusion alone is not enough;
    the next FE win needs to remove the remaining host gap or redesign the
    round loop more deeply, not just fuse the final tiny launches.

Current accepted baseline remains:
`benchmark-results/gpuprof/20260703-nc-chunk4.json`.

## 2026-07-03 — iteration 24: reuse DEC form scratch buffers

Hypothesis: the accepted baseline is
`benchmark-results/gpuprof/20260703-nc-chunk2.json` with CUDA median about
840.0ms and byte-identical e2e proof output. The current levers table ranks
`fold.superneo.pi_dec.open_children.forms` second: about 84.2ms wall, 27.5ms
GPU busy, and 56.5ms recoverable, mostly host/API gap. That path still
allocates zeroed `chi_r` and forms buffers per fold even though the tensor and
CSR kernels overwrite every consumed word.

Change: keep DEC-owned scratch buffers for the device `chi_r` table and the
`[2t][blocks][D]` form matrix, growing only when needed and reusing them across
folds. Use uninitialized device allocation only where the following kernels
fully write the buffer before any read. Leave the tiny challenge-vector upload
unchanged to avoid adding a synchronizing host copy.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower CUDA API/memset/free overhead in `fold.superneo.pi_dec.open_children.forms`.
- Accept only if repeat-3 e2e median improves by at least 2% against
  `benchmark-results/gpuprof/20260703-nc-chunk2.json` and `gpuprof check`
  passes; otherwise revert this scratch reuse.

Outcome:
- Rejected and reverted.
- Correctness was preserved while active:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
  - `parity quick`: green, byte-identical.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2113.91ms,
    CUDA 851.97ms, 2.48x.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json benchmark-results/gpuprof/20260703-dec-form-scratch.json`:
    green, byte-identical; median CPU 2103.0ms, CUDA 846.9ms, speedup 2.48x,
    CUDA repeats `[846.9, 845.9, 849.2]`, spread 0.4%, residency clean.
  - `gpuprof.py check benchmark-results/gpuprof/20260703-nc-chunk2.json
    benchmark-results/gpuprof/20260703-dec-form-scratch.json`: clean under the
    tolerance gate, but not accepted because the median regressed instead of
    improving by at least 2%.
- Local effect:
  - `fold.superneo.pi_dec.open_children.forms` stayed about the same/worse
    (85.7ms vs 84.2ms baseline), with the same ~58ms recoverable host/API gap.
  - Reusing the `chi_r`/forms allocation does not address the real DEC forms
    bottleneck; the cost is dominated by CSR form kernel/API scheduling and
    surrounding host gap, not just buffer allocation or memset.

Current accepted baseline remains:
`benchmark-results/gpuprof/20260703-nc-chunk2.json`.

## 2026-07-03 — iteration 25: contiguous deferred FE row tables

Hypothesis: the accepted baseline remains
`benchmark-results/gpuprof/20260703-nc-chunk2.json` with CUDA median about
840.0ms. The current levers table ranks `fold.superneo.pi_ccs.oracle.F` third:
about 59.8ms wall, 3.0ms GPU busy, and 56.7ms recoverable, mostly host gap.
The resident FE path builds one device row-table buffer per f-variable and then
copies each table into the FE arena separately. That keeps the data resident,
but it still pays repeated device allocations, zeroed output buffers, and tiny
copy launches.

Change: build all f-var row tables for one MCS into a single contiguous device
buffer, add a source-offset device-copy helper, and let the FE arena copy each
needed table slice from that contiguous buffer. This should reduce allocation
and object churn in `oracle.F` without changing the row-table math or transcript.

Expected outcome:
- Byte-identical `quick` and `e2e_bench`.
- Lower `fold.superneo.pi_ccs.oracle.F` wall/API gap and possibly lower
  `fold.superneo.pi_ccs.oracle.upload` overhead.
- Accept only if repeat-3 e2e median improves by at least 2% against
  `benchmark-results/gpuprof/20260703-nc-chunk2.json` and `gpuprof check`
  passes; otherwise revert this row-table packing change.

Outcome:
- Aborted before measurement and reverted before build.
- Reason: this was another local improvement to the current partial-GPU
  implementation. The better priority is expanding the protocol boundary so
  the repeated SuperNeo prover loop is GPU-owned: device transcript/control,
  resident Pi_CCS/Pi_RLC/Pi_DEC state, and final proof export only.
- No benchmark result accepted. Current accepted baseline remains:
  `benchmark-results/gpuprof/20260703-nc-chunk2.json`.

## 2026-07-03 — iteration 27: device-derived Π_RLC public X

Hypothesis: the accepted baseline remains
`benchmark-results/gpuprof/20260703-nc-chunk2.json` with CUDA median about
840.0ms. The priority for this slice is protocol migration, not a headline
speedup: Π_RLC already keeps `Z_mix` device-resident for Π_DEC, but the
combined CE claim's public `X` surface was still only assembled by host claim
algebra. Since `X = project_x_from_witness_mat(Z_mix)`, it can be derived from
resident device state after the witness mix.

Change: add `kernels/pi_rlc.rs` with `rlc_pack_public_x`, a `reduce::rlc`
wrapper that downloads the packed `Mat<F>`, and wire the adapter so
`fold.superneo.pi_rlc.output.X` overwrites the host-combined `X` before
`validate_combined` and Π_DEC. The RLC parity gate now explicitly asserts the
device-derived `X` equals the CPU-combined `X`.

Outcome:
- Accepted as a migration slice, not as a speed baseline.
- Correctness:
  - `cargo fmt --all`: green, with the existing stable rustfmt warning.
  - `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`: green.
  - `parity rlc`: green, including device `X` parity; CPU 3.02ms, GPU 1.51ms.
  - `parity e2e_bench`: green, byte-identical; direct print CPU 2108.91ms,
    CUDA 879.13ms, 2.40x.
  - `cargo check --workspace --release`: green.
  - `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
    --assert-residency --json benchmark-results/gpuprof/20260703-rlc-device-x-r3.json`:
    green, byte-identical; median CPU 2090.7ms, CUDA 872.0ms, speedup 2.39x,
    CUDA repeats `[872.0, 871.1, 877.9]`, spread 0.8%, residency clean.
- Measured effect:
  - New `fold.superneo.pi_rlc.output.X` leaf is visible in gpuprof:
    about 0.2ms across 4 folds, 4 launches of `rlc_pack_public_x`.
  - `Z_mix` remains device-resident; no D2H transfer on
    `fold.superneo.pi_rlc.mix_witness`.
  - This is intentionally not the performance baseline because it adds a tiny
    D2H proof-surface download while preserving the larger migration direction.
- Rejected in this slice:
  - Attempted to derive Π_RLC `y_zcol` from resident `Z_mix` via
    `Z_mix · χ_s`. The RLC parity gate rejected it, and the code was removed.
    Reason: Π_CCS' `y_zcol` is the NC digit-channel surface; Π_RLC must combine
    the input `y_zcol` claims under `Σρ_i`, not recompute it from `Z_mix`.

Current accepted performance baseline remains:
`benchmark-results/gpuprof/20260703-nc-chunk2.json`.
