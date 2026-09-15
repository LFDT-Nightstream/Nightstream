# PiDEC commitment replay performance — 2026-09-14

The complete replay command for the retained generated 13,680-block case takes
**8.50 seconds with eight workers**, below the requested ten-second target.
The unchanged baseline takes 37.27 seconds with the same worker setting:
**4.38 times faster**, a 77.2% reduction in command time.

| Workers | Baseline command | Optimized command | Baseline compute/read/write | Optimized compute/read/write |
| --- | ---: | ---: | ---: | ---: |
| 8 | 37.27 s | 8.50 s | 32,965 ms | 6,038 ms |
| 10 | 35.71 s | 10.59 s | 30,125 ms | 6,804 ms |

These are isolated measurements on the same Apple M1 Max with 64 GiB RAM.
All subagents were stopped before the baseline rebuild. No other build, test,
profiler, or agent work ran during a timed command. The baseline was rebuilt
from unchanged Nightstream `f13e74d6900ec7cf36bc513fd17bba1639871426` and
Lean `884cd4a5dc0dbc8d99189c333466d36d916c7af1`. The optimized source and runtime
were then restored and rebuilt. Each build finished before measurement.
Baseline and optimized runs used the same input, command, hardware, and worker
setting. Complete output files were compared after each run.

Eight workers match this host's reported eight performance cores; its two
other cores are efficiency cores. Threads were not pinned. The existing
`LEAN_NUM_THREADS` setting selects the count. The default remains unchanged
and selects ten workers on this Mac. The eight-worker peak RSS fell from
1,499,447,296 to 775,946,240 bytes, a 48.3% reduction. Compilation and startup
are not charged to each block. Earlier development timings are provisional
and are not used for the comparison above.

The retained changes are:

- Read the low GMP magnitude bits directly for 64-bit conversion on platforms
  with a sufficiently wide `unsigned long`. Unsigned negation restores the
  original residue for negative values. The 32-bit and LLP64 fallback remains.
- Reduce each exact ChaCha coefficient with three native Goldilocks wide
  reductions. A structural proof equates the result with the original 256-bit
  Nat reduction. Seed, row, block, lane, rounds, and feed-forward are unchanged.
- Keep commitment accumulators in canonical machine words. Each task slot
  carries its sum across batches. The same block tasks and collection points
  are retained, and a proved slot-update equality preserves the field sum.
- Prepare each child's zero decision and words once per block, then reuse them
  across all 22 key rows. A proved Boolean scan replaces the finite universal
  zero test. Arbitrary field inputs retain the generic multiplication path.
- Delay base rows and combination constraints until requested. This avoids
  eager construction during replay startup and preserves their exact bodies.

The existing specification and `honestMessages` equalities are retained.
New arithmetic and scheduling value theorems are audited. There are no new
axioms, unchecked production externs, protocol parameters, or hash families.
The production profile remains `b = 2`, `k_rho = 16`, `B = 65536`.

Validation passed:

- Full library build, static checks, axiom audit, and canonical identity check.
- All 19,008 final commitment coefficients and all 191,700 output bytes, at
  both eight and ten workers, against the rebuilt unchanged baseline.
- All 8,119,393 parent bytes, 3,477,319 split bytes, 3,052,541 accepted public
  PiRLC output bytes, and 128,098,921 canonical package bytes.
- Every key and product byte for the first two stored blocks, plus complete
  dense and sparse-tail output comparisons. The sparse case covers the full
  4,685,394-block carrier with four stored blocks and omitted zero blocks.
- Strict norm and canonicality bounds, duplicates, malformed signed masks,
  partial batches, trailing data, incomplete merges, and mixed-error order.
  Error and partial-output comparisons passed at both worker settings.
- Compiled and interpreted Lean integer-conversion tests. The final C++
  boundary test output also matches the unchanged runtime byte for byte.
- Rust accepted the complete public C-to-R handoff and rejected all 62 existing
  PiRLC mutations, including changed public targets.

The generated source masks are not claimed to open the accepted C commitments.
They test computation and before/after equality. The public C/R checks are
separate. The original Linux production capture and its 5,982-second complete
scan were not rerun; these results do not establish a production-scan time.

The Lean change is `14f2fed9c9782048e896c924f424624585acd772`, on `nico/lean-performance` in
`nicarq/lean4-optimized`. The [portable runtime patch](lean4-4.30.0-mod64.patch)
applies after `884cd4a5dc0dbc8d99189c333466d36d916c7af1` on Lean 4.30.0.
The build uses checked-in stage0 followed by stage1. Its embedded Lean hash
still identifies release base `d024af099ca4bf2c86f649261ebf59565dc8c622`;
source and runtime artifact records identify the changed implementation.
Both runtime variants and the shared library were rebuilt, and final executable
traces were removed to force relinking. The C++ runtime remains a trusted
implementation link.

An isolated external Metal experiment completed the same valid computation
and all comparisons in **0.50 seconds**. Host processing before reading any
expected output took 460.291 ms; the full host path through comparisons took
468.613 ms. All final output bytes and the first two complete key/product
outputs matched Lean. The product and key buffers total about 2.21 GB; process
RSS does not account for all GPU memory. This is an empirical prototype.
The shader, host buffer mapping, and execution have no checked Lean connection,
and no production executable uses it.

Trial geometry, bounded-index, and generic-fallback variants did not establish
a useful gain and were removed. Direct word-key output was not added because
its measured conversion cost was too small to close the remaining gap.

All raw inputs, outputs, profiles, command logs, process inventories, source
patches, and the Metal prototype are outside Git at:

`/Users/nijaar/starstream/nightstream-stage1-evidence/pidec-sub10-f13e74d6-dhh_tmyt`

`isolated-*-metrics.json` holds the final timings. `source-manifest.json` records
both signed source commits and artifact hashes; hashes are provenance records,
not substitutes for the complete byte comparisons. Lean commands used caps of
at most 1,500 seconds; native tests used caps of at most 300 seconds.

To repeat the eight-worker command after building `replayPiDECCommitment`, run
from `formal/nightstream-fprime` and use a fresh output path:

```bash
LEAN_NUM_THREADS=8 /usr/bin/time -l timeout --signal=KILL 1500s \
  bash scripts/validate.sh pi-dec-commitment-replay \
  /Users/nijaar/starstream/nightstream-stage1-evidence/lean-performance-f721ae23-j8vxsqym/pirlc-runtime-measured.jsonl \
  /tmp/pidec-commitments-new.json 0 13680
```
