# PiDEC signed product checkpoint — 2026-09-14

The generated 13,680-block replay takes **7.12 s** in the final isolated run with eight workers,
compared with **8.57 s** for unchanged Nightstream
`34797904ec8d0693a962c0573d7010d4461e5951`. Compute/read/write time falls
from **6,054 ms to 4,537 ms**. This is a 25.1% reduction in that phase
and a 16.9% reduction in total command time. The requested 2× command
improvement was not reached.

Both runs used the same Apple M1 Max, input, wrapper command and eight-worker
setting. Builds finished before timing. All subagents were stopped; no other
build, test or profiler ran during a timed command. Peak RSS was
776,880,128 bytes (740.9 MiB), compared with 776,388,608 bytes before the change.
Startup is included in command time and is not charged to each block.

| Version | Workers | Command | Compute/read/write | All output bytes match |
| --- | ---: | ---: | ---: | --- |
| Unchanged baseline | 8 | 8.57 s | 6,054 ms | Yes |
| First signed variant | 8 | 8.43 s | 4,594 ms | Yes |
| Byte guard, before full validation | 8 | 8.24 s | 4,464 ms | Yes |
| Final checkpoint | 8 | 7.12 s | 4,537 ms | Yes |
| Final checkpoint | 10 | 7.37 s | 4,793 ms | Yes |

Time outside the replay timer varied: 3.776 s in the first byte-guard run,
2.583 s in the final eight-worker run. The source keeps both results; the
smaller compute-time variation does not explain the full command difference.
No claim of a repeatable 2× overall gain is made.

The replay now prepares the exact 81-word cyclic key representation and stores
positive and negative child positions as byte offsets. Signed products use
separate addition and subtraction loops. Unsupported field digits use the
existing general product. Universal proofs preserve canonical values, the
exact accumulator equality, and the existing equality with `honestMessages`.
The seed/index mapping, production key, protocol, package, and
`b = 2`, `k_rho = 16`, `B = 65536` profile are preserved.

The generated C uses an unboxed UInt64 accumulator in the signed loops.
The first signed variant's profile replaces 15,921 old product-loop leaf
samples with 7,062 signed-loop leaf samples. ChaCha remains at about 6,168 leaf samples. These
samples locate work; they are not elapsed-time proportions.

A scalar ChaCha attempt was stopped after the three proof rounds required by
`formal/nightstream-fprime/AGENTS.md`. Its wiring proof exceeded the existing
proof limits. The draft and failed logs are outside Git. It was not timed or
retained. No new Metal path or Lean runtime change is in this checkpoint.
The unchanged Lean fork source is
`14f2fed9c9782048e896c924f424624585acd772` on `nico/lean-performance`,
based on Lean 4.30.0. Its captured stage1 configuration still embeds release
hash `d024af099ca4bf2c86f649261ebf59565dc8c622`.

Validation passed:

- Static checks, full library build, all axiom audits, and canonical identity.
- All 19,008 commitment coefficients and all 191,700 output bytes with eight
  and ten workers.
- All 128,098,921 package bytes, 8,119,393 PiRLC parent bytes, 3,477,319 PiDEC
  split bytes, and 3,052,541 accepted public PiRLC output bytes.
- 6,051 native product comparisons: all basis positions with both signs,
  zero, cancellation, nonzero initial sums, and general field fallback.
- Complete dense and sparse-tail outputs, first-two-block key and product
  bytes, omitted-zero behavior, malformed inputs, strict bounds, partial
  batches, merge checks, and mixed-error order at both worker settings.
- Rust public C/R handoff and all 62 existing PiRLC mutations, including
  changed public targets. The first Rust invocation used the wrong identity
  type; it was corrected to the pinned structural identifier before this pass.

Validation results are recorded in `source-manifest.json` and the command logs
at `/Users/nijaar/starstream/nightstream-stage1-evidence/pidec-next-34797904-lhd1_q6l`.
Large inputs, outputs, profiles, and failed drafts remain there, outside Git.

The input is generated test data. The original Linux production capture and
its 5,982-second scan were not rerun. These measurements do not establish a
production-scan speedup. Generated masks are not claimed to open the accepted
C commitments; the public C/R checks are separate.

The September 4 review's full Stage 1 profile, conformance and lifecycle work
is outside this checkpoint. The September 6 review's global closure, pin and
production migration work is also outside this checkpoint. The benchmark
fixture caveat and exact-identity checks apply and are retained here.
