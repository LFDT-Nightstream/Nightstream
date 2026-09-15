# Lean replay performance checkpoint — 2026-09-14

This checkpoint extends Nightstream `f721ae232c1c0a3d7591b60e9db324137accf621`.
Measurements used an Apple M1 Max, 64 GiB RAM, ten CPU workers selected from
the host CPU count, and a local source build of Lean 4.30.0. Builds finished
before timed replay commands. These are single-run development measurements
on generated data, not a new measurement of the original Linux production scan.

| Command | Baseline seconds | Optimized seconds | Reduction |
| --- | ---: | ---: | ---: |
| PiRLC witness replay, 13,680 blocks | 51.62 | 31.77 | 38.5% |
| PiDEC commitment replay of that parent | 41.49 | 34.18 | 17.6% |
| PiDEC dense replay, 1,240 blocks | 11.30 | 9.78 | 13.5% |
| PiDEC sparse full-carrier replay | 6.18 | 4.31 | 30.3% |
| PiRLC empty-source replay | 11.72 | 9.03 | 23.0% |
| Accepted PiRLC input check | 19.97 | 17.18 | 14.0% |

PiDEC `compute_read_write_ms` fell from 35,315 to 29,994 for the generated
PiRLC parent, and from 5,104 to 4,274 for the dense case. Initialization and
compilation are not charged to each block. PiRLC `kernel_nanos` is a sum of
task durations, not elapsed command time. Optimized peak RSS was 1.39–1.41 GiB;
this checkpoint does not establish a memory reduction.

Profiles selected three changes:

- Calculate prior-state hash chunk count from input length. This removes
  repeated list slicing. Existing equality statements are retained; the
  assembler's private count proof now uses the equality theorem.
- Use a direct field fold for the signed PiRLC action. The compiler emits a
  specialized loop. A structural proof equates the fold with the previous
  ordered sum; signed-input checks and the generic fallback stay intact.
- Construct GMP values directly from `uint64` when `unsigned long` has enough
  value bits. This removes temporary GMP values from a hot runtime conversion.
  The existing split conversion remains on 32-bit and LLP64 platforms.

The Lean runtime change is commit
`884cd4a5dc0dbc8d99189c333466d36d916c7af1`, based on Lean v4.30.0 commit
`d024af099ca4bf2c86f649261ebf59565dc8c622`. The portable
[runtime patch](lean4-4.30.0-uint64-mpz.patch) includes the compiled/interpreted
regression test. It can be applied with `git am` to that release base.
The local clone is `external/lean4`, branch `nico/nightstream-runtime-performance`.

The measured build used checked-in stage0 to build stage1. Its embedded Lean
hash still identifies the release base; the changed runtime objects and final
executable links were recorded separately. Both `leanrt` and `leanshared`
targets were rebuilt. Lake does not detect a runtime-only change from the
unchanged embedded hash: final executable traces were removed and executables
were relinked with the existing Lake artifact-cache control disabled.
The C++ runtime remains an existing trusted implementation link.

Validation passed:

- Every tested replay output byte, including 738,720 PiRLC coefficients and
  every 19,008-coefficient commitment result, matched the unchanged baseline.
- All 128,098,921 bytes of the canonical package matched the committed file.
  The canonical binding and Rust identity pins also matched.
- Static checks, the complete library build, and the axiom audit passed.
  No theorem statement, profile, production key, or protocol hash changed.
- Integer boundary checks passed against independent decimal GMP values.
  The Lean regression passed compiled and interpreted execution.
- Sparse tail coverage, zero blocks, invalid bounds, noncanonical parents,
  duplicate entries, and malformed signed masks passed their checks.
- Rust matched the accepted public C-to-R handoff and rejected all 62 existing
  PiRLC mutations, including changed public targets.

The generated private source masks are not claimed to open the accepted C
commitments. They test arithmetic and before/after replay equality. The public
C/R check is separate. The original private production capture remains unavailable
on this Mac; the 5,982-second complete production scan was not rerun.

An external Metal product probe matched all 38,016 coefficients and output
bytes for two blocks. GPU command times were 0.092 and 0.090 ms; complete host
dispatch and conversion took 2.857 and 0.715 ms. Shader/pipeline setup took
133.206 ms. JSON encoding and writes are additional. This is not a measured
complete-replay speedup. The shader and host conversion have no checked Lean
connection, and no production executable uses the probe.

Raw profiles, fixtures, outputs, timings, runtime object records, and the Metal
probe are outside Git at
`/Users/nijaar/starstream/nightstream-stage1-evidence/lean-performance-f721ae23-j8vxsqym`.
The sibling `pirlc-generated-fixtures-f721ae23-p60zobnw` directory records how
the accepted public fixture was obtained from the repository evidence archive.
Every Lean command used the repository's 1,500-second cap; native test commands
used its 300-second cap. All builds and tests used one queue.
