# Profile-driven Metal opening changes — 2026-09-21

The [GPU profile](../metal-gpu-profile-20260921) identified commitments,
sparse ring products, and geometric forms as the main device costs. Two
read-only subagent reviews inspected commitments and sparse products while
the main agent inspected geometric forms. Builds and GPU runs were serialized.

Two changes remain:

- Each geometric coefficient is computed once for both extension components.
  The two output sums keep their original term order and buffer layout.
- Sparse products skip multiplication by one. Form words are canonical, so
  this is exact as a machine word as well as modulo Goldilocks. Other
  magnitudes keep the existing multiplication.

The circuit, key, transcript, profile, buffers, and Cargo features are unchanged.
The production profile is Nightstream Goldilocks with `b = 2`, `k_rho = 16`.

## Full lifecycle measurements

Every run uses three steps: preparation, base proving, two active extends, and
terminal verification. All runs verified and have identical inputs, profile,
circuit identity, and final state. Each used the existing 300-second timeout
and 16 GiB RSS guard for the owner's approximate 16 GB pass. GB below is decimal.

| Run | Seconds | Peak RSS, GB |
| --- | ---: | ---: |
| Saved baseline, before encoder labels | 179.51 | 16.78 |
| Shared geometric coefficient | 174.80 | 16.27 |
| Shared coefficient and skipped unit multiplication, retained | 167.80 | 16.86 |
| Added center-out commitment order, rejected | 171.18 | 16.78 |
| Repeated baseline, with encoder labels and unchanged arithmetic | 193.31 | 16.75 |

The retained run uses 6.5% less time than the saved baseline and 13.2% less
than the repeated baseline. These are individual runs, with visible variation;
the summary uses the smaller measured difference. No full CPU lifecycle ratio
is established. [Timings](timings.json) contain exact phase and RSS values.
The binary paths and build command are recorded in each benchmark receipt.

The center-out commitment order passed CPU equality tests but showed no gain
over the two opening changes. It and its added fixture were removed. Its
measurement is retained to explain the decision.

## Exact CPU comparison

Both opening tests pass against CPU results. The geometric fixture also covers
overlapping runs, field reduction in a long ratio-2 run, zero and negative
ratios, distinct extension components, and partial blocks. An initial fixture
edit failed the builder's sorted-run check; sorting its input resolved that
test setup error before the successful runs.

The final build produced the second complete production C/R/D proof from the
stored nonzero-carried inputs. All **945,983 canonical proof bytes** match the
CPU reference. A direct comparison also matches all **16 returned witness
matrices**, every claim and opening, identities, and transcript values. The
source files were hardlinked from the unchanged CPU source directory. The
phase took 86.42 s including preparation and file work; peak RSS was
16,322,904,064 bytes. See [the comparison receipt](profiled-openings-second-output-comparison.json).

The phase replays the verifier as an additional check. Acceptance comes from
the exact CPU comparison, including output openings. No Lean command ran.
Cargo fmt, the release builds, and the diff whitespace check passed.

The saved production inputs, output matrices, proof, and benchmark executables
remain in the private run directory named in the receipts. This folder keeps
the logs and comparison records. The 5× CPU/Metal lifecycle target and a memory
bound for every supported circuit remain open.
