# Bounded Metal application tables

Application values now have one private owner. It retains a table when the
resident prefix and its next fold fit the available workspace. Otherwise,
it reconstructs row windows from the original matrices, witness masks, and
all prior challenges. Both the row calculations and their folds run on Metal.

The production workspace comes from the existing 16,000,000,000-byte device
allocation ceiling, minus live allocations and derived scratch requirements.
No new production threshold, Cargo feature, or environment variable was added.
This bounds application-table payloads. It is not a bound on total RSS.

## Exact replay

For round `r`, folded row `u` is the sum of original rows `(u << r) + v`,
weighted by the equality polynomial of the preceding challenges at `v`.
The implementation evaluates power-of-two source windows. Each window uses
the lower challenges in the original fold order. A GPU accumulation kernel
applies the remaining challenge weights between windows. No full equality
tensor is allocated.

The first base-to-extension fold uses the same 16 bytes for each input pair
and output value. Later folds use a separate buffer half the size of the
preceding buffer. Completed commands and old buffers are released before the
next stage. Once a complete folded prefix fits, it can remain resident.

Round kernels retain global pair addresses for equality, assignment, and
carried terms. Application reads use local window addresses. The assignment
suffix is included even when it extends beyond the application rows.

Terminal verification also evaluates exact row windows and returns a global
failure index. It never uses the prover's selective satisfied-row substitution.

## Tests

- The seven- and thirteen-row fixtures force a 672-byte application workspace:
  two base rows plus two extension rows for 14 matrices. The thirteen-row case
  still requires replay after two prior challenges.
- The seventeen-row fixture uses 1,792 bytes: eight source base rows, four
  base-row units for a later fold, and four for the two extension output rows.
  It exercises both a window folded more than once and later GPU accumulation.
- Each fixture compares complete PiCCS proof bytes, every round coefficient
  and challenge, transcript state and cursor, and all output openings against
  Optimized and resident Metal. Each checks the application payload peak and
  the transition to resident storage.
- A terminal test accepts the original witness and identifies changed global
  row 6 in the final window. CPU, resident Metal, and bounded Metal agree.
- All 22 Metal library tests pass. The expanded replay test also passes.
  All nine supported Nightstream engine comparisons pass; two CUDA checks
  remain ignored because the kernel is absent.

These internal workspace values are derived test inputs, not production limits.
All tests and test builds use the 300-second cap from `AGENTS.md`. No Lean
command ran. Formatting and whitespace checks pass.

## Production comparison

A fresh nonzero-carried second fold matches all **945,983 saved CPU proof
bytes** and all **16 returned witness matrices**. The parent, claims,
openings, identities, and transcript also match. All 19 source files are
byte-identical to the prior Metal inputs. See the input and output comparison
receipts in this directory.

The complete test takes 85.86 s, including preparation and file work. Its C/R/D
phase takes 47.99 s. Peak RSS is **16,384,065,536 bytes** (16.38 GB), below the
current 16 GiB test guard. This is a production phase check, not a full
lifecycle measurement.

The subsequent complete three-step Metal lifecycle passes in **164.661508625 s**
with **16,813,703,168 bytes peak RSS** (16.81 GB). Preparation takes 34.50 s,
base proving 6.80 s, the two extends 45.61 s and 57.81 s, and terminal
verification 19.94 s. All starting inputs, the selected profile, circuit
identity, and final state match the prior Metal benchmark. The run uses
command-scoped sleep prevention and a release executable with function symbols
retained. It is an individual measurement; no speed gain over prior runs is
inferred. The complete CPU run must use this same saved executable.

## Remaining scope

The old maximum accepted row shape required 27,505,776,032 bytes for its initial
application table and 55,011,552,064 bytes while its first fold overlapped.
The application owner no longer requires those full allocations. This is a
source-derived comparison; no giant circuit was constructed.

Host application rows, the CPU matrix cache, device matrix indexes, and other
live witness/oracle data still require bounded storage. Generic seeded Metal
plans also retain their separate partial allocation. Current Nightstream row
plans use explicit entries and geometric runs. These results do not establish
total RSS for every supported circuit or the 5× full lifecycle speed target.

The exact benchmark executable is retained with release function symbols for
later profiling. Its UUID and build settings are in `application-replay-binary.json`.
Source snapshots, binaries, and full proof artifacts remain in:

`/Users/nijaar/Library/Application Support/Nightstream/runs/metal-norm-20260921-bgylh03h`
