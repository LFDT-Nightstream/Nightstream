# Sealed application records

The Rust application builder writes original rows and recipe syntax to private
files. Row offsets and recipe links also stay in files. Finishing the writer
closes write handles and returns one immutable owner. An append failure prevents
sealing. On Unix, the private file names are removed immediately.

The assembler passes this owner with the relocated fixed envelope. The loader
uses the same records for source rows, witness execution, validation, and both
occurrences in the canonical identity preimage. It computes the existing
Poseidon2 identities without retaining all application words. No saved Lean
artifact or protocol value changes.

The new tests check exact term and recipe order, zero and duplicate terms,
read-only sharing, partial append failure, corrupt length, variable scope,
causal recipes, output dependencies, row coverage, and global assertion errors.
The complete identity test compares every word against the original saved
numeric-array value before checking the pinned circuit and key identities.
See `scope.json` for the current test results and remaining checks.

This change removes retained row and recipe data that grew with application
row count. It does not claim a universal process RSS bound. Current row
reconstruction keeps only distinct variables in one row. The fixed production
key limits the public lifecycle to 7,709 application variables; see
`memory-bound-review.md` for the calculation and the remaining RSS boundary.
Caller-owned affine expressions remain outside the record owner.

The native engine review in `engine-memory-accounting.md` found no missing
large allocation that grows with rows or width. The current CPU SumCheck
payload equation has a 16,000,000,000-byte ceiling. Metal derives windows from
its remaining budget and counts host and device metadata together.
Allocator and driver RSS are still measured overhead, not a formal guarantee.

The complete second Metal fold matches all 945,983 saved CPU proof bytes and
all sixteen returned matrices. It takes 83.36 s with 13.19 GB peak RSS. The
full public Metal lifecycle verifies in 264.52 s with 12.75 GB peak RSS,
including preparation, base proving, both extensions and terminal verification.
Matched Time Profiler runs on this exact image take 1,308.78 s on CPU and
277.00 s on Metal, or **4.7249×**, below the 5× target. Both verify and their
complete start/finish records match after excluding engine and elapsed time.
Observed lifetime RSS peaks are 15.79 GB for CPU and 12.63 GB for Metal; final
post-exit profile peaks are unavailable. The raw Metal result above supplies
a complete native process peak for that separate run.

The matched traces report Nominal thermal state throughout. The current Metal
trace attributes 97.132 seconds of summed CPU sample weight to matrix-window
construction: 59.258 seconds to counting and 37.677 seconds to filling. This
is the measured remaining host cost. These values are not wall time or GPU
time. All 5,317 native sample addresses resolve against the saved image.
See `streamed-template-metal-count-fill-attribution.json` and the compact
function table for coverage and unassigned samples.
