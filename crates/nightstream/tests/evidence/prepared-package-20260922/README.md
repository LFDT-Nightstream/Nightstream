# Prepared circuit packages

Compilation computes the existing circuit and application identities once.
The native package retains its fixed execution envelope in a private immutable
file. Saving streams that envelope and the original ordered application rows
and recipe syntax into one versioned package. Publication refuses to overwrite
an existing destination and exposes only the completed file.

Loading checks the format, selected profile, dimensions, row/witness coverage,
recipe grammar and causal references. It rebuilds private record indices and
seals an independent snapshot. It imports cached structural/application identity
components and reconstructs the small binding context; it does not rerun the
whole-circuit identity calculation. No authentication policy or certificate is
required by the crate. The caller selects the expected verifier configuration;
proof data cannot replace it.

The outer format contains eight version bytes and four output-form tags. The
application builder always appends four final output rows; its other counts
are derived from the validated application plan and records. The inner format
contains its version, fixed-envelope length, cached identity components, the
fixed envelope, and streamed rows/recipes. The reader rejects trailing data.

Checks pass: ten record-codec tests, nine prepared-package tests, one compiler
shape regression, four public storage tests, two engine-selection tests, and
nine supported engine comparisons. The guard checks take 99.89 seconds for
the prepared-package tests, 2.89 seconds for the compiler regression, and
40.42 seconds for the four public tests. The previously run Metal constraint
rejection test remains ignored in the public test batch. Two CUDA checks remain
ignored. An initial public test incorrectly passed
physical row positions to a logical-row API; the original circuit rejected those
positions. Correcting the test to logical endpoints made all four public tests
pass. The full workspace release check with all targets and Metal/CUDA passes.

The ignored Metal test compiles two applications with the same dimensions and
public state. The original requires a private input to be zero; the altered one
removes that constraint while preserving raw term arity. It copies only the
original cached identity components into the altered package. The loaded prover
produces a proof with matching claimed bindings and public state. The verifier
configured from the original application rejects the false constraint at
logical row 6,369,850. This takes 86.31 seconds and 10,319,380,480 bytes native
peak RSS under the 300-second cap.

The measurements below predate the fixed-metadata node guard. Saved benchmark
image `5BC2196F-B02A-3B3B-9E52-3C201C15449A` has schema2 and
scope `prepared_package_load_prove_verify`. The `compile` command runs once;
CPU and Metal `run` commands use the same saved artifact in fresh processes.
The measured path starts before package open and includes loading, validation,
engine setup, the base proof, two active folds, and terminal verification.
No preparation time is subtracted from a historical measurement.

Compilation takes 34.775985292 seconds and saving takes 0.134325333 seconds.
The saved package is 127,306,104 bytes and has the same circuit identity as the
prior compilation-inclusive measurements. The raw Metal load/prove/verify run passes in 212.528718625 seconds with
11,202,560,000 bytes final native peak RSS. Loading and creating both Metal
capabilities takes 2.332506375 seconds. Base proving takes 6.69484875 seconds,
the two extensions 69.961736792 and 94.710985292 seconds, and terminal verification
38.828422417 seconds. It verifies the same final state and circuit identity.
Matched Time Profiler runs on that image, before the node guard, take
1,263.701827917 seconds on CPU and 213.495212792 seconds on Metal:
**5.919110838×**. Complete inputs,
configuration and final result records agree. Both captures are serial with
matching settings/device metadata and Nominal thermal state. Observed lifetime
RSS peaks are 14,301,118,464 bytes for CPU and 11,449,663,488 bytes for Metal;
final post-exit peaks are unavailable.

A later review found unbounded redundant entries in structurally accepted fixed
metadata. The compiler-derived [fixed-source bound](fixed-source-bound.md) is
now in the decoder. The saved measurement above predates that guard;
the guard checks and final-image measurements are complete.
The 5.2782× historical compilation-inclusive result has a different timing scope.

## Final build with the fixed-metadata bound

Image `7DCAD994-817D-33E5-95BB-666FA9543762` passes all nine prepared-package
checks, the maximum/zero compiler-shape check, and four public save/load checks.
The new source files and tracked source diff match the saved image snapshot.
The final package has the same 127,306,104 bytes as the package before the guard;
the comparison checks every byte. Compilation takes 34.857374625 seconds and
saving takes 0.133257583 seconds in a separate process.

Matched Time Profiler runs take **1,262.100768458 seconds on CPU** and
**212.71809575 seconds on Metal**, or **5.933208287×**. Both load the same package,
prove the base and two recursive steps, and pass terminal verification. Complete
start and finish records agree after removing engine and elapsed time. The
captures use the same device and settings, run separately, and report Nominal
thermal state. See [the checked comparison](bounded-package-matched-time-profile-comparison.json).

Observed kernel RSS peaks are 14,448,033,792 bytes for CPU and 11,474,714,624 bytes
for Metal; final post-exit profile peaks are unavailable. The separate raw Metal
run passes in 212.469842375 seconds with 11,374,968,832 bytes final native peak
RSS. Its package loading and engine setup take 2.152479833 seconds.

The first final Metal recorder did not save its trace after the target passed
and exited. Its normal stop request did not resolve the wait. Only that owned
recorder was stopped. The attempt is retained as incomplete; a fresh normal
capture completed and is the Metal capture used in the comparison.

Stored data and working buffers are bounded across supported native shapes.
These results do not establish a universal process-RSS guarantee for every
circuit: allocator, runtime, driver, and simultaneous residency also contribute.
Package provenance and selection of the expected verifier configuration remain
the caller's responsibility. Loading adds no authentication policy or
whole-circuit identity calculation.
