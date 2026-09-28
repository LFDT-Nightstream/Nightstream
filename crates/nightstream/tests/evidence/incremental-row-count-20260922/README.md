# Incremental row-storage accounting

The count pass keeps cumulative candidate counts and a snapshot of the last
accepted row. Each run adds its exact encoded payload cost. The first explicit
or geometric run in a matrix adds one offset family. A complete candidate of
R rows uses fixed storage + run payload + 4*(R+1)*active families, plus the
existing count vectors and caller row payload. The extra zero offset and all
preceding empty rows are included when a family first appears.

An accepted row copies candidate counts to the snapshot. A rejected row leaves
the accepted snapshot and prefix unchanged. Filling allocates from that
snapshot, as before. No new array, pass, worker, feature or environment variable
is required.

Template substitution also skips normalization for zero or one canonical input
form. Nonzero field scaling preserves its ordered, distinct run keys. Multiple
inputs still use the existing normalization.

Twelve matrix-row tests, sixteen saved formula tests and twenty-three Metal
tests pass. The new row test independently enumerates late offset families and
checks exact and one-byte-short workspace boundaries. The production second-fold
comparison also passes: all 945,983 proof bytes, sixteen returned matrices,
parent claims, openings, transcript and identities match. The nineteen source
files match across all 542,178,475 bytes.

The new `89776242-C7BC-32FF-8174-47B117EBFF79` image completes the verified raw
Metal lifecycle in **250.277271958 seconds**, with a final native peak RSS of
**12,681,920,512 bytes**. Its start and finish records match the prior run apart
from elapsed seconds.
See [scope.json](scope.json) and the
[saved image record](incremental-count-benchmark-image.json).

The last matched Time Profiler ratio, **4.7248758153884545×**, belongs to the
older `8B18A367-06BA-3915-8254-40C42C9C0620` image: CPU 1308.784675084 seconds
and Metal 276.998745834 seconds. A matched CPU/Metal result for the new image
is still pending. The raw Metal result is not paired with that older CPU time.
See the [matched comparison](../sealed-application-records-20260921/streamed-template-matched-time-profile-comparison.json)
and the [profile I/O review](profile-io-review.md).

## Recipe-stack reuse

The next build reuses one private disk-backed continuation stack for each native
recipe batch. Each recipe resets its logical length before reading nodes, so a
failed variable lookup cannot leave operands for the next call. Stored records
stay immutable. Disk use follows the maximum depth reached during the batch;
resident scratch does not grow with recipe depth or count. Empty batches still
perform no stack I/O.

Nineteen application-record and native-loader tests pass, including nested
recipes, causal assignments and reuse after lookup failure. Four Poseidon2
tests pass against the stored Lean rows, witness program and execution values.
Workspace release checks pass with all targets and Metal/CUDA. No Lean command
was run.

Saved image `9F975E6F-3954-3D37-8C7D-97A6BBD37CDA` completes the raw Metal
lifecycle in 244.83093 seconds with 12,717,260,800 bytes final native peak RSS.
Its complete start and finish records match the preceding run except for time.
The Metal Time Profiler capture finishes in 246.453710167 seconds, with
12,711,673,856 bytes observed kernel lifetime peak RSS. Its final post-exit peak
is unavailable. The matched CPU capture takes 1,300.842536625 seconds, for **5.278242862×**.
Both verify and have complete matching start records except engine and finish
records except elapsed time. Settings and device metadata match; the captures
are serial, both report Nominal thermal state, and both finish within the
1,800-second cap including trace finalization. CPU observed peak RSS is
15,783,100,416 bytes; its final post-exit peak is also unavailable.

These are compilation-inclusive results. The owner's revised target includes
prepared-package loading, proving and terminal verification, with one-time
compilation measured separately. The new package feature and its benchmark
remain to be implemented; subtracting preparation time is not a load measurement.
