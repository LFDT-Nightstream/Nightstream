# Matched full lifecycle profiles — 2026-09-21

The same saved executable completes the three-step lifecycle in
**1,256.020431625 s on Optimized and 164.737161375 s on Metal**, a **7.62×**
ratio under matching Time Profiler instrumentation. This includes preparation,
base proving, two active folds, and terminal verification. Both runs verify the
same final state. The comparison without Instruments remains unmeasured on CPU;
the request for one 30-minute CPU run is pending and that command has not run.

| Phase | Optimized CPU (s) | Metal (s) |
| --- | ---: | ---: |
| Prepare | 34.524239208 | 34.526846625 |
| Prove base | 59.835702167 | 6.804702334 |
| Extend 2 | 396.958310208 | 45.611830292 |
| Extend 3 | 496.363860083 | 57.861076000 |
| Verify | 268.338163042 | 19.932587750 |
| **Full lifecycle** | **1,256.020431625** | **164.737161375** |

`matched-lifecycle-comparison.json` compares all starting fields except engine
and all final fields except time. Inputs, profile, circuit identity, step count,
and final state match. The native logs are stored here. They contain timings and
accepted states. The separate
[production comparison](../bounded-application-20260921) proves equality of all
945,983 CPU/Metal proof bytes, all sixteen returned matrices, claims, openings,
parent, transcript and identities for the current second-fold implementation.

The executable is `application-replay-benchmark-binary`, with UUID
`D66506D5-8D84-3D01-8EEB-E21332765BD7`. Both captures use that exact file.
`benchmark-binary.json` records the build and identity. Release optimization is
unchanged; the build retains function symbols. Captures run serially on the
same M1 Max MacBook Pro with 64 GiB RAM, macOS 26.6.2, and Instruments 27.0.
Both use one-millisecond CPU samples with the same settings, captured in
`trace-metadata.json`. Command-scoped `caffeinate -is` prevents sleep on AC.
The exported thermal tables report nominal state for each complete trace.

The capture driver uses the 1,800-second Instruments cap approved in
`AGENTS.md`, including target execution and trace finalization. It records target
and recorder identities, checks their command and start time before cleanup,
and runs a separate deadline watcher. Both recorders and native targets exit
normally. Complete controller times are 1,388.24 s for CPU and 171.20 s for
Metal. The ordinary five-minute test cap was not changed.

Observed kernel lifetime RSS peaks are **16,133,701,632 bytes on CPU** and
**16,628,383,744 bytes on Metal**. Each capture uses the existing 16 GiB RSS
guard for this approximate-16-GB pass. Post-exit peak reads are unavailable;
the final unsampled intervals are not proved. Physical footprint is diagnostic
data. The separate Metal run without Instruments records 164.661508625 s and a
final native maximum RSS of 16,813,703,168 bytes; see the preceding evidence.

The complete exports contain 6,861,913 CPU backtrace samples and 125,243 Metal
backtrace samples. They also contain 416 and 1,648 sentinel rows, respectively,
which have no backtrace. Every sample has a one-millisecond weight. Sample
weights add CPU work across target threads and are not wall times. Inclusive
function totals overlap; recursion is counted once per function in each stack.
Native symbols use the original executable's matching UUID and recorded load
address. Unknown frames remain unknown.

Random-block generation accounts for **54.07%** of the CPU run's backtrace
weight. The leading named CPU costs on the Metal run are the shared Poseidon2
permutation (**51.89%**) and signed-column PiRLC accumulation (**22.74%**).
These percentages describe host CPU samples. GPU kernel costs come from the
separate [device profile](../metal-gpu-profile-20260921), not these CPU samples.
Attribution receipts and complete compact function totals preserve the sample
denominators, unknowns, and all self/inclusive function weights.

The universal RSS requirement is still unmet. `matrix-cache-bound.json` derives
a supported row shape with 239,217,423 repeated three-term equality assertions
plus four output rows. It adds no variables. Matrix A alone then has at least
717,652,269 geometric runs, each stored as three `u64` values. The host payload
is at least **17,223,654,456 bytes**; retaining its full device copy doubles that
payload. Row indexes, other matrices, witnesses, spare capacity and eager
application rows add further memory. No large circuit was generated. Matrix
windows must be generated directly from the package before a full cache is
built; eager application storage is a separate preparation-memory gap.

Raw traces, the 1.66 GB CPU sample XML, the 34.65 MB Metal sample XML, exact
executable, and complete backtrace data are retained in the private run folder:

```text
/Users/nijaar/Library/Application Support/Nightstream/runs/metal-norm-20260921-bgylh03h
```

`export-receipt.json` records completed exports and thermal intervals. The
capture logs, resource records, native phase logs, and compact analysis are
archived here. This slice changes documentation and evidence only. No Lean
command, production source change, new feature, or environment variable was
needed for these captures.
