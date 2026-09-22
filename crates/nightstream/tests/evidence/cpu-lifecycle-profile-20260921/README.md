# CPU lifecycle and terminal profiling

These runs use the saved three-step Poseidon2 inputs on the Apple M1 Max.
Instruments runs use the owner-approved 1,800-second cap from `AGENTS.md`,
including trace finalization. The current process RSS guard is 16 GiB
(17,179,869,184 bytes). No new memory exception was used.

## Results

| Run | Result | Observed kernel peak RSS |
|---|---|---:|
| Full Optimized lifecycle | Stopped during terminal verification at the Instruments deadline | 16,397,549,568 bytes |
| Terminal phase with piped stdin | Blocked before preparation; stopped | No verification result |
| Terminal phase with regular-file stdin | Passed CPU verification and wrong-state rejection | 13,893,877,760 bytes |

The full run completed preparation in 34.87 s, base proving in 103.33 s,
and the two extends in 558.73 s and 765.77 s. It has no completed terminal
result or `benchmark_finished` event. Its trace could not be exported.
These partial times do not establish a full CPU/Metal speed ratio.

The successful terminal run took 448.09 s, including 34.85 s of cold circuit
preparation. It loaded the stored Metal step-3 envelope, built a new CPU cache,
verified the complete witness commitments and openings, and rejected a wrong
final state. This is not a warm verification time or a full lifecycle time.

[The comparison receipt](terminal-comparison.json) records exact equality of
all 19 input files and every result field except the engine name. The result
contains all 16 running claims, the fresh claim, circuit identity, initial
state, and final state. The CPU run did not generate a new folding proof.
The earlier production comparison remains the proof-byte parity evidence.

RSS uses `MACH_TASK_BASIC_INFO.resident_size_max`, read while the native target
was live. The OS peak could not be read after exit. The final unsampled
interval is therefore not proved. Current-RSS samples and physical-footprint
samples are retained separately; neither replaces the kernel RSS observation.

## Trace and clock limits

The successful Time Profiler trace contains 2,723,146 samples for native PID
74135. Their summed weight is 2,723.146 CPU seconds across threads, not wall
time. Of these samples, 268 have no stack. The native image UUID and load
address match the preserved executable.

The release build strips Rust symbols. An unstripped release rebuild has a
different code section: 2,506,492 bytes versus 2,507,316 bytes in the captured
image. This does not permit a global address mapping. See the
[code comparison](optimized-terminal-file-profile-text-comparison.json) and
[sample summary](terminal-samples-summary.json).

A separate check matched complete function byte ranges with unique symbol
ownership in both executables. It identified 14 sampled functions, covering
1,989.327 sampled CPU seconds. The largest identified costs are:

| Function | Self CPU sample weight | Share of all samples |
|---|---:|---:|
| `BlockRng::generate_and_set` | 1,074.611 s | 39.46% |
| `u128_div_rem` | 812.913 s | 29.85% |
| Goldilocks NEON Poseidon2 width-8 permutation | 63.659 s | 2.34% |
| `__umodti3` | 35.748 s | 1.31% |

These are thread-summed self weights, not elapsed phase times. The full byte
ranges, including padding, must match exactly and occur once in each image.
The rebuilt range must have one symbol at its start and no interior symbol.
Unmatched ranges remain unnamed. This proves the listed instruction-to-symbol
mapping; it does not prove whole-program identity or identify unnamed callers.
See the [function matching receipt](terminal-function-matches.json).

Power logs show Maintenance Sleep during both CPU captures. The native and
Python monotonic clocks stop during sleep on this host; the deadline uses
wall time. The records keep the original times. They do not subtract sleep
intervals or treat different measurement windows as equal. Future measurements
will use `caffeinate -is` for the measured command while the Mac is on AC power.
This does not change permanent power settings. See the
[clock evidence](sleep-and-clock-evidence.json).

## Memory gap

The accepted row bounds permit 239,217,427 application rows and 245,587,286
logical rows. Before bounded replay, one fresh witness and 14 matrices required
`8 × 14 × 245,587,286 = 27,505,776,032` application-table bytes.
The device guard rejected this request without providing bounded evaluation.
This is a result derived from the source and manifest, not a measured large
circuit run. See the [derivation](accepted-row-table-bound.json).

The later [bounded application change](../bounded-application-20260921) replaces
that full allocation with row replay. Other live storage, including the
application rows, CPU matrix cache, device matrix index, and witnesses, must
also fit. This record does not establish the memory bound for all supported
circuits. It does not propose a smaller supported circuit limit.

## Retained records

The JSON summaries identify the scope of each run. The benchmark logs,
Instruments logs, RSS observations, deadline records, request, and terminal
comparison are retained here. The raw trace, 639 MB sample export, complete
sample aggregation, executable copies, and analysis scripts remain in:

`/Users/nijaar/Library/Application Support/Nightstream/runs/metal-norm-20260921-bgylh03h`

No benchmark, recorder, or export process remains live. No production code or
Lean artifact changed in this profiling pass. The 5× full lifecycle target
and the memory bound for all supported circuits remain open.
