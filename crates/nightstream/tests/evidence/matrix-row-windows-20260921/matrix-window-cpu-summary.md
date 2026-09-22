The matched Time Profiler runs took **1330.702711417 s on CPU** and
**291.840242833 s on Metal**. The measured CPU/Metal ratio is
**4.5596957379811025×**. This instrumented pair is below the requested 5× target.
It is not a comparison of uninstrumented runs.

Both runs verified. Their complete start records match after removal of the
engine field. Their complete finish records match after removal of elapsed
seconds. This includes the initial state, message, three steps, two active
extends, selected `b = 2`, `k_rho = 16`, `B = 65536`, final state and circuit
identity. These records do not compare serialized intermediate proofs.

| Phase | CPU seconds | Metal seconds | CPU / Metal |
|---|---:|---:|---:|
| Prepare | 34.540521416 | 34.555801584 | 0.99956× |
| Base proof | 60.087123834 | 6.794190417 | 8.84390× |
| Extend 2 | 420.104987000 | 86.644211709 | 4.84862× |
| Extend 3 | 530.468488000 | 115.295510916 | 4.60095× |
| Verify | 285.501396209 | 48.550385125 | 5.88052× |

Both captures used the same saved executable, with UUID
`5841982D-F5C1-35FA-903D-66E7053B74CB`, and the same Time Profiler settings.
They ran in sequence. This result applies to the saved
`matrix-window-borrowed-source.tar.gz` build. The later sealed-record and
identity-streaming changes are outside this measurement.

The CPU trace reports that the target exited normally. Its recorded duration
is 1331.474612 s. The controller completed recording and trace finalization in
1461.617523541 s on its monotonic clock, within the 1800 s Instruments cap.
Both traces report one non-induced Nominal thermal interval for the complete
capture. Nominal state does not prove equal clock frequencies.

The full CPU sample export is 1,704,334,689 bytes. It was parsed with
`ElementTree.iterparse`; each row was removed after processing. No complete XML
tree or sample timestamp table was retained. The capture has 6,925,984 samples,
with 6925.984 s of summed CPU sample weight. Of these, 6,925,530 samples have
backtraces and 454 are sentinel rows without a backtrace. No other missing or
empty backtraces were found. All 10,176 native addresses resolved against the
exact saved image and its recorded load address. There are 1,734 unresolved top
samples in other images; no top frames lack an image annotation.

Selected self sample weights are:

| Function | CPU sample seconds |
|---|---:|
| `BlockRng::generate_and_set` | 3703.521 |
| `FnMut::call_mut::hf91b8f9d3f0d7af3` | 841.296 |
| Rayon `helper::h5c9bf7c451fe5f7c` | 515.776 |
| `Rq::mul` | 380.850 |
| `RingEvalScratch::bar_active` | 280.576 |

These weights sum work across target threads. They are not elapsed time or GPU
shader time. Inclusive weights overlap. The complete function table contains
3,620 entries. Timestamp references were counted but not retained, so this is
whole-capture attribution, not attribution by lifecycle phase.

The observed lifetime kernel RSS peaks were 16,790,093,824 bytes on CPU and
13,880,573,952 bytes on Metal. The CPU value exceeds 16,000,000,000 bytes; both
recorders used a 16 GiB guard. Final post-exit peaks are unavailable, so the final
unsampled intervals are not proved. These observations do not prove a general
process memory bound.

No matched uninstrumented CPU run, CPU/GPU clock-frequency data, or GPU limiter
counters are present. Time Profiler excluded waiting threads. The raw trace,
complete XML, descriptor data, full symbol summary and saved executable remain
in the private run directory named in the receipts.

The exact record comparison is in
[matrix-window-cpu-matched-lifecycle-comparison.json](matrix-window-cpu-matched-lifecycle-comparison.json).
The full compact function table is in
[matrix-window-cpu-lifecycle-profile-functions.json](matrix-window-cpu-lifecycle-profile-functions.json).
The export, parser and exact-image symbolication receipt is in
[matrix-window-cpu-lifecycle-profile-export-receipt.json](matrix-window-cpu-lifecycle-profile-export-receipt.json).
