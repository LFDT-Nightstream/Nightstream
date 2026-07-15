# Metal NIFS Backend

`neo-prover-metal` implements `NifsProverAdapter` without owning protocol
semantics. `neo-fold-clean` remains authoritative for transcript ordering,
claim algebra, proof assembly, and verification.

## M6 Data Flow

```text
fresh low-norm assignment
  -> Metal Ajtai commitment
  -> canonical CcsInstance
  -> Pi_CCS
       canonical sparse CPU row rounds and transcript replay
       Metal Ajtai Y_eval from signed-unit witness masks
       per-fold ring forms retained on-device for Pi_DEC
       Metal compact NC column rounds initialized from signed-unit masks
  -> Pi_RLC
       canonical rho sampling and claim algebra
       Metal witness random-linear-combination
       resident child planes reused as the running tail
       canonical projection schedule
  -> Pi_DEC
       Metal base-2 split into resident child planes
       device range and recomposition validation
       static explicit bar-matrix data retained on Metal
       exact-chi-matched Pi_CCS ring forms reused when available
       otherwise per-fold ring forms built from chi_r on Metal
       child y_ring projection and Phi81 reduction on Metal
       child Ajtai commitments from the verifier-owned matrix on Metal
       canonical child-claim construction
  -> shared deferred fold output
       one canonical RunningInstance owns Pi_RLC and Pi_DEC authority
       proof carrier reconstructs its public reduction surfaces at egress
       opaque Metal generation id for next-fold residency
       recursive F' compilation consumes a verifier-visible post-fold summary
```

The selected route is deliberately hybrid. Canonical sparse CPU row rounds beat
the existing device-owned FE candidate because that candidate duplicates table
preparation. The CPU still owns those rounds, transcript challenges, Ajtai-tail
sumcheck algebra, proof assembly, and verification. Metal owns the deterministic
Ajtai `Y_eval`, compact NC column rounds, bulk RLC witness mix, base-2 split and
validation, explicit-matrix ring-form construction, child opening projection,
child commitments, and cross-fold child residency. The `Y_eval` form buffer is
retained and reused by Pi_DEC only after an exact comparison of its chi table,
cache identity, matrix digest, and effective row count. Geometric runs are
already merged into the explicit row-block cache used by both Metal plans.
Compact seeded ring-form construction still uses the canonical dense-form
fallback until its structured generator has a Metal implementation. The report
records every selected routing decision.

The next Pi_CCS call still needs a canonical host witness mirror, so M6 does
not claim that every private byte remains device-only. It does remove the
Pi_RLC-to-Pi_DEC commitment boundary and keeps the authoritative child planes
resident for the next recursive fold. Pi_DEC downloads only validated child
sign masks. Their per-column representation is consumed directly by Pi_CCS
and the NC table builder instead of being transposed and rescanned on the host.

## Authority Boundary

The verifier never trusts a Metal-only digest, buffer, or generation id.
Every proof materializes as an ordinary `NifsProof` and is accepted by the
canonical CPU verifier. The accumulator digest and F' post-fold summary are
recomputed from canonical post-fold claims and parent authority.

The proof and running carriers share one fold-output authority. The proof
reconstructs Pi_RLC and Pi_DEC from that object's parent and children rather
than retaining duplicate claims. `MetalSession` owns the corresponding device
buffer. A stale or branched generation cannot select the wrong buffer: it
falls back to the materialized witness path. Terminal finalization consumes
the same carrier before materializing the final protocol object.

## Resident Sumcheck Candidates

The FE and compact NC implementations keep evaluation tables, transcript
snapshots, and fold state on Metal across rounds. The FE candidate remains
unselected because it duplicates row-table preparation. The old NC table-upload
path was not selected in the measured M5 Max result. The mask-native rewrite is
selected on macOS after real MSL compilation, exact proof parity, M6 crossover
and sustained gates, and Metal System Trace all passed on an M1 Max. Physical
iPhone acceptance remains separate.

The compact NC table begins with one `K` value per assignment column. Each
fold doubles a strided lane window until two windows would overlap in the
54-lane ring, then converts to dense rows. Storage remains approximately one
`K` value per original assignment column per witness.

The development candidate consumes the existing signed-unit column masks
directly for the first NC round, then folds them straight into the resident
width-two table after sampling the challenge. It never materializes the
width-one `K` table. At 15 witnesses and `ell_m = 19`, this removes one dispatch
and cuts the first-round/fold digit-source traffic from 480 MiB to 240 MiB
before cache effects; weights, equality values, and later folds are unchanged.
The host keeps only the masks on the successful path and reconstructs the
canonical width-one table lazily if Metal fails. While two compact windows are
disjoint, later round kernels visit only their `2 * width` live lanes instead
of all 54 lanes. This reduces the static per-witness column-round lane
evaluations from 28,311,498 to 3,506,122 (8.08x). These are traffic and loop-work
counts rather than an isolated NC speedup; the measured lifecycle and GPU-stage
results are recorded below.

An explicit-CSR MCS row-table offload was prototyped and rejected before
selection. On the production SHA shape, canonical MCS row construction costs
only 6–15 ms per fold. The proposed dispatch would move about 3.36 MB of witness
data and return about 4.19 MB of row tables, followed by host conversion into
the canonical table layout. Its best possible lifecycle gain is too small to
justify the extra ownership surface without a physical-GPU result.

Ajtai `Y_eval` is now implemented as the first high-value Pi_CCS candidate:
static bar matrices and signed-unit witness masks are execution data, while
transcript order, challenge sampling, sumcheck rounds, proof logs, and verifier
semantics remain in the canonical engine. The CPU oracle computes `Y_eval`
lazily only when the backend declines the shape. The Metal backend uploads the
chi table plus signed-unit masks, uses the existing ring-form/projection kernels,
downloads only `Y_eval`, and retains the form buffer for exact-checked Pi_DEC
reuse. This route passes real shader compilation and proof parity. Metal System
Trace also confirms that `Y_eval` executes on the GPU and that the retained
forms feed Pi_DEC without changing canonical proof bytes.

## M4 Gates

- CPU and Metal fresh commitments and complete proof bytes are identical.
- FE and NC candidate round logs match the canonical prover.
- The canonical verifier accepts Metal proofs and rejects a tampered child
  commitment.
- Recursive and terminal folds consume the resident running carrier.
- Explicit bar matrices are uploaded once; each fold builds forms and child
  openings on Metal without changing canonical child claims.
- CPU row-round transcripts with Metal `Y_eval` produce identical Pi_CCS proof
  logs, and the retained forms match a freshly derived Pi_DEC row point exactly.
- Device validation rejects an out-of-range or non-recomposing split, and
  Metal child commitments match the canonical Ajtai commitments exactly.

## M5 Gates

- Base, bootstrap, and consecutive steady folds match the canonical CPU
  running state and proof bytes.
- Every recursive proof uses the deferred proof carrier; recursive running
  outputs retain their Metal generation through the deferred running carrier.
- Pi_RLC and Pi_DEC proof surfaces materialize from the same shared fold
  output used by the next prover call.
- Transcript-prefix tampering and mutation of an earlier running commitment
  are rejected by the canonical verifier.
- Recursive F' compilation consumes the backend's complete post-fold summary,
  so the redundant prover-side NIFS.V replay is disabled without changing
  verifier behavior.
- Four-chunk SHA-256 and two-step Nebula memory lifecycles prove and verify.
- Stage time, command buffers, dispatches, waits, uploads, downloads, and
  resident/deferred fold counts are emitted in the benchmark report.

## M6 Gates

- Five measured four-chunk SHA-256 lifecycle pairs follow one warm-up per
  backend, alternate which backend runs first, retain ordered raw samples,
  and include synthesis, folds, and terminal materialization.
- CPU and Metal produce identical canonical proof authority and reduction
  transcripts accepted by the canonical verifier on every sample.
- Metal must be at least 1.52x faster at the median and 1.50x faster at p95,
  so the previously measured 1.510x result cannot pass unchanged.
- Synthesis of the next chunk overlaps the current fold, and the report records
  the work and saved overlap separately.
- Independent 60-second CPU and Metal runs must retain exact proof validity and
  achieve at least 1.15x Metal throughput.

## Measured Development Result

On an Apple M5 Max, the pre-oracle three-sample development report measured the
four-chunk SHA lifecycle at a 1.298-second median versus 1.960 seconds on CPU:
1.510x at the median and 1.499x at p95. Across four folds, Pi_CCS was 918 ms,
Pi_RLC 133 ms, and Pi_DEC 153 ms. The report recorded four deferred proof
folds, three deferred running folds, no recursive compile replay, and 7.36 MB
downloaded per lifecycle. The full five-sample crossover and independent
60-second-per-backend sustained acceptance test also passed.

The explicit-CSR Pi_CCS oracle slice described above was added after that
measurement, then removed after exact CPU profiling showed its low ceiling. The
old 1.510x number must not be attributed to any later Pi_CCS experiment.

On the current M1 Max, an exact four-fold CPU SHA trace measured 2,708 ms in
Pi_CCS. Ajtai `Y_eval` accounted for 983 ms (36%), NC for 1,322 ms (49%), and
MCS row construction for 41 ms (1.5%). The steady-fold `Y_eval` range was
220–304 ms and the NC range was 344–522 ms. These numbers are not comparable
to the M5 Max wall times, but they identify the same optimization order:
`Y_eval` first, then direct mask-native NC rounds and folds.

After accepting the Xcode 26.6 license and installing Metal Toolchain 17F109,
the real MSL build, primitive GPU arithmetic, and end-to-end CPU/Metal NIFS
proof-byte parity all passed. A complete M6 rerun measured five alternating
paired production SHA lifecycles. CPU and Metal medians were 3,692.4 ms and
1,691.5 ms (2.183x); p95 values were 3,741.8 ms and 1,722.7 ms (2.172x).
Every sample retained exact proof authority, clearing the 1.52x median and
1.50x p95 gates. Independent 60-second runs completed 17 CPU proofs and 37
Metal proofs for 2.263x normalized throughput, clearing the 1.15x sustained
gate. A counter-enabled diagnostic capture separately measured 3,991 ms versus
1,709 ms (2.335x); that single profiled sample is supporting evidence, not the
statistical acceptance result.

Metal System Trace measured steady-fold NC command buffers at approximately
204.5--206.5 ms, Pi_DEC at 36--42 ms, `Y_eval` at 27.8--31.7 ms, and resident
RLC at 10.9--12.6 ms. Peak Metal allocation was 736.8 MiB. Shader Timeline
attributes the largest aggregate work to generic `nc_round_partials`, followed
by `dec_ring_partials`, Poseidon2 uniform hashing, and mask-native first-round
NC. Compact folds and reductions are materially smaller. The next optimization
target is therefore NC round arithmetic; buffer reuse is secondary unless an
allocation or bandwidth counter identifies a concrete pressure point.

The small two-step Nebula fixture measured 121 ms on Metal versus 115 ms on
CPU. It remains useful for parity and command-floor accounting, but M6 requires
crossover on the production-core SHA workload rather than on tiny fixtures.

An authentic reduced WAT fixture now runs through WASM execution, Nebula trace
and relation construction, NIFS proving, and canonical verification. It loads
42 from linear memory, multiplies by 6, returns 252, produces five execution
rows, and lowers to 11,836,121 relation rows across 13,106,340 columns. Every
sample preserves terminal authority and exact CPU/Metal proof parity. On this
shape CPU proving measured 52.38 seconds versus 56.75 seconds on Metal (0.923x).
This proves the WASM+Nebula integration, but it is not a crossover claim: its
seeded Phi81 matrices intentionally retain canonical CPU ring-form fallback and
its current workload is not dominated by the accelerated SHA hot path.

Those macOS figures are development evidence, not an iPhone performance
claim. Physical-iPhone crossover, energy, and sustained thermal behavior must
still be measured before release. Pi_CCS is now the largest remaining
lifecycle target, while the small Nebula result exposes the command/submission
floor for short proofs.
