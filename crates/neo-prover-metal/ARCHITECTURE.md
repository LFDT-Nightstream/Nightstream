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
       selected: canonical sparse CPU prover and transcript replay
       available candidates: resident Metal FE and compact NC sumchecks
  -> Pi_RLC
       canonical rho sampling and claim algebra
       Metal witness random-linear-combination
       resident child planes reused as the running tail
       canonical projection schedule
  -> Pi_DEC
       Metal base-2 split into resident child planes
       device range and recomposition validation
       static explicit bar-matrix data retained on Metal
       per-fold ring forms built from chi_r on Metal
       child y_ring projection and Phi81 reduction on Metal
       child Ajtai commitments from the verifier-owned matrix on Metal
       canonical child-claim construction
  -> shared deferred fold output
       one canonical RunningInstance owns Pi_RLC and Pi_DEC authority
       proof carrier reconstructs its public reduction surfaces at egress
       opaque Metal generation id for next-fold residency
       recursive F' compilation consumes a verifier-visible post-fold summary
```

The selected route is deliberately hybrid. On current Apple hardware, the
canonical sparse Pi_CCS algorithm beats the current dense Metal candidates.
Metal owns the bulk RLC witness mix, the base-2 split and validation,
explicit-matrix ring-form construction, child opening projection, child
commitments, and cross-fold child residency. Compact seeded or geometric
matrices use the canonical dense-form fallback until their structured
generators have a Metal implementation. The report records every routing
decision.

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
snapshots, and fold state on Metal across rounds. They have exact round-log
parity tests but are not selected on the measured M5 Max path because the
canonical sparse CPU implementation is faster.

The compact NC table begins with one `K` value per assignment column. Each
fold doubles a strided lane window until two windows would overlap in the
54-lane ring, then converts to dense rows. Storage remains approximately one
`K` value per original assignment column per witness.

## M4 Gates

- CPU and Metal fresh commitments and complete proof bytes are identical.
- FE and NC candidate round logs match the canonical prover.
- The canonical verifier accepts Metal proofs and rejects a tampered child
  commitment.
- Recursive and terminal folds consume the resident running carrier.
- Explicit bar matrices are uploaded once; each fold builds forms and child
  openings on Metal without changing canonical child claims.
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

- Five measured four-chunk SHA-256 lifecycles follow one warm-up per backend
  and include synthesis, folds, and terminal materialization.
- CPU and Metal produce byte-identical proofs accepted by the canonical
  verifier on every sample.
- Metal must be at least 1.5x faster at the median and 1.25x faster at p95.
- Synthesis of the next chunk overlaps the current fold, and the report records
  the work and saved overlap separately.
- Independent 60-second CPU and Metal runs must retain exact proof validity and
  achieve at least 1.15x Metal throughput.

## Measured Development Result

On an Apple M5 Max, the final three-sample development report measured the
four-chunk SHA lifecycle at a 1.298-second median versus 1.960 seconds on CPU:
1.510x at the median and 1.499x at p95. Across four folds, Pi_CCS was 918 ms,
Pi_RLC 133 ms, and Pi_DEC 153 ms. The report recorded four deferred proof
folds, three deferred running folds, no recursive compile replay, and 7.36 MB
downloaded per lifecycle. The full five-sample crossover and independent
60-second-per-backend sustained acceptance test also passed.

The small two-step Nebula fixture measured 121 ms on Metal versus 115 ms on
CPU. It remains useful for parity and command-floor accounting, but M6 requires
crossover on the production-core SHA workload rather than on tiny fixtures.

Those macOS figures are development evidence, not an iPhone performance
claim. Physical-iPhone crossover, energy, and sustained thermal behavior must
still be measured before release. Pi_CCS is now the largest remaining
lifecycle target, while the small Nebula result exposes the command/submission
floor for short proofs.
