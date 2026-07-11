# CIR-SOUND / CIR-COMPLETE

```text
property_ids: CIR-SOUND, CIR-COMPLETE
claim:
  The exact supported artifact is the 4,076,614-row plain/stateless [1,1]
  full-history profile with one recursive invocation, a terminal fold, direct
  terminal CE, and the minimal-supported-bit-carrier relation. Standalone base
  and recursive owners do not, by themselves, implement the closed Step.Holds
  relation: a producer step cannot constrain the public inputs of a batch that
  is consumed later. The next recursive owner checks that delayed link, and the
  terminal-fold owner checks it for the trailing batch.

  Soundness is therefore a composed claim: every satisfying full-history
  generated assignment decodes to a sequence of Step.LocalHolds edges or a
  named circuit bad event. In the exact branch, every edge receives exactly one
  next-consumer or terminal OutgoingLinked proof, base seeds and adjacent
  states are pinned, and Step.closeLocal yields the same closed Step.Holds
  trace used by TRACE-VALID.

  Completeness quantifies over successful compiler executions for that same
  fixed profile. `CompilerWitness` carries independent source/interpreter
  executions and direct semantic inputs, never `Satisfies` or an accepted
  verifier conclusion. From those executions, the theorem constructs an
  assignment satisfying every exact composed row.
assumptions:
  - Goldilocks field arithmetic and the explicit Euclid-prime boundary used by
    canonical-u64 proofs.
  - Poseidon2 collision resistance is not needed for deterministic circuit
    correspondence; only functional correctness of its generated rows is.
  - Cryptographic soundness of NIFS is not assumed as deterministic circuit
    correspondence. Production PiRLC uses a one-point polynomial identity, so
    correspondence itself returns exact native verification or `BadRoot`.
    M6 must bound that root event using the degree-106 bound, commit-then-
    challenge schedule, SIS projection-preimage binding, and Fiat-Shamir model.
  - The artifact importer is a trusted translation boundary until its parser
    and Rust row extraction are themselves verified. It may establish row
    identity/inclusion, never a local protocol conclusion.
non_goals:
  - Claiming standalone producer rows enforce a future batch's public link.
  - Treating native preflight success as an R1CS constraint.
  - Treating an honest fixed witness as universal soundness or completeness.
  - Compact terminal SNARK soundness (DEC-SOUND/M6).
  - Stateful, Nebula, other batch schedules, multiple recursive invocations,
    alternate carriers, or parameterized circuit-family correspondence.
paper_sources:
  - HyperNova Construction 2 steps 3-5 and the terminal compiler verifier.
  - HyperNova NIVC-compatible encoder requirement.
  - SuperNeo PiCCS/PiRLC/PiDEC verifier composition.
rust_surfaces:
  - paper/f_prime/r1cs.rs::{enforce_f_prime_base_step_circuit,
    enforce_f_prime_recursive_step_circuit}
  - frontends/r1cs_f_prime/compiler.rs (stateful application composition)
  - engine/decider.rs::{synthesize_statement_r1cs,
    enforce_base_state_constants,enforce_state_link,emit_terminal_fold,
    enforce_terminal_latest_link}
  - paper/f_prime/native.rs::{prove_with_semantic_state,verify}
circuit_or_encoding_artifacts:
  - Exact 4,076,614-row composed full-history artifact for the fixed
    plain/stateless [1,1] profile, including one recursive invocation, terminal
    fold, direct terminal CE, and minimal-supported-bit-carrier relation.
  - Exact compact ownership manifests for every top-level, recursive, terminal,
    nested NIFS, projection, accumulator, continuity, public-pin, and direct-CE
    row family used by that artifact.
  - Existing exact subartifacts: canonical-u64, u64 increment/addition,
    recursive counter block, F' public encoding, the complete 12,498-row
    checked base program, base-state pins, state links,
    terminal delayed links, the full 6,661-row chunk-digest program, the
    600-row production Poseidon2 permutation, and one-claim full CE
    child/running continuity.
failure_class:
  A producer-local theorem is mislabeled as a closed step theorem; host shape
  rejection is omitted; base authority is left witness-selected; adjacent
  state wires are disconnected; stateful semantic output is only hashed and
  not computed; a prior or trailing fresh batch is unlinked; a fixed honest
  vector is generalized without a universal row argument; a one-point
  projection is incorrectly promoted to coefficient-wise equality without a
  polynomial-root bad event.
counterexample_or_witness:
  tests/FPrimeStep.lean constructs an unlinked installed batch for which
  Step.checkLocal=true but Step.check=false. Rust lifecycle tests separately
  mutate the prior link, terminal latest link, base constants, state links,
  accumulator continuity, and semantic state. The exact 714-row PiRLC
  projection helper also accepts `E(X)=X-7` at beta 7 while the full
  coefficient identity is false; this is the pinned `BadRoot` branch.
lean_theorems:
  - Nightstream.Protocol.FPrime.Step.holds_iff_local_and_outgoing
  - Nightstream.Protocol.FPrime.Step.fPrimeBaseLocal_sound
  - Nightstream.Protocol.FPrime.Step.fPrimeRecursiveLocal_sound
  - Nightstream.Protocol.FPrime.Step.closeLocal
  - Nightstream.Assurance.FPrimeCircuit.split_check_eq_true_iff
  - Nightstream.Implementation.R1CS.CheckedProgram.sound
  - Nightstream.Implementation.R1CS.CheckedProgram.complete
  - Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_sound
  - Nightstream.Implementation.R1CS.FPrimeEncodingSound.fPrimeEncoding_sound
  - Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.topLevel_covers_program
  - Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.nifs_covers_block
  - Nightstream.Assurance.FPrimeRecursiveCircuit.decodedChecks_sound
  - Nightstream.Assurance.FPrimeRecursiveCircuit.decodedChecks_local_sound
  - Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot
  - Nightstream.Assurance.FPrimeRecursiveCircuit.projectedChecks_local_sound_or_badRoot
  - Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_sound_or_bad
  - Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_complete
axiom_report:
  Local/closed decomposition theorems use only [propext] and are guarded in
  tests/Axioms.lean. Artifact theorems must add fail-closed reports.
proof_hash:
  Recorded per artifact and theorem in assurance/evidence-ledger.jsonl.
conformance_status:
  CIR-SOUND is artifact-checked for the exact fixed profile above.
  `fPrimeCircuit_sound_or_bad` consumes `Satisfies
  FPrimeFullHistoryRows.fullRows assignment`, reconstructs every semantic row
  owner, and returns a two-edge `ValidExecution` with the direct terminal
  predicate or `BadEvent.recursiveRoot` / `BadEvent.terminalRoot`. The generated
  artifact has 4,076,614 rows, 3,298,653 columns, and 125,877,402 nonzero sparse
  entries. The two root-event probability bounds remain M6 obligations.
  CIR-COMPLETE is artifact-checked for the same profile:
  `fPrimeCircuit_complete` reassembles an independent `CompilerWitness` into
  satisfaction of every exact `fullRows` row. No stateful, Nebula,
  general-schedule, or parameterized-profile correspondence is claimed.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cargo test -p neo-fold-clean --release --test f_prime_r1cs
  - cargo test -p neo-fold-clean --release --test system_lifecycle_f_prime_link
  - cargo test -p neo-fold-clean --release --test system_decider_r1cs
```

## M4 row-owner ledger

| Owner | Current evidence | Remaining obligation |
|---|---|---|
| Plain base step | Its exact rows feed the closed base edge, and its execution witness reconstructs those rows | Stateful/Nebula profiles |
| Chunk digest / Poseidon2 | Exact generated programs are included in the fixed-profile soundness path | Other profiles and circuit-family generalization |
| Counter / `enc_inst` | Exact recursive counter and encoding owners are sound and reconstructible from their compiler witnesses | Other schedules and profiles |
| Base pins / state links / terminal latest link | Exact owners close verifier authority and both delayed producer/consumer links | Other schedules and profiles |
| CE continuity | All 14 fixed-profile child/running continuity shards are included | Parameterized claim counts and Nebula-adv layouts |
| Recursive NIFS step | Exact PiCCS, PiRLC, PiDEC, point-binding, accumulator, and residual owners yield recursive validity or `BadEvent.recursiveRoot` | M6 probability bound and other profiles |
| Application / Nebula frontend | The supported profile is stateless and has no Nebula rows | Stateful and Nebula artifact correspondence |
| Terminal fold / direct CE closure | Exact terminal NIFS, authority, link, accumulator, continuity, public-pin, and direct-CE owners yield terminal validity or `BadEvent.terminalRoot` | M6 probability bound and other terminal profiles |
| Full-history composition | Exact `fullRows` satisfaction yields a two-edge `ValidExecution` plus direct terminal validity or a named root event; independent successful compiler executions reconstruct all exact rows | Generalized circuit families and M6 probability bounds |
