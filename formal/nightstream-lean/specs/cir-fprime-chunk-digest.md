# CIR-FPR-CHUNK-BIND

```text
property_id: CIR-FPR-CHUNK-BIND
claim:
  The exact schema-1 chunk-shape digest artifact is one deterministic
  straight-line field program from `(constant_one, start_step)` to all 6,659
  derived columns, including the four public chunk_digest lanes. Every
  satisfying assignment agrees with that executable program, two satisfying
  assignments with equal inputs have equal public digests, and executing the
  program constructs a satisfying witness. The smaller final-four-row theorem
  remains available for local output-binding composition.
assumptions:
  - Canonical assignment representatives and constant-one column zero.
  - Schema-1 profile: start index 9, three claims, D and production kappa,
    and m_in=257. Other supported cardinalities/layouts need their own hashes.
non_goals:
  - Native Rust helper refinement. The exact circuit program is authoritative
    at M4; equality with `f_prime_chunk_public_digest` is an M5 obligation.
  - Chunk-content authentication. The native F' digest is intentionally a
    step/shape digest to avoid the recursive-link fixed point; NIFS and the
    accumulator path own content authority.
paper_sources:
  - HyperNova Construction 2 state boundary update.
rust_surfaces:
  - crates/neo-fold-clean/src/paper/f_prime/digest_circuit.rs
    (enforce_f_prime_chunk_public_digest_circuit)
  - crates/neo-fold-clean/src/paper/f_prime/r1cs.rs
    (base and recursive branch bindings)
circuit_or_encoding_artifacts:
  - FPrimeChunkDigestArtifact.lean plus six generated definition shards,
    schema 1, all 6,661 exact rows.
failure_class:
  Native preflight supplies an honest digest but the proof relation permits a
  prover-selected value, or the F' state consumes a value disconnected from
  the digest gadget output.
counterexample_or_witness:
  Before the production fix the base circuit accepted arbitrary digests. The
  retained Rust forgery builds the complete honest Poseidon witness but changes
  claimed lane zero; the first binding row rejects it.
lean_theorems:
  - Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_sound
  - Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_claim_unique
  - Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_complete
  - Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_binding_sound
axiom_report:
  Whole-program theorems use [propext, Lean.ofReduceBool,
  Lean.trustCompiler, Quot.sound] because the 6,661-entry structural
  certificate is discharged with `native_decide`. The local binding theorem
  uses only [propext, Quot.sound]. Both reports are guarded fail-closed.
proof_hash:
  Full row artifact cb9587540e95fe4cef093a9bfbe9957281a37144978cc4625ba69bd512a17b7b.
  Witness artifact f273609f24aedadfc97ced41aa4d4c47e29ba7d52f0bb273d850508dad6b9a8f.
conformance_status:
  Artifact-checked for exact row functionality, public-output uniqueness, and
  completeness for this fixed plain profile. CIR-SOUND remains open on the
  other exact step owners and their composed decoding into Step.LocalHolds.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cargo test -p neo-fold-clean --release --test gadgets_f_prime_chunk_digest_lean_artifact
  - cargo test -p neo-fold-clean --release --test f_prime_digest_circuit
```
