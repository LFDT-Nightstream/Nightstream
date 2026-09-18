# FPR-HASH

```text
property_id: FPR-HASH
claim:
  The canonical F' output is one domain-separated Poseidon2-family message over
  vk_fs_digest, the PiCCS header, chunk_count, step_count, pc, current boundary,
  the stateful semantic lane when present, the Construction-2 accumulator, and
  the optional Nebula marker/digest. Params, structure, public-input length,
  and initial semantic state are bound transitively by vk_fs_digest; z0 is the
  verifier-derived initial-boundary digest. For post-step states, public_trace
  equals the current boundary and the stateless semantic lane equals the
  accumulator. Equal x_out values therefore imply equality of the full pinned
  authority view, including the actual Nebula lane, or an explicit outer-hash
  or inner-Nebula-digest collision.
assumptions:
  - StatePinned holds for both states compared by the binding theorem.
  - Production will instantiate Semantics.hash and Semantics.nebulaDigest with
    the mapped Poseidon2 encodings. Collision resistance is a later
    cryptographic assumption; this theorem exposes both collision events and
    does not assume either away.
non_goals:
  - Concrete Poseidon2 permutation correctness or collision probability.
  - Rust byte/field serialization refinement or circuit correspondence.
  - Treating a digest as authority for running/latest content; Step separately
    recomputes the accumulator and invokes NIFS.V.
paper_sources:
  - HyperNova Construction 2 recursive-link hash in section 6.3.
rust_surfaces:
  - crates/neo-fold-clean/src/paper/digest.rs
    (vk_fs_digest, initial_boundary_digest, public_trace_seed_digest,
    state_x_out_digest_with_mode)
  - crates/neo-fold-clean/src/paper/construction2/transition.rs (compute_x_out)
  - crates/neo-fold-clean/src/paper/f_prime/digest_circuit.rs
circuit_or_encoding_artifacts:
  - CIR-POSEIDON2 proves exact permutation-row functionality.
  - CIR-FPR-CHUNK-BIND proves the complete fixed-profile chunk digest program.
  - CIR-FPR-BASE-PROGRAM includes the exact plain base x_out computation.
failure_class:
  A prover mutates a directly absorbed coordinate, relabels a verifier-derived
  coordinate, changes an omitted equality-pinned coordinate, switches semantic
  mode, or changes the optional Nebula lane while retaining the same x_out.
counterexample_or_witness:
  tests/FPrimeXOut.lean mutates every direct lane, demonstrates the stateless
  omission plus pinning rule, rejects z0/initial-state/public-trace relabeling,
  forces the outer-collision branch under a constant hash, and constructs an
  explicit collision at the inner Nebula compression boundary.
lean_theorems:
  - Nightstream.Protocol.FPrime.XOut.xOut_binding_or_collision
axiom_report:
  [propext, Classical.choice, Quot.sound], guarded fail-closed in
  tests/Axioms.lean. These are Lean foundations used by equality/case analysis;
  there is no protocol or collision-resistance axiom.
proof_hash:
  sha256:f42302115a7adb683f0120aa760c4a713e4724e98c457b4c72ffd0b9f87f82b7
conformance_status:
  model-proved with several fixed circuit profiles artifact-checked. General
  stateful/Nebula x_out profiles, native helper equality, serialization, and
  control-flow refinement remain pending.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cd formal/nightstream-lean && lake build tests.FPrimeXOut tests.Axioms
  - perl -e 'alarm shift; exec @ARGV' 300 cargo test -p neo-fold-clean --release --test system_lifecycle_f_prime_link
```
