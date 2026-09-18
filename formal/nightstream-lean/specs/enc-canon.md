property_id: ENC-CANON

claim: The F' recursive-link encoding accepts exactly four canonical Goldilocks lanes, represented as 256 little-endian bits, and the full CCS public input is the verifier-fixed affine-one coordinate followed by those bits. Raw byte and bit containers are length-checked. Encoding followed by decoding is the identity on accepted values, and equal accepted encodings imply equal field lanes. Every canonical-residue assignment satisfying the exact production encoding row program has those properties.

assumptions: Lean's standard `BitVec` semantics; canonical assignment representatives; the constant-one R1CS wire; the Euclid divisor property of the Goldilocks modulus used by the canonical-u64 row theorem; the Rust-to-Lean artifact exporter and drift test for source parity.

non_goals: Collision resistance of the digest whose lanes are encoded; correctness of Poseidon2; serialization formats outside the F' 32-byte digest and `[1 | enc_inst(x_out)]` boundary; full recursive F' circuit correspondence.

paper_sources: HyperNova Construction 2 step 4 and NIVC-compatible encoding requirement; SuperNeo Definition 12 low-norm public assignment requirement.

rust_surfaces: `paper/f_prime/r1cs.rs::{encode_x_out_public_bits,encode_f_prime_public_input,enforce_public_bits_encode_digest}`; `engine/decider.rs::canonical_digest32_fields`; `lifecycle/mod.rs::validate_semantic_state_digest_canonical`.

circuit_or_encoding_artifacts: `FPrimeEncodingArtifact.lean`, schema version 1, 532 rows, 525 columns; source anchor `enforce_public_bits_encode_digest`; Rust drift target `gadgets_f_prime_encoding_lean_artifact`.

failure_class: Noncanonical field alias, wrong byte/bit length, omitted affine-one coordinate, lane-order mismatch, or public bit disconnected from the decomposed digest.

counterexample_or_witness: A 255-bit body is rejected before row emission; a public bit flip in an otherwise honest assignment first fails equality row 69; a 64-bit lane equal to the Goldilocks modulus is rejected by canonical decoding.

lean_theorems: `Encoding.FPrime.digestBytes_roundtrip`, `Encoding.FPrime.digestBytes_injective`, `Encoding.FPrime.encInst_roundtrip`, `Encoding.FPrime.encInst_injective`, `Encoding.FPrime.encInst_bits_injective`, `FPrimeEncodingSound.fPrimeEncoding_sound`, `FPrimeEncodingSound.accepted_public_bits_injective`.

axiom_report: Recorded fail-closed in `tests/Axioms.lean`.

proof_hash: Recorded in `assurance/evidence-ledger.jsonl` after final gates.

conformance_status: Artifact row program and Lean negative witnesses pass; Rust exporter/drift gate must pass against the current builder before the property is reported `artifact-checked`.

retest_commands: `lake build`; `lake exe check`; `cargo test -p neo-fold-clean --release --test gadgets_f_prime_encoding_lean_artifact` (hard timeout 300 seconds).
