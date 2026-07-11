# CIR-FPR-TERMINAL-LINK

```text
property_id: CIR-FPR-TERMINAL-LINK
claim:
  For the plain one-claim terminal layout, every canonical-residue assignment
  satisfying the exact 257 rows emitted by enforce_terminal_latest_link has
  fresh.x[0]=1 and fresh.x[1+i]=last_x_out_bits[i] for all 0<=i<256. Empty
  batches, wrong fresh public-input length, and wrong x_out bit length are
  rejected by verifier-owned host checks before row emission.
assumptions:
  - Canonical assignment representatives and constant-one column zero.
  - Rust-to-Lean artifact extraction through the existing decider test
    isolation wrapper.
non_goals:
  - Canonicality of last_x_out_bits; ENC-CANON owns that producer property.
  - NIFS or terminal CE soundness.
  - Layouts with application/Nebula suffixes or more than one trailing claim;
    those require separately hashed artifacts, though the theorem pattern is
    pointwise identical.
paper_sources:
  - HyperNova Construction 2 trailing instance/compiler verification.
rust_surfaces:
  - crates/neo-fold-clean/src/engine/decider.rs
    (enforce_terminal_latest_link)
  - crates/neo-fold-clean/src/engine/decider_test_isolation.rs
    (enforce_terminal_latest_link_against)
circuit_or_encoding_artifacts:
  - FPrimeTerminalLinkArtifact.lean, schema 1, 257 rows, 514 columns.
failure_class:
  Trailing latest batch is omitted, has malformed public shape, omits the CCS
  affine-one slot, or is disconnected bitwise from the final producer x_out.
counterexample_or_witness:
  Rust and Lean twins reject affine-one mutation at row 0 and public bit 37 at
  row 38. Rust separately rejects empty and wrong-length containers.
lean_theorems:
  - Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.fPrimeTerminalLink_sound
axiom_report:
  Guarded fail-closed in tests/Axioms.lean.
proof_hash:
  Recorded in assurance/evidence-ledger.jsonl after final gates.
conformance_status:
  Artifact-checked for the plain one-claim terminal layout. Whole-trace
  CIR-SOUND remains open until producer-local, consumer, application, base-pin,
  state-link, and NIFS circuit properties compose.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cargo test -p neo-fold-clean --release --test gadgets_f_prime_terminal_link_lean_artifact
```
