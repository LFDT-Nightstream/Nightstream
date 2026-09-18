# FPR-RECURSIVE

```text
property_id: FPR-RECURSIVE
claim:
  An accepted Active/Recursive step proves that the prior state is pinned, its
  carried accumulator digest is recomputed from the carried running value, its
  nonempty latest batch encodes the prior x_out, and the executable NIFS.V sees
  the typed prior-state/next-chunk transcript context and returns a specific
  next running accumulator. The step then installs a
  nonempty new batch, validates stateless/stateful application semantics and
  Nebula opening/advance, performs the exact counter/boundary/state advance,
  and recomputes the output x_out. The new batch's public link is deliberately
  deferred: the next recursive invocation checks it as its prior batch, or the
  terminal fold checks it when the batch is trailing. RecursiveLocalHolds is
  the standalone boundary; RecursiveHolds is the closed edge.
assumptions:
  - Environment supplies executable functions for NIFS.V, fresh-link decoding,
    application semantics, running/chunk digests, and Nebula verification. Its
    NIFS closure owns fixed verifier-key/structure context; the relation passes
    the varying prior public state and derived next chunk digest explicitly.
  - NIFS rejection is represented by none; success returns the unique
    verifier-computed nextRunning. There is no existential choice hidden in the
    checker.
non_goals:
  - A proof that the production Rust NIFS implementation is this function.
  - Generated F' R1CS correspondence or concrete enc_inst canonicality.
  - Re-proving the M2 folding reduction inside F'; M2 owns its relation and bad
    events, while M5 must refine the concrete NIFS.V call to that contract.
paper_sources:
  - HyperNova Construction 2 recursive steps 4-5.
  - SuperNeo PiCCS -> PiRLC -> PiDEC multi-fold contract.
rust_surfaces:
  - crates/neo-fold-clean/src/paper/f_prime/native.rs (Active/Recursive branch)
  - crates/neo-fold-clean/src/paper/nifs/verifier.rs
  - crates/neo-fold-clean/src/paper/construction2/transition.rs
  - crates/neo-fold-clean/src/paper/construction2/nebula_lane.rs
circuit_or_encoding_artifacts:
  - Exact enc_inst, counter, chunk-digest, Poseidon2, state-link, terminal-link,
    and CE-continuity subartifacts exist. The 1,824,444-row recursive NIFS
    program still requires generic family certificates; it is not represented
    by a fixed golden artifact.
failure_class:
  Wrong fold variant, stale/forged prior link, mismatched running digest, failed
  NIFS proof, empty new batch, invalid application or Nebula transition, forged
  state advance, output x_out mismatch, or a missing consumer/terminal link.
counterexample_or_witness:
  tests/FPrimeStep.lean contains one-coordinate rejections for every listed
  class, a replay-context mutation rejected by NIFS, and honest stateless and
  stateful executions.
lean_theorems:
  - Nightstream.Protocol.FPrime.Step.checkLocal_sound
  - Nightstream.Protocol.FPrime.Step.fPrimeRecursiveLocal_sound
  - Nightstream.Protocol.FPrime.Step.holds_iff_local_and_outgoing
  - Nightstream.Protocol.FPrime.Step.closeLocal
  - Nightstream.Protocol.FPrime.Step.check_sound
  - Nightstream.Protocol.FPrime.Step.check_eq_true_iff_holds
  - Nightstream.Protocol.FPrime.Step.fPrimeRecursive_sound
  - Nightstream.Protocol.FPrime.Step.next_state_pinned
  - Nightstream.Protocol.FPrime.Step.holds_advance_facts
axiom_report:
  Each theorem uses only [propext], guarded fail-closed in tests/Axioms.lean.
proof_hash:
  sha256:4066afda436f1182ed3e6729c3d94c2575afda3d5fd6038040e48312fb61bf2a
conformance_status:
  model-proved with the producer/consumer timing made explicit. Rust native
  verification and the standalone recursive R1CS own RecursiveLocalHolds;
  composed history/terminal rows must supply OutgoingLinked in M4. The
  executable semantic parameters are not yet rust-conformant.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cd formal/nightstream-lean && lake build tests.FPrimeStep tests.Axioms
  - perl -e 'alarm shift; exec @ARGV' 300 cargo test -p neo-fold-clean --release --test system_lifecycle_f_prime_link
```
