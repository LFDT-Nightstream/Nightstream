# SUM-CLAIM / SUM-SOUND — semantic SumCheck assurance

```text
property_ids: SUM-CLAIM, SUM-SOUND
claim:
  SumCheck claim truth is equality between the prover's initial target and the
  independently supplied actual hypercube sum. The executable verifier checks
  only per-round degree bounds, p_i(0)+p_i(1)=target_i, challenge forwarding,
  and the terminal evaluation. Its Boolean result is equivalent to the logical
  Accepted predicate. If an accepted claim is false while the independent
  semantic path is correct, some bounded-degree claimed polynomial differs
  from the semantic polynomial but agrees at the sampled verifier challenge.
assumptions:
  - TruthPath is instantiated from the actual arithmetized polynomial and
    terminal evaluation; it is independent of the prover's claimed chain.
  - Classical reasoning is used to distinguish arbitrary functions and values.
non_goals:
  - The Schwartz-Zippel/root-counting probability theorem, strong-sampling-set
    distribution, Fiat-Shamir transform, or production transcript refinement.
  - A claim that verifier acceptance by itself establishes claim truth.
paper_sources:
  - SuperNeo section 7.3 and Appendix D.4.
  - SumCheck Definition 6 and the Lund/Schwartz-Zippel error boundary.
rust_surfaces:
  - crates/neo-fold-clean/src/paper/reductions/pi_ccs.rs (verify)
  - crates/neo-reductions/src/sumcheck.rs
circuit_or_encoding_artifacts:
  - none; this property is model-proved, not artifact-checked.
failure_class:
  A false initial sum passes because a different prover polynomial evaluates to
  the semantic polynomial at a verifier challenge.
counterexample_or_witness:
  tests/SumCheck.lean and tests/Folding.lean contain a false claim 8 versus
  actual sum 5. It passes the claimed chain only because both polynomials
  evaluate to 7 at challenge 2. A target 9 mutation is rejected.
lean_theorems:
  - Nightstream.SuperNeo.SumCheck.check_eq_true_iff_accepted
  - Nightstream.SuperNeo.SumCheck.complete
  - Nightstream.SuperNeo.SumCheck.false_acceptance_implies_bad_challenge
axiom_report:
  check_eq_true_iff_accepted depends on [propext, Quot.sound]. The false-
  acceptance reduction depends on [propext, Classical.choice, Quot.sound].
  Both are guarded fail-closed in tests/Axioms.lean.
proof_hash:
  sha256:376dd915038c1f8b9c549c0ef9065c1e749b3586e403fa6a0b0310297c804b4a
conformance_status:
  model-proved. The executable Lean probes run in lake exe check; production
  Rust transcript and probability refinement remain pending.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
```
