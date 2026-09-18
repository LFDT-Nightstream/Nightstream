# PARAM-GLOBAL — production verifier-owned parameter regime

```text
property_id: PARAM-GLOBAL
claim:
  The production Appendix-B.2 profile fixes q, b, k, B, K_max, T, eta, d,
  kappa, extension degree, and security bits. Definition 14 holds at K_max
  and therefore at every K <= K_max. Norm stages and the 8*T*B MSIS regime
  are derived from this verifier-owned profile, never statement fields.
assumptions:
  - These constants are the selected production deployment profile.
non_goals:
  - Proof of Module-SIS hardness, strong-sampling-set negligibility, or the
    Rust parameter parser/refinement theorem.
paper_sources:
  - SuperNeo Definition 14 and Appendix B.2.
rust_surfaces:
  - crates/neo-params/src/lib.rs (goldilocks_paper_b2)
  - crates/neo-fold-clean/src/paper/params.rs (Params::production)
circuit_or_encoding_artifacts:
  - Rust/Lean literal golden vector in tests/system/production_params.rs and
    tests/ConcreteRelations.lean.
failure_class:
  A prover selects its own norm or fold arity, or a deployment admits K for
  which (K+k)T(b-1) >= B.
counterexample_or_witness:
  A K above 61 is outside the typed production theorem. Every K <= 61 inherits
  the stored maximum-arity inequality.
lean_theorems:
  - Nightstream.SuperNeo.GlobalParams.rlc_bound_for
  - Nightstream.SuperNeo.Concrete.production_parameter_values
  - Nightstream.SuperNeo.Concrete.production_norm_stages
  - Nightstream.SuperNeo.Concrete.production_msis_norm_bound
axiom_report:
  rlc_bound_for does not depend on any axioms, guarded fail-closed.
proof_hash:
  sha256:2d157a72f791398b4f5c307737b41874ef6d9b6978e0ae9aca61155e5098794e
conformance_status:
  model-proved with a Rust literal golden-vector regression. This does not
  promote the full Rust verifier to rust-conformant.
retest_commands:
  - cargo test -p neo-fold-clean --release --test system_production_params
  - cd formal/nightstream-lean && lake build && lake exe check
```
