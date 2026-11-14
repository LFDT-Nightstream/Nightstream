import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression

/-! Axiom audits for the links of the compression argument to Rust (`Spec/Compression/Rust`,
`Spec/Compression/Verifier/Run.lean`). -/

#audit_axioms NightstreamFPrime.Spec.Compression.Verifier.game_costLe
#audit_axioms NightstreamFPrime.Spec.Compression.Verifier.run_soundness
#audit_axioms NightstreamFPrime.Spec.Compression.Rust.runCost_le_runError
#audit_axioms NightstreamFPrime.Spec.Compression.Rust.error_le_runError
#audit_axioms NightstreamFPrime.Spec.Compression.Rust.params_run_soundness
#audit_axioms NightstreamFPrime.Spec.Compression.Rust.nextMultiple_spec
#audit_axioms NightstreamFPrime.Spec.Compression.Rust.placeAll_spec
#audit_axioms NightstreamFPrime.Spec.Compression.Rust.lookup_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Rust.lookup_isSome
#audit_axioms NightstreamFPrime.Spec.Compression.Rust.lookup_slice
