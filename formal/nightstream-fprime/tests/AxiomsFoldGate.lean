import tests.AxiomAudit
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate

/-! Axiom audits for the fold gate of Rust `Params::for_ccs_shape` (`FoldGate`). -/

#audit_axioms NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate.production_numerator
#audit_axioms NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate.error_eq
#audit_axioms NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate.count_pos
#audit_axioms NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate.bits_spec
#audit_axioms NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate.error_le_bits
#audit_axioms NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate.production_bits
#audit_axioms NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate.refresh_statisticalError_le
#audit_axioms NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate.light_statisticalError_le
#audit_axioms NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate.refresh_statisticalError_le_bits
#audit_axioms NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate.light_statisticalError_le_bits
