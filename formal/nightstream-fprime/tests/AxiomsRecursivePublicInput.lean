import tests.AxiomAudit
import NightstreamFPrime.Lifecycle.XOut

/-! Axiom audits for the canonical recursive public-input encoding. -/

#audit_axioms NightstreamFPrime.Lifecycle.encHash_marker
#audit_axioms NightstreamFPrime.Lifecycle.encHash_digestBitNat
#audit_axioms NightstreamFPrime.Lifecycle.encHash_digestBit
#audit_axioms NightstreamFPrime.Lifecycle.encHash_tail
#audit_axioms NightstreamFPrime.Lifecycle.encHash_norm
#audit_axioms NightstreamFPrime.Lifecycle.decodeHashWord_encHash
#audit_axioms NightstreamFPrime.Lifecycle.decodeHash_encHash
#audit_axioms NightstreamFPrime.Lifecycle.encHash_injective_fixed
