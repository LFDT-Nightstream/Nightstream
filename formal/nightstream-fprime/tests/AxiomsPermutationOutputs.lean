import tests.AxiomAudit
import NightstreamFPrime.Gadgets.Poseidon2.Permutation

/-! Audit the executable projections that avoid rebuilding permutation recipes. -/

#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Permutation.scheduleOutput_eq_compile
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Permutation.fullSboxState_eq_fullSboxStateDirect
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Permutation.partialSboxState_eq_partialSboxStateDirect
