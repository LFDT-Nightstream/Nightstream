import tests.AxiomAudit
import NightstreamFPrime.Spec.Folding.Refresh.Bound
import NightstreamFPrime.Spec.Folding.Schedule

#audit_axioms NightstreamFPrime.Spec.Folding.Refresh.honest_inputs
#audit_axioms NightstreamFPrime.Spec.Folding.Refresh.production_countNormLaw
#audit_axioms NightstreamFPrime.Spec.Folding.Refresh.parent_bounded
#audit_axioms NightstreamFPrime.Spec.Folding.Refresh.complete_bounded
#audit_axioms NightstreamFPrime.Spec.Folding.Schedule.depths_le_cap
#audit_axioms NightstreamFPrime.Spec.Folding.Schedule.accumulatorBound_succ
#audit_axioms NightstreamFPrime.Spec.Folding.Schedule.light_bound
#audit_axioms NightstreamFPrime.Spec.Folding.Schedule.honest_bounds
#audit_axioms NightstreamFPrime.Spec.Folding.Schedule.recoveredBound_le
#audit_axioms NightstreamFPrime.Spec.Folding.Schedule.recoveredBounds_le
#audit_axioms NightstreamFPrime.Spec.Folding.Schedule.msisChainBound_zero
#audit_axioms NightstreamFPrime.Spec.Folding.Schedule.production_cap_fits
#audit_axioms NightstreamFPrime.Spec.Folding.Schedule.production_msisChainBound
