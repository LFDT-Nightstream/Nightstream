import tests.AxiomAudit
import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth

open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth

#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.PerCircuit.transition_ofPaper_iff
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.PerCircuit.recursive_selected
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.accepts_iff
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.admissible_of_accepted
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.accepted_bounds
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.transition_depthStep
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.runDepths_lights
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.runDepths_refreshPeriod
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.runDepths_refreshSchedule

/-- One refresh per period starves slot 1: it takes a light fold each time the light circuit
follows the refresh circuit and is never refreshed. -/
example : (runDepths 2 (fun _ => 0) 1 (List.replicate 3 [0, 0, 1]).flatten).isNone = true := by
  decide
