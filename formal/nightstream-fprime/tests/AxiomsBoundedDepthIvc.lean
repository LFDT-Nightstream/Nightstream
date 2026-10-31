import tests.AxiomAudit
import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth
import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge

open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth

#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.PerCircuit.transition_ofPaper_iff
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.PerCircuit.recursive_selected
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.accepts_iff
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.admissible_of_accepted
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.accepted_bounds
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.transition_depthStep
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.runDepths_lights
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.slot_cases
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.depthStep_light
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.depthStep_refresh
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.runDepths_refreshBlock
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.runDepths_refreshBlocks
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.runDepths_refreshPeriod
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.runDepths_refreshSchedule
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.count_flatten_replicate
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.refreshPeriod_count
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.runDepths_light_count
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth.light_count_le
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge.Runs.acceptedFrom
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge.ValidHistory.head
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge.reverse_extracts
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge.reverse_extracts_within
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge.slot_runs
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge.execution_extracts

/-- One refresh per period starves slot 1: it takes a light fold each time the light circuit
follows the refresh circuit and is never refreshed. -/
example : (runDepths 2 (fun _ => 0) 1 (List.replicate 3 [0, 0, 1]).flatten).isNone = true := by
  decide
