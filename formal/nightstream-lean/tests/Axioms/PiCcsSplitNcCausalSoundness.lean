import Nightstream.Protocol.FPrime.Frozen
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for model-level production Split-NC
causal SumCheck soundness. This does not claim Fiat--Shamir or production
field instantiation.
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.CausalFixedPhase.detects_probability_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.CausalFixedPhase.detects_probability_le

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.CausalFixedPhase.badChallenge_implies_detects' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.CausalFixedPhase.badChallenge_implies_detects

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.feRoundRepresentable' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.feRoundRepresentable

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.fe_uniformRounds_eq_generated' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.fe_uniformRounds_eq_generated

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.fe_roundCollision_implies_detects' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.fe_roundCollision_implies_detects

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.nc_roundCollision_implies_detects' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.nc_roundCollision_implies_detects

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.split_detects_probability_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.split_detects_probability_le

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.CausalSoundness.rawRoundRepresentable' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.rawRoundRepresentable

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.CausalSoundness.splitCollision_implies_detects' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.splitCollision_implies_detects

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.CausalSoundness.splitCollision_probability_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.splitCollision_probability_le

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.accepted_implies_paper_or_algebraic_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.accepted_implies_paper_or_algebraic_failure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.not_transcriptFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.not_transcriptFailure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.not_bindingFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.not_bindingFailure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.blockLaneCombinedNc_refines_paperNc' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.blockLaneCombinedNc_refines_paperNc

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.everyCoordinate_has_exact_owner' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.everyCoordinate_has_exact_owner

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.delayedProjection_refines_rawRecomposition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.delayedProjection_refines_rawRecomposition

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.honest_complete_with_output' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.honest_complete_with_output

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.accepted_output_suitable_for_piRlc' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.accepted_output_suitable_for_piRlc

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Lifecycle.Trace.base_owns_no_predecessor' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.base_owns_no_predecessor

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Lifecycle.Trace.edge_owns_production_and_consumption' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.edge_owns_production_and_consumption

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Lifecycle.Trace.terminal_owns_discharge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.terminal_owns_discharge

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Lifecycle.Trace.terminalCount_eq_one' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.terminalCount_eq_one

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Lifecycle.Trace.closedTrace_reduces_to_paper_transitions_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.closedTrace_reduces_to_paper_transitions_or_named_failure
