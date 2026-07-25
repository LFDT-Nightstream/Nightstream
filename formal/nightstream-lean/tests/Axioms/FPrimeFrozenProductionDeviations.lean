import Nightstream.Protocol.FPrime.Frozen
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for the model-proved block/lane combined-NC
and one-fold delayed packed-`yZcol` production deviation.

This file does not upgrade the deviation to Rust/R1CS conformance.
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.expectedRound_quartic' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.expectedRound_quartic

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionProjection.authoritativeRunningProjection_eq_projectedRawRecomposition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.authoritativeRunningProjection_eq_projectedRawRecomposition

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs.accepted_implies_paper_and_yRingBound_or_yRingUnbound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.accepted_implies_paper_and_yRingBound_or_yRingUnbound_or_badEvent

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.ChallengeAuthority.holds' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.holds

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Checker.check_eq_true_iff_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.check_eq_true_iff_accepted

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Checker.baseCheck_eq_true_iff_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.baseCheck_eq_true_iff_accepted

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.PaperStep.accepted_and_packed_implies_transition_or_yRingUnbound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.accepted_and_packed_implies_transition_or_yRingUnbound_or_badEvent

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Terminal.projectionOpeningAccepted_implies_packedYZcolBound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.projectionOpeningAccepted_implies_packedYZcolBound_or_badEvent

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Edge.acceptedPair_of_nextPacked_implies_previousClosed_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.acceptedPair_of_nextPacked_implies_previousClosed_or_failure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Trace.closedTrace_implies_baseAndAllClosed_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.closedTrace_implies_baseAndAllClosed_or_failure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.SemanticComposition.fold_extraction_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.fold_extraction_or_named_failure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.OutputAuthority.accepted_implies_paper_and_authority_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.accepted_implies_paper_and_authority_or_named_failure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment.not_literalResidualSlotAlignment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.not_literalResidualSlotAlignment

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment.accepted_implies_paper_or_residual_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.ProductionDeviations.BlockLaneCombinedNc.accepted_implies_paper_or_residual_failure
