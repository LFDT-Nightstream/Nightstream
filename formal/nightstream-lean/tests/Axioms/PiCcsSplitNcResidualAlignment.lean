import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment
import tests.Axioms.Support

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment.not_literalResidualSlotAlignment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment.not_literalResidualSlotAlignment

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment.semanticResidualsZero_iff_paperHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment.semanticResidualsZero_iff_paperHolds

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment.accepted_implies_paper_or_residual_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.ResidualAlignment.accepted_implies_paper_or_residual_failure
