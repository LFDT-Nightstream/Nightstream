import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.Acceptance
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc

/-! Compatibility import for the relocated implementation-free combined-NC
acceptance theorem. New protocol code imports the lower-layer owner directly. -/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.Acceptance

export Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.Acceptance
  (sumcheckPolynomial_coordinates_eq_combinedAtPoint
    semanticInitial_eq_ordinary_add_weightedProjection residualWeightIdentity
    ResidualWeightRoot residualWeightIdentity_exact_iff
    expectedRoundsRepresentable
    accepted_implies_truth_and_parentProjection_or_badEvent)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.Acceptance
