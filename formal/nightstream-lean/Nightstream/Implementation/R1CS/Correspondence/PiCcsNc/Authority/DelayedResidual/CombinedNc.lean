import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc

/-! Compatibility import for the relocated implementation-free combined-NC
semantics. New protocol code imports the lower-layer owner directly. -/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc

export Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc
  (RunningWeights authoritativeRunningValueAt authoritativeRunningValueAt_live
    betaPowerSelector delayedAtPoint combinedAtPoint
    authoritativeRunningProjection delayedHypercubeSum combinedHypercubeSum
    delayedHypercubeSum_eq_weightedProjection
    combinedHypercubeSum_eq_ordinary_add_weightedProjection
    delayedTerminalRhs delayedAtPoint_eq_terminalRhs terminalFromMessage
    combinedAtPoint_eq_terminalFromMessage_of_bound
    delayedAtPoint_eq_zero_of_batchWeight_eq_zero
    combinedAtPoint_eq_ordinary_of_batchWeight_eq_zero
    delayedAtPoint_block_quadratic delayedAtPoint_lane_quadratic
    combinedAtPoint_block_quartic combinedAtPoint_lane_quartic
    sumcheckPolynomial sumcheckPolynomial_slice_quartic
    expectedRound_quartic expectedRound_has_five_coefficients)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
