import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.Acceptance

/-!
Protocol-owned public surface for the block/lane combined-NC deviation.

Owns: the typed combined polynomial, its exact quartic SumCheck contract, the
raw-assignment terminal, and the fixed-production acceptance partition.

Does not own: one-fold lifecycle continuity, Rust/R1CS refinement, transcript
primitive internals, costs, or row removal.

Emits constraints: no.

Authority boundary: the underlying declarations live in their protocol and
SuperNeo owners. This facade is the only surface new protocol code should
import.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.block_lane_combined_nc.polynomial` | expose the raw-assignment combined polynomial, terminal identity, and quartic round contract | computed/derived | `DelayedCombinedNc` |
| `fprime.block_lane_combined_nc.projection` | expose the fixed-production weights and raw recomposition projection theorem | derived | `ProductionProjection` |
| `fprime.block_lane_combined_nc.acceptance` | expose message/raw acceptance and its exact paper-or-named-event partitions | checked/security partition | `DelayedCombinedNc.Acceptance`, `ProductionPiCcs` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc

export Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc
  (RunningWeights authoritativeRunningValueAt authoritativeRunningProjection
    sumcheckPolynomial delayedHypercubeSum_eq_weightedProjection
    combinedHypercubeSum_eq_ordinary_add_weightedProjection
    combinedAtPoint_eq_terminalFromMessage_of_bound
    expectedRound_quartic expectedRound_has_five_coefficients
    combinedAtPoint_eq_ordinary_of_batchWeight_eq_zero)

export Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.Acceptance
  (ResidualWeightRoot accepted_implies_truth_and_parentProjection_or_badEvent)

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionProjection
  (productionWeights
    authoritativeRunningProjection_eq_projectedRawRecomposition)

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs
  (fePoint ncPoint ncTranscriptState rawInitial rawPolynomial messageTerminal
    NcMessageAccepted NcAccepted MessageAccepted Accepted BadEvent YRingUnbound
    YRingBound
    accepted_of_messageAccepted_and_packed
    ncAccepted_implies_truth_or_badEvent
    accepted_implies_paper_or_yRingUnbound_or_badEvent
    accepted_implies_paper_and_yRingBound_or_yRingUnbound_or_badEvent)

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc
