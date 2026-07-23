import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc

/-! Focused interface regression for the bounded running-`X` prefix decoder. -/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointPiCcsNcRawRunningDecoder

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

#check chunkOrdinal_bijective
#check Exact.generatedChunkExact
#check Exact.allocationAt_exact
#check Exact.coordinate_unique_chunkOwner
#check Exact.sourceArmColumn_injective
#check Exact.finalStart_injective
#check Exact.finalIntervals_nonoverlap
#check Exact.sourceRecord_eq_recordAt
#check ArtifactRefinement.rawRunningAssignments_decodedData
#check ArtifactRefinement.sourceAllocationRowsBind_decodedData
#check ArtifactRefinement.decodedScalar_eq_rawRunningAssignment
#check ArtifactRefinement.decodedVirtual_live_eq_rawRunningAssignment
#check ArtifactRefinement.decodedVirtual_paddingLane_zero
#check ArtifactRefinement.allocation_uniqueOwner
#check ArtifactRefinement.allocation_intervals_nonoverlap
#check Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.decodedRunning_live_eq_generatedAllocation
#check Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker.check_eq_true_iff_accepted
#check Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff
#check Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.check_eq_true_iff_accepted
#check Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.acceptedPair_implies_previousSemanticFold_or_badEvent
#check Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedPair_implies_previousSemanticFold_or_badEvent
#check Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedPair_of_stateChecks_implies_previousSemanticFold_or_badEvent
#check Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedTerminal_implies_semanticFold_or_badEvent
#check Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedPair_implies_previousSemanticFold_or_namedFailure
#check Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionBoundary.baseNcAccepted_iff

example : recordCount = 3780 /\
    chunkCount = 15 /\
    chunkLength = 252 /\
    chunkLength <= 256 := by
  exact ⟨profile_counts.1, profile_counts.2.1,
    profile_counts.2.2.1, profile_counts.2.2.2.2⟩

example : childCount = 14 /\
    logicalColumnCount = 270 /\
    packedLaneCount = 54 /\
    liveBlockCount = 5 /\
    virtualLaneCount - packedLaneCount = 10 := by
  decide

end Nightstream.Tests.FPrimeSelectiveFixedPointPiCcsNcRawRunningDecoder
