import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder

/-! Focused interface regression for the bounded fresh public-`X` decoder. -/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointPiCcsNcFreshSourceDecoder

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder

#check Exact.metadata_exact
#check Exact.chunk0_coordinateColumns_exact
#check Exact.chunk1_coordinateColumns_exact
#check Exact.chunk_lengths_and_coverage
#check Exact.chunk0_all_rangeValid
#check Exact.chunk1_all_rangeValid
#check Exact.record_formula
#check Exact.sourceColumn_has_uniqueLogicalOwner
#check Exact.sourceColumn_injective
#check Exact.records_all_wellFormed
#check Refinement.coordinateValueBindings_and_dataflow_imply_freshPublicInput

example : Generated.Chunk0.records.length = 256 /\
    Generated.Chunk1.records.length = 14 /\
    256 + 14 = logicalColumnCount :=
  Exact.chunk_lengths_and_coverage

example : sourceCount = 1 /\ logicalColumnCount = 270 := by
  decide

end Nightstream.Tests.FPrimeSelectiveFixedPointPiCcsNcFreshSourceDecoder
