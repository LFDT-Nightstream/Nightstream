import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation

/-! Focused checks for the separate padded-identity CE evaluation. -/

namespace tests.FPrimeCarrier270PaddedIdentityEvaluation

open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation

#check shape_carrierWidth
#check shape_blockCount
#check semanticShape_carrierWidth
#check paddedRowNumber
#check paddedRowNumber_eq_iff
#check decodeRow?
#check decodeRow?_eq_some_iff
#check paddedIdentityMatrix
#check system
#check paddedMatrixEntry_eq
#check expectedRowRing
#check rowRing_eq_expectedRowRing
#check matrixEvaluation_eq_expectedRows
#check asPackedAssignment
#check zeroLaneRow
#check rowNumber_zeroLaneRow
#check decodeRow?_zeroLaneRow_of_live
#check decodeRow?_zeroLaneRow_of_padding
#check expectedRowRing_zeroLane_of_live
#check expectedRowRing_zeroLane_of_padding
#check embeddedExpected_zeroLane_eq_blockRows
#check packedPoint
#check matrixEvaluation_packedPoint_eq_packedYZcol
#check claimedEvaluation_eq_packedYZcol_of_evaluationsBound
#check claimedEvaluation_eq_packedYZcol_of_ceHolds

end tests.FPrimeCarrier270PaddedIdentityEvaluation
