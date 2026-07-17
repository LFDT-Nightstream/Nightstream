import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.PayloadRefinement

/-! Focused surface checks for model-level fixed-point payload placement. -/

namespace Tests.FPrimeFullHistorySelectiveCcsPayloadRefinement

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.PayloadRefinement

#check FixedSnapshot
#check PayloadBundle
#check Refinement
#check Refinement.decoded_matrixCount_eq_13
#check Refinement.decodedRelation_roleMatrix
#check Refinement.toRelation
#check Refinement.toStructure
#check Refinement.toStructure_constraintPolynomial
#check Refinement.toStructure_roleMatrix

end Tests.FPrimeFullHistorySelectiveCcsPayloadRefinement
