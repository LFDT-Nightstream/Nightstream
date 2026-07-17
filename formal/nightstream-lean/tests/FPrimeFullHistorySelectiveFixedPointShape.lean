import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.FixedPointShape

/-! Focused theorem-surface checks for the model-level fixed-point shape contract. -/

namespace tests.FPrimeFullHistorySelectiveFixedPointShape

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Wire

#check RawHeader
#check RawMaterializedHeader
#check RawSnapshot
#check Refinement
#check Refinement.terminalInput_eq_materialized
#check Refinement.materialized_publicInputLength_eq_270
#check Refinement.materialized_columns_ring_aligned
#check Refinement.materialized_matrixCount_eq_13
#check Refinement.materialized_polynomialArity_eq_13
#check Refinement.materialized_polynomialTerms_eq
#check Refinement.materialized_rows_covered
#check Refinement.toProfile
#check Refinement.profile_shape_matrixCount_eq_13
#check Refinement.profile_shape_matrixCount_ne_three

example {raw : RawSnapshot} (refinement : Refinement raw) :
    raw.terminalInput = raw.materialized.verifier :=
  refinement.terminalInput_eq_materialized

example {raw : RawSnapshot} (refinement : Refinement raw) :
    raw.materialized.matrixCount = 13 :=
  refinement.materialized_matrixCount_eq_13

end tests.FPrimeFullHistorySelectiveFixedPointShape
