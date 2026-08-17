import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetained

/-! Focused checks for the normalized PiRLC algebra retained-row scan. -/

namespace tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained

example : reducedProductNnz = 3996 := reduced_product_nnz_exact
example := decoder_prefix_exact
example := retained_intervals_exact
example :
    audit.sourceNnz = sourceNnzExpected ∧
      audit.finalPortNnz = finalPortNnzExpected :=
  nonzero_census_exact
example : AuditValid := audit_valid

end tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained
