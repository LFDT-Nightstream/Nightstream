import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetained
import tests.Axioms.Support

/-! Dependency audit for the normalized PiRLC algebra retained-row scan. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.reduced_product_nnz_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms reduced_product_nnz_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.nonzero_census_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms nonzero_census_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.decoder_prefix_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms decoder_prefix_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.retained_intervals_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms retained_intervals_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.audit_valid' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms audit_valid
