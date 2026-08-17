import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder
import tests.Axioms.Support

/-! Dependency audit for the production PiRLC family-body decoder. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder.dimensions_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms dimensions_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder.certificate_input_lengths_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms certificate_input_lengths_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder.even_column_census_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms even_column_census_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder.odd_column_census_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms odd_column_census_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder.even_valid' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms even_valid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder.odd_valid' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms odd_valid
