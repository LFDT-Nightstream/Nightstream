import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRows
import tests.Axioms.Support

/-! Dependency audit for the normalized PiRLC opening-row scan receipt. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows.exact_receipt' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms exact_receipt

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows.audited_row_count_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms audited_row_count_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows.row_ledger_canonical_census_join' does not depend on any axioms -/
#guard_msgs in
#audit_axioms row_ledger_canonical_census_join

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows.decoder_opening_template_join' does not depend on any axioms -/
#guard_msgs in
#audit_axioms decoder_opening_template_join

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows.generic_artifact_join' does not depend on any axioms -/
#guard_msgs in
#audit_axioms generic_artifact_join
