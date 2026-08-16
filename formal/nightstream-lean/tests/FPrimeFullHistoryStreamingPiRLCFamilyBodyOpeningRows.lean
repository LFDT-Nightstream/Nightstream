import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRows

/-! Focused checks for the normalized PiRLC opening-row scan receipt. -/

namespace tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows

example : ExactReceipt := exact_receipt
example : auditedRowCount = 50707 := audited_row_count_exact
example := row_ledger_canonical_census_join
example := decoder_opening_template_join
example := generic_artifact_join

end tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows
