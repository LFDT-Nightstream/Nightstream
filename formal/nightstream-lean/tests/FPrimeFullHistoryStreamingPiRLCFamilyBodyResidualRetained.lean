import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetained

/-! Focused checks for the normalized PiRLC residual retained-row scan. -/

namespace tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained

example : audit.sourceRows = 108 := by decide
example : decoderCoherent := decoder_run_exact
example : rowLedgerCoherent := retained_intervals_exact
example : AuditValid := audit_valid

end tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained
