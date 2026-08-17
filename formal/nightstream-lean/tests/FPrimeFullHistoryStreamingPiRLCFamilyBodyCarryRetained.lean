import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetained

/-! Focused checks for the normalized PiRLC carry retained-row scan. -/

namespace tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained

example : audit.sourceRows = 1837 := by decide
example : decoderCoherent := decoder_run_exact
example : rowLedgerCoherent := retained_intervals_exact
example : AuditValid := audit_valid

end tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained
