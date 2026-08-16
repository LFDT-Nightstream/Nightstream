import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger

/-! Focused checks for the normalized production PiRLC body row ledger. -/

namespace tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger

example : LedgerValid := ledger_valid

example :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger.ledger.rows =
      279089 :=
  dimensions_exact.2.2.2.1

example :
    rewriteInstanceCount
        Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger.ledger
        .poseidon2 = 1376 := by
  native_decide

example :
    rewriteEmittedRowCount
        Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger.ledger
        .poseidon2 = 118352 := by
  native_decide

end tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger
