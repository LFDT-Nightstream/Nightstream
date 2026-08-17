import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger

/-! Focused checks for the normalized production PiRLC body row ledger. -/

namespace tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger

example : LedgerValid := ledger_valid

example :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger.ledger.rows =
      491046 :=
  dimensions_exact.2.2.2.1

example :
    rewriteInstanceCount
        Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger.ledger
        .poseidon2 = 3762 := by
  decide

example :
    rewriteEmittedRowCount
        Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger.ledger
        .poseidon2 = 323548 := by
  decide

end tests.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger
