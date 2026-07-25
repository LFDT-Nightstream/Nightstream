import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalLinkDrift

/-!
Focused elaboration boundary for the full-history terminal-link drift
obstruction.
-/

namespace NightstreamTests.FPrimeFullHistoryTerminalLinkDrift

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift

example :
    FPrimeFullHistoryTerminalLink.rowCount = 1 + 256 :=
  generatedSnapshot_rowCount_eq_logicalWidth

example :
    FPrimeTerminalLink.rowCount = 1 + 256 + 13 :=
  currentPlainOwner_rowCount_eq_logicalPlusPadding

example :
    FPrimeFullHistoryTerminalLink.rowCount ≠ FPrimeTerminalLink.rowCount :=
  generatedSnapshot_ne_currentPlainOwner

example :
    FPrimeTerminalLink.rowCount -
        FPrimeFullHistoryTerminalLink.rowCount = 13 :=
  generatedSnapshot_missingPlainPaddingRows

example :
    FPrimeFullHistoryTerminalLink.rows ≠ FPrimeTerminalLink.rows :=
  generatedSnapshotRows_ne_currentPlainOwnerRows

end NightstreamTests.FPrimeFullHistoryTerminalLinkDrift
