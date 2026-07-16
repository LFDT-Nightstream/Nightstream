import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursiveAccumulatorCoreArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursiveAccumulatorRunningLinkArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursiveAccumulatorOutputLinkArtifact

/-! Exact aggregate for the recursive accumulator owner. Hashes are drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulator

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "1590f426113c2902f21e2e96d664b1b5bb67962401f27b4b9cd54a5925198500"
def rowStart : Nat := 863857
def rowEnd : Nat := 1118776
def rowCount : Nat := 254919

def runningDigestRows : List Row :=
  FPrimeFullHistoryRecursiveAccumulatorRunningLink.rows

def coreRows : List Row :=
  FPrimeFullHistoryRecursiveAccumulatorCore.rows

def outputDigestRows : List Row :=
  FPrimeFullHistoryRecursiveAccumulatorOutputLink.rows

def rowPieces : List (List Row) :=
  [runningDigestRows, coreRows, outputDigestRows]

def rows : List Row := rowPieces.flatten

theorem rows_length : rows.length = rowCount := by
  simp [rows, rowPieces, runningDigestRows, coreRows, outputDigestRows,
    FPrimeFullHistoryRecursiveAccumulatorRunningLink.rows_length,
    FPrimeFullHistoryRecursiveAccumulatorCore.rows_length,
    FPrimeFullHistoryRecursiveAccumulatorOutputLink.rows_length, rowCount]
  native_decide

def runningAccumulatorDigestColumns : List Nat :=
  FPrimeFullHistoryRecursiveAccumulatorRunningLink.pairs.map Prod.fst

def stateInputAccumulatorDigestColumns : List Nat :=
  FPrimeFullHistoryRecursiveAccumulatorRunningLink.pairs.map Prod.snd

def claimedAccumulatorDigestColumns : List Nat :=
  FPrimeFullHistoryRecursiveAccumulatorOutputLink.pairs.map Prod.fst

def recomputedAccumulatorDigestColumns : List Nat :=
  FPrimeFullHistoryRecursiveAccumulatorOutputLink.pairs.map Prod.snd

def accumulatorClaimSourceColumns : List Nat :=
  FPrimeFullHistoryRecursiveAccumulatorCore.accumulatorClaimSourceColumns

def accumulatorDigestColumns : List Nat :=
  recomputedAccumulatorDigestColumns

def stateOutputAccumulatorDigestColumns : List Nat :=
  [1127468, 1127469, 1127470, 1127471]

theorem exact_owner_partition :
    FPrimeFullHistoryRecursiveAccumulatorRunningLink.rowStart = rowStart ∧
    FPrimeFullHistoryRecursiveAccumulatorRunningLink.rowEnd = 863861 ∧
    FPrimeFullHistoryRecursiveAccumulatorCore.rowStart = 863861 ∧
    FPrimeFullHistoryRecursiveAccumulatorCore.rowEnd = 1118772 ∧
    FPrimeFullHistoryRecursiveAccumulatorOutputLink.rowStart = 1118772 ∧
    FPrimeFullHistoryRecursiveAccumulatorOutputLink.rowEnd = rowEnd := by native_decide

theorem recomputed_is_core_output :
    recomputedAccumulatorDigestColumns =
      FPrimeFullHistoryRecursiveAccumulatorCore.accumulatorDigestColumns := by native_decide

theorem recomputed_is_state_output :
    recomputedAccumulatorDigestColumns =
      stateOutputAccumulatorDigestColumns := by native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulator
