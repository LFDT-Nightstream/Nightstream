import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursiveAccumulatorCoreArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursiveAccumulatorRunningLinkArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursiveAccumulatorOutputLinkArtifact

/-! Exact aggregate for the recursive accumulator owner. Hashes are drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulator

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "465a73e16e64461e346c2a3370368ba5eec506d419775634bbda397c400dac11"
def rowStart : Nat := 887388
def rowEnd : Nat := 924691
def rowCount : Nat := 37303

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

def parentCeDigestColumns : List Nat :=
  FPrimeFullHistoryRecursiveAccumulatorCore.parentCeDigestColumns

def accumulatorDigestColumns : List Nat :=
  recomputedAccumulatorDigestColumns

def stateOutputAccumulatorDigestColumns : List Nat :=
  [924511, 924512, 924513, 924514]

theorem exact_owner_partition :
    FPrimeFullHistoryRecursiveAccumulatorRunningLink.rowStart = rowStart ∧
    FPrimeFullHistoryRecursiveAccumulatorRunningLink.rowEnd = 887392 ∧
    FPrimeFullHistoryRecursiveAccumulatorCore.rowStart = 887392 ∧
    FPrimeFullHistoryRecursiveAccumulatorCore.rowEnd = 924687 ∧
    FPrimeFullHistoryRecursiveAccumulatorOutputLink.rowStart = 924687 ∧
    FPrimeFullHistoryRecursiveAccumulatorOutputLink.rowEnd = rowEnd := by native_decide

theorem recomputed_is_core_output :
    recomputedAccumulatorDigestColumns =
      FPrimeFullHistoryRecursiveAccumulatorCore.accumulatorDigestColumns := by native_decide

theorem recomputed_is_state_output :
    recomputedAccumulatorDigestColumns =
      stateOutputAccumulatorDigestColumns := by native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulator
