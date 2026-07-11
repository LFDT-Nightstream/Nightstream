import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreArtifact
import Nightstream.Implementation.R1CS.Relabel
import Nightstream.Implementation.R1CS.ShiftedTernary

/-! Exact classification of every recursive-accumulator-core assertion row. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage

open Nightstream.Implementation.R1CS

set_option maxRecDepth 1048576

def defaultRow : Row := ⟨[], [], []⟩

def columnMap
    (map : FPrimeFullHistoryRecursiveAccumulatorCore.ShiftedTernaryMap) :
    List Nat :=
  [0, map.fieldColumn] ++ List.replicate 56 0 ++
    map.digitColumns ++ map.negativeColumns ++ map.borrowColumns

def shiftedOwnerRows
    (map : FPrimeFullHistoryRecursiveAccumulatorCore.ShiftedTernaryMap) :
    List Row :=
  if map.rowStart < FPrimeFullHistoryRecursiveAccumulatorCore.segment1RowStart then
    FPrimeFullHistoryRecursiveAccumulatorCore.segment0Rows
  else FPrimeFullHistoryRecursiveAccumulatorCore.segment1Rows

def shiftedLocalRowStart
    (map : FPrimeFullHistoryRecursiveAccumulatorCore.ShiftedTernaryMap) : Nat :=
  if map.rowStart < FPrimeFullHistoryRecursiveAccumulatorCore.segment1RowStart then
    map.rowStart
  else map.rowStart - FPrimeFullHistoryRecursiveAccumulatorCore.segment1RowStart

def checkPattern0 : List Nat :=
  ((List.range 89).map (fun index => 0 + 1 * index)) ++
    [90] ++
    ((List.range 4).map (fun index => 92 + 1 * index)) ++
    ((List.range 4).map (fun index => 97 + 1 * index)) ++
    ((List.range 6).map (fun index => 102 + 1 * index)) ++
    [109, 112, 115, 116] ++
    ((List.range 4).map (fun index => 120 + 1 * index))

def checkPattern1 : List Nat :=
  ((List.range 82).map (fun index => 0 + 1 * index)) ++
    ((List.range 6).map (fun index => 83 + 1 * index)) ++
    [90] ++
    ((List.range 4).map (fun index => 92 + 1 * index)) ++
    ((List.range 4).map (fun index => 97 + 1 * index)) ++
    ((List.range 6).map (fun index => 102 + 1 * index)) ++
    [109, 112, 115, 116] ++
    ((List.range 4).map (fun index => 120 + 1 * index))

def checkPatterns : List (List Nat) := [checkPattern0, checkPattern1]

def checkPatternTags : List Nat :=
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def checksForMapIndex (mapIndex : Nat) : List Row :=
  let map := FPrimeFullHistoryRecursiveAccumulatorCore.shiftedTernaryMaps.getD mapIndex default
  let patternTag := checkPatternTags.getD mapIndex 0
  (checkPatterns.getD patternTag []).map fun rowIndex =>
    Relabel.row (columnMap map)
      (ShiftedTernaryCompiler.canonicalRows.getD rowIndex defaultRow)

theorem checkPatternTags_length :
    checkPatternTags.length =
      FPrimeFullHistoryRecursiveAccumulatorCore.shiftedTernaryMaps.length := by native_decide

theorem checkPatterns_bounded :
    ∀ pattern ∈ checkPatterns, ∀ rowIndex ∈ pattern,
      rowIndex < ShiftedTernaryCompiler.canonicalRows.length := by native_decide

def classifiedCheckCount : Nat :=
  (checkPatternTags.map fun tag => (checkPatterns.getD tag []).length).sum

def residualCheckCount : Nat := 0

theorem classification_count : classifiedCheckCount = 21438 := by native_decide

def segment0MapIndices : List Nat :=
  ((List.range 84).map (fun index => 0 + 1 * index))

def segment0ExpectedChecks : List Row :=
  segment0MapIndices.flatMap checksForMapIndex

theorem segment0_checks_covered :
    CheckedProgram.checks
        FPrimeFullHistoryRecursiveAccumulatorCore.segment0Instructions =
      segment0ExpectedChecks := by native_decide

def segment1MapIndices : List Nat :=
  ((List.range 108).map (fun index => 84 + 1 * index))

def segment1ExpectedChecks : List Row :=
  segment1MapIndices.flatMap checksForMapIndex

theorem segment1_checks_covered :
    CheckedProgram.checks
        FPrimeFullHistoryRecursiveAccumulatorCore.segment1Instructions =
      segment1ExpectedChecks := by native_decide

def segment2MapIndices : List Nat :=
  []

def segment2ExpectedChecks : List Row :=
  segment2MapIndices.flatMap checksForMapIndex

theorem segment2_checks_covered :
    CheckedProgram.checks
        FPrimeFullHistoryRecursiveAccumulatorCore.segment2Instructions =
      segment2ExpectedChecks := by native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage
