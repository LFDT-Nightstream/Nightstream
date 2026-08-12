import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.CandidateClassificationRows
import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.FirstAcceptedRows

/-!
Contract: exact indexed first-accepted selectors for all 15 x 54 V2 PiRLC
coefficients.

Each occurrence reads the three accept and residue outputs of its matching
candidate classifiers. Its eight-column allocation follows the complete
classification allocation. The indexed family avoids materializing a large
list while retaining exact row and column counts.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2

def sourceCount : Nat := ProductPiRlcTranscriptRows.scalarCount
def coefficientCount : Nat := ProductPiRlcTranscriptRows.coefficientCount
def attemptCount : Nat := ProductPiRlcTranscriptRows.attemptCount

structure CoordinateIndex where
  source : Fin sourceCount
  coefficient : Fin coefficientCount
deriving DecidableEq

def CoordinateIndex.flat (index : CoordinateIndex) : Nat :=
  index.source.val * coefficientCount + index.coefficient.val

def coordinateCount : Nat := sourceCount * coefficientCount

theorem coordinateCount_eq : coordinateCount = 810 := by decide

theorem CoordinateIndex.flat_lt (index : CoordinateIndex) :
    index.flat < coordinateCount := by
  have sourceLt := index.source.isLt
  have coefficientLt := index.coefficient.isLt
  norm_num [CoordinateIndex.flat, coordinateCount, sourceCount,
    coefficientCount, ProductPiRlcTranscriptRows.scalarCount,
    ProductPiRlcTranscriptRows.coefficientCount] at *
  omega

theorem CoordinateIndex.flat_injective :
    Function.Injective CoordinateIndex.flat := by
  intro left right equal
  have coefficientEqual : left.coefficient.val = right.coefficient.val := by
    have modulo := congrArg (fun value => value % coefficientCount) equal
    simpa [CoordinateIndex.flat, Nat.add_mod, Nat.mul_mod,
      Nat.mod_eq_of_lt left.coefficient.isLt,
      Nat.mod_eq_of_lt right.coefficient.isLt] using modulo
  have sourceEqual : left.source.val = right.source.val := by
    norm_num [CoordinateIndex.flat, coefficientCount,
      ProductPiRlcTranscriptRows.coefficientCount] at equal
    omega
  have sourceFin : left.source = right.source := Fin.ext sourceEqual
  have coefficientFin : left.coefficient = right.coefficient :=
    Fin.ext coefficientEqual
  cases left with
  | mk leftSource leftCoefficient =>
      cases right with
      | mk rightSource rightCoefficient =>
          simp only at sourceFin coefficientFin
          subst rightSource
          subst rightCoefficient
          rfl

def candidateIndex
    (coordinate : CoordinateIndex) (attempt : Fin attemptCount) :
    ProductPiRlcTranscriptRows.CandidateIndex where
  source := Fin.cast (by rfl) coordinate.source
  coefficient := Fin.cast (by rfl) coordinate.coefficient
  attempt := Fin.cast (by rfl) attempt

def selectionStart (input : ProductPiRlcTranscriptRows.Input) : Nat :=
  ProductPiRlcCandidateClassificationRows.classificationStart input +
    ProductPiRlcCandidateClassificationRows.aggregateAuxiliaryCount

def coordinateBase
    (input : ProductPiRlcTranscriptRows.Input) (index : CoordinateIndex) : Nat :=
  selectionStart input + index.flat *
    ProductPiRlcFirstAcceptedRows.auxiliaryCount

def layout
    (input : ProductPiRlcTranscriptRows.Input)
    (index : CoordinateIndex) : ProductPiRlcFirstAcceptedRows.Layout where
  base := coordinateBase input index
  accept := fun attempt =>
    ProductPiRlcFullFieldCandidateRows.acceptColumn
      (ProductPiRlcCandidateClassificationRows.layout input
        (candidateIndex index attempt))
  residue := fun attempt =>
    ProductPiRlcFullFieldCandidateRows.residueColumn
      (ProductPiRlcCandidateClassificationRows.layout input
        (candidateIndex index attempt))

def rows
    (input : ProductPiRlcTranscriptRows.Input) (index : CoordinateIndex) :
    List Row :=
  ProductPiRlcFirstAcceptedRows.rows (layout input index)

theorem rows_length
    (input : ProductPiRlcTranscriptRows.Input) (index : CoordinateIndex) :
    (rows input index).length = 9 :=
  ProductPiRlcFirstAcceptedRows.rows_length (layout input index)

def RowsHold
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat) : Prop :=
  forall index, Satisfies (rows input index) assignment

def aggregateRowCount : Nat := coordinateCount * 9
def aggregateAuxiliaryCount : Nat :=
  coordinateCount * ProductPiRlcFirstAcceptedRows.auxiliaryCount

theorem aggregateRowCount_eq : aggregateRowCount = 7290 := by decide
theorem aggregateAuxiliaryCount_eq : aggregateAuxiliaryCount = 6480 := by decide

theorem allocation_window
    (input : ProductPiRlcTranscriptRows.Input) (index : CoordinateIndex)
    (column : Nat)
    (member : column ∈ ProductPiRlcFirstAcceptedRows.allocation
      (layout input index)) :
    selectionStart input ≤ column ∧
      column < selectionStart input + aggregateAuxiliaryCount := by
  have localWindow :=
    (ProductPiRlcFirstAcceptedRows.allocation_mem_iff
      (layout input index) column).mp member
  have flatLt := index.flat_lt
  simp only [layout, coordinateBase, aggregateAuxiliaryCount] at localWindow ⊢
  norm_num [coordinateCount, sourceCount, coefficientCount,
    ProductPiRlcTranscriptRows.scalarCount,
    ProductPiRlcTranscriptRows.coefficientCount,
    ProductPiRlcFirstAcceptedRows.auxiliaryCount] at flatLt localWindow ⊢
  omega

theorem allocations_disjoint
    (input : ProductPiRlcTranscriptRows.Input)
    (left right : CoordinateIndex) (different : left ≠ right) :
    coordinateBase input left + ProductPiRlcFirstAcceptedRows.auxiliaryCount ≤
        coordinateBase input right ∨
      coordinateBase input right + ProductPiRlcFirstAcceptedRows.auxiliaryCount ≤
        coordinateBase input left := by
  have flatDifferent : left.flat ≠ right.flat := by
    intro equal
    exact different (CoordinateIndex.flat_injective equal)
  norm_num [coordinateBase, ProductPiRlcFirstAcceptedRows.auxiliaryCount] at *
  omega

end Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchRows
