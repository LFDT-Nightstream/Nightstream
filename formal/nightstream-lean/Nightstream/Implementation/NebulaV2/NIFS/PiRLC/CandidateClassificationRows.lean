import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.TranscriptRows
import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.FullFieldCandidateRows

/-!
Contract: exact indexed classification rows for all V2 PiRLC candidates.

The 2,430 occurrences follow the transcript's source-major,
coefficient-major, attempt-minor order. Each occurrence reads its matching
full-field candidate expression and owns a disjoint 78-column window after
the complete sampler-transcript allocation.

The family is indexed instead of materialized as one large Lean list.
`RowsHold` means that every exact occurrence is satisfied.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcCandidateClassificationRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductPiRlcFullFieldCandidateRows

abbrev CandidateIndex := ProductPiRlcTranscriptRows.CandidateIndex

/-- The complete transcript window ends before classification allocations. -/
def classificationStart (input : ProductPiRlcTranscriptRows.Input) : Nat :=
  input.transcriptBase + ProductPiRlcTranscriptRows.aggregateRowCount

def candidateBase
    (input : ProductPiRlcTranscriptRows.Input) (index : CandidateIndex) : Nat :=
  classificationStart input + index.flat * auxiliaryCount

def layout
    (input : ProductPiRlcTranscriptRows.Input) (index : CandidateIndex) : Layout where
  base := candidateBase input index
  candidate := ProductPiRlcTranscriptRows.candidate input index

def rows
    (input : ProductPiRlcTranscriptRows.Input) (index : CandidateIndex) : List Row :=
  ProductPiRlcFullFieldCandidateRows.rows (layout input index)

theorem rows_length
    (input : ProductPiRlcTranscriptRows.Input) (index : CandidateIndex) :
    (rows input index).length = 89 :=
  ProductPiRlcFullFieldCandidateRows.rows_length (layout input index)

/-- Satisfaction of every exact classification occurrence. -/
def RowsHold
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat) : Prop :=
  forall index, Satisfies (rows input index) assignment

def aggregateRowCount : Nat :=
  ProductPiRlcTranscriptRows.candidateCount * 89

def aggregateAuxiliaryCount : Nat :=
  ProductPiRlcTranscriptRows.candidateCount * auxiliaryCount

theorem aggregateRowCount_eq : aggregateRowCount = 216270 := by decide

theorem aggregateAuxiliaryCount_eq :
    aggregateAuxiliaryCount = 189540 := by decide

theorem allocation_window
    (input : ProductPiRlcTranscriptRows.Input) (index : CandidateIndex)
    (column : Nat)
    (member : column ∈ allocation (layout input index)) :
    classificationStart input ≤ column ∧
      column < classificationStart input + aggregateAuxiliaryCount := by
  have localWindow :=
    (allocation_mem_iff (layout input index) column).mp member
  have flatLt := index.flat_lt
  simp only [layout, candidateBase, aggregateAuxiliaryCount] at localWindow ⊢
  norm_num [ProductPiRlcTranscriptRows.candidateCount,
    ProductPiRlcTranscriptRows.scalarCount,
    ProductPiRlcTranscriptRows.coefficientCount,
    ProductPiRlcTranscriptRows.attemptCount,
    auxiliaryCount] at flatLt localWindow ⊢
  omega

theorem allocations_disjoint
    (input : ProductPiRlcTranscriptRows.Input)
    (left right : CandidateIndex) (different : left ≠ right) :
    candidateBase input left + auxiliaryCount ≤ candidateBase input right ∨
      candidateBase input right + auxiliaryCount ≤ candidateBase input left := by
  have flatDifferent : left.flat ≠ right.flat := by
    intro equal
    exact different (ProductPiRlcTranscriptRows.CandidateIndex.flat_injective equal)
  norm_num [candidateBase, auxiliaryCount] at *
  omega

theorem starts_after_transcript
    (input : ProductPiRlcTranscriptRows.Input) :
    input.transcriptBase + ProductPiRlcTranscriptRows.aggregateRowCount =
      classificationStart input := rfl

end Nightstream.Implementation.NebulaV2.ProductPiRlcCandidateClassificationRows
