import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedPublicInput

/-!
CCS relation transport across the ring-aligned F' public boundary.

Owns: insertion of the same thirteen zero coordinates into every CCS matrix
row; preservation of field dot products, matrix images, and row points; shape
preservation; and equivalence of concrete CCS satisfaction before and after
the public-boundary alignment.

Does not own: commitment-key transport, CE evaluation/public-carrier
refinement, Π_CCS soundness, Rust/R1CS column maps, or constraint removal.

Emits constraints: no.

Authority boundary: this module proves a relation isomorphism. It does not use
an existing circuit, trace, digest, or measured row count as a premise.

| Branch | Mathematical obligation | Result | Assurance tier |
|---|---|---|---|
| row arithmetic | inserted zero coordinates do not change `row · z` | `dotF_insertPublicPadding` | kernel theorem |
| matrix image | every transformed `M z` equals the original image | `matrixVector_align` | kernel theorem |
| verifier shape | rows gain exactly 13 columns; all other structure is unchanged | `alignStructure_wellFormed` | kernel theorem |
| row semantics | transformed CCS row points equal original row points | `rowPoint_align` | kernel theorem |
| relation | transformed CCS accepts exactly the original satisfying assignments | `ccsSatisfied_align_iff` | kernel theorem |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCcsRelation

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput

/-- Matrix rows use the same coordinate insertion as assignments. -/
def alignRow (row : List F) : List F := insertPublicPadding row

def alignMatrix (matrix : Matrix) : Matrix := matrix.map alignRow

/-- Transform only the assignment width and matrix columns. The constraint
polynomial, row domain, and evaluation point domain remain unchanged. -/
def alignStructure (system : Structure) : Structure where
  matrices := system.matrices.map alignMatrix
  polynomial := system.polynomial
  rows := system.rows
  columns := system.columns + paddingWidth
  pointDimension := system.pointDimension

private theorem fadd_assoc (left middle right : F) :
    (left + middle) + right = left + (middle + right) := by
  apply Fin.ext
  simp only [Fin.val_add]
  rw [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

private theorem fzero_add (value : F) : 0 + value = value := by
  apply Fin.ext
  simp

private theorem dotF_append
    {leftPrefix leftSuffix rightPrefix rightSuffix : List F}
    (samePrefixLength : leftPrefix.length = rightPrefix.length) :
    dotF (leftPrefix ++ leftSuffix) (rightPrefix ++ rightSuffix) =
      dotF leftPrefix rightPrefix + dotF leftSuffix rightSuffix := by
  induction leftPrefix generalizing rightPrefix with
  | nil =>
      cases rightPrefix <;> simp_all [dotF]
  | cons leftHead leftTail inductionHypothesis =>
      cases rightPrefix with
      | nil => simp_all
      | cons rightHead rightTail =>
          simp only [List.length_cons, Nat.succ.injEq] at samePrefixLength
          simpa only [List.cons_append, dotF,
            inductionHypothesis samePrefixLength] using
            (fadd_assoc (leftHead * rightHead) (dotF leftTail rightTail)
              (dotF leftSuffix rightSuffix)).symm

private theorem dotF_publicPadding : dotF publicPadding publicPadding = 0 := by
  decide

/-- Simultaneously inserting the fixed-zero public coordinates into a matrix
row and its assignment preserves their scalar product. -/
theorem dotF_insertPublicPadding (left right : List F)
    (leftHasPublic : logicalPublicWidth ≤ left.length)
    (rightHasPublic : logicalPublicWidth ≤ right.length) :
    dotF (insertPublicPadding left) (insertPublicPadding right) =
      dotF left right := by
  have takeLengths :
      (left.take logicalPublicWidth).length =
        (right.take logicalPublicWidth).length := by
    have left257 : 257 ≤ left.length := by
      simpa [logicalPublicWidth] using leftHasPublic
    have right257 : 257 ≤ right.length := by
      simpa [logicalPublicWidth] using rightHasPublic
    simp [logicalPublicWidth, left257, right257]
  calc
    dotF (insertPublicPadding left) (insertPublicPadding right) =
        dotF (left.take logicalPublicWidth) (right.take logicalPublicWidth) +
          dotF (publicPadding ++ left.drop logicalPublicWidth)
            (publicPadding ++ right.drop logicalPublicWidth) := by
      simpa only [insertPublicPadding, List.append_assoc] using
        (dotF_append (leftSuffix := publicPadding ++ left.drop logicalPublicWidth)
          (rightSuffix := publicPadding ++ right.drop logicalPublicWidth)
          takeLengths)
    _ = dotF (left.take logicalPublicWidth) (right.take logicalPublicWidth) +
          (dotF publicPadding publicPadding +
            dotF (left.drop logicalPublicWidth)
              (right.drop logicalPublicWidth)) := by
      rw [dotF_append (show publicPadding.length = publicPadding.length from rfl)]
    _ = dotF (left.take logicalPublicWidth) (right.take logicalPublicWidth) +
          dotF (left.drop logicalPublicWidth)
            (right.drop logicalPublicWidth) := by
      rw [dotF_publicPadding, fzero_add]
    _ = dotF
          (left.take logicalPublicWidth ++ left.drop logicalPublicWidth)
          (right.take logicalPublicWidth ++ right.drop logicalPublicWidth) := by
      exact (dotF_append takeLengths).symm
    _ = dotF left right := by
      rw [List.take_append_drop, List.take_append_drop]

theorem alignRow_length (row : List F)
    (hasPublic : logicalPublicWidth ≤ row.length) :
    (alignRow row).length = row.length + paddingWidth := by
  exact insertPublicPadding_length row hasPublic

theorem alignMatrix_wellFormed
    {rows columns : Nat} {matrix : Matrix}
    (hasPublic : logicalPublicWidth ≤ columns)
    (wellFormed : MatrixWellFormed rows columns matrix) :
    MatrixWellFormed rows (columns + paddingWidth) (alignMatrix matrix) := by
  constructor
  · simpa [alignMatrix] using wellFormed.1
  · intro alignedRow alignedRowMember
    rcases List.mem_map.mp alignedRowMember with ⟨row, rowMember, rfl⟩
    have rowLength := wellFormed.2 row rowMember
    have rowHasPublic : logicalPublicWidth ≤ row.length := by
      rw [rowLength]
      exact hasPublic
    rw [alignRow_length row rowHasPublic, rowLength]

/-- Every transformed matrix image is definitionally the old image once the
row/assignment widths are verifier-owned. -/
theorem matrixVector_align (matrix : Matrix) (assignment : Assignment)
    (hasPublic : logicalPublicWidth ≤ assignment.length)
    (rowLengths : ∀ row ∈ matrix, row.length = assignment.length) :
    matrixVector (alignMatrix matrix) (insertPublicPadding assignment) =
      matrixVector matrix assignment := by
  simp only [matrixVector, alignMatrix, List.map_map]
  apply List.map_congr_left
  intro row rowMember
  simp only [Function.comp_apply, alignRow]
  exact dotF_insertPublicPadding row assignment
    (rowLengths row rowMember ▸ hasPublic) hasPublic

theorem alignStructure_wellFormed (system : Structure)
    (hasPublic : logicalPublicWidth ≤ system.columns)
    (wellFormed : system.WellFormed) :
    (alignStructure system).WellFormed := by
  rcases wellFormed with ⟨nonempty, matrices, polynomial, rowDomain⟩
  constructor
  · simpa [alignStructure] using nonempty
  constructor
  · intro alignedMatrix alignedMember
    rcases List.mem_map.mp alignedMember with ⟨matrix, matrixMember, rfl⟩
    exact alignMatrix_wellFormed hasPublic (matrices matrix matrixMember)
  constructor
  · simpa [alignStructure] using polynomial
  · simpa [alignStructure] using rowDomain

/-- The transformed relation evaluates every matrix to the same field vector,
so its polynomial row point is unchanged. -/
theorem rowPoint_align (system : Structure) (assignment : Assignment)
    (hasPublic : logicalPublicWidth ≤ system.columns)
    (assignmentLength : assignment.length = system.columns)
    (wellFormed : system.WellFormed) (row : Nat) :
    rowPoint (alignStructure system) (insertPublicPadding assignment) row =
      rowPoint system assignment row := by
  unfold rowPoint
  simp only [alignStructure, List.map_map]
  apply List.map_congr_left
  intro matrix matrixMember
  simp only [Function.comp_apply]
  have assignmentHasPublic : logicalPublicWidth ≤ assignment.length := by
    rw [assignmentLength]
    exact hasPublic
  have imagesEqual := matrixVector_align matrix assignment assignmentHasPublic (by
    intro matrixRow matrixRowMember
    have rowLength := (wellFormed.2.1 matrix matrixMember).2 matrixRow matrixRowMember
    exact rowLength.trans assignmentLength.symm)
  rw [imagesEqual]

/-- The zero-column transport preserves and reflects the concrete CCS
relation. This is the semantic authorization needed before a Rust compiler may
insert the aligned public block. -/
theorem ccsSatisfied_align_iff (system : Structure) (assignment : Assignment)
    (hasPublic : logicalPublicWidth ≤ system.columns)
    (assignmentLength : assignment.length = system.columns)
    (wellFormed : system.WellFormed) :
    ccsSatisfied (alignStructure system) (insertPublicPadding assignment) ↔
      ccsSatisfied system assignment := by
  have alignedWellFormed := alignStructure_wellFormed system hasPublic wellFormed
  have assignmentHasPublic : logicalPublicWidth ≤ assignment.length := by
    rw [assignmentLength]
    exact hasPublic
  have alignedLength :
      (insertPublicPadding assignment).length =
        (alignStructure system).columns := by
    rw [insertPublicPadding_length assignment assignmentHasPublic, assignmentLength]
    rfl
  constructor
  · intro accepted
    refine ⟨wellFormed, assignmentLength, ?_⟩
    intro row rowLt
    have rowAccepted := accepted.2.2 row rowLt
    rwa [rowPoint_align system assignment hasPublic assignmentLength wellFormed row]
      at rowAccepted
  · intro accepted
    refine ⟨alignedWellFormed, alignedLength, ?_⟩
    intro row rowLt
    rw [rowPoint_align system assignment hasPublic assignmentLength wellFormed row]
    exact accepted.2.2 row rowLt

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCcsRelation
