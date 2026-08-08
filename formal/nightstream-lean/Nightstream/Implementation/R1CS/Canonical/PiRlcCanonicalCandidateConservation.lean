import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidate

/-!
Contract: categorical column conservation for one canonical `Pi_RLC`
candidate occurrence.

Every row operand is the shared constant wire, one of the sixteen declared
source-bit reads, a column mentioned by the declared prior-count expression,
or one of the occurrence's exact 22 allocated columns.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateConservation

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidate
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

def Allowed (layout : Layout) (column : Nat) : Prop :=
  column = 0 ∨
    (∃ index : Fin sourceBitCount, column = layout.sourceBit index) ∨
    Mentions layout.prior column ∨
    column ∈ allocation layout

private def CombAllowed (layout : Layout) (comb : LinComb) : Prop :=
  ∀ column, Mentions comb column → Allowed layout column

private def RowAllowed (layout : Layout) (row : Row) : Prop :=
  CombAllowed layout row.a ∧
    CombAllowed layout row.b ∧
      CombAllowed layout row.c

private theorem allowed_constant (layout : Layout) :
    Allowed layout 0 :=
  Or.inl rfl

private theorem allowed_allocated
    (layout : Layout) (column : Nat)
    (lower : layout.base ≤ column)
    (upper : column < layout.base + auxiliaryCount) :
    Allowed layout column := by
  right
  right
  right
  exact (allocation_mem_iff layout column).mpr ⟨lower, upper⟩

private theorem combAllowed_nil (layout : Layout) :
    CombAllowed layout [] := by
  intro column mentioned
  simp [Mentions] at mentioned

private theorem combAllowed_single
    (layout : Layout) (column coefficient : Nat)
    (allowed : Allowed layout column) :
    CombAllowed layout [(column, coefficient)] := by
  intro target mentioned
  have equal := (mentions_single column target coefficient).mp mentioned
  subst target
  exact allowed

private theorem combAllowed_append
    (layout : Layout) (left right : LinComb)
    (leftAllowed : CombAllowed layout left)
    (rightAllowed : CombAllowed layout right) :
    CombAllowed layout (left ++ right) := by
  intro column mentioned
  rw [mentions_append] at mentioned
  exact mentioned.elim (leftAllowed column) (rightAllowed column)

private theorem combAllowed_pair
    (layout : Layout)
    (leftColumn leftCoefficient rightColumn rightCoefficient : Nat)
    (leftAllowed : Allowed layout leftColumn)
    (rightAllowed : Allowed layout rightColumn) :
    CombAllowed layout
      [(leftColumn, leftCoefficient), (rightColumn, rightCoefficient)] :=
  combAllowed_append layout
    [(leftColumn, leftCoefficient)] [(rightColumn, rightCoefficient)]
    (combAllowed_single layout leftColumn leftCoefficient leftAllowed)
    (combAllowed_single layout rightColumn rightCoefficient rightAllowed)

private theorem chunkTerms_allowed (layout : Layout) :
    CombAllowed layout (chunkTerms layout) := by
  unfold chunkTerms
  apply combAllowed_append
  · exact combAllowed_single layout 0 rejectionBucket
      (allowed_constant layout)
  · intro column mentioned
    unfold Mentions at mentioned
    rw [List.map_map] at mentioned
    change
      column ∈ (List.finRange sourceBitCount).map layout.sourceBit at mentioned
    rcases List.mem_map.mp mentioned with ⟨index, _, rfl⟩
    exact Or.inr (Or.inl ⟨index, rfl⟩)

private theorem quotientTerms_allowed (layout : Layout) :
    CombAllowed layout (quotientTerms layout) := by
  intro column mentioned
  unfold quotientTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change
    column ∈
      (List.range quotientBitCount).map
        (quotientBitColumn layout) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, inRange, rfl⟩
  have bounded := List.mem_range.mp inRange
  apply allowed_allocated
  · unfold quotientBitColumn
    omega
  · simp only [quotientBitColumn, quotientBitCount, auxiliaryCount] at *
    omega

private theorem prior_allowed (layout : Layout) :
    CombAllowed layout layout.prior := by
  intro column mentioned
  exact Or.inr (Or.inr (Or.inl mentioned))

private theorem differenceTerms_allowed (layout : Layout) :
    CombAllowed layout (differenceTerms layout) := by
  unfold differenceTerms
  apply combAllowed_append
  · exact chunkTerms_allowed layout
  · exact combAllowed_single layout 0 (goldilocksP - rejectionBucket)
      (allowed_constant layout)

private theorem accept_allowed (layout : Layout) :
    Allowed layout (acceptColumn layout) := by
  apply allowed_allocated
  · unfold acceptColumn
    omega
  · simp only [acceptColumn, auxiliaryCount]
    omega

private theorem inverse_allowed (layout : Layout) :
    Allowed layout (inverseColumn layout) := by
  apply allowed_allocated
  · unfold inverseColumn
    omega
  · simp only [inverseColumn, auxiliaryCount]
    omega

private theorem residue_allowed (layout : Layout) :
    Allowed layout (residueColumn layout) := by
  apply allowed_allocated
  · unfold residueColumn
    omega
  · simp only [residueColumn, auxiliaryCount]
    omega

private theorem quotient_allowed (layout : Layout) :
    Allowed layout (quotientColumn layout) := by
  apply allowed_allocated
  · unfold quotientColumn
    omega
  · simp only [quotientColumn, auxiliaryCount]
    omega

private theorem product_allowed (layout : Layout) (stage : Nat)
    (bounded : stage < 3) :
    Allowed layout (productColumn layout stage) := by
  apply allowed_allocated
  · unfold productColumn
    omega
  · simp only [productColumn, auxiliaryCount] at *
    omega

private theorem quotientBit_allowed
    (layout : Layout) (offset : Nat) (bounded : offset < quotientBitCount) :
    Allowed layout (quotientBitColumn layout offset) := by
  apply allowed_allocated
  · unfold quotientBitColumn
    omega
  · simp only [quotientBitColumn, quotientBitCount, auxiliaryCount] at *
    omega

private theorem cumulative_allowed (layout : Layout) :
    Allowed layout (cumulativeColumn layout) := by
  apply allowed_allocated
  · unfold cumulativeColumn
    omega
  · simp only [cumulativeColumn, auxiliaryCount]
    omega

private theorem oneMinusAccept_allowed (layout : Layout) :
    CombAllowed layout (oneMinusAccept layout) :=
  combAllowed_pair layout
    (acceptColumn layout) (goldilocksP - 1) 0 1
    (accept_allowed layout) (allowed_constant layout)

private theorem bitRow_allowed
    (layout : Layout) (column : Nat)
    (allowed : Allowed layout column) :
    RowAllowed layout (bitRow column) := by
  unfold RowAllowed bitRow
  exact
    ⟨combAllowed_single layout column 1 allowed,
      combAllowed_pair layout column 1 0 (goldilocksP - 1)
        allowed (allowed_constant layout),
      combAllowed_nil layout⟩

private theorem acceptanceRows_allowed
    (layout : Layout) (row : Row)
    (member : row ∈ acceptanceRows layout) :
    RowAllowed layout row := by
  simp only [acceptanceRows, List.mem_cons, List.not_mem_nil, or_false]
    at member
  rcases member with rfl | rfl | rfl | rfl
  · exact bitRow_allowed layout _ (accept_allowed layout)
  · exact
      ⟨oneMinusAccept_allowed layout, differenceTerms_allowed layout,
        combAllowed_nil layout⟩
  · exact
      ⟨differenceTerms_allowed layout,
        combAllowed_single layout _ 1 (inverse_allowed layout),
        combAllowed_single layout _ 1 (accept_allowed layout)⟩
  · exact
      ⟨oneMinusAccept_allowed layout,
        combAllowed_single layout _ 1 (inverse_allowed layout),
        combAllowed_nil layout⟩

private theorem residueRangeRows_allowed
    (layout : Layout) (row : Row)
    (member : row ∈ residueRangeRows layout) :
    RowAllowed layout row := by
  simp only [residueRangeRows, List.mem_cons, List.not_mem_nil, or_false]
    at member
  rcases member with rfl | rfl | rfl | rfl
  · exact
      ⟨combAllowed_single layout _ 1 (residue_allowed layout),
        combAllowed_pair layout _ 1 0 (goldilocksP - 1)
          (residue_allowed layout) (allowed_constant layout),
        combAllowed_single layout _ 1 (product_allowed layout 0 (by omega))⟩
  · exact
      ⟨combAllowed_single layout _ 1 (product_allowed layout 0 (by omega)),
        combAllowed_pair layout _ 1 0 (goldilocksP - 2)
          (residue_allowed layout) (allowed_constant layout),
        combAllowed_single layout _ 1 (product_allowed layout 1 (by omega))⟩
  · exact
      ⟨combAllowed_single layout _ 1 (product_allowed layout 1 (by omega)),
        combAllowed_pair layout _ 1 0 (goldilocksP - 3)
          (residue_allowed layout) (allowed_constant layout),
        combAllowed_single layout _ 1 (product_allowed layout 2 (by omega))⟩
  · exact
      ⟨combAllowed_single layout _ 1 (product_allowed layout 2 (by omega)),
        combAllowed_pair layout _ 1 0 (goldilocksP - 4)
          (residue_allowed layout) (allowed_constant layout),
        combAllowed_nil layout⟩

private theorem quotientBitRows_allowed
    (layout : Layout) (row : Row)
    (member : row ∈ quotientBitRows layout) :
    RowAllowed layout row := by
  rcases List.mem_map.mp member with ⟨offset, inRange, rfl⟩
  exact bitRow_allowed layout _
    (quotientBit_allowed layout offset (List.mem_range.mp inRange))

private theorem quotientRecompositionRow_allowed (layout : Layout) :
    RowAllowed layout (quotientRecompositionRow layout) :=
  ⟨combAllowed_single layout _ 1 (quotient_allowed layout),
    combAllowed_single layout 0 1 (allowed_constant layout),
    quotientTerms_allowed layout⟩

private theorem decompositionRow_allowed (layout : Layout) :
    RowAllowed layout (decompositionRow layout) :=
  ⟨chunkTerms_allowed layout,
    combAllowed_single layout 0 1 (allowed_constant layout),
    combAllowed_pair layout _ 5 _ 1
      (quotient_allowed layout) (residue_allowed layout)⟩

private theorem cumulativeRow_allowed (layout : Layout) :
    RowAllowed layout (cumulativeRow layout) :=
  ⟨combAllowed_single layout _ 1 (cumulative_allowed layout),
    combAllowed_single layout 0 1 (allowed_constant layout),
    combAllowed_append layout layout.prior [(acceptColumn layout, 1)]
      (prior_allowed layout)
      (combAllowed_single layout _ 1 (accept_allowed layout))⟩

/-- Every operand of every emitted candidate row belongs to one explicit
source or allocation class. -/
theorem rows_conservation
    (layout : Layout) (row : Row) (rowMember : row ∈ rows layout)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    Allowed layout column := by
  simp only [rows, List.mem_append] at rowMember
  rcases rowMember with ((inHead | inBits) | inFinal)
  · rcases inHead with inAcceptance | inResidue
    · have allowed := acceptanceRows_allowed layout row inAcceptance
      exact mentioned.elim (allowed.1 column)
        (fun side => side.elim (allowed.2.1 column) (allowed.2.2 column))
    · have allowed := residueRangeRows_allowed layout row inResidue
      exact mentioned.elim (allowed.1 column)
        (fun side => side.elim (allowed.2.1 column) (allowed.2.2 column))
  · have allowed := quotientBitRows_allowed layout row inBits
    exact mentioned.elim (allowed.1 column)
      (fun side => side.elim (allowed.2.1 column) (allowed.2.2 column))
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at inFinal
    rcases inFinal with rfl | rfl | rfl
    · have allowed := quotientRecompositionRow_allowed layout
      exact mentioned.elim (allowed.1 column)
        (fun side => side.elim (allowed.2.1 column) (allowed.2.2 column))
    · have allowed := decompositionRow_allowed layout
      exact mentioned.elim (allowed.1 column)
        (fun side => side.elim (allowed.2.1 column) (allowed.2.2 column))
    · have allowed := cumulativeRow_allowed layout
      exact mentioned.elim (allowed.1 column)
        (fun side => side.elim (allowed.2.1 column) (allowed.2.2 column))

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateConservation
