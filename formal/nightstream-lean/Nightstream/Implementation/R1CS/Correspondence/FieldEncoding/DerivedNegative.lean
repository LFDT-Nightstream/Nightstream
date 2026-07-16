import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.NormDischarged

/-!
Contract: model-level elimination of shifted-ternary negative-indicator
witnesses after the authoritative SuperNeo `b = 2` norm is available.

Owns: the exact quadratic formula `n = d(d - 1) / 2` in Goldilocks, proof
that it reconstructs the unique negative indicator for a centered digit, and
equivalence between the conservative 82-row predicate and a 41-obligation borrow
predicate when the omitted indicators are reconstructed from the digits.

Does not own: a concrete substituted CCS gate, polynomial-degree model, R1CS product-wire lowering,
production row deletion, slot projection, Rust materialization, or proof that
the production verifier supplies the outer norm premise.

Emits constraints: no. `derivedBorrowSchedule` counts candidate semantic
gates; it is not an R1CS or generated CCS artifact.

Authority boundary: the digit alphabet comes only from the verifier-checked
outer norm. The negative indicator is derived from that digit and is never a
prover-authoritative field.

| Branch | Mathematical obligation | Conservative rows | Derived candidate | Tier |
|---|---|---:|---:|---|
| digit alphabet | `d` is centered under `normBounded 2` | 0 | 0 | kernel model |
| negative indicator | `2n = d(d-1)` | 41 | 0 | kernel model |
| borrow transition | radix-three canonicality | 41 R1CS | 41 substituted obligations | model candidate |
-/

namespace Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernaryComplete
open Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged

set_option maxRecDepth 262144

/-- Multiplicative inverse of two in the canonical Goldilocks residue range. -/
def inverseTwo : Nat := (goldilocksP + 1) / 2

theorem two_mul_inverseTwo_mod :
    2 * inverseTwo % goldilocksP = 1 := by
  native_decide

/-- Canonical field subtraction by one. -/
def fieldPred (value : Nat) : Nat :=
  (value + (goldilocksP - 1)) % goldilocksP

/-- The negative indicator as a quadratic function of a centered digit:
`d(d-1)/2` in Goldilocks. -/
def derivedNegative (value : Nat) : Nat :=
  value * fieldPred value * inverseTwo % goldilocksP

theorem derivedNegative_lt (value : Nat) :
    derivedNegative value < goldilocksP := by
  unfold derivedNegative
  exact Nat.mod_lt _ (by native_decide)

/-- On the norm-authorized alphabet, the quadratic formula is exactly the
existing semantic negative indicator. -/
theorem derivedNegative_eq_indicator
    {value : Nat} (centered : CenteredResidue value) :
    derivedNegative value = negativeIndicator value := by
  rcases centered with rfl | rfl | rfl <;>
    native_decide

theorem digit_negative_eq_indicator
    {value negative : Nat} (digit : Digit value negative) :
    negative = negativeIndicator value := by
  cases digit with
  | neg valueEq negativeEq => simp [negativeIndicator, valueEq, negativeEq]
  | zero valueEq negativeEq =>
      simp [negativeIndicator, valueEq, negativeEq, goldilocksP]
  | pos valueEq negativeEq =>
      simp [negativeIndicator, valueEq, negativeEq, goldilocksP]

/-- A conservative accepted opening cannot choose a different negative
indicator once the external norm fixes the centered digit. -/
theorem accepted_negative_eq_derived
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (norm : DigitNormBoundTwo assignment)
    (accepted : CenteredTernaryNormDischarged.Accepts assignment)
    {index : Nat} (indexLt : index < digitCount) :
    assignment (ShiftedTernary.negativeCols.getD index 0) =
      derivedNegative
        (assignment (ShiftedTernary.digitCols.getD index 0)) := by
  let digitColumn := ShiftedTernary.digitCols.getD index 0
  let negativeColumn := ShiftedTernary.negativeCols.getD index 0
  have bounded := norm index indexLt
  have centered : CenteredResidue (assignment digitColumn) :=
    normBoundTwo_iff_centeredResidue.mp bounded
  have gate : CenteredUnitGateHolds (assignment digitColumn) :=
    (centeredUnitGate_iff prime bounded.1).mpr centered
  have digit : Digit (assignment digitColumn) (assignment negativeColumn) :=
    digit_of_centeredUnit_and_definition prime canonical one gate
      (accepted.negativeDefinition index indexLt)
  calc
    assignment negativeColumn = negativeIndicator (assignment digitColumn) :=
      digit_negative_eq_indicator digit
    _ = derivedNegative (assignment digitColumn) :=
      (derivedNegative_eq_indicator centered).symm

/-- Artifact-checked contiguous source-column formula for the 41 digits. -/
theorem digitColumns_eq_range :
    ShiftedTernary.digitCols = List.range' 58 digitCount := by
  decide

theorem digitColumn_formula {index : Nat} (indexLt : index < digitCount) :
    ShiftedTernary.digitCols.getD index 0 = 58 + index := by
  rw [digitColumns_eq_range]
  simp [List.getD_eq_getElem?_getD, indexLt]

/-- Artifact-checked contiguous source-column formula for the 41 omitted
negative indicators. -/
theorem negativeColumns_eq_range :
    ShiftedTernary.negativeCols = List.range' 99 digitCount := by
  decide

theorem negativeColumn_formula {index : Nat} (indexLt : index < digitCount) :
    ShiftedTernary.negativeCols.getD index 0 = 99 + index := by
  rw [negativeColumns_eq_range]
  simp [List.getD_eq_getElem?_getD, indexLt]

/-- Reconstruct all omitted negative-indicator columns from a reduced
assignment. Columns outside the exact artifact interval `[99, 140)` are
preserved byte-for-byte. -/
def materializeNegatives (assignment : Nat → Nat) : Nat → Nat :=
  fun column =>
    if 99 ≤ column ∧ column < 140 then
      derivedNegative (assignment (column - 41))
    else
      assignment column

theorem materializeNegatives_outside
    {assignment : Nat → Nat} {column : Nat}
    (outside : column < 99 ∨ 140 ≤ column) :
    materializeNegatives assignment column = assignment column := by
  unfold materializeNegatives
  split <;> omega

theorem materializeNegatives_digit
    {assignment : Nat → Nat} {index : Nat}
    (indexLt : index < digitCount) :
    materializeNegatives assignment
        (ShiftedTernary.digitCols.getD index 0) =
      assignment (ShiftedTernary.digitCols.getD index 0) := by
  have columnEq := digitColumn_formula indexLt
  apply materializeNegatives_outside
  left
  rw [columnEq]
  simp [digitCount] at indexLt
  omega

theorem materializeNegatives_negative
    {assignment : Nat → Nat} {index : Nat}
    (indexLt : index < digitCount) :
    materializeNegatives assignment
        (ShiftedTernary.negativeCols.getD index 0) =
      derivedNegative
        (assignment (ShiftedTernary.digitCols.getD index 0)) := by
  have digitEq := digitColumn_formula indexLt
  have negativeEq := negativeColumn_formula indexLt
  rw [digitEq, negativeEq]
  unfold materializeNegatives
  have indexBound : index < 41 := by simpa [digitCount] using indexLt
  have interval : 99 ≤ 99 + index ∧ 99 + index < 140 := by omega
  rw [if_pos interval]
  congr 2
  omega

theorem materializeNegatives_one
    {assignment : Nat → Nat} :
    materializeNegatives assignment 0 = assignment 0 := by
  exact materializeNegatives_outside (Or.inl (by omega))

theorem materializeNegatives_canonical
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP) :
    ∀ column, materializeNegatives assignment column < goldilocksP := by
  intro column
  unfold materializeNegatives
  split
  · exact derivedNegative_lt _
  · exact canonical column

theorem materializeNegatives_norm
    {assignment : Nat → Nat}
    (norm : DigitNormBoundTwo assignment) :
    DigitNormBoundTwo (materializeNegatives assignment) := by
  intro index indexLt
  rw [materializeNegatives_digit indexLt]
  exact norm index indexLt

/-- One borrow transition after semantic substitution of all omitted
negative indicators. No polynomial AST or degree theorem is claimed here. -/
def DerivedBorrowHolds (assignment : Nat → Nat) (index : Nat) : Prop :=
  RowHolds (materializeNegatives assignment) (borrowRow index)

/-- Candidate 41-obligation predicate after eliminating negative-indicator columns. -/
def DerivedAccepts (assignment : Nat → Nat) : Prop :=
  ∀ index, index < digitCount → DerivedBorrowHolds assignment index

/-- The omitted columns, if retained in an old-layout assignment, are uniquely
reconstructed rather than supplied by the prover. -/
def NegativesMaterialized (assignment : Nat → Nat) : Prop :=
  ∀ index, index < digitCount →
    assignment (ShiftedTernary.negativeCols.getD index 0) =
      derivedNegative
        (assignment (ShiftedTernary.digitCols.getD index 0))

theorem materializeNegatives_is_materialized
    {assignment : Nat → Nat} :
    NegativesMaterialized (materializeNegatives assignment) := by
  intro index indexLt
  rw [materializeNegatives_negative indexLt,
    materializeNegatives_digit indexLt]

theorem materializeNegatives_eq_of_materialized
    {assignment : Nat → Nat}
    (materialized : NegativesMaterialized assignment) :
    materializeNegatives assignment = assignment := by
  funext column
  by_cases interval : 99 ≤ column ∧ column < 140
  · have indexLt : column - 99 < digitCount := by
      simp [digitCount]
      omega
    have digitEq := digitColumn_formula indexLt
    have negativeEq := negativeColumn_formula indexLt
    have negativeColumnEq :
        ShiftedTernary.negativeCols.getD (column - 99) 0 = column := by
      rw [negativeEq]
      omega
    have digitColumnEq :
        ShiftedTernary.digitCols.getD (column - 99) 0 = column - 41 := by
      rw [digitEq]
      omega
    have valueEq := materialized (column - 99) indexLt
    rw [negativeColumnEq, digitColumnEq] at valueEq
    unfold materializeNegatives
    rw [if_pos interval, ← valueEq]
  · unfold materializeNegatives
    rw [if_neg interval]

/-- Exact semantic gate schedule for the candidate; each index owns one
substituted borrow equation. -/
def derivedBorrowSchedule : List Nat := List.range digitCount

theorem derivedBorrowSchedule_length :
    derivedBorrowSchedule.length = 41 := by
  decide

theorem accepted_materializes_negatives
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (norm : DigitNormBoundTwo assignment)
    (accepted : CenteredTernaryNormDischarged.Accepts assignment) :
    NegativesMaterialized assignment := by
  intro index indexLt
  exact accepted_negative_eq_derived prime canonical one norm accepted indexLt

theorem accepted_implies_derived
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (norm : DigitNormBoundTwo assignment)
    (accepted : CenteredTernaryNormDischarged.Accepts assignment) :
    DerivedAccepts assignment := by
  have materialized :=
    accepted_materializes_negatives prime canonical one norm accepted
  have assignmentEq := materializeNegatives_eq_of_materialized materialized
  intro index indexLt
  simpa [DerivedBorrowHolds, assignmentEq] using
    accepted.borrowTransition index indexLt

theorem derived_implies_accepted
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (norm : DigitNormBoundTwo assignment)
    (materialized : NegativesMaterialized assignment)
    (accepted : DerivedAccepts assignment) :
    CenteredTernaryNormDischarged.Accepts assignment := by
  have assignmentEq := materializeNegatives_eq_of_materialized materialized
  constructor
  · intro index indexLt
    let digitColumn := ShiftedTernary.digitCols.getD index 0
    let negativeColumn := ShiftedTernary.negativeCols.getD index 0
    have centered : CenteredResidue (assignment digitColumn) :=
      normBoundTwo_iff_centeredResidue.mp (norm index indexLt)
    have indicatorEq :
        assignment negativeColumn = negativeIndicator (assignment digitColumn) := by
      calc
        assignment negativeColumn = derivedNegative (assignment digitColumn) :=
          materialized index indexLt
        _ = negativeIndicator (assignment digitColumn) :=
          derivedNegative_eq_indicator centered
    have digit : Digit (assignment digitColumn) (assignment negativeColumn) := by
      rw [indicatorEq]
      exact digit_of_centeredResidue centered
    exact digitDefinition_complete one digit
  · intro index indexLt
    simpa [DerivedBorrowHolds, assignmentEq] using accepted index indexLt

/-- Reduced-layout reconstruction theorem. The input assignment's 41 old
negative columns are ignored; materialization reconstructs them. Conservative
82-row acceptance of that extension is exactly the 41 substituted borrow
obligations, with no `NegativesMaterialized` premise on the reduced input. -/
theorem materialized_accepts_iff_derived
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (norm : DigitNormBoundTwo assignment) :
    CenteredTernaryNormDischarged.Accepts
        (materializeNegatives assignment) ↔
      DerivedAccepts assignment := by
  have materializedOne : materializeNegatives assignment 0 = 1 := by
    rw [materializeNegatives_one]
    exact one
  constructor
  · intro accepted index indexLt
    exact accepted.borrowTransition index indexLt
  · intro accepted
    constructor
    · intro index indexLt
      let digitColumn := ShiftedTernary.digitCols.getD index 0
      let negativeColumn := ShiftedTernary.negativeCols.getD index 0
      have centered : CenteredResidue (assignment digitColumn) :=
        normBoundTwo_iff_centeredResidue.mp (norm index indexLt)
      have digit :
          Digit
            (materializeNegatives assignment digitColumn)
            (materializeNegatives assignment negativeColumn) := by
        rw [materializeNegatives_digit indexLt,
          materializeNegatives_negative indexLt,
          derivedNegative_eq_indicator centered]
        exact digit_of_centeredResidue centered
      exact digitDefinition_complete materializedOne digit
    · intro index indexLt
      exact accepted index indexLt

/-- Exact old-layout model boundary: the conservative 82-row predicate equals
the 41-obligation substituted predicate plus unique reconstruction of the deleted
negative-indicator columns. -/
theorem conservative_iff_derived_and_materialized
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (norm : DigitNormBoundTwo assignment) :
    CenteredTernaryNormDischarged.Accepts assignment ↔
      DerivedAccepts assignment ∧ NegativesMaterialized assignment := by
  constructor
  · intro accepted
    exact ⟨accepted_implies_derived prime canonical one norm accepted,
      accepted_materializes_negatives prime canonical one norm accepted⟩
  · rintro ⟨derived, materialized⟩
    exact derived_implies_accepted one norm materialized derived

end Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative
