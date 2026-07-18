import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredTernary
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Contract: discharge internal centered-unit gates from the authoritative
SuperNeo `b = 2` opening norm.

Owns: the exact equivalence between the Goldilocks centered norm window and
the three centered residues, and the model-level 82-row canonical
shifted-ternary core obtained by retaining only negative-definition and borrow
transition gates.

Does not own: proof that a production acceptance path checks the SuperNeo norm,
fixed-F-prime slot membership, Rust row removal, selector noninterference, or
standalone R1CS backends that omit the norm predicate.

Emits constraints: no. `normDischargedGates` is a costed model schedule.

Authority boundary: the removed cubic equations are implied only by the
verifier-owned SuperNeo opening predicate `normBounded 2 z`. A prover claim,
digest, witness generator check, or plain matrix-satisfaction result is not a
replacement for that premise.

| Branch | Mathematical obligation | Internal gates | External premise | Tier |
|---|---|---:|---|---|
| ordinary centered field | each coordinate is in `{-1,0,1}` | 0 | `normBounded 2` | model-level |
| canonical shifted opening | negative indicators and radix-three range | 82 | `normBounded 2` on digit coordinates | model-level |
-/

namespace Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore
open Nightstream.Implementation.R1CS.CenteredTernaryField

set_option maxRecDepth 262144

/-- Canonical-residue form of the authoritative strict `b = 2` norm. -/
def NormBoundTwo (value : Nat) : Prop :=
  value < goldilocksP ∧ centeredMagnitude value < 2

/-- Over canonical Goldilocks residues, the external strict norm bound has
exactly the same three roots as the internal cubic alphabet gate. -/
theorem normBoundTwo_iff_centeredResidue {value : Nat} :
    NormBoundTwo value ↔ CenteredResidue value := by
  constructor
  · rintro ⟨canonical, bounded⟩
    unfold centeredMagnitude at bounded
    by_cases lowerHalf : value ≤ goldilocksP - value
    · rw [Nat.min_eq_left lowerHalf] at bounded
      rcases Nat.eq_zero_or_pos value with rfl | positive
      · exact Or.inr (Or.inl rfl)
      · have valueEq : value = 1 := by omega
        exact Or.inr (Or.inr valueEq)
    · have upperHalf : goldilocksP - value ≤ value := by omega
      rw [Nat.min_eq_right upperHalf] at bounded
      have differencePositive : 0 < goldilocksP - value :=
        Nat.sub_pos_of_lt canonical
      have differenceEq : goldilocksP - value = 1 := by omega
      have valueEq : value = goldilocksP - 1 := by omega
      exact Or.inl valueEq
  · intro centered
    constructor
    · rcases centered with negative | zero | one
      · rw [negative]; decide
      · rw [zero]; decide
      · rw [one]; decide
    · exact centeredResidue_norm_lt_two centered

theorem normBoundTwo_iff_centeredUnitGate
    (prime : EuclidPrime goldilocksP) {value : Nat} :
    NormBoundTwo value ↔
      value < goldilocksP ∧ CenteredUnitGateHolds value := by
  constructor
  · intro bounded
    exact ⟨bounded.1, (centeredUnitGate_iff prime bounded.1).mpr
      (normBoundTwo_iff_centeredResidue.mp bounded)⟩
  · rintro ⟨canonical, gate⟩
    exact normBoundTwo_iff_centeredResidue.mpr
      ((centeredUnitGate_iff prime canonical).mp gate)

/-- Direct bridge to the concrete SuperNeo norm used by CCS and CE openings. -/
theorem concrete_norm_two_iff_centeredResidue
    (value : Nightstream.SuperNeo.Concrete.F) :
    Nightstream.SuperNeo.Concrete.centeredMagnitude value < 2 ↔
      CenteredResidue value.val := by
  constructor
  · intro bounded
    apply normBoundTwo_iff_centeredResidue.mp
    constructor
    · exact value.isLt
    · simpa [Nightstream.SuperNeo.Concrete.centeredMagnitude,
        Nightstream.SuperNeo.Concrete.goldilocksModulus,
        centeredMagnitude, goldilocksP] using bounded
  · intro centered
    have bounded := (normBoundTwo_iff_centeredResidue.mpr centered).2
    simpa [Nightstream.SuperNeo.Concrete.centeredMagnitude,
      Nightstream.SuperNeo.Concrete.goldilocksModulus,
      centeredMagnitude, goldilocksP] using bounded

theorem concrete_normBounded_two_implies_centered
    {assignment : List Nightstream.SuperNeo.Concrete.F}
    (norm : Nightstream.SuperNeo.Concrete.normBounded 2 assignment)
    {value : Nightstream.SuperNeo.Concrete.F}
    (member : value ∈ assignment) :
    CenteredResidue value.val := by
  exact (concrete_norm_two_iff_centeredResidue value).mp
    (norm value member)

/-- External norm premise restricted to the shifted-opening digit columns. -/
def DigitNormBoundTwo (assignment : Nat → Nat) : Prop :=
  ∀ index, index < digitCount →
    NormBoundTwo
      (assignment (ShiftedTernary.digitCols.getD index 0))

/-- Internal canonical-opening obligations after the centered alphabet is
discharged to the outer relation. -/
structure Accepts (assignment : Nat → Nat) : Prop where
  negativeDefinition : ∀ index, index < digitCount →
    RowHolds assignment (negativeDefinitionRow
      (ShiftedTernary.digitCols.getD index 0)
      (ShiftedTernary.negativeCols.getD index 0))
  borrowTransition : ∀ index, index < digitCount →
    RowHolds assignment (borrowRow index)

/-- Exact retained model schedule: 41 negative definitions and 41 borrow
transitions. -/
def normDischargedGates : List Gate :=
  negativeDefinitionGates ++ borrowTransitionGates

theorem normDischargedGates_length :
    normDischargedGates.length = 82 := by
  decide

/-- The 82 retained gates plus the external norm are exactly the existing
123-gate reduced predicate. -/
theorem accepts_iff_reduced
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (norm : DigitNormBoundTwo assignment) :
    Accepts assignment ↔
      ShiftedTernaryReducedCore.Accepts assignment := by
  constructor
  · intro accepted
    constructor
    · intro index indexLt
      have bounded := norm index indexLt
      exact (centeredUnitGate_iff prime bounded.1).mpr
        (normBoundTwo_iff_centeredResidue.mp bounded)
    · exact accepted.negativeDefinition
    · exact accepted.borrowTransition
  · intro accepted
    exact ⟨accepted.negativeDefinition, accepted.borrowTransition⟩

/-- With the shared field/digit alias, the 82-row core and the external norm
are model-equivalent to all 124 old canonical-opening rows. -/
theorem accepts_iff_canonicalRows
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (norm : DigitNormBoundTwo assignment) :
    (Accepts assignment ∧ SharedFieldDigitAlias assignment) ↔
      Satisfies canonicalRows assignment := by
  rw [accepts_iff_reduced prime norm]
  exact reduced_iff_canonicalRows prime canonical one

end Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
