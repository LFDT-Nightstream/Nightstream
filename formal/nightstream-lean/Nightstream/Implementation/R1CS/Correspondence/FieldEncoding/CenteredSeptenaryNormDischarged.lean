import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenary
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Model-level discharge of the centered-septenary alphabet from the
verifier-owned SuperNeo `b = 4` opening norm.

Assurance tier: model-level.

Owns: the exact equivalence between the strict Goldilocks norm window
`|x| < 4` and the seven canonical residues, plus construction of a
23-coordinate chosen witness from an externally norm-checked assignment.

Does not own: proof that a production acceptance path supplies this norm,
generated-coordinate membership, Rust assignment conformance, or complete
F-prime relation soundness.

Emits constraints: no. The external norm is verifier authority; witness data
or an internal digest is not a replacement for it.
-/

namespace Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.CenteredSeptenaryField

/-- Canonical-residue form of the authoritative strict `b = 4` norm. -/
def NormBoundFour (value : Nat) : Prop :=
  value < goldilocksP ∧ centeredMagnitude value < 4

theorem normBoundFour_iff_centeredResidue {value : Nat} :
    NormBoundFour value ↔ CenteredResidue value := by
  constructor
  · rintro ⟨canonical, bounded⟩
    unfold centeredMagnitude at bounded
    by_cases lowerHalf : value ≤ goldilocksP - value
    · rw [Nat.min_eq_left lowerHalf] at bounded
      have cases : value = 0 ∨ value = 1 ∨ value = 2 ∨ value = 3 := by
        omega
      rcases cases with zero | one | two | three
      all_goals simp [CenteredResidue, *]
    · have upperHalf : goldilocksP - value ≤ value := by omega
      rw [Nat.min_eq_right upperHalf] at bounded
      have differencePositive : 0 < goldilocksP - value :=
        Nat.sub_pos_of_lt canonical
      have cases :
          goldilocksP - value = 1 ∨
          goldilocksP - value = 2 ∨
          goldilocksP - value = 3 := by
        omega
      rcases cases with one | two | three
      · right; right; left
        omega
      · right; left
        omega
      · left
        omega
  · intro centered
    rcases centered with rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals constructor <;> decide

theorem concrete_norm_four_iff_centeredResidue
    (value : Nightstream.SuperNeo.Concrete.F) :
    Nightstream.SuperNeo.Concrete.centeredMagnitude value < 4 ↔
      CenteredResidue value.val := by
  constructor
  · intro bounded
    apply normBoundFour_iff_centeredResidue.mp
    constructor
    · exact value.isLt
    · simpa [Nightstream.SuperNeo.Concrete.centeredMagnitude,
        Nightstream.SuperNeo.Concrete.goldilocksModulus,
        centeredMagnitude, goldilocksP] using bounded
  · intro centered
    have bounded := (normBoundFour_iff_centeredResidue.mpr centered).2
    simpa [Nightstream.SuperNeo.Concrete.centeredMagnitude,
      Nightstream.SuperNeo.Concrete.goldilocksModulus,
      centeredMagnitude, goldilocksP] using bounded

theorem concrete_normBounded_four_implies_centered
    {assignment : List Nightstream.SuperNeo.Concrete.F}
    (norm : Nightstream.SuperNeo.Concrete.normBounded 4 assignment)
    {value : Nightstream.SuperNeo.Concrete.F}
    (member : value ∈ assignment) :
    CenteredResidue value.val := by
  exact (concrete_norm_four_iff_centeredResidue value).mp
    (norm value member)

def finiteWordOfField
    (digits : Fin CenteredSeptenaryField.digitCount →
      Nightstream.SuperNeo.Concrete.F) : FiniteWord :=
  fun index => (digits index).val

theorem finiteWordOfField_alphabet_of_outer_norm
    {assignment : List Nightstream.SuperNeo.Concrete.F}
    (norm : Nightstream.SuperNeo.Concrete.normBounded 4 assignment)
    (digits : Fin CenteredSeptenaryField.digitCount →
      Nightstream.SuperNeo.Concrete.F)
    (members : ∀ index, digits index ∈ assignment) :
    FiniteAlphabetWord (finiteWordOfField digits) := by
  intro index
  exact concrete_normBounded_four_implies_centered norm (members index)

def chosenWitnessOfOuterNorm
    {assignment : List Nightstream.SuperNeo.Concrete.F}
    (norm : Nightstream.SuperNeo.Concrete.normBounded 4 assignment)
    (digits : Fin CenteredSeptenaryField.digitCount →
      Nightstream.SuperNeo.Concrete.F)
    (members : ∀ index, digits index ∈ assignment) : ChosenWitness :=
  {
    digits := finiteWordOfField digits
    alphabet := finiteWordOfField_alphabet_of_outer_norm norm digits members
  }

theorem reconstructed_source_exists_of_outer_norm
    {sourcePredicate : Nat → Prop}
    {assignment : List Nightstream.SuperNeo.Concrete.F}
    (norm : Nightstream.SuperNeo.Concrete.normBounded 4 assignment)
    (digits : Fin CenteredSeptenaryField.digitCount →
      Nightstream.SuperNeo.Concrete.F)
    (members : ∀ index, digits index ∈ assignment)
    (accepted : sourcePredicate
      (decodeFiniteWord (finiteWordOfField digits))) :
    ∃ source, source < goldilocksP ∧ sourcePredicate source := by
  let witness := chosenWitnessOfOuterNorm norm digits members
  exact augmentedRelation_sound (witness := witness) accepted

end Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged
