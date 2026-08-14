import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenaryNormDischarged

/-!
Model-level assignment-layout bridge for 23-coordinate centered-septenary
words.

Assurance tier: model-level.

Owns: derivation of coordinate membership from an exact assignment length and
bounded word starts, plus reconstruction of every encoded source field under
the verifier-owned strict `b = 4` norm.

Does not own: generated Rust word starts, proof that the production outer
opening supplies the norm, source-row semantics, or complete F-prime
same-assignment conformance.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.CenteredSeptenaryLayout

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CenteredSeptenaryField
open Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged

abbrev Field := Nightstream.SuperNeo.Concrete.F

structure Layout (fieldCount : Nat) where
  encodedColumnCount : Nat
  wordStart : Fin fieldCount → Nat
  wordFits : ∀ field,
    wordStart field + CenteredSeptenaryField.digitCount ≤ encodedColumnCount

def wordDigits {fieldCount : Nat}
    (layout : Layout fieldCount) (assignment : List Field)
    (field : Fin fieldCount) :
    Fin CenteredSeptenaryField.digitCount → Field :=
  fun digit => assignment.getD (layout.wordStart field + digit.val) 0

theorem word_coordinate_lt {fieldCount : Nat}
    (layout : Layout fieldCount) {assignment : List Field}
    (lengthExact : assignment.length = layout.encodedColumnCount)
    (field : Fin fieldCount)
    (digit : Fin CenteredSeptenaryField.digitCount) :
    layout.wordStart field + digit.val < assignment.length := by
  rw [lengthExact]
  have fits := layout.wordFits field
  omega

private theorem getD_mem_of_lt {values : List Field}
    {index : Nat} (indexLt : index < values.length) :
    values.getD index 0 ∈ values := by
  have member := List.getElem_mem (l := values) indexLt
  rwa [List.getElem_eq_getD 0] at member

theorem word_digit_mem {fieldCount : Nat}
    (layout : Layout fieldCount) {assignment : List Field}
    (lengthExact : assignment.length = layout.encodedColumnCount)
    (field : Fin fieldCount)
    (digit : Fin CenteredSeptenaryField.digitCount) :
    wordDigits layout assignment field digit ∈ assignment := by
  exact getD_mem_of_lt (word_coordinate_lt layout lengthExact field digit)

theorem every_word_has_septenary_alphabet {fieldCount : Nat}
    (layout : Layout fieldCount) {assignment : List Field}
    (lengthExact : assignment.length = layout.encodedColumnCount)
    (norm : Nightstream.SuperNeo.Concrete.normBounded 4 assignment) :
    ∀ field, FiniteAlphabetWord
      (finiteWordOfField (wordDigits layout assignment field)) := by
  intro field
  exact finiteWordOfField_alphabet_of_outer_norm norm
    (wordDigits layout assignment field)
    (word_digit_mem layout lengthExact field)

def chosenWord {fieldCount : Nat}
    (layout : Layout fieldCount) {assignment : List Field}
    (lengthExact : assignment.length = layout.encodedColumnCount)
    (norm : Nightstream.SuperNeo.Concrete.normBounded 4 assignment)
    (field : Fin fieldCount) : ChosenWitness :=
  chosenWitnessOfOuterNorm norm (wordDigits layout assignment field)
    (word_digit_mem layout lengthExact field)

def decodedAssignment {fieldCount : Nat}
    (layout : Layout fieldCount) {assignment : List Field}
    (lengthExact : assignment.length = layout.encodedColumnCount)
    (norm : Nightstream.SuperNeo.Concrete.normBounded 4 assignment) :
    Fin fieldCount → Nat :=
  fun field => (chosenWord layout lengthExact norm field).source

theorem decodedAssignment_canonical {fieldCount : Nat}
    (layout : Layout fieldCount) {assignment : List Field}
    (lengthExact : assignment.length = layout.encodedColumnCount)
    (norm : Nightstream.SuperNeo.Concrete.normBounded 4 assignment) :
    ∀ field, decodedAssignment layout lengthExact norm field <
      goldilocksP := by
  intro field
  exact (chosenWord layout lengthExact norm field).source_canonical

/-- Representation-soundness boundary: every accepted encoded assignment with
the exact layout and outer norm yields one canonical source assignment. -/
theorem accepted_reconstructs_canonical_source {fieldCount : Nat}
    (layout : Layout fieldCount) {assignment : List Field}
    (lengthExact : assignment.length = layout.encodedColumnCount)
    (norm : Nightstream.SuperNeo.Concrete.normBounded 4 assignment)
    (sourcePredicate : (Fin fieldCount → Nat) → Prop)
    (accepted : sourcePredicate
      (decodedAssignment layout lengthExact norm)) :
    ∃ source,
      (∀ field, source field < goldilocksP) ∧
        sourcePredicate source := by
  exact ⟨decodedAssignment layout lengthExact norm,
    decodedAssignment_canonical layout lengthExact norm, accepted⟩

end Nightstream.Implementation.R1CS.CenteredSeptenaryLayout
