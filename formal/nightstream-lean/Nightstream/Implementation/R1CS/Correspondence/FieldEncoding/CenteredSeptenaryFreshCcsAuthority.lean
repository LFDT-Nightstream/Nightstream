import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenaryLayout
import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.Concrete.Phi81Relation.Semantics

/-!
Contract: exact model-level transfer of the verifier-owned fresh-CCS
`b = 4` norm to any bounded 23-coordinate centered-septenary layout.

Assurance tier: model-level for property
`FPRIME-R4-OUTER-NORM-DISCHARGES-WORD-DOMAINS`.

Owns: typed same-assignment coordinates, their exact width bound, extraction
of the strict radix-four norm from one fresh `CCS.Holds` fact, and the
seven-symbol alphabet for every word in an arbitrary bounded layout.

Does not own: generated Rust word starts, selective-matrix substitution,
source-row reconstruction, proof that every omitted production row belongs
to this family, or complete F-prime relation soundness.

Emits constraints: no. The verifier-owned outer norm is the authority. A
digest or a prover-carried domain claim cannot replace it.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.CenteredSeptenaryFreshCcsAuthority

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CenteredSeptenaryField
open Nightstream.Implementation.R1CS.CenteredSeptenaryLayout
open Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

universe uCommitment

/-- Exact typed carrier coordinate for one digit of one generated word. The
result cannot name padding beyond the complete outer assignment. -/
def wordCoordinate {fieldCount : Nat}
    (layout : Layout fieldCount)
    {shape : Shape}
    (widthExact : layout.encodedColumnCount = shape.carrierWidth)
    (field : Fin fieldCount)
    (digit : Fin CenteredSeptenaryField.digitCount) :
    Fin shape.carrierWidth :=
  ⟨layout.wordStart field + digit.val, by
    rw [← widthExact]
    have fits := layout.wordFits field
    omega⟩

/-- The exact word projected from the same typed assignment that the outer
CCS opening commits and norm-checks. -/
def typedWordDigits {fieldCount : Nat}
    (layout : Layout fieldCount)
    {shape : Shape}
    (widthExact : layout.encodedColumnCount = shape.carrierWidth)
    (assignment : Assignment shape)
    (field : Fin fieldCount) :
    Fin CenteredSeptenaryField.digitCount → Field :=
  fun digit => assignment (wordCoordinate layout widthExact field digit)

theorem every_word_has_septenary_alphabet_of_norm
    {fieldCount : Nat}
    (layout : Layout fieldCount)
    {shape : Shape}
    (widthExact : layout.encodedColumnCount = shape.carrierWidth)
    (assignment : Assignment shape)
    (norm : assignmentNormBounded 4 assignment) :
    ∀ field, FiniteAlphabetWord
      (finiteWordOfField
        (typedWordDigits layout widthExact assignment field)) := by
  intro field digit
  exact (concrete_norm_four_iff_centeredResidue
    (assignment (wordCoordinate layout widthExact field digit))).mp
      (norm (wordCoordinate layout widthExact field digit))

/-- A fresh CCS opening under verifier-owned `b = 4` gives the exact norm
used to justify removal of per-word alphabet rows. -/
theorem norm_four_of_fresh_ccsHolds
    {shape : Shape}
    {Commitment : Type uCommitment}
    (commit : Assignment shape → Commitment)
    (params : GlobalParams)
    (baseFour : params.b = 4)
    (statement : CCSStatement shape Commitment)
    (fresh : statement.stage = .fresh)
    (assignment : Assignment shape)
    (holds : CCS.Holds (relationSemantics commit) params statement assignment) :
    assignmentNormBounded 4 assignment := by
  have bounded := holds.1.2.2
  simpa [NormStage.bound, fresh, baseFour] using bounded

/-- Conditional row-discharge boundary for an arbitrary exact layout. This
uses the same assignment in the CCS opening and in every projected word. -/
theorem every_word_has_septenary_alphabet_of_fresh_ccsHolds
    {fieldCount : Nat}
    (layout : Layout fieldCount)
    {shape : Shape}
    (widthExact : layout.encodedColumnCount = shape.carrierWidth)
    {Commitment : Type uCommitment}
    (commit : Assignment shape → Commitment)
    (params : GlobalParams)
    (baseFour : params.b = 4)
    (statement : CCSStatement shape Commitment)
    (fresh : statement.stage = .fresh)
    (assignment : Assignment shape)
    (holds : CCS.Holds (relationSemantics commit) params statement assignment) :
    ∀ field, FiniteAlphabetWord
      (finiteWordOfField
        (typedWordDigits layout widthExact assignment field)) := by
  exact every_word_has_septenary_alphabet_of_norm layout widthExact assignment
    (norm_four_of_fresh_ccsHolds commit params baseFour statement fresh
      assignment holds)

/-- Supported-profile specialization. It fixes the exact radix-four
candidate parameters rather than accepting a caller-selected bound. -/
theorem radixFourCandidate_every_word_has_septenary_alphabet
    {fieldCount : Nat}
    (layout : Layout fieldCount)
    {shape : Shape}
    (widthExact : layout.encodedColumnCount = shape.carrierWidth)
    {Commitment : Type uCommitment}
    (commit : Assignment shape → Commitment)
    (statement : CCSStatement shape Commitment)
    (fresh : statement.stage = .fresh)
    (assignment : Assignment shape)
    (holds : CCS.Holds (relationSemantics commit)
      Radix4Candidate.globalParams statement assignment) :
    ∀ field, FiniteAlphabetWord
      (finiteWordOfField
        (typedWordDigits layout widthExact assignment field)) := by
  exact every_word_has_septenary_alphabet_of_fresh_ccsHolds layout widthExact
    commit Radix4Candidate.globalParams rfl statement fresh assignment holds

end Nightstream.Implementation.R1CS.CenteredSeptenaryFreshCcsAuthority
