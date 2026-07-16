import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.SharedSlots
import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.CenteredZero

namespace NightstreamTests.ShiftedTernary

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernary
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.ShiftedTernaryComplete

def honestAssignment : Nat → Nat := fun column => honestWitness.getD column 0

def forgedAssignment : Nat → Nat := fun column => forgedWitness.getD column 0

/-- The exact Rust-generated honest witness satisfies all 180 rows. -/
theorem honest_satisfies : Satisfies rows honestAssignment := by native_decide

/-- The `x + p` alternate opening fails at the terminal borrow row. -/
example : ¬ Satisfies rows forgedAssignment := by native_decide

/-- The compact seeded-commitment compiler expands to the exact generated
commitment suffix, so its generic theorem applies to this artifact. -/
example : commitmentBlock.Valid ∧ commitmentBlock.rows = rows.drop 126 := by
  exact ⟨commitmentBlock_valid, commitmentRows_eq_artifact⟩

/-- The production Goldilocks fixture receives the full semantic theorem:
canonical field opening, fixed shape, and exact seeded Phi81 commitment. -/
example (prime : EuclidPrime goldilocksP)
    (canonical : ∀ column, honestAssignment column < goldilocksP) :
    OneFieldSound honestAssignment := by
  apply oneField_sound prime
  · exact canonical
  · rfl
  · exact honest_satisfies

/-- Completeness is driven by the native witness-generator relation, not by
an assumed acceptance result or a duplicate row predicate. -/
example (witness : CanonicalWitness honestAssignment) :
    Satisfies ShiftedTernaryCompiler.canonicalRows honestAssignment :=
  canonicalRows_complete witness

/-- The generated Rust aliases and coefficients instantiate the exact
production shared-slot expansion. -/
example :
    ShiftedTernarySharedSlots.sourceExpansion
        ShiftedTernarySharedSlots.productionLayout
        ShiftedTernarySharedSlotsArtifact.sourceFieldCol =
      ShiftedTernarySharedSlotsArtifact.fieldTerms :=
  ShiftedTernarySharedSlots.production_sourceExpansion_instantiates.2.1

/-- Schema 3 pins 123 logical retained obligations, their exact 103 physical
rows, and the 123 omitted obligations. -/
example : ShiftedTernarySharedSlotsArtifact.schemaVersion = 3 :=
  ShiftedTernarySharedSlots.production_schema

/-- The exported CCS gate polynomial, evaluated on the generated sparse
matrix rows, is exactly the reduced production acceptance predicate under the
explicit Goldilocks projective-nonresidue premise. -/
example
    (projectiveNonresidue :
      ShiftedTernarySharedSlots.ProjectiveSevenNonresidue)
    (encoded : Nat → Nat)
    (one : encoded ShiftedTernarySharedSlotsArtifact.oneColumn = 1) :
    ShiftedTernarySharedSlots.ArtifactGateAccepts encoded ↔
      ShiftedTernarySharedSlots.ProductionAccepts encoded :=
  ShiftedTernarySharedSlots.artifactGateAccepts_iff_productionAccepts
    projectiveNonresidue encoded one

/-- The actual 20 residual-pair rows, one tail, and 82 product rows are
equivalent to all 124 canonical source rows under the explicit algebraic
premise and fixed-one convention. -/
example
    (projectiveNonresidue :
      ShiftedTernarySharedSlots.ProjectiveSevenNonresidue)
    (prime : EuclidPrime goldilocksP) (encoded : Nat → Nat)
    (one : encoded ShiftedTernarySharedSlotsArtifact.oneColumn = 1) :
    ShiftedTernarySharedSlots.ArtifactGateAccepts encoded ↔
      Satisfies ShiftedTernaryCompiler.canonicalRows
        (ShiftedTernarySharedSlots.decodedAssignment
          ShiftedTernarySharedSlots.productionLayout encoded) :=
  ShiftedTernarySharedSlots.artifactGateAccepts_iff_canonicalRows
    projectiveNonresidue prime one

/-- Completeness remains driven by the independent native witness relation,
not by replaying the omitted rows. -/
example (encoded : Nat → Nat)
    (witness : CanonicalWitness
      (ShiftedTernarySharedSlots.decodedAssignment
        ShiftedTernarySharedSlots.productionLayout encoded)) :
    ShiftedTernarySharedSlots.ProductionAccepts encoded :=
  ShiftedTernarySharedSlots.production_complete witness

/-- The reduced inactive-opening argument is independent of borrow rows: a
centered-unit word with zero weighted field value is coordinate-wise zero. -/
example {value negative : Nat → Nat}
    (digits : ∀ index, index < ShiftedTernaryCompiler.digitCount →
      ShiftedTernaryCompiler.Digit (value index) (negative index))
    (weightedZero : ShiftedTernarySound.lowValue value
      ShiftedTernaryCompiler.digitCount % goldilocksP = 0) :
    ∀ index, index < ShiftedTernaryCompiler.digitCount → value index = 0 :=
  ShiftedTernaryCenteredZero.centered_zero_unique digits weightedZero

end NightstreamTests.ShiftedTernary
