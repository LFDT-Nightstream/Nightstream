import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePins
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.ProductionPublicWriteTrace

/-!
Concrete active public-write execution for the fixed post-PiDEC artifact.

Assurance tier: artifact-checked and fixed-profile Rust-conformant.

Owns: interpretation of the exact generated runtime values as a normalized
assignment and matching typed legacy public source; derivation of
`ActivePublicWritesBound` and constant one from the generated execution data;
and the premise-free fixed-artifact typed Carrier270 conclusion.

Does not own: application/source-relation satisfaction of the constructed
legacy suffix, private assignment decoding, final sparse A/B/C equality,
CCS/CE membership, commitment-key alignment, protocol acceptance, or row
removal.

Emits constraints: none.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.post_pidec.public_write.values` | generated runtime values populate the normalized public prefix | direct dataflow artifact |
| `f_prime.post_pidec.public_write.sources` | direct sources populate the same typed legacy public prefix | derived |
| `f_prime.post_pidec.public_write.bound` | actual generated writes satisfy `ActivePublicWritesBound` | derived |
| `f_prime.post_pidec.public_write.carrier270` | generated normalized prefix equals the typed Carrier270 projection | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePublicWriteExecution

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.ProductionPublicWriteTrace
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePins
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

/-- Fixed runtime assignment projected from the exact generated public-write
values. Coordinates outside the public prefix are deliberately not claimed. -/
def executionAssignment : Nat -> Nat :=
  fun column =>
    if inPublic : column < PublicDecoder.alignedPublicWidth then
      (productionRawWrite ⟨column, inPublic⟩).value
    else
      0

/-- One typed legacy assignment whose public prefix is the exact runtime
source vector. Its private suffix is zero because this leaf owns only the
public carrier. -/
def executionLegacy (dimensions : Dimensions) : LegacyAssignment dimensions :=
  fun column =>
    if inPublic : column.val < legacyPublicWidth then
      fieldResidue
        (productionRawWrite ⟨column.val, by
          exact Nat.lt_trans inPublic (by
            decide)⟩).value
    else
      0

private theorem generatedPackedCoordinate_valid
    {coordinate :
      Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.RawPackedPublicCoordinate}
    (member : coordinate ∈ generatedPackedCoordinates) :
    PackedCoordinateValid coordinate := by
  rcases generated_packed_coordinate_decodes member with ⟨decoded, decodes⟩
  unfold decodePackedCoordinate at decodes
  split at decodes
  · assumption
  · simp at decodes

theorem executionAssignment_at
    (column : Fin PublicDecoder.alignedPublicWidth) :
    fieldResidue (executionAssignment column.val) =
      fieldResidue (productionRawWrite column).value := by
  simp [executionAssignment, column.isLt]

/-- The exact generated constant runtime write supplies the independently
named constant-one source fact. -/
theorem executionLegacy_constantOne (dimensions : Dimensions) :
    ActiveSourceConstantOne dimensions (executionLegacy dimensions) := by
  let column : Fin PublicDecoder.alignedPublicWidth := ⟨0, by decide⟩
  have rawOne : (productionRawWrite column).value = 1 :=
    (productionRawWrite_valueShape column).1 rfl
  simp [ActiveSourceConstantOne, executionLegacy, legacyPublicWidth,
    column, rawOne]

/-- The generated runtime values discharge the formerly external live-write
contract for this exact execution artifact. No source label or digest is used
as value authority: each value is the Rust-checked builder/normalized/packed
join exported in `productionRawWrite`. -/
theorem execution_activePublicWritesBound (dimensions : Dimensions) :
    ActivePublicWritesBound executionAssignment dimensions
      (executionLegacy dimensions) := by
  constructor
  intro coordinate member
  have valid := generatedPackedCoordinate_valid member
  let column : Fin PublicDecoder.alignedPublicWidth :=
    ⟨coordinate.column, valid.2.1⟩
  rw [executionAssignment_at column, valid.2.2.2.2.2]
  by_cases zero : coordinate.column = 0
  · have rawOne : (productionRawWrite column).value = 1 :=
      (productionRawWrite_valueShape column).1 zero
    simp [expectedPublicSource, zero, legacyPublicWidth,
      interpretedActivePublicSource, rawOne]
  · by_cases inLegacy : coordinate.column < legacyPublicWidth
    · have inLegacyLiteral : coordinate.column < 257 := by
        simpa [legacyPublicWidth] using inLegacy
      simp [expectedPublicSource, zero, inLegacyLiteral, legacyPublicWidth,
        interpretedActivePublicSource, executionLegacy, column]
    · have padding : legacyPublicWidth <= coordinate.column :=
        Nat.le_of_not_gt inLegacy
      have notInLegacyLiteral : ¬ coordinate.column < 257 := by
        simpa [legacyPublicWidth] using inLegacy
      have rawZero : (productionRawWrite column).value = 0 :=
        (productionRawWrite_valueShape column).2 padding
      simp [expectedPublicSource, zero, notInLegacyLiteral, legacyPublicWidth,
        interpretedActivePublicSource, rawZero]

/-- Premise-free fixed-artifact public-carrier refinement.  This eliminates
`ActivePublicWritesBound` for the exported execution profile; it does not
claim that the zero-filled private suffix satisfies the production relation. -/
theorem execution_normalizedPublicInput_eq_projectPublicInput
    (dimensions : Dimensions) :
    normalizedPublicInput executionAssignment dimensions =
      projectPublicInput (assignment dimensions (executionLegacy dimensions)) := by
  exact activePublicWritesBound_implies_typedPublicAssignment dimensions
    (executionLegacy dimensions)
    (execution_activePublicWritesBound dimensions)
    (executionLegacy_constantOne dimensions)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePublicWriteExecution
