import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LayoutManifest

/-!
Contract: derive the encoded- and CE-width floor of one generated fixed-F'
field-layout manifest from its exact coordinate partitions.

Owns: total-run reconciliation, the eligible-coordinate subtotal, and the
conditional theorem turning a generated exact eligible census into the
`eligible fields × 41` architecture floor.

Does not own: concrete base/recursive manifests, source-role classification,
manifest generation, selector composition, or a claim that 41 coordinates per
field is globally optimal across different encodings.

Emits constraints: no.

Authority boundary: the floor follows only after a generated artifact proves
its exact eligible-coordinate subtotal. A profiler count or handwritten field
count is not an input to these theorems.

| Theorem | Mathematical obligation | Production evidence required | Consequence |
|---|---|---|---|
| `encodedEligibleLength_le_total` | eligible owner runs are a subtotal of all encoded runs | valid encoded partition | subtotal ≤ encoded columns |
| `ceEligibleLength_le_total` | eligible owner runs are a subtotal of all CE runs | valid CE partition | subtotal ≤ CE assignment length |
| `encoded_width_floor` | eligible subtotal is exactly `eligibleCount * 41` | generated exact census | encoded width floor |
| `ce_width_floor` | eligible subtotal is exactly `eligibleCount * 41` | generated exact census | CE width floor |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFieldLayout

open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler

/-- Sum the lengths of all runs in one exact partition. -/
def totalRunLength {Owner : Type}
    (runOf : Owner → CoordinateRun) (owners : List Owner) : Nat :=
  (owners.map fun owner => (runOf owner).length).sum

namespace ExactPartition

private theorem totalRunLengthFrom_eq
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {cursor count : Nat} {owners : List Owner}
    (partition : ExactPartitionFrom runOf cursor count owners) :
    cursor + totalRunLength runOf owners = count := by
  induction owners generalizing cursor with
  | nil =>
      simpa [ExactPartitionFrom, totalRunLength] using partition
  | cons head tail inductionHypothesis =>
      simp only [ExactPartitionFrom] at partition
      rcases partition with ⟨headStart, headPositive, tailPartition⟩
      have tailTotal := inductionHypothesis tailPartition
      simp only [totalRunLength, List.map_cons, List.sum_cons,
        CoordinateRun.endExclusive] at tailTotal ⊢
      omega

/-- An exact partition's universe size is the sum of its run lengths. -/
theorem totalRunLength_eq
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {count : Nat} {owners : List Owner}
    (partition : ExactPartition runOf count owners) :
    totalRunLength runOf owners = count := by
  have total := totalRunLengthFrom_eq partition
  simpa using total

end ExactPartition

/-- Encoded-coordinate subtotal owned by ordinary private source fields. -/
def eligibleEncodedLength : List CoordinateOwnerRun → Nat
  | [] => 0
  | owner :: tail =>
      (if owner.role.Eligible then owner.encoded.length else 0) +
        eligibleEncodedLength tail

/-- CE-coordinate subtotal owned by ordinary private source fields. -/
def eligibleCeLength : List CoordinateOwnerRun → Nat
  | [] => 0
  | owner :: tail =>
      (if owner.role.Eligible then owner.ce.length else 0) +
        eligibleCeLength tail

private theorem eligibleEncodedLength_le_totalRunLength
    (owners : List CoordinateOwnerRun) :
    eligibleEncodedLength owners ≤
      totalRunLength CoordinateOwnerRun.encoded owners := by
  induction owners with
  | nil => simp [eligibleEncodedLength, totalRunLength]
  | cons owner tail inductionHypothesis =>
      simp only [totalRunLength] at inductionHypothesis
      simp only [eligibleEncodedLength, totalRunLength, List.map_cons,
        List.sum_cons]
      split <;> omega

private theorem eligibleCeLength_le_totalRunLength
    (owners : List CoordinateOwnerRun) :
    eligibleCeLength owners ≤
      totalRunLength CoordinateOwnerRun.ce owners := by
  induction owners with
  | nil => simp [eligibleCeLength, totalRunLength]
  | cons owner tail inductionHypothesis =>
      simp only [totalRunLength] at inductionHypothesis
      simp only [eligibleCeLength, totalRunLength, List.map_cons,
        List.sum_cons]
      split <;> omega

/-- The eligible encoded-coordinate subtotal cannot exceed the manifest's
exact encoded-coordinate universe. -/
theorem encodedEligibleLength_le_total
    {manifest : Manifest} (valid : manifest.Valid) :
    eligibleEncodedLength manifest.coordinateOwners ≤
      manifest.encodedColumnCount := by
  rw [← ExactPartition.totalRunLength_eq valid.encodedPartition]
  exact eligibleEncodedLength_le_totalRunLength manifest.coordinateOwners

/-- The eligible CE-coordinate subtotal cannot exceed the manifest's exact
committed-assignment universe. -/
theorem ceEligibleLength_le_total
    {manifest : Manifest} (valid : manifest.Valid) :
    eligibleCeLength manifest.coordinateOwners ≤
      manifest.ceAssignmentLength := by
  rw [← ExactPartition.totalRunLength_eq valid.cePartition]
  exact eligibleCeLength_le_totalRunLength manifest.coordinateOwners

/-- Current per-field centered encoding width. This is an architecture
constant, not a claim about arbitrary cross-field compression. -/
theorem digitCount_eq_41 : digitCount = 41 := by
  rfl

/-- Once generated data proves the exact ordinary-field subtotal, every
eligible field contributes 41 encoded columns to this architecture. -/
theorem encoded_width_floor
    {artifact : GeneratedArtifact}
    (exactEligible :
      eligibleEncodedLength artifact.manifest.coordinateOwners =
        artifact.manifest.eligibleCount * digitCount) :
    artifact.manifest.eligibleCount * 41 ≤
      artifact.manifest.encodedColumnCount := by
  rw [← digitCount_eq_41, ← exactEligible]
  exact encodedEligibleLength_le_total artifact.valid

/-- Once generated data proves the exact ordinary-field subtotal, every
eligible field contributes 41 coordinates to the committed CE assignment. -/
theorem ce_width_floor
    {artifact : GeneratedArtifact}
    (exactEligible :
      eligibleCeLength artifact.manifest.coordinateOwners =
        artifact.manifest.eligibleCount * digitCount) :
    artifact.manifest.eligibleCount * 41 ≤
      artifact.manifest.ceAssignmentLength := by
  rw [← digitCount_eq_41, ← exactEligible]
  exact ceEligibleLength_le_total artifact.valid

end Nightstream.Implementation.R1CS.FPrimeFieldLayout
