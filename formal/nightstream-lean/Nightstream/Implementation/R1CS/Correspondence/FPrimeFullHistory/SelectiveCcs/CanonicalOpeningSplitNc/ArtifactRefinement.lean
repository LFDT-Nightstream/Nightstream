import Nightstream.Implementation.R1CS.Artifacts.ShiftedTernary
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningArtifactRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningSplitNc

/-!
Contract: instantiate the optimized canonical-opening layout from the generated
shifted-ternary selective artifact.

Assurance tier: artifact-checked column-layout refinement composed with the
model-level Split-NC/canonicality theorem.

Owns: the exact 41 digit and 20 retained-borrow coordinates exported through
the shifted-ternary artifact facade; their bounds, disjointness, and injective
embedding into `ProductionLayout`; specialization of Split-NC coverage to that
layout; and composition of the exact generated rows with that coverage.

Does not own: production opening multiplicity, conversion of selective-CCS
acceptance into the exact row-residual premise, Rust conformance beyond the
generated artifact, verifier acceptance, or Fiat-Shamir soundness.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
open Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

private theorem rangeMap_getD_of_lt
    (base count : Nat) {index : Nat} (indexLt : index < count) :
    ((List.range count).map fun offset => base + offset).getD index 0 =
      base + index := by
  simp [List.getD_eq_getElem?_getD, indexLt]

/-- The generated digit coordinates are exactly one 41-column interval. -/
theorem digitCoordinates_eq_rangeMap :
    ShiftedTernarySelectiveArtifact.digitCoordinates =
      (List.range digitCount).map fun index => 108 + index := by
  decide

/-- The generated retained-borrow coordinates are exactly one 20-column
interval. -/
theorem borrowCoordinates_eq_rangeMap :
    ShiftedTernarySelectiveArtifact.borrowCoordinates =
      (List.range chunkBorrowCount).map fun index => 2363 + index := by
  decide

theorem digitCoordinates_length :
    ShiftedTernarySelectiveArtifact.digitCoordinates.length = digitCount := by
  rw [digitCoordinates_eq_rangeMap]
  simp

theorem borrowCoordinates_length :
    ShiftedTernarySelectiveArtifact.borrowCoordinates.length =
      chunkBorrowCount := by
  rw [borrowCoordinates_eq_rangeMap]
  simp

/-- The artifact exports exactly 61 optimized opening coordinates. -/
theorem openingCoordinates_length :
    (ShiftedTernarySelectiveArtifact.digitCoordinates ++
      ShiftedTernarySelectiveArtifact.borrowCoordinates).length = 61 := by
  rw [List.length_append, digitCoordinates_length, borrowCoordinates_length]
  decide

/-- No generated digit or retained-borrow coordinate aliases another. -/
theorem openingCoordinates_nodup :
    (ShiftedTernarySelectiveArtifact.digitCoordinates ++
      ShiftedTernarySelectiveArtifact.borrowCoordinates).Nodup := by
  decide

/-- The artifact identifies exactly 21 physical row indices. This theorem
records multiplicity only; it does not interpret those rows. -/
theorem canonicalRows_length :
    ShiftedTernarySelectiveArtifact.canonicalRows.length = chunkCount := by
  decide

theorem digitCoordinate_formula (index : Fin digitCount) :
    ShiftedTernarySelectiveArtifact.digitCoordinates.getD index.val 0 =
      108 + index.val := by
  rw [digitCoordinates_eq_rangeMap]
  exact rangeMap_getD_of_lt 108 digitCount index.isLt

theorem borrowCoordinate_formula (index : Fin chunkBorrowCount) :
    ShiftedTernarySelectiveArtifact.borrowCoordinates.getD index.val 0 =
      2363 + index.val := by
  rw [borrowCoordinates_eq_rangeMap]
  exact rangeMap_getD_of_lt 2363 chunkBorrowCount index.isLt

/-- Every generated digit coordinate lies inside the generated structure. -/
theorem digitCoordinate_lt_structure (index : Fin digitCount) :
    ShiftedTernarySelectiveArtifact.digitCoordinates.getD index.val 0 <
      ShiftedTernarySelectiveArtifact.structureColumnCount := by
  rw [digitCoordinate_formula]
  have indexLt := index.isLt
  change index.val < 41 at indexLt
  change 108 + index.val < 2430
  omega

/-- Every generated retained-borrow coordinate lies inside the generated
structure. -/
theorem borrowCoordinate_lt_structure (index : Fin chunkBorrowCount) :
    ShiftedTernarySelectiveArtifact.borrowCoordinates.getD index.val 0 <
      ShiftedTernarySelectiveArtifact.structureColumnCount := by
  rw [borrowCoordinate_formula]
  have indexLt := index.isLt
  change index.val < 20 at indexLt
  change 2363 + index.val < 2430
  omega

/-- Exact artifact column selected by one coordinate of the single generated
opening. -/
def artifactCoordinateNat : Coordinate 1 → Nat
  | .digit _ index =>
      ShiftedTernarySelectiveArtifact.digitCoordinates.getD index.val 0
  | .borrow _ index =>
      ShiftedTernarySelectiveArtifact.borrowCoordinates.getD index.val 0

theorem artifactCoordinateNat_lt_structure
    (coordinate : Coordinate 1) :
    artifactCoordinateNat coordinate <
      ShiftedTernarySelectiveArtifact.structureColumnCount := by
  cases coordinate with
  | digit _ index => exact digitCoordinate_lt_structure index
  | borrow _ index => exact borrowCoordinate_lt_structure index

/-- The generated coordinate map is injective across both coordinate kinds. -/
theorem artifactCoordinateNat_injective :
    Function.Injective artifactCoordinateNat := by
  intro left right equal
  cases left with
  | digit leftOpening leftIndex =>
      cases right with
      | digit rightOpening rightIndex =>
          change
            ShiftedTernarySelectiveArtifact.digitCoordinates.getD
                leftIndex.val 0 =
              ShiftedTernarySelectiveArtifact.digitCoordinates.getD
                rightIndex.val 0 at equal
          rw [digitCoordinate_formula, digitCoordinate_formula] at equal
          have openingEq : leftOpening = rightOpening :=
            Subsingleton.elim _ _
          have indexEq : leftIndex = rightIndex := Fin.ext (by omega)
          cases openingEq
          cases indexEq
          rfl
      | borrow _ rightIndex =>
          change
            ShiftedTernarySelectiveArtifact.digitCoordinates.getD
                leftIndex.val 0 =
              ShiftedTernarySelectiveArtifact.borrowCoordinates.getD
                rightIndex.val 0 at equal
          rw [digitCoordinate_formula, borrowCoordinate_formula] at equal
          have leftLt := leftIndex.isLt
          have rightLt := rightIndex.isLt
          change leftIndex.val < 41 at leftLt
          change rightIndex.val < 20 at rightLt
          omega
  | borrow leftOpening leftIndex =>
      cases right with
      | digit _ rightIndex =>
          change
            ShiftedTernarySelectiveArtifact.borrowCoordinates.getD
                leftIndex.val 0 =
              ShiftedTernarySelectiveArtifact.digitCoordinates.getD
                rightIndex.val 0 at equal
          rw [borrowCoordinate_formula, digitCoordinate_formula] at equal
          have leftLt := leftIndex.isLt
          have rightLt := rightIndex.isLt
          change leftIndex.val < 20 at leftLt
          change rightIndex.val < 41 at rightLt
          omega
      | borrow rightOpening rightIndex =>
          change
            ShiftedTernarySelectiveArtifact.borrowCoordinates.getD
                leftIndex.val 0 =
              ShiftedTernarySelectiveArtifact.borrowCoordinates.getD
                rightIndex.val 0 at equal
          rw [borrowCoordinate_formula, borrowCoordinate_formula] at equal
          have openingEq : leftOpening = rightOpening :=
            Subsingleton.elim _ _
          have indexEq : leftIndex = rightIndex := Fin.ext (by omega)
          cases openingEq
          cases indexEq
          rfl

/-- The exact single-opening layout exported by the selective artifact. -/
def generatedLayout :
    ProductionLayout ShiftedTernarySelectiveArtifact.structureColumnCount 1 where
  column coordinate :=
    ⟨artifactCoordinateNat coordinate,
      artifactCoordinateNat_lt_structure coordinate⟩
  injective := by
    intro left right equal
    apply artifactCoordinateNat_injective
    exact congrArg Fin.val equal

@[simp] theorem generatedLayout_digit
    (opening : Fin 1) (index : Fin digitCount) :
    (generatedLayout.column (.digit opening index)).val =
      ShiftedTernarySelectiveArtifact.digitCoordinates.getD index.val 0 := by
  rfl

@[simp] theorem generatedLayout_borrow
    (opening : Fin 1) (index : Fin chunkBorrowCount) :
    (generatedLayout.column (.borrow opening index)).val =
      ShiftedTernarySelectiveArtifact.borrowCoordinates.getD index.val 0 := by
  rfl

/-- Split-NC covers every generated digit and retained-borrow coordinate. -/
theorem splitNc_covers_generated_opening
    {rows freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows
      ShiftedTernarySelectiveArtifact.structureColumnCount}
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin 1) :
    OpeningCoordinatesBoundTwo generatedLayout data source opening := by
  exact splitNc_covers_opening generatedLayout data truth source opening

/-- Interpret every physical artifact column as the matching authoritative
logical coordinate of one complete Split-NC source. -/
def artifactAssignment
    {rows freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows
      ShiftedTernarySelectiveArtifact.structureColumnCount}
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount) :
    Nat → Nat :=
  fun column =>
    if inRange :
        column < ShiftedTernarySelectiveArtifact.structureColumnCount then
      (data.assignment source
        (Phi81CarrierLayout.embedLogical ⟨column, inRange⟩)).val
    else
      0

theorem artifactAssignment_digit
    {rows freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows
      ShiftedTernarySelectiveArtifact.structureColumnCount}
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin 1) (index : Fin digitCount) :
    artifactAssignment data source
        (ShiftedTernarySelectiveArtifact.digitCoordinates.getD
          index.val 0) =
      (coordinateValue generatedLayout data source
        (.digit opening index)).val := by
  unfold artifactAssignment coordinateValue carrierColumn generatedLayout
  rw [dif_pos (digitCoordinate_lt_structure index)]
  rfl

theorem artifactAssignment_borrow
    {rows freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows
      ShiftedTernarySelectiveArtifact.structureColumnCount}
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin 1) (index : Fin chunkBorrowCount) :
    artifactAssignment data source
        (ShiftedTernarySelectiveArtifact.borrowCoordinates.getD
          index.val 0) =
      (coordinateValue generatedLayout data source
        (.borrow opening index)).val := by
  unfold artifactAssignment coordinateValue carrierColumn generatedLayout
  rw [dif_pos (borrowCoordinate_lt_structure index)]
  rfl

theorem artifactLocalAssignment_digit
    {rows freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows
      ShiftedTernarySelectiveArtifact.structureColumnCount}
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin 1) (index : Fin digitCount) :
    CanonicalOpeningArtifactRows.localAssignment
        (artifactAssignment data source)
        (ShiftedTernary.digitCols.getD index.val 0) =
      localAssignment generatedLayout data source opening
        (ShiftedTernary.digitCols.getD index.val 0) := by
  rw [CanonicalOpeningArtifactRows.localAssignment_digit
      (artifactAssignment data source) index.isLt,
    localAssignment_digit]
  change artifactAssignment data source (108 + index.val) =
    (coordinateValue generatedLayout data source
      (.digit opening index)).val
  rw [show 108 + index.val =
      ShiftedTernarySelectiveArtifact.digitCoordinates.getD
        index.val 0 by
      symm
      exact digitCoordinate_formula index]
  exact artifactAssignment_digit data source opening index

theorem artifactLocalAssignment_borrow
    {rows freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows
      ShiftedTernarySelectiveArtifact.structureColumnCount}
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin 1) (index : Fin chunkBorrowCount) :
    CanonicalOpeningArtifactRows.localAssignment
        (artifactAssignment data source)
        (chunkBorrowColumnBase + index.val) =
      localAssignment generatedLayout data source opening
        (chunkBorrowColumnBase + index.val) := by
  rw [CanonicalOpeningArtifactRows.localAssignment_borrow
      (artifactAssignment data source) index.isLt,
    localAssignment_borrow]
  change artifactAssignment data source (2363 + index.val) =
    (coordinateValue generatedLayout data source
      (.borrow opening index)).val
  rw [show 2363 + index.val =
      ShiftedTernarySelectiveArtifact.borrowCoordinates.getD
        index.val 0 by
      symm
      exact borrowCoordinate_formula index]
  exact artifactAssignment_borrow data source opening index

theorem splitNc_supplies_artifactDigitNorm
    {rows freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows
      ShiftedTernarySelectiveArtifact.structureColumnCount}
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin 1) :
    DigitNormBoundTwo
      (CanonicalOpeningArtifactRows.localAssignment
        (artifactAssignment data source)) := by
  intro index indexLt
  rw [artifactLocalAssignment_digit data source opening
    ⟨index, indexLt⟩]
  exact splitNc_covers_digit generatedLayout data truth source opening
    ⟨index, indexLt⟩

theorem splitNc_supplies_artifactBorrowNorm
    {rows freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows
      ShiftedTernarySelectiveArtifact.structureColumnCount}
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin 1) :
    ∀ index : Fin chunkBorrowCount,
      NormBoundTwo
        (CanonicalOpeningArtifactRows.localAssignment
          (artifactAssignment data source)
          (chunkBorrowColumnBase + index.val)) := by
  intro index
  rw [artifactLocalAssignment_borrow data source opening index]
  exact splitNc_covers_borrow generatedLayout data truth source opening index

private theorem lowValue_artifact_eq_generated
    {rows freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows
      ShiftedTernarySelectiveArtifact.structureColumnCount}
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin 1) :
    lowValue
      (assignmentTritMod
        (CanonicalOpeningArtifactRows.localAssignment
          (artifactAssignment data source)))
      digitCount =
    lowValue
      (assignmentTritMod
        (localAssignment generatedLayout data source opening))
      digitCount := by
  apply
    Nightstream.Implementation.R1CS.ShiftedTernaryComplete.lowValue_congr
  intro index indexLt
  unfold assignmentTritMod
  rw [artifactLocalAssignment_digit data source opening
    ⟨index, indexLt⟩]

/-- End-to-end artifact theorem. Split-NC supplies all 61 coordinate bounds;
the exact generated 21 rows then establish the canonical Goldilocks opening.
No caller-supplied `CanonicalRowsHold` or borrow-Boolean premise remains. -/
theorem splitNc_and_generatedArtifactRows_encoded_lt_modulus
    {rows freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows
      ShiftedTernarySelectiveArtifact.structureColumnCount}
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin 1)
    (one : artifactAssignment data source 0 = 1)
    (selector :
      artifactAssignment data source
          ShiftedTernarySelectiveArtifact.selectorColumn = 1)
    (rowsHold :
      CanonicalOpeningArtifactRows.ArtifactRowsHold
        (artifactAssignment data source)) :
    lowValue
        (assignmentTritMod
          (localAssignment generatedLayout data source opening))
        digitCount <
      goldilocksP := by
  have digitNorm :=
    splitNc_supplies_artifactDigitNorm data truth source opening
  have borrowNorm :=
    splitNc_supplies_artifactBorrowNorm data truth source opening
  have schedule :=
    CanonicalOpeningArtifactRows.artifactRows_imply_chunkScheduleHolds
      (artifactAssignment data source) one selector digitNorm borrowNorm
      rowsHold
  have canonical :=
    chunkSchedule_encoded_lt_modulus digitNorm schedule
  rw [lowValue_artifact_eq_generated data source opening] at canonical
  exact canonical

/-- Artifact-layout specialization of the composed canonicality theorem.

`rowsHold` remains explicit: the generated row indices and coefficients are
not treated as semantic authority by this refinement. -/
theorem splitNc_and_generatedLayoutCanonicalRows_encoded_lt_modulus
    {rows freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows
      ShiftedTernarySelectiveArtifact.structureColumnCount}
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin 1)
    (rowsHold : CanonicalRowsHold generatedLayout data source opening) :
    lowValue
        (assignmentTritMod
          (localAssignment generatedLayout data source opening))
        digitCount <
      goldilocksP := by
  exact splitNc_and_canonicalRows_encoded_lt_modulus
    generatedLayout data truth source opening rowsHold

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement
