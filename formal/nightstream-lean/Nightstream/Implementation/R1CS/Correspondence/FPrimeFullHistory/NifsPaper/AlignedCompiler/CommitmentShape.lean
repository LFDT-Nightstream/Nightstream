import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.MatrixMap
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedNifsRepair

/-!
Ajtai packing and setup-shape refinement for the aligned F' compiler.

Owns: the mathematical setup-column formula; exact agreement with the
54-lane assignment packing; verifier-owned key shape; and a concrete proof
that reusing an old key can change the commitment even when old and aligned
assignments require the same number of Ajtai columns.

Does not own: Rust `usize::div_ceil`, the process-global Ajtai registry, key
generation or serialization, a production key artifact, Poseidon2 key
digests, emitted opening rows, or permission to reuse a legacy commitment.

Emits constraints: no.

Authority boundary: setup dimensions authorize only array shape. The aligned
commitment must be recomputed from the complete aligned assignment under the
verifier-owned aligned key; equal dimensions do not authorize key or
commitment reuse.

| Protocol | Phase | Constraint family | Mathematical obligation | Result |
|---|---|---|---|---|
| SuperNeo CE | setup | ring columns | setup uses exactly `ceil(columns / 54)` | `setupColumns_eq_blockCount` |
| SuperNeo CE | assignment packing | packed width | packing emits exactly the setup-column count | `packAssignment_length` |
| SuperNeo CE | verifier key | row shape | every verifier-owned key row matches the packed opening | `alignedKeyShape_matches_packing` |
| F' carrier | public projection | active ring columns | 257 and repaired 270 public scalars both occupy five columns | `fixedPublicSetupColumns` |
| F' carrier | total opening | paper alignment | a paper-aligned total width has its exact declared ring-column count | `paperDimensions_setupColumns` |
| SuperNeo CE | commitment authority | legacy-key reuse | equal setup dimensions do not preserve the commitment | `sameShape_legacyKey_commitment_not_preserved` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCcsRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair

/-- Mathematical counterpart of the production Ajtai setup width. The Rust
correspondence remains a separate obligation. -/
def setupColumns (columns : Nat) : Nat :=
  (columns + ringDegree - 1) / ringDegree

theorem setupColumns_eq_blockCount (columns : Nat) :
    setupColumns columns = Phi81ColumnLayout.blockCount columns := rfl

/-- The semantic assignment packer and setup formula allocate the same number
of ring columns for every scalar width. -/
theorem packAssignment_length (assignment : Assignment) :
    (packAssignment assignment).length = setupColumns assignment.length := by
  simp [packAssignment, setupColumns]

/-- Verifier-owned Ajtai key shape: `kappa` rows, each spanning every packed
ring column of the authoritative opening. -/
def KeyShape (kappa columns : Nat) (key : AjtaiKey) : Prop :=
  key.length = kappa ∧
    ∀ row ∈ key, row.length = setupColumns columns

def AlignedKeyShape (kappa : Nat) (system : Structure) (key : AjtaiKey) : Prop :=
  KeyShape kappa (alignStructure system).columns key

/-- Every row of a correctly shaped aligned key spans exactly the packed
aligned assignment. -/
theorem alignedKeyShape_matches_packing
    (kappa : Nat) (system : Structure) (key : AjtaiKey)
    (assignment : Assignment)
    (keyShape : AlignedKeyShape kappa system key)
    (hasPublic : logicalPublicWidth ≤ system.columns)
    (assignmentLength : assignment.length = system.columns)
    (row : List RingF) (rowMember : row ∈ key) :
    row.length = (packAssignment (insertPublicPadding assignment)).length := by
  rw [packAssignment_length]
  rw [insertPublicPadding_length assignment (assignmentLength ▸ hasPublic)]
  simpa [AlignedKeyShape, alignStructure, assignmentLength] using
    keyShape.2 row rowMember

theorem setupColumns_exactRingWidth (ringColumns : Nat) :
    setupColumns (ringDegree * ringColumns) = ringColumns := by
  simp [setupColumns, ringDegree]
  omega

/-- Paper dimension eligibility determines the exact setup width, rather than
merely an upper bound or a padded carrier size. -/
theorem paperDimensions_setupColumns (system : Structure)
    (dimensions : PaperDimensions system) :
    ∃ ringColumns,
      (alignStructure system).columns = ringDegree * ringColumns ∧
      setupColumns (alignStructure system).columns = ringColumns := by
  rcases dimensions.2.2 with ⟨ringColumns, exactWidth⟩
  refine ⟨ringColumns, exactWidth, ?_⟩
  rw [exactWidth]
  exact setupColumns_exactRingWidth ringColumns

/-- The logical and paper-visible public carriers have the same setup column
count, although their coefficient ownership is different. -/
theorem fixedPublicSetupColumns :
    setupColumns logicalPublicWidth = 5 ∧
      setupColumns alignedPublicWidth = 5 := by
  simpa [setupColumns_eq_blockCount] using
    Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.fixedCarrier_blockCounts

/-! ## Same-shape key-reuse counterexample -/

/-- An eligible legacy width `311 = 54 * 6 - 13`. The nonzero scalar is the
first legacy-private coordinate and moves from block 4/lane 41 to block
5/lane 0 after alignment. -/
def sameShapeWitnessAssignment : Assignment :=
  List.replicate logicalPublicWidth 0 ++ [1] ++ List.replicate 53 0

/-- One six-column key row selects old block four and assigns zero authority
to block five. It has valid row width before and after alignment. -/
def sameShapeSelectionKey : AjtaiKey :=
  [List.replicate 4 ringFZero ++ [ringFOne, ringFZero]]

def selectedCoefficient (assignment : Assignment) : F :=
  (ajtaiCommit sameShapeSelectionKey assignment).getD 0 ringFZero
    ⟨41, by decide⟩

set_option maxRecDepth 524288 in
theorem sameShape_setupFacts :
    sameShapeWitnessAssignment.length = 311 ∧
      (insertPublicPadding sameShapeWitnessAssignment).length = 324 ∧
      setupColumns sameShapeWitnessAssignment.length = 6 ∧
      setupColumns (insertPublicPadding sameShapeWitnessAssignment).length = 6 ∧
      KeyShape 1 sameShapeWitnessAssignment.length sameShapeSelectionKey ∧
      KeyShape 1 (insertPublicPadding sameShapeWitnessAssignment).length
        sameShapeSelectionKey := by
  have oldLength : sameShapeWitnessAssignment.length = 311 := by decide
  have alignedLength :
      (insertPublicPadding sameShapeWitnessAssignment).length = 324 := by
    decide
  have oldColumns : setupColumns sameShapeWitnessAssignment.length = 6 := by
    decide
  have alignedColumns :
      setupColumns (insertPublicPadding sameShapeWitnessAssignment).length = 6 := by
    decide
  have keyAtOld :
      KeyShape 1 sameShapeWitnessAssignment.length sameShapeSelectionKey := by
    constructor
    · decide
    · intro row rowMember
      have rowEqual :
          row = List.replicate 4 ringFZero ++ [ringFOne, ringFZero] := by
        simpa [sameShapeSelectionKey] using rowMember
      subst row
      calc
        (List.replicate 4 ringFZero ++ [ringFOne, ringFZero]).length = 6 := by
          decide
        _ = setupColumns sameShapeWitnessAssignment.length := oldColumns.symm
  have keyAtAligned :
      KeyShape 1 (insertPublicPadding sameShapeWitnessAssignment).length
        sameShapeSelectionKey := by
    constructor
    · decide
    · intro row rowMember
      have rowEqual :
          row = List.replicate 4 ringFZero ++ [ringFOne, ringFZero] := by
        simpa [sameShapeSelectionKey] using rowMember
      subst row
      calc
        (List.replicate 4 ringFZero ++ [ringFOne, ringFZero]).length = 6 := by
          decide
        _ = setupColumns
            (insertPublicPadding sameShapeWitnessAssignment).length :=
          alignedColumns.symm
  exact ⟨oldLength, alignedLength, oldColumns, alignedColumns,
    keyAtOld, keyAtAligned⟩

set_option maxRecDepth 524288 in
/-- Kernel-checked necessity witness: setup dimensions and key row dimensions
can remain identical while the coefficient relocation changes the committed
value. -/
theorem sameShape_legacyKey_commitment_changes :
    selectedCoefficient sameShapeWitnessAssignment = 1 ∧
      selectedCoefficient
        (insertPublicPadding sameShapeWitnessAssignment) = 0 := by
  decide

theorem sameShape_legacyKey_commitment_not_preserved :
    ajtaiCommit sameShapeSelectionKey
        (insertPublicPadding sameShapeWitnessAssignment) ≠
      ajtaiCommit sameShapeSelectionKey sameShapeWitnessAssignment := by
  intro equalCommitments
  have equalCoefficient := congrArg
    (fun commitment => commitment.getD 0 ringFZero ⟨41, by decide⟩)
    equalCommitments
  change selectedCoefficient (insertPublicPadding sameShapeWitnessAssignment) =
    selectedCoefficient sameShapeWitnessAssignment at equalCoefficient
  rw [sameShape_legacyKey_commitment_changes.2,
    sameShape_legacyKey_commitment_changes.1] at equalCoefficient
  have zeroNotOne : (0 : F) ≠ 1 := by decide
  exact zeroNotOne equalCoefficient

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape
