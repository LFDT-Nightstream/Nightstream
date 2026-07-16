import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCcsRelation

/-!
CE and norm transport across the ring-aligned F' public boundary.

Owns: preservation and reflection of the verifier-owned norm predicate;
preservation of the CE evaluation-point domain; equality of all matrix
evaluations after zero-column transport; and construction of an honest aligned
canonical CE statement under a newly verifier-owned commitment key.

Does not own: equality of old and aligned commitments, a production packed-X
permutation, Ajtai-key migration, PiCCS acceptance, Rust/R1CS column maps, or
constraint removal.

Emits constraints: no.

Authority boundary: the aligned statement recomputes its commitment from the
aligned opening under its verifier-owned key. This module does not treat an old
commitment or digest as authority for the new opening.

| Branch | Mathematical obligation | Result | Assurance tier |
|---|---|---|---|
| opening.norm | thirteen inserted zeros preserve and reflect the norm bound | `normBounded_insertPublicPadding_iff` | kernel theorem |
| ce.point | the transformed CCS has the same evaluation-point domain | `evaluationPointValid_align_iff` | kernel theorem |
| ce.evaluations | every aligned matrix evaluation equals the original | `matrixEvaluations_align` | kernel theorem |
| opening.commitment | reusing the unmodified legacy key need not preserve a commitment | `legacyKey_commitment_not_preserved` | kernel counterexample |
| ce.completeness | an honest old CE opening yields an honest aligned canonical CE opening | `alignedCanonicalCE_holds` | kernel theorem |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCeRelation

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCcsRelation

private theorem normBounded_append (bound : Nat) (left right : List F) :
    normBounded bound (left ++ right) ↔
      normBounded bound left ∧ normBounded bound right := by
  constructor
  · intro bounded
    constructor
    · intro value member
      exact bounded value (List.mem_append_left right member)
    · intro value member
      exact bounded value (List.mem_append_right left member)
  · rintro ⟨leftBounded, rightBounded⟩ value member
    rcases List.mem_append.mp member with member | member
    · exact leftBounded value member
    · exact rightBounded value member

private theorem normBounded_publicPadding (bound : Nat) (positive : 0 < bound) :
    normBounded bound publicPadding := by
  intro value member
  have valueZero : value = 0 := by
    have facts : paddingWidth ≠ 0 ∧ value = 0 := by
      simpa [publicPadding] using member
    exact facts.2
  subst value
  simpa [centeredMagnitude] using positive

/-- Inserting verifier-fixed zeros cannot hide a norm violation, and it adds
no violation when the verifier-owned bound is positive. -/
theorem normBounded_insertPublicPadding_iff (bound : Nat) (positive : 0 < bound)
    (assignment : Assignment) :
    normBounded bound (insertPublicPadding assignment) ↔
      normBounded bound assignment := by
  unfold insertPublicPadding
  rw [normBounded_append, normBounded_append]
  simp only [normBounded_publicPadding bound positive, and_true]
  rw [← normBounded_append, List.take_append_drop]

/-- Alignment changes only the matrix column carrier, never the row domain or
the multilinear evaluation-point dimension. -/
theorem evaluationPointValid_align_iff (system : Structure) (point : Point)
    (hasPublic : logicalPublicWidth ≤ system.columns)
    (wellFormed : system.WellFormed) :
    evaluationPointValid (alignStructure system) point ↔
      evaluationPointValid system point := by
  constructor
  · intro valid
    exact ⟨wellFormed, by simpa [alignStructure] using valid.2⟩
  · intro valid
    exact ⟨alignStructure_wellFormed system hasPublic wellFormed,
      by simpa [alignStructure] using valid.2⟩

/-- Because every aligned matrix image is exactly the original image, all
ring-valued multilinear evaluations agree coefficient by coefficient. -/
theorem matrixEvaluations_align (system : Structure) (assignment : Assignment)
    (point : Point)
    (hasPublic : logicalPublicWidth ≤ system.columns)
    (assignmentLength : assignment.length = system.columns)
    (wellFormed : system.WellFormed) :
    matrixEvaluations (alignStructure system)
        (insertPublicPadding assignment) point =
      matrixEvaluations system assignment point := by
  unfold matrixEvaluations
  apply congrArg List.toArray
  simp only [alignStructure, List.map_map]
  apply List.map_congr_left
  intro matrix matrixMember
  simp only [Function.comp_apply]
  have assignmentHasPublic : logicalPublicWidth ≤ assignment.length := by
    rw [assignmentLength]
    exact hasPublic
  have imagesEqual := matrixVector_align matrix assignment assignmentHasPublic (by
    intro row rowMember
    have rowLength := (wellFormed.2.1 matrix matrixMember).2 row rowMember
    exact rowLength.trans assignmentLength.symm)
  rw [imagesEqual]

/-! ## Commitment migration necessity -/

/-- The first legacy-private scalar is nonzero and every preceding scalar is
zero. It occupies coefficient 41 of packed ring column four before alignment,
then coefficient zero of packed ring column five after alignment. -/
def commitmentWitnessAssignment : Assignment :=
  List.replicate logicalPublicWidth 0 ++ [1]

/-- One legacy Ajtai row selecting packed ring column four. The unchanged row
contains no coefficient for the new sixth packed column. -/
def legacySelectionKey : AjtaiKey :=
  [List.replicate 4 ringFZero ++ [ringFOne]]

def selectedCommitmentCoefficient (assignment : Assignment) : F :=
  (ajtaiCommit legacySelectionKey assignment).getD 0 ringFZero
    ⟨41, by decide⟩

set_option maxRecDepth 524288 in
/-- Concrete necessity witness: an unmodified legacy key commits to different
values before and after the public-padding insertion. The semantic migration
must therefore recompute commitments under a deliberately specified aligned
key; old commitment equality cannot be assumed. -/
theorem legacyKey_commitment_changes :
    selectedCommitmentCoefficient commitmentWitnessAssignment = 1 ∧
      selectedCommitmentCoefficient
        (insertPublicPadding commitmentWitnessAssignment) = 0 := by
  decide

theorem legacyKey_commitment_not_preserved :
    ajtaiCommit legacySelectionKey
        (insertPublicPadding commitmentWitnessAssignment) ≠
      ajtaiCommit legacySelectionKey commitmentWitnessAssignment := by
  intro equalCommitments
  have equalCoefficient := congrArg
    (fun commitment => commitment.getD 0 ringFZero ⟨41, by decide⟩)
    equalCommitments
  change selectedCommitmentCoefficient
      (insertPublicPadding commitmentWitnessAssignment) =
    selectedCommitmentCoefficient commitmentWitnessAssignment
    at equalCoefficient
  rw [legacyKey_commitment_changes.2, legacyKey_commitment_changes.1]
    at equalCoefficient
  have zeroNotOne : (0 : F) ≠ 1 := by decide
  exact zeroNotOne equalCoefficient

/-- The aligned relation uses a new verifier-owned Ajtai key and exposes all
270 aligned scalar coordinates as public. -/
def alignedContext (ajtaiKey : AjtaiKey) : Context where
  publicWidth := alignedPublicWidth
  ajtaiKey := ajtaiKey

/-- Completeness of the aligned CE opening. This theorem deliberately
recomputes the aligned commitment; it does not claim equality with a legacy
commitment whose packed assignment dimension differs. -/
theorem alignedCanonicalCE_holds (params : GlobalParams) (ajtaiKey : AjtaiKey)
    (system : Structure) (stage : NormStage) (point : Point)
    (assignment : Assignment)
    (positiveBound : 0 < stage.bound params)
    (bounded : normBounded (stage.bound params) assignment)
    (hasPublic : logicalPublicWidth ≤ system.columns)
    (pointValid : evaluationPointValid system point) :
    CE.Holds (relationSemantics (alignedContext ajtaiKey)) params
      (canonicalCEStatement (alignedContext ajtaiKey) (alignStructure system)
        stage point (insertPublicPadding assignment))
      (insertPublicPadding assignment) := by
  apply canonicalCE_holds
  · exact (normBounded_insertPublicPadding_iff
      (stage.bound params) positiveBound assignment).2 bounded
  · exact (evaluationPointValid_align_iff system point hasPublic
      pointValid.1).2 pointValid

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCeRelation
