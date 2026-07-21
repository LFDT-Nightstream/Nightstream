import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PublicAssignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding

/-!
Physical public-prefix contract for the production fixed-point assignment.

Owns: the concrete projection from the first 270 cells of the 11,725,506-cell
production assignment; the exact source-write obligation for cells `0..256`;
and the composition of that dataflow with the 13 generated public-padding
rows.

Does not own: the generated source-write certificate, private-column decoding,
matrix semantics, commitment-key alignment, or protocol acceptance.  In
particular, `PublicSourceDataflow` is the exact Rust-export boundary and is not
semantic authority by itself.

Emits constraints: no.

| Stable stage path | Obligation | Authority class | Lean owner |
|---|---|---|---|
| `f_prime.fixed_point.assignment.public_source` | physical cells `0..256` equal the authoritative legacy source | direct dataflow, pending artifact | `PublicSourceDataflow` |
| `f_prime.fixed_point.assignment.public_padding` | physical cells `257..269` are zero from the exact generated rows | checked plus derived | `projectPhysical270_eq_projectPublicInput` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PhysicalPublicAssignment

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicAssignment
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPaddingRefinement

/-- Embed one typed public coordinate into the exact production assignment
width.  The width is pinned by the generated public-padding artifact rather
than repeated here. -/
def physicalPublicColumn (dimensions : Dimensions)
    (column : Fin dimensions.shape.publicWidth) :
    Fin PublicPaddingRefinement.Artifact.relationColumns :=
  ⟨column.val, by
    have columnBound := column.isLt
    simp only [Dimensions.shape_publicWidth] at columnBound
    simp only [PublicPaddingRefinement.Artifact.relationColumns,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.relationColumns]
    omega⟩

/-- The physical production assignment projected to its typed 270-coordinate
public carrier. -/
def projectPhysical270 (dimensions : Dimensions)
    (encoded : Fin PublicPaddingRefinement.Artifact.relationColumns → F) :
    PublicInput dimensions.shape :=
  fun column => encoded (physicalPublicColumn dimensions column)

/-- Replay the generated public-owner program into the exact physical
assignment width, leaving the private suffix unchanged.  The public owner
program itself is Rust-generated and fail-closed in `PublicDecoder`. -/
def replayPhysicalPublicAssignment (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns → F) :
    Fin PublicPaddingRefinement.Artifact.relationColumns → F :=
  fun column =>
    if isPublic : column.val < PublicDecoder.alignedPublicWidth then
      artifactPublicValue (sourcePublicPrefix dimensions legacy)
        ⟨column.val, isPublic⟩
    else
      suffix column

theorem replayPhysicalPublicAssignment_at_public
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns → F)
    (column : Fin dimensions.shape.publicWidth) :
    replayPhysicalPublicAssignment dimensions legacy suffix
        (physicalPublicColumn dimensions column) =
      artifactPublicInput dimensions legacy column := by
  have publicBound :
      (physicalPublicColumn dimensions column).val <
        PublicDecoder.alignedPublicWidth := by
    have columnBound := column.isLt
    simpa [Dimensions.shape_publicWidth, PublicDecoder.alignedPublicWidth]
      using columnBound
  rw [replayPhysicalPublicAssignment, dif_pos publicBound]
  unfold artifactPublicInput
  congr 1

/-- The generated owner replay, embedded into the full physical width, is
exactly the artifact-decoded public assignment. -/
theorem projectPhysical270_replay_eq_artifactPublicInput
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns → F) :
    projectPhysical270 dimensions
        (replayPhysicalPublicAssignment dimensions legacy suffix) =
      artifactPublicInput dimensions legacy := by
  funext column
  exact replayPhysicalPublicAssignment_at_public dimensions legacy suffix column

/-- Artifact-checked physical replay of the 270 generated owners equals the
independently typed carrier.  The only semantic premise is the conventional
constant-one condition on the authoritative source assignment. -/
theorem projectPhysical270_replay_eq_projectPublicInput
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns → F)
    (constantOne : SourceConstantOne dimensions legacy) :
    projectPhysical270 dimensions
        (replayPhysicalPublicAssignment dimensions legacy suffix) =
      projectPublicInput (assignment dimensions legacy) := by
  exact (projectPhysical270_replay_eq_artifactPublicInput
    dimensions legacy suffix).trans
      (artifactPublicInput_eq_projectPublicInput dimensions legacy constantOne)

/-- Exact dataflow obligation for the non-padding public prefix.  The owning
Rust exporter must discharge this against the actual assignment encoder; a
digest or prover-carried public vector cannot discharge it. -/
def PublicSourceDataflow (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (encoded : Fin PublicPaddingRefinement.Artifact.relationColumns → F) : Prop :=
  ∀ (column : Fin dimensions.shape.publicWidth)
    (isLegacy : column.val < legacyPublicWidth),
      encoded (physicalPublicColumn dimensions column) =
        legacy ⟨column.val,
          Nat.lt_of_lt_of_le isLegacy dimensions.legacyPublicFits⟩

private theorem physicalConstantOne
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (encoded : Fin PublicPaddingRefinement.Artifact.relationColumns → F)
    (sourceDataflow : PublicSourceDataflow dimensions legacy encoded)
    (constantOne : SourceConstantOne dimensions legacy) :
    encoded PublicPaddingRefinement.constantColumn = 1 := by
  let publicZero : Fin dimensions.shape.publicWidth :=
    ⟨0, by simp [Dimensions.shape_publicWidth]⟩
  have publicZeroLegacy : publicZero.val < legacyPublicWidth := by
    change 0 < 257
    decide
  have dataflow := sourceDataflow publicZero publicZeroLegacy
  have samePhysical :
      physicalPublicColumn dimensions publicZero =
        PublicPaddingRefinement.constantColumn := by
    apply Fin.ext
    rfl
  rw [samePhysical] at dataflow
  simpa [SourceConstantOne, publicZero] using dataflow.trans constantOne

/-- The generated public-owner replay discharges the concrete source-write
contract; no caller-provided coordinate equality remains. -/
theorem replayPhysicalPublicAssignment_sourceDataflow
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns → F)
    (constantOne : SourceConstantOne dimensions legacy) :
    PublicSourceDataflow dimensions legacy
      (replayPhysicalPublicAssignment dimensions legacy suffix) := by
  intro column isLegacy
  have projected := congrFun
    (projectPhysical270_replay_eq_projectPublicInput dimensions legacy suffix
      constantOne) column
  rw [projectPublicInput_exact] at projected
  simpa [projectPhysical270, expectedPublicInput, isLegacy] using projected

theorem replayPhysicalPublicAssignment_constantOne
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns → F)
    (constantOne : SourceConstantOne dimensions legacy) :
    replayPhysicalPublicAssignment dimensions legacy suffix
        PublicPaddingRefinement.constantColumn = 1 := by
  exact physicalConstantOne dimensions legacy
    (replayPhysicalPublicAssignment dimensions legacy suffix)
    (replayPhysicalPublicAssignment_sourceDataflow dimensions legacy suffix
      constantOne)
    constantOne

/-- Actual source writes plus the generated padding equations identify the
first 270 physical cells with the independently typed carrier projection.

This is deliberately not the final artifact theorem: the next generated leaf
must derive `PublicSourceDataflow` from the production assignment encoder. -/
theorem projectPhysical270_eq_projectPublicInput
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (encoded : Fin PublicPaddingRefinement.Artifact.relationColumns → F)
    (sourceDataflow : PublicSourceDataflow dimensions legacy encoded)
    (constantOne : SourceConstantOne dimensions legacy)
    (paddingRows : PublicPaddingRefinement.GeneratedRowsSatisfied encoded) :
    projectPhysical270 dimensions encoded =
      projectPublicInput (assignment dimensions legacy) := by
  rw [projectPublicInput_exact]
  funext column
  unfold projectPhysical270 expectedPublicInput
  by_cases isLegacy : column.val < legacyPublicWidth
  · rw [dif_pos isLegacy]
    exact sourceDataflow column isLegacy
  · rw [dif_neg isLegacy]
    have columnBound := column.isLt
    have lower : legacyPublicWidth ≤ column.val := Nat.not_lt.mp isLegacy
    have upper : column.val < legacyPublicWidth +
        PublicPaddingRefinement.Artifact.paddingWidth := by
      simp only [Dimensions.shape_publicWidth, legacyPublicWidth,
        PublicPaddingRefinement.Artifact.paddingWidth,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.paddingWidth]
        at columnBound ⊢
      omega
    let offset : Fin PublicPaddingRefinement.Artifact.paddingWidth :=
      ⟨column.val - legacyPublicWidth, by omega⟩
    have samePhysical :
        physicalPublicColumn dimensions column =
          PublicPaddingRefinement.paddingColumn offset := by
      apply Fin.ext
      change column.val = 257 + (column.val - 257)
      have lowerConcrete : 257 ≤ column.val := by
        simpa [legacyPublicWidth] using lower
      omega
    have paddingZero :=
      (PublicPaddingRefinement.generatedRowsSatisfied_iff_padding_zero encoded
        (physicalConstantOne dimensions legacy encoded sourceDataflow constantOne)).1
        paddingRows offset
    rw [samePhysical]
    exact paddingZero

/-- The exact generated public-padding rows hold on the generated physical
owner replay.  This joins the owner schedule to the already checked physical
row equations without treating either artifact as semantic authority. -/
theorem replayPhysicalPublicAssignment_paddingRowsSatisfied
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns → F)
    (constantOne : SourceConstantOne dimensions legacy) :
    PublicPaddingRefinement.GeneratedRowsSatisfied
      (replayPhysicalPublicAssignment dimensions legacy suffix) := by
  let encoded := replayPhysicalPublicAssignment dimensions legacy suffix
  have sourceDataflow : PublicSourceDataflow dimensions legacy encoded :=
    replayPhysicalPublicAssignment_sourceDataflow dimensions legacy suffix
      constantOne
  have encodedOne : encoded PublicPaddingRefinement.constantColumn = 1 :=
    physicalConstantOne dimensions legacy encoded sourceDataflow constantOne
  rw [PublicPaddingRefinement.generatedRowsSatisfied_iff_padding_zero encoded
    encodedOne]
  intro offset
  let publicColumn : Fin dimensions.shape.publicWidth :=
    ⟨257 + offset.val, by
      have offsetBound := offset.isLt
      simp only [Dimensions.shape_publicWidth,
        PublicPaddingRefinement.Artifact.paddingWidth,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.paddingWidth]
        at offsetBound ⊢
      omega⟩
  have physicalEq :
      physicalPublicColumn dimensions publicColumn =
        PublicPaddingRefinement.paddingColumn offset := by
    apply Fin.ext
    rfl
  have projected := congrFun
    (projectPhysical270_replay_eq_projectPublicInput dimensions legacy suffix
      constantOne) publicColumn
  have typedZero := assignment_fixedPublicPadding dimensions legacy
    (PublicPaddingRefinement.typedOffset offset)
  have carrierEq :
      dimensions.shape.publicColumn publicColumn =
        paddingCarrierColumn dimensions
          (PublicPaddingRefinement.typedOffset offset) := by
    apply Fin.ext
    rfl
  rw [projectPhysical270, physicalEq,
    Nightstream.SuperNeo.Concrete.Phi81Relation.projectPublicInput, carrierEq,
    typedZero] at projected
  simpa [encoded] using projected

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PhysicalPublicAssignment
