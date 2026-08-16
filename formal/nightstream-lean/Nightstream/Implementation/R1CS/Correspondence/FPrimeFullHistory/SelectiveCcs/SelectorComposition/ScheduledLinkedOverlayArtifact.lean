import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.ScheduledLinkedOverlayFixture
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.ScheduledLinkedOverlay
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Rows

/-!
Contract: exact generated-row bridge for the schedule-linked overlay fixture.

Owns fail-closed fixture decoding, exact overlay selector placement, radix-three
decoding of both linked private fields, and selective-polynomial evaluation of
every emitted overlay equality, activation, field-link, and padding row.

The Rust owner compares all thirteen matrix ports with these row recipes. Lean
proves that zero residuals are exactly the overlay fields used by
`ScheduledLinkedOverlay.LinkedAccepts`.

Does not own source component semantics, production dimensions, or recursive
and terminal F-prime relations.

Emits constraints: eight checked rows in the generated fixture.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledLinkedOverlayArtifact

open Nightstream.Implementation.R1CS.FPrimeFullHistoryScheduledLinkedOverlayFixture.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics
open Nightstream.SuperNeo.Concrete

abbrev rawArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.ScheduledLinkedOverlayFixture.rawArtifact

theorem rawArtifact_valid : Valid rawArtifact := by
  decide

def artifact : Decoded := ⟨rawArtifact, rawArtifact_valid⟩

theorem decode_rawArtifact : decode rawArtifact = some artifact := by
  simp [decode, artifact, rawArtifact_valid]

@[simp] theorem rows_exact : rawArtifact.rows = 384 := by
  rfl

@[simp] theorem columns_exact : rawArtifact.columns = 540 := by
  rfl

@[simp] theorem publicColumns_exact : rawArtifact.publicColumns = 54 := by
  rfl

@[simp] theorem row_boundaries_exact :
    (rawArtifact.scheduledRowEnd,
      rawArtifact.overlayRowEnd,
      rawArtifact.overlayKindEqualityRowEnd,
      rawArtifact.overlayActivationRowEnd,
      rawArtifact.fieldLinkRowEnd,
      rawArtifact.ringPaddingRowEnd) =
    (348, 376, 378, 381, 383, 384) := by
  rfl

@[simp] theorem field_geometry_exact :
    (rawArtifact.phaseFieldStarts, rawArtifact.overlayFieldStarts,
      rawArtifact.fieldWidths, rawArtifact.fieldRadices) =
    ([270, 270], [434, 434], [41, 41], [3, 3]) := by
  rfl

def scheduleSelector : Fin 3 → Fin 540
  | ⟨0, _⟩ => ⟨378, by decide⟩
  | ⟨1, _⟩ => ⟨379, by decide⟩
  | ⟨2, _⟩ => ⟨380, by decide⟩

def overlaySelector : Fin 2 → Fin 540
  | ⟨0, _⟩ => ⟨432, by decide⟩
  | ⟨1, _⟩ => ⟨433, by decide⟩

def lifecycleOf : Fin 3 → Fin 2
  | ⟨0, _⟩ => 0
  | ⟨1, _⟩ => 1
  | ⟨2, _⟩ => 1

def phaseKindOf : Fin 3 → Fin 2
  | ⟨0, _⟩ => 0
  | ⟨1, _⟩ => 1
  | ⟨2, _⟩ => 0

def overlayKindOf : Fin 3 → Fin 2
  | ⟨0, _⟩ => 0
  | ⟨1, _⟩ => 1
  | ⟨2, _⟩ => 0

@[simp] theorem scheduleSelector_values :
    (List.ofFn fun arm : Fin 3 => (scheduleSelector arm).val) =
      [378, 379, 380] := by
  decide

@[simp] theorem overlaySelector_values :
    (List.ofFn fun kind : Fin 2 => (overlaySelector kind).val) =
      [432, 433] := by
  decide

@[simp] theorem schedule_maps_exact :
    ((List.ofFn fun arm : Fin 3 => (lifecycleOf arm).val),
      (List.ofFn fun arm : Fin 3 => (phaseKindOf arm).val),
      (List.ofFn fun arm : Fin 3 => (overlayKindOf arm).val)) =
    ([0, 1, 1], [0, 1, 0], [0, 1, 0]) := by
  decide

def scheduleWeights (assignment : Fin 540 → F) : Fin 3 → F :=
  fun arm => assignment (scheduleSelector arm)

def storedOverlayWeights (assignment : Fin 540 → F) : Fin 2 → F :=
  fun kind => assignment (overlaySelector kind)

/-- Little-endian Horner decoding of one retained low-norm word. -/
def decodeRadix (radix : F) : List F → F
  | [] => 0
  | digit :: rest => digit + radix * decodeRadix radix rest

/-- Exact radix-three value of the linked phase word. -/
def phaseFieldValue (assignment : Fin 540 → F) (_kind : Fin 2) : F :=
  decodeRadix 3 <| List.ofFn fun digit : Fin 41 =>
    assignment ⟨270 + digit.val, by omega⟩

/-- Exact radix-three value of the linked overlay word. -/
def overlayFieldValue (assignment : Fin 540 → F) (_kind : Fin 2) : F :=
  decodeRadix 3 <| List.ofFn fun digit : Fin 41 =>
    assignment ⟨434 + digit.val, by omega⟩

def overlayEqualityGap
    (assignment : Fin 540 → F) (kind : Fin 2) : F :=
  storedOverlayWeights assignment kind -
    groupWeight overlayKindOf (scheduleWeights assignment) kind

def overlayActivationGap
    (assignment : Fin 540 → F) (arm : Fin 3) : F :=
  scheduleWeights assignment arm *
      storedOverlayWeights assignment (overlayKindOf arm) -
    scheduleWeights assignment arm

def fieldLinkGap
    (assignment : Fin 540 → F) (kind : Fin 2) : F :=
  phaseFieldValue assignment kind - overlayFieldValue assignment kind

inductive FieldLinkRow where
  | exactValue

def fieldLinkResidual
    (assignment : Fin 540 → F) (kind : Fin 2) : FieldLinkRow → F
  | .exactValue => fieldLinkGap assignment kind

def fieldLinkRows
    (assignment : Fin 540 → F) (kind : Fin 2) : ResidualFamily where
  Row := FieldLinkRow
  residual := fieldLinkResidual assignment kind

/-- Exact matrix-image point of one overlay-selector equality row. -/
def overlayEqualityPoint
    (assignment : Fin 540 → F) (kind : Fin 2) : Fin 13 → F :=
  productPoint 1 0 0 (overlayEqualityGap assignment kind)

/-- Exact matrix-image point of one overlay activation row. -/
def overlayActivationPoint
    (assignment : Fin 540 → F) (arm : Fin 3) : Fin 13 → F :=
  productPoint 1 (scheduleWeights assignment arm)
    (storedOverlayWeights assignment (overlayKindOf arm))
    (scheduleWeights assignment arm)

/-- Exact matrix-image point of one decoded-field equality row. -/
def fieldLinkPoint
    (assignment : Fin 540 → F) (kind : Fin 2) : Fin 13 → F :=
  productPoint 1 (storedOverlayWeights assignment kind)
    (fieldLinkGap assignment kind) 0

/-- Exact matrix-image point of the one ring-padding row. -/
def paddingPoint (assignment : Fin 540 → F) : Fin 13 → F :=
  productPoint 1 0 0 (assignment ⟨539, by decide⟩)

private theorem evaluate_linearPoint (gap : F) :
    evaluate (productPoint 1 0 0 gap) = -gap := by
  rw [evaluate_productPoint]
  simp [productResidual, productPoint, sparsePoint,
    Role.index, Fin.mul_zero, Fin.one_mul, Fin.zero_add]

private theorem neg_eq_zero_iff (value : F) :
    -value = 0 ↔ value = 0 := by
  constructor
  · intro negZero
    have := congrArg Neg.neg negZero
    simpa only [Lean.Grind.AddCommGroup.neg_neg,
      Lean.Grind.AddCommGroup.neg_zero] using this
  · intro zero
    rw [zero, Lean.Grind.AddCommGroup.neg_zero]

theorem overlayEqualityPoint_zero_iff
    (assignment : Fin 540 → F) (kind : Fin 2) :
    evaluate (overlayEqualityPoint assignment kind) = 0 ↔
      storedOverlayWeights assignment kind =
        groupWeight overlayKindOf (scheduleWeights assignment) kind := by
  rw [overlayEqualityPoint, evaluate_linearPoint, neg_eq_zero_iff,
    overlayEqualityGap, Lean.Grind.AddCommGroup.sub_eq_zero_iff]

theorem evaluate_overlayActivationPoint
    (assignment : Fin 540 → F) (arm : Fin 3) :
    evaluate (overlayActivationPoint assignment arm) =
      overlayActivationGap assignment arm := by
  unfold overlayActivationPoint
  rw [evaluate_productPoint]
  simp [productResidual, overlayActivationGap, productPoint,
    sparsePoint, Role.index, Fin.one_mul, Fin.sub_eq_add_neg]

theorem overlayActivationPoint_zero_iff
    (assignment : Fin 540 → F) (arm : Fin 3) :
    evaluate (overlayActivationPoint assignment arm) = 0 ↔
      scheduleWeights assignment arm *
          storedOverlayWeights assignment (overlayKindOf arm) =
        scheduleWeights assignment arm := by
  rw [evaluate_overlayActivationPoint, overlayActivationGap,
    Lean.Grind.AddCommGroup.sub_eq_zero_iff]

theorem evaluate_fieldLinkPoint
    (assignment : Fin 540 → F) (kind : Fin 2) :
    evaluate (fieldLinkPoint assignment kind) =
      storedOverlayWeights assignment kind * fieldLinkGap assignment kind := by
  unfold fieldLinkPoint
  rw [evaluate_productPoint]
  simp [productResidual, productPoint, sparsePoint,
    Role.index, Fin.one_mul, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

theorem fieldLinkPoint_zero_iff_gated
    (assignment : Fin 540 → F) (kind : Fin 2) :
    evaluate (fieldLinkPoint assignment kind) = 0 ↔
      GatedRowsZero (storedOverlayWeights assignment kind)
        (fieldLinkRows assignment kind) := by
  rw [evaluate_fieldLinkPoint]
  constructor
  · intro zero row
    cases row with
    | exactValue => exact zero
  · intro gated
    exact gated .exactValue

theorem paddingPoint_zero_iff (assignment : Fin 540 → F) :
    evaluate (paddingPoint assignment) = 0 ↔
      assignment ⟨539, by decide⟩ = 0 := by
  rw [paddingPoint, evaluate_linearPoint, neg_eq_zero_iff]

/-- Satisfaction of all eight generated overlay-link and padding rows. -/
structure LinkRowsHold (assignment : Fin 540 → F) : Prop where
  overlayEqualities : ∀ kind,
    evaluate (overlayEqualityPoint assignment kind) = 0
  overlayActivations : ∀ arm,
    evaluate (overlayActivationPoint assignment arm) = 0
  fieldLinks : ∀ kind,
    evaluate (fieldLinkPoint assignment kind) = 0
  padding : evaluate (paddingPoint assignment) = 0

theorem linkRowsHold_iff (assignment : Fin 540 → F) :
    LinkRowsHold assignment ↔
      (∀ kind,
        storedOverlayWeights assignment kind =
          groupWeight overlayKindOf (scheduleWeights assignment) kind) ∧
      (∀ arm,
        scheduleWeights assignment arm *
            storedOverlayWeights assignment (overlayKindOf arm) =
          scheduleWeights assignment arm) ∧
      (∀ kind,
        GatedRowsZero (storedOverlayWeights assignment kind)
          (fieldLinkRows assignment kind)) ∧
      assignment ⟨539, by decide⟩ = 0 := by
  constructor
  · intro rows
    exact ⟨
      fun kind => (overlayEqualityPoint_zero_iff assignment kind).mp
        (rows.overlayEqualities kind),
      fun arm => (overlayActivationPoint_zero_iff assignment arm).mp
        (rows.overlayActivations arm),
      fun kind => (fieldLinkPoint_zero_iff_gated assignment kind).mp
        (rows.fieldLinks kind),
      (paddingPoint_zero_iff assignment).mp rows.padding⟩
  · rintro ⟨overlayEqualities, overlayActivations, fieldLinks, padding⟩
    exact {
      overlayEqualities := fun kind =>
        (overlayEqualityPoint_zero_iff assignment kind).mpr
          (overlayEqualities kind)
      overlayActivations := fun arm =>
        (overlayActivationPoint_zero_iff assignment arm).mpr
          (overlayActivations arm)
      fieldLinks := fun kind =>
        (fieldLinkPoint_zero_iff_gated assignment kind).mpr
          (fieldLinks kind)
      padding := (paddingPoint_zero_iff assignment).mpr padding
    }

/-- Source component rows and every generated overlay-link row on one
assignment. -/
def ComposedRowsHold
    (assignment : Fin 540 → F)
    (lifecycleWeights phaseKindWeights : Fin 2 → F)
    (common phaseKinds : Fin 2 → ResidualFamily)
    (schedule : Fin 3 → ResidualFamily)
    (overlays : Fin 2 → ResidualFamily) : Prop :=
  ScheduledGrouped.LinkedAccepts lifecycleOf phaseKindOf
      (scheduleWeights assignment) lifecycleWeights phaseKindWeights common
      phaseKinds schedule ∧
    (∀ kind,
      GatedRowsZero (storedOverlayWeights assignment kind) (overlays kind)) ∧
    LinkRowsHold assignment

theorem composedRowsHold_iff_linkedAccepts_and_padding
    (assignment : Fin 540 → F)
    (lifecycleWeights phaseKindWeights : Fin 2 → F)
    (common phaseKinds : Fin 2 → ResidualFamily)
    (schedule : Fin 3 → ResidualFamily)
    (overlays : Fin 2 → ResidualFamily) :
    ComposedRowsHold assignment lifecycleWeights phaseKindWeights common
        phaseKinds schedule overlays ↔
      ScheduledLinkedOverlay.LinkedAccepts lifecycleOf phaseKindOf
          overlayKindOf (scheduleWeights assignment) lifecycleWeights
          phaseKindWeights (storedOverlayWeights assignment) common phaseKinds
          schedule overlays (fieldLinkRows assignment) ∧
        assignment ⟨539, by decide⟩ = 0 := by
  constructor
  · rintro ⟨scheduled, overlayGated, linkRows⟩
    rcases (linkRowsHold_iff assignment).mp linkRows with
      ⟨overlayEqualities, overlayLinks, fieldLinksGated, padding⟩
    exact ⟨{
      scheduled := scheduled
      overlayEqualities := overlayEqualities
      overlayGated := overlayGated
      overlayLinks := overlayLinks
      fieldLinksGated := fieldLinksGated
    }, padding⟩
  · rintro ⟨accepted, padding⟩
    refine ⟨accepted.scheduled, accepted.overlayGated, ?_⟩
    exact (linkRowsHold_iff assignment).mpr ⟨accepted.overlayEqualities,
      accepted.overlayLinks, accepted.fieldLinksGated, padding⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledLinkedOverlayArtifact
