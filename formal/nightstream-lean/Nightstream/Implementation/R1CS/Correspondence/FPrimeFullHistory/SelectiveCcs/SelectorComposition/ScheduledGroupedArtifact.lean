import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.ScheduledGroupedPhaseFixture
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.ScheduledGrouped
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Rows

/-!
Contract: exact generated-row bridge for the scheduled grouped-phase fixture.

Owns fail-closed fixture decoding, the exact lifecycle, phase-kind, and
schedule selector placement, and selective-polynomial evaluation of every
emitted schedule total, selector equality, activation, and cursor row.

The Rust owner compares all thirteen matrix ports with these row recipes.
Lean proves that their zero residuals are exactly the link and schedule fields
used by `ScheduledGrouped.LinkedAccepts`.

Does not own either component relation, production schedule dimensions,
phase semantics, or the recursive and terminal F-prime relations.

Emits constraints: seventeen checked link rows in the generated fixture.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGroupedArtifact

open Nightstream.Implementation.R1CS.FPrimeFullHistoryScheduledGroupedPhaseFixture.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGrouped
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics
open Nightstream.SuperNeo.Concrete

abbrev rawArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.ScheduledGroupedPhaseFixture.rawArtifact

theorem rawArtifact_valid : Valid rawArtifact := by
  decide

def artifact : Decoded := ⟨rawArtifact, rawArtifact_valid⟩

theorem decode_rawArtifact : decode rawArtifact = some artifact := by
  simp [decode, artifact, rawArtifact_valid]

@[simp] theorem rows_exact : rawArtifact.rows = 406 := by
  rfl

@[simp] theorem columns_exact : rawArtifact.columns = 324 := by
  rfl

@[simp] theorem publicColumns_exact : rawArtifact.publicColumns = 54 := by
  rfl

@[simp] theorem commonRowEnd_exact : rawArtifact.commonRowEnd = 169 := by
  rfl

@[simp] theorem phaseRowEnd_exact : rawArtifact.phaseRowEnd = 338 := by
  rfl

@[simp] theorem scheduleTotalRowEnd_exact :
    rawArtifact.scheduleTotalRowEnd = 339 := by
  rfl

@[simp] theorem lifecycleEqualityRowEnd_exact :
    rawArtifact.lifecycleEqualityRowEnd = 341 := by
  rfl

@[simp] theorem phaseKindEqualityRowEnd_exact :
    rawArtifact.phaseKindEqualityRowEnd = 343 := by
  rfl

@[simp] theorem lifecycleActivationRowEnd_exact :
    rawArtifact.lifecycleActivationRowEnd = 346 := by
  rfl

@[simp] theorem phaseKindActivationRowEnd_exact :
    rawArtifact.phaseKindActivationRowEnd = 349 := by
  rfl

@[simp] theorem cursorBindingRowEnd_exact :
    rawArtifact.cursorBindingRowEnd = 355 := by
  rfl

def commonSelector : Fin 2 → Fin 324
  | ⟨0, _⟩ => ⟨54, by decide⟩
  | ⟨1, _⟩ => ⟨55, by decide⟩

def phaseKindSelector : Fin 2 → Fin 324
  | ⟨0, _⟩ => ⟨162, by decide⟩
  | ⟨1, _⟩ => ⟨163, by decide⟩

def scheduleSelector : Fin 3 → Fin 324
  | ⟨0, _⟩ => ⟨270, by decide⟩
  | ⟨1, _⟩ => ⟨271, by decide⟩
  | ⟨2, _⟩ => ⟨272, by decide⟩

def lifecycleOf : Fin 3 → Fin 2
  | ⟨0, _⟩ => 0
  | ⟨1, _⟩ => 1
  | ⟨2, _⟩ => 1

def phaseKindOf : Fin 3 → Fin 2
  | ⟨0, _⟩ => 0
  | ⟨1, _⟩ => 1
  | ⟨2, _⟩ => 0

@[simp] theorem commonSelector_values :
    (List.ofFn fun group : Fin 2 => (commonSelector group).val) = [54, 55] := by
  decide

@[simp] theorem phaseKindSelector_values :
    (List.ofFn fun kind : Fin 2 => (phaseKindSelector kind).val) =
      [162, 163] := by
  decide

@[simp] theorem scheduleSelector_values :
    (List.ofFn fun arm : Fin 3 => (scheduleSelector arm).val) =
      [270, 271, 272] := by
  decide

@[simp] theorem lifecycleOf_values :
    (List.ofFn fun arm : Fin 3 => (lifecycleOf arm).val) = [0, 1, 1] := by
  decide

@[simp] theorem phaseKindOf_values :
    (List.ofFn fun arm : Fin 3 => (phaseKindOf arm).val) = [0, 1, 0] := by
  decide

def scheduleWeights (assignment : Fin 324 → F) : Fin 3 → F :=
  fun arm => assignment (scheduleSelector arm)

def storedLifecycleWeights (assignment : Fin 324 → F) : Fin 2 → F :=
  fun group => assignment (commonSelector group)

def storedPhaseKindWeights (assignment : Fin 324 → F) : Fin 2 → F :=
  fun kind => assignment (phaseKindSelector kind)

def beforeCursorValue (assignment : Fin 324 → F) : F :=
  assignment ⟨1, by decide⟩ + 2 * assignment ⟨2, by decide⟩

def afterCursorValue (assignment : Fin 324 → F) : F :=
  assignment ⟨3, by decide⟩ + 2 * assignment ⟨4, by decide⟩

def armValue (arm : Fin 3) : F :=
  ⟨arm.val, Nat.lt_trans arm.isLt (by decide)⟩

def nextArmValue (arm : Fin 3) : F :=
  ⟨arm.val + 1, by
    have modulusLarge : 3 < goldilocksModulus := by decide
    omega⟩

def scheduleTotalGap (assignment : Fin 324 → F) : F :=
  1 - selectorSum (scheduleWeights assignment)

def lifecycleEqualityGap
    (assignment : Fin 324 → F) (group : Fin 2) : F :=
  storedLifecycleWeights assignment group -
    groupWeight lifecycleOf (scheduleWeights assignment) group

def phaseKindEqualityGap
    (assignment : Fin 324 → F) (kind : Fin 2) : F :=
  storedPhaseKindWeights assignment kind -
    groupWeight phaseKindOf (scheduleWeights assignment) kind

def lifecycleActivationGap
    (assignment : Fin 324 → F) (arm : Fin 3) : F :=
  scheduleWeights assignment arm *
      storedLifecycleWeights assignment (lifecycleOf arm) -
    scheduleWeights assignment arm

def phaseKindActivationGap
    (assignment : Fin 324 → F) (arm : Fin 3) : F :=
  scheduleWeights assignment arm *
      storedPhaseKindWeights assignment (phaseKindOf arm) -
    scheduleWeights assignment arm

def beforeCursorGap
    (assignment : Fin 324 → F) (arm : Fin 3) : F :=
  beforeCursorValue assignment - armValue arm

def afterCursorGap
    (assignment : Fin 324 → F) (arm : Fin 3) : F :=
  afterCursorValue assignment - nextArmValue arm

inductive CursorRow where
  | before
  | after

def cursorResidual
    (assignment : Fin 324 → F) (arm : Fin 3) : CursorRow → F
  | .before => beforeCursorGap assignment arm
  | .after => afterCursorGap assignment arm

def scheduleRows
    (assignment : Fin 324 → F) (arm : Fin 3) : ResidualFamily where
  Row := CursorRow
  residual := cursorResidual assignment arm

/-- Exact matrix-image point of the generated schedule-total row. -/
def scheduleTotalPoint (assignment : Fin 324 → F) : Fin 13 → F :=
  productPoint 1 0 0 (scheduleTotalGap assignment)

/-- Exact matrix-image point of one lifecycle-selector equality row. -/
def lifecycleEqualityPoint
    (assignment : Fin 324 → F) (group : Fin 2) : Fin 13 → F :=
  productPoint 1 0 0 (lifecycleEqualityGap assignment group)

/-- Exact matrix-image point of one phase-kind-selector equality row. -/
def phaseKindEqualityPoint
    (assignment : Fin 324 → F) (kind : Fin 2) : Fin 13 → F :=
  productPoint 1 0 0 (phaseKindEqualityGap assignment kind)

/-- Exact matrix-image point of one lifecycle activation row. -/
def lifecycleActivationPoint
    (assignment : Fin 324 → F) (arm : Fin 3) : Fin 13 → F :=
  productPoint 1 (scheduleWeights assignment arm)
    (storedLifecycleWeights assignment (lifecycleOf arm))
    (scheduleWeights assignment arm)

/-- Exact matrix-image point of one phase-kind activation row. -/
def phaseKindActivationPoint
    (assignment : Fin 324 → F) (arm : Fin 3) : Fin 13 → F :=
  productPoint 1 (scheduleWeights assignment arm)
    (storedPhaseKindWeights assignment (phaseKindOf arm))
    (scheduleWeights assignment arm)

/-- Exact matrix-image point of one before-cursor binding row. -/
def beforeCursorPoint
    (assignment : Fin 324 → F) (arm : Fin 3) : Fin 13 → F :=
  productPoint 1 (scheduleWeights assignment arm)
    (beforeCursorGap assignment arm) 0

/-- Exact matrix-image point of one after-cursor binding row. -/
def afterCursorPoint
    (assignment : Fin 324 → F) (arm : Fin 3) : Fin 13 → F :=
  productPoint 1 (scheduleWeights assignment arm)
    (afterCursorGap assignment arm) 0

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

theorem scheduleTotalPoint_zero_iff
    (assignment : Fin 324 → F) :
    evaluate (scheduleTotalPoint assignment) = 0 ↔
      SelectorTotal (scheduleWeights assignment) := by
  rw [scheduleTotalPoint, evaluate_linearPoint, neg_eq_zero_iff]
  unfold scheduleTotalGap SelectorTotal
  constructor
  · intro gapZero
    exact (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp gapZero).symm
  · intro total
    exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr total.symm

theorem lifecycleEqualityPoint_zero_iff
    (assignment : Fin 324 → F) (group : Fin 2) :
    evaluate (lifecycleEqualityPoint assignment group) = 0 ↔
      storedLifecycleWeights assignment group =
        groupWeight lifecycleOf (scheduleWeights assignment) group := by
  rw [lifecycleEqualityPoint, evaluate_linearPoint, neg_eq_zero_iff,
    lifecycleEqualityGap, Lean.Grind.AddCommGroup.sub_eq_zero_iff]

theorem phaseKindEqualityPoint_zero_iff
    (assignment : Fin 324 → F) (kind : Fin 2) :
    evaluate (phaseKindEqualityPoint assignment kind) = 0 ↔
      storedPhaseKindWeights assignment kind =
        groupWeight phaseKindOf (scheduleWeights assignment) kind := by
  rw [phaseKindEqualityPoint, evaluate_linearPoint, neg_eq_zero_iff,
    phaseKindEqualityGap, Lean.Grind.AddCommGroup.sub_eq_zero_iff]

theorem evaluate_lifecycleActivationPoint
    (assignment : Fin 324 → F) (arm : Fin 3) :
    evaluate (lifecycleActivationPoint assignment arm) =
      lifecycleActivationGap assignment arm := by
  unfold lifecycleActivationPoint
  rw [evaluate_productPoint]
  simp [productResidual, lifecycleActivationGap, productPoint,
    sparsePoint, Role.index, Fin.one_mul, Fin.sub_eq_add_neg]

theorem evaluate_phaseKindActivationPoint
    (assignment : Fin 324 → F) (arm : Fin 3) :
    evaluate (phaseKindActivationPoint assignment arm) =
      phaseKindActivationGap assignment arm := by
  unfold phaseKindActivationPoint
  rw [evaluate_productPoint]
  simp [productResidual, phaseKindActivationGap, productPoint,
    sparsePoint, Role.index, Fin.one_mul, Fin.sub_eq_add_neg]

theorem lifecycleActivationPoint_zero_iff
    (assignment : Fin 324 → F) (arm : Fin 3) :
    evaluate (lifecycleActivationPoint assignment arm) = 0 ↔
      scheduleWeights assignment arm *
          storedLifecycleWeights assignment (lifecycleOf arm) =
        scheduleWeights assignment arm := by
  rw [evaluate_lifecycleActivationPoint, lifecycleActivationGap,
    Lean.Grind.AddCommGroup.sub_eq_zero_iff]

theorem phaseKindActivationPoint_zero_iff
    (assignment : Fin 324 → F) (arm : Fin 3) :
    evaluate (phaseKindActivationPoint assignment arm) = 0 ↔
      scheduleWeights assignment arm *
          storedPhaseKindWeights assignment (phaseKindOf arm) =
        scheduleWeights assignment arm := by
  rw [evaluate_phaseKindActivationPoint, phaseKindActivationGap,
    Lean.Grind.AddCommGroup.sub_eq_zero_iff]

theorem evaluate_beforeCursorPoint
    (assignment : Fin 324 → F) (arm : Fin 3) :
    evaluate (beforeCursorPoint assignment arm) =
      scheduleWeights assignment arm * beforeCursorGap assignment arm := by
  unfold beforeCursorPoint
  rw [evaluate_productPoint]
  simp [productResidual, productPoint, sparsePoint,
    Role.index, Fin.one_mul, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

theorem evaluate_afterCursorPoint
    (assignment : Fin 324 → F) (arm : Fin 3) :
    evaluate (afterCursorPoint assignment arm) =
      scheduleWeights assignment arm * afterCursorGap assignment arm := by
  unfold afterCursorPoint
  rw [evaluate_productPoint]
  simp [productResidual, productPoint, sparsePoint,
    Role.index, Fin.one_mul, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

theorem cursorPoints_zero_iff_scheduleGated
    (assignment : Fin 324 → F) (arm : Fin 3) :
    (evaluate (beforeCursorPoint assignment arm) = 0 ∧
      evaluate (afterCursorPoint assignment arm) = 0) ↔
        GatedRowsZero (scheduleWeights assignment arm)
          (scheduleRows assignment arm) := by
  rw [evaluate_beforeCursorPoint, evaluate_afterCursorPoint]
  constructor
  · rintro ⟨beforeZero, afterZero⟩ row
    cases row with
    | before => exact beforeZero
    | after => exact afterZero
  · intro gated
    exact ⟨gated .before, gated .after⟩

/-- Satisfaction of all seventeen generated schedule-link rows. -/
structure LinkRowsHold (assignment : Fin 324 → F) : Prop where
  scheduleTotal : evaluate (scheduleTotalPoint assignment) = 0
  lifecycleEqualities : ∀ group,
    evaluate (lifecycleEqualityPoint assignment group) = 0
  phaseKindEqualities : ∀ kind,
    evaluate (phaseKindEqualityPoint assignment kind) = 0
  lifecycleActivations : ∀ arm,
    evaluate (lifecycleActivationPoint assignment arm) = 0
  phaseKindActivations : ∀ arm,
    evaluate (phaseKindActivationPoint assignment arm) = 0
  cursorBindings : ∀ arm,
    evaluate (beforeCursorPoint assignment arm) = 0 ∧
      evaluate (afterCursorPoint assignment arm) = 0

theorem linkRowsHold_iff
    (assignment : Fin 324 → F) :
    LinkRowsHold assignment ↔
      SelectorTotal (scheduleWeights assignment) ∧
      (∀ group,
        storedLifecycleWeights assignment group =
          groupWeight lifecycleOf (scheduleWeights assignment) group) ∧
      (∀ kind,
        storedPhaseKindWeights assignment kind =
          groupWeight phaseKindOf (scheduleWeights assignment) kind) ∧
      (∀ arm,
        scheduleWeights assignment arm *
            storedLifecycleWeights assignment (lifecycleOf arm) =
          scheduleWeights assignment arm) ∧
      (∀ arm,
        scheduleWeights assignment arm *
            storedPhaseKindWeights assignment (phaseKindOf arm) =
          scheduleWeights assignment arm) ∧
      ∀ arm,
        GatedRowsZero (scheduleWeights assignment arm)
          (scheduleRows assignment arm) := by
  constructor
  · intro rows
    exact ⟨
      (scheduleTotalPoint_zero_iff assignment).mp rows.scheduleTotal,
      fun group => (lifecycleEqualityPoint_zero_iff assignment group).mp
        (rows.lifecycleEqualities group),
      fun kind => (phaseKindEqualityPoint_zero_iff assignment kind).mp
        (rows.phaseKindEqualities kind),
      fun arm => (lifecycleActivationPoint_zero_iff assignment arm).mp
        (rows.lifecycleActivations arm),
      fun arm => (phaseKindActivationPoint_zero_iff assignment arm).mp
        (rows.phaseKindActivations arm),
      fun arm => (cursorPoints_zero_iff_scheduleGated assignment arm).mp
        (rows.cursorBindings arm)⟩
  · rintro ⟨total, lifecycleEqualities, phaseKindEqualities,
      lifecycleLinks, phaseKindLinks, scheduleGated⟩
    exact {
      scheduleTotal := (scheduleTotalPoint_zero_iff assignment).mpr total
      lifecycleEqualities := fun group =>
        (lifecycleEqualityPoint_zero_iff assignment group).mpr
          (lifecycleEqualities group)
      phaseKindEqualities := fun kind =>
        (phaseKindEqualityPoint_zero_iff assignment kind).mpr
          (phaseKindEqualities kind)
      lifecycleActivations := fun arm =>
        (lifecycleActivationPoint_zero_iff assignment arm).mpr
          (lifecycleLinks arm)
      phaseKindActivations := fun arm =>
        (phaseKindActivationPoint_zero_iff assignment arm).mpr
          (phaseKindLinks arm)
      cursorBindings := fun arm =>
        (cursorPoints_zero_iff_scheduleGated assignment arm).mpr
          (scheduleGated arm)
    }

/-- All source-family rows and all exact generated link rows on one assignment. -/
def ComposedRowsHold
    (assignment : Fin 324 → F)
    (common phaseKinds : Fin 2 → ResidualFamily) : Prop :=
  LinkRowsHold assignment ∧
    (∀ group,
      GatedRowsZero (storedLifecycleWeights assignment group) (common group)) ∧
    ∀ kind,
      GatedRowsZero (storedPhaseKindWeights assignment kind) (phaseKinds kind)

theorem composedRowsHold_iff_linkedAccepts
    (assignment : Fin 324 → F)
    (common phaseKinds : Fin 2 → ResidualFamily) :
    ComposedRowsHold assignment common phaseKinds ↔
      ScheduledGrouped.LinkedAccepts lifecycleOf phaseKindOf
        (scheduleWeights assignment) (storedLifecycleWeights assignment)
        (storedPhaseKindWeights assignment) common phaseKinds
        (scheduleRows assignment) := by
  constructor
  · rintro ⟨linkRows, commonGated, phaseKindGated⟩
    rcases (linkRowsHold_iff assignment).mp linkRows with
      ⟨total, lifecycleEqualities, phaseKindEqualities, lifecycleLinks,
        phaseKindLinks, scheduleGated⟩
    exact ⟨total, lifecycleEqualities, phaseKindEqualities, commonGated,
      phaseKindGated, scheduleGated, lifecycleLinks, phaseKindLinks⟩
  · intro accepted
    refine ⟨?_, accepted.commonGated, accepted.phaseKindGated⟩
    exact (linkRowsHold_iff assignment).mpr ⟨accepted.total,
      accepted.lifecycleEqualities, accepted.phaseKindEqualities,
      accepted.lifecycleLinks, accepted.phaseKindLinks,
      accepted.scheduleGated⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGroupedArtifact
