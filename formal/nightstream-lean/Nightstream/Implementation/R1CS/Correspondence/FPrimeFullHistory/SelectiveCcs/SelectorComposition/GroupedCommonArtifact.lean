import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.GroupedPhaseFixture
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.GroupedCommon
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Rows

/-!
Contract: exact generated-row bridge for the grouped common-row composer.

Owns fail-closed fixture decoding, the exact two-group and three-phase selector
placement, and selective-polynomial evaluation of every emitted group-equality
and phase-activation row.

The Rust owner exhaustively compares all thirteen matrix rows with the same
recipe. Lean proves that zero residuals are exactly the two link families used
by `GroupedCommon.LinkedAccepts`.

Does not own either component relation, production phase counts, phase
semantics, or the final recursive and terminal F-prime relations.

Emits constraints: two group-equality rows and three phase-activation rows in
the generated fixture.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommonArtifact

open Nightstream.Implementation.R1CS.FPrimeFullHistoryGroupedPhaseFixture.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics
open Nightstream.SuperNeo.Concrete

abbrev rawArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.GroupedPhaseFixture.rawArtifact

theorem rawArtifact_valid : Valid rawArtifact := by
  decide

def artifact : Decoded := ⟨rawArtifact, rawArtifact_valid⟩

theorem decode_rawArtifact : decode rawArtifact = some artifact := by
  simp [decode, artifact, rawArtifact_valid]

@[simp] theorem rows_exact : rawArtifact.rows = 340 := by
  rfl

@[simp] theorem columns_exact : rawArtifact.columns = 270 := by
  rfl

@[simp] theorem publicColumns_exact : rawArtifact.publicColumns = 54 := by
  rfl

@[simp] theorem commonRowEnd_exact : rawArtifact.commonRowEnd = 166 := by
  rfl

@[simp] theorem phaseRowEnd_exact : rawArtifact.phaseRowEnd = 335 := by
  rfl

@[simp] theorem groupEqualityRowEnd_exact :
    rawArtifact.groupEqualityRowEnd = 337 := by
  rfl

@[simp] theorem phaseActivationRowEnd_exact :
    rawArtifact.phaseActivationRowEnd = 340 := by
  rfl

def commonSelector : Fin 2 -> Fin 270
  | ⟨0, _⟩ => ⟨54, by decide⟩
  | ⟨1, _⟩ => ⟨55, by decide⟩

def phaseSelector : Fin 3 -> Fin 270
  | ⟨0, _⟩ => ⟨162, by decide⟩
  | ⟨1, _⟩ => ⟨163, by decide⟩
  | ⟨2, _⟩ => ⟨164, by decide⟩

def groupOf : Fin 3 -> Fin 2
  | ⟨0, _⟩ => 0
  | ⟨1, _⟩ => 1
  | ⟨2, _⟩ => 1

@[simp] theorem commonSelector_values :
    (List.ofFn fun group : Fin 2 => (commonSelector group).val) = [54, 55] := by
  decide

@[simp] theorem phaseSelector_values :
    (List.ofFn fun phase : Fin 3 => (phaseSelector phase).val) =
      [162, 163, 164] := by
  decide

@[simp] theorem groupOf_values :
    (List.ofFn fun phase : Fin 3 => (groupOf phase).val) = [0, 1, 1] := by
  decide

def phaseWeights (assignment : Fin 270 -> F) : Fin 3 -> F :=
  fun phase => assignment (phaseSelector phase)

def storedGroupWeights (assignment : Fin 270 -> F) : Fin 2 -> F :=
  fun group => assignment (commonSelector group)

def groupGap (assignment : Fin 270 -> F) (group : Fin 2) : F :=
  storedGroupWeights assignment group -
    groupWeight groupOf (phaseWeights assignment) group

def activationGap (assignment : Fin 270 -> F) (phase : Fin 3) : F :=
  phaseWeights assignment phase *
      storedGroupWeights assignment (groupOf phase) -
    phaseWeights assignment phase

/-- Exact matrix-image point of one generated group-equality row. -/
def groupEqualityPoint (assignment : Fin 270 -> F) (group : Fin 2) :
    Fin 13 -> F :=
  productPoint 1 0 0 (groupGap assignment group)

/-- Exact matrix-image point of one generated phase-activation row. -/
def phaseActivationPoint (assignment : Fin 270 -> F) (phase : Fin 3) :
    Fin 13 -> F :=
  productPoint 1 (phaseWeights assignment phase)
    (storedGroupWeights assignment (groupOf phase))
    (phaseWeights assignment phase)

theorem evaluate_groupEqualityPoint
    (assignment : Fin 270 -> F) (group : Fin 2) :
    evaluate (groupEqualityPoint assignment group) =
      -(groupGap assignment group) := by
  unfold groupEqualityPoint
  rw [evaluate_productPoint]
  simp [productResidual, productPoint, sparsePoint,
    Role.index, Fin.mul_zero, Fin.one_mul, Fin.zero_add]

theorem evaluate_phaseActivationPoint
    (assignment : Fin 270 -> F) (phase : Fin 3) :
    evaluate (phaseActivationPoint assignment phase) =
      activationGap assignment phase := by
  unfold phaseActivationPoint
  rw [evaluate_productPoint]
  simp [productResidual, activationGap, productPoint,
    sparsePoint, Role.index, Fin.one_mul, Fin.sub_eq_add_neg]

theorem groupEqualityPoint_zero_iff
    (assignment : Fin 270 -> F) (group : Fin 2) :
    evaluate (groupEqualityPoint assignment group) = 0 ↔
      storedGroupWeights assignment group =
        groupWeight groupOf (phaseWeights assignment) group := by
  rw [evaluate_groupEqualityPoint]
  constructor
  · intro negZero
    have gapZero : groupGap assignment group = 0 := by
      simpa only [Lean.Grind.AddCommGroup.neg_neg,
        Lean.Grind.AddCommGroup.neg_zero] using congrArg Neg.neg negZero
    exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp gapZero
  · intro equal
    rw [groupGap, Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr equal]
    rfl

theorem phaseActivationPoint_zero_iff
    (assignment : Fin 270 -> F) (phase : Fin 3) :
    evaluate (phaseActivationPoint assignment phase) = 0 ↔
      phaseWeights assignment phase *
          storedGroupWeights assignment (groupOf phase) =
        phaseWeights assignment phase := by
  rw [evaluate_phaseActivationPoint, activationGap,
    Lean.Grind.AddCommGroup.sub_eq_zero_iff]

/-- Satisfaction of all five generated link rows. -/
def LinkRowsHold (assignment : Fin 270 -> F) : Prop :=
  (∀ group, evaluate (groupEqualityPoint assignment group) = 0) /\
    ∀ phase, evaluate (phaseActivationPoint assignment phase) = 0

theorem linkRowsHold_iff
    (assignment : Fin 270 -> F) :
    LinkRowsHold assignment ↔
      (∀ group,
        storedGroupWeights assignment group =
          groupWeight groupOf (phaseWeights assignment) group) /\
      ∀ phase,
        phaseWeights assignment phase *
            storedGroupWeights assignment (groupOf phase) =
          phaseWeights assignment phase := by
  constructor
  · rintro ⟨groupRows, phaseRows⟩
    exact ⟨
      fun group =>
        (groupEqualityPoint_zero_iff assignment group).mp (groupRows group),
      fun phase =>
        (phaseActivationPoint_zero_iff assignment phase).mp
          (phaseRows phase)⟩
  · rintro ⟨groupEqualities, phaseLinks⟩
    exact ⟨
      fun group =>
        (groupEqualityPoint_zero_iff assignment group).mpr
          (groupEqualities group),
      fun phase =>
        (phaseActivationPoint_zero_iff assignment phase).mpr
          (phaseLinks phase)⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommonArtifact
