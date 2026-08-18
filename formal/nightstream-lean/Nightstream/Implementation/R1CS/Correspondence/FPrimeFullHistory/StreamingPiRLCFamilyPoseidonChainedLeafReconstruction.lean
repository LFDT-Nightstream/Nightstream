import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonChainedLeafCertificate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafReconstruction

/-!
Contract: same-assignment source reconstruction for one generated chained
production PiRLC Poseidon2 leaf.

Assurance tier: artifact-checked leaf reconstruction.

Owns decoding of the four exact prior-output images, linear port-image
algebra, structural matching of all 86 chained rows, and derivation of the
shared S-box equations from one final assignment.

Does not own absolute Rust placement, replay-batch coverage, selector
authority, recursive orchestration, or cryptographic security.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonChainedLeaf
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafCertificate
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

private abbrev sharedSteps :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedSteps

private abbrev LeafStepRealized
    (source : SourceAssignment) (final : FinalAssignment)
    (step : Wire.Step) (row : Wire.Row) : Prop :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.StepRealized
    source final step row

private abbrev LeafStepSboxHolds
    (source : SourceAssignment) (step : Wire.Step) : Prop :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.StepSboxHolds
    source step

structure SourceImage where
  lane : Fin 4
  port : Wire.Port

def decodeSourceImage (raw : RawSourceImage) : Option SourceImage := do
  if bounded : raw.lane < 4 then
    let port ← Wire.decodePort raw.port
    pure { lane := ⟨raw.lane, bounded⟩, port }
  else
    none

def decodedImages : List SourceImage :=
  (rawImages.mapM decodeSourceImage).getD []

theorem raw_images_decode :
    rawImages.mapM decodeSourceImage = some decodedImages := by
  rfl

theorem decoded_images_length : decodedImages.length = 4 := by
  rfl

theorem decoded_image_lanes :
    decodedImages.map (fun image => image.lane.val) = [0, 1, 2, 3] := by
  rfl

def emptyImage : SourceImage where
  lane := ⟨0, by decide⟩
  port := emptyPort

def priorImagePort (lane : Fin 4) : Wire.Port :=
  (decodedImages.getD lane.val emptyImage).port

def unitSlotPort (slot : Wire.Slot) : Wire.Port where
  explicit := []
  geometric := [{ slot, initial := 1, ratio := 3 }]

def sourceColumnPort : Wire.SourceColumn → Wire.Port
  | .externalA lane => unitSlotPort (.externalA lane)
  | .externalB lane => priorImagePort lane
  | column@(.local _) =>
      match sourceSlot sharedSteps column with
      | some slot => unitSlotPort slot
      | none => emptyPort

def addPort (left right : Wire.Port) : Wire.Port where
  explicit := left.explicit ++ right.explicit
  geometric := left.geometric ++ right.geometric

def scaleRun (coefficient : F) (run : Wire.GeometricRun) : Wire.GeometricRun where
  slot := run.slot
  initial := coefficient * run.initial
  ratio := run.ratio

def scalePort (coefficient : F) (port : Wire.Port) : Wire.Port where
  explicit := port.explicit.map fun term =>
    { term with coefficient := coefficient * term.coefficient }
  geometric := port.geometric.map (scaleRun coefficient)

def constantPort (constant : F) : Wire.Port where
  explicit := expectedExplicit constant
  geometric := []

def termsPort : List Wire.SourceTerm → Wire.Port
  | [] => emptyPort
  | term :: tail =>
      addPort (scalePort term.coefficient (sourceColumnPort term.column))
        (termsPort tail)

def sourceLinearPort (value : Wire.SourceLinearCombination) : Wire.Port :=
  addPort (constantPort value.constant) (termsPort value.terms)

def reconstructedSource (final : FinalAssignment) : SourceAssignment where
  externalA := fun lane => portAction (sourceColumnPort (.externalA lane)) final
  externalB := fun lane => portAction (sourceColumnPort (.externalB lane)) final
  localValue := fun offset => portAction (sourceColumnPort (.local offset)) final

private theorem sum_append (left right : List F) :
    sum (left ++ right) = sum left + sum right := by
  induction left with
  | nil => simp [sum]
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, sum]
      rw [inductionHypothesis, Lean.Grind.Fin.add_assoc]

private theorem sum_map_mul_left (coefficient : F) :
    ∀ values : List F,
      sum (values.map fun value => coefficient * value) =
        coefficient * sum values
  | [] => by exact (Fin.mul_zero coefficient).symm
  | head :: tail => by
      simp only [List.map_cons, sum, sum_map_mul_left coefficient tail]
      exact (Lean.Grind.Fin.left_distrib coefficient head (sum tail)).symm

private theorem geometricCoefficient_scale
    (coefficient initial ratio : F) (index : Nat) :
    geometricCoefficient (coefficient * initial) ratio index =
      coefficient * geometricCoefficient initial ratio index := by
  induction index with
  | zero => rfl
  | succ index inductionHypothesis =>
      simp only [geometricCoefficient]
      rw [inductionHypothesis, Fin.mul_assoc]

private theorem geometricAction_scale
    (coefficient : F) (run : Wire.GeometricRun)
    (final : FinalAssignment) :
    geometricAction (scaleRun coefficient run) final =
      coefficient * geometricAction run final := by
  unfold geometricAction scaleRun
  simp only [geometricCoefficient_scale, Fin.mul_assoc]
  simpa only [List.map_ofFn, Function.comp_apply] using
    sum_map_mul_left coefficient
      (List.ofFn fun index : Fin 41 =>
        geometricCoefficient run.initial run.ratio index.val *
          final.digit run.slot index)

private theorem add_interchange (a b c d : F) :
    (a + b) + (c + d) = (a + c) + (b + d) := by
  rw [Lean.Grind.Fin.add_assoc a b (c + d),
    ← Lean.Grind.Fin.add_assoc b c d,
    Lean.Grind.Fin.add_comm b c,
    Lean.Grind.Fin.add_assoc c b d,
    ← Lean.Grind.Fin.add_assoc a c (b + d)]

private theorem portAction_add
    (left right : Wire.Port) (final : FinalAssignment) :
    portAction (addPort left right) final =
      portAction left final + portAction right final := by
  unfold portAction addPort
  simp only [List.map_append, sum_append]
  exact add_interchange _ _ _ _

private theorem portAction_scale
    (coefficient : F) (port : Wire.Port) (final : FinalAssignment) :
    portAction (scalePort coefficient port) final =
      coefficient * portAction port final := by
  have explicitScale :
      ∀ terms : List Wire.ExplicitTerm,
        sum ((terms.map fun term =>
            { term with coefficient := coefficient * term.coefficient }).map
          fun term => term.coefficient * final.explicit term.column) =
        coefficient * sum (terms.map fun term =>
          term.coefficient * final.explicit term.column) := by
    intro terms
    induction terms with
    | nil => exact (Fin.mul_zero coefficient).symm
    | cons term tail inductionHypothesis =>
        simp only [List.map_cons, sum]
        rw [Fin.mul_assoc, inductionHypothesis]
        exact (Lean.Grind.Fin.left_distrib _ _ _).symm
  have geometricScale :
      ∀ runs : List Wire.GeometricRun,
        sum ((runs.map (scaleRun coefficient)).map fun run =>
          geometricAction run final) =
        coefficient * sum (runs.map fun run =>
          geometricAction run final) := by
    intro runs
    induction runs with
    | nil => exact (Fin.mul_zero coefficient).symm
    | cons run tail inductionHypothesis =>
        simp only [List.map_cons, sum]
        rw [geometricAction_scale, inductionHypothesis]
        exact (Lean.Grind.Fin.left_distrib _ _ _).symm
  unfold portAction scalePort
  rw [explicitScale, geometricScale]
  exact (Lean.Grind.Fin.left_distrib _ _ _).symm

private theorem portAction_constant
    (constant : F) (final : FinalAssignment)
    (one : final.explicit .one = 1) :
    portAction (constantPort constant) final = constant := by
  unfold constantPort portAction expectedExplicit
  split <;> rename_i constantZero
  · simp [constantZero, sum]
  · simp [one, sum, Fin.mul_one]

private theorem sourceColumnPort_action
    (column : Wire.SourceColumn) (final : FinalAssignment) :
    portAction (sourceColumnPort column) final =
      sourceValue (reconstructedSource final) column := by
  cases column <;> rfl

private theorem termsPort_action
    (terms : List Wire.SourceTerm) (final : FinalAssignment) :
    portAction (termsPort terms) final =
      sum (terms.map fun term =>
        term.coefficient * sourceValue (reconstructedSource final) term.column) := by
  induction terms with
  | nil => rfl
  | cons term tail inductionHypothesis =>
      simp only [termsPort, List.map_cons, sum]
      rw [portAction_add, portAction_scale, sourceColumnPort_action,
        inductionHypothesis]

theorem sourceLinearPort_action
    (value : Wire.SourceLinearCombination) (final : FinalAssignment)
    (one : final.explicit .one = 1) :
    portAction (sourceLinearPort value) final =
      sourceAction value (reconstructedSource final) := by
  unfold sourceLinearPort sourceAction
  rw [portAction_add, portAction_constant value.constant final one,
    termsPort_action]

structure StepLayoutMatches
    (step : Wire.Step) (row : Wire.Row) : Prop where
  input : row.port Role.sboxInput.index = sourceLinearPort step.input
  output : row.port Role.c.index = sourceLinearPort step.output

instance (step : Wire.Step) (row : Wire.Row) :
    Decidable (StepLayoutMatches step row) :=
  if input : row.port Role.sboxInput.index = sourceLinearPort step.input then
    if output : row.port Role.c.index = sourceLinearPort step.output then
      isTrue ⟨input, output⟩
    else
      isFalse (fun matched => output matched.output)
  else
    isFalse (fun matched => input matched.input)

def stepLayoutCheck (step : Wire.Step) (row : Wire.Row) : Bool :=
  decide (StepLayoutMatches step row)

def allStepLayoutsCheck : List Wire.Step → List Wire.Row → Bool
  | [], [] => true
  | step :: stepTail, row :: rowTail =>
      stepLayoutCheck step row && allStepLayoutsCheck stepTail rowTail
  | _, _ => false

private theorem stepLayoutCheck_sound
    (step : Wire.Step) (row : Wire.Row)
    (checked : stepLayoutCheck step row = true) :
    StepLayoutMatches step row := by
  unfold stepLayoutCheck at checked
  exact of_decide_eq_true checked

private theorem allStepLayoutsCheck_sound :
    ∀ (steps : List Wire.Step) (rows : List Wire.Row),
      allStepLayoutsCheck steps rows = true →
      List.Forall₂ StepLayoutMatches steps rows
  | [], [], _ => .nil
  | [], _ :: _, checked => by simp [allStepLayoutsCheck] at checked
  | _ :: _, [], checked => by simp [allStepLayoutsCheck] at checked
  | step :: stepTail, row :: rowTail, checked => by
      simp only [allStepLayoutsCheck, Bool.and_eq_true] at checked
      exact .cons (stepLayoutCheck_sound step row checked.1)
        (allStepLayoutsCheck_sound stepTail rowTail checked.2)

theorem decoded_head_layouts_checked :
    allStepLayoutsCheck (sharedSteps.take 43) decodedRowHead = true := by
  rfl

theorem decoded_tail_layouts_checked :
    allStepLayoutsCheck (sharedSteps.drop 43) decodedRowTail = true := by
  rfl

private theorem forall₂_append
    {α β : Type} {relation : α → β → Prop}
    {leftSteps rightSteps : List α} {leftRows rightRows : List β}
    (left : List.Forall₂ relation leftSteps leftRows)
    (right : List.Forall₂ relation rightSteps rightRows) :
    List.Forall₂ relation (leftSteps ++ rightSteps) (leftRows ++ rightRows) := by
  induction left with
  | nil => exact right
  | cons head tail inductionHypothesis =>
      exact .cons head inductionHypothesis

theorem decoded_step_layouts :
    List.Forall₂ StepLayoutMatches sharedSteps decodedRows := by
  rw [show sharedSteps = sharedSteps.take 43 ++ sharedSteps.drop 43 by simp]
  exact forall₂_append
    (allStepLayoutsCheck_sound _ _ decoded_head_layouts_checked)
    (allStepLayoutsCheck_sound _ _ decoded_tail_layouts_checked)

private theorem realized_of_layouts
    (final : FinalAssignment) (one : final.explicit .one = 1) :
    ∀ {steps : List Wire.Step} {rows : List Wire.Row},
      List.Forall₂ StepLayoutMatches steps rows →
      List.Forall₂ (LeafStepRealized (reconstructedSource final) final)
        steps rows := by
  intro steps rows layouts
  induction layouts with
  | nil => exact .nil
  | cons head tail inductionHypothesis =>
      exact .cons
        { input := by
            rw [head.input]
            exact sourceLinearPort_action _ _ one
          output := by
            rw [head.output]
            exact sourceLinearPort_action _ _ one }
        inductionHypothesis

theorem decoded_steps_realized
    (final : FinalAssignment) (one : final.explicit .one = 1) :
    List.Forall₂ (LeafStepRealized (reconstructedSource final) final)
      sharedSteps decodedRows :=
  realized_of_layouts final one decoded_step_layouts

/-- All active chained rows imply all 86 shared source S-box equations on the
source assignment reconstructed from those same final coordinates. -/
theorem decoded_rows_imply_reconstructed_step_sboxes
    (final : FinalAssignment)
    (one : final.explicit .one = 1)
    (selectorOne : final.explicit .selector = 1)
    (holds : ∀ row ∈ decodedRows, residual row final = 0) :
    ∀ step ∈ sharedSteps,
      LeafStepSboxHolds (reconstructedSource final) step :=
  decoded_rows_imply_shared_step_sboxes
    (reconstructedSource final) final selectorOne
    (decoded_steps_realized final one) holds

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction
