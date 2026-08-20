import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonChainedLeafReconstruction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeafCertificate

/-!
Contract: same-assignment source reconstruction for the first generated
recursive-terminal XOut Poseidon2 leaf.

Assurance tier: artifact-checked leaf reconstruction.

Owns: decoding of five exact source images, constant-term aggregation,
structural matching of all 86 generated rows, and derivation of all source
S-box equations from one final leaf assignment.

Does not own: absolute final-column placement, complete hash replay,
lifecycle authority, or cryptographic security.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeafReconstruction

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeaf
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeafCertificate

private abbrev LeafStepRealized
    (source : SourceAssignment) (final : FinalAssignment)
    (step : Wire.Step) (row : Wire.Row) : Prop :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.StepRealized
    source final step row

private abbrev LeafStepSboxHolds
    (source : SourceAssignment) (step : Wire.Step) : Prop :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.StepSboxHolds
    source step

private abbrev addPort :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.addPort
private abbrev scalePort :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.scalePort
private abbrev constantPort :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.constantPort
private abbrev unitSlotPort :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.unitSlotPort

structure SourceImage where
  lane : Fin 5
  port : Wire.Port

def decodeSourceImage (raw : RawSourceImage) : Option SourceImage := do
  if bounded : raw.lane < 5 then
    let port ← Wire.decodePort raw.port
    pure { lane := ⟨raw.lane, bounded⟩, port }
  else
    none

def decodedImages : List SourceImage :=
  (rawImages.mapM decodeSourceImage).getD []

theorem raw_images_decode :
    rawImages.mapM decodeSourceImage = some decodedImages := by
  rfl

theorem decoded_images_length : decodedImages.length = 5 := by
  rfl

theorem decoded_image_lanes :
    decodedImages.map (fun image => image.lane.val) = [0, 1, 2, 3, 4] := by
  rfl

def emptyImage : SourceImage where
  lane := ⟨0, by decide⟩
  port := emptyPort

def sourceImagePort (index : Fin 5) : Wire.Port :=
  (decodedImages.getD index.val emptyImage).port

def sourceColumnPort : Wire.SourceColumn → Wire.Port
  | .externalA lane => sourceImagePort ⟨lane.val + 1, by omega⟩
  | .externalB lane =>
      if lane.val = 0 then sourceImagePort ⟨0, by decide⟩ else emptyPort
  | column@(.local _) =>
      match
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction.sourceSlot
          decodedSteps column
      with
      | some slot => unitSlotPort slot
      | none => emptyPort

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
      rw [Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.portAction_add,
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.portAction_scale,
        sourceColumnPort_action, inductionHypothesis]

theorem sourceLinearPort_action
    (value : Wire.SourceLinearCombination) (final : FinalAssignment)
    (one : final.explicit .one = 1) :
    portAction (sourceLinearPort value) final =
      sourceAction value (reconstructedSource final) := by
  unfold sourceLinearPort sourceAction
  rw [Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.portAction_add,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.portAction_constant,
    termsPort_action]
  exact one

def ExplicitOne (port : Wire.Port) : Prop :=
  ∀ term ∈ port.explicit, term.column = .one

instance (port : Wire.Port) : Decidable (ExplicitOne port) := by
  unfold ExplicitOne
  infer_instance

def explicitCoefficient (port : Wire.Port) : F :=
  sum (port.explicit.map fun term => term.coefficient)

structure PortEquivalent (left right : Wire.Port) : Prop where
  leftOne : ExplicitOne left
  rightOne : ExplicitOne right
  explicit : explicitCoefficient left = explicitCoefficient right
  geometric : left.geometric = right.geometric

instance (left right : Wire.Port) : Decidable (PortEquivalent left right) := by
  by_cases leftOne : ExplicitOne left
  · by_cases rightOne : ExplicitOne right
    · by_cases explicit : explicitCoefficient left = explicitCoefficient right
      · by_cases geometric : left.geometric = right.geometric
        · exact isTrue ⟨leftOne, rightOne, explicit, geometric⟩
        · exact isFalse (fun equivalent => geometric equivalent.geometric)
      · exact isFalse (fun equivalent => explicit equivalent.explicit)
    · exact isFalse (fun equivalent => rightOne equivalent.rightOne)
  · exact isFalse (fun equivalent => leftOne equivalent.leftOne)

private theorem explicit_terms_action_eq_coefficient
    (terms : List Wire.ExplicitTerm) (final : FinalAssignment)
    (one : final.explicit .one = 1)
    (onlyOne : ∀ term ∈ terms, term.column = .one) :
    sum (terms.map fun term =>
      term.coefficient * final.explicit term.column) =
      sum (terms.map fun term => term.coefficient) := by
  induction terms with
  | nil => rfl
  | cons term tail inductionHypothesis =>
      simp only [List.map_cons, sum]
      rw [onlyOne term List.mem_cons_self, one, Fin.mul_one]
      rw [inductionHypothesis (fun candidate member =>
        onlyOne candidate (List.mem_cons_of_mem term member))]

private theorem explicit_action_eq_coefficient
    (port : Wire.Port) (final : FinalAssignment)
    (one : final.explicit .one = 1)
    (onlyOne : ExplicitOne port) :
    sum (port.explicit.map fun term =>
      term.coefficient * final.explicit term.column) =
      explicitCoefficient port := by
  exact explicit_terms_action_eq_coefficient port.explicit final one onlyOne

theorem portEquivalent_action
    {left right : Wire.Port} (equivalent : PortEquivalent left right)
    (final : FinalAssignment) (one : final.explicit .one = 1) :
    portAction left final = portAction right final := by
  unfold portAction
  rw [explicit_action_eq_coefficient left final one equivalent.leftOne,
    explicit_action_eq_coefficient right final one equivalent.rightOne,
    equivalent.explicit, equivalent.geometric]

structure StepLayoutMatches (step : Wire.Step) (row : Wire.Row) : Prop where
  input : PortEquivalent (row.port Role.sboxInput.index)
    (sourceLinearPort step.input)
  output : PortEquivalent (row.port Role.c.index)
    (sourceLinearPort step.output)

instance (step : Wire.Step) (row : Wire.Row) :
    Decidable (StepLayoutMatches step row) := by
  by_cases input : PortEquivalent (row.port Role.sboxInput.index)
      (sourceLinearPort step.input)
  · by_cases output : PortEquivalent (row.port Role.c.index)
        (sourceLinearPort step.output)
    · exact isTrue ⟨input, output⟩
    · exact isFalse (fun matched => output matched.output)
  · exact isFalse (fun matched => input matched.input)

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
    allStepLayoutsCheck decodedStepHead decodedRowHead = true := by
  rfl

theorem decoded_tail_layouts_checked :
    allStepLayoutsCheck decodedStepTail decodedRowTail = true := by
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
    List.Forall₂ StepLayoutMatches decodedSteps decodedRows :=
  forall₂_append
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
        { input := (portEquivalent_action head.input final one).trans
            (sourceLinearPort_action _ _ one)
          output := (portEquivalent_action head.output final one).trans
            (sourceLinearPort_action _ _ one) }
        inductionHypothesis

theorem decoded_steps_realized
    (final : FinalAssignment) (one : final.explicit .one = 1) :
    List.Forall₂ (LeafStepRealized (reconstructedSource final) final)
      decodedSteps decodedRows :=
  realized_of_layouts final one decoded_step_layouts

/-- All active generated rows imply all 86 typed source S-box equations on
the source assignment reconstructed from the five exact source images and
the local radix-3 slots of that same final assignment. -/
theorem decoded_rows_imply_reconstructed_step_sboxes
    (final : FinalAssignment)
    (one : final.explicit .one = 1)
    (selectorOne : final.explicit .selector = 1)
    (holds : ∀ row ∈ decodedRows, residual row final = 0) :
    ∀ step ∈ decodedSteps,
      LeafStepSboxHolds (reconstructedSource final) step :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.realized_rows_imply_step_sboxes
    (reconstructedSource final) final selectorOne
    (decoded_steps_realized final one) decoded_rows_have_sbox_shape holds

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeafReconstruction
