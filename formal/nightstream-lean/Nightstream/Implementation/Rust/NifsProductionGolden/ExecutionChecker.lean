import Nightstream.Implementation.Rust.NifsProductionGolden.PiCcsChecker
import Nightstream.Implementation.Rust.NifsProductionGolden.PiDecChecker
import Nightstream.Implementation.Rust.NifsProductionGolden.PiRlcChecker

/-!
One fail-closed checker for the recorded production
`Pi_CCS -> Pi_RLC -> Pi_DEC` execution.

The cross-phase check binds the complete active `Pi_CCS` output to the sole
`Pi_RLC` input, binds its point to the SumCheck point, and replays the exact
fold-digest and `rho` handoff. The three phase checkers own the paper equations.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Rust.NifsProductionGolden.ExecutionChecker

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Rust.NifsProductionGolden
open Nightstream.Implementation.Rust.PiCcsExecution

def activeOutputMatches (receipt : ProductionReceipt)
    (input : RawClaim) : Bool :=
  (List.range 4).all fun matrix =>
    (List.range 54).all fun coefficient =>
      decide (receipt.piCcsProof.fullOutput.getD
          (matrix * 54 + coefficient) default =
        input.evaluations.getD (matrix * 64 + coefficient) default)

def pointMatches (decoded : PiCcsChecker.DecodedReceipt)
    (input : RawClaim) : Bool :=
  decide (input.point.map RawK.decode = decoded.roundPoint.coordinates)

def crossPhaseCheck (receipt : ProductionReceipt) : Bool :=
  match PiCcsChecker.decodeReceiptCertified receipt, receipt.piRlcInputs with
  | some decoded, [input] =>
      match PiRlcReplay.handoff? receipt decoded.finalTranscript with
      | none => false
      | some _ => activeOutputMatches receipt input && pointMatches decoded input
  | _, _ => false

def CrossPhaseBound (receipt : ProductionReceipt) : Prop :=
  exists decoded input handoff,
    PiCcsChecker.decodeReceiptCertified receipt = some decoded /\
      receipt.piRlcInputs = [input] /\
      PiRlcReplay.handoff? receipt decoded.finalTranscript = some handoff /\
      activeOutputMatches receipt input = true /\
      pointMatches decoded input = true

theorem crossPhaseCheck_sound (receipt : ProductionReceipt)
    (checked : crossPhaseCheck receipt = true) :
    CrossPhaseBound receipt := by
  unfold crossPhaseCheck at checked
  cases decodedEq : PiCcsChecker.decodeReceiptCertified receipt with
  | none => simp [decodedEq] at checked
  | some decoded =>
    cases inputsEq : receipt.piRlcInputs with
    | nil => simp [decodedEq, inputsEq] at checked
    | cons input inputs =>
      cases inputs with
      | cons second rest => simp [decodedEq, inputsEq] at checked
      | nil =>
        cases handoffEq : PiRlcReplay.handoff? receipt
            decoded.finalTranscript with
        | none => simp [decodedEq, inputsEq, handoffEq] at checked
        | some handoff =>
          have components : activeOutputMatches receipt input = true /\
              pointMatches decoded input = true := by
            simpa only [decodedEq, inputsEq, handoffEq, Bool.and_eq_true]
              using checked
          exact ⟨decoded, input, handoff, decodedEq, inputsEq, handoffEq,
            components.1, components.2⟩

def checkReceipt (receipt : ProductionReceipt) : Bool :=
  NifsProductionGolden.receiptShapeCheck receipt &&
    (crossPhaseCheck receipt &&
      (PiCcsChecker.checkReceipt receipt &&
        (PiRlcChecker.checkReceipt receipt &&
          PiDecChecker.checkReceipt receipt)))

namespace PaperExecution

def Accepts (receipt : ProductionReceipt) : Prop :=
  CrossPhaseBound receipt /\
    PiCcsChecker.PaperPiCCS.Accepts receipt /\
    PiRlcChecker.PaperPiRLC.Accepts receipt /\
    PiDecChecker.PaperPiDEC.Accepts receipt

end PaperExecution

theorem checkReceipt_sound (receipt : ProductionReceipt) :
    checkReceipt receipt = true -> PaperExecution.Accepts receipt := by
  intro checked
  have components :
      NifsProductionGolden.receiptShapeCheck receipt = true /\
      crossPhaseCheck receipt = true /\
      PiCcsChecker.checkReceipt receipt = true /\
      PiRlcChecker.checkReceipt receipt = true /\
      PiDecChecker.checkReceipt receipt = true := by
    simpa only [checkReceipt, Bool.and_eq_true] using checked
  exact ⟨crossPhaseCheck_sound receipt components.2.1,
    PiCcsChecker.checkReceipt_sound receipt components.2.2.1,
    PiRlcChecker.checkReceipt_sound receipt components.2.2.2.1,
    PiDecChecker.checkReceipt_sound receipt components.2.2.2.2⟩

end Nightstream.Implementation.Rust.NifsProductionGolden.ExecutionChecker
