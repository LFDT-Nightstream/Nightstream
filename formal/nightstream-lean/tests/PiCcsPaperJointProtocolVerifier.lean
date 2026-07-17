import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier

#check Statement
#check ReplayInput
#check TranscriptReplayCollision
#check TranscriptStateCollision
#check OutputAbsorptionCollision
#check Certificate.toFinite_rounds_length
#check derive_coins_eq_transcript
#check derive_outgoingState_eq_absorbOutput
#check check_eq_true_iff_accepted
#check check_complete_of_accepted
#check check_implies_tableTruth_or_badEvent

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.Tests

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial

def emptyShape : Shape where
  cubeVariables := 0
  freshCount := 0
  runningCount := 0
  matrixCount := 0
  coefficientCount := 0

def emptyPolynomial : CCSResidualTable.ConstraintPolynomial Nat 0 where
  degreeBound := 0
  terms := []
  termsBelowDegree := by simp

def emptyInput : VerifierInput Nat emptyShape where
  constraintPolynomial := emptyPolynomial
  priorPoint := { coordinates := [], dimension := rfl }
  claimedCoefficient := fun coordinate => Fin.elim0 coordinate.running

def emptyOutput : OutputMessage Nat emptyShape where
  freshMatrixImage := fun source => Fin.elim0 source
  sourceAssignment := fun source => Fin.elim0 source
  carriedImage := fun coordinate => Fin.elim0 coordinate.running

/-- Deliberately omitting abstract oracle used only to prove that the named
failure predicates fail closed. It is not a protocol instantiation. -/
def omittingOracle : Oracle Nat Nat emptyShape where
  transcript := {
    initialState := fun _ => 0
    absorbRound := fun _ round _ => Fin.elim0 round
    squeeze := fun state _ => (0, state)
  }
  absorbOutput := fun _ _ => 0

def emptyRounds : FiatShamir.Certificate Nat emptyShape where
  rounds := fun round => Fin.elim0 round

def leftReplay : ReplayInput Nat Nat emptyShape where
  statement := { priorState := 0, input := emptyInput }
  rounds := emptyRounds

def rightReplay : ReplayInput Nat Nat emptyShape where
  statement := { priorState := 1, input := emptyInput }
  rounds := emptyRounds

private theorem leftReplay_ne_rightReplay : leftReplay ≠ rightReplay := by
  intro equal
  have priorStateEqual := congrArg
    (fun replay => replay.statement.priorState) equal
  change (0 : Nat) = 1 at priorStateEqual
  omega

/-- A transcript that ignores the complete statement is caught at the
challenge view, even though its initialization function is no longer the
only surface being audited. -/
theorem omittingOracle_has_replayCollision :
    TranscriptReplayCollision omittingOracle leftReplay rightReplay := by
  refine ⟨leftReplay_ne_rightReplay, ?_⟩
  constructor
  · rfl
  · constructor <;> rfl

/-- The same omitting transcript is independently caught at the state handed
to output absorption. -/
theorem omittingOracle_has_stateCollision :
    TranscriptStateCollision omittingOracle leftReplay rightReplay := by
  refine ⟨leftReplay_ne_rightReplay, ?_⟩
  rfl

def oneVariableShape : Shape where
  cubeVariables := 1
  freshCount := 0
  runningCount := 0
  matrixCount := 0
  coefficientCount := 0

def oneVariablePolynomial : CCSResidualTable.ConstraintPolynomial Nat 0 where
  degreeBound := 0
  terms := []
  termsBelowDegree := by simp

def oneVariableInput : VerifierInput Nat oneVariableShape where
  constraintPolynomial := oneVariablePolynomial
  priorPoint := { coordinates := [0], dimension := rfl }
  claimedCoefficient := fun coordinate => Fin.elim0 coordinate.running

def zeroRoundMessage : SumCheck.Finite.Message Nat where
  coefficients := [0]

def oneRoundMessage : SumCheck.Finite.Message Nat where
  coefficients := [1]

def zeroRoundCertificate : FiatShamir.Certificate Nat oneVariableShape where
  rounds := fun _ => zeroRoundMessage

def oneRoundCertificate : FiatShamir.Certificate Nat oneVariableShape where
  rounds := fun _ => oneRoundMessage

/-- This oracle threads a nonconstant state through the prescribed schedule,
but deliberately omits the round polynomial payload from `absorbRound`. -/
def roundOmittingOracle : Oracle Nat Nat oneVariableShape where
  transcript := {
    initialState := fun statement => statement.priorState
    absorbRound := fun state _ _ => state
    squeeze := fun state _ => (state, state + 1)
  }
  absorbOutput := fun state _ => state

def zeroRoundReplay : ReplayInput Nat Nat oneVariableShape where
  statement := { priorState := 0, input := oneVariableInput }
  rounds := zeroRoundCertificate

def oneRoundReplay : ReplayInput Nat Nat oneVariableShape where
  statement := { priorState := 0, input := oneVariableInput }
  rounds := oneRoundCertificate

private theorem zeroRoundReplay_ne_oneRoundReplay :
    zeroRoundReplay ≠ oneRoundReplay := by
  intro equal
  have messageEqual := congrArg
    (fun replay =>
      (replay.rounds.rounds
        (⟨0, by simp [oneVariableShape]⟩ :
          Fin oneVariableShape.cubeVariables)).coefficients)
    equal
  simp [zeroRoundReplay, oneRoundReplay, zeroRoundCertificate,
    oneRoundCertificate, zeroRoundMessage, oneRoundMessage] at messageEqual

/-- Two executions that differ only in their nonempty round message expose a
whole-replay collision when round absorption ignores that message. -/
theorem roundOmittingOracle_has_replayCollision :
    TranscriptReplayCollision roundOmittingOracle zeroRoundReplay oneRoundReplay := by
  refine ⟨zeroRoundReplay_ne_oneRoundReplay, ?_⟩
  constructor
  · rfl
  · constructor <;> rfl

/-- The same one-round omission also loses the message at the exact state
handed to output absorption. -/
theorem roundOmittingOracle_has_stateCollision :
    TranscriptStateCollision roundOmittingOracle zeroRoundReplay oneRoundReplay := by
  refine ⟨zeroRoundReplay_ne_oneRoundReplay, ?_⟩
  rfl

/-- Output absorption that ignores its incoming transcript is caught even
when the output message itself is unchanged. -/
theorem omittingOracle_has_outputStateCollision :
    OutputAbsorptionCollision omittingOracle 0 1 emptyOutput emptyOutput := by
  constructor
  · intro equal
    have stateEqual := congrArg Prod.fst equal
    change (0 : Nat) = 1 at stateEqual
    omega
  · rfl

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.Tests
