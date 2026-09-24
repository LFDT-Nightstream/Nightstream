import NightstreamFPrime.Layout.Stage1.AccumulatorSemantics
import NightstreamFPrime.Layout.Stage1.Wide.PiRLCInputs
import NightstreamFPrime.Lifecycle.PiRLC.Wide.Key

/-! The wide phase consumes the same PiCCS outputs and constructs the exact
parent computed by the wide NIFS key. Its transcript response is total. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.AccumulatorSemantics

open NightstreamFPrime.Circuit NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

theorem inputs_eq_keyOutputs
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) (env : Env)
    (phase : PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (Stage1.AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      Stage1.PiCCSInputs.phaseOffset env (Stage1.AccumulatorInputs.proof relation env)) :
    PiRLC.Wide.Semantics.evalInputs relation
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset env =
      (PiRLC.Wide.Key.key relation ajtai).piCcsOutputs
        (Stage1.AccumulatorInputs.running logicalWidth publicFits env)
        (Stage1.AccumulatorInputs.fresh logicalWidth publicFits env) (Stage1.AccumulatorInputs.proof relation env) := by
  rw [PiRLC.Wide.Key.piCcsOutputs_unchanged]
  have same : PiRLC.Wide.Semantics.evalInputs relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset env =
      PiRLC.v1_1.Semantics.evalInputs relation
        (Stage1.PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) Stage1.PiRLCInputs.phaseOffset env := by
    unfold PiRLC.Wide.Semantics.evalInputs PiRLC.v1_1.Semantics.evalInputs
    rfl
  exact same.trans (Stage1.AccumulatorSemantics.piRlcInputs_eq_keyOutputs relation ajtai env phase)

theorem challenges_eq_key
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env) (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (interface : PiRLC.Wide.Formal.Interface logicalWidth publicFits) (offset : Nat)
    (phase : PiRLC.Wide.Semantics.PhaseHolds relation ajtai interface offset env)
    (initial : PiRLC.Wide.Scalar.evalState env (interface.initialState offset) =
      ((PiRLC.Wide.Key.key relation ajtai).piCcsExecution running fresh proof).outgoingState) :
    (PiRLC.Wide.Key.key relation ajtai).piRlcChallenges running fresh proof =
      some (PiRLC.Wide.Semantics.evalChallenges interface offset env) := by
  change (PiRLC.Wide.Key.key relation ajtai).piRlcResponse
    ((PiRLC.Wide.Key.key relation ajtai).piCcsExecution running fresh proof).outgoingState = _
  rw [← initial]
  exact PiRLC.Wide.Key.phase_response relation ajtai interface offset env phase

theorem output_eq_keyParent
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env) (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (interface : PiRLC.Wide.Formal.Interface logicalWidth publicFits) (offset : Nat)
    (phase : PiRLC.Wide.Semantics.PhaseHolds relation ajtai interface offset env)
    (inputs : PiRLC.Wide.Semantics.evalInputs relation interface offset env =
      (PiRLC.Wide.Key.key relation ajtai).piCcsOutputs running fresh proof) :
    PiRLC.Wide.Semantics.evalOutput relation interface offset env =
      (PiRLC.Wide.Key.key relation ajtai).parentForChallenges running fresh proof
        (PiRLC.Wide.Semantics.evalChallenges interface offset env) := by
  let first : Fin Nifs.PaperProfile.arity.total := ⟨0, by decide⟩
  have inputStructure := congrArg (fun values => (values first).constraintSystem) inputs
  have inputPoint := congrArg (fun values => (values first).point) inputs
  change (PiRLC.Wide.Semantics.evalInputs relation interface offset env first).constraintSystem =
    (PiRLC.Wide.Key.key relation ajtai).relationSource at inputStructure
  change (PiRLC.Wide.Semantics.evalInputs relation interface offset env first).point =
    ((PiRLC.Wide.Key.key relation ajtai).piCcsExecution running fresh proof).coins.roundPoint at inputPoint
  have outputStructure : (PiRLC.Wide.Semantics.evalOutput relation interface offset env).constraintSystem =
      (PiRLC.Wide.Key.key relation ajtai).relationSource := (phase.accepted.sameStructure first).symm.trans inputStructure
  have outputPoint : (PiRLC.Wide.Semantics.evalOutput relation interface offset env).point =
      ((PiRLC.Wide.Key.key relation ajtai).piCcsExecution running fresh proof).coins.roundPoint :=
    (phase.accepted.samePoint first).symm.trans inputPoint
  rw [PiRLC.Wide.Semantics.output_eq_combinedOutput relation ajtai interface offset env phase]
  unfold Nifs.PaperNonInteractive.Key.parentForChallenges
  rw [inputs, outputStructure, outputPoint]
  rfl

end NightstreamFPrime.Layout.Stage1.Wide.AccumulatorSemantics
