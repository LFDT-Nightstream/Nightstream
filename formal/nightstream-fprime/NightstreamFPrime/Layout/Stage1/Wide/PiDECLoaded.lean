import NightstreamFPrime.Layout.Stage1.Wide.PiDECInputBounds
import NightstreamFPrime.Layout.Stage1.Wide.PiDECProofInputs
import NightstreamFPrime.Layout.Stage1.PiDECProtocolCompleteness
import NightstreamFPrime.Lifecycle.PiRLC.Wide.Key

/-! Load PiDEC's actual proof fields for the wide parent. The resulting
semantic phase follows from the verifier's checks on that exact parent. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.PiDECLoaded

open NightstreamFPrime.Circuit NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint
open Stage1.PiDECProtocolCompleteness
  (evaluation_ext message_ext attempt_ext instance_ext children_outputAccepted)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)

def parent (env : Env) := PiRLC.Wide.Semantics.evalOutput relation
  (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset env

theorem parent_eq_of_agree (before after : Env)
    (agrees : ∀ index, index < PiRLCStarts.phaseFreshStart → before index = after index) :
    parent relation before = parent relation after := by
  exact PiRLC.v1_1.OutputBinding.evalOutput_eq_of_agree relation
    (PiDECInputs.piRlcOutputInterface logicalWidth publicFits) PiRLCStarts.outputLogicalStart
    PiRLCStarts.phaseFreshStart before after (PiDECInputs.parentBelow relation) agrees

theorem parent_preserved (env : Env) (proof : Proof (ProductionKey.degreeBound relation))
    (publicInput : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    parent relation (PiDECProofInputs.load env proof publicInput) = parent relation env := by
  apply parent_eq_of_agree
  intro index below
  exact PiDECProofInputs.load_agreesOutside env proof publicInput index
    (Or.inl (lt_of_lt_of_le below PiDECInputs.parentEnd_le_proofInputStart))

theorem inputParent (env : Env) :
    (PiDEC.v1_1.Semantics.inputAttempt relation (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset env).parent = parent relation env := rfl

variable (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

private theorem loaded_message (env : Env) (proof : Proof (ProductionKey.degreeBound relation))
    (publicInput : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex) :
    PiDEC.v1_1.InputBinding.evalMessage (PiDECInputs.message child) (PiDECProofInputs.load env proof publicInput) =
      { commitment := proof.piDecCommitments child, evaluations := #[proof.piDecEvaluations child] } := by
  apply message_ext
  · dsimp only [PiDEC.v1_1.InputBinding.evalMessage, PiDECInputs.message]
    funext row lane
    exact PiDECProofInputs.eval_childCommitment (logicalWidth := logicalWidth) (publicFits := publicFits) env proof publicInput child row lane
  · apply congrArg (fun value : PaperAlgebra.Evaluation => #[value])
    apply evaluation_ext
    · funext coefficient
      exact PiDECProofInputs.eval_childEvalK (logicalWidth := logicalWidth) (publicFits := publicFits) env proof publicInput child coefficient
    · funext matrix coefficient
      exact PiDECProofInputs.eval_childEvalA (logicalWidth := logicalWidth) (publicFits := publicFits) env proof publicInput child matrix coefficient

theorem loaded_attempt (env : Env) (proof : Proof (ProductionKey.degreeBound relation))
    (publicInput : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    PiDEC.v1_1.Semantics.inputAttempt relation (PiDECInputs.interface logicalWidth publicFits)
        PiDECInputs.phaseOffset (PiDECProofInputs.load env proof publicInput) =
      (PiRLC.Wide.Key.key relation ajtai).piDecAttemptForParent proof (parent relation env) := by
  apply attempt_ext
  · exact (inputParent relation _).trans (parent_preserved relation env proof publicInput)
  · funext child
    exact loaded_message relation env proof publicInput child

theorem loaded_output (env : Env) (proof : Proof (ProductionKey.degreeBound relation)) :
    PiDEC.v1_1.Semantics.output relation (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset
        (PiDECProofInputs.load env proof (parent relation env).publicInput) =
      PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai)
        ((PiRLC.Wide.Key.key relation ajtai).piDecAttemptForParent proof (parent relation env)) := by
  funext child
  apply instance_ext
  · rfl
  · dsimp only [PiDEC.v1_1.Semantics.output, PiDEC.v1_1.OutputBinding.evalOutput,
      PiDEC.v1_1.Formal.outputBindingInterface, PiDEC.v1_1.Formal.atOffset,
      PiDECInputs.interface, PiDECInputs.message, PiDEC.PaperVerifier.children,
      Nifs.PaperNonInteractive.Key.piDecAttemptForParent]
    funext row lane
    exact PiDECProofInputs.eval_childCommitment (logicalWidth := logicalWidth) (publicFits := publicFits) env proof (parent relation env).publicInput child row lane
  · dsimp only [PiDEC.v1_1.Semantics.output, PiDEC.v1_1.OutputBinding.evalOutput,
      PiDEC.v1_1.Formal.outputBindingInterface, PiDEC.v1_1.Formal.atOffset,
      PiDECInputs.interface, PiDEC.PaperVerifier.children,
      Nifs.PaperNonInteractive.Key.piDecAttemptForParent, PaperAlgebra.publicInputSplit]
    funext coordinate
    exact PiDECProofInputs.eval_childPublicInput (logicalWidth := logicalWidth) (publicFits := publicFits) env proof (parent relation env).publicInput child coordinate
  · exact congrArg (fun value => value.point) (parent_preserved relation env proof (parent relation env).publicInput)
  · exact congrArg (fun message => message.evaluations)
      (loaded_message relation env proof (parent relation env).publicInput child)
  · rfl

theorem loaded_phase (env : Env) (proof : Proof (ProductionKey.degreeBound relation))
    (checks : PiDEC.PaperVerifier.Accepted (PiRLC.Wide.Key.key relation ajtai).piDecAlgebra
      (PiRLC.Wide.Key.key relation ajtai).piDecPublicInputSplit (PiRLC.Wide.Key.key relation ajtai).piDecEvaluationArity
      ((PiRLC.Wide.Key.key relation ajtai).piDecAttemptForParent proof (parent relation env))) :
    PiDEC.v1_1.Semantics.PhaseHolds relation ajtai (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset
      (PiDECProofInputs.load env proof (parent relation env).publicInput) := by
  unfold PiDEC.v1_1.Semantics.PhaseHolds
  rw [loaded_attempt relation ajtai, loaded_output relation ajtai]
  exact children_outputAccepted ajtai _ checks

end NightstreamFPrime.Layout.Stage1.Wide.PiDECLoaded
