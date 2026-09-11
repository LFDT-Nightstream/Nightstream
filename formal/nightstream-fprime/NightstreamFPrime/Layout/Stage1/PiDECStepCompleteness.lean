import NightstreamFPrime.Layout.Stage1.StepWitnessPrefix
import NightstreamFPrime.Layout.Stage1.PiCCSProofReadback
import NightstreamFPrime.Layout.Stage1.StateEncoding
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1
import NightstreamFPrime.Lifecycle.Stage1.Terminal

/-!
Owns the selected recursive step adapter to pilot, C/R/D, running-transition,
and next-preimage witness construction. The actual semantic step supplies
NIFS acceptance and the prior public link. The selected state validity and
successor bound supply canonical state framing. Application construction and
physical lowering remain separate.
-/

namespace NightstreamFPrime.Layout.Stage1.PiDECStepCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (context : VerifierContext.Digest4)
  (input : Input KeyDigest AppState AppWitness
    (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (Proof (ProductionKey.degreeBound relation)) slotCount)
  (output : Output Digest AppState
    (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)

private theorem recursive_inputs
    (step : StepHoldsFor relation ajtai context.toList
      Lifecycle.Stage1.Poseidon2HashChainV1.program input output)
    (positive : 0 < input.iteration) :
    input.priorPc = 1 ∧
    input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation ajtai context.toList) input)) ∧
    Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      (input.running functionIndex) input.fresh input.nifsProof =
        some (output.runningNext functionIndex) := by
  rcases step.2.2.2 with base | recursive
  · exact False.elim ((Nat.ne_of_gt positive) base.1)
  · rcases recursive with ⟨pcValid, _, publicLink, accepted, _⟩
    have pc : input.priorPc = 1 := by
      change 1 ≤ input.priorPc ∧ input.priorPc ≤ 1 at pcValid
      omega
    have selected : selectedIndex pcValid = functionIndex := by
      apply Fin.ext
      have bound := (selectedIndex pcValid).isLt
      change (selectedIndex pcValid).val < 1 at bound
      change (selectedIndex pcValid).val = 0
      omega
    change input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation ajtai context.toList) input)) at publicLink
    dsimp only [HyperNova.NonInteractiveMultiFold.Accepts, setup, nifsVerifier] at accepted
    have selectedAccepted := Eq.mp (congrArg (fun index : Fin slotCount =>
      Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
        (input.running index) input.fresh input.nifsProof =
          some (output.runningNext index)) selected) accepted
    exact ⟨pc, publicLink, selectedAccepted⟩

/-- A positive selected step constructs pilot, C/R/D, running-transition,
and next-preimage rows in one environment. Its actual proof supplies sampler
availability and the exact NIFS result. State validity and the successor
bound permit exact natural-counter readback. No generated row, phase value,
or source agreement is assumed. -/
theorem recursive_completePrefix
    (valid : Lifecycle.Stage1.Terminal.StatementValid
      { iteration := input.iteration, z0 := input.z0, zi := input.zi })
    (step : StepHoldsFor relation ajtai context.toList
      Lifecycle.Stage1.Poseidon2HashChainV1.program input output)
    (positive : 0 < input.iteration)
    (successor : input.iteration + 1 < goldilocksModulus) :
    let prior := priorHashPreimage (setup relation ajtai context.toList) input
    let next := nextHashPreimage (setup relation ajtai context.toList) input output
    let values := PiCCSProofReadback.ofProof
      (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof
    ∃ (priorWellFormed : StateEncoding.WellFormed prior)
      (nextWellFormed : StateEncoding.WellFormed next),
    ∃ (digestFixed : output.x.length = PilotProduction.digestWords),
      ∃ p : Sequence.Prefix
          (PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) next output.x
            priorWellFormed.1 nextWellFormed.1 digestFixed values context) PilotProduction.witnessOffset,
        ∃ c : Sequence.Prefix p.current PiCCSInputs.phaseOffset,
          ∃ r : Sequence.Prefix c.current PiRLCInputs.phaseOffset,
            ∃ d : Sequence.Prefix
                (PiDECProofInputs.load r.current input.nifsProof
                  (PiRLC.v1_1.Semantics.evalOutput relation
                    (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
                    PiRLCInputs.phaseOffset r.current).publicInput) PiDECInputs.phaseOffset,
              ∃ t : Sequence.Prefix d.current RunningTransitionInputs.phaseOffset,
                flatConstraints p.operations = Pilot.logicalConstraints PilotProduction.interface PilotProduction.witnessOffset ∧
                c.operations = PiCCS.v1_1.Formal.opsAt relation (PiCCSProofInputs.relationInterface relation) PiCCSInputs.phaseOffset ∧
                r.operations = PiRLC.v1_1.Formal.opsAt relation PiRLCInputs.interface PiRLCInputs.phaseOffset ∧
                d.operations = PiDEC.v1_1.Formal.opsAt relation (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset ∧
                t.operations = Lifecycle.Stage1.RunningTransition.operations
                  (RunningTransitionInputs.interface logicalWidth publicFits) RunningTransitionInputs.phaseOffset ∧
                holdsFlat t.current p.operations ∧ holdsFlat t.current c.operations ∧
                holdsFlat t.current r.operations ∧ holdsFlat t.current d.operations ∧
                holdsFlat t.current (Lifecycle.Stage1.NextPreimage.opsAt NextPreimageInputs.sourceInterface
                  RunningTransitionInputs.phaseOffset) ∧
                Lifecycle.Pilot.SpecHolds PilotProduction.interface PilotProduction.witnessOffset t.current ∧
                PiDEC.v1_1.Semantics.PhaseHolds relation ajtai (PiDECInputs.interface logicalWidth publicFits)
                  PiDECInputs.phaseOffset t.current ∧
                Lifecycle.Stage1.RunningTransition.SpecHolds (RunningTransitionInputs.interface logicalWidth publicFits)
                  RunningTransitionInputs.phaseOffset t.current ∧
                Lifecycle.Stage1.NextPreimage.SpecHolds NextPreimageInputs.sourceInterface RunningTransitionInputs.phaseOffset t.current ∧
                RunningTransitionInputs.piDecRunningOutput relation t.current = output.runningNext functionIndex ∧
                (∀ index : Fin PilotProduction.stateHashWords,
                  t.current (PilotProduction.priorPreimageStart + index.val) =
                    (serializePreimage (publicFits := publicFits) prior).getD index.val 0) ∧
                (∀ index : Fin PilotProduction.stateHashWords,
                  t.current (PilotProduction.outputPreimageStart + index.val) =
                    (serializePreimage (publicFits := publicFits) next).getD index.val 0) := by
  let prior := priorHashPreimage (setup relation ajtai context.toList) input
  let next := nextHashPreimage (setup relation ajtai context.toList) input output
  obtain ⟨priorPc, publicLink, accepted⟩ := recursive_inputs relation ajtai context input output step positive
  have priorFixed : PilotProduction.FixedPreimage prior :=
    ⟨context.toList_length, valid.2.1, valid.2.2⟩
  have nextWidth : output.zNext.length = PilotProduction.digestWords := by
    have applicationStep := step.2.1
    change output.zNext = Lifecycle.Stage1.Poseidon2HashChainV1.step input.zi input.witness at applicationStep
    rw [applicationStep]
    exact Lifecycle.Stage1.Poseidon2HashChainV1.step_output_length input.zi input.witness
  have nextFixed : PilotProduction.FixedPreimage next :=
    ⟨context.toList_length, valid.2.1, nextWidth⟩
  have nextPc : next.pc = 1 := by
    change oneBased output.pcNext = 1
    rw [step.1]
    rfl
  have priorWellFormed : StateEncoding.WellFormed prior := ⟨priorFixed, valid.1, priorPc⟩
  have nextWellFormed : StateEncoding.WellFormed next := ⟨nextFixed, successor, nextPc⟩
  exact ⟨priorWellFormed, nextWellFormed,
    StepWitnessPrefix.completePrefix relation ajtai context input output
      (output.runningNext functionIndex) step priorWellFormed nextWellFormed
      publicLink accepted (fun _ => rfl)⟩

end NightstreamFPrime.Layout.Stage1.PiDECStepCompleteness
