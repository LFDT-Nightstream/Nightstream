import NightstreamFPrime.Layout.Stage1.PiDECProtocolCompleteness
import NightstreamFPrime.Layout.Stage1.PiCCSProofReadback
import NightstreamFPrime.Layout.Stage1.StateEncoding
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1
import NightstreamFPrime.Lifecycle.Stage1.Terminal

/-!
Owns the selected recursive HyperNova step's inputs to local C/R/D witness
construction. The semantic step supplies its actual NIFS acceptance and prior
public link. The existing state-validity and selected application contracts
supply fixed state widths. This module does not construct outer phase rows.
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

/-- A positive valid step for the selected Poseidon2 application constructs
local C/R/D witnesses from its own fresh instance and NIFS proof. Framing,
context, sampler availability, D checks, and the exact running output are
derived. The result covers these local phase rows; application, pilot,
transition, lowering, and low-norm assignment construction remain separate. -/
theorem recursive_completePrefix
    (valid : Lifecycle.Stage1.Terminal.StatementValid
      { iteration := input.iteration, z0 := input.z0, zi := input.zi })
    (step : StepHoldsFor relation ajtai context.toList
      Lifecycle.Stage1.Poseidon2HashChainV1.program input output)
    (positive : 0 < input.iteration) :
    let prior := priorHashPreimage (setup relation ajtai context.toList) input
    let next := nextHashPreimage (setup relation ajtai context.toList) input output
    let values := PiCCSProofReadback.ofProof
      (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof
    ∃ (priorFixed : PilotProduction.FixedPreimage prior)
      (nextFixed : PilotProduction.FixedPreimage next)
      (digestFixed : output.x.length = PilotProduction.digestWords),
      ∃ c : Sequence.Prefix
          (PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) next output.x
            priorFixed nextFixed digestFixed values context) PiCCSInputs.phaseOffset,
        ∃ r : Sequence.Prefix c.current PiRLCInputs.phaseOffset,
          ∃ d : Sequence.Prefix
              (PiDECProofInputs.load r.current input.nifsProof
                (PiRLC.v1_1.Semantics.evalOutput relation
                  (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
                  PiRLCInputs.phaseOffset r.current).publicInput) PiDECInputs.phaseOffset,
            c.operations = PiCCS.v1_1.Formal.opsAt relation
              (PiCCSProofInputs.relationInterface relation) PiCCSInputs.phaseOffset ∧
            r.operations = PiRLC.v1_1.Formal.opsAt relation
              (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
              PiRLCInputs.phaseOffset ∧
            d.operations = PiDEC.v1_1.Formal.opsAt relation
              (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset ∧
            holdsFlat d.current c.operations ∧ holdsFlat d.current r.operations ∧
            PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
              (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset d.current ∧
            RunningTransitionInputs.piDecRunningOutput relation d.current =
              output.runningNext functionIndex := by
  let prior := priorHashPreimage (setup relation ajtai context.toList) input
  let next := nextHashPreimage (setup relation ajtai context.toList) input output
  let values := PiCCSProofReadback.ofProof
    (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof
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
  have digestFixed : output.x.length = PilotProduction.digestWords := by
    have hash := step.2.2.1
    change output.x = stateHash next at hash
    rw [hash]
    exact StateEncoding.stateHash_length next
  have nextPc : next.pc = 1 := by
    change oneBased output.pcNext = 1
    rw [step.1]
    rfl
  have fresh : PiCCSProofInputs.protocolFresh logicalWidth publicFits
      (encHash (stateHash prior)) values = input.fresh := by
    have readback := PiCCSProofReadback.protocolFresh_ofProof relation input.fresh input.nifsProof
    rw [publicLink] at readback
    exact readback
  have proofReadback : PiCCSProofInputs.relationProof relation values input.nifsProof =
      input.nifsProof := PiCCSProofReadback.relationProof_ofProof relation
    (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof
  have actualAccepted : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      (prior.running functionIndex)
      (PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values)
      (PiCCSProofInputs.relationProof relation values input.nifsProof) =
        some (output.runningNext functionIndex) := by
    rw [fresh]
    rw [proofReadback]
    exact accepted
  refine ⟨priorFixed, nextFixed, digestFixed, ?_⟩
  have constructed := PiDECProtocolCompleteness.completePrefix relation ajtai prior
    (encHash (stateHash prior)) next output.x priorFixed nextFixed digestFixed values context
    input.nifsProof (output.runningNext functionIndex) priorPc nextPc rfl rfl actualAccepted
  simpa only [proofReadback] using constructed

end NightstreamFPrime.Layout.Stage1.PiDECStepCompleteness
