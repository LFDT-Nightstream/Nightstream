import NightstreamFPrime.Layout.Stage1.PiDECStepCompleteness
import NightstreamFPrime.Lifecycle.Nifs.BaseVerifierCompleteness

/-!
Owns canonical base advice and its pilot/C/R/D/transition witness construction. The
HyperNova base branch permits dummy advice; this constructor selects the
existing zero proof and fresh claim, and preserves the same semantic output.
The verifier's dummy D output is separate from the base transition's default
running output. The production sampler is total.
-/

namespace NightstreamFPrime.Layout.Stage1.PiDECBaseCompleteness

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

noncomputable def canonicalInput : Input KeyDigest AppState AppWitness
    (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (Proof (ProductionKey.degreeBound relation)) slotCount :=
  let normalized := { input with running := fun _ => defaultRunning, priorPc := 1 }
  { normalized with
    fresh := Nifs.BaseCompleteness.baseFresh
      (priorHashPreimage (setup relation ajtai context.toList) normalized)
    nifsProof := Nifs.BaseCompleteness.zeroProof }

variable
  (output : Output Digest AppState
    (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)

theorem canonicalInput_preserves_base
    (program : Lifecycle.Stage1.Application.Program)
    (step : StepHoldsFor relation ajtai context.toList program input output)
    (zero : input.iteration = 0) :
    StepHoldsFor relation ajtai context.toList program
      (canonicalInput relation ajtai context input) output := by
  rcases step.2.2.2 with base | recursive
  · exact ⟨step.1, step.2.1, step.2.2.1, Or.inl ⟨zero, base.2.1, base.2.2⟩⟩
  · rcases recursive with ⟨_, positive, _⟩
    exact False.elim ((Nat.ne_of_gt positive) zero)

theorem base_completePrefix
    (valid : Lifecycle.Stage1.Terminal.StatementValid
      { iteration := input.iteration, z0 := input.z0, zi := input.zi })
    (program : Lifecycle.Stage1.Application.Program)
    (step : StepHoldsFor relation ajtai context.toList program input output)
    (zero : input.iteration = 0) :
    let normalized := canonicalInput relation ajtai context input
    let prior := priorHashPreimage (setup relation ajtai context.toList) normalized
    let next := nextHashPreimage (setup relation ajtai context.toList) normalized output
    let values := PiCCSProofReadback.ofProof
      (normalized.fresh.commitments ⟨0, by decide⟩) normalized.nifsProof
    StepHoldsFor relation ajtai context.toList program
      normalized output ∧
    ∃ (priorWellFormed : StateEncoding.WellFormed prior)
      (nextWellFormed : StateEncoding.WellFormed next),
      ∃ result, Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
          (prior.running functionIndex) normalized.fresh normalized.nifsProof = some result ∧
    ∃ (digestFixed : output.x.length = PilotProduction.digestWords),
      ∃ p : Sequence.Prefix
          (PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) next output.x
            priorWellFormed.1 nextWellFormed.1 digestFixed values context) PilotProduction.witnessOffset,
        ∃ c : Sequence.Prefix p.current PiCCSInputs.phaseOffset,
          ∃ r : Sequence.Prefix c.current PiRLCInputs.phaseOffset,
            ∃ d : Sequence.Prefix
                (PiDECProofInputs.load r.current normalized.nifsProof
                  (PiRLC.v1_2.Semantics.evalOutput relation
                    (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
                    PiRLCInputs.phaseOffset r.current).publicInput) PiDECInputs.phaseOffset,
              ∃ t : Sequence.Prefix d.current RunningTransitionInputs.phaseOffset,
                flatConstraints p.operations = Pilot.logicalConstraints PilotProduction.interface PilotProduction.witnessOffset ∧
                c.operations = PiCCS.v1_2.Formal.opsAt relation (PiCCSProofInputs.relationInterface relation) PiCCSInputs.phaseOffset ∧
                r.operations = PiRLC.v1_2.Formal.opsAt relation PiRLCInputs.interface PiRLCInputs.phaseOffset ∧
                d.operations = PiDEC.v1_2.Formal.opsAt relation (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset ∧
                t.operations = Lifecycle.Stage1.RunningTransition.operations
                  (RunningTransitionInputs.interface logicalWidth publicFits) RunningTransitionInputs.phaseOffset ∧
                holdsFlat t.current p.operations ∧ holdsFlat t.current c.operations ∧
                holdsFlat t.current r.operations ∧ holdsFlat t.current d.operations ∧
                holdsFlat t.current (Lifecycle.Stage1.NextPreimage.opsAt NextPreimageInputs.sourceInterface
                  RunningTransitionInputs.phaseOffset) ∧
                Lifecycle.Pilot.SpecHolds PilotProduction.interface PilotProduction.witnessOffset t.current ∧
                PiDEC.v1_2.Semantics.PhaseHolds relation ajtai (PiDECInputs.interface logicalWidth publicFits)
                  PiDECInputs.phaseOffset t.current ∧
                Lifecycle.Stage1.RunningTransition.SpecHolds (RunningTransitionInputs.interface logicalWidth publicFits)
                  RunningTransitionInputs.phaseOffset t.current ∧
                Lifecycle.Stage1.NextPreimage.SpecHolds NextPreimageInputs.sourceInterface RunningTransitionInputs.phaseOffset t.current ∧
                RunningTransitionInputs.piDecRunningOutput relation t.current = result ∧
                (∀ index : Fin PilotProduction.stateHashWords,
                  t.current (PilotProduction.priorPreimageStart + index.val) =
                    (serializePreimage (publicFits := publicFits) prior).getD index.val 0) ∧
                (∀ index : Fin PilotProduction.stateHashWords,
                  t.current (PilotProduction.outputPreimageStart + index.val) =
                    (serializePreimage (publicFits := publicFits) next).getD index.val 0) := by
  let normalized := canonicalInput relation ajtai context input
  let prior := priorHashPreimage (setup relation ajtai context.toList) normalized
  let next := nextHashPreimage (setup relation ajtai context.toList) normalized output
  have normalizedStep := canonicalInput_preserves_base relation ajtai context input output program step zero
  have priorFixed : PilotProduction.FixedPreimage prior :=
    ⟨context.toList_length, valid.2.1, valid.2.2⟩
  have nextWidth : output.zNext.length = PilotProduction.digestWords := by
    have applicationStep := step.2.1
    change output.zNext = program.step input.zi input.witness at applicationStep
    rw [applicationStep]
    exact program.step_length input.zi input.witness valid.2.2
  have nextFixed : PilotProduction.FixedPreimage next :=
    ⟨context.toList_length, valid.2.1, nextWidth⟩
  have nextPc : next.pc = 1 := by
    change oneBased output.pcNext = 1
    rw [step.1]
    rfl
  have successor : next.iteration < goldilocksModulus := by
    change input.iteration + 1 < goldilocksModulus
    rw [zero]
    norm_num [goldilocksModulus]
  have nextCanonical : Lifecycle.ChildrenCanonical (next.running functionIndex) := by
    rcases step.2.2.2 with base | recursive
    · change Lifecycle.ChildrenCanonical (output.runningNext functionIndex)
      rw [congrFun base.2.2 functionIndex]
      exact StateEncoding.defaultRunning_canonical
    · rcases recursive with ⟨_, positive, _⟩
      exact False.elim ((Nat.ne_of_gt positive) zero)
  have priorWellFormed : StateEncoding.WellFormed prior :=
    ⟨priorFixed, valid.1, rfl, StateEncoding.defaultRunning_canonical⟩
  have nextWellFormed : StateEncoding.WellFormed next :=
    ⟨nextFixed, successor, nextPc, nextCanonical⟩
  obtain ⟨result, accepted⟩ := Nifs.BaseCompleteness.zeroProof_verify relation ajtai prior
  refine ⟨normalizedStep, priorWellFormed, nextWellFormed, result, accepted, ?_⟩
  apply StepWitnessPrefix.completePrefix relation ajtai context normalized output result
    program normalizedStep priorWellFormed nextWellFormed rfl accepted
  intro positive
  change 0 < input.iteration at positive
  omega

end NightstreamFPrime.Layout.Stage1.PiDECBaseCompleteness
