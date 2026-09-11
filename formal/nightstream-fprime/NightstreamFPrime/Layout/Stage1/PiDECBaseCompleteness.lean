import NightstreamFPrime.Layout.Stage1.PiDECStepCompleteness
import NightstreamFPrime.Lifecycle.Nifs.BaseVerifierCompleteness

/-!
Owns canonical base advice and its local C/R/D witness construction. The
HyperNova base branch permits dummy advice; this constructor selects the
existing zero proof and fresh claim, and preserves the same semantic output.
The verifier's dummy D output is separate from the base transition's default
running output. Bounded sampler success remains an explicit execution premise.
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

/-- Select the existing canonical dummy advice in the existing semantic
input record. State, iteration, and application witness are preserved. -/
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

/-- Canonical dummy advice preserves the same accepted base step, including
its application witness and public output. No NIFS acceptance or opening is
needed for this semantic normalization. -/
theorem canonicalInput_preserves_base
    (step : StepHoldsFor relation ajtai context.toList
      Lifecycle.Stage1.Poseidon2HashChainV1.program input output)
    (zero : input.iteration = 0) :
    StepHoldsFor relation ajtai context.toList Lifecycle.Stage1.Poseidon2HashChainV1.program
      (canonicalInput relation ajtai context input) output := by
  rcases step.2.2.2 with base | recursive
  · exact ⟨step.1, step.2.1, step.2.2.1, Or.inl ⟨zero, base.2.1, base.2.2⟩⟩
  · rcases recursive with ⟨_, positive, _⟩
    exact False.elim ((Nat.ne_of_gt positive) zero)

/-- The canonical base advice constructs local C/R/D rows when its actual
bounded sampler succeeds. The returned dummy D result is exactly the NIFS
verifier result. The same outer semantic base step still installs
`defaultRunning`; no child opening or equality with the dummy D result is
claimed. -/
theorem base_completePrefix
    (valid : Lifecycle.Stage1.Terminal.StatementValid
      { iteration := input.iteration, z0 := input.z0, zi := input.zi })
    (step : StepHoldsFor relation ajtai context.toList
      Lifecycle.Stage1.Poseidon2HashChainV1.program input output)
    (zero : input.iteration = 0)
    (challenges : Fin (ProductionKey.key relation ajtai).arity.total → RingF)
    (sampled : (ProductionKey.key relation ajtai).piRlcChallenges defaultRunning
      (Nifs.BaseCompleteness.baseFresh (priorHashPreimage (setup relation ajtai context.toList)
        (canonicalInput relation ajtai context input))) Nifs.BaseCompleteness.zeroProof = some challenges) :
    let normalized := canonicalInput relation ajtai context input
    let prior := priorHashPreimage (setup relation ajtai context.toList) normalized
    let next := nextHashPreimage (setup relation ajtai context.toList) normalized output
    let values := PiCCSProofReadback.ofProof
      (normalized.fresh.commitments ⟨0, by decide⟩) normalized.nifsProof
    StepHoldsFor relation ajtai context.toList Lifecycle.Stage1.Poseidon2HashChainV1.program
      normalized output ∧
    ∃ (priorFixed : PilotProduction.FixedPreimage prior)
      (nextFixed : PilotProduction.FixedPreimage next)
      (digestFixed : output.x.length = PilotProduction.digestWords),
      ∃ result, Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
          (prior.running functionIndex) normalized.fresh normalized.nifsProof = some result ∧
        ∃ c : Sequence.Prefix
            (PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) next output.x
              priorFixed nextFixed digestFixed values context) PiCCSInputs.phaseOffset,
          ∃ r : Sequence.Prefix c.current PiRLCInputs.phaseOffset,
            ∃ d : Sequence.Prefix
                (PiDECProofInputs.load r.current normalized.nifsProof
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
              RunningTransitionInputs.piDecRunningOutput relation d.current = result := by
  let normalized := canonicalInput relation ajtai context input
  let prior := priorHashPreimage (setup relation ajtai context.toList) normalized
  let next := nextHashPreimage (setup relation ajtai context.toList) normalized output
  let values := PiCCSProofReadback.ofProof
    (normalized.fresh.commitments ⟨0, by decide⟩) normalized.nifsProof
  have normalizedStep := canonicalInput_preserves_base relation ajtai context input output step zero
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
    have hash := normalizedStep.2.2.1
    change output.x = stateHash next at hash
    rw [hash]
    exact StateEncoding.stateHash_length next
  have nextPc : next.pc = 1 := by
    change oneBased output.pcNext = 1
    rw [step.1]
    rfl
  have fresh : PiCCSProofInputs.protocolFresh logicalWidth publicFits
      (encHash (stateHash prior)) values = normalized.fresh :=
    PiCCSProofReadback.protocolFresh_ofProof relation normalized.fresh normalized.nifsProof
  have proofReadback : PiCCSProofInputs.relationProof relation values normalized.nifsProof =
      normalized.nifsProof := PiCCSProofReadback.relationProof_ofProof relation
    (normalized.fresh.commitments ⟨0, by decide⟩) normalized.nifsProof
  obtain ⟨result, accepted⟩ := Nifs.BaseCompleteness.zeroProof_verify_of_sampler relation ajtai
    prior challenges sampled
  have actualAccepted : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      (prior.running functionIndex)
      (PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values)
      (PiCCSProofInputs.relationProof relation values normalized.nifsProof) = some result := by
    rw [fresh, proofReadback]
    exact accepted
  refine ⟨normalizedStep, priorFixed, nextFixed, digestFixed, result, accepted, ?_⟩
  have constructed := PiDECProtocolCompleteness.completePrefix relation ajtai prior
    (encHash (stateHash prior)) next output.x priorFixed nextFixed digestFixed values context
    normalized.nifsProof result rfl nextPc rfl rfl actualAccepted
  simpa only [proofReadback] using constructed

end NightstreamFPrime.Layout.Stage1.PiDECBaseCompleteness
