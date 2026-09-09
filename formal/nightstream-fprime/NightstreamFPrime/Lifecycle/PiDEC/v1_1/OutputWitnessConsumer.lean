import NightstreamFPrime.Spec.Folding.PiDEC.OutputWitnessConsumer
import NightstreamFPrime.Lifecycle.Stage1.Terminal

/-!
Consume the existing concrete terminal relation's 16 CE(b) witnesses at the
exact NIFS output. The verifier-selected relation and Ajtai key are shared by
terminal membership, NIFS acceptance, and PiDEC parent extraction.

This is the caller link for SuperNeo Theorem 13. Bare public acceptance does
not establish terminal membership, private openings, or a probability bound.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputWitnessConsumer

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.ProductionKey

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

variable (relation : LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The generic output consumer checks the same CE statement as the concrete
terminal relation, with no additional adapter premise. -/
theorem runningStatement_eq
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (index : Fin productionShape.runningCount) :
    Spec.Folding.PiDEC.OutputWitnessConsumer.runningStatement
        (key relation ajtai) result index =
      Lifecycle.runningStatement relation result index := by
  rfl

private theorem terminalOutputValid
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputWitness : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputFresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : Stage1.Terminal.FreshWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (terminal : Lifecycle.TerminalHolds relation ajtai result outputWitness
      outputFresh freshWitness) :
    ∀ index, CE.Holds (key relation ajtai).piRlcSemantics
      (key relation ajtai).params
      (Spec.Folding.PiDEC.OutputWitnessConsumer.runningStatement
        (key relation ajtai) result index) (outputWitness index) := by
  intro index
  rw [runningStatement_eq relation ajtai result index]
  exact terminal.1 index

variable (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (proof : Proof (degreeBound relation))
  (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (attempt : Spec.Folding.PiDEC.PaperVerifier.Attempt
    (PaperAlgebra.Structure logicalWidth)
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment
    productionGlobalParams)

/-- Terminal membership supplies all child openings for the actual accepted
NIFS output. Acceptance includes the literal `key.output = some result` link. -/
theorem terminalHolds_supplies_childOpenings
    (attemptEq : (key relation ajtai).piDecAttempt running fresh proof = some attempt)
    (accepted : Nifs.PaperNonInteractive.verify
      (key relation ajtai) running fresh proof = some result)
    (outputWitness : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputFresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : Stage1.Terminal.FreshWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (terminal : Lifecycle.TerminalHolds relation ajtai result outputWitness
      outputFresh freshWitness) :
    Nonempty (Nifs.PaperSecurityComposition.ChildOpenings
      (key relation ajtai) attempt) := by
  exact ⟨Spec.Folding.PiDEC.OutputWitnessConsumer.childOpeningsOfOutput
    (key relation ajtai) running fresh proof result attempt attemptEq accepted
    outputWitness (terminalOutputValid relation ajtai result outputWitness
      outputFresh freshWitness terminal)⟩

/-- The terminal's checked witness vector opens the PiDEC parent after the
same radix recomposition used by the production key. -/
theorem terminalHolds_extracts_parent
    (attemptEq : (key relation ajtai).piDecAttempt running fresh proof = some attempt)
    (accepted : Nifs.PaperNonInteractive.verify
      (key relation ajtai) running fresh proof = some result)
    (outputWitness : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputFresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : Stage1.Terminal.FreshWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (terminal : Lifecycle.TerminalHolds relation ajtai result outputWitness
      outputFresh freshWitness) :
    CE.Holds (semantics ajtai) productionGlobalParams attempt.parent
      ((piDecAlgebra ajtai).recomposeAssignment outputWitness) := by
  exact Spec.Folding.PiDEC.OutputWitnessConsumer.accepted_outputWitness_extracts_parent
    (key relation ajtai) running fresh proof result attempt attemptEq accepted
    outputWitness (terminalOutputValid relation ajtai result outputWitness
      outputFresh freshWitness terminal)

/-- The actual outer terminal payload supplies the openings. Its one running
slot must be the exact result of the accepted NIFS call. No commitment or
digest alone supplies this output-relation witness. -/
theorem recursiveTerminal_supplies_childOpenings
    (vk : KeyDigest) (application : Stage1.Application.Program)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Stage1.Terminal.RunningWitness
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Stage1.Terminal.FreshWitness
        (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (attemptEq : (key relation ajtai).piDecAttempt running fresh proof = some attempt)
    (accepted : Nifs.PaperNonInteractive.verify (key relation ajtai)
      running fresh proof = some (payload.running functionIndex))
    (terminal : Stage1.Terminal.HoldsFor relation ajtai vk application
      statement (.recursive payload)) :
    Nonempty (Nifs.PaperSecurityComposition.ChildOpenings
      (key relation ajtai) attempt) := by
  rcases (Stage1.Terminal.holdsFor_recursive_iff relation ajtai vk application
    statement payload).mp terminal with
    ⟨_pcValid, _positive, _publicLink, runningValid, freshValid⟩
  have memberships : Lifecycle.TerminalHolds relation ajtai
      (payload.running functionIndex) (payload.runningWitness functionIndex)
      payload.fresh payload.freshWitness :=
    ⟨runningValid functionIndex, freshValid⟩
  exact terminalHolds_supplies_childOpenings relation ajtai running fresh proof
    (payload.running functionIndex) attempt attemptEq accepted
    (payload.runningWitness functionIndex) payload.fresh payload.freshWitness
    memberships

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputWitnessConsumer
