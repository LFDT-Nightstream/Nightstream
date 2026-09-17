import NightstreamFPrime.Export.Stage1.CheckedReplayNifs

/-! Bind the checked local replay to its exact augmented-function state.
The prior openings are supplied in the prior accepted payload. The exact
next running claim comes from the checked D fields. Successor opening
construction and terminal acceptance remain separate. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CheckedReplayStep

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup productionAjtaiKey)

/-- The accepted prior payload contains the exact C input claims and supplied
openings. This constructor adds no correctness equality as a premise. -/
def prior (input : PiCCSInputCheck.Input)
    (runningWitness : Stage1.Terminal.RunningWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (freshWitness : Stage1.Terminal.FreshWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application)) : HyperNovaHistory.Payload where
  running := fun _ => PiCCSInputCheck.running input
  runningWitness := fun _ => runningWitness
  fresh := PiCCSInputCheck.fresh input
  freshWitness := freshWitness
  pc := 1

/-- Actual successful C/R/D computations determine the exact semantic next
state. The premise accepts only the prior state; it does not supply a verified
local proof, a next-state equality, or an accepted successor. -/
theorem checked_step
    (statement : HyperNovaHistory.Statement)
    (input : PiCCSInputCheck.Input) (batch : PiRLCParent.Batch)
    (parent : PiRLCParent.Values) (messages : PiDECInputCheck.Messages)
    (runningWitness : Stage1.Terminal.RunningWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (freshWitness : Stage1.Terminal.FreshWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (advice : AppWitness)
    (sampled : PiRLCInputCheck.sampled input = some batch)
    (returned : PiRLCParent.computedParent input batch = some parent)
    (checked : PiDECInputCheck.accepted parent messages = true)
    (accepted : PerApplicationTerminal.Holds application fits productionSetup
      statement (.recursive (prior input runningWitness freshWitness)))
    (nonwrap : statement.iteration + 1 < goldilocksModulus) :
    let relation := PerApplicationFixedPoint.relation application fits
    let context := (PerApplicationCanonicalPackage.verifierContextDescriptor fits productionSetup).digest4
    let before := HyperNovaStepData.input statement (prior input runningWitness freshWitness)
      advice (CheckedReplayNifs.proof input messages)
    let after := HyperNovaStepData.output statement advice
      (PiCCSInputCheck.runningFromInput messages)
    StepHoldsFor relation productionAjtaiKey context.toList application before after ∧
    StateEncoding.WellFormed (priorHashPreimage (setup relation productionAjtaiKey context.toList) before) ∧
    StateEncoding.WellFormed (nextHashPreimage (setup relation productionAjtaiKey context.toList) before after) ∧
    before.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation productionAjtaiKey context.toList) before)) ∧
    PiCCSInputCheck.runningFromInput messages = after.runningNext functionIndex := by
  exact HyperNovaStepData.stepHolds_and_wellFormed statement
    (prior input runningWitness freshWitness) advice
    (CheckedReplayNifs.proof input messages) (PiCCSInputCheck.runningFromInput messages)
    accepted (CheckedReplayNifs.checked_verifies_selected input batch parent messages
      sampled returned checked) nonwrap

end NightstreamFPrime.Export.Stage1.CheckedReplayStep
