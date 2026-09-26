import NightstreamFPrime.Export.Stage1.HyperNovaCompleteness
import NightstreamFPrime.Layout.Stage1.StateEncoding
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1
import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns the selected semantic step data after the honest NIFS call returns.
The accepted prior payload supplies the public link and prior framing.
The output, next hash, and both WellFormed facts are constructed here.
Physical and complete assignment construction remain separate.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaStepData

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1
variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The actual prior payload, selected application advice, and returned local
proof are the existing augmented-function input fields. -/
def input (statement : TerminalStatement AppState)
    (payload : TerminalProof
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Stage1.Terminal.RunningWitness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Stage1.Terminal.FreshWitness (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (advice : AppWitness) (proof : Lifecycle.Proof 9) :
    Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth)
        (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth)
        (publicFits := publicFits))
      (Lifecycle.Proof 9) slotCount where
  iteration := statement.iteration
  z0 := statement.z0
  zi := statement.zi
  running := payload.running
  fresh := payload.fresh
  priorPc := payload.pc
  witness := advice
  nifsProof := proof

/-- Compute the selected application result and the full next-state digest.
The NIFS result fills the one outer running slot; no output is supplied. -/
def output (context : VerifierContext.Digest4) (application : Stage1.Application.Program) (statement : TerminalStatement AppState) (advice : AppWitness)
    (result : Running (logicalWidth := logicalWidth)
      (publicFits := publicFits)) :
    Output Digest AppState
      (Running (logicalWidth := logicalWidth)
        (publicFits := publicFits)) slotCount where
  zNext := application.step statement.zi advice
  runningNext := fun _ => result
  pcNext := functionIndex
  x := stateHash {
    verifierKeys := fun _ => context.toList
    iteration := statement.iteration + 1
    z0 := statement.z0
    current := application.step statement.zi advice
    running := fun _ => result
    pc := 1 }

/-- A returned honest NIFS result determines the exact recursive semantic
step and both canonical state frames. Prior framing and the public link are
derived from terminal acceptance; next framing uses only counter nonwrap.
Advice width belongs to later application witness wiring, not this result. -/
private abbrev application := Stage1.Poseidon2HashChainV1.program

theorem stepHolds_and_wellFormed
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : VerifierContext.Digest4)
    (statement : TerminalStatement AppState)
        (payload : TerminalProof
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Stage1.Terminal.RunningWitness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Stage1.Terminal.FreshWitness (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (advice : AppWitness) (proof : Lifecycle.Proof 9)
    (result : Running (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (accepted : Stage1.Terminal.HoldsFor relation ajtai context.toList application
      statement (.recursive payload))
    (verified : Nifs.PaperNonInteractive.verify
      (ProductionKey.key relation ajtai)
      (payload.running functionIndex) payload.fresh proof = some result)
    (nonwrap : statement.iteration + 1 < goldilocksModulus) :
    let before := input statement payload advice proof
    let after := output context application statement advice result
    StepHoldsFor relation ajtai context.toList application before after ∧
    StateEncoding.WellFormed (priorHashPreimage (setup relation ajtai context.toList) before) ∧
    StateEncoding.WellFormed (nextHashPreimage (setup relation ajtai context.toList) before after) ∧
    before.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation ajtai context.toList) before)) ∧
    result = after.runningNext functionIndex := by
  let before := input statement payload advice proof
  let after := output context application statement advice result
  obtain ⟨valid, pcValid, positive, publicLink, _running, _fresh⟩ :=
    (Stage1.Terminal.holdsFor_recursive_iff relation ajtai context.toList application statement payload).mp accepted
  have pc : payload.pc = 1 := by
    change 1 ≤ payload.pc ∧ payload.pc ≤ 1 at pcValid
    omega
  have selected : selectedIndex pcValid = functionIndex := by
    apply Fin.ext
    have bound := (selectedIndex pcValid).isLt
    change (selectedIndex pcValid).val < 1 at bound
    change (selectedIndex pcValid).val = 0
    omega
  have link : before.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation ajtai context.toList) before)) :=
    publicLink
  have step : StepHoldsFor relation ajtai context.toList application before after := by
    refine ⟨rfl, rfl, rfl, Or.inr ⟨pcValid, positive, link, ?_, ?_⟩⟩
    · dsimp only [HyperNova.NonInteractiveMultiFold.Accepts, setup, nifsVerifier,
        before, after, input, output]
      exact Eq.mpr (congrArg (fun index : Fin slotCount =>
        Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
          (payload.running index) payload.fresh proof = some result) selected) verified
    · intro slot different
      have same : slot = selectedIndex pcValid := by
        apply Fin.ext
        have slotBound := slot.isLt
        have selectedBound := (selectedIndex pcValid).isLt
        change slot.val < 1 at slotBound
        change (selectedIndex pcValid).val < 1 at selectedBound
        omega
      exact False.elim (different same)
  have priorFixed : NightstreamFPrime.Layout.PilotProduction.FixedPreimage
      (priorHashPreimage (setup relation ajtai context.toList) before) :=
    ⟨context.toList_length, valid.2.1, valid.2.2⟩
  have nextWidth : after.zNext.length = Stage1.Application.stateWordCount :=
    Stage1.Poseidon2HashChainV1.step_output_length statement.zi advice
  have nextFixed : NightstreamFPrime.Layout.PilotProduction.FixedPreimage
      (nextHashPreimage (setup relation ajtai context.toList) before after) :=
    ⟨context.toList_length, valid.2.1, nextWidth⟩
  exact ⟨step, ⟨priorFixed, valid.1, pc⟩, ⟨nextFixed, nonwrap, rfl⟩, link, rfl⟩

end NightstreamFPrime.Export.Stage1.HyperNovaStepData
