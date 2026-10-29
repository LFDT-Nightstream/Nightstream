import NightstreamFPrime.Export.Stage1.HyperNovaCompleteness
import NightstreamFPrime.Layout.Stage1.StateEncoding

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaStepData

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1

variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup application)

def input (statement : HyperNovaHistory.Statement) (payload : HyperNovaHistory.Payload application)
    (advice : AppWitness) (proof : Lifecycle.Proof 8) :
    Input KeyDigest AppState AppWitness
      (Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Lifecycle.Proof 8) slotCount where
  iteration := statement.iteration
  z0 := statement.z0
  zi := statement.zi
  running := payload.running
  fresh := payload.fresh
  priorPc := payload.pc
  witness := advice
  nifsProof := proof

def output (statement : HyperNovaHistory.Statement) (advice : AppWitness)
    (result : Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application)) :
    Output Digest AppState
      (Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application)) slotCount where
  zNext := application.step statement.zi advice
  runningNext := fun _ => result
  pcNext := functionIndex
  x := stateHash {
    verifierKeys := fun _ => PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup
    iteration := statement.iteration + 1
    z0 := statement.z0
    current := application.step statement.zi advice
    running := fun _ => result
    pc := 1 }

theorem stepHolds_and_wellFormed
    (statement : HyperNovaHistory.Statement) (payload : HyperNovaHistory.Payload application)
    (advice : AppWitness) (proof : Lifecycle.Proof 8)
    (result : Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (accepted : PerApplicationTerminal.Holds application fits commitmentSetup
      statement (.recursive payload))
    (verified : Nifs.PaperNonInteractive.verify
      (ProductionKey.key (PerApplicationFixedPoint.relation application fits) (PerApplicationCanonicalPackage.commitmentKey commitmentSetup))
      (payload.running functionIndex) payload.fresh proof = some result)
    (nonwrap : statement.iteration + 1 < goldilocksModulus) :
    let relation := PerApplicationFixedPoint.relation application fits
    let context := (PerApplicationCanonicalPackage.verifierContextDescriptor fits commitmentSetup).digest4
    let before := input application statement payload advice proof
    let after := output application fits commitmentSetup statement advice result
    StepHoldsFor relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList application before after ∧
    StateEncoding.WellFormed (priorHashPreimage (setup relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList) before) ∧
    StateEncoding.WellFormed (nextHashPreimage (setup relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList) before after) ∧
    before.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList) before)) ∧
    result = after.runningNext functionIndex := by
  let relation := PerApplicationFixedPoint.relation application fits
  let context := (PerApplicationCanonicalPackage.verifierContextDescriptor fits commitmentSetup).digest4
  let before := input application statement payload advice proof
  let after := output application fits commitmentSetup statement advice result
  obtain ⟨valid, canonical, pcValid, positive, publicLink, _running, _fresh⟩ :=
    (PerApplicationTerminal.holds_recursive_iff application fits commitmentSetup statement payload).mp accepted
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
      encHash (stateHash (priorHashPreimage (setup relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList) before)) :=
    publicLink
  have step : StepHoldsFor relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList application before after := by
    refine ⟨rfl, rfl, rfl, Or.inr ⟨pcValid, positive, link, ?_, ?_⟩⟩
    · dsimp only [HyperNova.NonInteractiveMultiFold.Accepts, setup, nifsVerifier,
        before, after, input, output]
      exact Eq.mpr (congrArg (fun index : Fin slotCount =>
        Nifs.PaperNonInteractive.verify (ProductionKey.key relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup))
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
      (priorHashPreimage (setup relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList) before) :=
    ⟨context.toList_length, valid.2.1, valid.2.2⟩
  have nextWidth : after.zNext.length = Stage1.Application.stateWordCount :=
    application.step_length statement.zi advice valid.2.2
  have nextFixed : NightstreamFPrime.Layout.PilotProduction.FixedPreimage
      (nextHashPreimage (setup relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList) before after) :=
    ⟨context.toList_length, valid.2.1, nextWidth⟩
  have nextCanonical := StateEncoding.output_canonical relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
    ((Nifs.PaperNonInteractive.verify_eq_some_iff _ _ _ _ _).mp verified).2.2
  exact ⟨step, ⟨priorFixed, valid.1, pc, canonical functionIndex⟩,
    ⟨nextFixed, nonwrap, rfl, nextCanonical⟩, link, rfl⟩

end NightstreamFPrime.Export.Stage1.HyperNovaStepData
