import NightstreamFPrime.Export.Stage1.ActualContextSecurity

/-!
Owns the deterministic reverse step of HyperNova Construction 2, Appendix
H.3. An accepted terminal opening identifies the decoded application step
and its predecessor, or supplies the existing state-hash collision.

The recursive case consumes valid openings for the exact decoded source
instances. Their extraction is a separate obligation. The iteration-one
case returns the bottom proof without any source-opening premise.
-/

namespace NightstreamFPrime.Export.Stage1.HyperNovaPredecessor

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open ActualContextSecurity

private theorem step_implies_predecessor
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : KeyDigest) (application : Stage1.Application.Program)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (runningWitness : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : Stage1.Terminal.FreshWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (valid : Stage1.Terminal.StatementValid
      { iteration := input.iteration, z0 := input.z0, zi := input.zi })
    (step : StepHoldsFor relation ajtai context application input output)
    (sources : 0 < input.iteration → Lifecycle.TerminalHolds relation ajtai
      (input.running functionIndex) runningWitness input.fresh freshWitness) :
    Stage1.Terminal.HoldsFor relation ajtai context application
      { iteration := input.iteration, z0 := input.z0, zi := input.zi }
      (if input.iteration = 0 then .bottom else .recursive {
        running := input.running
        runningWitness := fun _ => runningWitness
        fresh := input.fresh
        freshWitness := freshWitness
        pc := input.priorPc }) := by
  change FixedAugmentedTransition (Lifecycle.setup relation ajtai context)
    (Lifecycle.machineFor publicFits application) functionIndex input output at step
  rcases step.2.2.2 with base | recursive
  · rw [if_pos base.1]
    exact (Stage1.Terminal.holdsFor_bottom_iff relation ajtai context application _).mpr
      ⟨valid, base.1, base.2.1.symm⟩
  · rcases recursive with ⟨pcValid, positive, publicLink, _fold, _unchanged⟩
    rw [if_neg (Nat.ne_of_gt positive)]
    apply (Stage1.Terminal.holdsFor_recursive_iff relation ajtai context application _ _).mpr
    refine ⟨valid, pcValid, positive, publicLink, ?_, ?_⟩
    · intro slot
      have selected : slot = functionIndex := by
        apply Fin.ext
        have bound := slot.isLt
        change slot.val < 1 at bound
        change slot.val = 0
        omega
      rw [selected]
      exact (sources positive).1
    · exact (sources positive).2

/-- The exact decoded source openings construct an accepted predecessor.
The four success fields identify its natural counter, initial state,
recovered application transition, and terminal proof. The source-opening
premise is used only when the decoded predecessor counter is positive. -/
theorem terminal_implies_predecessorOrCollision
    (application : Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup application)
    (statement : TerminalStatement AppState) (payload : TerminalPayload application)
    (runningWitness : Stage1.Terminal.RunningWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (freshWitness : Stage1.Terminal.FreshWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (terminal : Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload))
    (sourceHolds :
      let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
      let input := ActualStep.input application fits assignment
        (ActualStep.decodedFresh application assignment)
        (ActualPiDECMessages.proof application fits assignment)
      0 < input.iteration → Lifecycle.TerminalHolds
        (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (input.running functionIndex) runningWitness input.fresh freshWitness) :
    let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
    let input := ActualStep.input application fits assignment
      (ActualStep.decodedFresh application assignment)
      (ActualPiDECMessages.proof application fits assignment)
    (input.iteration + 1 = statement.iteration ∧
      input.z0 = statement.z0 ∧
      statement.zi = application.step input.zi input.witness ∧
      Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
        application
        { iteration := input.iteration, z0 := input.z0, zi := input.zi }
        (if input.iteration = 0 then .bottom else .recursive {
          running := input.running
          runningWitness := fun _ => runningWitness
          fresh := input.fresh
          freshWitness := freshWitness
          pc := input.priorPc })) ∨
      PiCCSSecurity.StateHashCollision (decodedNext application assignment)
        (terminalPreimage application fits commitmentSetup statement payload) := by
  let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
  let input := ActualStep.input application fits assignment
    (ActualStep.decodedFresh application assignment)
    (ActualPiDECMessages.proof application fits assignment)
  let output := ActualStep.output application assignment
    (stateHash (terminalPreimage application fits commitmentSetup statement payload))
  rcases terminal_implies_matchingStepOrCollision application fits commitmentSetup
      statement payload terminal with ⟨step, same⟩ | collision
  · apply Or.inl
    have iteration : input.iteration + 1 = statement.iteration :=
      congrArg (fun preimage => preimage.iteration) same
    have initial : input.z0 = statement.z0 :=
      congrArg (fun preimage => preimage.z0) same
    have current : output.zNext = statement.zi :=
      congrArg (fun preimage => preimage.current) same
    have applicationStep : output.zNext = application.step input.zi input.witness :=
      step.2.1
    have valid : Stage1.Terminal.StatementValid
        { iteration := input.iteration, z0 := input.z0, zi := input.zi } := by
      refine ⟨?_, ?_, ?_⟩
      · exact (StateDecoder.preimage_wellFormed
          (PerApplicationFixedPoint.logicalWidth application)
          (PerApplicationFixedPoint.publicFits application)
          (ActualStep.priorState application assignment)).2.1
      · exact StateDecoder.initialState_length (ActualStep.priorState application assignment)
      · exact StateDecoder.currentState_length (ActualStep.priorState application assignment)
    exact ⟨iteration, initial, current.symm.trans applicationStep,
      step_implies_predecessor (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
        application input output runningWitness freshWitness valid step sourceHolds⟩
  · exact Or.inr collision

/-- Reverse iteration one uses the base branch, recovers its actual
application witness, and accepts the unique bottom predecessor. It requires
no NIFS call, extracted source witness, or source-membership premise. -/
theorem terminal_one_implies_baseOrCollision
    (application : Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup application)
    (statement : TerminalStatement AppState) (payload : TerminalPayload application)
    (first : statement.iteration = 1)
    (terminal : Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload)) :
    let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
    let input := ActualStep.input application fits assignment
      (ActualStep.decodedFresh application assignment)
      (ActualPiDECMessages.proof application fits assignment)
    (Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
        application { iteration := 0, z0 := statement.z0, zi := statement.z0 } .bottom ∧
      statement.zi = application.step statement.z0 input.witness) ∨
      PiCCSSecurity.StateHashCollision (decodedNext application assignment)
        (terminalPreimage application fits commitmentSetup statement payload) := by
  let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
  let input := ActualStep.input application fits assignment
    (ActualStep.decodedFresh application assignment)
    (ActualPiDECMessages.proof application fits assignment)
  let output := ActualStep.output application assignment
    (stateHash (terminalPreimage application fits commitmentSetup statement payload))
  rcases terminal_implies_matchingStepOrCollision application fits commitmentSetup
      statement payload terminal with ⟨step, same⟩ | collision
  · apply Or.inl
    have iteration : input.iteration + 1 = statement.iteration :=
      congrArg (fun preimage => preimage.iteration) same
    have zero : input.iteration = 0 := by omega
    have initial : input.z0 = statement.z0 :=
      congrArg (fun preimage => preimage.z0) same
    have current : output.zNext = statement.zi :=
      congrArg (fun preimage => preimage.current) same
    have applicationStep : output.zNext = application.step input.zi input.witness :=
      step.2.1
    have baseState : input.z0 = input.zi := by
      rcases step.2.2.2 with base | recursive
      · exact base.2.1
      · rcases recursive with ⟨_pcValid, positive, _⟩
        exact False.elim ((Nat.ne_of_gt positive) zero)
    have priorState : input.zi = statement.z0 := baseState.symm.trans initial
    refine ⟨?_, ?_⟩
    · apply (Stage1.Terminal.holdsFor_bottom_iff
        (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
        application _).mpr
      refine ⟨⟨?_, terminal.1.2.1, terminal.1.2.1⟩, rfl, rfl⟩
      have bound := terminal.1.1
      change 0 < goldilocksModulus
      omega
    · exact current.symm.trans (by simpa only [priorState] using applicationStep)
  · exact Or.inr collision

end NightstreamFPrime.Export.Stage1.HyperNovaPredecessor
