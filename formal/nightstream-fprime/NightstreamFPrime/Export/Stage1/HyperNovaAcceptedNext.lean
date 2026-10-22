import NightstreamFPrime.Export.Stage1.HyperNovaStepData
import NightstreamFPrime.Export.Stage1.SelectedAssignmentCompleteness
import NightstreamFPrime.Layout.ProductionRelation.CcsOpening
import NightstreamFPrime.Layout.Stage1.PiDECBaseCompleteness
import NightstreamFPrime.Lifecycle.PilotZeroRunning

/-!
Owns accepted-next completeness from actual old openings, for every
application that fits the `2^28` profile and every commitment setup.
No security or work law is claimed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaAcceptedNext

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open PerApplicationCanonicalAssignment
open _root_.NightstreamFPrime.Spec.SumCheck.Finite

private theorem completeAssignment_eq_extend {program : Program} (raw : RawValues program) :
    raw.completeAssignment = Phi81CarrierLayout.extendAssignment 0 raw.assignment := by
  funext column
  by_cases below : column.val < PerApplicationFixedPoint.logicalWidth program
  · simp only [RawValues.completeAssignment, Phi81CarrierLayout.extendAssignment,
      Phi81CarrierLayout.logicalColumn?, dif_pos below]
  · simp only [RawValues.completeAssignment, Phi81CarrierLayout.extendAssignment,
      Phi81CarrierLayout.logicalColumn?, dif_neg below]

theorem freshHolds_of_rows
    (program : Program) (fit : PerApplicationFixedPoint.FitsTwoPow28 program)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program))
    (raw : RawValues program)
    (rows : (PerApplicationFixedPoint.structuralPlan program fit).RowsZero raw.assignment)
    (bounded : ∀ column, centeredMagnitude (raw.completeAssignment column) < 2) :
    CCS.Holds (semantics ajtai) productionGlobalParams
      (Lifecycle.freshStatement (PerApplicationFixedPoint.relation program fit)
        { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai raw.completeAssignment
          publicInputs := fun _ => encHash raw.outputDigest }) raw.completeAssignment := by
  have completed := completeAssignment_eq_extend raw
  have logicalBound : ∀ column, centeredMagnitude (raw.assignment column) < 2 := by
    intro column
    have same : raw.completeAssignment (Phi81CarrierLayout.embedLogical column) =
        raw.assignment column :=
      (congrFun completed (Phi81CarrierLayout.embedLogical column)).trans
        (Phi81CarrierLayout.extendAssignment_embedLogical (0 : F) raw.assignment column)
    exact Eq.mp (congrArg (fun value : F => centeredMagnitude value < 2) same)
      (bounded (Phi81CarrierLayout.embedLogical column))
  have publicInput := PerApplicationCanonicalAssignment.projectPublicInput_completeAssignment raw
  rw [completed] at publicInput
  have member := Plan.rowsZero_implies_freshHolds
    (PerApplicationFixedPoint.structuralPlan program fit) fit.carrier ajtai
    raw.assignment (encHash raw.outputDigest) rows logicalBound publicInput
  exact Eq.mpr (congrArg (fun assignment : PaperAlgebra.Assignment
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program) =>
    CCS.Holds (semantics ajtai) productionGlobalParams
      (Lifecycle.freshStatement (PerApplicationFixedPoint.relation program fit)
        { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai assignment
          publicInputs := fun _ => encHash raw.outputDigest }) assignment) completed) member

theorem terminal_of_memberships
    (program : Program) (fit : PerApplicationFixedPoint.FitsTwoPow28 program)
    (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup program)
    (statement : TerminalStatement AppState)
    (running : Running (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program))
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program))
    (raw : RawValues program)
    (valid : Stage1.Terminal.StatementValid statement)
    (positive : 0 < statement.iteration)
    (digest : raw.outputDigest = stateHash {
      verifierKeys := fun _ => PerApplicationCanonicalPackage.verifierContextDigest fit commitmentSetup
      iteration := statement.iteration
      z0 := statement.z0
      current := statement.zi
      running := fun _ => running
      pc := 1 })
    (canonical : ChildrenCanonical running)
    (runningMember : ∀ child,
      CE.Holds (semantics (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)) productionGlobalParams
        (Lifecycle.runningStatement (PerApplicationFixedPoint.relation program fit) running child)
        (children child))
    (freshMember : CCS.Holds
      (semantics (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)) productionGlobalParams
      (Lifecycle.freshStatement (PerApplicationFixedPoint.relation program fit)
        { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit
            (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) raw.completeAssignment
          publicInputs := fun _ => encHash raw.outputDigest }) raw.completeAssignment) :
    PerApplicationTerminal.Holds program fit commitmentSetup statement (.recursive {
      running := fun _ => running
      runningWitness := fun _ => children
      fresh := {
        commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit
          (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) raw.completeAssignment
        publicInputs := fun _ => encHash raw.outputDigest }
      freshWitness := raw.completeAssignment
      pc := 1 }) := by
  apply (PerApplicationTerminal.holds_recursive_iff program fit commitmentSetup statement _).mpr
  refine ⟨valid, fun _ => canonical, ?_⟩
  refine ⟨(show InRange slotCount 1 from ⟨Nat.le_refl 1, Nat.le_refl 1⟩),
    positive, ?_, ?_, ?_⟩
  · exact congrArg (encHash (publicFits := PerApplicationFixedPoint.publicFits program)) digest
  · intro slot
    exact runningMember
  · exact freshMember

variable (application : Program) (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup application)

theorem recursive_extend
    (statement : HyperNovaHistory.Statement) (payload : HyperNovaHistory.Payload application)
    (advice : AppWitness)
    (accepted : PerApplicationTerminal.Holds application fits commitmentSetup
      statement (.recursive payload))
    (adviceWidth : advice.length = application.witnessWordCount)
    (nonwrap : statement.iteration + 1 < goldilocksModulus) :
    let relation := PerApplicationFixedPoint.relation application fits
    let key := ProductionKey.key relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
    ∃ (messages : Fin productionShape.cubeVariables → FixedPolynomial K 8)
      (fullOutput : FullOutputCoordinates.FullOutput K productionShape),
      let coins := FiatShamir.derive key.oracle.transcript
        ({ priorState := key.publicInputState (payload.running functionIndex) payload.fresh
           input := (key.statement (payload.running functionIndex) payload.fresh).verifierInput key.lift } :
          PiCCS.TranscriptReplay.Statement K Transcript.State productionShape)
        { rounds := fun round => (messages round).toMessage }
      ∃ rho : Fin key.arity.total → RingF,
        key.piRlcResponse (key.absorbPiCcsOutput coins.finalState fullOutput) = some rho ∧
        ∃ (proof : Lifecycle.Proof 8)
          (result : Running
            (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
            (publicFits := PerApplicationFixedPoint.publicFits application))
          (children : Stage1.Terminal.RunningWitness
            (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
            (publicFits := PerApplicationFixedPoint.publicFits application))
          (raw : RawValues application),
          proof.piCcsRounds = messages ∧ proof.piCcsOutput = fullOutput ∧
          key.piRlcChallenges (payload.running functionIndex) payload.fresh proof = some rho ∧
          Nifs.PaperNonInteractive.verify key (payload.running functionIndex) payload.fresh proof = some result ∧
          (∀ child, CE.Holds (semantics (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)) productionGlobalParams
            (Lifecycle.runningStatement relation result child) (children child)) ∧
          Stage1.Application.witnessValue (ApplicationInputs.interface application)
            (ApplicationInputs.localStart application) (SourceCompiler.sourceEnv raw.base) = advice ∧
          raw.outputDigest = (HyperNovaStepData.output application fits commitmentSetup statement advice result).x ∧
          PerApplicationTerminal.Holds application fits commitmentSetup
            { iteration := statement.iteration + 1
              z0 := statement.z0
              zi := application.step statement.zi advice } (.recursive {
              running := fun _ => result
              runningWitness := fun _ => children
              fresh := {
                commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit
                  (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) raw.completeAssignment
                publicInputs := fun _ => encHash raw.outputDigest }
              freshWitness := raw.completeAssignment
              pc := 1 }) := by
  let relation := PerApplicationFixedPoint.relation application fits
  let key := ProductionKey.key relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
  let context := (PerApplicationCanonicalPackage.verifierContextDescriptor fits commitmentSetup).digest4
  obtain ⟨messages, fullOutput, continuation⟩ :=
    HyperNovaCompleteness.recursive_nifs application fits commitmentSetup statement payload accepted
  refine ⟨messages, fullOutput, ?_⟩
  intro coins
  obtain ⟨rho, sampled, proof, result, children, roundsEq, outputEq, sampleEq, verified, childrenMember⟩ :=
    continuation
  refine ⟨rho, sampled, ?_⟩
  let before := HyperNovaStepData.input application statement payload advice proof
  let after := HyperNovaStepData.output application fits commitmentSetup statement advice result
  obtain ⟨step, priorWellFormed, nextWellFormed, freshPublic, resultEq⟩ :=
    HyperNovaStepData.stepHolds_and_wellFormed application fits commitmentSetup statement payload advice proof result
      accepted verified nonwrap
  obtain ⟨raw, rows, bounded, _publicInput, actualAdvice, digest⟩ :=
    SelectedAssignmentCompleteness.complete application fits (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context before after result
      step priorWellFormed nextWellFormed freshPublic verified (fun _ => resultEq) adviceWidth
  have freshMember := freshHolds_of_rows application fits (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) raw rows bounded
  have priorValid := ((PerApplicationTerminal.holds_recursive_iff
    application fits commitmentSetup statement payload).mp accepted).1
  let next : HyperNovaHistory.Statement :=
    { iteration := statement.iteration + 1, z0 := statement.z0,
      zi := application.step statement.zi advice }
  have nextValid : Stage1.Terminal.StatementValid next :=
    ⟨nonwrap, priorValid.2.1, application.step_length statement.zi advice priorValid.2.2⟩
  have nextAccepted := terminal_of_memberships application fits commitmentSetup next
    result children raw nextValid (Nat.zero_lt_succ statement.iteration)
    digest nextWellFormed.2.2.2 childrenMember freshMember
  exact ⟨proof, result, children, raw, roundsEq, outputEq, sampleEq, verified,
    childrenMember, actualAdvice, digest, nextAccepted⟩

theorem base_extend
    (statement : HyperNovaHistory.Statement) (advice : AppWitness)
    (accepted : PerApplicationTerminal.Holds application fits commitmentSetup statement .bottom)
    (adviceWidth : advice.length = application.witnessWordCount) :
    let relation := PerApplicationFixedPoint.relation application fits
    let key := ProductionKey.key relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
    let context := (PerApplicationCanonicalPackage.verifierContextDescriptor fits commitmentSetup).digest4
    let prior : HashPreimage
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application) := {
      verifierKeys := fun _ => context.toList
      iteration := statement.iteration
      z0 := statement.z0
      current := statement.zi
      running := fun _ => defaultRunning
      pc := 1 }
      ∃ (dummyResult : Running
          (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
          (publicFits := PerApplicationFixedPoint.publicFits application))
        (raw : RawValues application),
        Nifs.PaperNonInteractive.verify key defaultRunning
          (Nifs.BaseCompleteness.baseFresh prior) Nifs.BaseCompleteness.zeroProof = some dummyResult ∧
        Stage1.Application.witnessValue (ApplicationInputs.interface application)
          (ApplicationInputs.localStart application) (SourceCompiler.sourceEnv raw.base) = advice ∧
        raw.outputDigest = (HyperNovaStepData.output application fits commitmentSetup statement advice defaultRunning).x ∧
        PerApplicationTerminal.Holds application fits commitmentSetup
          { iteration := statement.iteration + 1
            z0 := statement.z0
            zi := application.step statement.zi advice } (.recursive {
            running := fun _ => defaultRunning
            runningWitness := fun _ _ => Phi81Relation.EvaluationHomomorphism.BaseLinear.assignmentZero
            fresh := {
              commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit
                (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) raw.completeAssignment
              publicInputs := fun _ => encHash raw.outputDigest }
            freshWitness := raw.completeAssignment
            pc := 1 }) := by
  intro relation key context prior
  obtain ⟨valid, zero, sameInitial⟩ :=
    (PerApplicationTerminal.holds_bottom_iff application fits commitmentSetup statement).mp accepted
  let seeded : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Lifecycle.Proof 8) slotCount := {
    iteration := statement.iteration
    z0 := statement.z0
    zi := statement.zi
    running := fun _ => defaultRunning
    fresh := Nifs.BaseCompleteness.baseFresh prior
    priorPc := 1
    witness := advice
    nifsProof := Nifs.BaseCompleteness.zeroProof }
  let before := PiDECBaseCompleteness.canonicalInput relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context seeded
  let after := HyperNovaStepData.output application fits commitmentSetup statement advice defaultRunning
  have seededStep : StepHoldsFor relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList application seeded after :=
    ⟨rfl, rfl, rfl, Or.inl ⟨zero, sameInitial.symm, rfl⟩⟩
  have step := PiDECBaseCompleteness.canonicalInput_preserves_base
    relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context seeded after application
    seededStep zero
  have priorFixed : PilotProduction.FixedPreimage
      (priorHashPreimage (setup relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList) before) :=
    ⟨context.toList_length, valid.2.1, valid.2.2⟩
  have nextFixed : PilotProduction.FixedPreimage
      (nextHashPreimage (setup relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList) before after) :=
    ⟨context.toList_length, valid.2.1,
      application.step_length statement.zi advice valid.2.2⟩
  have nonwrap : statement.iteration + 1 < goldilocksModulus := by
    rw [zero]
    exact (by decide : 0 + 1 < goldilocksModulus)
  have priorWellFormed : StateEncoding.WellFormed
      (priorHashPreimage (setup relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList) before) :=
    ⟨priorFixed, valid.1, rfl, StateEncoding.defaultRunning_canonical⟩
  have nextWellFormed : StateEncoding.WellFormed
      (nextHashPreimage (setup relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList) before after) :=
    ⟨nextFixed, nonwrap, rfl, StateEncoding.defaultRunning_canonical⟩
  have freshPublic : before.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context.toList) before)) := by
    rfl
  obtain ⟨dummyResult, verified⟩ := Nifs.BaseCompleteness.zeroProof_verify
    relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) prior
  have beforeVerified : Nifs.PaperNonInteractive.verify key
      (before.running functionIndex) before.fresh before.nifsProof = some dummyResult := by
    simpa only [before, PiDECBaseCompleteness.canonicalInput, seeded] using! verified
  have recursiveResult : 0 < before.iteration → dummyResult = after.runningNext functionIndex := by
    intro positive
    have beforeZero : before.iteration = 0 := zero
    exact False.elim ((Nat.ne_of_gt positive) beforeZero)
  obtain ⟨raw, rows, bounded, _publicInput, actualAdvice, digest⟩ :=
    SelectedAssignmentCompleteness.complete application fits (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) context before after dummyResult
      step priorWellFormed nextWellFormed freshPublic beforeVerified recursiveResult adviceWidth
  have freshMember := freshHolds_of_rows application fits (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) raw rows bounded
  let next : HyperNovaHistory.Statement := {
    iteration := statement.iteration + 1
    z0 := statement.z0
    zi := application.step statement.zi advice }
  have nextValid : Stage1.Terminal.StatementValid next :=
    ⟨nonwrap, valid.2.1, application.step_length statement.zi advice valid.2.2⟩
  have nextAccepted := terminal_of_memberships application fits commitmentSetup next
    defaultRunning (fun _ => Phi81Relation.EvaluationHomomorphism.BaseLinear.assignmentZero) raw
    nextValid (Nat.zero_lt_succ statement.iteration) digest
    StateEncoding.defaultRunning_canonical
    (PilotZeroRunning.defaultRunning_holds relation (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)) freshMember
  exact ⟨dummyResult, raw, verified, actualAdvice, digest, nextAccepted⟩

end NightstreamFPrime.Export.Stage1.HyperNovaAcceptedNext
