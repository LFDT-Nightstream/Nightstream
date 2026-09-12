import NightstreamFPrime.Export.Stage1.HyperNovaStepData
import NightstreamFPrime.Export.Stage1.SelectedAssignmentCompleteness
import NightstreamFPrime.Layout.ProductionRelation.CcsOpening
import NightstreamFPrime.Layout.Stage1.PiDECBaseCompleteness
import NightstreamFPrime.Lifecycle.PilotZeroRunning

/-!
Owns selected accepted-next completeness from actual old openings.
The recursive branch fixes its C prefix before actual sampler success. The
base branch uses canonical dummy advice and retains the default running claims.
Both call the actual selected assignment constructor and compute the fresh
commitment from that same carrier. No security or work law is claimed.

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
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup productionAjtaiKey)

private theorem completeAssignment_eq_extend {program : Program} (raw : RawValues program) :
    raw.completeAssignment = Phi81CarrierLayout.extendAssignment 0 raw.assignment := by
  funext column
  by_cases below : column.val < PerApplicationFixedPoint.logicalWidth program
  · simp only [RawValues.completeAssignment, Phi81CarrierLayout.extendAssignment,
      Phi81CarrierLayout.logicalColumn?, dif_pos below]
  · simp only [RawValues.completeAssignment, Phi81CarrierLayout.extendAssignment,
      Phi81CarrierLayout.logicalColumn?, dif_neg below]

private theorem freshHolds_of_rows
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

private theorem terminal_of_memberships
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
  refine ⟨valid, ?_⟩
  refine ⟨(show InRange slotCount 1 from ⟨Nat.le_refl 1, Nat.le_refl 1⟩),
    positive, ?_, ?_, ?_⟩
  · exact congrArg (encHash (publicFits := PerApplicationFixedPoint.publicFits program)) digest
  · intro slot
    exact runningMember
  · exact freshMember

/-- An accepted prior payload and the actual sampler success construct the
normal local proof and an accepted successor envelope. The same returned
child assignments and actual canonical fresh carrier occur in that envelope.
No accepted local proof, child validity, physical rows, or assignment-correctness
callback is a caller premise. This is deterministic completeness only. -/
theorem recursive_extend_of_sampler_success
    (statement : HyperNovaHistory.Statement) (payload : HyperNovaHistory.Payload)
    (advice : AppWitness)
    (accepted : PerApplicationTerminal.Holds application fits productionSetup
      statement (.recursive payload))
    (adviceWidth : advice.length = Stage1.Poseidon2HashChainV1.messageWordCount)
    (nonwrap : statement.iteration + 1 < goldilocksModulus) :
    let relation := PerApplicationFixedPoint.relation application fits
    let key := ProductionKey.key relation productionAjtaiKey
    ∃ (messages : Fin productionShape.cubeVariables → FixedPolynomial K 9)
      (fullOutput : FullOutputCoordinates.FullOutput K productionShape),
      let coins := FiatShamir.derive key.oracle.transcript
        ({ priorState := key.publicInputState (payload.running functionIndex) payload.fresh
           input := (key.statement (payload.running functionIndex) payload.fresh).verifierInput key.lift } :
          PiCCS.TranscriptReplay.Statement K Transcript.State productionShape)
        { rounds := fun round => (messages round).toMessage }
      ∀ rho : Fin key.arity.total → RingF,
        key.piRlcResponse (key.absorbPiCcsOutput coins.finalState fullOutput) = some rho →
        ∃ (proof : Lifecycle.Proof 9)
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
          (∀ child, CE.Holds (semantics productionAjtaiKey) productionGlobalParams
            (Lifecycle.runningStatement relation result child) (children child)) ∧
          Stage1.Application.witnessValue (ApplicationInputs.interface application)
            (ApplicationInputs.localStart application) (SourceCompiler.sourceEnv raw.base) = advice ∧
          raw.outputDigest = (HyperNovaStepData.output statement advice result).x ∧
          PerApplicationTerminal.Holds application fits productionSetup
            { iteration := statement.iteration + 1
              z0 := statement.z0
              zi := application.step statement.zi advice } (.recursive {
              running := fun _ => result
              runningWitness := fun _ => children
              fresh := {
                commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit
                  productionAjtaiKey raw.completeAssignment
                publicInputs := fun _ => encHash raw.outputDigest }
              freshWitness := raw.completeAssignment
              pc := 1 }) := by
  let relation := PerApplicationFixedPoint.relation application fits
  let key := ProductionKey.key relation productionAjtaiKey
  let context := (PerApplicationCanonicalPackage.verifierContextDescriptor fits productionSetup).digest4
  obtain ⟨messages, fullOutput, continuation⟩ :=
    HyperNovaCompleteness.recursive_nifs_of_sampler_success statement payload accepted
  refine ⟨messages, fullOutput, ?_⟩
  intro coins rho sampled
  obtain ⟨proof, result, children, roundsEq, outputEq, sampleEq, verified, childrenMember⟩ :=
    continuation rho sampled
  let before := HyperNovaStepData.input statement payload advice proof
  let after := HyperNovaStepData.output statement advice result
  obtain ⟨step, priorWellFormed, nextWellFormed, freshPublic, resultEq⟩ :=
    HyperNovaStepData.stepHolds_and_wellFormed statement payload advice proof result
      accepted verified nonwrap
  obtain ⟨raw, rows, bounded, _publicInput, actualAdvice, digest⟩ :=
    SelectedAssignmentCompleteness.complete productionAjtaiKey context before after result
      step priorWellFormed nextWellFormed freshPublic verified (fun _ => resultEq) adviceWidth
  have freshMember := freshHolds_of_rows application fits productionAjtaiKey raw rows bounded
  have priorValid := ((PerApplicationTerminal.holds_recursive_iff
    application fits productionSetup statement payload).mp accepted).1
  let next : HyperNovaHistory.Statement :=
    { iteration := statement.iteration + 1, z0 := statement.z0,
      zi := application.step statement.zi advice }
  have nextValid : Stage1.Terminal.StatementValid next :=
    ⟨nonwrap, priorValid.2.1, Stage1.Poseidon2HashChainV1.step_output_length statement.zi advice⟩
  have nextAccepted := terminal_of_memberships application fits productionSetup next
    result children raw nextValid (Nat.zero_lt_succ statement.iteration)
    digest childrenMember freshMember
  exact ⟨proof, result, children, raw, roundsEq, outputEq, sampleEq, verified,
    childrenMember, actualAdvice, digest, nextAccepted⟩

/-- The accepted initial envelope constructs a first canonical step when
the actual sampler on the existing zero proof succeeds. The dummy verifier
result is used by the physical constructor. The new terminal envelope keeps
the default running claims and their proved zero openings. No opening of the
dummy fresh claim or its verifier-produced children is assumed or claimed. -/
theorem base_extend_of_sampler_success
    (statement : HyperNovaHistory.Statement) (advice : AppWitness)
    (accepted : PerApplicationTerminal.Holds application fits productionSetup statement .bottom)
    (adviceWidth : advice.length = Stage1.Poseidon2HashChainV1.messageWordCount) :
    let relation := PerApplicationFixedPoint.relation application fits
    let key := ProductionKey.key relation productionAjtaiKey
    let context := (PerApplicationCanonicalPackage.verifierContextDescriptor fits productionSetup).digest4
    let prior : HashPreimage
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application) := {
      verifierKeys := fun _ => context.toList
      iteration := statement.iteration
      z0 := statement.z0
      current := statement.zi
      running := fun _ => defaultRunning
      pc := 1 }
    ∀ rho : Fin key.arity.total → RingF,
      key.piRlcChallenges defaultRunning (Nifs.BaseCompleteness.baseFresh prior)
        Nifs.BaseCompleteness.zeroProof = some rho →
      ∃ (dummyResult : Running
          (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
          (publicFits := PerApplicationFixedPoint.publicFits application))
        (raw : RawValues application),
        Nifs.PaperNonInteractive.verify key defaultRunning
          (Nifs.BaseCompleteness.baseFresh prior) Nifs.BaseCompleteness.zeroProof = some dummyResult ∧
        Stage1.Application.witnessValue (ApplicationInputs.interface application)
          (ApplicationInputs.localStart application) (SourceCompiler.sourceEnv raw.base) = advice ∧
        raw.outputDigest = (HyperNovaStepData.output statement advice defaultRunning).x ∧
        PerApplicationTerminal.Holds application fits productionSetup
          { iteration := statement.iteration + 1
            z0 := statement.z0
            zi := application.step statement.zi advice } (.recursive {
            running := fun _ => defaultRunning
            runningWitness := fun _ _ => Phi81Relation.EvaluationHomomorphism.BaseLinear.assignmentZero
            fresh := {
              commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit
                productionAjtaiKey raw.completeAssignment
              publicInputs := fun _ => encHash raw.outputDigest }
            freshWitness := raw.completeAssignment
            pc := 1 }) := by
  intro relation key context prior rho sampled
  obtain ⟨valid, zero, sameInitial⟩ :=
    (PerApplicationTerminal.holds_bottom_iff application fits productionSetup statement).mp accepted
  let seeded : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Lifecycle.Proof 9) slotCount := {
    iteration := statement.iteration
    z0 := statement.z0
    zi := statement.zi
    running := fun _ => defaultRunning
    fresh := Nifs.BaseCompleteness.baseFresh prior
    priorPc := 1
    witness := advice
    nifsProof := Nifs.BaseCompleteness.zeroProof }
  let before := PiDECBaseCompleteness.canonicalInput relation productionAjtaiKey context seeded
  let after := HyperNovaStepData.output statement advice defaultRunning
  have seededStep : StepHoldsFor relation productionAjtaiKey context.toList application seeded after :=
    ⟨rfl, rfl, rfl, Or.inl ⟨zero, sameInitial.symm, rfl⟩⟩
  have step := PiDECBaseCompleteness.canonicalInput_preserves_base
    relation productionAjtaiKey context seeded after seededStep zero
  have priorFixed : PilotProduction.FixedPreimage
      (priorHashPreimage (setup relation productionAjtaiKey context.toList) before) :=
    ⟨context.toList_length, valid.2.1, valid.2.2⟩
  have nextFixed : PilotProduction.FixedPreimage
      (nextHashPreimage (setup relation productionAjtaiKey context.toList) before after) :=
    ⟨context.toList_length, valid.2.1,
      Stage1.Poseidon2HashChainV1.step_output_length statement.zi advice⟩
  have nonwrap : statement.iteration + 1 < goldilocksModulus := by
    rw [zero]
    exact (by decide : 0 + 1 < goldilocksModulus)
  have priorWellFormed : StateEncoding.WellFormed
      (priorHashPreimage (setup relation productionAjtaiKey context.toList) before) :=
    ⟨priorFixed, valid.1, rfl⟩
  have nextWellFormed : StateEncoding.WellFormed
      (nextHashPreimage (setup relation productionAjtaiKey context.toList) before after) :=
    ⟨nextFixed, nonwrap, rfl⟩
  have freshPublic : before.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation productionAjtaiKey context.toList) before)) := by
    rfl
  obtain ⟨dummyResult, verified⟩ := Nifs.BaseCompleteness.zeroProof_verify_of_sampler
    relation productionAjtaiKey prior rho sampled
  have beforeVerified : Nifs.PaperNonInteractive.verify key
      (before.running functionIndex) before.fresh before.nifsProof = some dummyResult := by
    simpa only [before, PiDECBaseCompleteness.canonicalInput, seeded] using verified
  have recursiveResult : 0 < before.iteration → dummyResult = after.runningNext functionIndex := by
    intro positive
    have beforeZero : before.iteration = 0 := zero
    exact False.elim ((Nat.ne_of_gt positive) beforeZero)
  obtain ⟨raw, rows, bounded, _publicInput, actualAdvice, digest⟩ :=
    SelectedAssignmentCompleteness.complete productionAjtaiKey context before after dummyResult
      step priorWellFormed nextWellFormed freshPublic beforeVerified recursiveResult adviceWidth
  have freshMember := freshHolds_of_rows application fits productionAjtaiKey raw rows bounded
  let next : HyperNovaHistory.Statement := {
    iteration := statement.iteration + 1
    z0 := statement.z0
    zi := application.step statement.zi advice }
  have nextValid : Stage1.Terminal.StatementValid next :=
    ⟨nonwrap, valid.2.1, Stage1.Poseidon2HashChainV1.step_output_length statement.zi advice⟩
  have nextAccepted := terminal_of_memberships application fits productionSetup next
    defaultRunning (fun _ => Phi81Relation.EvaluationHomomorphism.BaseLinear.assignmentZero) raw
    nextValid (Nat.zero_lt_succ statement.iteration) digest
    (PilotZeroRunning.defaultRunning_holds relation productionAjtaiKey) freshMember
  exact ⟨dummyResult, raw, verified, actualAdvice, digest, nextAccepted⟩

end NightstreamFPrime.Export.Stage1.HyperNovaAcceptedNext
