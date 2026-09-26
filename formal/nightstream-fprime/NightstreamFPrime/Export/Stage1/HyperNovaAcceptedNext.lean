import NightstreamFPrime.Export.Stage1.HyperNovaStepData
import NightstreamFPrime.Export.Stage1.Wide.SelectedAssignmentCompleteness
import NightstreamFPrime.Export.Stage1.Wide.TerminalSecurity
import NightstreamFPrime.Layout.ProductionRelation.CcsOpening
import NightstreamFPrime.Lifecycle.Nifs.BaseVerifierCompleteness
import NightstreamFPrime.Lifecycle.PilotZeroRunning

/-! Accepted-successor completeness for the selected package. The fresh
commitment opens the constructed carrier; recursive running claims retain
the honest fold's exact child openings. No security or runtime bound is claimed. -/

namespace NightstreamFPrime.Export.Stage1.HyperNovaAcceptedNext

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint
open Spec.HyperNova.Construction2.Paper Layout.Stage1 ProductionRelation
open Wide
open Wide.SetupBinding (application fits productionAjtaiKey)

private theorem freshHolds_of_rows
    (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (env : Env) (message : Fin 4 → F)
    (rows : (FixedPoint.structuralPlan application compiled fits).RowsZero
      (SourceAssignment.assignment application env (ApplicationCompletedAssignment.suffix env message)))
    (bounded : ∀ column, centeredMagnitude
      (CarrierAssignment.values application env (ApplicationCompletedAssignment.suffix env message) column) < 2) :
    let carrier := CarrierAssignment.values application env (ApplicationCompletedAssignment.suffix env message)
    let digest := (SourceAssignment.raw application env (ApplicationCompletedAssignment.suffix env message)).outputDigest
    CCS.Holds (semantics productionAjtaiKey) productionGlobalParams
      (Lifecycle.freshStatement (FixedPoint.relation application compiled fits)
        { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit productionAjtaiKey carrier
          publicInputs := fun _ => encHash digest }) carrier := by
  let suffix := ApplicationCompletedAssignment.suffix env message
  have logicalBound : ∀ column, centeredMagnitude
      (SourceAssignment.assignment application env suffix column) < 2 := by
    intro column
    rw [← CarrierAssignment.logical_value application env suffix column]
    exact bounded (Phi81CarrierLayout.embedLogical column)
  exact Plan.rowsZero_implies_freshHolds
    (FixedPoint.structuralPlan application compiled fits) (FixedPoint.carrierFits application fits)
    productionAjtaiKey (SourceAssignment.assignment application env suffix)
    (encHash (SourceAssignment.raw application env suffix).outputDigest) rows logicalBound
    (CarrierAssignment.publicInput application env suffix)

private theorem terminal_of_memberships
    (target : Wide.Target) (statement : TerminalStatement AppState)
    (running : Running (logicalWidth := target.logicalWidth) (publicFits := target.publicFits))
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := target.logicalWidth) (publicFits := target.publicFits))
    (carrier : PaperAlgebra.Assignment
      (logicalWidth := target.logicalWidth) (publicFits := target.publicFits))
    (digest : Digest)
    (valid : Stage1.Terminal.StatementValid statement) (positive : 0 < statement.iteration)
    (hash : digest = stateHash {
      verifierKeys := fun _ => target.context
      iteration := statement.iteration
      z0 := statement.z0
      current := statement.zi
      running := fun _ => running
      pc := 1 })
    (runningMember : ∀ child, CE.Holds (semantics target.ajtai) productionGlobalParams
      (Lifecycle.runningStatement target.relation running child) (children child))
    (freshMember : CCS.Holds (semantics target.ajtai) productionGlobalParams
      (Lifecycle.freshStatement target.relation
        { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit target.ajtai carrier
          publicInputs := fun _ => encHash digest }) carrier) :
    target.Holds statement (.recursive {
      running := fun _ => running
      runningWitness := fun _ => children
      fresh := {
        commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit target.ajtai carrier
        publicInputs := fun _ => encHash digest }
      freshWitness := carrier
      pc := 1 }) := by
  apply (Stage1.Terminal.holdsFor_recursive_iff target.relation target.ajtai target.context
    target.program statement _).mpr
  refine ⟨valid, ⟨Nat.le_refl 1, Nat.le_refl 1⟩, positive, ?_, ?_, freshMember⟩
  · exact congrArg (encHash (publicFits := target.publicFits)) hash
  · intro slot
    exact runningMember

/-- An accepted selected envelope extends to an accepted successor with the
actual new fresh opening and the fold's exact child openings. Advice width
and counter nonwrap are the same input restrictions as before the sampler switch. -/
theorem recursive_extend
    (compiled : PiRlcWideSampler.RangePlan.Compiled) (parts : AuthorityStream.Parts)
    (statement : TerminalStatement AppState) (payload : (Wide.selected compiled parts).Payload)
    (advice : AppWitness)
    (accepted : (Wide.selected compiled parts).Holds statement (.recursive payload))
    (adviceWidth : advice.length = Stage1.Poseidon2HashChainV1.messageWordCount)
    (nonwrap : statement.iteration + 1 < goldilocksModulus) :
    let target := Wide.selected compiled parts
    ∃ (proof : Lifecycle.Proof 9)
      (result : Running (logicalWidth := target.logicalWidth) (publicFits := target.publicFits))
      (children : Stage1.Terminal.RunningWitness
        (logicalWidth := target.logicalWidth) (publicFits := target.publicFits))
      (env : Env) (message : Fin 4 → F),
      let suffix := ApplicationCompletedAssignment.suffix env message
      let carrier := CarrierAssignment.values application env suffix
      let raw := SourceAssignment.raw application env suffix
      Nifs.PaperNonInteractive.verify (ProductionKey.key target.relation target.ajtai)
        (payload.running functionIndex) payload.fresh proof = some result ∧
      (∀ child, CE.Holds (semantics target.ajtai) productionGlobalParams
        (Lifecycle.runningStatement target.relation result child) (children child)) ∧
      Stage1.Application.witnessValue (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) (SourceCompiler.sourceEnv raw.base) = advice ∧
      target.Holds {
        iteration := statement.iteration + 1
        z0 := statement.z0
        zi := application.step statement.zi advice } (.recursive {
        running := fun _ => result
        runningWitness := fun _ => children
        fresh := {
          commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit target.ajtai carrier
          publicInputs := fun _ => encHash raw.outputDigest }
        freshWitness := carrier
        pc := 1 }) := by
  let target := Wide.selected compiled parts
  let context := SetupBinding.contextDigest (SetupBinding.descriptor parts)
  obtain ⟨proof, result, children, verified, childrenMember⟩ :=
    HyperNovaCompleteness.recursive_nifs target.relation target.ajtai target.context target.program
      statement payload accepted
  let before := HyperNovaStepData.input statement payload advice proof
  let after := HyperNovaStepData.output context application statement advice result
  obtain ⟨step, priorWellFormed, nextWellFormed, freshPublic, resultEq⟩ :=
    HyperNovaStepData.stepHolds_and_wellFormed target.relation target.ajtai context
      statement payload advice proof result accepted verified nonwrap
  obtain ⟨env, message, rows, bounded, _publicInput, actualAdvice, digest⟩ :=
    SelectedAssignmentCompleteness.complete compiled productionAjtaiKey context before after result
      step priorWellFormed nextWellFormed freshPublic verified (fun _ => resultEq) adviceWidth
  have freshMember := freshHolds_of_rows compiled env message rows bounded
  have priorValid := accepted.1
  let next : TerminalStatement AppState := {
    iteration := statement.iteration + 1
    z0 := statement.z0
    zi := application.step statement.zi advice }
  have nextValid : Stage1.Terminal.StatementValid next :=
    ⟨nonwrap, priorValid.2.1, Stage1.Poseidon2HashChainV1.step_output_length statement.zi advice⟩
  have nextAccepted := terminal_of_memberships target next result children
    (CarrierAssignment.values application env (ApplicationCompletedAssignment.suffix env message))
    (SourceAssignment.raw application env (ApplicationCompletedAssignment.suffix env message)).outputDigest
    nextValid (Nat.zero_lt_succ statement.iteration) digest childrenMember freshMember
  exact ⟨proof, result, children, env, message, verified, childrenMember, actualAdvice, nextAccepted⟩

/-- The empty accepted envelope has a first accepted application step.
The auxiliary zero NIFS proof constructs rows; the new envelope keeps the
default running claims with their zero openings and the actual fresh carrier. -/
theorem base_extend
    (compiled : PiRlcWideSampler.RangePlan.Compiled) (parts : AuthorityStream.Parts)
    (statement : TerminalStatement AppState) (advice : AppWitness)
    (accepted : (Wide.selected compiled parts).Holds statement .bottom)
    (adviceWidth : advice.length = Stage1.Poseidon2HashChainV1.messageWordCount) :
    let target := Wide.selected compiled parts
    ∃ (env : Env) (message : Fin 4 → F),
      let suffix := ApplicationCompletedAssignment.suffix env message
      let carrier := CarrierAssignment.values application env suffix
      let raw := SourceAssignment.raw application env suffix
      Stage1.Application.witnessValue (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) (SourceCompiler.sourceEnv raw.base) = advice ∧
      target.Holds {
        iteration := statement.iteration + 1
        z0 := statement.z0
        zi := application.step statement.zi advice } (.recursive {
        running := fun _ => defaultRunning
        runningWitness := fun _ _ => Phi81Relation.EvaluationHomomorphism.BaseLinear.assignmentZero
        fresh := {
          commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit target.ajtai carrier
          publicInputs := fun _ => encHash raw.outputDigest }
        freshWitness := carrier
        pc := 1 }) := by
  let target := Wide.selected compiled parts
  let relation := target.relation
  let context := SetupBinding.contextDigest (SetupBinding.descriptor parts)
  obtain ⟨valid, zero, sameInitial⟩ :=
    (Stage1.Terminal.holdsFor_bottom_iff target.relation target.ajtai target.context
      target.program statement).mp accepted
  let prior : HashPreimage
      (logicalWidth := target.logicalWidth) (publicFits := target.publicFits) := {
    verifierKeys := fun _ => context.toList
    iteration := statement.iteration
    z0 := statement.z0
    current := statement.zi
    running := fun _ => defaultRunning
    pc := 1 }
  let before : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := target.logicalWidth) (publicFits := target.publicFits))
      (Fresh (logicalWidth := target.logicalWidth) (publicFits := target.publicFits))
      (Lifecycle.Proof 9) slotCount := {
    iteration := statement.iteration
    z0 := statement.z0
    zi := statement.zi
    running := fun _ => defaultRunning
    fresh := Nifs.BaseCompleteness.baseFresh prior
    priorPc := 1
    witness := advice
    nifsProof := Nifs.BaseCompleteness.zeroProof }
  let after := HyperNovaStepData.output context application statement advice
    (defaultRunning (logicalWidth := target.logicalWidth) (publicFits := target.publicFits))
  have step : StepHoldsFor relation target.ajtai context.toList application before after :=
    ⟨rfl, rfl, rfl, Or.inl ⟨zero, sameInitial.symm, rfl⟩⟩
  have priorFixed : PilotProduction.FixedPreimage
      (priorHashPreimage (setup relation target.ajtai context.toList) before) :=
    ⟨context.toList_length, valid.2.1, valid.2.2⟩
  have nextFixed : PilotProduction.FixedPreimage
      (nextHashPreimage (setup relation target.ajtai context.toList) before after) :=
    ⟨context.toList_length, valid.2.1,
      Stage1.Poseidon2HashChainV1.step_output_length statement.zi advice⟩
  have nonwrap : statement.iteration + 1 < goldilocksModulus := by
    rw [zero]
    decide
  have priorWellFormed : StateEncoding.WellFormed
      (priorHashPreimage (setup relation target.ajtai context.toList) before) :=
    ⟨priorFixed, valid.1, rfl⟩
  have nextWellFormed : StateEncoding.WellFormed
      (nextHashPreimage (setup relation target.ajtai context.toList) before after) :=
    ⟨nextFixed, nonwrap, rfl⟩
  have freshPublic : before.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation target.ajtai context.toList) before)) := rfl
  obtain ⟨dummyResult, verified⟩ := Nifs.BaseCompleteness.zeroProof_verify_of_sampler
    relation target.ajtai prior _ rfl
  have recursiveResult : 0 < before.iteration → dummyResult = after.runningNext functionIndex := by
    intro positive
    exact False.elim ((Nat.ne_of_gt positive) zero)
  obtain ⟨env, message, rows, bounded, _publicInput, actualAdvice, digest⟩ :=
    SelectedAssignmentCompleteness.complete compiled productionAjtaiKey context before after dummyResult
      step priorWellFormed nextWellFormed freshPublic verified recursiveResult adviceWidth
  have freshMember := freshHolds_of_rows compiled env message rows bounded
  let next : TerminalStatement AppState := {
    iteration := statement.iteration + 1
    z0 := statement.z0
    zi := application.step statement.zi advice }
  have nextValid : Stage1.Terminal.StatementValid next :=
    ⟨nonwrap, valid.2.1, Stage1.Poseidon2HashChainV1.step_output_length statement.zi advice⟩
  have nextAccepted := terminal_of_memberships target next defaultRunning
    (fun _ => Phi81Relation.EvaluationHomomorphism.BaseLinear.assignmentZero)
    (CarrierAssignment.values application env (ApplicationCompletedAssignment.suffix env message))
    (SourceAssignment.raw application env (ApplicationCompletedAssignment.suffix env message)).outputDigest
    nextValid (Nat.zero_lt_succ statement.iteration) digest
    (PilotZeroRunning.defaultRunning_holds relation target.ajtai) freshMember
  exact ⟨env, message, actualAdvice, nextAccepted⟩

end NightstreamFPrime.Export.Stage1.HyperNovaAcceptedNext
