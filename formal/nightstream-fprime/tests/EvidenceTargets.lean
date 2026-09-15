import NightstreamFPrime.Export.Stage1.PiCCSSourceImagesPreservation
import NightstreamFPrime.Export.Stage1.PiCCSAggregatedImagesPreservation
import NightstreamFPrime.Export.Stage1.PiCCSFirstRound
import NightstreamFPrime.Export.Stage1.PilotDecodedPhase
import NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase
import NightstreamFPrime.Export.Stage1.ActualPiCCSInputs
import NightstreamFPrime.Export.Stage1.ActualPiDECOutput
import NightstreamFPrime.Export.Stage1.ActualContextSecurity
import NightstreamFPrime.Export.Stage1.ActualTerminalSecurity
import NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity
import NightstreamFPrime.Export.Stage1.HyperNovaFalseAcceptance
import NightstreamFPrime.Export.Stage1.PiRLCWitnessHonestResponse
import NightstreamFPrime.Export.Stage1.PiDECStoredSplitHonestWitness
import NightstreamFPrime.Export.Stage1.PiDECCommitmentHonestMessages
import NightstreamFPrime.Export.Stage1.PiDECEvaluationHonestMessages
import NightstreamFPrime.Export.Stage1.PiDECEvaluationFromBlocks
import tests.EvidenceMetadata

/-! Exact assignment targets for pilot/PiCCS and terminal opening extraction.
Terminal matching does not close full history or production conformance.
-/

namespace LeanGraph.Targets

open NightstreamFPrime
open Circuit Layout Layout.Stage1 Spec
open Export.Stage1 Lifecycle Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def PilotAssignment : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : Export.Stage1.PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (Export.Stage1.PerApplicationFixedPoint.logicalWidth application)),
    assignment (ApplicationRetainedGeometry.oneColumn
      (Export.Stage1.PerApplicationFixedPoint.geometry application)) = 1 →
    (Export.Stage1.PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment →
    Lifecycle.Pilot.SpecHolds PilotProduction.interface PilotProduction.witnessOffset
      (PilotSpartan.pullback (Export.Stage1.PilotDecodedEnvironment.env
        (Export.Stage1.DirectApplicationPrefixPlan.pilotOrdinaryGeometry
          (Export.Stage1.PerApplicationFixedPoint.geometry application)) assignment))

theorem pilotAssignment : PilotAssignment :=
  Export.Stage1.PilotDecodedPhase.selectedRowsZero_implies_specHolds

def PiCCSAssignment : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : Export.Stage1.PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : Lifecycle.PaperAlgebra.AjtaiKey
      (logicalWidth := Export.Stage1.PerApplicationFixedPoint.logicalWidth application)
      (publicFits := Export.Stage1.PerApplicationFixedPoint.publicFits application))
    (template : Lifecycle.Proof (Lifecycle.ProductionKey.degreeBound
      (Export.Stage1.PerApplicationFixedPoint.relation application fits)))
    (assignment : Assignment F (Export.Stage1.PerApplicationFixedPoint.logicalWidth application)),
    assignment (ApplicationRetainedGeometry.oneColumn
      (Export.Stage1.PerApplicationFixedPoint.geometry application)) = 1 →
    (Export.Stage1.PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment →
    Lifecycle.PiCCS.v1_1.Formal.PhaseHolds
      (Export.Stage1.PerApplicationFixedPoint.relation application fits) ajtai
      (PiCCSInvocations.parentInterface
        (Export.Stage1.PerApplicationFixedPoint.logicalWidth application)
        (Export.Stage1.PerApplicationFixedPoint.publicFits application))
      PiCCSInputs.phaseOffset
      (Spartan.pullback (Export.Stage1.PiCCSAssignmentSoundness.decodedEnv
        (Export.Stage1.DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
          (Export.Stage1.PerApplicationFixedPoint.geometry application)) assignment)) template

theorem piCCSAssignment : PiCCSAssignment :=
  Export.Stage1.PiCCSDecodedPhase.selectedRowsZero_implies_phaseHolds

def PiCCSPublicAssignment : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (template : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits)))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest),
    digest.length = 4 →
    Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest →
    (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment →
    let geometry := PerApplicationFixedPoint.geometry application
    let relation := PerApplicationFixedPoint.relation application fits
    let interface := PiCCSInvocations.parentInterface
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
    let env := Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
    let prior := StateDecoder.preimage (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
      (ActualPreimageFraming.priorState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
    let next := ActualHashSlots.nextPreimage
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
      (ActualPreimageFraming.priorState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
      (ActualPreimageFraming.outputState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
    PiCCS.v1_1.Formal.PhaseHolds relation ajtai interface
        PiCCSInputs.phaseOffset env template ∧
      PiCCS.v1_1.Formal.evalRunning interface PiCCSInputs.phaseOffset env =
        prior.running functionIndex ∧
      (∀ source, (PiCCS.v1_1.Formal.evalFresh interface
          PiCCSInputs.phaseOffset env).publicInputs source =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application)
          (stateHash (publicFits := PerApplicationFixedPoint.publicFits application) prior)) ∧
      digest = stateHash (publicFits := PerApplicationFixedPoint.publicFits application) next

theorem piCCSPublicAssignment : PiCCSPublicAssignment :=
  ActualPiCCSInputs.selectedRowsAndPublic_imply_phaseAndHashes

/-- Arbitrary selected rows and their actual public input imply the full
typed step at the decoded context. Verifier-context binding is separate. -/
def Stage1Assignment : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest),
    digest.length = 4 →
    Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest →
    (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment →
    StepHoldsFor (PerApplicationFixedPoint.relation application fits) ajtai
      (ActualStep.contextKey application assignment) application
      (ActualStep.input application fits assignment
        (ActualStep.decodedFresh application assignment)
        (ActualPiDECMessages.proof application fits assignment))
      (ActualStep.output application assignment digest)

theorem stage1Assignment : Stage1Assignment := by
  intro application fits ajtai assignment digest fixed publicEqual rows
  exact ActualPiDECOutput.selectedRowsAndPublic_imply_step
    application fits ajtai assignment digest publicEqual rows fixed

#audit_axioms stage1Assignment

/-- Actual terminal membership must supply the arbitrary opening and all
row/public premises, plus exact advertised-state matching or a named collision. -/
def Stage1TerminalAssignment : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup application)
    (statement : Spec.HyperNova.Construction2.Paper.TerminalStatement AppState)
    (payload : ActualContextSecurity.TerminalPayload application),
    Lifecycle.Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload) →
    let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
    (StepHoldsFor (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup) application
        (ActualStep.input application fits assignment (ActualStep.decodedFresh application assignment)
          (ActualPiDECMessages.proof application fits assignment))
        (ActualStep.output application assignment
          (stateHash (ActualContextSecurity.terminalPreimage
            application fits commitmentSetup statement payload))) ∧
      ActualContextSecurity.decodedNext application assignment =
        ActualContextSecurity.terminalPreimage application fits commitmentSetup statement payload) ∨
      PiCCSSecurity.StateHashCollision (ActualContextSecurity.decodedNext application assignment)
        (ActualContextSecurity.terminalPreimage application fits commitmentSetup statement payload)

theorem stage1TerminalAssignment : Stage1TerminalAssignment :=
  ActualContextSecurity.terminal_implies_matchingStepOrCollision

#audit_axioms stage1TerminalAssignment

/-- The terminal's actual witnesses must open the exact decoded PiDEC parent.
The first step needs no NIFS extraction; collisions remain named events. -/
def Stage1TerminalParent : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup application)
    (statement : Spec.HyperNova.Construction2.Paper.TerminalStatement AppState)
    (payload : ActualContextSecurity.TerminalPayload application),
    Lifecycle.Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload) →
    let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
    let input := ActualStep.input application fits assignment
      (ActualStep.decodedFresh application assignment)
      (ActualPiDECMessages.proof application fits assignment)
    let relation := PerApplicationFixedPoint.relation application fits
    let ajtai := PerApplicationCanonicalPackage.commitmentKey commitmentSetup
    let key := ProductionKey.key relation ajtai
    input.iteration = 0 ∨
      (0 < input.iteration ∧ ∃ attempt,
        key.piDecAttempt (input.running functionIndex) input.fresh input.nifsProof = some attempt ∧
        Spec.CE.Holds (semantics ajtai) productionGlobalParams attempt.parent
          ((PaperAlgebra.piDecAlgebra ajtai).recomposeAssignment
            (payload.runningWitness functionIndex))) ∨
      PiCCSSecurity.StateHashCollision (ActualContextSecurity.decodedNext application assignment)
        (ActualContextSecurity.terminalPreimage application fits commitmentSetup statement payload)

theorem stage1TerminalParent : Stage1TerminalParent :=
  ActualTerminalSecurity.terminal_implies_parentOrBaseOrCollision

#audit_axioms stage1TerminalParent

#audit_axioms piCCSPublicAssignment

#audit_axioms pilotAssignment
#audit_axioms piCCSAssignment

section HyperNovaSecurity

open scoped BigOperators ENNReal
open Spec.Folding Spec.Folding.Nifs Lifecycle.Nifs
open StrongReduction
open PiRLC.CoordinateForkLaw (Challenge)
open HyperNovaHistory (Statement Envelope)
open HyperNovaVisitedLaw (Visit goodActive visitedLaw guardedDraw)
open HyperNovaGuardedSourceLaw (inputs realLaw guardedPrefix)
open Poseidon2HashChainV1Setup (productionAjtaiKey)
open PiDECInputCheck (relation)

/-- Exact final linear history criterion for ordinary state and tape types.
All source, FS, depth, invertibility and primitive-clock premises are stated
here. This criterion does not assert hardness or an efficient FS translation. -/
def HyperNovaLinearSecurity : Prop :=
  ∀ (State Tape : Type)
  (tapes : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → PMF Tape)
  (rawCall : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State →
      PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) NifsExtractionProvider.rlc)
  (checkClock : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.CheckClock)
  (storageClock : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.StorageClock)
  (parentClock : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.ParentClock)
  (storageBound : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → Nat)
  (storageBounded : ∀ visit coins output state assignments,
    storageClock visit coins output state assignments ≤ storageBound visit coins output state)
  (baseSummable : ∀ visit coins output state vector, Summable fun tape =>
    (tapes visit coins output state tape).toReal *
      (PaperWeakOracle.baseWork NifsExtractionProvider.rlc
        (NifsExtractionProvider.suffixProgram (NifsExtractionProvider.batchAt inputs visit coins output)
          (checkClock visit coins output state) (storageClock visit coins output state))
        (rawCall visit coins output state) vector tape : ℝ))
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (_depthBound : ∀ input ∈ initial.support, input.1.iteration ≤ depth)
    (originalFirstPhase : Visit → InteractivePrefix.Prover State productionShape 9)
    (abortTape : Tape) (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (queries : Fin depth → Nat)
    (scalarSubClock : RingF → RingF → Nat) (inverseAdapterClock : RingF → Nat)
    (assignmentSubClock : PiRLCExtractionPrimitives.Assignment → PiRLCExtractionPrimitives.Assignment → Nat)
    (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment → Nat)
    (sourceCheckClock : Visit → PiCCSStoredSourceProbability.CheckClock)
    (accessClock : Visit → PiCCSStoredSourceProbability.AccessClock)
    (_lowNorm : Phi81StrongSet.LowNormInvertibility)
    (bounds : PiRLC.PaperForkExtractionWork.PrimitiveBounds)
    (_bounded : PiRLC.PaperForkExtractionWork.Bounded
      (PaperExtractionAlgebra.extractionAlgebra productionAjtaiKey).ring
      (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock) bounds),
    let continuation := NifsProviderLaw.continuation inputs tapes rawCall checkClock storageClock parentClock
      storageBound storageBounded baseSummable
    let program := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let source := HyperNovaGuardedSourceLaw.source originalFirstPhase continuation program
    let visits := fun j : Fin depth => visitedLaw source initial j.val
    let running := fun visit => PiCCSInputCheck.running (inputs visit)
    let fresh := fun visit => PiCCSInputCheck.fresh (inputs visit)
    let firstPhase := guardedPrefix originalFirstPhase
    let checked := InteractiveComposition.firstPhase firstPhase (SupportedExtraction.publicCheck running)
    let contexts := fun j : Fin depth => FiatShamirTransfer.contextLaw relation (realLaw (visits j))
    let provider := fun j : Fin depth =>
      NifsProviderLaw.supportedProvider inputs tapes rawCall checkClock storageClock parentClock
        storageBound storageBounded baseSummable (contexts j) checked
    let extended := fun j : Fin depth =>
      SupportedContinuation.extension relation productionAjtaiKey running fresh (contexts j) checked
        abortTape (provider j)
    (∀ j : Fin depth,
      FiatShamirTransfer.FiatShamirModel relation productionAjtaiKey running fresh
        (realLaw (visits j)) firstPhase abortTape (provider j) g deltaFS (queries j)) →
    (initial.toOuterMeasure {input |
      PerApplicationTerminal.Holds Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Setup.productionSetup input.1 input.2}).toReal ≤
      ((HyperNovaHistoryLaw.law source initial).toOuterMeasure
        {sample | HyperNovaHistoryProbability.AdviceReturned sample}).toReal +
        ∑ j : Fin depth,
          (((visits j).toOuterMeasure
              {visit | HyperNovaFirstFailure.MarkedHashCollision visit}).toReal +
            (((visits j).toOuterMeasure {visit | goodActive visit}).toReal -
              g (queries j) ((visits j).toOuterMeasure {visit | goodActive visit}).toReal +
              deltaFS (queries j) + InteractiveComposition.weakLoss relation productionAjtaiKey +
              IndependentExecution.testError productionShape 9 +
              AdaptiveBindingProbability.successProbability relation productionAjtaiKey program running fresh
                firstPhase (SupportedExtraction.publicCheck running) (extended j)
                (fun visit => PiCCSStoredSourceProbability.sourceProgram (inputs visit)
                  (sourceCheckClock visit) (accessClock visit)) (contexts j) * PaperProfile.arity.total))

/-- The final selected history theorem discharges the literal registered
probability criterion, including its exact operational source and events. -/
theorem hyperNovaLinearSecurity : HyperNovaLinearSecurity :=
  @HyperNovaVisitedSecurity.history_probability_linear_bound

#audit_axioms hyperNovaLinearSecurity

/-- The selected terminal false-acceptance event and its exact symbolic loss. -/
def HyperNovaTerminalFalseAcceptance : Prop :=
  ∀ (State Tape : Type)
  (tapes : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → PMF Tape)
  (rawCall : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State →
      PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) NifsExtractionProvider.rlc)
  (checkClock : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.CheckClock)
  (storageClock : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.StorageClock)
  (parentClock : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.ParentClock)
  (storageBound : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → Nat)
  (storageBounded : ∀ visit coins output state assignments,
    storageClock visit coins output state assignments ≤ storageBound visit coins output state)
  (baseSummable : ∀ visit coins output state vector, Summable fun tape =>
    (tapes visit coins output state tape).toReal *
      (PaperWeakOracle.baseWork NifsExtractionProvider.rlc
        (NifsExtractionProvider.suffixProgram (NifsExtractionProvider.batchAt inputs visit coins output)
          (checkClock visit coins output state) (storageClock visit coins output state))
        (rawCall visit coins output state) vector tape : ℝ))
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (_depthBound : ∀ input ∈ initial.support, input.1.iteration ≤ depth)
    (originalFirstPhase : Visit → InteractivePrefix.Prover State productionShape 9)
    (abortTape : Tape) (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (queries : Fin depth → Nat)
    (scalarSubClock : RingF → RingF → Nat) (inverseAdapterClock : RingF → Nat)
    (assignmentSubClock : PiRLCExtractionPrimitives.Assignment → PiRLCExtractionPrimitives.Assignment → Nat)
    (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment → Nat)
    (sourceCheckClock : Visit → PiCCSStoredSourceProbability.CheckClock)
    (accessClock : Visit → PiCCSStoredSourceProbability.AccessClock)
    (_lowNorm : Phi81StrongSet.LowNormInvertibility)
    (bounds : PiRLC.PaperForkExtractionWork.PrimitiveBounds)
    (_bounded : PiRLC.PaperForkExtractionWork.Bounded
      (PaperExtractionAlgebra.extractionAlgebra productionAjtaiKey).ring
      (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock) bounds),
    let continuation := NifsProviderLaw.continuation inputs tapes rawCall checkClock storageClock parentClock
      storageBound storageBounded baseSummable
    let program := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let source := HyperNovaGuardedSourceLaw.source originalFirstPhase continuation program
    let visits := fun j : Fin depth => visitedLaw source initial j.val
    let running := fun visit => PiCCSInputCheck.running (inputs visit)
    let fresh := fun visit => PiCCSInputCheck.fresh (inputs visit)
    let firstPhase := guardedPrefix originalFirstPhase
    let checked := InteractiveComposition.firstPhase firstPhase (SupportedExtraction.publicCheck running)
    let contexts := fun j : Fin depth => FiatShamirTransfer.contextLaw relation (realLaw (visits j))
    let provider := fun j : Fin depth =>
      NifsProviderLaw.supportedProvider inputs tapes rawCall checkClock storageClock parentClock
        storageBound storageBounded baseSummable (contexts j) checked
    let extended := fun j : Fin depth =>
      SupportedContinuation.extension relation productionAjtaiKey running fresh (contexts j) checked
        abortTape (provider j)
    (∀ j : Fin depth,
      FiatShamirTransfer.FiatShamirModel relation productionAjtaiKey running fresh
        (realLaw (visits j)) firstPhase abortTape (provider j) g deltaFS (queries j)) →
    (initial.toOuterMeasure {input | HyperNovaFalseAcceptance.FalseAcceptance input}).toReal ≤
      ∑ j : Fin depth,
          (((visits j).toOuterMeasure
              {visit | HyperNovaFirstFailure.MarkedHashCollision visit}).toReal +
            (((visits j).toOuterMeasure {visit | goodActive visit}).toReal -
              g (queries j) ((visits j).toOuterMeasure {visit | goodActive visit}).toReal +
              deltaFS (queries j) + InteractiveComposition.weakLoss relation productionAjtaiKey +
              IndependentExecution.testError productionShape 9 +
              AdaptiveBindingProbability.successProbability relation productionAjtaiKey program running fresh
                firstPhase (SupportedExtraction.publicCheck running) (extended j)
                (fun visit => PiCCSStoredSourceProbability.sourceProgram (inputs visit)
                  (sourceCheckClock visit) (accessClock visit)) (contexts j) * PaperProfile.arity.total))

/-- The original mixed-law event bridge discharges the registered criterion. -/
theorem hyperNovaTerminalFalseAcceptance : HyperNovaTerminalFalseAcceptance :=
  @HyperNovaFalseAcceptance.probability_linear_bound

#audit_axioms hyperNovaTerminalFalseAcceptance

end HyperNovaSecurity

/-- The prepared executable computes every full-carrier block of the existing
PiRLC assignment combination. Challenges and all 17 source assignments are inputs;
no expected Rust parent or source-validity premise occurs in the statement. -/
def PiRLCWitnessReplay : Prop :=
  ∀ (shape : Phi81Relation.Shape)
    (challenges : Fin PiRLCNonzero.SourceCount → RingF)
    (assignments : Fin PiRLCNonzero.SourceCount → Phi81Relation.Assignment shape)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)),
    ((PiRLCWitnessBlock.preparedWitnessBlockPartials challenges
      (PiRLCWitnessBlock.prepareWitnessActions challenges)
      (fun source => PiRLCPartialTrace.MaterializedRingF.ofRing
        (Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock
          (assignments source) block))).map PiRLCPartialTrace.MaterializedRingF.toRing).getLast? =
      some (Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock
        (Phi81Relation.EvaluationHomomorphism.PiRLCFinite.combineAssignments
          challenges assignments) block)

/-- The audited block equality supplies the literal replay criterion. -/
theorem piRLCWitnessReplay : PiRLCWitnessReplay :=
  fun _ => PiRLCWitnessBlock.preparedWitnessBlockPartials_getLast?

#audit_axioms piRLCWitnessReplay

/-- The executable returns all canonical private digits exactly when the
complete supplied parent is bounded. No expected Rust child is an input. -/
def PiDECWitnessReplay : Prop :=
  ∀ (width : Nat) (parent : Spec.Folding.Nifs.StoredAssignmentArithmetic.StoredAssignment width),
    Phi81Relation.PiDECAlgebra.StoredSplit.splitChecked parent =
      if ∀ column, centeredMagnitude (parent.get column) <
          Phi81Relation.PiDECAlgebra.Radix.combinedBound then
        some (Vector.ofFn fun child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex =>
          Vector.ofFn fun column : Fin width =>
            Phi81Relation.PiDECAlgebra.Radix.splitScalar (parent.get column) child)
      else none

/-- The total kernel equality includes successful and rejected inputs. -/
theorem piDECWitnessReplay : PiDECWitnessReplay :=
  fun _ => Phi81Relation.PiDECAlgebra.StoredSplit.kernel_eq_spec

#audit_axioms piDECWitnessReplay

/-- Every child and every production commitment row, using the same successful
stored split and summing the actual computed block contributions. -/
def PiDECCommitmentReplay : Prop :=
  let shape := PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)
  ∀ (parent : Spec.CE.Instance (PaperAlgebra.Structure shape.logicalWidth)
      (Phi81Relation.PublicInput shape) PaperAlgebra.Point
      PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (parentWitness : Spec.Folding.Nifs.StoredAssignmentArithmetic.StoredAssignment shape.carrierWidth)
    (childWitnesses : Vector (Spec.Folding.Nifs.StoredAssignmentArithmetic.StoredAssignment shape.carrierWidth)
      productionGlobalParams.k),
    Phi81Relation.PiDECAlgebra.StoredSplit.splitChecked parentWitness = some childWitnesses →
    ∀ (child : Fin productionGlobalParams.k)
      (row : Fin Poseidon2HashChainV1Setup.verifierRows),
      (PiDECCommitmentFold.sum fun block =>
        (PiDECCommitmentBlock.contributions Poseidon2HashChainV1Setup.productionSetup
          row block (PiDECCommitmentFold.childBlocks
            (shape := shape) childWitnesses block)).get child).get =
        (Spec.Folding.PiDEC.PaperVerifier.honestMessages
          (PaperAlgebra.piDecAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey)
          parent (Spec.Folding.Nifs.StoredAssignmentArithmetic.view parentWitness) child).commitment row

theorem piDECCommitmentReplay : PiDECCommitmentReplay := by
  dsimp only [PiDECCommitmentReplay]
  intro parent parentWitness childWitnesses success child row
  exact PiDECCommitmentHonestMessages.sum_contributions_honestMessages
    parent parentWitness childWitnesses success child row

#audit_axioms piDECCommitmentReplay

/-- Every complete child evaluation family at the common parent point is
computed from the same successfully split witness. No expected message,
opening or cryptographic premise is supplied to the evaluation kernel. -/
def PiDECChildEvaluationReplay : Prop :=
  let shape := PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)
  ∀ (values : PiDECInputCheck.ParentValues)
    (parentWitness : Spec.Folding.Nifs.StoredAssignmentArithmetic.StoredAssignment shape.carrierWidth)
    (childWitnesses : Vector (Spec.Folding.Nifs.StoredAssignmentArithmetic.StoredAssignment shape.carrierWidth)
      productionGlobalParams.k),
    Phi81Relation.PiDECAlgebra.StoredSplit.splitChecked parentWitness = some childWitnesses →
    ∀ child : Fin productionGlobalParams.k,
      #[(PiDECEvaluationFromBlocks.familyFromBlocks
        (PiDECCommitmentFold.childBlocks (shape := shape) childWitnesses) values.point).get child] =
        (Spec.Folding.PiDEC.PaperVerifier.honestMessages
          (PaperAlgebra.piDecAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey)
          (PiDECInputCheck.parent values)
          (Spec.Folding.Nifs.StoredAssignmentArithmetic.view parentWitness) child).evaluations

theorem piDECChildEvaluationReplay : PiDECChildEvaluationReplay := by
  dsimp only [PiDECChildEvaluationReplay]
  exact PiDECEvaluationFromBlocks.familyFromBlocks_honestMessages

#audit_axioms piDECChildEvaluationReplay

/-- The computable complete first-round polynomial evaluates to the existing
Q completion sum. This kernel target does not assert source-image loading or
an executed comparison with Rust. Those remain separate replay obligations. -/
def PiCCSFirstRoundKernel : Prop :=
  ∀ (data : ProtocolPolynomial.Data K Lifecycle.productionShape)
    (alpha : CubePoint K Lifecycle.productionShape.cubeVariables)
    (gamma value : K),
    (PiCCSFirstRound.firstRound ConcreteCarrier.extensionOps data alpha gamma
      (remaining := 27) (by decide)).evaluate ConcreteCarrier.extensionOps.toOps value =
      Spec.SumCheck.Finite.HypercubeTruth.sumCompletions ConcreteCarrier.extensionOps.toOps
        (ProtocolPolynomial.polynomial ConcreteCarrier.extensionOps data alpha gamma) [value] 27

/-- The complete sum theorem supplies the exact coefficient-kernel target. -/
theorem piCCSFirstRoundKernel : PiCCSFirstRoundKernel :=
  fun data alpha gamma value => PiCCSFirstRound.firstRound_evaluate
    ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws data alpha gamma (by decide) value

#audit_axioms piCCSFirstRoundKernel

/-- The executable image assembly uses the original complete witnesses and
returns the existing source-connected protocol message. -/
def PiCCSOriginalImages : Prop :=
  let relation := PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
    Poseidon2HashChainV1Package.fits
  ∀ (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (vertex : BooleanVertex cubeVariables),
    PiCCSSourceImages.images?
      (PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
      (fun row => (PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)[row]?)
      (PiDECParentSparseRead.prepare ())
      (PiRLC.v1_1.InputBinding.relationSource relation).cubeLayout witness.assignments vertex =
        some (ProtocolPolynomial.vertexMessage
          (((ProductionKey.key relation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
            (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)).sourceProtocolData
              K.embed witness) vertex)

theorem piCCSOriginalImages : PiCCSOriginalImages :=
  PiCCSSourceImages.images_sourceProtocolData

/-- The aggregated constructor, including zeroed unused message fields,
preserves the complete coefficient object with prepared gamma lookup. -/
def PiCCSPreparedPairKernel : Prop :=
  ∀ (input : ProtocolPolynomial.VerifierInput K productionShape) (gamma : K)
    (alphaSelector priorSelector : Spec.SumCheck.Finite.FixedPolynomial K 1)
    (low high : ProtocolPolynomial.OutputMessage K productionShape),
    PiCCSFirstRoundPair.pairPolynomialWithTotals ConcreteCarrier.extensionOps input
      (PiCCSGammaPowers.lookup ConcreteCarrier.extensionOps.toOps gamma
        (PiCCSGammaPowers.prepare ConcreteCarrier.extensionOps.toOps gamma
          (PiCCSFirstRoundPair.powerCount productionShape)))
      alphaSelector priorSelector
      { low with padImage := fun _ => K.zero, matrixImage := fun _ => K.zero }
      { high with padImage := fun _ => K.zero, matrixImage := fun _ => K.zero }
      (FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps (canonicalPadCoordinates productionShape)
        (fun coordinate => K.mul
          (TargetPolynomial.power ConcreteCarrier.extensionOps.toOps gamma coordinate.localGammaExponent)
          (low.padImage coordinate)))
      (FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps (canonicalPadCoordinates productionShape)
        (fun coordinate => K.mul
          (TargetPolynomial.power ConcreteCarrier.extensionOps.toOps gamma coordinate.localGammaExponent)
          (high.padImage coordinate)))
      (FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps (canonicalMatrixCoordinates productionShape)
        (fun coordinate => K.mul
          (TargetPolynomial.power ConcreteCarrier.extensionOps.toOps gamma coordinate.localGammaExponent)
          (low.matrixImage coordinate)))
      (FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps (canonicalMatrixCoordinates productionShape)
        (fun coordinate => K.mul
          (TargetPolynomial.power ConcreteCarrier.extensionOps.toOps gamma coordinate.localGammaExponent)
          (high.matrixImage coordinate))) =
        PiCCSFirstRoundPair.pairPolynomial ConcreteCarrier.extensionOps input gamma
          alphaSelector priorSelector low high

theorem piCCSPreparedPairKernel : PiCCSPreparedPairKernel := by
  intro input gamma alphaSelector priorSelector low high
  have powers := funext (PiCCSGammaPowers.lookup_prepare ConcreteCarrier.extensionOps.toOps
    gamma (PiCCSFirstRoundPair.powerCount productionShape))
  rw [powers]
  rw [PiCCSFirstRoundPair.pairPolynomialWithTotals_congr
    ConcreteCarrier.extensionOps input (TargetPolynomial.power ConcreteCarrier.extensionOps.toOps gamma)
    alphaSelector priorSelector
    { low with padImage := fun _ => K.zero, matrixImage := fun _ => K.zero }
    { high with padImage := fun _ => K.zero, matrixImage := fun _ => K.zero } low high
    _ _ _ _ rfl rfl rfl rfl]
  exact PiCCSFirstRoundPair.pairPolynomialWithTotals_eq
    ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws input _
      alphaSelector priorSelector low high

/-- The optimized source constructor supplies the selected vertex fields
and the complete canonical carried sums without image-success premises. -/
def PiCCSAggregatedEndpoints : Prop :=
  let relation := PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
    Poseidon2HashChainV1Package.fits
  ∀ (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (vertex : BooleanVertex cubeVariables),
    let powers := TargetPolynomial.power ConcreteCarrier.extensionOps.toOps gamma
    let prepared := PiCCSAggregatedImages.prepare (PiDECParentSparseRead.prepare ()) powers
    let message := ProtocolPolynomial.vertexMessage
      (((ProductionKey.key relation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
        (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)).sourceProtocolData
          K.embed witness) vertex
    PiCCSAggregatedImages.endpoint?
      (PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
      (fun row => (PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)[row]?)
      (PiRLC.v1_1.InputBinding.relationSource relation).cubeLayout witness.assignments
      prepared.1 prepared.2 (PiCCSAggregatedImages.combinedBlock powers witness.assignments)
      powers vertex = some (
        { message with padImage := fun _ => K.zero, matrixImage := fun _ => K.zero },
        FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps (canonicalPadCoordinates productionShape)
          (fun coordinate => K.mul (powers coordinate.localGammaExponent) (message.padImage coordinate)),
        FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps (canonicalMatrixCoordinates productionShape)
          (fun coordinate => K.mul (powers coordinate.localGammaExponent) (message.matrixImage coordinate)))

theorem piCCSAggregatedEndpoints : PiCCSAggregatedEndpoints :=
  PiCCSAggregatedImages.endpoint_sourceProtocolData

/-- Reused invocation rows have the same block semantics after the existing
loader succeeds. The runner retains its checked numeric fallback on a miss. -/
def PiCCSStoredInvocation : Prop :=
  ∀ (block : Layout.MatrixProgram.Poseidon.Block) (columns : Nat)
    (index : Fin block.invocationCount)
    (interface : Layout.ProductionRelation.PoseidonSboxPlan.Interface columns),
    PiDECPoseidonNumericBlock.loadInvocation? block columns index = some interface →
    ∀ (read : Fin columns → K) (row : Fin 94) (port : Fin Spec.ProductionRelation.matrixCount),
    some (((PiCCSLinearRows.invocation read interface).get row).get port) =
      (block.row? columns (Fin.encodeProd (index, row)).val).map (fun forms =>
        PiCCSSparseEvaluation.evaluateK
          (match Layout.ProductionRelation.meaningfulPort? port with
            | some meaningful => forms meaningful
            | none => Layout.ProductionRelation.SparseForm.empty) read)

theorem piCCSStoredInvocation : PiCCSStoredInvocation :=
  fun block _ => PiCCSLinearRows.invocation_loaded_value block

/-- Kernel closure combines complete completion-sum semantics, original-source
assembly, aggregated endpoints, and exact prepared coefficients. Executed full-round coverage and
Rust comparison are separate requirements in the same graph record. -/
def PiCCSFirstRoundReplayKernel : Prop :=
  PiCCSFirstRoundKernel ∧ PiCCSOriginalImages ∧ PiCCSPreparedPairKernel ∧
    PiCCSAggregatedEndpoints ∧ PiCCSStoredInvocation

theorem piCCSFirstRoundReplayKernel : PiCCSFirstRoundReplayKernel :=
  ⟨piCCSFirstRoundKernel, piCCSOriginalImages, piCCSPreparedPairKernel,
    piCCSAggregatedEndpoints, piCCSStoredInvocation⟩

#audit_axioms piCCSOriginalImages
#audit_axioms piCCSPreparedPairKernel
#audit_axioms piCCSAggregatedEndpoints
#audit_axioms piCCSStoredInvocation
#audit_axioms piCCSFirstRoundReplayKernel

end LeanGraph.Targets
