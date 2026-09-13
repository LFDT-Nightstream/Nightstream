import NightstreamFPrime.Export.Stage1.PilotDecodedPhase
import NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase
import NightstreamFPrime.Export.Stage1.ActualPiCCSInputs
import NightstreamFPrime.Export.Stage1.ActualPiDECOutput
import NightstreamFPrime.Export.Stage1.ActualContextSecurity
import NightstreamFPrime.Export.Stage1.ActualTerminalSecurity
import NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity
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

end HyperNovaSecurity

end LeanGraph.Targets
