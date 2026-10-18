import NightstreamFPrime.Export.Stage1.PiCCSPrefixComplete
import NightstreamFPrime.Export.Stage1.PiCCSFirstRoundComposition
import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixSupportedPreservation
import NightstreamFPrime.Export.Stage1.PiCCSOriginalPadPreservation
import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixRange
import NightstreamFPrime.Export.Stage1.PiCCSFreshPrefix
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PrefixFold
import NightstreamFPrime.Export.Stage1.PiCCSPrefixNormBuckets
import NightstreamFPrime.Export.Stage1.PiCCSPrefixCodeFold
import NightstreamFPrime.Export.Stage1.PiCCSFreshComplete
import NightstreamFPrime.Export.Stage1.PiCCSNormComplete
import NightstreamFPrime.Export.Stage1.PiCCSSourceImagesPreservation
import NightstreamFPrime.Export.Stage1.PiCCSAggregatedImagesPreservation
import NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache
import NightstreamFPrime.Export.Stage1.PiCCSCachedSelector
import NightstreamFPrime.Export.Stage1.PiCCSNormCache
import NightstreamFPrime.Export.Stage1.PiCCSFirstRound
import NightstreamFPrime.Export.Stage1.PilotDecodedPhase
import NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase
import NightstreamFPrime.Export.Stage1.ActualPiCCSInputs
import NightstreamFPrime.Export.Stage1.ActualPiDECOutput
import NightstreamFPrime.Export.Stage1.ActualContextSecurity
import NightstreamFPrime.Export.Stage1.ActualTerminalSecurity
import NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity
import NightstreamFPrime.Export.Stage1.HyperNovaFalseAcceptance
import NightstreamFPrime.Export.Stage1.RandomOracleSetup
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
    Lifecycle.PiCCS.v1_2.Formal.PhaseHolds
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
    PiCCS.v1_2.Formal.PhaseHolds relation ajtai interface
        PiCCSInputs.phaseOffset env template ∧
      PiCCS.v1_2.Formal.evalRunning interface PiCCSInputs.phaseOffset env =
        prior.running functionIndex ∧
      (∀ source, (PiCCS.v1_2.Formal.evalFresh interface
          PiCCSInputs.phaseOffset env).publicInputs source =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application)
          (stateHash (publicFits := PerApplicationFixedPoint.publicFits application) prior)) ∧
      digest = stateHash (publicFits := PerApplicationFixedPoint.publicFits application) next

theorem piCCSPublicAssignment : PiCCSPublicAssignment :=
  ActualPiCCSInputs.selectedRowsAndPublic_imply_phaseAndHashes

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
open HyperNovaVisitedSecurity (NifsAdversary NifsExtractor Assumption1 Closed IvcAdversary Stage
  reverseStages)

/-- HyperNova errata Assumption 1, Definition 7, Definition 11 (ii), and
Lemma 17. The random-oracle theorem `RandomOracleSetup.production_knowledge_error_lt`
motivates the value of `error`; it does not prove Assumption 1 for Poseidon2,
and no Lean statement derives `error` from it. -/
def HyperNovaLinearSecurity : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : Export.Stage1.PerApplicationFixedPoint.FitsTwoPow28 application)
    (setup : Export.Stage1.PerApplicationCanonicalPackage.CommitmentSetup application)
    (Admitted : NifsAdversary application fits setup → Prop)
    (StageAdmitted : Stage application fits setup → Prop)
    (Efficient : (adversary : NifsAdversary application fits setup) → NifsExtractor adversary → Prop)
    (error : NifsAdversary application fits setup → ℝ) (assumption : Assumption1 Admitted Efficient error)
    (closed : Closed Admitted StageAdmitted Efficient) (adversary : IvcAdversary application fits setup)
    (admitted : StageAdmitted (Stage.start adversary)) (depth : Nat)
    (_depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth),
    (∀ event : Set (HyperNovaHistory.Statement × HyperNovaHistory.Envelope application),
      adversary.tape.toOuterMeasure {tape | adversary.output tape ∈ event} =
        (reverseStages assumption closed adversary admitted depth).1.reverseLaw.toOuterMeasure
          {sample | (sample.1, sample.2.1) ∈ event}) ∧
    ((reverseStages assumption closed adversary admitted depth).1.reverseLaw.toOuterMeasure
        {sample | HyperNovaHistoryProbability.Accepted application fits setup sample ∧
          ¬ HyperNovaHistoryProbability.AdviceReturned application fits sample}).toReal ≤
      ∑ j : Fin depth,
        (((reverseStages assumption closed adversary admitted j.val).1.tape.toOuterMeasure
            {tape | HyperNovaFirstFailure.MarkedHashCollision application fits setup
              ((reverseStages assumption closed adversary admitted j.val).1.visit tape)}).toReal +
          error (reverseStages assumption closed adversary admitted j.val).1.nifs)

theorem hyperNovaLinearSecurity : HyperNovaLinearSecurity :=
  fun _ _ _ _ _ _ _ assumption closed adversary admitted depth depthBound =>
    ⟨HyperNovaVisitedSecurity.reverse_input_event assumption closed adversary admitted depth,
      HyperNovaVisitedSecurity.history_failure_le assumption closed adversary admitted depth
        depthBound⟩

#audit_axioms hyperNovaLinearSecurity

/-- HyperNova errata Assumption 1 and Lemma 17. -/
def HyperNovaTerminalFalseAcceptance : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : Export.Stage1.PerApplicationFixedPoint.FitsTwoPow28 application)
    (setup : Export.Stage1.PerApplicationCanonicalPackage.CommitmentSetup application)
    (Admitted : NifsAdversary application fits setup → Prop)
    (StageAdmitted : Stage application fits setup → Prop)
    (Efficient : (adversary : NifsAdversary application fits setup) → NifsExtractor adversary → Prop)
    (error : NifsAdversary application fits setup → ℝ) (assumption : Assumption1 Admitted Efficient error)
    (closed : Closed Admitted StageAdmitted Efficient) (adversary : IvcAdversary application fits setup)
    (admitted : StageAdmitted (Stage.start adversary)) (depth : Nat)
    (_depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth),
    (adversary.tape.toOuterMeasure
        {tape | HyperNovaFalseAcceptance.FalseAcceptance application fits setup (adversary.output tape)}).toReal ≤
      ∑ j : Fin depth,
        (((reverseStages assumption closed adversary admitted j.val).1.tape.toOuterMeasure
            {tape | HyperNovaFirstFailure.MarkedHashCollision application fits setup
              ((reverseStages assumption closed adversary admitted j.val).1.visit tape)}).toReal +
          error (reverseStages assumption closed adversary admitted j.val).1.nifs)

theorem hyperNovaTerminalFalseAcceptance : HyperNovaTerminalFalseAcceptance :=
  fun application fits setup _ _ _ _ assumption closed adversary admitted depth depthBound =>
    HyperNovaFalseAcceptance.probability_bound application fits setup assumption closed adversary
      admitted depth depthBound

#audit_axioms hyperNovaTerminalFalseAcceptance

end HyperNovaSecurity

section RomKnowledge

open scoped BigOperators
open Lifecycle.RandomOracleTest (Point Answer)
open Lifecycle.RandomOracleExtraction (Claim Succeeds)
open Export.Stage1.RandomOracleSetup (SetupIndex setupKey extraction hashCollisions msisAdvantage)

attribute [local instance low] Classical.propDecidable

/-- SHAKE128 and the Fiat–Shamir hash are random oracles in this statement;
MSIS and state-hash hardness are not assumed. -/
def RomKnowledgeSoundness : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : Export.Stage1.PerApplicationFixedPoint.FitsTwoPow28 application) {Output : Type}
    (adversary : (SetupIndex (Export.Stage1.PerApplicationFixedPoint.logicalWidth application)
        (Export.Stage1.PerApplicationFixedPoint.publicFits application) →
        Spec.AjtaiSetupV1.Programming.Chunk) →
      Spec.RandomOracle.OracleComp (Point (Export.Stage1.PerApplicationFixedPoint.logicalWidth application)
        (Export.Stage1.PerApplicationFixedPoint.publicFits application)
        (Lifecycle.ProductionKey.degreeBound
          (Export.Stage1.PerApplicationFixedPoint.relation application fits))) Answer Output)
    (claim : Output → Claim (Export.Stage1.PerApplicationFixedPoint.relation application fits))
    (prior : Output → Lifecycle.HashPreimage
      (logicalWidth := Export.Stage1.PerApplicationFixedPoint.logicalWidth application)
      (publicFits := Export.Stage1.PerApplicationFixedPoint.publicFits application))
    (contextDigest : Lifecycle.KeyDigest) (queries : Nat),
    (∀ chunks, (adversary chunks).QueryBound queries) →
    let relation := Export.Stage1.PerApplicationFixedPoint.relation application fits
    𝔼 chunks : SetupIndex (Export.Stage1.PerApplicationFixedPoint.logicalWidth application)
        (Export.Stage1.PerApplicationFixedPoint.publicFits application) →
        Spec.AjtaiSetupV1.Programming.Chunk, 𝔼 oracle,
        (if Succeeds relation (setupKey chunks) (adversary chunks)
            (Export.Stage1.RandomOracleLink.linkedClaim relation claim prior contextDigest)
            oracle then (1 : ℝ) else 0) <
      𝔼 chunks, (extraction relation adversary claim prior contextDigest chunks +
          hashCollisions relation adversary claim prior contextDigest chunks) +
        Lifecycle.RandomOracleKnowledge.statisticalError queries +
        msisAdvantage relation adversary claim prior contextDigest + 1 / 2 ^ 190

theorem romKnowledgeSoundness : RomKnowledgeSoundness :=
  fun application fits _ adversary claim prior contextDigest _ bounded =>
    Export.Stage1.RandomOracleSetup.production_knowledge_error_lt application fits adversary claim
      prior contextDigest bounded

#audit_axioms romKnowledgeSoundness

end RomKnowledge

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

theorem piRLCWitnessReplay : PiRLCWitnessReplay :=
  fun _ => PiRLCWitnessBlock.preparedWitnessBlockPartials_getLast?

#audit_axioms piRLCWitnessReplay

def PiDECWitnessReplay : Prop :=
  ∀ (width : Nat) (parent : Spec.Folding.Nifs.StoredAssignmentArithmetic.StoredAssignment width),
    Phi81Relation.PiDECAlgebra.StoredSplit.splitChecked parent =
      if ∀ column, centeredMagnitude (parent.get column) <
          Phi81Relation.PiDECAlgebra.Radix.combinedBound then
        some (Vector.ofFn fun child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex =>
          Vector.ofFn fun column : Fin width =>
            Phi81Relation.PiDECAlgebra.Radix.splitScalar (parent.get column) child)
      else none

theorem piDECWitnessReplay : PiDECWitnessReplay :=
  fun _ => Phi81Relation.PiDECAlgebra.StoredSplit.kernel_eq_spec

#audit_axioms piDECWitnessReplay

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

def PiCCSFirstRoundKernel : Prop :=
  ∀ (data : ProtocolPolynomial.Data K Lifecycle.productionShape)
    (alpha : CubePoint K Lifecycle.productionShape.cubeVariables)
    (gamma value : K),
    (PiCCSFirstRound.firstRound ConcreteCarrier.extensionOps data alpha gamma
      (remaining := 27) (by decide)).evaluate ConcreteCarrier.extensionOps.toOps value =
      Spec.SumCheck.Finite.HypercubeTruth.sumCompletions ConcreteCarrier.extensionOps.toOps
        (ProtocolPolynomial.polynomial ConcreteCarrier.extensionOps data alpha gamma) [value] 27

theorem piCCSFirstRoundKernel : PiCCSFirstRoundKernel :=
  fun data alpha gamma value => PiCCSFirstRound.firstRound_evaluate
    ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws data alpha gamma (by decide) value

#audit_axioms piCCSFirstRoundKernel

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
      (PiRLC.v1_2.InputBinding.relationSource relation).cubeLayout witness.assignments vertex =
        some (ProtocolPolynomial.vertexMessage
          (((ProductionKey.key relation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
            (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)).sourceProtocolData
              K.embed witness) vertex)

theorem piCCSOriginalImages : PiCCSOriginalImages :=
  PiCCSSourceImages.images_sourceProtocolData

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
      (PiRLC.v1_2.InputBinding.relationSource relation).cubeLayout witness.assignments
      prepared.1 prepared.2 (PiCCSAggregatedImages.combinedBlock powers witness.assignments)
      powers vertex = some (
        { message with padImage := fun _ => K.zero, matrixImage := fun _ => K.zero },
        FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps (canonicalPadCoordinates productionShape)
          (fun coordinate => K.mul (powers coordinate.localGammaExponent) (message.padImage coordinate)),
        FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps (canonicalMatrixCoordinates productionShape)
          (fun coordinate => K.mul (powers coordinate.localGammaExponent) (message.matrixImage coordinate)))

theorem piCCSAggregatedEndpoints : PiCCSAggregatedEndpoints :=
  PiCCSAggregatedImages.endpoint_sourceProtocolData

def PiCCSStoredInvocation : Prop :=
  ∀ (block : Layout.MatrixProgram.Poseidon.Block) (columns : Nat)
    (index : Fin block.invocationCount)
    (interface : Layout.ProductionRelation.PoseidonSboxPlan.Interface columns),
    PiDECPoseidonNumericBlock.loadInvocation? block columns index = some interface →
    ∀ (basis : PiRLCPartialTrace.FixedArray (Vector K ringDegree) ringDegree)
      (blocks : Nat → Vector K ringDegree) (row : Fin 150) (port : Fin Spec.ProductionRelation.matrixCount),
    some (((PiCCSCarriedReadCache.invocation basis blocks interface).get row).get port) =
      (block.row? columns (Fin.encodeProd (index, row)).val).map (fun forms =>
        PiCCSSparseEvaluation.evaluateK
          (match Layout.ProductionRelation.meaningfulPort? port with
            | some meaningful => forms meaningful
            | none => Layout.ProductionRelation.SparseForm.empty) (PiCCSCarriedRead.read basis blocks))

theorem piCCSStoredInvocation : PiCCSStoredInvocation := by
  intro block columns index interface loaded basis blocks row port
  rw [PiCCSCarriedReadCache.invocation_eq]
  exact PiCCSLinearRows.invocation_loaded_value block index interface loaded
    (PiCCSCarriedRead.read basis blocks) row port

def PiCCSCachedPairKernel : Prop :=
  ∀ (input : ProtocolPolynomial.VerifierInput K productionShape) (powers : Nat → K)
    (alpha : CubePoint K productionShape.cubeVariables) (suffix : BooleanVertex 27)
    (low high : ProtocolPolynomial.OutputMessage K productionShape)
    (padLow padHigh matrixLow matrixHigh : K),
    PiCCSFirstRoundPair.pairPolynomialWithNorm ConcreteCarrier.extensionOps input powers
      (PiCCSCachedSelector.equalitySelector ConcreteCarrier.extensionOps suffix alpha
        (PiCCSTensorWeights.prepare ConcreteCarrier.extensionOps alpha.coordinates.tail))
      (PiCCSCachedSelector.equalitySelector ConcreteCarrier.extensionOps suffix input.priorPoint
        (PiCCSTensorWeights.prepare ConcreteCarrier.extensionOps input.priorPoint.coordinates.tail))
      low high padLow padHigh matrixLow matrixHigh
      (PiCCSNormCache.sourceNorm (PiCCSNormCache.prepare powers) powers low high) =
        PiCCSFirstRoundPair.pairPolynomialWithTotals ConcreteCarrier.extensionOps input powers
          (PiCCSFirstRound.equalitySelector ConcreteCarrier.extensionOps suffix alpha)
          (PiCCSFirstRound.equalitySelector ConcreteCarrier.extensionOps suffix input.priorPoint)
          low high padLow padHigh matrixLow matrixHigh

theorem piCCSCachedPairKernel : PiCCSCachedPairKernel := by
  intro input powers alpha suffix low high padLow padHigh matrixLow matrixHigh
  rw [PiCCSCachedSelector.equalitySelector_prepare ConcreteCarrier.extensionOps
      ConcreteCarrier.extensionLaws (by decide) suffix alpha,
    PiCCSCachedSelector.equalitySelector_prepare ConcreteCarrier.extensionOps
      ConcreteCarrier.extensionLaws (by decide) suffix input.priorPoint,
    PiCCSNormCache.sourceNorm_eq,
    PiCCSFirstRoundPair.pairPolynomialWithNorm_eq]

def PiCCSCompleteNormKernel : Prop :=
  ∀
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (gamma : K) (alpha : CubePoint K productionShape.cubeVariables)
    (masks : Array (Array (Nat × Nat)))
    (freshLow freshHigh : Nat → Vector F ProductionRelation.matrixCount)
    (parts : Nat), 0 < parts →
    let powers := PiCCSGammaPowers.lookup ConcreteCarrier.extensionOps.toOps gamma
      (PiCCSGammaPowers.prepare ConcreteCarrier.extensionOps.toOps gamma productionShape.sourceCount)
    let weight := PiCCSTensorWeights.lookup ConcreteCarrier.extensionOps alpha.coordinates.tail
      (PiCCSTensorWeights.prepare ConcreteCarrier.extensionOps alpha.coordinates.tail)
    PiCCSNormContribution.normTerm input (TargetPolynomial.power ConcreteCarrier.extensionOps.toOps gamma) (PiCCSNormContribution.headSelector alpha)
        ((Array.ofFn (fun index : Fin parts =>
          PiCCSNormBuckets.finish powers
            (PiCCSNormScan.range weight masks (PiCCSSourceImages.blockCount * index.val / parts)
              (PiCCSSourceImages.blockCount * (index.val + 1) / parts -
                PiCCSSourceImages.blockCount * index.val / parts)))).foldl
          (Spec.SumCheck.Finite.FixedPolynomial.add ConcreteCarrier.extensionOps.toOps) (Spec.SumCheck.Finite.FixedPolynomial.zero ConcreteCarrier.extensionOps.toOps 3)) =
      PiCCSPolynomialRange.range ConcreteCarrier.extensionOps 0 (2 ^ (productionShape.cubeVariables - 1))
        (PiCCSNormComplete.numericPairNorm input (TargetPolynomial.power ConcreteCarrier.extensionOps.toOps gamma)
          alpha masks freshLow freshHigh)

theorem piCCSCompleteNormKernel : PiCCSCompleteNormKernel :=
  PiCCSNormComplete.prepared_workers_eq_fullPairSum

def PiCCSCompleteFreshKernel : Prop :=
  ∀ (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (layout : UnifiedSources.ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables),
    PiCCSFreshComplete.freshRange? input witness layout powers alpha 0
        PiCCSFreshComplete.activePairs =
      some (PiCCSPolynomialRange.range ConcreteCarrier.extensionOps 0 (2 ^ (cubeVariables - 1))
        (PiCCSFreshComplete.referencePair input witness powers alpha))

theorem piCCSCompleteFreshKernel : PiCCSCompleteFreshKernel :=
  PiCCSFreshComplete.freshRange_eq_fullPairSum

def PiCCSFirstRoundSourceCoefficients : Prop :=
  ∀ (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat))),
    PiCCSFirstRoundComposition.coefficients? input masks =
      some ((PiCCSFirstRound.firstRound ConcreteCarrier.extensionOps
        (((ProductionKey.key
          (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
            Poseidon2HashChainV1Package.fits)
          Poseidon2HashChainV1Setup.productionAjtaiKey).statement
          (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)).sourceProtocolData
            K.embed ⟨PiCCSNormSource.assignments masks⟩)
        (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
        (remaining := 27) (by decide)).coefficients)

theorem piCCSFirstRoundSourceCoefficients : PiCCSFirstRoundSourceCoefficients :=
  PiCCSFirstRoundComposition.coefficients_eq_firstRound

#audit_axioms piCCSFirstRoundSourceCoefficients

def PiCCSFirstRoundReplayKernel : Prop :=
  PiCCSFirstRoundKernel ∧ PiCCSOriginalImages ∧ PiCCSPreparedPairKernel ∧
    PiCCSAggregatedEndpoints ∧ PiCCSStoredInvocation ∧ PiCCSCachedPairKernel ∧
    PiCCSCompleteNormKernel ∧ PiCCSCompleteFreshKernel

theorem piCCSFirstRoundReplayKernel : PiCCSFirstRoundReplayKernel :=
  ⟨piCCSFirstRoundKernel, piCCSOriginalImages, piCCSPreparedPairKernel,
    piCCSAggregatedEndpoints, piCCSStoredInvocation, piCCSCachedPairKernel,
    piCCSCompleteNormKernel, piCCSCompleteFreshKernel⟩

#audit_axioms piCCSOriginalImages
#audit_axioms piCCSPreparedPairKernel
#audit_axioms piCCSAggregatedEndpoints
#audit_axioms piCCSStoredInvocation
#audit_axioms piCCSCompleteNormKernel
#audit_axioms piCCSCompleteFreshKernel
#audit_axioms piCCSCachedPairKernel
#audit_axioms piCCSFirstRoundReplayKernel

def PiCCSNormPrefixKernel : Prop :=
  ∀ (firstChallenge secondChallenge : K) (codes : Nat → Fin 3) (start count : Nat),
    PiCCSPrefixCodeFold.decode
        (PiCCSPrefixCodeFold.pairedTable (PiCCSPrefixNorm.values firstChallenge) secondChallenge)
        (PiCCSPrefixCodeFold.quadCodes codes start count) =
      PrefixFold.foldOne ConcreteCarrier.extensionOps
        (PrefixFold.foldOne ConcreteCarrier.extensionOps
          (Array.ofFn fun index : Fin (4 * count) =>
            K.embed (PiCCSNormCache.signedValue (codes (4 * start + index.val))))
          firstChallenge)
        secondChallenge

theorem piCCSNormPrefixKernel : PiCCSNormPrefixKernel :=
  PiCCSPrefixCodeFold.decode_quadCodes_twoFolds

#audit_axioms piCCSNormPrefixKernel

def PiCCSPrefixNormAccumulation : Prop :=
  (∀ (codeCount : Nat) (values : Vector K codeCount)
      (buckets : Vector (Vector K codeCount) codeCount)
      (low high : Nat → Fin codeCount) (weight : Nat → K) (start count : Nat),
    PiCCSPrefixNormBuckets.finish values
        (PiCCSPrefixNormBuckets.accumulate buckets low high weight start count) =
      Spec.SumCheck.Finite.FixedPolynomial.add ConcreteCarrier.extensionOps.toOps
        (PiCCSPrefixNormBuckets.finish values buckets)
        (PiCCSPolynomialRange.range ConcreteCarrier.extensionOps start count (fun index =>
          Spec.SumCheck.Finite.FixedPolynomial.scale ConcreteCarrier.extensionOps.toOps
            (weight index) (PiCCSFirstRoundPair.normPair ConcreteCarrier.extensionOps
              (values.get (low index)) (values.get (high index)))))) ∧
  (∀ (codeCount : Nat) (buckets : Vector (Vector K codeCount) codeCount)
      (low high : Nat → Fin codeCount) (weight : Nat → K)
      (start leftCount rightCount : Nat),
    PiCCSPrefixNormBuckets.accumulate buckets low high weight start (leftCount + rightCount) =
      PiCCSPrefixNormBuckets.accumulate
        (PiCCSPrefixNormBuckets.accumulate buckets low high weight start leftCount)
        low high weight (start + leftCount) rightCount)

theorem piCCSPrefixNormAccumulation : PiCCSPrefixNormAccumulation :=
  ⟨@PiCCSPrefixNormBuckets.finish_accumulate_eq_add_range,
    @PiCCSPrefixNormBuckets.accumulate_append⟩

#audit_axioms piCCSPrefixNormAccumulation

def PiCCSRetainedPrefixKernel : Prop :=
  (∀ (values : Array K) (challenges : List K) (remaining : Nat)
      (suffix : CubePoint K remaining),
    values.size ≤ 2 ^ (remaining + challenges.length) →
    (PrefixFold.zeroExtend ConcreteCarrier.extensionOps remaining
        (PrefixFold.foldPrefix ConcreteCarrier.extensionOps values challenges)).evaluate
          ConcreteCarrier.extensionOps suffix =
      (PrefixFold.zeroExtend ConcreteCarrier.extensionOps
        (remaining + challenges.length) values).evaluate ConcreteCarrier.extensionOps
          ⟨challenges ++ suffix.coordinates, by simp [suffix.dimension, Nat.add_comm]⟩) ∧
  (∀ (matrixCount : Nat) (rows : Array (Vector K matrixCount))
      (challenge : K) (port : Fin matrixCount),
    PiCCSFreshPrefix.portValues (PiCCSFreshPrefix.foldRows rows challenge) port =
      PrefixFold.foldOne ConcreteCarrier.extensionOps
        (PiCCSFreshPrefix.portValues rows port) challenge)

theorem piCCSRetainedPrefixKernel : PiCCSRetainedPrefixKernel := by
  constructor
  · intro values challenges remaining suffix fits
    exact PrefixFold.foldPrefix_evaluate ConcreteCarrier.extensionOps
      ConcreteCarrier.extensionLaws values challenges suffix fits
  · intro matrixCount rows challenge port
    exact PiCCSFreshPrefix.portValues_foldRows rows challenge port

#audit_axioms piCCSRetainedPrefixKernel


def PiCCSOriginalEvaluationKernel : Prop :=
  (∀ (columns : Nat) (masks : Array (Array (Nat × Nat)))
      (source : Fin productionShape.sourceCount) (output : Fin ringDegree),
    (PiCCSOriginalReads.read (PiDECParentSparseRead.prepare ()) masks source output :
      Fin columns → F) =
      PiCCSSourceImages.kernelRead (PiCCSOriginalReads.assignment masks source) output) ∧
  (∀ (point : PaperAlgebra.Point) (masks : Array (Array (Nat × Nat)))
      (source : Fin productionShape.sourceCount),
    ((PiCCSOriginalPad.range 0 PiCCSSourceImages.blockCount point masks).get source).toRing =
      (PaperAlgebra.evaluationFamily
        (Lifecycle.PiRLC.v1_2.InputBinding.relationSource PiDECInputCheck.relation)
        (PiCCSOriginalReads.assignment masks source) point).pad) ∧
  (∀ (masks : Array (Array (Nat × Nat))) (point : PaperAlgebra.Point)
      (source : Fin productionShape.sourceCount) (port : Fin Spec.ProductionRelation.matrixCount),
    ((PiCCSOriginalMatrixRange.originalRange masks point 0
      (PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application).rowCount
        source).get port).toRing =
      (PaperAlgebra.evaluationFamily
        (Lifecycle.PiRLC.v1_2.InputBinding.relationSource
          (PerApplicationFixedPoint.relation
            Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits))
        (PiCCSOriginalReads.assignment masks source) point).matrix port) ∧
  (∀ (columns arity count : Nat)
      (tables : PiRLCPartialTrace.FixedArray
        (PiRLCPartialTrace.FixedArray (Layout.ProductionRelation.SparseForm ringDegree)
          ringDegree) ringDegree)
      (masks : Array (Array (Nat × Nat))) (firstRow : Nat) (point : CubePoint K arity)
      (forms : Vector (Layout.MatrixProgram.RowForms columns) count)
      (source : Fin productionShape.sourceCount) (port : Fin Spec.ProductionRelation.matrixCount),
    ((PiCCSOriginalMatrixSupported.weighted (PiCCSOriginalSupport.isZero masks)
      firstRow point (PiCCSOriginalReads.read tables masks) forms).get
        (Fin.encodeProd (source, port))).toRing =
      ((PiCCSOriginalMatrixBatch.sum firstRow point
        (PiCCSOriginalReads.read tables masks) forms).get (Fin.encodeProd (source, port))).toRing) ∧
  (∀ (columns arity count : Nat)
      (tables : PiRLCPartialTrace.FixedArray
        (PiRLCPartialTrace.FixedArray (Layout.ProductionRelation.SparseForm ringDegree)
          ringDegree) ringDegree)
      (masks : Array (Array (Nat × Nat))) (firstRow : Nat) (point : CubePoint K arity)
      (interfaces : Vector (Layout.ProductionRelation.PoseidonSboxPlan.Interface columns) count)
      (source : Fin productionShape.sourceCount) (port : Fin Spec.ProductionRelation.matrixCount),
    ((PiCCSOriginalMatrixSupported.invocations (PiCCSOriginalSupport.isZero masks)
      firstRow point (PiCCSOriginalReads.read tables masks) interfaces).get
        (Fin.encodeProd (source, port))).toRing =
      ((PiCCSOriginalMatrixBatch.sumInvocations firstRow point
        (PiCCSOriginalReads.read tables masks) interfaces).get
          (Fin.encodeProd (source, port))).toRing)

theorem piCCSOriginalEvaluationKernel : PiCCSOriginalEvaluationKernel := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro columns masks source output
    exact PiCCSOriginalReads.read_eq_kernelRead masks source output
  · intro point masks source
    exact PiCCSOriginalPad.complete_eq_evaluationFamily point masks source
  · intro masks point source port
    exact (PiCCSOriginalMatrixRange.range_eq_matrix masks point source port).trans
      (PiCCSOriginalMatrixPreservation.matrix_eq_evaluationFamily masks point source port)

  · intro columns arity count tables masks firstRow point forms source port
    exact PiCCSOriginalMatrixSupported.weighted_isZero_source_port
      tables masks firstRow point forms source port
  · intro columns arity count tables masks firstRow point interfaces source port
    exact PiCCSOriginalMatrixSupported.invocations_isZero_source_port
      tables masks firstRow point interfaces source port

#audit_axioms piCCSOriginalEvaluationKernel

def PiCCSAllRoundSourceCoefficients : Prop :=
  PiCCSFirstRoundSourceCoefficients ∧
  (∀ (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat))),
    masks.size ≤ PiCCSSourceImages.blockCount →
    ∀ (challenges : List K), challenges ≠ [] →
    ∀ (remaining : Nat)
      (dimension : cubeVariables = challenges.length + remaining + 1),
      PiCCSPrefixComplete.coefficients? input masks challenges remaining =
        some ((PiCCSPrefixRound.roundPolynomial ConcreteCarrier.extensionOps
          (((ProductionKey.key
            (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
              Poseidon2HashChainV1Package.fits)
            Poseidon2HashChainV1Setup.productionAjtaiKey).statement
            (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)).sourceProtocolData
              K.embed ⟨PiCCSNormSource.assignments masks⟩)
          (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
          challenges (remaining := remaining) dimension).coefficients)) ∧
  (∀ (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat))),
    masks.size ≤ PiCCSSourceImages.blockCount →
    ∀ (first second : K) (remaining : Nat)
      (dimension : cubeVariables = [first, second].length + remaining + 1),
      PiCCSPrefixComplete.bucketCoefficients? input masks first second remaining =
        some ((PiCCSPrefixRound.roundPolynomial ConcreteCarrier.extensionOps
          (((ProductionKey.key
            (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
              Poseidon2HashChainV1Package.fits)
            Poseidon2HashChainV1Setup.productionAjtaiKey).statement
            (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)).sourceProtocolData
              K.embed ⟨PiCCSNormSource.assignments masks⟩)
          (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
          [first, second] (remaining := remaining) dimension).coefficients))

theorem piCCSAllRoundSourceCoefficients : PiCCSAllRoundSourceCoefficients := by
  refine ⟨piCCSFirstRoundSourceCoefficients, ?_, ?_⟩
  · intro input masks loaded challenges nonempty remaining dimension
    exact PiCCSPrefixComplete.coefficients_eq_roundPolynomial input masks loaded challenges nonempty dimension
  · intro input masks loaded first second remaining dimension
    exact PiCCSPrefixComplete.bucketCoefficients_eq_roundPolynomial input masks loaded first second dimension

#audit_axioms piCCSAllRoundSourceCoefficients


end LeanGraph.Targets
