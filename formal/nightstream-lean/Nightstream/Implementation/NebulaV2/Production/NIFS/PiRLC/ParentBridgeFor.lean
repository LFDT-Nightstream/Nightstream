import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.ChallengeBridge
import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.AlgebraSoundFor
import Nightstream.Implementation.NebulaV2.Production.NIFS.PiRLC.PostPiCcsBridgeFor

/-!
Contract: exact exponent-indexed row-derived production PiRLC parent.

One `rowVariables` parameter selects the generated relation, paper NIFS key,
PiCCS rows, and parent type. Satisfied transcript, classifier, selector, and
algebra rows derive every field of the verifier-computed PiRLC parent.
`Placement` contains only physical identities between authority-bearing PiCCS
outputs and algebra input wires. It does not contain a challenge, parent
field, algebra equation, or verifier result.

Does not own PiDEC, generated placement, terminal verification,
cryptographic security, or Rust refinement.

Assurance tier: exponent-indexed row-to-paper refinement.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.NebulaV2.ProductionProductPiRlcParentBridgeFor

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows
open Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSoundFor
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Exact sampler input after complete PiCCS output absorption. -/
noncomputable def samplerInput
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat) : ProductPiRlcTranscriptRows.Input :=
  ProductionProductPiRlcPostPiCcsBridgeFor.samplerInput
    (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId config
      artifact running fresh wires) samplerBase

/-- Physical links into one exact exponent-indexed PiRLC parent. -/
structure Placement
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat) (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Prop where
  challengeSymbols : ProductPiRlcChallengeBridge.Placement
    (samplerInput candidate statementId config artifact running fresh wires
      samplerBase) algebraLayout
  inputBundle : forall source,
    ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId config
      artifact).piCcsOutputs running fresh proof source).commitment =
      decodeInputBundles algebraLayout assignment canonical source
  inputPublic : forall source,
    ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId config
      artifact).piCcsOutputs running fresh proof source).publicInput =
      decodeInputPublic (rowVariables := rowVariables) algebraLayout assignment
        canonical source
  inputEvaluation : forall source,
    ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId config
      artifact).piCcsOutputs running fresh proof source).evaluations =
      #[decodeInputEvaluations rowVariables algebraLayout assignment canonical
        source]

/-- Combining singleton evaluation arrays gives the exact one-family parent. -/
theorem combineEvaluations_singletons
    (rowVariables : Nat) (challenges : Source -> RingF)
    (families : Source -> ProductPaperAlgebraFor.Evaluation rowVariables) :
    ProductPaperAlgebraFor.combineEvaluations rowVariables challenges
        (fun source => #[families source]) =
      #[ProductPaperAlgebraFor.combineEvaluationFamily challenges families] := by
  rfl

/-- Selected symbols equal the paper key's PiRLC challenges. The sampler state
is derived from PiCCS rows. -/
theorem challenges_eq_selected
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat) (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (piCcsPlacement : ProductionProductPiCcsTypedBridgeFor.Placement candidate
      statementId config artifact running fresh proof wires assignment)
    (piCcsRows : Satisfies
      (ProductPiCcsTranscriptRowsFor.rows
        (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
          config artifact running fresh wires)) assignment)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold
      (samplerInput candidate statementId config artifact running fresh wires
        samplerBase) assignment)
    (classificationRows : ProductPiRlcCandidateClassificationRows.RowsHold
      (samplerInput candidate statementId config artifact running fresh wires
        samplerBase) assignment)
    (selectorRows : ProductPiRlcFirstAcceptedBatchRows.RowsHold
      (samplerInput candidate statementId config artifact running fresh wires
        samplerBase) assignment)
    (challengePlacement : ProductPiRlcChallengeBridge.Placement
      (samplerInput candidate statementId config artifact running fresh wires
        samplerBase) algebraLayout) :
    decodeChallenges algebraLayout assignment
        (ProductPiRlcChallengeBridge.challengeSymbol_range
          (samplerInput candidate statementId config artifact running fresh
            wires samplerBase) algebraLayout assignment canonical one
          transcriptRows classificationRows selectorRows challengePlacement) =
      (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
        config artifact).piRlcChallenges running fresh proof := by
  let sampleInput := samplerInput candidate statementId config artifact running
    fresh wires samplerBase
  have decoded := ProductPiRlcChallengeBridge.decodeChallenges_eq_piRlcResponse
    sampleInput algebraLayout assignment canonical one transcriptRows
    classificationRows selectorRows challengePlacement
  have stateEq :=
    ProductionProductPiRlcPostPiCcsBridgeFor.valueStart_eq_executionOutgoing
      candidate statementId config artifact running fresh proof wires assignment
      samplerBase canonical one piCcsPlacement piCcsRows
  change ProductPiRlcFirstAcceptedBatchSound.samplerState sampleInput assignment =
      ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
        config artifact).piCcsExecution running fresh proof).outgoingState
    at stateEq
  change ProductPiRlcAlgebraSound.decodeChallenges algebraLayout assignment _ =
    (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
      config artifact).piRlcChallenges running fresh proof
  rw [decoded, stateEq]
  rfl

/-- All row-derived fields equal the verifier-computed exponent-indexed paper
parent. No parent field is an assumption. -/
theorem parentFields_of_rows
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat) (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (piCcsPlacement : ProductionProductPiCcsTypedBridgeFor.Placement candidate
      statementId config artifact running fresh proof wires assignment)
    (piCcsRows : Satisfies
      (ProductPiCcsTranscriptRowsFor.rows
        (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
          config artifact running fresh wires)) assignment)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold
      (samplerInput candidate statementId config artifact running fresh wires
        samplerBase) assignment)
    (classificationRows : ProductPiRlcCandidateClassificationRows.RowsHold
      (samplerInput candidate statementId config artifact running fresh wires
        samplerBase) assignment)
    (selectorRows : ProductPiRlcFirstAcceptedBatchRows.RowsHold
      (samplerInput candidate statementId config artifact running fresh wires
        samplerBase) assignment)
    (algebraRows : Satisfies (ProductPiRlcAlgebraRows.rows algebraLayout)
      assignment)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires samplerBase algebraLayout assignment canonical) :
    let parent := (ProductionProductPiCcsTypedBridgeFor.paperKey candidate
      statementId config artifact).parent running fresh proof
    decodeOutputBundle algebraLayout assignment canonical = parent.commitment /\
      decodeOutputPublic (rowVariables := rowVariables) algebraLayout assignment
        canonical = parent.publicInput /\
      #[decodeOutputEvaluation rowVariables algebraLayout assignment canonical] =
        parent.evaluations := by
  dsimp only
  let range := ProductPiRlcChallengeBridge.challengeSymbol_range
    (samplerInput candidate statementId config artifact running fresh wires
      samplerBase) algebraLayout assignment canonical one transcriptRows
    classificationRows selectorRows placement.challengeSymbols
  have challenges := challenges_eq_selected candidate statementId config
    artifact running fresh proof wires samplerBase algebraLayout assignment
    canonical one piCcsPlacement piCcsRows transcriptRows classificationRows
    selectorRows placement.challengeSymbols
  have equations := typedEquations_of_rows
    (rowVariables := rowVariables) (logicalWidth := logicalWidth)
    (publicFits := publicFits) canonical one range algebraRows
  have bundles :
      decodeInputBundles algebraLayout assignment canonical =
        fun source =>
          ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
            config artifact).piCcsOutputs running fresh proof source
            ).commitment := by
    funext source
    exact (placement.inputBundle source).symm
  have publics :
      decodeInputPublic (rowVariables := rowVariables) algebraLayout assignment
          canonical =
        fun source =>
          ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
            config artifact).piCcsOutputs running fresh proof source
            ).publicInput := by
    funext source
    exact (placement.inputPublic source).symm
  have evaluations :
      (fun source =>
          ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
            config artifact).piCcsOutputs running fresh proof source
            ).evaluations) =
        fun source =>
          #[decodeInputEvaluations rowVariables algebraLayout assignment
            canonical source] := by
    funext source
    exact placement.inputEvaluation source
  constructor
  · calc
      decodeOutputBundle algebraLayout assignment canonical =
          ProductCommitmentAlgebra.combineBundles
            (decodeChallenges algebraLayout assignment range)
            (decodeInputBundles algebraLayout assignment canonical) :=
        equations.1
      _ = ProductCommitmentAlgebra.combineBundles
          ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
            config artifact).piRlcChallenges running fresh proof)
          (fun source =>
            ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
              config artifact).piCcsOutputs running fresh proof source
              ).commitment) := by
        rw [challenges, bundles]
        rfl
      _ = ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
          config artifact).parent running fresh proof).commitment := by
        rfl
  constructor
  · calc
      decodeOutputPublic (rowVariables := rowVariables) algebraLayout assignment
          canonical =
          Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
            (decodeChallenges algebraLayout assignment range)
            (decodeInputPublic (rowVariables := rowVariables) algebraLayout
              assignment canonical) := equations.2.1
      _ = Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
          ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
            config artifact).piRlcChallenges running fresh proof)
          (fun source =>
            ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
              config artifact).piCcsOutputs running fresh proof source
              ).publicInput) := by
        rw [challenges, publics]
        rfl
      _ = ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
          config artifact).parent running fresh proof).publicInput := by
        rfl
  · calc
      #[decodeOutputEvaluation rowVariables algebraLayout assignment canonical] =
          #[ProductPaperAlgebraFor.combineEvaluationFamily
            (decodeChallenges algebraLayout assignment range)
            (decodeInputEvaluations rowVariables algebraLayout assignment
              canonical)] := congrArg (fun value => #[value]) equations.2.2
      _ = ProductPaperAlgebraFor.combineEvaluations rowVariables
          (decodeChallenges algebraLayout assignment range)
          (fun source =>
            #[decodeInputEvaluations rowVariables algebraLayout assignment
              canonical source]) :=
        (combineEvaluations_singletons rowVariables _ _).symm
      _ = ProductPaperAlgebraFor.combineEvaluations rowVariables
          ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
            config artifact).piRlcChallenges running fresh proof)
          (fun source =>
            ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
              config artifact).piCcsOutputs running fresh proof source
              ).evaluations) := by
        rw [challenges, evaluations]
        rfl
      _ = ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
          config artifact).parent running fresh proof).evaluations := by
        rfl

end Nightstream.Implementation.NebulaV2.ProductionProductPiRlcParentBridgeFor
