import Nightstream.Implementation.Nebula.NIFS.PiRLC.ChallengeBridge
import Nightstream.Implementation.Nebula.NIFS.PiRLC.PostPiCcsBridge

/-!
Contract: exact row-derived V2 PiRLC parent.

This file connects the complete PiCCS output, post-output Poseidon2 state,
full-field sampler, and all 110 PiRLC algebra families. Satisfaction of those
rows derives the commitment, public input, and complete evaluation family of
the verifier-computed paper parent.

The placement contains only physical column identities and equality between
decoded PiRLC input wires and verifier-computed PiCCS output fields. It does
not contain challenges, parent fields, algebra equations, or a verifier
result.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 3000000

namespace Nightstream.Implementation.Nebula.ProductPiRlcParentBridge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows
open Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Exact history-free sampler input after the complete PiCCS output. -/
noncomputable def samplerInput
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifs.StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifs.RelationArtifact logicalWidth publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (wires : ProductPiCcsTypedBridge.Wires) (samplerBase : Nat) :
    ProductPiRlcTranscriptRows.Input :=
  ProductPiRlcPostPiCcsBridge.samplerInput
    (ProductPiCcsTypedBridge.rowInput statementId config artifact running fresh
      wires) samplerBase

/-- Physical links into one exact PiRLC parent computation.

The output columns are not linked to a claimed parent here. Their values are
derived from the algebra rows. -/
structure Placement
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifs.StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifs.RelationArtifact logicalWidth publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (proof : ProductPiCcsTypedBridge.ExactProof)
    (wires : ProductPiCcsTypedBridge.Wires) (samplerBase : Nat)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Prop where
  challengeSymbols : ProductPiRlcChallengeBridge.Placement
    (samplerInput statementId config artifact running fresh wires samplerBase)
    algebraLayout
  inputBundle : forall source,
    ((ProductConcreteNifs.key statementId config artifact).piCcsOutputs
      running fresh proof source).commitment =
      decodeInputBundles algebraLayout assignment canonical source
  inputPublic : forall source,
    ((ProductConcreteNifs.key statementId config artifact).piCcsOutputs
      running fresh proof source).publicInput =
      decodeInputPublic algebraLayout assignment canonical source
  inputEvaluation : forall source,
    ((ProductConcreteNifs.key statementId config artifact).piCcsOutputs
      running fresh proof source).evaluations =
      #[decodeInputEvaluations algebraLayout assignment canonical source]

/-- Combining fifteen singleton evaluation arrays produces the one exact
combined evaluation family. -/
theorem combineEvaluations_singletons
    (challenges : Source -> RingF)
    (families : Source -> ProductPaperAlgebra.Evaluation) :
    ProductPaperAlgebra.combineEvaluations challenges
        (fun source => #[families source]) =
      #[ProductPaperAlgebra.combineEvaluationFamily challenges families] := by
  rfl

/-- The selected challenge symbols are exactly the paper key's PiRLC
challenges. The post-PiCCS state is derived from PiCCS rows. -/
theorem challenges_eq_selected
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifs.StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifs.RelationArtifact logicalWidth publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (proof : ProductPiCcsTypedBridge.ExactProof)
    (wires : ProductPiCcsTypedBridge.Wires) (samplerBase : Nat)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (piCcsPlacement : ProductPiCcsTypedBridge.Placement statementId config
      artifact running fresh proof wires assignment)
    (piCcsRows : Satisfies
      (ProductPiCcsTranscriptRows.rows
        (ProductPiCcsTypedBridge.rowInput statementId config artifact running
          fresh wires)) assignment)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold
      (samplerInput statementId config artifact running fresh wires samplerBase)
      assignment)
    (classificationRows : ProductPiRlcCandidateClassificationRows.RowsHold
      (samplerInput statementId config artifact running fresh wires samplerBase)
      assignment)
    (selectorRows : ProductPiRlcFirstAcceptedBatchRows.RowsHold
      (samplerInput statementId config artifact running fresh wires samplerBase)
      assignment)
    (challengePlacement : ProductPiRlcChallengeBridge.Placement
      (samplerInput statementId config artifact running fresh wires samplerBase)
      algebraLayout) :
    decodeChallenges algebraLayout assignment
        (ProductPiRlcChallengeBridge.challengeSymbol_range
          (samplerInput statementId config artifact running fresh wires
            samplerBase) algebraLayout assignment canonical one transcriptRows
          classificationRows selectorRows challengePlacement) =
      (ProductConcreteNifs.key statementId config artifact).piRlcChallenges
        running fresh proof := by
  let sampleInput :=
    samplerInput statementId config artifact running fresh wires samplerBase
  have decoded := ProductPiRlcChallengeBridge.decodeChallenges_eq_piRlcResponse
    sampleInput algebraLayout assignment canonical one transcriptRows
    classificationRows selectorRows challengePlacement
  have stateEq :=
    ProductPiRlcPostPiCcsBridge.valueStart_eq_executionOutgoing statementId
      config artifact running fresh proof wires assignment samplerBase canonical
      one piCcsPlacement piCcsRows
  change ProductPiRlcFirstAcceptedBatchSound.samplerState sampleInput assignment =
      ((ProductConcreteNifs.key statementId config artifact).piCcsExecution
        running fresh proof).outgoingState at stateEq
  rw [decoded, stateEq]
  rfl

/-- All row-derived PiRLC fields equal the fields of the verifier-computed
paper parent. No parent field is an assumption. -/
theorem parentFields_of_rows
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifs.StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifs.RelationArtifact logicalWidth publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (proof : ProductPiCcsTypedBridge.ExactProof)
    (wires : ProductPiCcsTypedBridge.Wires) (samplerBase : Nat)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (piCcsPlacement : ProductPiCcsTypedBridge.Placement statementId config
      artifact running fresh proof wires assignment)
    (piCcsRows : Satisfies
      (ProductPiCcsTranscriptRows.rows
        (ProductPiCcsTypedBridge.rowInput statementId config artifact running
          fresh wires)) assignment)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold
      (samplerInput statementId config artifact running fresh wires samplerBase)
      assignment)
    (classificationRows : ProductPiRlcCandidateClassificationRows.RowsHold
      (samplerInput statementId config artifact running fresh wires samplerBase)
      assignment)
    (selectorRows : ProductPiRlcFirstAcceptedBatchRows.RowsHold
      (samplerInput statementId config artifact running fresh wires samplerBase)
      assignment)
    (algebraRows : Satisfies (ProductPiRlcAlgebraRows.rows algebraLayout)
      assignment)
    (placement : Placement statementId config artifact running fresh proof
      wires samplerBase algebraLayout assignment canonical) :
    let parent := (ProductConcreteNifs.key statementId config artifact).parent
      running fresh proof
    decodeOutputBundle algebraLayout assignment canonical = parent.commitment /\
      decodeOutputPublic algebraLayout assignment canonical = parent.publicInput /\
      #[decodeOutputEvaluation algebraLayout assignment canonical] =
        parent.evaluations := by
  dsimp only
  let range := ProductPiRlcChallengeBridge.challengeSymbol_range
    (samplerInput statementId config artifact running fresh wires samplerBase)
    algebraLayout assignment canonical one transcriptRows classificationRows
    selectorRows placement.challengeSymbols
  have challenges := challenges_eq_selected statementId config artifact running
    fresh proof wires samplerBase algebraLayout assignment canonical one
    piCcsPlacement piCcsRows transcriptRows classificationRows selectorRows
    placement.challengeSymbols
  have equations := typedEquations_of_rows
    (logicalWidth := logicalWidth) (publicFits := publicFits)
    canonical one range algebraRows
  have bundles :
      decodeInputBundles algebraLayout assignment canonical =
        fun source =>
          ((ProductConcreteNifs.key statementId config artifact).piCcsOutputs
            running fresh proof source).commitment := by
    funext source
    exact (placement.inputBundle source).symm
  have publics :
      decodeInputPublic algebraLayout assignment canonical =
        fun source =>
          ((ProductConcreteNifs.key statementId config artifact).piCcsOutputs
            running fresh proof source).publicInput := by
    funext source
    exact (placement.inputPublic source).symm
  have evaluations :
      (fun source =>
          ((ProductConcreteNifs.key statementId config artifact).piCcsOutputs
            running fresh proof source).evaluations) =
        fun source =>
          #[decodeInputEvaluations algebraLayout assignment canonical source] := by
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
          ((ProductConcreteNifs.key statementId config artifact).piRlcChallenges
            running fresh proof)
          (fun source =>
            ((ProductConcreteNifs.key statementId config artifact).piCcsOutputs
              running fresh proof source).commitment) := by
        rw [challenges, bundles]
        rfl
      _ = ((ProductConcreteNifs.key statementId config artifact).parent
          running fresh proof).commitment := by
        rfl
  constructor
  · calc
      decodeOutputPublic algebraLayout assignment canonical =
          Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
            (decodeChallenges algebraLayout assignment range)
            (decodeInputPublic algebraLayout assignment canonical) :=
        equations.2.1
      _ = Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
          ((ProductConcreteNifs.key statementId config artifact).piRlcChallenges
            running fresh proof)
          (fun source =>
            ((ProductConcreteNifs.key statementId config artifact).piCcsOutputs
              running fresh proof source).publicInput) := by
        rw [challenges, publics]
        rfl
      _ = ((ProductConcreteNifs.key statementId config artifact).parent
          running fresh proof).publicInput := by
        rfl
  · calc
      #[decodeOutputEvaluation algebraLayout assignment canonical] =
          #[ProductPaperAlgebra.combineEvaluationFamily
            (decodeChallenges algebraLayout assignment range)
            (decodeInputEvaluations algebraLayout assignment canonical)] :=
        congrArg (fun value => #[value]) equations.2.2
      _ = ProductPaperAlgebra.combineEvaluations
          (decodeChallenges algebraLayout assignment range)
          (fun source =>
            #[decodeInputEvaluations algebraLayout assignment canonical
              source]) :=
        (combineEvaluations_singletons _ _).symm
      _ = ProductPaperAlgebra.combineEvaluations
          ((ProductConcreteNifs.key statementId config artifact).piRlcChallenges
            running fresh proof)
          (fun source =>
            ((ProductConcreteNifs.key statementId config artifact).piCcsOutputs
              running fresh proof source).evaluations) := by
        rw [challenges, evaluations]
        rfl
      _ = ((ProductConcreteNifs.key statementId config artifact).parent
          running fresh proof).evaluations := by
        rfl

end Nightstream.Implementation.Nebula.ProductPiRlcParentBridge
