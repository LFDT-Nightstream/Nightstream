import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TypedReplayFor
import Nightstream.Implementation.NebulaV2.Production.NIFS.Core.ConcreteNifs

/-!
Contract: exponent-indexed production refinement of the complete product
PiCCS row program.

The generated augmented relation, NIFS key, public frame, alpha vector,
SumCheck certificate, full output, and transcript all use one
`rowVariables`. Satisfying rows derive the exact paper PiCCS check and the
post-PiCCS state. Placement contains only equalities from physical fields to
independent key, statement, and proof data. It contains no challenge,
transcript result, SumCheck chain, PiCCS Boolean, or NIFS verdict.

This module does not select a production exponent or prove fixed-point
closure. It also does not own PiRLC, PiDEC, terminal verification,
cryptographic security, generated-column placement, or Rust refinement.

Assurance tier: exponent-indexed typed row refinement.
-/

set_option autoImplicit false
set_option maxHeartbeats 1800000
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionProductPiCcsTypedBridgeFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptRowsFor
open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptSemanticsFor
open Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifsKey
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources

abbrev ExactProof (rowVariables : Nat) :=
  Proof K ProductPaperAlgebraFor.Commitment
    (ProductNifsCodec.shapeFor rowVariables) 9

/-- Physical fields for one PiCCS call. Verifier-owned structure data is not
part of this value. -/
structure Wires (rowVariables : Nat) where
  publicNifsFields : List LinCombNormal.LinComb
  publicNifsFields_length :
    publicNifsFields.length = publicFieldCount rowVariables
  priorPoint : Fin (Shape rowVariables).cubeVariables -> KMul.Carried
  claimedCoefficient : CarriedCoordinate (Shape rowVariables) -> KMul.Carried
  rounds : Fin (Shape rowVariables).cubeVariables ->
    KFixedPhaseSumCheck.Round 9
  fullOutput : Fin (Shape rowVariables).sourceCount ->
    Fin (Shape rowVariables).matrixCount ->
    Fin (Shape rowVariables).coefficientCount -> KMul.Carried
  current : KMul.Carried
  terminal : KMul.Carried
  transcriptBase : Nat

/-- The candidate key used by the production call. -/
noncomputable def paperKey
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits) :=
  (selectedKey candidate statementId config artifact).paper

@[simp] theorem paperKey_lift
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits) :
    (paperKey candidate statementId config artifact).lift = K.embed :=
  (selectedKey candidate statementId config artifact).lift_eq

@[simp] theorem paperKey_matrixSource
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits) :
    (paperKey candidate statementId config artifact).matrixSource =
      ProductPaperAlgebraFor.matrixSource artifact.system :=
  selectedKey_matrixSource candidate statementId config artifact

theorem paperKey_publicInputState
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
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)) :
    (paperKey candidate statementId config artifact).publicInputState
        running fresh =
      publicAbsorber candidate
        (ProductPoseidon2.initialStateForStatement statementId)
        running fresh :=
  SelectedKey.publicInputState_eq
    (selectedKey candidate statementId config artifact) running fresh

@[simp] theorem paperKey_absorbPiCcsOutput
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits) :
    (paperKey candidate statementId config artifact).absorbPiCcsOutput =
      ProductPoseidon2.absorbFullOutputFor rowVariables :=
  (selectedKey candidate statementId config artifact).absorbPiCcsOutput_eq

@[simp] theorem paperKey_piDecPublicInputSplit
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits) :
    (paperKey candidate statementId config artifact).piDecPublicInputSplit =
      ProductPaperAlgebraFor.publicInputSplit config := by
  change
    (ProductConcreteNifsFor.keyWithPublicAbsorption statementId config artifact
      (ProductionProductConcreteNifsKey.publicAbsorber candidate)
      ).piDecPublicInputSplit =
        ProductPaperAlgebraFor.publicInputSplit config
  exact ProductConcreteNifsFor.keyWithPublicAbsorption_piDecPublicInputSplit
    statementId config artifact
      (ProductionProductConcreteNifsKey.publicAbsorber candidate)

/-- Typed verifier input selected by the exact candidate key. -/
noncomputable def exactVerifierInput
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
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)) :
    ProtocolPolynomial.VerifierInput K (Shape rowVariables) :=
  ((paperKey candidate statementId config artifact).statement running fresh
    ).verifierInput (paperKey candidate statementId config artifact).lift

/-- Install verifier-owned structure data around physical prover fields. -/
noncomputable def rowInput
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
    (wires : Wires rowVariables) : Input rowVariables where
  statementId := statementId
  constraintPolynomial :=
    (exactVerifierInput candidate statementId config artifact running fresh
      ).constraintPolynomial
  publicNifsFields := wires.publicNifsFields
  publicNifsFields_length := wires.publicNifsFields_length
  priorPoint := wires.priorPoint
  claimedCoefficient := wires.claimedCoefficient
  rounds := wires.rounds
  fullOutput := wires.fullOutput
  current := wires.current
  terminal := wires.terminal
  transcriptBase := wires.transcriptBase

def decodeK (assignment : Nat -> Nat) (value : KMul.Carried) : K :=
  ofProjection (decoded assignment value)

/-- Physical placement. Each field binds a row-visible value to independent
key, statement, or proof data. -/
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat) : Prop where
  publicSerialization :
    fieldValues assignment wires.publicNifsFields =
      ProductionProductNifsPublicTranscript.publicNifsFields candidate 9
        running fresh
  statementSerialization :
    fieldValues assignment
        (statementFields
          (rowInput candidate statementId config artifact running fresh wires)) =
      ProductPoseidon2.statementFieldsFor rowVariables
        ({ priorState :=
            (paperKey candidate statementId config artifact
              ).publicInputState running fresh
           input := exactVerifierInput candidate statementId config artifact
             running fresh } : ProductPiCcsTypedReplayFor.PaperStatement
               rowVariables)
  roundSerialization : forall round,
    fieldValues assignment (roundFields round.val (wires.rounds round)) =
      ProductPoseidon2.roundFieldsFor round
        ((proof.piCcsRounds round).toMessage)
  outputSerialization :
    fieldValues assignment
        (fullOutputFields
          (rowInput candidate statementId config artifact running fresh wires)) =
      ProductPoseidon2.outputFieldsFor rowVariables proof.piCcsOutput
  priorPoint :
    (KPiCcsOccurrence.decodedVerifierInput
      (occurrenceInput
        (rowInput candidate statementId config artifact running fresh wires))
      assignment).priorPoint =
        (exactVerifierInput candidate statementId config artifact running fresh
          ).priorPoint
  claimedCoefficient : forall coordinate,
    (KPiCcsOccurrence.decodedVerifierInput
      (occurrenceInput
        (rowInput candidate statementId config artifact running fresh wires))
      assignment).claimedCoefficient coordinate =
        (exactVerifierInput candidate statementId config artifact running fresh
          ).claimedCoefficient coordinate
  roundPolynomial : forall round,
    KPiCcsOccurrence.decodedRound (wires.rounds round) assignment =
      proof.piCcsRounds round
  fullOutputCoordinate : forall source matrix coefficient,
    decodeK assignment (wires.fullOutput source matrix coefficient) =
      proof.piCcsOutput.coordinate source matrix coefficient

private theorem fixedCertificate_eq_of_rounds_eq
    {left right : Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Certificate
      K 9}
    (equal : left.rounds = right.rounds) : left = right := by
  cases left
  cases right
  simp only at equal
  subst equal
  rfl

private theorem cubePoint_eq_of_coordinates_eq
    {Field : Type} {dimensionCount : Nat}
    {left right : CubePoint Field dimensionCount}
    (equal : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp only at equal
  subst equal
  rfl

/-- The arithmetic occurrence reads the exact candidate-key verifier input. -/
theorem decodedVerifierInput_eq
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment) :
    KPiCcsOccurrence.decodedVerifierInput
        (occurrenceInput
          (rowInput candidate statementId config artifact running fresh wires))
        assignment =
      exactVerifierInput candidate statementId config artifact running fresh := by
  apply ProtocolPolynomial.VerifierInput.ext
  · rfl
  · exact placement.priorPoint
  · funext coordinate
    exact placement.claimedCoefficient coordinate

/-- The row messages decode to the proof's exact fixed-phase certificate. -/
theorem decodedCertificate_eq
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment) :
    KPiCcsOccurrence.decodedCertificate
        (occurrenceInput
          (rowInput candidate statementId config artifact running fresh wires))
        assignment =
      (paperKey candidate statementId config artifact
        ).piCcsFixedCertificate running fresh proof := by
  apply fixedCertificate_eq_of_rounds_eq
  unfold KPiCcsOccurrence.decodedCertificate occurrenceInput
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.piCcsFixedCertificate
  simp only [List.map_ofFn]
  apply congrArg List.ofFn
  funext round
  exact placement.roundPolynomial round

/-- The arithmetic occurrence reads projections of the complete output
family carried by the proof. -/
theorem decodedMessage_eq
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment) :
    KPiCcsOccurrence.decodedMessage
        (occurrenceInput
          (rowInput candidate statementId config artifact running fresh wires))
        assignment =
      ((paperKey candidate statementId config artifact
        ).piCcsCertificate running fresh proof).output := by
  apply ProtocolPolynomial.OutputMessage.ext
  · funext source matrix
    simpa [KPiCcsOccurrence.decodedMessage,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedMessage,
      KPiCcsTerminal.decoded, occurrenceInput, projectedFresh,
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.piCcsCertificate,
      StrongReduction.Statement.projectOutput, decodeK] using
        placement.fullOutputCoordinate
          (freshSourceIndex source) matrix
          (constantCoefficient rowVariables)
  · funext source
    simpa [KPiCcsOccurrence.decodedMessage,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedMessage,
      KPiCcsTerminal.decoded, occurrenceInput, projectedAssignment,
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.piCcsCertificate,
      StrongReduction.Statement.projectOutput, decodeK,
      firstMatrix, constantCoefficient, paperKey_matrixSource,
      ProductPaperAlgebraFor.matrixSource,
      Phi81CoefficientKernel.phi81Kernel] using
        placement.fullOutputCoordinate source
          (firstMatrix rowVariables) (constantCoefficient rowVariables)
  · funext coordinate
    simpa [KPiCcsOccurrence.decodedMessage,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedMessage,
      KPiCcsTerminal.decoded, occurrenceInput, projectedCarried,
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.piCcsCertificate,
      StrongReduction.Statement.projectOutput, decodeK] using
        placement.fullOutputCoordinate
          (runningSourceIndex coordinate.running)
          coordinate.matrix coordinate.coefficient

/-- Public row fields reach the selected key's exact public-input state. -/
private theorem publicReplayState
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment) :
    valueAbsorbPublic assignment
        (rowInput candidate statementId config artifact running fresh wires) =
      (paperKey candidate statementId config artifact
        ).publicInputState running fresh := by
  unfold valueAbsorbPublic
  change Poseidon2Duplex.absorbList ProductPoseidon2.constants
      (fieldValues assignment wires.publicNifsFields)
      (ProductPoseidon2.initialStateForStatement statementId) = _
  rw [placement.publicSerialization,
    paperKey_publicInputState candidate statementId config artifact running fresh]
  rfl

/-- Round fields serialize the exact transcript certificate. -/
private theorem roundReplaySerialization
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment)
    (round : Fin (Shape rowVariables).cubeVariables) :
    fieldValues assignment
        (roundFields round.val
          ((rowInput candidate statementId config artifact running fresh wires
            ).rounds round)) =
      ProductPoseidon2.roundFieldsFor round
        ((proof.piCcsRounds round).toMessage) := by
  simpa only [rowInput] using placement.roundSerialization round

/-- Row replay produces the exact Poseidon2 coins for the selected prior
state, verifier input, and proof messages. -/
theorem valueReplay_eq_expectedCoins
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment) :
    let input := rowInput candidate statementId config artifact running fresh wires
    let statement : ProductPiCcsTypedReplayFor.PaperStatement rowVariables :=
      { priorState := (paperKey candidate statementId config artifact
          ).publicInputState running fresh
        input := exactVerifierInput candidate statementId config artifact
          running fresh }
    let certificate : ProductPiCcsTypedReplayFor.PaperCertificate rowVariables :=
      { rounds := fun round => (proof.piCcsRounds round).toMessage }
    let expected := FiatShamir.derive
      (ProductPoseidon2.transcriptFor rowVariables) statement certificate
    (valueDeriveAlpha assignment input).1.map ofProjection =
        expected.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 = expected.gamma /\
      (valueReplayRounds assignment input).challenges.map ofProjection =
        expected.roundPoint.coordinates /\
      (valueReplayRounds assignment input).state = expected.finalState := by
  dsimp only
  exact ProductPiCcsTypedReplayFor.valueReplay_eq_derived_of_components
    (rowInput candidate statementId config artifact running fresh wires)
    ((paperKey candidate statementId config artifact
      ).publicInputState running fresh)
    (exactVerifierInput candidate statementId config artifact running fresh)
    (fun round => (proof.piCcsRounds round).toMessage) assignment one
    (publicReplayState candidate statementId config artifact running fresh
      proof wires assignment placement)
    placement.statementSerialization
    (roundReplaySerialization candidate statementId config artifact running
      fresh proof wires assignment placement)

/-- The expected Poseidon2 replay is the coin record used by the selected
paper key. -/
theorem executionCoins_eq_expected
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
    (proof : ExactProof rowVariables) :
    ((paperKey candidate statementId config artifact
      ).piCcsExecution running fresh proof).coins =
      FiatShamir.derive (ProductPoseidon2.transcriptFor rowVariables)
        ({ priorState := (paperKey candidate statementId config artifact
              ).publicInputState running fresh
           input := exactVerifierInput candidate statementId config artifact
             running fresh } : ProductPiCcsTypedReplayFor.PaperStatement
               rowVariables)
        ({ rounds := fun round => (proof.piCcsRounds round).toMessage } :
          ProductPiCcsTypedReplayFor.PaperCertificate rowVariables) := by
  rw [Key.piCcsExecution_coins_eq_derive]
  have transcriptEq :
      (paperKey candidate statementId config artifact).oracle.transcript =
        ProductPoseidon2.transcriptFor rowVariables := by
    exact congrArg ProtocolVerifier.Oracle.transcript
      (selectedKey candidate statementId config artifact).oracle_eq
  rw [transcriptEq]
  rfl

/-- Row replay produces exactly the coin record used by the selected key. -/
theorem valueReplay_eq_executionCoins
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment) :
    let input := rowInput candidate statementId config artifact running fresh wires
    let execution := (paperKey candidate statementId config artifact
      ).piCcsExecution running fresh proof
    (valueDeriveAlpha assignment input).1.map ofProjection =
        execution.coins.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 =
        execution.coins.gamma /\
      (valueReplayRounds assignment input).challenges.map ofProjection =
        execution.coins.roundPoint.coordinates /\
      (valueReplayRounds assignment input).state =
        execution.coins.finalState := by
  have replay := valueReplay_eq_expectedCoins candidate statementId config
    artifact running fresh proof wires assignment one placement
  have exact := executionCoins_eq_expected candidate statementId config artifact
    running fresh proof
  dsimp only at replay ⊢
  rw [← exact] at replay
  exact replay

/-- Satisfying transcript rows decode the exact alpha, gamma, and SumCheck
point used by the selected paper execution. -/
theorem decodedCoins_eq_executionCoins
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment)
    (satisfied : Satisfies
      (rows
        (rowInput candidate statementId config artifact running fresh wires))
      assignment) :
    let input := rowInput candidate statementId config artifact running fresh wires
    let occurrence := occurrenceInput input
    let execution := (paperKey candidate statementId config artifact
      ).piCcsExecution running fresh proof
    (KPiCcsOccurrence.decodedAlpha occurrence assignment).coordinates =
        execution.coins.alpha.coordinates /\
      KPiCcsOccurrence.decodedGamma occurrence assignment =
        execution.coins.gamma /\
      (KPiCcsOccurrence.decodedPoint occurrence assignment).coordinates =
        execution.coins.roundPoint.coordinates := by
  dsimp only
  let input := rowInput candidate statementId config artifact running fresh wires
  let occurrence := occurrenceInput input
  let execution := (paperKey candidate statementId config artifact
    ).piCcsExecution running fresh proof
  have physical := rows_replay_semantics assignment input residues one satisfied
  have exact := valueReplay_eq_executionCoins candidate statementId config
    artifact running fresh proof wires assignment one placement
  change
    (valueDeriveAlpha assignment input).1.map ofProjection =
        execution.coins.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 =
        execution.coins.gamma /\
      (valueReplayRounds assignment input).challenges.map ofProjection =
        execution.coins.roundPoint.coordinates /\
      (valueReplayRounds assignment input).state =
        execution.coins.finalState at exact
  have alphaPhysical := congrArg (List.map ofProjection) physical.1
  have gammaPhysical := congrArg ofProjection physical.2.1
  have pointPhysical := congrArg (List.map ofProjection) physical.2.2.1
  simp only [List.map_map] at alphaPhysical pointPhysical
  have alphaDecoded := ProductPiCcsTypedReplayFor.decodedAlpha_coordinates_eq
    input assignment
  have pointDecoded := ProductPiCcsTypedReplayFor.decodedPoint_coordinates_eq
    input assignment
  change
    (KPiCcsOccurrence.decodedAlpha occurrence assignment).coordinates =
      (deriveAlpha input).1.map fun value =>
        ofProjection (decoded assignment value) at alphaDecoded
  change
    (KPiCcsOccurrence.decodedPoint occurrence assignment).coordinates =
      (replayRounds input).challenges.map
        fun value => ofProjection (decoded assignment value) at pointDecoded
  have gammaDecoded :
      KPiCcsOccurrence.decodedGamma occurrence assignment =
        ofProjection (decoded assignment (deriveGamma input).1) := by
    rfl
  exact
    ⟨alphaDecoded.trans (alphaPhysical.trans exact.1),
      gammaDecoded.trans (gammaPhysical.trans exact.2.1),
      pointDecoded.trans (pointPhysical.trans exact.2.2.1)⟩

/-- Satisfying production rows imply the exact typed SumCheck chain used by
the selected key. The chain is a conclusion. -/
theorem rows_imply_piCcsChain
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment)
    (satisfied : Satisfies
      (rows
        (rowInput candidate statementId config artifact running fresh wires))
      assignment) :
    let key := paperKey candidate statementId config artifact
    let verifierInput := exactVerifierInput candidate statementId config artifact
      running fresh
    let execution := key.piCcsExecution running fresh proof
    SumCheck.Finite.FixedPhase.Chain key.extensionOps.toOps
      (verifierInput.initial key.extensionOps execution.coins.gamma)
      (key.piCcsFixedCertificate running fresh proof).rounds
      execution.coins.roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage key.extensionOps
        verifierInput execution.coins.alpha execution.coins.gamma
        execution.coins.roundPoint
        (key.piCcsCertificate running fresh proof).output) := by
  dsimp only
  let input := rowInput candidate statementId config artifact running fresh wires
  let occurrence := occurrenceInput input
  have chain := arithmetic_rows_sound input assignment one satisfied
  have inputEq := decodedVerifierInput_eq candidate statementId config artifact
    running fresh proof wires assignment placement
  have certificateEq := decodedCertificate_eq candidate statementId config
    artifact running fresh proof wires assignment placement
  have messageEq := decodedMessage_eq candidate statementId config artifact
    running fresh proof wires assignment placement
  have coinsEq := decodedCoins_eq_executionCoins candidate statementId config
    artifact running fresh proof wires assignment residues one placement satisfied
  change
    (KPiCcsOccurrence.decodedAlpha occurrence assignment).coordinates =
        ((paperKey candidate statementId config artifact
          ).piCcsExecution running fresh proof).coins.alpha.coordinates /\
      KPiCcsOccurrence.decodedGamma occurrence assignment =
        ((paperKey candidate statementId config artifact
          ).piCcsExecution running fresh proof).coins.gamma /\
      (KPiCcsOccurrence.decodedPoint occurrence assignment).coordinates =
        ((paperKey candidate statementId config artifact
          ).piCcsExecution running fresh proof).coins.roundPoint.coordinates
      at coinsEq
  have alphaEq :
      KPiCcsOccurrence.decodedAlpha (occurrenceInput input) assignment =
        ((paperKey candidate statementId config artifact
          ).piCcsExecution running fresh proof).coins.alpha := by
    apply cubePoint_eq_of_coordinates_eq
    simpa only [occurrence] using coinsEq.1
  have gammaEq :
      KPiCcsOccurrence.decodedGamma (occurrenceInput input) assignment =
        ((paperKey candidate statementId config artifact
          ).piCcsExecution running fresh proof).coins.gamma := by
    simpa only [occurrence] using coinsEq.2.1
  have pointEq :
      KPiCcsOccurrence.decodedPoint (occurrenceInput input) assignment =
        ((paperKey candidate statementId config artifact
          ).piCcsExecution running fresh proof).coins.roundPoint := by
    apply cubePoint_eq_of_coordinates_eq
    simpa only [occurrence] using coinsEq.2.2
  rw [inputEq, certificateEq, messageEq, alphaEq, gammaEq, pointEq] at chain
  exact chain

/-- Exact exponent-indexed PiCCS row soundness. No verifier result,
challenge, SumCheck chain, or paper acceptance result is an assumption. -/
theorem rows_imply_piCcsCheck_true
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment)
    (satisfied : Satisfies
      (rows
        (rowInput candidate statementId config artifact running fresh wires))
      assignment) :
    piCcsCheck (paperKey candidate statementId config artifact)
      running fresh proof = true := by
  apply (piCcsCheck_eq_true_iff
    (paperKey candidate statementId config artifact) running fresh proof).2
  exact rows_imply_piCcsChain candidate statementId config artifact running fresh
    proof wires assignment residues one placement satisfied

/-- Complete output absorption in the row replay is the selected PiCCS
outgoing state. -/
theorem valueAfterFullOutput_eq_executionOutgoing
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment) :
    valueAfterFullOutput assignment
        (rowInput candidate statementId config artifact running fresh wires) =
      ((paperKey candidate statementId config artifact
        ).piCcsExecution running fresh proof).outgoingState := by
  let input := rowInput candidate statementId config artifact running fresh wires
  let execution := (paperKey candidate statementId config artifact
    ).piCcsExecution running fresh proof
  have coins := valueReplay_eq_executionCoins candidate statementId config
    artifact running fresh proof wires assignment one placement
  change
    (valueDeriveAlpha assignment input).1.map ofProjection =
        execution.coins.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 =
        execution.coins.gamma /\
      (valueReplayRounds assignment input).challenges.map ofProjection =
        execution.coins.roundPoint.coordinates /\
      (valueReplayRounds assignment input).state =
        execution.coins.finalState at coins
  unfold valueAfterFullOutput
  rw [placement.outputSerialization, coins.2.2.2]
  change ProductPoseidon2.absorbFullOutputFor rowVariables
      execution.coins.finalState proof.piCcsOutput = execution.outgoingState
  have outgoing := Key.piCcsExecution_outgoingState_eq_absorbPiCcsOutput
    (paperKey candidate statementId config artifact) running fresh proof
  rw [paperKey_absorbPiCcsOutput candidate statementId config artifact] at outgoing
  exact outgoing.symm

/-- Row satisfaction derives the exact state handed to PiRLC after complete
PiCCS output absorption. -/
theorem rows_imply_outgoingState
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
    (proof : ExactProof rowVariables) (wires : Wires rowVariables)
    (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires assignment)
    (satisfied : Satisfies
      (rows
        (rowInput candidate statementId config artifact running fresh wires))
      assignment) :
    SymbolicDuplexSemantics.decodedBuilder assignment
        (afterFullOutput
          (rowInput candidate statementId config artifact running fresh wires)) =
      ((paperKey candidate statementId config artifact
        ).piCcsExecution running fresh proof).outgoingState := by
  let input := rowInput candidate statementId config artifact running fresh wires
  have physical := rows_replay_semantics assignment input residues one satisfied
  exact physical.2.2.2.2.trans
    (valueAfterFullOutput_eq_executionOutgoing candidate statementId config
      artifact running fresh proof wires assignment one placement)

end Nightstream.Implementation.NebulaV2.ProductionProductPiCcsTypedBridgeFor
