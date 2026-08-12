import Nightstream.Implementation.NebulaV2.NIFS.Core.Concrete
import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TranscriptSemantics
import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge

/-!
Contract: typed refinement of the exact V2 PiCCS transcript and arithmetic rows.

Owns the bridge from decoded row fields to the exact `ProductConcreteNifs`
paper verifier input, fixed-width certificate, complete PiCCS output, and
Poseidon2 transcript replay. Row satisfaction then implies that the concrete
paper PiCCS Boolean is true.

The placement interface contains only equality between decoded physical
fields and typed verifier or proof fields. It does not contain a challenge,
transcript state, SumCheck chain, PiCCS Boolean, or NIFS acceptance result.

Does not own the generated recursive column map, PiRLC, PiDEC, Rust
refinement, cryptographic transcript security, or extraction.

Emits constraints: no; it proves the meaning of
`ProductPiCcsTranscriptRows.rows`.
-/

set_option autoImplicit false
set_option maxHeartbeats 1200000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.NebulaV2.ProductPiCcsTypedBridge

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptRows
open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptSemantics
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources

abbrev ExactProof :=
  Proof K ProductPaperAlgebra.Commitment ProductNifsCodec.shape 9

/-- Physical fields for one exact V2 PiCCS call. The verifier-owned
constraint polynomial and statement identifier are not prover wires. -/
structure Wires where
  publicNifsFields : List LinCombNormal.LinComb
  publicNifsFields_length : publicNifsFields.length = 87655
  priorPoint : Fin ProductNifsCodec.shape.cubeVariables -> KMul.Carried
  claimedCoefficient :
    CarriedCoordinate ProductNifsCodec.shape -> KMul.Carried
  rounds : Fin ProductNifsCodec.shape.cubeVariables ->
    KFixedPhaseSumCheck.Round 9
  fullOutput : Fin ProductNifsCodec.shape.sourceCount ->
    Fin ProductNifsCodec.shape.matrixCount ->
    Fin ProductNifsCodec.shape.coefficientCount -> KMul.Carried
  current : KMul.Carried
  terminal : KMul.Carried
  transcriptBase : Nat

/-- The exact typed verifier input selected by the concrete V2 key. -/
noncomputable def exactVerifierInput
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
      (ProductPaperAlgebra.FullShape logicalWidth publicFits)) :
    ProtocolPolynomial.VerifierInput K ProductNifsCodec.shape :=
  let selected := ProductConcreteNifs.key statementId config artifact
  (selected.statement running fresh).verifierInput selected.lift

/-- Bind the verifier-owned structure fields and retain only physical prover
values as row expressions. -/
noncomputable def rowInput
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
    (wires : Wires) : ProductPiCcsTranscriptRows.Input where
  statementId := statementId
  constraintPolynomial :=
    (exactVerifierInput statementId config artifact running fresh
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

/-- Decode one row-carrier value into the paper's concrete extension field. -/
def decodeK (assignment : Nat -> Nat) (value : KMul.Carried) : K :=
  KConcreteFixedPhaseBridge.ofProjection
    (ProductPiCcsTranscriptSemantics.decoded assignment value)

@[simp] theorem decodeK_eq_pointDecoded
    (assignment : Nat -> Nat) (value : KMul.Carried) :
    decodeK assignment value = KPointEquality.decoded assignment value := by
  rfl

/-- Field placement for the exact call.

The four serialization equalities bind the physical transcript fields. The
remaining equalities bind the same physical values to the typed arithmetic
input and proof. None of these fields states or implies verifier acceptance
without the transcript and arithmetic rows. -/
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
    (proof : ExactProof)
    (wires : Wires) (assignment : Nat -> Nat) : Prop where
  publicSerialization :
    fieldValues assignment wires.publicNifsFields =
      ProductPoseidon2.publicNifsFields 9 running fresh
  statementSerialization :
    fieldValues assignment
        (ProductPiCcsTranscriptRows.statementFields
          (rowInput statementId config artifact running fresh wires)) =
      ProductPoseidon2.statementFields
        ({ priorState :=
            (ProductConcreteNifs.key statementId config artifact
              ).publicInputState running fresh
           input := exactVerifierInput statementId config artifact running fresh } :
          ProtocolVerifier.Statement K ProductPoseidon2.State
            ProductNifsCodec.shape)
  roundSerialization : forall round,
    fieldValues assignment
        (ProductPiCcsTranscriptRows.roundFields round.val
          (wires.rounds round)) =
      ProductPoseidon2.roundFields round
        ((proof.piCcsRounds round).toMessage)
  outputSerialization :
    fieldValues assignment
        (ProductPiCcsTranscriptRows.fullOutputFields
          (rowInput statementId config artifact running fresh wires)) =
      ProductPoseidon2.outputFields proof.piCcsOutput
  priorPoint :
    (KPiCcsOccurrence.decodedVerifierInput
      (ProductPiCcsTranscriptRows.occurrenceInput
        (rowInput statementId config artifact running fresh wires))
      assignment).priorPoint =
        (exactVerifierInput statementId config artifact running fresh).priorPoint
  claimedCoefficient : forall coordinate,
    (KPiCcsOccurrence.decodedVerifierInput
      (ProductPiCcsTranscriptRows.occurrenceInput
        (rowInput statementId config artifact running fresh wires))
      assignment).claimedCoefficient coordinate =
        (exactVerifierInput statementId config artifact running fresh
          ).claimedCoefficient coordinate
  roundPolynomial : forall round,
    KPiCcsOccurrence.decodedRound (wires.rounds round) assignment =
      proof.piCcsRounds round
  fullOutputCoordinate : forall source matrix coefficient,
    decodeK assignment (wires.fullOutput source matrix coefficient) =
      proof.piCcsOutput.coordinate source matrix coefficient

private theorem fixedCertificate_eq_of_rounds_eq
    {left right : Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Certificate K 9}
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

/-- The arithmetic occurrence reads the exact verifier-owned input. -/
theorem decodedVerifierInput_eq
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment) :
    KPiCcsOccurrence.decodedVerifierInput
        (ProductPiCcsTranscriptRows.occurrenceInput
          (rowInput statementId config artifact running fresh wires))
        assignment =
      exactVerifierInput statementId config artifact running fresh := by
  apply ProtocolPolynomial.VerifierInput.ext
  · rfl
  · exact placement.priorPoint
  · funext coordinate
    exact placement.claimedCoefficient coordinate

/-- The fixed-width row messages decode to exactly the proof's 25 paper
round polynomials. -/
theorem decodedCertificate_eq
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment) :
    KPiCcsOccurrence.decodedCertificate
        (ProductPiCcsTranscriptRows.occurrenceInput
          (rowInput statementId config artifact running fresh wires))
        assignment =
      (ProductConcreteNifs.key statementId config artifact
        ).piCcsFixedCertificate running fresh proof := by
  apply fixedCertificate_eq_of_rounds_eq
  unfold KPiCcsOccurrence.decodedCertificate
    ProductPiCcsTranscriptRows.occurrenceInput
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.piCcsFixedCertificate
  simp only [List.map_ofFn]
  apply congrArg List.ofFn
  funext round
  exact placement.roundPolynomial round

/-- The arithmetic occurrence reads the exact projection of the complete
coefficient family carried by the proof. -/
theorem decodedMessage_eq
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment) :
    KPiCcsOccurrence.decodedMessage
        (ProductPiCcsTranscriptRows.occurrenceInput
          (rowInput statementId config artifact running fresh wires))
        assignment =
      ((ProductConcreteNifs.key statementId config artifact
        ).piCcsCertificate running fresh proof).output := by
  apply ProtocolPolynomial.OutputMessage.ext
  · funext source matrix
    simpa [KPiCcsOccurrence.decodedMessage,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedMessage,
      KPiCcsTerminal.decoded, ProductPiCcsTranscriptRows.occurrenceInput,
      ProductPiCcsTranscriptRows.projectedFresh,
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.piCcsCertificate,
      StrongReduction.Statement.projectOutput, decodeK] using
        placement.fullOutputCoordinate
          (freshSourceIndex source) matrix
          ProductPiCcsTranscriptRows.constantCoefficient
  · funext source
    simpa [KPiCcsOccurrence.decodedMessage,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedMessage,
      KPiCcsTerminal.decoded, ProductPiCcsTranscriptRows.occurrenceInput,
      ProductPiCcsTranscriptRows.projectedAssignment,
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.piCcsCertificate,
      StrongReduction.Statement.projectOutput, decodeK,
      ProductPiCcsTranscriptRows.firstMatrix,
      ProductPiCcsTranscriptRows.constantCoefficient,
      ProductConcreteNifs.key, ProductPaperAlgebra.matrixSource,
      Phi81CoefficientKernel.phi81Kernel] using
        placement.fullOutputCoordinate source
          ProductPiCcsTranscriptRows.firstMatrix
          ProductPiCcsTranscriptRows.constantCoefficient
  · funext coordinate
    simpa [KPiCcsOccurrence.decodedMessage,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedMessage,
      KPiCcsTerminal.decoded, ProductPiCcsTranscriptRows.occurrenceInput,
      ProductPiCcsTranscriptRows.projectedCarried,
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.piCcsCertificate,
      StrongReduction.Statement.projectOutput, decodeK] using
        placement.fullOutputCoordinate
          (runningSourceIndex coordinate.running)
          coordinate.matrix coordinate.coefficient

/-! ## Exact Poseidon2 replay -/

/-- The symbolic Construction-3 challenge frame evaluates to the exact
value-level frame used by the selected V2 oracle. -/
theorem challengeFields_eq
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) :
    fieldValues assignment
        (ProductPiCcsTranscriptRows.verifierChallengeFields
          eventIndex challengeIndex challengeType coordinates) =
      ProductPoseidon2.verifierChallengeFields
        eventIndex challengeIndex challengeType coordinates := by
  unfold ProductPiCcsTranscriptRows.verifierChallengeFields
    ProductPoseidon2.verifierChallengeFields
  simp only [fieldValues, List.map_append, List.map_map,
    Function.comp_apply, lcEval_word assignment one]
  simp [
    ProductPoseidon2.construction3DomainFields,
    ProductPoseidon2.verifierChallengeLabelFields,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.construction3DomainFields,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.verifierChallengeLabelFields,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.stringFields,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.construction3DomainBytes,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.verifierChallengeLabelBytes,
    ProductPiCcsTranscriptRows.word, lcEval,
    Nightstream.Implementation.R1CS.Canonical.LinCombNormal.rawSum, one,
    ProductPoseidon2.word,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.word,
    goldilocksP, goldilocksModulus]

/-- One row-level extension squeeze is the same coordinate pair and state as
the selected concrete V2 squeeze. -/
theorem valueSqueeze_eq_concrete
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) (state : ProductPoseidon2.State) :
    (ofProjection
        (valueSqueezeVerifierChallenge assignment eventIndex challengeIndex
          challengeType coordinates state).1,
      (valueSqueezeVerifierChallenge assignment eventIndex challengeIndex
        challengeType coordinates state).2) =
      ProductPoseidon2.squeezeVerifierChallenge eventIndex challengeIndex
        challengeType coordinates state := by
  unfold valueSqueezeVerifierChallenge
    ProductPoseidon2.squeezeVerifierChallenge
  rw [challengeFields_eq assignment one]
  unfold SymbolicDuplexSemantics.squeezeKValue ProductPoseidon2.squeezeK
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.squeezeK
    SymbolicDuplexSemantics.challengeValue
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.challengeValue
  rfl

/-- Concrete alpha replay over explicit natural-number coordinates. -/
def concreteAlphaIndices :
    List Nat -> ProductPoseidon2.State -> List K × ProductPoseidon2.State
  | [], state => ([], state)
  | index :: rest, state =>
      let sampled := ProductPoseidon2.squeezeVerifierChallenge
        1 1 42 [index] state
      let tail := concreteAlphaIndices rest sampled.2
      (sampled.1 :: tail.1, tail.2)

/-- Mapping the row carrier to the paper carrier commutes with the complete
alpha recurrence. -/
theorem valueAlphaGo_eq_concrete
    (assignment : Nat -> Nat) (one : assignment 0 = 1) (input : Input) :
    forall index count state,
      ((valueDeriveAlphaGo assignment input index count state).1.map
          ofProjection,
        (valueDeriveAlphaGo assignment input index count state).2) =
      concreteAlphaIndices (List.range' index count) state
  | _, 0, _ => rfl
  | index, count + 1, state => by
      let valueSample := valueSqueezeVerifierChallenge assignment 1 1 42
        [index] state
      let concreteSample := ProductPoseidon2.squeezeVerifierChallenge
        1 1 42 [index] state
      have sampleEq := valueSqueeze_eq_concrete assignment one 1 1 42
        [index] state
      have sampleValueEq := congrArg Prod.fst sampleEq
      have sampleStateEq := congrArg Prod.snd sampleEq
      simp only at sampleValueEq sampleStateEq
      simp only [valueDeriveAlphaGo, List.range'_succ,
        concreteAlphaIndices, List.map_cons]
      rw [sampleValueEq, sampleStateEq]
      exact congrArg
        (fun tail : List K × ProductPoseidon2.State =>
          (concreteSample.1 :: tail.1, tail.2))
        (valueAlphaGo_eq_concrete assignment one input
          (index + 1) count concreteSample.2)

/-- The paper alpha schedule over an explicit finite-index list depends only
on the canonical natural coordinate order. -/
theorem squeezeMany_alpha_eq_indices
    (indices : List (Fin ProductNifsCodec.shape.cubeVariables))
    (state : ProductPoseidon2.State) :
    FiatShamir.squeezeMany ProductPoseidon2.transcript state
        (indices.map FiatShamir.ChallengeLabel.alpha) =
      concreteAlphaIndices (indices.map fun index => index.val) state := by
  induction indices generalizing state with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      let sampled := ProductPoseidon2.squeezeVerifierChallenge
        1 1 42 [index.val] state
      have tailEq := inductionHypothesis sampled.2
      simpa [FiatShamir.squeezeMany, ProductPoseidon2.transcript,
        concreteAlphaIndices, sampled] using
          congrArg
            (fun tail : List K × ProductPoseidon2.State =>
              (sampled.1 :: tail.1, tail.2)) tailEq

theorem canonicalFinIndices_values (count : Nat) :
    (canonicalFinIndices count).map (fun index => index.val) =
      List.range count := by
  apply List.ext_getElem
  · simp [canonicalFinIndices]
  · intro index leftBound rightBound
    simp [canonicalFinIndices]

/-- The value-level alpha recurrence is exactly the selected paper-oracle
alpha schedule. -/
theorem valueAlpha_eq_paper
    (assignment : Nat -> Nat) (one : assignment 0 = 1) (input : Input)
    (initialState : ProductPoseidon2.State) :
    ((valueDeriveAlphaGo assignment input 0
        ProductNifsCodec.shape.cubeVariables initialState).1.map ofProjection,
      (valueDeriveAlphaGo assignment input 0
        ProductNifsCodec.shape.cubeVariables initialState).2) =
      FiatShamir.squeezeMany ProductPoseidon2.transcript initialState
        (FiatShamir.alphaLabels ProductNifsCodec.shape) := by
  have valueEq := valueAlphaGo_eq_concrete assignment one input 0
    ProductNifsCodec.shape.cubeVariables initialState
  have paperEq := squeezeMany_alpha_eq_indices
    (canonicalFinIndices ProductNifsCodec.shape.cubeVariables) initialState
  unfold FiatShamir.alphaLabels at paperEq
  rw [canonicalFinIndices_values, List.range_eq_range'] at paperEq
  exact valueEq.trans paperEq.symm

/-- The two row absorption phases end at the exact initial state used by the
selected paper PiCCS transcript. -/
theorem valueAbsorbStatement_eq_paperInitial
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment) :
    valueAbsorbStatement assignment
        (rowInput statementId config artifact running fresh wires) =
      ProductPoseidon2.transcript.initialState
        ({ priorState :=
            (ProductConcreteNifs.key statementId config artifact
              ).publicInputState running fresh
           input := exactVerifierInput statementId config artifact running fresh } :
          ProtocolVerifier.Statement K ProductPoseidon2.State
            ProductNifsCodec.shape) := by
  have publicEq :
      fieldValues assignment
          (rowInput statementId config artifact running fresh wires
            ).publicNifsFields =
        ProductPoseidon2.publicNifsFields 9 running fresh := by
    simpa only [rowInput] using placement.publicSerialization
  unfold valueAbsorbStatement valueAbsorbPublic
  rw [publicEq, placement.statementSerialization]
  rfl

/-- The complete row pre-SumCheck replay is the exact selected paper replay.
In particular, gamma is not a placement field or a caller input. -/
theorem valuePreSumcheck_eq_paper
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment) :
    let input := rowInput statementId config artifact running fresh wires
    let statement :
        ProtocolVerifier.Statement K ProductPoseidon2.State
          ProductNifsCodec.shape :=
      { priorState :=
          (ProductConcreteNifs.key statementId config artifact
            ).publicInputState running fresh
        input := exactVerifierInput statementId config artifact running fresh }
    let paper :=
      FiatShamir.derivePreSumcheck ProductPoseidon2.transcript statement
    (valueDeriveAlpha assignment input).1.map ofProjection =
        paper.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 = paper.gamma /\
      (valueDeriveGamma assignment input).2 = paper.state := by
  dsimp only
  let input := rowInput statementId config artifact running fresh wires
  let statement :
      ProtocolVerifier.Statement K ProductPoseidon2.State
        ProductNifsCodec.shape :=
    { priorState :=
        (ProductConcreteNifs.key statementId config artifact
          ).publicInputState running fresh
      input := exactVerifierInput statementId config artifact running fresh }
  have initialEq :
      valueAbsorbStatement assignment input =
        ProductPoseidon2.transcript.initialState statement := by
    exact valueAbsorbStatement_eq_paperInitial statementId config artifact
      running fresh proof wires assignment placement
  have alphaEq := valueAlpha_eq_paper assignment one input
    (valueAbsorbStatement assignment input)
  have gammaEq := valueSqueeze_eq_concrete assignment one 2 2 43 []
    (valueDeriveAlpha assignment input).2
  change
    ((valueDeriveAlpha assignment input).1.map ofProjection,
      (valueDeriveAlpha assignment input).2) =
      FiatShamir.squeezeMany ProductPoseidon2.transcript
        (valueAbsorbStatement assignment input)
        (FiatShamir.alphaLabels ProductNifsCodec.shape) at alphaEq
  rw [initialEq] at alphaEq
  have alphaValuesEq := congrArg Prod.fst alphaEq
  have alphaStateEq := congrArg Prod.snd alphaEq
  simp only at alphaValuesEq alphaStateEq
  have gammaValueEq := congrArg Prod.fst gammaEq
  have gammaStateEq := congrArg Prod.snd gammaEq
  simp only at gammaValueEq gammaStateEq
  unfold FiatShamir.derivePreSumcheck
  change
    (valueDeriveAlpha assignment input).1.map ofProjection =
        (FiatShamir.squeezeMany ProductPoseidon2.transcript
          (ProductPoseidon2.transcript.initialState statement)
          (FiatShamir.alphaLabels ProductNifsCodec.shape)).1 /\
      ofProjection (valueDeriveGamma assignment input).1 =
        (ProductPoseidon2.transcript.squeeze
          (FiatShamir.squeezeMany ProductPoseidon2.transcript
            (ProductPoseidon2.transcript.initialState statement)
            (FiatShamir.alphaLabels ProductNifsCodec.shape)).2
          .gamma).1 /\
      (valueDeriveGamma assignment input).2 =
        (ProductPoseidon2.transcript.squeeze
          (FiatShamir.squeezeMany ProductPoseidon2.transcript
            (ProductPoseidon2.transcript.initialState statement)
            (FiatShamir.alphaLabels ProductNifsCodec.shape)).2
          .gamma).2
  refine ⟨alphaValuesEq, ?_, ?_⟩
  · rw [← alphaStateEq]
    simpa [valueDeriveGamma, ProductPoseidon2.transcript] using gammaValueEq
  · rw [← alphaStateEq]
    simpa [valueDeriveGamma, ProductPoseidon2.transcript] using gammaStateEq

/-! ## Exact SumCheck-round replay -/

/-- The selected certificate exposes exactly the typed proof message at one
round. -/
@[simp] theorem exactCertificate_round
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
    (proof : ExactProof)
    (round : Fin ProductNifsCodec.shape.cubeVariables) :
    ((ProductConcreteNifs.key statementId config artifact
      ).piCcsCertificate running fresh proof).rounds round =
      (proof.piCcsRounds round).toMessage := by
  rfl

/-- Replay an arbitrary canonical suffix of rounds. Each prover polynomial is
absorbed before its row-derived challenge, with the exact Construction-3
indices used by the selected paper oracle. -/
theorem valueReplayRoundsGo_eq_paper
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment) :
    forall
      (indices : List (Fin ProductNifsCodec.shape.cubeVariables))
      (rounds : List Round) (index : Nat)
      (state : ProductPoseidon2.State),
      indices.map wires.rounds = rounds ->
      indices.map (fun coordinate => coordinate.val) =
        List.range' index rounds.length ->
      ((valueReplayRoundsGo assignment
          (rowInput statementId config artifact running fresh wires)
          rounds index state).challenges.map ofProjection,
        (valueReplayRoundsGo assignment
          (rowInput statementId config artifact running fresh wires)
          rounds index state).state) =
        FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
          ((ProductConcreteNifs.key statementId config artifact
            ).piCcsCertificate running fresh proof).rounds state indices
  | [], rounds, _, _, roundsEq, _ => by
      have empty : rounds = [] := by
        simpa only [List.map_nil] using roundsEq.symm
      subst rounds
      rfl
  | _ :: _, [], _, _, roundsEq, _ => by
      simp only [List.map_cons, List.cons_ne_nil] at roundsEq
  | coordinate :: rest, round :: rounds, index, state,
      roundsEq, indicesEq => by
      simp only [List.map_cons, List.cons.injEq] at roundsEq
      simp only [List.map_cons, List.length_cons, List.range'_succ,
        List.cons.injEq] at indicesEq
      have roundEq := roundsEq.1
      have restRoundsEq := roundsEq.2
      have coordinateEq := indicesEq.1
      have restIndicesEq := indicesEq.2
      subst round
      subst index
      let valueAbsorbed :=
        Poseidon2Duplex.absorbList ProductPoseidon2.constants
          (fieldValues assignment
            (ProductPiCcsTranscriptRows.roundFields coordinate.val
              (wires.rounds coordinate))) state
      let paperAbsorbed :=
        Poseidon2Duplex.absorbList ProductPoseidon2.constants
          (ProductPoseidon2.roundFields coordinate
            (proof.piCcsRounds coordinate).toMessage) state
      have absorbedEq : valueAbsorbed = paperAbsorbed := by
        unfold valueAbsorbed paperAbsorbed
        rw [placement.roundSerialization coordinate]
      let valueSample := valueSqueezeVerifierChallenge assignment
        (4 + 2 * coordinate.val) (3 + coordinate.val) 46 [] valueAbsorbed
      let paperSample := ProductPoseidon2.squeezeVerifierChallenge
        (4 + 2 * coordinate.val) (3 + coordinate.val) 46 [] paperAbsorbed
      have sampleEq := valueSqueeze_eq_concrete assignment one
        (4 + 2 * coordinate.val) (3 + coordinate.val) 46 [] valueAbsorbed
      change (ofProjection valueSample.1, valueSample.2) = _ at sampleEq
      rw [absorbedEq] at sampleEq
      change (ofProjection valueSample.1, valueSample.2) = paperSample at sampleEq
      have sampleValueEq := congrArg Prod.fst sampleEq
      have sampleStateEq := congrArg Prod.snd sampleEq
      simp only at sampleValueEq sampleStateEq
      simp only [valueReplayRoundsGo, FiatShamir.deriveRoundsFrom,
        ProductPoseidon2.transcript, exactCertificate_round, List.map_cons]
      change
        (ofProjection valueSample.1 ::
            (valueReplayRoundsGo assignment
              (rowInput statementId config artifact running fresh wires)
              rounds (coordinate.val + 1) valueSample.2).challenges.map
                ofProjection,
          (valueReplayRoundsGo assignment
            (rowInput statementId config artifact running fresh wires)
            rounds (coordinate.val + 1) valueSample.2).state) =
          (paperSample.1 ::
            (FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
              ((ProductConcreteNifs.key statementId config artifact
                ).piCcsCertificate running fresh proof).rounds
              paperSample.2 rest).1,
            (FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
              ((ProductConcreteNifs.key statementId config artifact
                ).piCcsCertificate running fresh proof).rounds
              paperSample.2 rest).2)
      rw [sampleValueEq, sampleStateEq]
      exact congrArg
        (fun tail : List K × ProductPoseidon2.State =>
          (paperSample.1 :: tail.1, tail.2))
        (valueReplayRoundsGo_eq_paper statementId config artifact running
          fresh proof wires assignment one placement rest rounds
          (coordinate.val + 1) paperSample.2 restRoundsEq restIndicesEq)

/-- Canonical finite indices select the fixed physical round function in the
same order as `List.ofFn`. -/
theorem canonicalRoundWires (wires : Wires) :
    (canonicalFinIndices ProductNifsCodec.shape.cubeVariables).map
        wires.rounds =
      List.ofFn wires.rounds := by
  apply List.ext_getElem
  · simp only [List.length_map, canonicalFinIndices_length,
      List.length_ofFn]
  · intro index leftBound rightBound
    simp only [List.getElem_map]
    have sourceBound :
        index <
          (canonicalFinIndices ProductNifsCodec.shape.cubeVariables).length := by
      simpa only [List.length_map] using leftBound
    change wires.rounds
        ((canonicalFinIndices ProductNifsCodec.shape.cubeVariables
          )[index]'sourceBound) =
      (List.ofFn wires.rounds)[index]
    simp only [canonicalFinIndices, List.getElem_ofFn]
    congr 1

/-- The full fixed-width row replay returns the exact paper round point and
pre-output transcript state. -/
theorem valueRounds_eq_paper
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment) :
    let input := rowInput statementId config artifact running fresh wires
    let statement :
        ProtocolVerifier.Statement K ProductPoseidon2.State
          ProductNifsCodec.shape :=
      { priorState :=
          (ProductConcreteNifs.key statementId config artifact
            ).publicInputState running fresh
        input := exactVerifierInput statementId config artifact running fresh }
    let pre := FiatShamir.derivePreSumcheck ProductPoseidon2.transcript statement
    ((valueReplayRounds assignment input).challenges.map ofProjection,
      (valueReplayRounds assignment input).state) =
      FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
        ((ProductConcreteNifs.key statementId config artifact
          ).piCcsCertificate running fresh proof).rounds
        pre.state (canonicalFinIndices ProductNifsCodec.shape.cubeVariables) := by
  dsimp only
  let input := rowInput statementId config artifact running fresh wires
  let statement :
      ProtocolVerifier.Statement K ProductPoseidon2.State
        ProductNifsCodec.shape :=
    { priorState :=
        (ProductConcreteNifs.key statementId config artifact
          ).publicInputState running fresh
      input := exactVerifierInput statementId config artifact running fresh }
  let pre := FiatShamir.derivePreSumcheck ProductPoseidon2.transcript statement
  have preEq := valuePreSumcheck_eq_paper statementId config artifact running
    fresh proof wires assignment one placement
  change
    (valueDeriveAlpha assignment input).1.map ofProjection =
        pre.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 = pre.gamma /\
      (valueDeriveGamma assignment input).2 = pre.state at preEq
  have replay := valueReplayRoundsGo_eq_paper statementId config artifact
    running fresh proof wires assignment one placement
    (canonicalFinIndices ProductNifsCodec.shape.cubeVariables)
    (List.ofFn wires.rounds) 0 (valueDeriveGamma assignment input).2
    (canonicalRoundWires wires)
    (by rw [canonicalFinIndices_values, List.length_ofFn,
      List.range_eq_range'])
  change
    ((valueReplayRounds assignment input).challenges.map ofProjection,
      (valueReplayRounds assignment input).state) =
      FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
        ((ProductConcreteNifs.key statementId config artifact
          ).piCcsCertificate running fresh proof).rounds
        (valueDeriveGamma assignment input).2
        (canonicalFinIndices ProductNifsCodec.shape.cubeVariables) at replay
  exact replay.trans (congrArg
    (fun state =>
      FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
        ((ProductConcreteNifs.key statementId config artifact
          ).piCcsCertificate running fresh proof).rounds state
        (canonicalFinIndices ProductNifsCodec.shape.cubeVariables))
    preEq.2.2)

/-! ## Row-derived paper coins and PiCCS decision -/

private theorem derive_components
    {Context Field State : Type}
    {shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape}
    (oracle : FiatShamir.Oracle Context Field State shape)
    (context : Context) (certificate : FiatShamir.Certificate Field shape) :
    let pre := FiatShamir.derivePreSumcheck oracle context
    let rounds := FiatShamir.deriveRoundsFrom oracle certificate.rounds
      pre.state (canonicalFinIndices shape.cubeVariables)
    let derived := FiatShamir.derive oracle context certificate
    derived.alpha = pre.alpha /\
      derived.gamma = pre.gamma /\
      derived.roundPoint.coordinates = rounds.1 /\
      derived.finalState = rounds.2 := by
  exact ⟨rfl, rfl, rfl, rfl⟩

private theorem piCcsExecution_outgoing_exact
    {Extension Commitment PublicInput Scalar State : Type}
    {shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    (key.piCcsExecution running fresh proof).outgoingState =
      key.absorbPiCcsOutput
        (key.piCcsExecution running fresh proof).coins.finalState
        proof.piCcsOutput := by
  rfl

/-- The complete value replay is the coin record used by the exact selected
PiCCS execution. -/
theorem valueReplay_eq_executionCoins
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment) :
    let input := rowInput statementId config artifact running fresh wires
    let execution :=
      (ProductConcreteNifs.key statementId config artifact
        ).piCcsExecution running fresh proof
    (valueDeriveAlpha assignment input).1.map ofProjection =
        execution.coins.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 =
        execution.coins.gamma /\
      (valueReplayRounds assignment input).challenges.map ofProjection =
        execution.coins.roundPoint.coordinates /\
      (valueReplayRounds assignment input).state =
        execution.coins.finalState := by
  dsimp only
  let input := rowInput statementId config artifact running fresh wires
  let statement :
      ProtocolVerifier.Statement K ProductPoseidon2.State
        ProductNifsCodec.shape :=
    { priorState :=
        (ProductConcreteNifs.key statementId config artifact
          ).publicInputState running fresh
      input := exactVerifierInput statementId config artifact running fresh }
  let pre := FiatShamir.derivePreSumcheck ProductPoseidon2.transcript statement
  let roundResult :=
    FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
      ((ProductConcreteNifs.key statementId config artifact
        ).piCcsCertificate running fresh proof).rounds pre.state
      (canonicalFinIndices ProductNifsCodec.shape.cubeVariables)
  let certificate :=
    ((ProductConcreteNifs.key statementId config artifact
      ).piCcsCertificate running fresh proof).toTranscript
  let derived := FiatShamir.derive ProductPoseidon2.transcript statement
    certificate
  have preEq := valuePreSumcheck_eq_paper statementId config artifact running
    fresh proof wires assignment one placement
  change
    (valueDeriveAlpha assignment input).1.map ofProjection =
        pre.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 = pre.gamma /\
      (valueDeriveGamma assignment input).2 = pre.state at preEq
  have roundsEq := valueRounds_eq_paper statementId config artifact running
    fresh proof wires assignment one placement
  change
    ((valueReplayRounds assignment input).challenges.map ofProjection,
      (valueReplayRounds assignment input).state) = roundResult at roundsEq
  have roundValuesEq := congrArg Prod.fst roundsEq
  have roundStateEq := congrArg Prod.snd roundsEq
  simp only at roundValuesEq roundStateEq
  have components := derive_components ProductPoseidon2.transcript statement
    certificate
  change derived.alpha = pre.alpha /\
      derived.gamma = pre.gamma /\
      derived.roundPoint.coordinates = roundResult.1 /\
      derived.finalState = roundResult.2 at components
  have executionCoins :
      ((ProductConcreteNifs.key statementId config artifact
        ).piCcsExecution running fresh proof).coins = derived := by
    rfl
  rw [executionCoins]
  exact
    ⟨preEq.1.trans (congrArg CubePoint.coordinates components.1.symm),
      preEq.2.1.trans components.2.1.symm,
      roundValuesEq.trans components.2.2.1.symm,
      roundStateEq.trans components.2.2.2.symm⟩

/-- The occurrence's typed alpha point is exactly the canonical list decoded
from the row replay. -/
theorem decodedAlpha_coordinates_eq
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
    (wires : Wires) (assignment : Nat -> Nat) :
    let input := rowInput statementId config artifact running fresh wires
    (KPiCcsOccurrence.decodedAlpha
      (ProductPiCcsTranscriptRows.occurrenceInput input) assignment
      ).coordinates =
      (ProductPiCcsTranscriptRows.deriveAlpha input).1.map fun value =>
        ofProjection (decoded assignment value) := by
  dsimp only
  let input := rowInput statementId config artifact running fresh wires
  apply List.ext_getElem
  · simp only [KPiCcsOccurrence.decodedAlpha,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedAlpha,
      KPiCcsTerminal.alphaEqualityInput, KPointEquality.decodedRight,
      KPointEquality.indices, List.length_map, List.length_ofFn,
      deriveAlpha_length]
  · intro index leftBound rightBound
    simp only [KPiCcsOccurrence.decodedAlpha,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedAlpha,
      KPiCcsTerminal.alphaEqualityInput, KPointEquality.decodedRight,
      KPointEquality.indices, List.getElem_map, List.getElem_ofFn,
      ProductPiCcsTranscriptRows.occurrenceInput,
      ProductPiCcsTranscriptRows.alphaAt, KPointEquality.decoded,
      decodeK, ProductPiCcsTranscriptSemantics.decoded]
    congr 3

/-- The occurrence's typed SumCheck point is exactly the canonical challenge
list decoded from the row replay. -/
theorem decodedPoint_coordinates_eq
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
    (wires : Wires) (assignment : Nat -> Nat) :
    let input := rowInput statementId config artifact running fresh wires
    (KPiCcsOccurrence.decodedPoint
      (ProductPiCcsTranscriptRows.occurrenceInput input) assignment
      ).coordinates =
      (ProductPiCcsTranscriptRows.replayRounds input).challenges.map fun value =>
        ofProjection (decoded assignment value) := by
  dsimp only
  let input := rowInput statementId config artifact running fresh wires
  apply List.ext_getElem
  · simp only [KPiCcsOccurrence.decodedPoint,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedPoint,
      KPiCcsTerminal.alphaEqualityInput, KPointEquality.decodedLeft,
      KPointEquality.indices, List.length_map, List.length_ofFn,
      replayRounds_length]
  · intro index leftBound rightBound
    simp only [KPiCcsOccurrence.decodedPoint,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedPoint,
      KPiCcsTerminal.alphaEqualityInput, KPointEquality.decodedLeft,
      KPointEquality.indices, List.getElem_map, List.getElem_ofFn,
      ProductPiCcsTranscriptRows.occurrenceInput,
      ProductPiCcsTranscriptRows.pointAt, KPointEquality.decoded,
      decodeK, ProductPiCcsTranscriptSemantics.decoded]
    congr 3

/-- Satisfying transcript rows decode the exact alpha, gamma, and SumCheck
point used by the selected paper execution. -/
theorem decodedCoins_eq_executionCoins
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment)
    (satisfied : Satisfies
      (ProductPiCcsTranscriptRows.rows
        (rowInput statementId config artifact running fresh wires))
      assignment) :
    let input := rowInput statementId config artifact running fresh wires
    let occurrence := ProductPiCcsTranscriptRows.occurrenceInput input
    let execution :=
      (ProductConcreteNifs.key statementId config artifact
        ).piCcsExecution running fresh proof
    (KPiCcsOccurrence.decodedAlpha occurrence assignment).coordinates =
        execution.coins.alpha.coordinates /\
      KPiCcsOccurrence.decodedGamma occurrence assignment =
        execution.coins.gamma /\
      (KPiCcsOccurrence.decodedPoint occurrence assignment).coordinates =
        execution.coins.roundPoint.coordinates := by
  dsimp only
  let input := rowInput statementId config artifact running fresh wires
  let occurrence := ProductPiCcsTranscriptRows.occurrenceInput input
  let execution :=
    (ProductConcreteNifs.key statementId config artifact
      ).piCcsExecution running fresh proof
  have physical := rows_replay_semantics assignment input residues one satisfied
  have exact := valueReplay_eq_executionCoins statementId config artifact
    running fresh proof wires assignment one placement
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
  have alphaDecoded := decodedAlpha_coordinates_eq statementId config artifact
    running fresh wires assignment
  have pointDecoded := decodedPoint_coordinates_eq statementId config artifact
    running fresh wires assignment
  change
    (KPiCcsOccurrence.decodedAlpha occurrence assignment).coordinates =
      (ProductPiCcsTranscriptRows.deriveAlpha input).1.map fun value =>
        ofProjection (decoded assignment value) at alphaDecoded
  change
    (KPiCcsOccurrence.decodedPoint occurrence assignment).coordinates =
      (ProductPiCcsTranscriptRows.replayRounds input).challenges.map
        fun value => ofProjection (decoded assignment value) at pointDecoded
  have gammaDecoded :
      KPiCcsOccurrence.decodedGamma occurrence assignment =
        ofProjection (decoded assignment (deriveGamma input).1) := by
    rfl
  exact
    ⟨alphaDecoded.trans (alphaPhysical.trans exact.1),
      gammaDecoded.trans (gammaPhysical.trans exact.2.1),
      pointDecoded.trans (pointPhysical.trans exact.2.2.1)⟩

/-- **Exact V2 PiCCS row soundness.** No verifier result, transcript
challenge, SumCheck chain, or paper acceptance result is an assumption. -/
theorem rows_imply_piCcsCheck_true
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment)
    (satisfied : Satisfies
      (ProductPiCcsTranscriptRows.rows
        (rowInput statementId config artifact running fresh wires))
      assignment) :
    piCcsCheck (ProductConcreteNifs.key statementId config artifact)
      running fresh proof = true := by
  let input := rowInput statementId config artifact running fresh wires
  let occurrence := ProductPiCcsTranscriptRows.occurrenceInput input
  have chain := ProductPiCcsTranscriptRows.arithmetic_rows_sound input
    assignment one satisfied
  have inputEq := decodedVerifierInput_eq statementId config artifact running
    fresh proof wires assignment placement
  have certificateEq := decodedCertificate_eq statementId config artifact
    running fresh proof wires assignment placement
  have messageEq := decodedMessage_eq statementId config artifact running
    fresh proof wires assignment placement
  have coinsEq := decodedCoins_eq_executionCoins statementId config artifact
    running fresh proof wires assignment residues one placement satisfied
  change
    (KPiCcsOccurrence.decodedAlpha occurrence assignment).coordinates =
        ((ProductConcreteNifs.key statementId config artifact
          ).piCcsExecution running fresh proof).coins.alpha.coordinates /\
      KPiCcsOccurrence.decodedGamma occurrence assignment =
        ((ProductConcreteNifs.key statementId config artifact
          ).piCcsExecution running fresh proof).coins.gamma /\
      (KPiCcsOccurrence.decodedPoint occurrence assignment).coordinates =
        ((ProductConcreteNifs.key statementId config artifact
          ).piCcsExecution running fresh proof).coins.roundPoint.coordinates
      at coinsEq
  have alphaEq :
      KPiCcsOccurrence.decodedAlpha
          (ProductPiCcsTranscriptRows.occurrenceInput input) assignment =
        ((ProductConcreteNifs.key statementId config artifact
          ).piCcsExecution running fresh proof).coins.alpha := by
    apply cubePoint_eq_of_coordinates_eq
    simpa only [occurrence] using coinsEq.1
  have gammaEq :
      KPiCcsOccurrence.decodedGamma
          (ProductPiCcsTranscriptRows.occurrenceInput input) assignment =
        ((ProductConcreteNifs.key statementId config artifact
          ).piCcsExecution running fresh proof).coins.gamma := by
    simpa only [occurrence] using coinsEq.2.1
  have pointEq :
      KPiCcsOccurrence.decodedPoint
          (ProductPiCcsTranscriptRows.occurrenceInput input) assignment =
        ((ProductConcreteNifs.key statementId config artifact
          ).piCcsExecution running fresh proof).coins.roundPoint := by
    apply cubePoint_eq_of_coordinates_eq
    simpa only [occurrence] using coinsEq.2.2
  rw [inputEq, certificateEq, messageEq, alphaEq, gammaEq, pointEq] at chain
  exact (piCcsCheck_eq_true_iff
    (ProductConcreteNifs.key statementId config artifact)
    running fresh proof).2 chain

/-- The complete output absorption in the row replay is the selected PiCCS
outgoing state. The projected PiCCS message cannot replace this full output. -/
theorem valueAfterFullOutput_eq_executionOutgoing
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment) :
    valueAfterFullOutput assignment
        (rowInput statementId config artifact running fresh wires) =
      ((ProductConcreteNifs.key statementId config artifact
        ).piCcsExecution running fresh proof).outgoingState := by
  let input := rowInput statementId config artifact running fresh wires
  let execution :=
    (ProductConcreteNifs.key statementId config artifact
      ).piCcsExecution running fresh proof
  have coins := valueReplay_eq_executionCoins statementId config artifact
    running fresh proof wires assignment one placement
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
  change ProductPoseidon2.absorbFullOutput execution.coins.finalState
      proof.piCcsOutput = execution.outgoingState
  have outgoing := piCcsExecution_outgoing_exact
    (ProductConcreteNifs.key statementId config artifact)
    running fresh proof
  rw [ProductConcreteNifs.key_absorbPiCcsOutput] at outgoing
  exact outgoing.symm

/-- Row satisfaction derives the state handed to PiRLC after complete PiCCS
output absorption. -/
theorem rows_imply_outgoingState
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
    (proof : ExactProof) (wires : Wires) (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : Placement statementId config artifact running fresh proof
      wires assignment)
    (satisfied : Satisfies
      (ProductPiCcsTranscriptRows.rows
        (rowInput statementId config artifact running fresh wires))
      assignment) :
    SymbolicDuplexSemantics.decodedBuilder assignment
        (ProductPiCcsTranscriptRows.afterFullOutput
          (rowInput statementId config artifact running fresh wires)) =
      ((ProductConcreteNifs.key statementId config artifact
        ).piCcsExecution running fresh proof).outgoingState := by
  let input := rowInput statementId config artifact running fresh wires
  have physical := rows_replay_semantics assignment input residues one satisfied
  exact physical.2.2.2.2.trans
    (valueAfterFullOutput_eq_executionOutgoing statementId config artifact
      running fresh proof wires assignment one placement)

end Nightstream.Implementation.NebulaV2.ProductPiCcsTypedBridge
