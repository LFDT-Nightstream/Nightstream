import Nightstream.Implementation.Nebula.Production.Carrier.StreamingStateBinding
import Nightstream.Implementation.Nebula.Production.NIFS.PiCCS.TypedBridgeFor
import Nightstream.Implementation.Nebula.NIFS.Running.RunningCoordinatesFor

/-!
Contract: exact claim-frame coordinates needed by the production PiCCS
statement.

Assurance tier: model-level serialization refinement.

Owns the two claim-frame windows that contain the prior point and carried
evaluations, their frame-order traversal, and the reordering from the
running-major claim codec to the coefficient-major PiCCS statement.

Does not own generated rows, a Poseidon2 binding circuit, PiCCS challenges,
collision resistance, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBinding

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFieldNativeFullClaim
open Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding
open Nightstream.Implementation.Nebula.ProductionProductNifsPublicTranscript
open Nightstream.Implementation.Encoding.NifsCanonicalCodec
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

def statementIdentifierFieldCount : Nat := 366
def publicPrefixFieldCount : Nat := 17
def runningPointFieldCount (rowVariables : Nat) : Nat := 2 * rowVariables
def runningCommitmentFieldCount : Nat := 14 * 3888
def runningPublicInputFieldCount : Nat := 14 * 540
def runningEvaluationFieldCount : Nat := 14 * 14 * ringDegree * 2

def runningFrameStart : Nat :=
  statementIdentifierFieldCount + publicPrefixFieldCount

def pointFrameStart : Nat := runningFrameStart

def evaluationRunningOffset (rowVariables : Nat) : Nat :=
  runningPointFieldCount rowVariables + runningCommitmentFieldCount +
    runningPublicInputFieldCount

def evaluationFrameStart (rowVariables : Nat) : Nat :=
  runningFrameStart + evaluationRunningOffset rowVariables

/-- The authoritative claim frame exposes the running codec sections in their
exact physical order. This theorem only unfolds the existing production
serializers; it does not define a second frame. -/
theorem authoritativeFrame_sections
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) :
    authoritativeFrame statementId degreeBound value =
      ProductPoseidon2.statementIdentifierFields statementId ++
      fixedPrefix candidate fullShape degreeBound ++
      nativeValues
          ((pointCodec fullShape.rowVariables).encode
            value.recursiveState.point) ++
      nativeValues
          (Codec.encodeFin ProductNifsCodec.bundleCodec 14
            value.recursiveState.commitments) ++
      nativeValues
          (Codec.encodeFin
            (publicInputCodec fullShape.publicWidth) 14
            value.recursiveState.publicInputs) ++
      nativeValues
          (Codec.encodeFin
            (ProductNifsCodec.evaluationCodecFor fullShape.rowVariables) 14
            value.recursiveState.evaluations) ++
      nativeValues (bundleFields value.commitmentBundle) ++
      value.ccsPublic.val := by
  rw [authoritativeFrame, frame, blocks, runningFields,
    ProductNifsRunningCoordinatesFor.runningCodecFor_sections]
  simp only [List.flatten_cons, List.flatten_nil,
    nativeValues, List.map_append]
  simp only [List.append_assoc, List.append_nil]

theorem production_window_geometry :
    pointFrameStart = 383 /\
      runningPointFieldCount 26 = 52 /\
      evaluationFrameStart 26 = 62427 /\
      runningEvaluationFieldCount = 21168 /\
      evaluationFrameStart 26 + runningEvaluationFieldCount = 83595 := by
  decide

private theorem drop_take_middle
    {Value : Type} (leading middle trailing : List Value) :
    ((leading ++ middle ++ trailing).drop leading.length).take middle.length =
      middle := by
  simp

/-- Exact authoritative-frame window for the prior evaluation point. -/
noncomputable def pointWindow
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) : List Nat :=
  (authoritativeFrame statementId degreeBound value).drop pointFrameStart |>
    List.take (runningPointFieldCount fullShape.rowVariables)

/-- Exact authoritative-frame window for all carried evaluations in
running-major claim-codec order. -/
noncomputable def evaluationWindow
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) : List Nat :=
  (authoritativeFrame statementId degreeBound value).drop
      (evaluationFrameStart fullShape.rowVariables) |>
    List.take runningEvaluationFieldCount

/-- The two non-contiguous authoritative-frame windows used by the variable
part of the PiCCS verifier input. Their order matches the claim codec. -/
noncomputable def selectedAuthoritativeFields
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) : List Nat :=
  pointWindow statementId degreeBound value ++
    evaluationWindow statementId degreeBound value

theorem pointWindow_eq
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) :
    pointWindow statementId degreeBound value =
      nativeValues
        ((pointCodec fullShape.rowVariables).encode
          value.recursiveState.point) := by
  let leading := ProductPoseidon2.statementIdentifierFields statementId ++
    fixedPrefix candidate fullShape degreeBound
  let point := nativeValues
    ((pointCodec fullShape.rowVariables).encode value.recursiveState.point)
  let trailing :=
    nativeValues
        (Codec.encodeFin ProductNifsCodec.bundleCodec 14
          value.recursiveState.commitments) ++
    nativeValues
        (Codec.encodeFin (publicInputCodec fullShape.publicWidth) 14
          value.recursiveState.publicInputs) ++
    nativeValues
        (Codec.encodeFin
          (ProductNifsCodec.evaluationCodecFor fullShape.rowVariables) 14
          value.recursiveState.evaluations) ++
    nativeValues (bundleFields value.commitmentBundle) ++
    value.ccsPublic.val
  have sections : authoritativeFrame statementId degreeBound value =
      leading ++ point ++ trailing := by
    rw [authoritativeFrame_sections]
    simp only [leading, point, trailing, List.append_assoc]
  have leadingLength : leading.length = pointFrameStart := by
    simp [leading, pointFrameStart, runningFrameStart,
      statementIdentifierFieldCount, publicPrefixFieldCount,
      ProductPoseidon2.statementIdentifierFields, fixedPrefix_length]
  have pointLength : point.length =
      runningPointFieldCount fullShape.rowVariables := by
    simp [point, nativeValues_length,
      ProductNifsRunningCoordinatesFor.point_encoded_length,
      runningPointFieldCount]
    omega
  rw [pointWindow, sections, ← leadingLength, ← pointLength]
  exact drop_take_middle leading point trailing

theorem evaluationWindow_eq
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) :
    evaluationWindow statementId degreeBound value =
      nativeValues
        (Codec.encodeFin
          (ProductNifsCodec.evaluationCodecFor fullShape.rowVariables) 14
          value.recursiveState.evaluations) := by
  let leading := ProductPoseidon2.statementIdentifierFields statementId ++
    fixedPrefix candidate fullShape degreeBound ++
    nativeValues
        ((pointCodec fullShape.rowVariables).encode
          value.recursiveState.point) ++
    nativeValues
        (Codec.encodeFin ProductNifsCodec.bundleCodec 14
          value.recursiveState.commitments) ++
    nativeValues
        (Codec.encodeFin (publicInputCodec fullShape.publicWidth) 14
          value.recursiveState.publicInputs)
  let evaluations := nativeValues
    (Codec.encodeFin
      (ProductNifsCodec.evaluationCodecFor fullShape.rowVariables) 14
      value.recursiveState.evaluations)
  let trailing := nativeValues (bundleFields value.commitmentBundle) ++
    value.ccsPublic.val
  have sections : authoritativeFrame statementId degreeBound value =
      leading ++ evaluations ++ trailing := by
    rw [authoritativeFrame_sections]
    simp only [leading, evaluations, trailing, List.append_assoc]
  have leadingLength : leading.length =
      evaluationFrameStart fullShape.rowVariables := by
    simp [leading, ProductPoseidon2.statementIdentifierFields,
      fixedPrefix_length, nativeValues_length,
      ProductNifsRunningCoordinatesFor.point_encoded_length,
      contract.publicWidth, MemoryBoundCcsPublic.coordinateCount,
      evaluationFrameStart, runningFrameStart, evaluationRunningOffset,
      statementIdentifierFieldCount, publicPrefixFieldCount,
      runningPointFieldCount, runningCommitmentFieldCount,
      runningPublicInputFieldCount]
    omega
  have evaluationsLength : evaluations.length =
      runningEvaluationFieldCount := by
    simp [evaluations, nativeValues_length, Codec.encodeFin_length,
      ProductNifsCodec.evaluationCodecFor_width,
      runningEvaluationFieldCount, ringDegree]
  rw [evaluationWindow, sections, ← leadingLength, ← evaluationsLength]
  exact drop_take_middle leading evaluations trailing

private theorem flatMap_flatMap_comm
    {Left Right Value : Type}
    (left : List Left) (right : List Right)
    (values : Left -> Right -> List Value) :
    (left.flatMap fun leftValue =>
      right.flatMap fun rightValue => values leftValue rightValue).Perm
    (right.flatMap fun rightValue =>
      left.flatMap fun leftValue => values leftValue rightValue) := by
  induction left with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons]
      exact (inductionHypothesis.append_left _).trans
        (List.flatMap_append_perm right (values head)
          (fun rightValue =>
            tail.flatMap fun leftValue => values leftValue rightValue))

private theorem flatMap_map_comm
    {Left Right Value : Type}
    (left : List Left) (right : List Right)
    (values : Left -> Right -> Value) :
    (left.flatMap fun leftValue =>
      right.map fun rightValue => values leftValue rightValue).Perm
    (right.flatMap fun rightValue =>
      left.map fun leftValue => values leftValue rightValue) := by
  simpa [← List.map_eq_flatMap] using
    flatMap_flatMap_comm left right
      (fun leftValue rightValue => [values leftValue rightValue])

private theorem nativeValues_encodeFin_values
    {Value : Type} (codec : Codec Value) (count : Nat)
    (values : Fin count -> Value) :
    nativeValues (Codec.encodeFin codec count values) =
      (List.ofFn values).flatMap fun value =>
        nativeValues (codec.encode value) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ]
      simp only [Codec.encodeFin, nativeValues, List.map_append,
        List.flatMap_cons]
      exact congrArg (fun tail =>
        List.map Fin.val (codec.encode (values 0)) ++ tail)
          (inductionHypothesis (fun index => values index.succ))

private theorem nativeValues_encodeFin
    {Value : Type} (codec : Codec Value) (count : Nat)
    (values : Fin count -> Value) :
    nativeValues (Codec.encodeFin codec count values) =
      (canonicalFinIndices count).flatMap fun index =>
        nativeValues (codec.encode (values index)) := by
  rw [nativeValues_encodeFin_values, canonicalFinIndices,
    List.ofFn_comp', List.flatMap_map]
  rfl

@[simp] theorem nativeValues_kCodec (value : K) :
    nativeValues (kCodec.encode value) = ProductPoseidon2.kFields value := by
  simp [nativeValues, ProductPoseidon2.kFields]

private theorem canonicalFinIndices_flatMap_getD
    {Value Field : Type} (values : List Value) (default : Value)
    (encode : Value -> List Field) :
    (canonicalFinIndices values.length).flatMap
        (fun index => encode (values.getD index.val default)) =
      values.flatMap encode := by
  have reindexed :
      (canonicalFinIndices values.length).map
          (fun index => values.getD index.val default) = values := by
    rw [canonicalFinIndices, List.map_ofFn]
    simpa [List.getD_eq_getElem?_getD] using List.ofFn_get values
  calc
    _ = ((canonicalFinIndices values.length).map
          (fun index => values.getD index.val default)).flatMap encode := by
        rw [List.flatMap_map]
    _ = values.flatMap encode :=
      congrArg (fun selected => selected.flatMap encode) reindexed

private theorem canonicalFinIndices_flatMap_getD_of_length
    {Value Field : Type} {count : Nat}
    (values : List Value) (default : Value) (encode : Value -> List Field)
    (dimension : values.length = count) :
    (canonicalFinIndices count).flatMap
        (fun index => encode (values.getD index.val default)) =
      values.flatMap encode := by
  subst count
  exact canonicalFinIndices_flatMap_getD values default encode

theorem nativePointFields_eq
    {rowVariables : Nat} (point : CubePoint K rowVariables) :
    nativeValues ((pointCodec rowVariables).encode point) =
      ProductPoseidon2.pointFields point := by
  rw [show (pointCodec rowVariables).encode point =
      Codec.encodeFin kCodec rowVariables
        (fun index => point.coordinates.getD index.val K.zero) by rfl]
  rw [nativeValues_encodeFin]
  simp only [nativeValues_kCodec]
  exact canonicalFinIndices_flatMap_getD_of_length point.coordinates K.zero
    ProductPoseidon2.kFields point.dimension

theorem nativeEvaluationFields_eq
    {rowVariables : Nat} (evaluation : ProductNifsCodec.EvaluationFor rowVariables) :
    nativeValues ((ProductNifsCodec.evaluationCodecFor rowVariables).encode
        evaluation) =
      (canonicalFinIndices 14).flatMap fun matrix =>
        (canonicalFinIndices ringDegree).flatMap fun coefficient =>
          ProductPoseidon2.kFields (evaluation matrix coefficient) := by
  rw [show (ProductNifsCodec.evaluationCodecFor rowVariables).encode
      evaluation =
        Codec.encodeFin
          (Codec.finFunction ringDegree kCodec) 14 evaluation by rfl]
  rw [nativeValues_encodeFin]
  apply List.flatMap_congr
  intro matrix _
  rw [show (Codec.finFunction ringDegree kCodec).encode
      (evaluation matrix) =
        Codec.encodeFin kCodec ringDegree (evaluation matrix) by rfl]
  rw [nativeValues_encodeFin]
  apply List.flatMap_congr
  intro coefficient _
  exact nativeValues_kCodec (evaluation matrix coefficient)

/-- Evaluation coordinates in the claim codec order: running changes
slowest, then matrix, then coefficient. -/
def frameOrderCarriedCoordinates
    (shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape) :
    List (CarriedCoordinate shape) :=
  (canonicalFinIndices shape.runningCount).flatMap fun running =>
    (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
      (canonicalFinIndices shape.coefficientCount).map fun coefficient =>
        { running, matrix, coefficient }

/-- The claim codec and PiCCS statement contain the same typed carried
coordinates. Only their traversal order differs. -/
theorem frameOrderCarriedCoordinates_perm_canonical
    (shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape) :
    (frameOrderCarriedCoordinates shape).Perm
      (canonicalCarriedCoordinates shape) := by
  let running := canonicalFinIndices shape.runningCount
  let matrix := canonicalFinIndices shape.matrixCount
  let coefficient := canonicalFinIndices shape.coefficientCount
  have runningMatrix :
      (running.flatMap fun runningValue =>
        matrix.flatMap fun matrixValue =>
          coefficient.map fun coefficientValue =>
            ({ running := runningValue
               matrix := matrixValue
               coefficient := coefficientValue } :
              CarriedCoordinate shape)).Perm
      (matrix.flatMap fun matrixValue =>
        running.flatMap fun runningValue =>
          coefficient.map fun coefficientValue =>
            ({ running := runningValue
               matrix := matrixValue
               coefficient := coefficientValue } :
              CarriedCoordinate shape)) :=
    flatMap_flatMap_comm running matrix fun runningValue matrixValue =>
      coefficient.map fun coefficientValue =>
        ({ running := runningValue
           matrix := matrixValue
           coefficient := coefficientValue } : CarriedCoordinate shape)
  have runningCoefficient :
      (matrix.flatMap fun matrixValue =>
        running.flatMap fun runningValue =>
          coefficient.map fun coefficientValue =>
            ({ running := runningValue
               matrix := matrixValue
               coefficient := coefficientValue } :
              CarriedCoordinate shape)).Perm
      (matrix.flatMap fun matrixValue =>
        coefficient.flatMap fun coefficientValue =>
          running.map fun runningValue =>
            ({ running := runningValue
               matrix := matrixValue
               coefficient := coefficientValue } :
              CarriedCoordinate shape)) := by
    apply List.Perm.flatMap_left
    intro matrixValue _
    exact flatMap_map_comm running coefficient fun runningValue
      coefficientValue =>
        ({ running := runningValue
           matrix := matrixValue
           coefficient := coefficientValue } : CarriedCoordinate shape)
  have matrixCoefficient :
      (matrix.flatMap fun matrixValue =>
        coefficient.flatMap fun coefficientValue =>
          running.map fun runningValue =>
            ({ running := runningValue
               matrix := matrixValue
               coefficient := coefficientValue } :
              CarriedCoordinate shape)).Perm
      (coefficient.flatMap fun coefficientValue =>
        matrix.flatMap fun matrixValue =>
          running.map fun runningValue =>
            ({ running := runningValue
               matrix := matrixValue
               coefficient := coefficientValue } :
              CarriedCoordinate shape)) :=
    flatMap_flatMap_comm matrix coefficient fun matrixValue coefficientValue =>
      running.map fun runningValue =>
        ({ running := runningValue
           matrix := matrixValue
           coefficient := coefficientValue } : CarriedCoordinate shape)
  exact runningMatrix.trans (runningCoefficient.trans matrixCoefficient)

def frameOrderClaimedFields
    {shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape}
    (claimed : CarriedCoordinate shape -> K) : List Nat :=
  (frameOrderCarriedCoordinates shape).flatMap fun coordinate =>
    ProductPoseidon2.kFields (claimed coordinate)

theorem frameOrderClaimedFields_perm_canonical
    {shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape}
    (claimed : CarriedCoordinate shape -> K) :
    (frameOrderClaimedFields claimed).Perm
      ((canonicalCarriedCoordinates shape).flatMap fun coordinate =>
        ProductPoseidon2.kFields (claimed coordinate)) := by
  exact (frameOrderCarriedCoordinates_perm_canonical shape).flatMap_right _

theorem nativeEvaluationsSection_eq
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (running : ProductNifsCodec.RunningFor rowVariables fullShape) :
    nativeValues
        (Codec.encodeFin (ProductNifsCodec.evaluationCodecFor rowVariables)
          14 running.evaluations) =
      frameOrderClaimedFields fun coordinate =>
        running.evaluations coordinate.running coordinate.matrix
          coordinate.coefficient := by
  rw [nativeValues_encodeFin]
  simp only [frameOrderClaimedFields, frameOrderCarriedCoordinates,
    ProductNifsCodec.shapeFor, List.flatMap_assoc, List.flatMap_map]
  apply List.flatMap_congr
  intro runningIndex _
  exact nativeEvaluationFields_eq (running.evaluations runningIndex)

/-- Variable PiCCS statement fields in claim-frame order. This order is for
the statement-binding digest only. The PiCCS transcript keeps its paper
coefficient-major order. -/
def frameOrderVariableFields
    {rowVariables : Nat}
    (input : ProtocolPolynomial.VerifierInput K
      (ProductNifsCodec.shapeFor rowVariables)) : List Nat :=
  ProductPoseidon2.pointFields input.priorPoint ++
    frameOrderClaimedFields input.claimedCoefficient

@[simp] theorem exactVerifierInput_priorPoint
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
    (ProductionProductPiCcsTypedBridgeFor.exactVerifierInput candidate
      statementId config artifact running fresh).priorPoint = running.point := by
  rfl

@[simp] theorem exactVerifierInput_claimedCoefficient
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
    (coordinate : CarriedCoordinate (ProductNifsCodec.shapeFor rowVariables)) :
    (ProductionProductPiCcsTypedBridgeFor.exactVerifierInput candidate
      statementId config artifact running fresh).claimedCoefficient coordinate =
        running.evaluations coordinate.running coordinate.matrix
          coordinate.coefficient := by
  rfl

/-- The exact production PiCCS input reads the same point and evaluations as
the field-native running claim. The right side is still in claim-frame order. -/
theorem frameOrderVariableFields_exactVerifierInput
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
    frameOrderVariableFields
        (ProductionProductPiCcsTypedBridgeFor.exactVerifierInput candidate
          statementId config artifact running fresh) =
      nativeValues ((pointCodec rowVariables).encode running.point) ++
      nativeValues
          (Codec.encodeFin (ProductNifsCodec.evaluationCodecFor rowVariables)
            14 running.evaluations) := by
  have claimedEqual :
      (ProductionProductPiCcsTypedBridgeFor.exactVerifierInput candidate
        statementId config artifact running fresh).claimedCoefficient =
      fun coordinate =>
        running.evaluations coordinate.running coordinate.matrix
          coordinate.coefficient := by
    funext coordinate
    exact exactVerifierInput_claimedCoefficient candidate statementId config
      artifact running fresh coordinate
  unfold frameOrderVariableFields
  rw [exactVerifierInput_priorPoint, claimedEqual, nativePointFields_eq,
    nativeEvaluationsSection_eq]
  rfl

/-- The variable point and claimed coefficients used by the exact production
PiCCS verifier input are the exact selected fields of the authoritative claim
frame. The equality is in claim-frame order; the earlier permutation theorem
accounts for the paper's coefficient-major PiCCS order. -/
theorem selectedAuthoritativeFields_exactVerifierInput
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (degreeBound : Nat)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (value : Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)) :
    selectedAuthoritativeFields statementId degreeBound value =
      frameOrderVariableFields
        (ProductionProductPiCcsTypedBridgeFor.exactVerifierInput candidate
          statementId config artifact value.recursiveState fresh) := by
  rw [selectedAuthoritativeFields, pointWindow_eq,
    evaluationWindow_eq
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits)]
  exact (frameOrderVariableFields_exactVerifierInput candidate statementId
    config artifact value.recursiveState fresh).symm

@[simp] theorem selectedAuthoritativeFields_length
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) :
    (selectedAuthoritativeFields statementId degreeBound value).length =
      2 * fullShape.rowVariables + 21168 := by
  rw [selectedAuthoritativeFields, List.length_append,
    pointWindow_eq, evaluationWindow_eq contract, nativeValues_length,
    nativeValues_length,
    ProductNifsRunningCoordinatesFor.point_encoded_length,
    Codec.encodeFin_length,
    ProductNifsCodec.evaluationCodecFor_width]
  simp
  omega

theorem selectedAuthoritativeFields_length_r26
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor 26 fullShape)
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) :
    (selectedAuthoritativeFields statementId degreeBound value).length =
      21220 := by
  rw [selectedAuthoritativeFields_length contract.toShape,
    contract.rowVariablesExact]

@[simp] theorem frameOrderCarriedCoordinates_length
    (shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape) :
    (frameOrderCarriedCoordinates shape).length =
      shape.runningCount * shape.matrixCount * shape.coefficientCount := by
  simp [frameOrderCarriedCoordinates, canonicalFinIndices_length]
  exact (Nat.mul_assoc _ _ _).symm

@[simp] theorem frameOrderClaimedFields_length
    {shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape}
    (claimed : CarriedCoordinate shape -> K) :
    (frameOrderClaimedFields claimed).length =
      2 * (shape.runningCount * shape.matrixCount *
        shape.coefficientCount) := by
  rw [frameOrderClaimedFields, List.length_flatMap]
  simp [ProductPoseidon2.kFields, frameOrderCarriedCoordinates_length]
  omega

@[simp] theorem frameOrderVariableFields_length
    {rowVariables : Nat}
    (input : ProtocolPolynomial.VerifierInput K
      (ProductNifsCodec.shapeFor rowVariables)) :
    (frameOrderVariableFields input).length =
      2 * rowVariables + 21168 := by
  simp [frameOrderVariableFields, ProductPoseidon2.pointFields,
    ProductPoseidon2.kFields, ProductNifsCodec.shapeFor,
    frameOrderClaimedFields_length, input.priorPoint.dimension, ringDegree]
  omega

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBinding
