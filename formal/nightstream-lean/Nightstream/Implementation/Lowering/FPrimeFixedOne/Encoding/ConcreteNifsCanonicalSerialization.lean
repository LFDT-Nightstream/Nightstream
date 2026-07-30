import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProfileViews
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCoverage

/-!
Contract: the Lean-owned dynamic statement and output serialization for the
selected fixed-active ConcretePhi81 NIFS.

The verifier setup is fixed by the selected Lean program.  The transcript
therefore binds one versioned shape header and every dynamic public field:
the checked parent, all fourteen running sources, the fresh source, the
PiCCS prior point and running claims, and the complete raw PiCCS output.

Owns: field order, framing words, source lists, exact serialization laws, and
the output cursor law.

Does not own: a relation matrix, application semantics, verifier acceptance,
physical rows, Rust, artifacts, or a security reduction across two different
verifier setups.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalSerialization

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProfileViews
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalViews
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private abbrev TranscriptState := Poseidon2Duplex.State

/-- Version of the selected dynamic statement encoding. -/
def statementVersion : Nat := 1

/-- Version of the selected raw-output encoding. -/
def outputVersion : Nat := 1

/-- Canonical field representative of one framing word. -/
def framingField (value : Nat) : Field :=
  residue value

/-- Read one physical coordinate of any fixed-width codec. -/
noncomputable def physicalView
    {α : Type}
    (codec : Codec α)
    (index : Fin codec.width) :
    FView codec (fun input => (codec.encode input).getD index.val 0) where
  index := index
  encodeValue := fun _ => rfl

/-- Exact field values of one codec encoding in physical order. -/
def physicalValues
    {α : Type}
    (codec : Codec α)
    (input : α) : List Field :=
  List.ofFn fun index : Fin codec.width =>
    (codec.encode input).getD index.val 0

theorem physicalValues_eq_encode
    {α : Type}
    (codec : Codec α)
    (input : α) :
    physicalValues codec input = codec.encode input := by
  apply List.ext_get
  · simp [physicalValues, codec.encode_length]
  · intro index leftLt rightLt
    simp only [physicalValues, List.get_eq_getElem, List.getElem_ofFn]
    rw [List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem rightLt]
    rfl

/-- Canonical low/high field order for one extension value. -/
def kFields (value : Nightstream.SuperNeo.Concrete.K) : List Nat :=
  [value.c0.val, value.c1.val]

/-- Fixed-size point fields in coordinate-major, low/high order. -/
def pointFields
    {variables : Nat}
    (point : CubePoint Nightstream.SuperNeo.Concrete.K variables) : List Nat :=
  (List.ofFn fun coordinate : Fin variables =>
    kFields
      (ConcreteNifsCarrierViews.pointCoordinate coordinate point)).flatten

/-- Fixed-size running-claim fields in running, matrix, lane, low/high order. -/
def claimedYRingFields
    {shape : SemanticShape}
    (claimed : Fin shape.runningCount → Fin shape.matrixCount → RingK) :
    List Nat :=
  (List.ofFn fun running : Fin shape.runningCount =>
    (List.ofFn fun matrix : Fin shape.matrixCount =>
      (List.ofFn fun lane : Fin ringDegree =>
        kFields (claimed running matrix lane)).flatten).flatten).flatten

/-- Complete raw output fields in source, matrix, lane, low/high order,
followed by the independent column-point product in source, lane, low/high
order. -/
def rawOutputFields
    {shape : SemanticShape}
    (output : OutputMessage shape) : List Nat :=
  [(framingField outputVersion).val,
   (framingField shape.sourceCount).val,
   (framingField shape.matrixCount).val] ++
    (List.ofFn fun source : Fin shape.sourceCount =>
      (List.ofFn fun matrix : Fin shape.matrixCount =>
        (List.ofFn fun lane : Fin ringDegree =>
          kFields (output.yRing source matrix lane)).flatten).flatten).flatten ++
    (List.ofFn fun source : Fin shape.sourceCount =>
      (List.ofFn fun lane : Fin ringDegree =>
        kFields (output.yZcol source lane)).flatten).flatten

/-- Forget only fields that the selected context fixes by construction. -/
def parentPayload
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (statement :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    FixedActive.Canonical.ParentPayload
      shape publicRingColumns publicFits verifierRows where
  commitment := statement.commitment
  publicInput := statement.publicInput
  point := statement.point
  evaluations := statement.evaluations

/-- Forget only fields that the selected context fixes by construction. -/
def runningPayload
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (statement :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    FixedActive.Canonical.RunningPayload
      shape publicRingColumns publicFits verifierRows where
  commitment := statement.commitment
  publicInput := statement.publicInput
  point := statement.point
  evaluations := statement.evaluations

/-- Forget only fields that the selected context fixes by construction. -/
def freshPayload
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (statement :
      Phi81Relation.CCSStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    FixedActive.Canonical.FreshPayload
      shape publicRingColumns publicFits verifierRows where
  commitment := statement.commitment
  publicInput := statement.publicInput

/-- Dynamic selected running fields.  A missing parent has a distinct tag and
cannot collide with the active-parent branch. -/
noncomputable def statementRunningFields
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (input :
      StatementInput shape publicRingColumns publicFits verifierRows
        FixedActive.arity) : List Nat :=
  match input.runningParent with
  | none => [0]
  | some parent =>
      1 ::
        ((runningCodec shape publicRingColumns verifierRows publicFits).encode
          {
            parent := parentPayload parent
            children := fun child =>
              runningPayload (input.sources.running child)
          }).map Fin.val

/-- Dynamic selected fresh fields.  The fixed-active arity has one fresh
source, so no default or truncating read occurs. -/
noncomputable def statementFreshFields
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
  (input :
      StatementInput shape publicRingColumns publicFits verifierRows
        FixedActive.arity) : List Nat :=
  ((freshCodec shape publicRingColumns verifierRows publicFits).encode
    (freshPayload
      (input.sources.fresh
        (Fin.cast FixedActive.arity_freshCount.symm (0 : Fin 1))))).map
          Fin.val

/-- Complete dynamic statement fields after the selected verifier setup is
fixed by the Lean program. -/
noncomputable def dynamicStatementFields
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (statement :
      Statement
        (VerifierKey shape publicRingColumns publicFits verifierRows)
        (StatementInput shape publicRingColumns publicFits verifierRows
          FixedActive.arity)) : List Nat :=
  [(framingField statementVersion).val,
   (framingField shape.rowVariables).val,
   (framingField shape.logicalWidth).val,
   (framingField shape.matrixCount).val,
   (framingField shape.freshCount).val,
   (framingField shape.runningCount).val,
   (framingField publicRingColumns).val,
   (framingField verifierRows).val,
   (framingField
      (if statement.input.pending.isSome then 1 else 0)).val] ++
    statementRunningFields statement.input ++
    statementFreshFields statement.input ++
    pointFields statement.input.polynomial.priorPoint ++
    claimedYRingFields statement.input.polynomial.claimedYRing

/-- Selected serialization used by the canonical Poseidon2 transcript. -/
noncomputable def serialization
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    KSplitNcPoseidonSchedule.Serialization
      (VerifierKey shape publicRingColumns publicFits verifierRows)
      (StatementInput shape publicRingColumns publicFits verifierRows
        FixedActive.arity)
      shape where
  statementFields := dynamicStatementFields
  outputFields := rawOutputFields

/-- Every physical running-codec field is one authoritative running source. -/
noncomputable def runningSources
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    List
      (FieldSource
        (runningCodec shape publicRingColumns verifierRows publicFits)
        (freshCodec shape publicRingColumns verifierRows publicFits)
        (proofCodec shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits)) :=
  List.ofFn fun index :
      Fin
        (runningCodec
          shape publicRingColumns verifierRows publicFits).width =>
    .running
      (fun running =>
        ((runningCodec
          shape publicRingColumns verifierRows publicFits).encode running).getD
            index.val 0)
      (physicalView
        (runningCodec shape publicRingColumns verifierRows publicFits) index)

/-- Every physical fresh-codec field is one authoritative fresh source. -/
noncomputable def freshSources
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    List
      (FieldSource
        (runningCodec shape publicRingColumns verifierRows publicFits)
        (freshCodec shape publicRingColumns verifierRows publicFits)
        (proofCodec shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits)) :=
  List.ofFn fun index :
      Fin
        (freshCodec shape publicRingColumns verifierRows publicFits).width =>
    .fresh
      (fun fresh =>
        ((freshCodec
          shape publicRingColumns verifierRows publicFits).encode fresh).getD
            index.val 0)
      (physicalView
        (freshCodec shape publicRingColumns verifierRows publicFits) index)

private def proofKSource
    {shape : SemanticShape}
    {constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount}
    {priorAbsorbed publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows →
        Nightstream.SuperNeo.Concrete.K}
    (view :
      KView
        (proofCodec shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits)
        value)
    (component : KComponent) :
    FieldSource
      (runningCodec shape publicRingColumns verifierRows publicFits)
      (freshCodec shape publicRingColumns verifierRows publicFits)
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits) :=
  .proof
    (fun proof => component.value (value proof))
    (component.view view)

/-- Public PiCCS input fields in prior-point, then running/matrix/lane order. -/
noncomputable def proofStatementSources
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    List
      (FieldSource
        (runningCodec shape publicRingColumns verifierRows publicFits)
        (freshCodec shape publicRingColumns verifierRows publicFits)
        (proofCodec shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits)) :=
  (List.ofFn fun coordinate : Fin shape.rowVariables =>
    [proofKSource
      (endpointViews shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits |>.priorPoint coordinate)
      .c0,
     proofKSource
      (endpointViews shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits |>.priorPoint coordinate)
      .c1]).flatten ++
  (List.ofFn fun running : Fin shape.runningCount =>
    (List.ofFn fun matrix : Fin shape.matrixCount =>
      (List.ofFn fun lane : Fin ringDegree =>
        [proofKSource
          (endpointViews shape constraintPolynomial priorAbsorbed
            publicRingColumns verifierRows publicFits |>.claimedYRing
              running matrix lane)
          .c0,
         proofKSource
          (endpointViews shape constraintPolynomial priorAbsorbed
            publicRingColumns verifierRows publicFits |>.claimedYRing
              running matrix lane)
          .c1]).flatten).flatten).flatten

/-- Framed dynamic statement sources in exactly the selected serialization
order. -/
noncomputable def statementSources
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    List
      (FieldSource
        (runningCodec shape publicRingColumns verifierRows publicFits)
        (freshCodec shape publicRingColumns verifierRows publicFits)
        (proofCodec shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits)) :=
  [.constant (framingField statementVersion),
   .constant (framingField shape.rowVariables),
   .constant (framingField shape.logicalWidth),
   .constant (framingField shape.matrixCount),
   .constant (framingField shape.freshCount),
   .constant (framingField shape.runningCount),
   .constant (framingField publicRingColumns),
   .constant (framingField verifierRows),
   .constant (framingField 0),
   .constant (framingField 1)] ++
    runningSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits ++
    freshSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits ++
    proofStatementSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits

/-- Raw output-ring sources in source, matrix, lane, low/high order. -/
noncomputable def outputYRingSources
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    List
      (FieldSource
        (runningCodec shape publicRingColumns verifierRows publicFits)
        (freshCodec shape publicRingColumns verifierRows publicFits)
        (proofCodec shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits)) :=
  (List.ofFn fun source : Fin shape.sourceCount =>
    (List.ofFn fun matrix : Fin shape.matrixCount =>
      (List.ofFn fun lane : Fin ringDegree =>
        [proofKSource
          (endpointViews shape constraintPolynomial priorAbsorbed
            publicRingColumns verifierRows publicFits |>.outputYRing
              source matrix lane)
          .c0,
         proofKSource
          (endpointViews shape constraintPolynomial priorAbsorbed
            publicRingColumns verifierRows publicFits |>.outputYRing
              source matrix lane)
          .c1]).flatten).flatten).flatten

/-- Raw output old-point sources in source, lane, low/high order. -/
noncomputable def outputYZcolSources
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    List
      (FieldSource
        (runningCodec shape publicRingColumns verifierRows publicFits)
        (freshCodec shape publicRingColumns verifierRows publicFits)
        (proofCodec shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits)) :=
  (List.ofFn fun source : Fin shape.sourceCount =>
    (List.ofFn fun lane : Fin ringDegree =>
      [proofKSource
        (endpointViews shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits |>.outputYZcol source lane)
        .c0,
       proofKSource
        (endpointViews shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits |>.outputYZcol source lane)
      .c1]).flatten).flatten

/-- Raw output sources, including the three framing words needed for a typed
and cursor-stable output domain. -/
noncomputable def outputSources
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    List
      (FieldSource
        (runningCodec shape publicRingColumns verifierRows publicFits)
        (freshCodec shape publicRingColumns verifierRows publicFits)
        (proofCodec shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits)) :=
  [.constant (framingField outputVersion),
   .constant (framingField shape.sourceCount),
   .constant (framingField shape.matrixCount)] ++
  outputYRingSources shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits ++
  outputYZcolSources shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits

@[simp] theorem runningSources_values
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    (runningSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).map
        (fun source => source.value running fresh proof) =
      (runningCodec
        shape publicRingColumns verifierRows publicFits).encode running := by
  simpa only [runningSources, List.map_ofFn, Function.comp_apply,
    FieldSource.value, physicalValues] using
    physicalValues_eq_encode
      (runningCodec shape publicRingColumns verifierRows publicFits) running

@[simp] theorem freshSources_values
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    (freshSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).map
        (fun source => source.value running fresh proof) =
      (freshCodec
        shape publicRingColumns verifierRows publicFits).encode fresh := by
  simpa only [freshSources, List.map_ofFn, Function.comp_apply,
    FieldSource.value, physicalValues] using
    physicalValues_eq_encode
      (freshCodec shape publicRingColumns verifierRows publicFits) fresh

@[simp] theorem proofStatementSources_values
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    (proofStatementSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).map
        (fun source => (source.value running fresh proof).val) =
      pointFields proof.piCcsInput.priorPoint ++
        claimedYRingFields proof.piCcsInput.claimedYRing := by
  unfold proofStatementSources pointFields claimedYRingFields
  rw [List.map_append]
  congr 1
  · rw [List.map_flatten, List.map_ofFn]
    apply congrArg List.flatten
    apply congrArg List.ofFn
    funext coordinate
    simp [Function.comp_apply, proofKSource, FieldSource.value, kFields,
      ConcreteNifsOperationalFrame.priorPointCoordinate,
      ConcreteNifsCarrierViews.pointCoordinate,
      KComponent.value, KComponent.view]
  · rw [List.map_flatten, List.map_ofFn]
    apply congrArg List.flatten
    apply congrArg List.ofFn
    funext runningIndex
    simp only [Function.comp_apply]
    rw [List.map_flatten, List.map_ofFn]
    apply congrArg List.flatten
    apply congrArg List.ofFn
    funext matrix
    simp only [Function.comp_apply]
    rw [List.map_flatten, List.map_ofFn]
    apply congrArg List.flatten
    apply congrArg List.ofFn
    funext lane
    simp [Function.comp_apply, proofKSource, FieldSource.value, kFields,
      ConcreteNifsOperationalFrame.claimedYRingCoordinate,
      KComponent.value, KComponent.view]

@[simp] theorem outputYRingSources_values
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    (outputYRingSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).map
        (fun source => (source.value running fresh proof).val) =
      (List.ofFn fun source : Fin shape.sourceCount =>
        (List.ofFn fun matrix : Fin shape.matrixCount =>
          (List.ofFn fun lane : Fin ringDegree =>
            kFields
              (proof.certificate.piCcs.output.yRing source matrix lane)
            ).flatten).flatten).flatten := by
  unfold outputYRingSources
  rw [List.map_flatten, List.map_ofFn]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext source
  simp only [Function.comp_apply]
  rw [List.map_flatten, List.map_ofFn]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext matrix
  simp only [Function.comp_apply]
  rw [List.map_flatten, List.map_ofFn]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext lane
  simp [Function.comp_apply, proofKSource, FieldSource.value, kFields,
    ConcreteNifsOperationalFrame.outputYRingCoordinate,
    KComponent.value, KComponent.view]

@[simp] theorem outputYZcolSources_values
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    (outputYZcolSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).map
        (fun source => (source.value running fresh proof).val) =
      (List.ofFn fun source : Fin shape.sourceCount =>
        (List.ofFn fun lane : Fin ringDegree =>
          kFields (proof.certificate.piCcs.output.yZcol source lane)
          ).flatten).flatten := by
  unfold outputYZcolSources
  rw [List.map_flatten, List.map_ofFn]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext source
  simp only [Function.comp_apply]
  rw [List.map_flatten, List.map_ofFn]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext lane
  simp [Function.comp_apply, proofKSource, FieldSource.value, kFields,
    ConcreteNifsOperationalFrame.outputYZcolCoordinate,
    KComponent.value, KComponent.view]

@[simp] theorem outputSources_values
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    (outputSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).map
        (fun source => (source.value running fresh proof).val) =
      rawOutputFields proof.certificate.piCcs.output := by
  unfold outputSources rawOutputFields
  rw [List.map_append, List.map_append,
    outputYRingSources_values shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits running fresh proof,
    outputYZcolSources_values shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits running fresh proof]
  rfl

@[simp] theorem running_projection_eq
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (system :
      Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    {
      parent := parentPayload (running.parent.materialize system)
      children := fun child =>
        runningPayload ((running.children child).materialize system)
    } = running := by
  cases running
  rfl

@[simp] theorem fresh_projection_eq
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (system :
      Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows) :
    freshPayload (fresh.materialize system) = fresh := by
  cases fresh
  rfl

/-- The canonical source list is exactly the selected statement
serialization after the verifier-owned context is materialized. -/
theorem statementSources_values
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    (statementSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).map
        (fun source => (source.value running fresh proof).val) =
      (serialization shape publicRingColumns verifierRows publicFits
        ).statementFields
        (ConcreteNifsParameters.context key running fresh proof
          ).materialize.piCcsStatement := by
  have runningValues :
      (runningSources shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits).map
          (fun source => (source.value running fresh proof).val) =
        ((runningCodec shape publicRingColumns verifierRows publicFits).encode
          running).map Fin.val := by
    simpa only [List.map_map, Function.comp_apply] using
      congrArg (List.map Fin.val)
        (runningSources_values shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits running fresh proof)
  have freshValues :
      (freshSources shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits).map
          (fun source => (source.value running fresh proof).val) =
        ((freshCodec shape publicRingColumns verifierRows publicFits).encode
          fresh).map Fin.val := by
    simpa only [List.map_map, Function.comp_apply] using
      congrArg (List.map Fin.val)
        (freshSources_values shape constraintPolynomial priorAbsorbed
          publicRingColumns verifierRows publicFits running fresh proof)
  simp only [statementSources, List.map_append, List.map_cons, List.map_nil]
  rw [runningValues, freshValues,
    proofStatementSources_values shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits running fresh proof]
  have oneModulus : 1 % goldilocksModulus = 1 := by decide
  simp [serialization, dynamicStatementFields,
    statementRunningFields, statementFreshFields,
    ConcreteNifsParameters.context,
    FixedActive.Canonical.Context.materialize,
    FixedActive.Canonical.Input.materialize,
    Context.piCcsStatement, fresh_projection_eq,
    statementVersion, framingField, NumericRowBridge.residue, oneModulus,
    FieldSource.value]
  exact congrArg
    (fun value =>
      ((runningCodec shape publicRingColumns verifierRows publicFits).encode
        value).map Fin.val)
    (running_projection_eq key.system running).symm

private theorem sum_ofFn_constant
    (count value : Nat) :
    (List.ofFn fun _ : Fin count => value).sum = count * value := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ, List.sum_cons, inductionHypothesis, Nat.succ_mul]
      omega

private theorem flatten_ofFn_length
    {Alpha : Type}
    {count width : Nat}
    (blocks : Fin count → List Alpha)
    (blockLength : ∀ index, (blocks index).length = width) :
    (List.ofFn blocks).flatten.length = count * width := by
  rw [List.length_flatten, List.map_ofFn]
  have lengths :
      List.ofFn (List.length ∘ blocks) =
        List.ofFn (fun _ : Fin count => width) := by
    apply congrArg List.ofFn
    funext index
    exact blockLength index
  rw [lengths, sum_ofFn_constant]

@[simp] theorem proofStatementSources_length
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (proofStatementSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).length =
      shape.rowVariables * 2 +
        shape.runningCount * shape.matrixCount * ringDegree * 2 := by
  unfold proofStatementSources
  rw [List.length_append]
  congr 1
  · apply flatten_ofFn_length
    intro coordinate
    rfl
  · calc
      (List.ofFn fun running : Fin shape.runningCount =>
          (List.ofFn fun matrix : Fin shape.matrixCount =>
            (List.ofFn fun lane : Fin ringDegree =>
              [proofKSource
                (endpointViews shape constraintPolynomial priorAbsorbed
                  publicRingColumns verifierRows publicFits |>.claimedYRing
                    running matrix lane)
                .c0,
               proofKSource
                (endpointViews shape constraintPolynomial priorAbsorbed
                  publicRingColumns verifierRows publicFits |>.claimedYRing
                    running matrix lane)
                .c1]).flatten).flatten).flatten.length =
          shape.runningCount * (shape.matrixCount * (ringDegree * 2)) := by
        apply flatten_ofFn_length
        intro running
        apply flatten_ofFn_length
        intro matrix
        apply flatten_ofFn_length
        intro lane
        rfl
      _ = shape.runningCount * shape.matrixCount * ringDegree * 2 := by
        simp only [Nat.mul_assoc]

@[simp] theorem statementSources_length
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (statementSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).length =
      10 +
        (runningCodec shape publicRingColumns verifierRows publicFits).width +
        (freshCodec shape publicRingColumns verifierRows publicFits).width +
        shape.rowVariables * 2 +
        shape.runningCount * shape.matrixCount * ringDegree * 2 := by
  simp [statementSources, runningSources, freshSources]
  omega

@[simp] theorem outputYRingSources_length
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (outputYRingSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).length =
      shape.sourceCount * shape.matrixCount * ringDegree * 2 := by
  unfold outputYRingSources
  calc
    (List.ofFn fun source : Fin shape.sourceCount =>
        (List.ofFn fun matrix : Fin shape.matrixCount =>
          (List.ofFn fun lane : Fin ringDegree =>
            [proofKSource
              (endpointViews shape constraintPolynomial priorAbsorbed
                publicRingColumns verifierRows publicFits |>.outputYRing
                  source matrix lane)
              .c0,
             proofKSource
              (endpointViews shape constraintPolynomial priorAbsorbed
                publicRingColumns verifierRows publicFits |>.outputYRing
                  source matrix lane)
              .c1]).flatten).flatten).flatten.length =
        shape.sourceCount * (shape.matrixCount * (ringDegree * 2)) := by
      apply flatten_ofFn_length
      intro source
      apply flatten_ofFn_length
      intro matrix
      apply flatten_ofFn_length
      intro lane
      rfl
    _ = shape.sourceCount * shape.matrixCount * ringDegree * 2 := by
      simp only [Nat.mul_assoc]

@[simp] theorem outputYZcolSources_length
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (outputYZcolSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).length =
      shape.sourceCount * ringDegree * 2 := by
  unfold outputYZcolSources
  calc
    (List.ofFn fun source : Fin shape.sourceCount =>
        (List.ofFn fun lane : Fin ringDegree =>
          [proofKSource
            (endpointViews shape constraintPolynomial priorAbsorbed
              publicRingColumns verifierRows publicFits |>.outputYZcol
                source lane)
            .c0,
           proofKSource
            (endpointViews shape constraintPolynomial priorAbsorbed
              publicRingColumns verifierRows publicFits |>.outputYZcol
                source lane)
              .c1]).flatten).flatten.length =
        shape.sourceCount * (ringDegree * 2) := by
      apply flatten_ofFn_length
      intro source
      apply flatten_ofFn_length
      intro lane
      rfl
    _ = shape.sourceCount * ringDegree * 2 := by
      simp only [Nat.mul_assoc]

theorem outputSources_length
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (outputSources shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits).length =
      3 +
        shape.sourceCount * shape.matrixCount * ringDegree * 2 +
        shape.sourceCount * ringDegree * 2 := by
  simp [outputSources]
  omega

/-- The three-word output frame and the even 54-lane extension products
leave the overwrite-duplex cursor at one for every selected shape. -/
theorem outputCursorOne
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    SymbolicDuplexCursor.after 0
        (2 +
          (outputSources shape constraintPolynomial priorAbsorbed
            publicRingColumns verifierRows publicFits).length) =
      1 := by
  rw [outputSources_length]
  have countShape :
      2 +
          (3 +
            shape.sourceCount * shape.matrixCount * ringDegree * 2 +
            shape.sourceCount * ringDegree * 2) =
        5 +
          (shape.sourceCount * shape.matrixCount * ringDegree * 2 +
            shape.sourceCount * ringDegree * 2) := by
    omega
  rw [countShape]
  have multiple :
      shape.sourceCount * shape.matrixCount * ringDegree * 2 +
          shape.sourceCount * ringDegree * 2 =
        4 *
          (shape.sourceCount * shape.matrixCount * 27 +
            shape.sourceCount * 27) := by
    simp [ringDegree]
    omega
  rw [multiple]
  rw [show
      5 +
          4 *
            (shape.sourceCount * shape.matrixCount * 27 +
              shape.sourceCount * 27) =
        1 +
          4 *
            (1 +
              (shape.sourceCount * shape.matrixCount * 27 +
                shape.sourceCount * 27)) by omega,
    SymbolicDuplexCursor.after_add]
  change
    SymbolicDuplexCursor.after 1
        (4 *
          (1 +
            (shape.sourceCount * shape.matrixCount * 27 +
              shape.sourceCount * 27))) =
      1
  exact SymbolicDuplexCursor.after_one_four_mul _

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalSerialization
