import Nightstream.Implementation.NebulaV2.ProductionFullClaimCarrierLayoutFor
import Nightstream.Implementation.NebulaV2.ProductionProductPiCcsTypedBridgeFor

/-!
Contract: exponent-indexed full-claim carrier to the exact paper-NIFS public
input.

The first seventeen fields are verifier-owned constants. The following
running window has `83160 + 2 * rowVariables` direct field columns. The
mandatory bundle and CCS public value also use direct columns. The theorem
derives public serialization from one physical carrier placement.

No equality in this module assumes NIFS acceptance, transcript challenges,
F-prime continuity, or execution.

Assurance tier: exponent-indexed carrier-to-PiCCS refinement.

Emits constraints: zero for native aliases.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.NebulaV2.ProductionFullClaimNifsPublicCarrierFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptSemanticsFor
open Nightstream.Implementation.NebulaV2.ProductionFullClaimCarrierLayoutFor
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev LinComb := LinCombNormal.LinComb

def columnField (column : Nat) : LinComb := [(column, 1)]

def runningColumns {candidate rowVariables}
    (layout : Layout candidate rowVariables) : List Nat :=
  List.ofFn layout.runningColumn

def bundleColumns {candidate rowVariables}
    (layout : Layout candidate rowVariables) : List Nat :=
  List.ofFn layout.bundleColumn

def ccsPublicColumns {candidate rowVariables}
    (layout : Layout candidate rowVariables) : List Nat :=
  List.ofFn layout.ccsPublicColumn

noncomputable def fields
    (candidate : Id) (rowVariables : Nat)
    (fullShape : Phi81Relation.Shape) (degreeBound : Nat)
    (layout : Layout candidate rowVariables) : List LinComb :=
  (ProductionProductNifsPublicTranscript.fixedPrefix
      candidate fullShape degreeBound).map ProductPiCcsTranscriptRowsFor.word ++
    (runningColumns layout).map columnField ++
    (bundleColumns layout).map columnField ++
    (ccsPublicColumns layout).map columnField

theorem runningColumns_length
    {candidate rowVariables} (layout : Layout candidate rowVariables) :
    (runningColumns layout).length = runningFieldCoordinatesFor rowVariables := by
  rw [runningColumns, List.length_ofFn]

theorem bundleColumns_length
    {candidate rowVariables} (layout : Layout candidate rowVariables) :
    (bundleColumns layout).length = bundleFieldCoordinates := by
  rw [bundleColumns, List.length_ofFn]

theorem ccsPublicColumns_length
    {candidate rowVariables} (layout : Layout candidate rowVariables) :
    (ccsPublicColumns layout).length = 540 := by
  rw [ccsPublicColumns, List.length_ofFn]

theorem fields_length
    (candidate : Id) (rowVariables : Nat)
    (fullShape : Phi81Relation.Shape) (degreeBound : Nat)
    (layout : Layout candidate rowVariables) :
    (fields candidate rowVariables fullShape degreeBound layout).length =
      ProductPiCcsTranscriptRowsFor.publicFieldCount rowVariables := by
  rw [fields, List.length_append, List.length_append, List.length_append,
    List.length_map, List.length_map, List.length_map, List.length_map,
    ProductionProductNifsPublicTranscript.fixedPrefix_length,
    runningColumns_length, bundleColumns_length, ccsPublicColumns_length]
  simp [ProductPiCcsTranscriptRowsFor.publicFieldCount,
    runningFieldCoordinatesFor, ProductNifsCodec.runningFieldCountFor,
    bundleFieldCoordinates]

def PrefixCanonical
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (degreeBound : Nat) : Prop :=
  forall value,
    value ∈ ProductionProductNifsPublicTranscript.fixedPrefix
      candidate fullShape degreeBound -> value < goldilocksP

private theorem fieldValues_append
    (assignment : Nat -> Nat) (left right : List LinComb) :
    fieldValues assignment (left ++ right) =
      fieldValues assignment left ++ fieldValues assignment right := by
  simp [ProductPiCcsTranscriptSemanticsFor.fieldValues]

private theorem fieldValues_columnFields
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (columns : List Nat) :
    fieldValues assignment (columns.map columnField) =
      columns.map assignment := by
  induction columns with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change lcEval assignment (columnField head) ::
          fieldValues assignment (tail.map columnField) =
        assignment head :: tail.map assignment
      rw [inductionHypothesis]
      simp [columnField, lcEval, Nat.mod_eq_of_lt (canonical head)]

private theorem mapOfFn_assignment_eq_nativeValues
    {n : Nat} {assignment : Nat -> Nat}
    (columns : Fin n -> Nat) (values : List F)
    (lengthExact : values.length = n)
    (placed : forall coordinate,
      assignment (columns coordinate) =
        (values.get (Fin.cast lengthExact.symm coordinate)).val) :
    (List.ofFn columns).map assignment =
      ProductionProductNifsPublicTranscript.nativeValues values := by
  rw [List.map_ofFn]
  have functionsEqual :
      assignment ∘ columns =
        fun coordinate : Fin n =>
          (values.get (Fin.cast lengthExact.symm coordinate)).val := by
    funext coordinate
    exact placed coordinate
  rw [functionsEqual]
  have reindexed := List.ofFn_congr lengthExact
    (fun coordinate : Fin values.length => (values.get coordinate).val)
  have source :
      List.ofFn (fun coordinate : Fin values.length =>
        (values.get coordinate).val) =
        ProductionProductNifsPublicTranscript.nativeValues values := by
    calc
      _ = (List.ofFn (List.get values)).map Fin.val :=
        List.ofFn_comp' (List.get values) Fin.val
      _ = values.map Fin.val := congrArg (List.map Fin.val)
        (List.ofFn_get values)
      _ = _ := rfl
  exact reindexed.symm.trans source

private theorem mapOfFn_assignment_eq_values
    {n : Nat} {assignment : Nat -> Nat}
    (columns : Fin n -> Nat) (values : List Nat)
    (lengthExact : values.length = n)
    (placed : forall coordinate,
      assignment (columns coordinate) =
        values.get (Fin.cast lengthExact.symm coordinate)) :
    (List.ofFn columns).map assignment = values := by
  rw [List.map_ofFn]
  have functionsEqual :
      assignment ∘ columns =
        fun coordinate : Fin n =>
          values.get (Fin.cast lengthExact.symm coordinate) := by
    funext coordinate
    exact placed coordinate
  rw [functionsEqual]
  have reindexed := List.ofFn_congr lengthExact
    (fun coordinate : Fin values.length => values.get coordinate)
  rw [List.ofFn_get] at reindexed
  exact reindexed.symm

private theorem prefixValues
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {degreeBound : Nat} {assignment : Nat -> Nat}
    (one : assignment 0 = 1)
    (canonical : PrefixCanonical candidate fullShape degreeBound) :
    fieldValues assignment
        ((ProductionProductNifsPublicTranscript.fixedPrefix
          candidate fullShape degreeBound).map
            ProductPiCcsTranscriptRowsFor.word) =
      ProductionProductNifsPublicTranscript.fixedPrefix
        candidate fullShape degreeBound := by
  have evaluated := ProductPiCcsTranscriptSemantics.fieldValues_words
    assignment one
    (ProductionProductNifsPublicTranscript.fixedPrefix
      candidate fullShape degreeBound)
  calc
    _ = (ProductionProductNifsPublicTranscript.fixedPrefix
          candidate fullShape degreeBound).map
          (fun value => value % goldilocksP) := by
      simpa only [ProductPiCcsTranscriptSemanticsFor.fieldValues,
        ProductPiCcsTranscriptSemantics.fieldValues,
        ProductPiCcsTranscriptRowsFor.word] using evaluated
    _ = (ProductionProductNifsPublicTranscript.fixedPrefix
          candidate fullShape degreeBound).map id := by
      apply List.map_congr_left
      intro value member
      exact Nat.mod_eq_of_lt (canonical value member)
    _ = _ := List.map_id _

private theorem runningValues
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {layout : Layout candidate rowVariables} {assignment : Nat -> Nat}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    (canonical : forall column, assignment column < goldilocksP)
    (placed : Placed contract layout assignment value) :
    fieldValues assignment ((runningColumns layout).map columnField) =
      ProductionProductNifsPublicTranscript.nativeValues
        (ProductionFieldNativeFullClaim.runningFields value.recursiveState) := by
  rw [fieldValues_columnFields canonical]
  change (List.ofFn layout.runningColumn).map assignment =
    ProductionFieldNativeFullClaim.runningNativeValues value.recursiveState
  have fieldLength := ProductionFieldNativeFullClaim.runningFields_lengthFor
    contract.toShape value.recursiveState
  have nativeLength :
      (ProductionFieldNativeFullClaim.runningNativeValues
        value.recursiveState).length = runningFieldCoordinatesFor rowVariables := by
    rw [ProductionFieldNativeFullClaim.runningNativeValues, List.length_map,
      fieldLength]
    simp [runningFieldCoordinatesFor, ProductNifsCodec.runningFieldCountFor,
      contract.rowVariablesExact]
  exact mapOfFn_assignment_eq_values layout.runningColumn
    (ProductionFieldNativeFullClaim.runningNativeValues value.recursiveState)
    nativeLength placed.running

private theorem bundleValues
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {layout : Layout candidate rowVariables} {assignment : Nat -> Nat}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    (canonical : forall column, assignment column < goldilocksP)
    (placed : Placed contract layout assignment value) :
    fieldValues assignment ((bundleColumns layout).map columnField) =
      ProductionProductNifsPublicTranscript.nativeValues
        (ProductionFieldNativeFullClaim.bundleFields
          value.commitmentBundle) := by
  rw [fieldValues_columnFields canonical]
  exact mapOfFn_assignment_eq_nativeValues layout.bundleColumn
    (ProductionFieldNativeFullClaim.bundleFields value.commitmentBundle)
    (ProductionFieldNativeFullClaim.bundleFields_length value.commitmentBundle)
    placed.bundle

private theorem ccsPublicValues
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {layout : Layout candidate rowVariables} {assignment : Nat -> Nat}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    (canonical : forall column, assignment column < goldilocksP)
    (placed : Placed contract layout assignment value) :
    fieldValues assignment ((ccsPublicColumns layout).map columnField) =
      value.ccsPublic.val := by
  rw [fieldValues_columnFields canonical]
  exact mapOfFn_assignment_eq_values layout.ccsPublicColumn
    value.ccsPublic.val value.ccsPublic.property.1 placed.ccsPublic

/-- The physical carrier evaluates to the exact selected public frame. -/
theorem fieldValues_eq_publicNifsFields
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (degreeBound : Nat) {layout : Layout candidate rowVariables}
    {assignment : Nat -> Nat}
    (assignmentCanonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (prefixCanonical : PrefixCanonical candidate fullShape degreeBound)
    (value : ProductionFieldNativeFullClaim.Value candidate fullShape)
    (placed : Placed contract layout assignment value) :
    fieldValues assignment
        (fields candidate rowVariables fullShape degreeBound layout) =
      ProductionProductNifsPublicTranscript.publicNifsFields candidate
        degreeBound value.recursiveState
        (ProductionFieldNativeFullClaim.freshOfValue contract.toShape value) := by
  rw [ProductionProductNifsPublicTranscript.publicNifsFields_of_value
    contract.toShape]
  simp only [ProductionProductNifsPublicTranscript.frame,
    ProductionProductNifsPublicTranscript.blocks, List.flatten_cons,
    List.flatten_nil, List.append_nil]
  rw [fields, fieldValues_append, fieldValues_append, fieldValues_append]
  rw [prefixValues one prefixCanonical,
    runningValues assignmentCanonical placed,
    bundleValues assignmentCanonical placed,
    ccsPublicValues assignmentCanonical placed]
  simp only [List.append_assoc]

noncomputable def bindPublicFields
    (candidate : Id) (rowVariables : Nat)
    (fullShape : Phi81Relation.Shape) (degreeBound : Nat)
    (layout : Layout candidate rowVariables)
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables) :
    ProductionProductPiCcsTypedBridgeFor.Wires rowVariables :=
  { wires with
    publicNifsFields := fields candidate rowVariables fullShape degreeBound layout
    publicNifsFields_length :=
      fields_length candidate rowVariables fullShape degreeBound layout }

/-- Placement of all PiCCS fields except public serialization. -/
structure RemainingPlacement
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
    (assignment : Nat -> Nat) : Prop where
  statementSerialization :
    fieldValues assignment
        (ProductPiCcsTranscriptRowsFor.statementFields
          (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
            config artifact running fresh wires)) =
      ProductPoseidon2.statementFieldsFor rowVariables
        ({ priorState :=
            (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
              config artifact).publicInputState running fresh
           input := ProductionProductPiCcsTypedBridgeFor.exactVerifierInput
             candidate statementId config artifact running fresh } :
          ProductPiCcsTypedReplayFor.PaperStatement rowVariables)
  roundSerialization : forall round,
    fieldValues assignment
        (ProductPiCcsTranscriptRowsFor.roundFields round.val
          (wires.rounds round)) =
      ProductPoseidon2.roundFieldsFor round
        ((proof.piCcsRounds round).toMessage)
  outputSerialization :
    fieldValues assignment
        (ProductPiCcsTranscriptRowsFor.fullOutputFields
          (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
            config artifact running fresh wires)) =
      ProductPoseidon2.outputFieldsFor rowVariables proof.piCcsOutput
  priorPoint :
    (KPiCcsOccurrence.decodedVerifierInput
      (ProductPiCcsTranscriptRowsFor.occurrenceInput
        (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
          config artifact running fresh wires)) assignment).priorPoint =
      (ProductionProductPiCcsTypedBridgeFor.exactVerifierInput candidate
        statementId config artifact running fresh).priorPoint
  claimedCoefficient : forall coordinate,
    (KPiCcsOccurrence.decodedVerifierInput
      (ProductPiCcsTranscriptRowsFor.occurrenceInput
        (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
          config artifact running fresh wires)) assignment).claimedCoefficient
        coordinate =
      (ProductionProductPiCcsTypedBridgeFor.exactVerifierInput candidate
        statementId config artifact running fresh).claimedCoefficient coordinate
  roundPolynomial : forall round,
    KPiCcsOccurrence.decodedRound (wires.rounds round) assignment =
      proof.piCcsRounds round
  fullOutputCoordinate : forall source matrix coefficient,
    ProductionProductPiCcsTypedBridgeFor.decodeK assignment
        (wires.fullOutput source matrix coefficient) =
      proof.piCcsOutput.coordinate source matrix coefficient

/-- Carrier placement supplies the authority-bearing public serialization. -/
theorem piCcsPlacement
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    {layout : Layout candidate rowVariables} {assignment : Nat -> Nat}
    (assignmentCanonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (prefixCanonical : PrefixCanonical candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits) 9)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (placed : Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) layout assignment value)
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (baseWires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (remaining : RemainingPlacement candidate statementId config artifact
      value.recursiveState
      (ProductionFieldNativeFullClaim.freshOfValue
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits).toShape value)
      proof (bindPublicFields candidate rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)
        9 layout baseWires) assignment) :
    ProductionProductPiCcsTypedBridgeFor.Placement candidate statementId config
      artifact value.recursiveState
      (ProductionFieldNativeFullClaim.freshOfValue
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits).toShape value)
      proof (bindPublicFields candidate rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)
        9 layout baseWires) assignment where
  publicSerialization := fieldValues_eq_publicNifsFields
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits) 9 assignmentCanonical one prefixCanonical value placed
  statementSerialization := remaining.statementSerialization
  roundSerialization := remaining.roundSerialization
  outputSerialization := remaining.outputSerialization
  priorPoint := remaining.priorPoint
  claimedCoefficient := remaining.claimedCoefficient
  roundPolynomial := remaining.roundPolynomial
  fullOutputCoordinate := remaining.fullOutputCoordinate

end Nightstream.Implementation.NebulaV2.ProductionFullClaimNifsPublicCarrierFor
