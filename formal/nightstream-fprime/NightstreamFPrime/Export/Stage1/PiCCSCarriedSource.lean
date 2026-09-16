import NightstreamFPrime.Export.Stage1.PiCCSAggregatedImagesPreservation
import NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout

/-! Proof-only source values for the two carried accumulators. Matrix and Pad
values are projections of the checked original-source endpoint constructor.
The scalar fold ordering and worker merge belong to the complete-sum proof. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedSource

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout
open PiRLCPartialTrace
open FiniteSumAlgebra (sumMap)

open PiCCSAggregatedImages
  (selectedRelation selectedProgram selectedSource selectedLayout selectedMessage)

private abbrev powers (gamma : K) := TargetPolynomial.power extensionOps.toOps gamma
private abbrev basis (gamma : K) :=
  PiCCSAggregatedImages.prepare (PiDECParentSparseRead.prepare ()) (powers gamma)
private abbrev blocks
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) := PiCCSAggregatedImages.combinedBlock (powers gamma) witness.assignments

/-- The same slot sum used by replayCarriedMatrix, before its global shift. -/
def slotTotal (gamma : K) (values : Vector K ProductionRelation.matrixCount) : K :=
  sumMap extensionOps (canonicalFinIndices productionShape.matrixCount) fun slot =>
    extensionOps.mul (powers gamma (productionShape.runningCount * slot.val)) (values.get slot)

/-- The numeric matrix lookup has the same zero suffix as the endpoint. -/
def matrixAt?
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (vertex : BooleanVertex cubeVariables) : Option (Vector K ProductionRelation.matrixCount) :=
  if NumericBooleanDomain.index vertex < selectedProgram.rowCount then
    PiCCSLinearRows.row? selectedProgram (columns := PiCCSSourceImages.logicalWidth) selectedSource
      (PiCCSCarriedRead.read (basis gamma).2 (blocks witness gamma)) (NumericBooleanDomain.index vertex)
  else some (Vector.replicate ProductionRelation.matrixCount K.zero)

private def padRead {columns : Nat} (layout : ColumnLayout cubeVariables columns)
    (prepared : FixedArray (Vector K ringDegree) ringDegree)
    (blockValues : Nat → Vector K ringDegree) (vertex : BooleanVertex cubeVariables) : K :=
  match layout.toColumn? vertex with
  | some column => PiCCSCarriedRead.read prepared blockValues column
  | none => K.zero

private theorem padRead_toVertex {columns : Nat} (layout : ColumnLayout cubeVariables columns)
    (prepared : FixedArray (Vector K ringDegree) ringDegree)
    (blockValues : Nat → Vector K ringDegree) (column : Fin columns) :
    padRead layout prepared blockValues (layout.toVertex column) =
      PiCCSCarriedRead.read prepared blockValues column := by
  rw [padRead, layout.toColumn_toVertex]

private theorem padRead_none {columns : Nat} (layout : ColumnLayout cubeVariables columns)
    (prepared : FixedArray (Vector K ringDegree) ringDegree)
    (blockValues : Nat → Vector K ringDegree) (vertex : BooleanVertex cubeVariables)
    (outside : layout.toColumn? vertex = none) :
    padRead layout prepared blockValues vertex = K.zero := by
  rw [padRead, outside]

/-- The decoded full-carrier Pad value, before its suffix weight. -/
def padAt
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (vertex : BooleanVertex cubeVariables) : K :=
  padRead selectedLayout (basis gamma).1 (blocks witness gamma) vertex

noncomputable def originalPad (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (vertex : BooleanVertex cubeVariables) : K :=
  sumMap extensionOps (canonicalPadCoordinates productionShape) fun coordinate =>
    extensionOps.mul (powers gamma coordinate.localGammaExponent)
      ((selectedMessage input witness vertex).padImage coordinate)

noncomputable def originalMatrix (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (vertex : BooleanVertex cubeVariables) : K :=
  sumMap extensionOps (canonicalMatrixCoordinates productionShape) fun coordinate =>
    extensionOps.mul (powers gamma coordinate.localGammaExponent)
      ((selectedMessage input witness vertex).matrixImage coordinate)

private theorem join_projections {Fresh Values Message : Type}
    (fresh : Option Fresh) (values : Option Values) (makeMessage : Fresh → Message)
    (pad : K) (total : Values → K) (wanted : Message) (wantedPad wantedMatrix : K)
    (returned : (do
      let fresh ← fresh
      let values ← values
      pure (makeMessage fresh, pad, total values)) = some (wanted, wantedPad, wantedMatrix)) :
    pad = wantedPad ∧ values.map total = some wantedMatrix := by
  cases fresh with
  | none => cases returned
  | some fresh =>
      cases values with
      | none => cases returned
      | some values =>
          have same : (makeMessage fresh, pad, total values) = (wanted, wantedPad, wantedMatrix) :=
            Option.some.inj returned
          exact ⟨congrArg (fun value => value.2.1) same,
            congrArg some (congrArg (fun value => value.2.2) same)⟩

private theorem fromRows_value
    (layout : ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (assignments : Fin productionShape.sourceCount → Phi81Relation.Assignment PiCCSSourceImages.shape)
    (padBasis : FixedArray (Vector K ringDegree) ringDegree)
    (blockValues : Nat → Vector K ringDegree) (power : Nat → K)
    (vertex : BooleanVertex cubeVariables)
    (fresh : Vector F ProductionRelation.matrixCount)
    (matrices : Vector K ProductionRelation.matrixCount) :
    PiCCSAggregatedImages.fromRows layout assignments padBasis blockValues power vertex fresh matrices =
      (PiCCSAggregatedImages.nonlinearMessage layout assignments vertex fresh,
       padRead layout padBasis blockValues vertex,
       sumMap extensionOps (canonicalFinIndices productionShape.matrixCount) (fun slot =>
         extensionOps.mul (power (productionShape.runningCount * slot.val)) (matrices.get slot))) := by
  cases decoded : layout.toColumn? vertex with
  | none => simp only [PiCCSAggregatedImages.fromRows, padRead, decoded]
  | some column => simp only [PiCCSAggregatedImages.fromRows, padRead, decoded]

private theorem endpoint_projections_generic
    (program : MatrixProgram.Program) (sourceRow : Nat → Option R1CS.Row)
    (layout : ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (assignments : Fin productionShape.sourceCount → Phi81Relation.Assignment PiCCSSourceImages.shape)
    (padBasis matrixBasis : FixedArray (Vector K ringDegree) ringDegree)
    (blockValues : Nat → Vector K ringDegree) (power : Nat → K)
    (vertex : BooleanVertex cubeVariables) (wanted : ProtocolPolynomial.OutputMessage K productionShape)
    (wantedPad wantedMatrix : K)
    (returned : PiCCSAggregatedImages.endpoint? program sourceRow layout assignments
      padBasis matrixBasis blockValues power vertex = some (wanted, wantedPad, wantedMatrix)) :
    padRead layout padBasis blockValues vertex = wantedPad ∧
    (if NumericBooleanDomain.index vertex < program.rowCount then
      PiCCSLinearRows.row? program (columns := PiCCSSourceImages.logicalWidth) sourceRow
        (PiCCSCarriedRead.read matrixBasis blockValues) (NumericBooleanDomain.index vertex)
     else some (Vector.replicate ProductionRelation.matrixCount K.zero)).map
      (fun values => sumMap extensionOps (canonicalFinIndices productionShape.matrixCount)
        (fun slot => extensionOps.mul (power (productionShape.runningCount * slot.val)) (values.get slot))) =
      some wantedMatrix := by
  by_cases active : NumericBooleanDomain.index vertex < program.rowCount
  · simp only [PiCCSAggregatedImages.endpoint?, if_pos active, fromRows_value] at returned
    simp only [if_pos active]
    exact join_projections
      (PiCCSSourceImages.freshMatrixImage? program sourceRow
        (assignments (freshSourceIndex ⟨0, by decide⟩)) vertex)
      (PiCCSLinearRows.row? program (columns := PiCCSSourceImages.logicalWidth) sourceRow
        (PiCCSCarriedRead.read matrixBasis blockValues) (NumericBooleanDomain.index vertex))
      (fun fresh => PiCCSAggregatedImages.nonlinearMessage layout assignments vertex fresh)
      (padRead layout padBasis blockValues vertex)
      (fun values => sumMap extensionOps (canonicalFinIndices productionShape.matrixCount)
        (fun slot => extensionOps.mul (power (productionShape.runningCount * slot.val)) (values.get slot)))
      wanted wantedPad wantedMatrix returned
  · simp only [PiCCSAggregatedImages.endpoint?, if_neg active, fromRows_value] at returned
    simp only [if_neg active]
    exact join_projections
      (PiCCSSourceImages.freshMatrixImage? program sourceRow
        (assignments (freshSourceIndex ⟨0, by decide⟩)) vertex)
      (some (Vector.replicate ProductionRelation.matrixCount K.zero))
      (fun fresh => PiCCSAggregatedImages.nonlinearMessage layout assignments vertex fresh)
      (padRead layout padBasis blockValues vertex)
      (fun values => sumMap extensionOps (canonicalFinIndices productionShape.matrixCount)
        (fun slot => extensionOps.mul (power (productionShape.runningCount * slot.val)) (values.get slot)))
      wanted wantedPad wantedMatrix returned

private theorem endpoint_projections (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (vertex : BooleanVertex cubeVariables) :
    padAt witness gamma vertex = originalPad input witness gamma vertex ∧
      (matrixAt? witness gamma vertex).map (slotTotal gamma) =
        some (originalMatrix input witness gamma vertex) := by
  have original := PiCCSAggregatedImages.endpoint_sourceProtocolData input witness gamma vertex
  dsimp only [padAt, matrixAt?, slotTotal]
  exact endpoint_projections_generic selectedProgram selectedSource selectedLayout witness.assignments
    (basis gamma).1 (basis gamma).2 (blocks witness gamma) (powers gamma) vertex
    { (selectedMessage input witness vertex) with padImage := fun _ => K.zero, matrixImage := fun _ => K.zero }
    (originalPad input witness gamma vertex) (originalMatrix input witness gamma vertex) original

/-- Every numeric matrix slot sum is the original full local-gamma total.
The Option equation discharges successful loads without a validity premise. -/
theorem matrix_sourceProtocolData (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (vertex : BooleanVertex cubeVariables) :
    (matrixAt? witness gamma vertex).map (slotTotal gamma) =
      some (originalMatrix input witness gamma vertex) :=
  (endpoint_projections input witness gamma vertex).2

/-- Every decoded Pad value is the original full local-gamma total. -/
theorem pad_sourceProtocolData (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (vertex : BooleanVertex cubeVariables) :
    padAt witness gamma vertex = originalPad input witness gamma vertex :=
  (endpoint_projections input witness gamma vertex).1

private theorem fullShape_carrierWidth (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :
    (PaperAlgebra.FullShape logicalWidth publicFits).carrierWidth =
      Phi81CarrierLayout.carrierWidth logicalWidth := rfl

/-- Transport a selected block index to the exact carrier-column owner's
block type without evaluating the selected logical-width expression. -/
theorem blockCount_eq_authority : PiCCSSourceImages.blockCount =
    Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth PiCCSSourceImages.logicalWidth) := by
  exact congrArg Phi81ColumnLayout.blockCount
    (fullShape_carrierWidth PiCCSSourceImages.logicalWidth PiCCSSourceImages.publicFits)

private theorem read_carrierColumn {logicalWidth : Nat}
    (prepared : FixedArray (Vector K ringDegree) ringDegree) (blockValues : Nat → Vector K ringDegree)
    (block : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth)))
    (lane : Fin ringDegree) :
    PiCCSCarriedRead.read prepared blockValues
        (Phi81CarrierLayout.carrierColumn (logicalWidth := logicalWidth) block lane) =
      PiCCSWeightedBasis.dotK (prepared.get lane) (blockValues block.val).get := by
  have decoded := Phi81CarrierLayout.decode_carrierColumn (logicalWidth := logicalWidth) block lane
  have blockIndex :
      (Phi81CarrierLayout.carrierColumn (logicalWidth := logicalWidth) block lane).val / ringDegree = block.val :=
    congrArg (fun value => value.1.val) decoded
  have laneIndex :
      (⟨(Phi81CarrierLayout.carrierColumn (logicalWidth := logicalWidth) block lane).val % ringDegree,
        Nat.mod_lt _ (by decide)⟩ : Fin ringDegree) = lane := congrArg Prod.snd decoded
  simp only [PiCCSCarriedRead.read, blockIndex, laneIndex]

/-- The direct block/lane dot product in replayCarriedPad is exactly the
original Pad total at that full-carrier column. Running tails are included. -/
theorem pad_block_sourceProtocolData (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K)
    (block : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth PiCCSSourceImages.logicalWidth)))
    (lane : Fin ringDegree) :
    PiCCSWeightedBasis.dotK ((basis gamma).1.get lane) (blocks witness gamma block.val).get =
      originalPad input witness gamma (selectedLayout.toVertex
        (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth) block lane)) := by
  have original := pad_sourceProtocolData input witness gamma (selectedLayout.toVertex
    (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth) block lane))
  rw [padAt, padRead_toVertex] at original
  exact (read_carrierColumn (basis gamma).1 (blocks witness gamma) block lane).symm.trans original

private theorem weighted_zero_vector (count : Nat) (weight : Fin count → K) :
    sumMap extensionOps (canonicalFinIndices count) (fun slot =>
      extensionOps.mul (weight slot) ((Vector.replicate count K.zero).get slot)) = extensionOps.zero := by
  have zeroGet (slot : Fin count) : (Vector.replicate count K.zero).get slot = extensionOps.zero := by
    change (Vector.replicate count K.zero)[slot.val] = extensionOps.zero
    rw [Vector.getElem_replicate]
    rfl
  simp only [zeroGet, extensionLaws.mul_zero, FiniteSumAlgebra.sumMap_zero extensionOps extensionLaws]

/-- Matrix completion past the active program is zero for every assignment. -/
theorem originalMatrix_padding (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (vertex : BooleanVertex cubeVariables)
    (beyond : selectedProgram.rowCount ≤ NumericBooleanDomain.index vertex) :
    originalMatrix input witness gamma vertex = extensionOps.zero := by
  have original := matrix_sourceProtocolData input witness gamma vertex
  have outside : ¬ NumericBooleanDomain.index vertex < selectedProgram.rowCount := by omega
  rw [matrixAt?, if_neg outside, Option.map_some] at original
  have zeroTotal : slotTotal gamma (Vector.replicate ProductionRelation.matrixCount K.zero) =
      extensionOps.zero := by
    exact weighted_zero_vector ProductionRelation.matrixCount
      (fun slot => powers gamma (productionShape.runningCount * slot.val))
  exact (Option.some.inj original).symm.trans zeroTotal

private theorem relationLayout_none {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (vertex : BooleanVertex cubeVariables)
    (beyond : Phi81CarrierLayout.carrierWidth logicalWidth ≤ NumericBooleanDomain.index vertex) :
    (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation).cubeLayout.toColumn? vertex = none := by
  exact (Folding.PiCCS.CanonicalRowLayout.toColumn?_eq_none_iff cubeVariables
    (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits vertex).2 beyond

/-- Pad completion begins after the complete carrier, not after the matrix
row count. No retained tail coordinate is dropped. -/
theorem originalPad_padding (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (vertex : BooleanVertex cubeVariables)
    (beyond : PiCCSSourceImages.shape.carrierWidth ≤ NumericBooleanDomain.index vertex) :
    originalPad input witness gamma vertex = extensionOps.zero := by
  have width : PiCCSSourceImages.shape.carrierWidth =
      Phi81CarrierLayout.carrierWidth PiCCSSourceImages.logicalWidth :=
    fullShape_carrierWidth PiCCSSourceImages.logicalWidth PiCCSSourceImages.publicFits
  have decoded : selectedLayout.toColumn? vertex = none :=
    relationLayout_none selectedRelation vertex (by rw [← width]; exact beyond)
  have original := pad_sourceProtocolData input witness gamma vertex
  rw [padAt, padRead_none selectedLayout (basis gamma).1 (blocks witness gamma) vertex decoded] at original
  exact original.symm

end NightstreamFPrime.Export.Stage1.PiCCSCarriedSource
