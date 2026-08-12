import Nightstream.Implementation.NebulaV2.ProductPiDecPublicSplitRows
import Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridgeFor
import Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor
import Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplayFor
import Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSoundFor
import Nightstream.Implementation.NebulaV2.ProductionProductNifsPaperRowsSoundFor
import Nightstream.Implementation.R1CS.Canonical.KEquality

/-!
Contract: exact field-native output carrier for one exponent-indexed paper
NIFS verifier call.

The rows copy the transcript-derived SumCheck point into the output carrier
and compute all fourteen public-input children from the PiRLC parent. Static
aliases place the PiDEC child commitments and evaluations into the same
carrier. The soundness theorem covers every flat carrier coordinate.

`Sources` is a named upstream ABI boundary. It contains only point, parent
public-input, PiDEC child commitment, and PiDEC child evaluation facts. A
separate theorem must derive it from the complete PiCCS/PiRLC/PiDEC rows.

Assurance tier: exponent-indexed row implementation.

Does not own the upstream NIFS section proof, complete recursive manifest,
Rust refinement, terminal verification, or cryptographic soundness.

Emits constraints: `2 * rowVariables + 23,760` R1CS rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.NebulaV2.ProductionProductNifsOutputRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Protocol.NebulaV2.CommitmentBundle
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

abbrev Running
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductNifsCodec.RunningFor rowVariables
    (FullShape rowVariables logicalWidth publicFits)

/-- Output carrier plus the auxiliary public-split columns. -/
structure Layout (rowVariables : Nat) where
  carrierColumn : Fin (ProductNifsCodec.runningFieldCountFor rowVariables) -> Nat
  publicSplit : ProductPiDecPublicSplitRows.Layout

def pointCarrier {rowVariables : Nat} (layout : Layout rowVariables)
    (coordinate : Fin rowVariables) : KMul.Carried where
  low := [(layout.carrierColumn
    ⟨pointCoordinateIndex coordinate 0,
      point_coordinate_bound coordinate 0⟩, 1)]
  high := [(layout.carrierColumn
    ⟨pointCoordinateIndex coordinate 1,
      point_coordinate_bound coordinate 1⟩, 1)]

def pointRows {rowVariables : Nat} (layout : Layout rowVariables)
    (point : Fin rowVariables -> KMul.Carried) : List Row :=
  (ProductPiDecRows.indices rowVariables).flatMap fun coordinate =>
    KEquality.rows (point coordinate) (pointCarrier layout coordinate)

def rows {rowVariables : Nat} (layout : Layout rowVariables)
    (point : Fin rowVariables -> KMul.Carried) : List Row :=
  pointRows layout point ++ ProductPiDecPublicSplitRows.rows layout.publicSplit

private theorem length_flatMap_uniform
    {Alpha Beta : Type} (items : List Alpha) (values : Alpha -> List Beta)
    (count : Nat) (uniform : forall item, (values item).length = count) :
    (items.flatMap values).length = items.length * count := by
  induction items with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp [uniform, inductionHypothesis, Nat.add_mul, Nat.add_comm]

theorem pointRows_length
    {rowVariables : Nat} (layout : Layout rowVariables)
    (point : Fin rowVariables -> KMul.Carried) :
    (pointRows layout point).length = rowVariables * 2 := by
  rw [pointRows, length_flatMap_uniform _ _ 2]
  · simp
  · intro coordinate
    exact KEquality.rows_length _ _

theorem rows_length
    {rowVariables : Nat} (layout : Layout rowVariables)
    (point : Fin rowVariables -> KMul.Carried) :
    (rows layout point).length = rowVariables * 2 + 23760 := by
  simp [rows, pointRows_length, ProductPiDecPublicSplitRows.rows_length]

def fullPublicColumn
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (column : Fin 540) : Fin fullShape.publicWidth :=
  ⟨column.val, by
    rw [contract.publicWidth]
    simpa only [MemoryBoundCcsPublic.coordinateCount] using column.isLt⟩

def publicBlock (column : Fin 540) : Fin 10 :=
  ⟨column.val / ringDegree, by
    have columnLt := column.isLt
    change column.val / 54 < 10
    omega⟩

def publicLane (column : Fin 540) : Fin ringDegree :=
  ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩

def pointIndex {rowVariables : Nat}
    (coordinate : Fin rowVariables) (limb : Fin 2) :
    Fin (ProductNifsCodec.runningFieldCountFor rowVariables) :=
  ⟨pointCoordinateIndex coordinate limb,
    point_coordinate_bound coordinate limb⟩

def commitmentIndex {rowVariables : Nat}
    (child : Fin 14) (component : Fin 4)
    (row : Fin ProductCommitmentAlgebra.Rank) (lane : Fin ringDegree) :
    Fin (ProductNifsCodec.runningFieldCountFor rowVariables) :=
  ⟨commitmentCoordinateIndex child (componentAt component) row lane,
    commitment_coordinate_bound child (componentAt component) row lane⟩

def publicIndex
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (child : Fin 14) (column : Fin 540) :
    Fin (ProductNifsCodec.runningFieldCountFor rowVariables) :=
  ⟨publicInputCoordinateIndex child (fullPublicColumn contract column),
    public_input_coordinate_bound contract child
      (fullPublicColumn contract column)⟩

def evaluationIndex
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (child matrix : Fin 14) (lane : Fin ringDegree) (limb : Fin 2) :
    Fin (ProductNifsCodec.runningFieldCountFor rowVariables) :=
  ⟨evaluationCoordinateIndex (fullShape := fullShape) child matrix lane limb,
    evaluation_coordinate_bound contract child matrix lane limb⟩

@[simp] theorem runningCoordinate_point_index
    {rowVariables : Nat} (coordinate : Fin rowVariables) (limb : Fin 2) :
    (RunningCoordinate.point coordinate limb).index =
      pointIndex coordinate limb := by
  apply Fin.ext
  simp [RunningCoordinate.index, RunningCoordinate.indexNat, pointIndex,
    pointCoordinateIndex, pointOffset]

@[simp] theorem runningCoordinate_commitment_index
    {rowVariables : Nat} (child : Fin 14) (component : Fin 4)
    (row : Fin ProductCommitmentAlgebra.Rank) (lane : Fin ringDegree) :
    (RunningCoordinate.commitment (rowVariables := rowVariables)
        child component row lane).index =
      commitmentIndex (rowVariables := rowVariables)
        child component row lane := by
  apply Fin.ext
  simp [RunningCoordinate.index, RunningCoordinate.indexNat,
    commitmentIndex, commitmentCoordinateIndex,
    componentIndex_componentAt]

@[simp] theorem runningCoordinate_public_index
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (child : Fin 14) (column : Fin 540) :
    (RunningCoordinate.publicInput (rowVariables := rowVariables)
        child column).index =
      publicIndex contract child column := by
  apply Fin.ext
  simp [RunningCoordinate.index, RunningCoordinate.indexNat, publicIndex,
    publicInputCoordinateIndex, fullPublicColumn, contract.publicWidth,
    MemoryBoundCcsPublic.coordinateCount]

@[simp] theorem runningCoordinate_evaluation_index
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (child matrix : Fin 14) (lane : Fin ringDegree) (limb : Fin 2) :
    (RunningCoordinate.evaluation (rowVariables := rowVariables)
        child matrix lane limb).index =
      evaluationIndex contract child matrix lane limb := by
  apply Fin.ext
  simp [RunningCoordinate.index, RunningCoordinate.indexNat,
    evaluationIndex, evaluationCoordinateIndex, evaluationsOffset,
    publicInputsFieldCount, contract.publicWidth,
    MemoryBoundCcsPublic.coordinateCount]

/-- Static zero-copy layout links. These are column equalities, not witness
value or verifier-result assumptions. -/
structure Layout.Valid
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (layout : Layout rowVariables)
    (algebra : ProductPiRlcAlgebraRows.Layout)
    (piDec : ProductPiDecRows.Layout) : Prop where
  parentPublic : forall column : Fin 540,
    layout.publicSplit.parentColumn column =
      algebra.outputPublic (publicBlock column) (publicLane column)
  childPublic : forall child column,
    layout.publicSplit.childColumn child column =
      layout.carrierColumn (publicIndex contract child column)
  childCommitment : forall child component row lane,
    layout.carrierColumn (commitmentIndex child component row lane) =
      (piDec.childBundle child).column (componentAt component) row lane
  childEvaluation : forall child matrix lane limb,
    layout.carrierColumn
        (evaluationIndex contract child matrix lane limb) =
      (piDec.childEvaluation child).column matrix lane limb

/-- Exact upstream values consumed by the output serialization rows. Each
field is narrower than full output-carrier placement. -/
structure Sources
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (layout : Layout rowVariables)
    (point : Fin rowVariables -> KMul.Carried)
    (algebra : ProductPiRlcAlgebraRows.Layout)
    (piDec : ProductPiDecRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (parentInput : PublicInput fullShape)
    (running : ProductNifsCodec.RunningFor rowVariables fullShape) : Prop where
  point : forall coordinate,
    ofProjection (KFixedPhaseSumCheck.decodeCarried assignment
      (point coordinate)) =
      running.point.coordinates.getD coordinate.val K.zero
  parentCoordinate : forall column : Fin 540,
    ProductPiDecLinearCombination.fieldAt assignment canonical
        (algebra.outputPublic (publicBlock column) (publicLane column)) =
      parentInput (fullPublicColumn contract column)
  publicInput : forall child column,
    running.publicInputs child (fullPublicColumn contract column) =
      PiDECAlgebra.Radix.splitScalar
        (parentInput (fullPublicColumn contract column)) child
  commitment : forall child component row lane,
    ProductPiDecLinearCombination.fieldAt assignment canonical
        ((piDec.childBundle child).column (componentAt component) row lane) =
      running.commitments child (componentAt component) row lane
  evaluation : forall child matrix lane,
    let decoded := ProductPiDecTypedBridgeFor.decodeEvaluation rowVariables
      (piDec.childEvaluation child) assignment canonical
    decoded matrix lane = running.evaluations child matrix lane

structure Placed
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (layout : Layout rowVariables) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (running : ProductNifsCodec.RunningFor rowVariables fullShape) : Prop where
  coordinate : forall index,
    ProductPiDecLinearCombination.fieldAt assignment canonical
        (layout.carrierColumn index) =
      ((ProductNifsCodec.runningCodecFor rowVariables fullShape).encode
        running).getD index.val 0

/-- Natural-number form of one exact output-carrier coordinate. Canonical
decoding makes the field coordinate equal to the physical assignment value. -/
theorem Placed.assignment_coordinate
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    {running : ProductNifsCodec.RunningFor rowVariables fullShape}
    (placed : Placed layout assignment canonical running)
    (index : Fin (ProductNifsCodec.runningFieldCountFor rowVariables)) :
    assignment (layout.carrierColumn index) =
      (((ProductNifsCodec.runningCodecFor rowVariables fullShape).encode
        running).getD index.val 0).val := by
  exact congrArg Fin.val (placed.coordinate index)

private theorem point_rows_hold
    {rowVariables : Nat} {layout : Layout rowVariables}
    {point : Fin rowVariables -> KMul.Carried} {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout point) assignment)
    (coordinate : Fin rowVariables) :
    Satisfies (KEquality.rows (point coordinate)
      (pointCarrier layout coordinate)) assignment := by
  intro row member
  apply holds row
  apply List.mem_append_left
  exact List.mem_flatMap.mpr
    ⟨coordinate, ProductPiDecRows.index_mem coordinate, member⟩

private theorem public_rows_hold
    {rowVariables : Nat} {layout : Layout rowVariables}
    {point : Fin rowVariables -> KMul.Carried} {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout point) assignment) :
    Satisfies (ProductPiDecPublicSplitRows.rows layout.publicSplit)
      assignment := by
  intro row member
  exact holds row (List.mem_append_right _ member)

theorem point_fields_of_rows
    {rowVariables : Nat} {layout : Layout rowVariables}
    {point : Fin rowVariables -> KMul.Carried} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout point) assignment)
    (coordinate : Fin rowVariables) :
    let decoded := ofProjection
      (KFixedPhaseSumCheck.decodeCarried assignment (point coordinate))
    ProductPiDecLinearCombination.fieldAt assignment canonical
        (layout.carrierColumn (pointIndex coordinate 0)) = decoded.c0 /\
      ProductPiDecLinearCombination.fieldAt assignment canonical
        (layout.carrierColumn (pointIndex coordinate 1)) = decoded.c1 := by
  dsimp only
  have equalities := KEquality.rows_sound assignment (point coordinate)
    (pointCarrier layout coordinate) one (point_rows_hold holds coordinate)
  constructor <;> apply Fin.ext
  · simpa [ProductPiDecLinearCombination.fieldAt, pointCarrier,
      pointIndex, KFixedPhaseSumCheck.decodeCarried,
      KConcreteFixedPhaseBridge.ofProjection, lcEval,
      Nat.mod_eq_of_lt (canonical _)] using equalities.1.symm
  · simpa [ProductPiDecLinearCombination.fieldAt, pointCarrier,
      pointIndex, KFixedPhaseSumCheck.decodeCarried,
      KConcreteFixedPhaseBridge.ofProjection, lcEval,
      Nat.mod_eq_of_lt (canonical _)] using equalities.2.symm

/-- Satisfied output rows and exact upstream sources bind every field in the
flat running carrier. The proof uses the exhaustive coordinate theorem. -/
theorem rows_sound
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {layout : Layout rowVariables}
    {point : Fin rowVariables -> KMul.Carried}
    {algebra : ProductPiRlcAlgebraRows.Layout}
    {piDec : ProductPiDecRows.Layout}
    {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    {parentInput : PublicInput fullShape}
    {running : ProductNifsCodec.RunningFor rowVariables fullShape}
    (valid : layout.Valid contract algebra piDec)
    (sources : Sources contract layout point algebra piDec assignment canonical
      parentInput running)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout point) assignment) :
    Placed layout assignment canonical running := by
  constructor
  intro index
  rcases runningCoordinate_surjective index with ⟨coordinate, equal⟩
  subst index
  cases coordinate with
  | point coordinate limb =>
      have fields := point_fields_of_rows canonical one holds coordinate
      have encoded :=
        ProductNifsRunningCoordinatesFor.runningCodecFor_point_getD
          running coordinate limb
      fin_cases limb
      · rw [runningCoordinate_point_index]
        calc
          ProductPiDecLinearCombination.fieldAt assignment canonical
                (layout.carrierColumn (pointIndex coordinate 0)) =
              (ofProjection (KFixedPhaseSumCheck.decodeCarried assignment
                (point coordinate))).c0 := fields.1
          _ = (running.point.coordinates.getD coordinate.val K.zero).c0 :=
            congrArg K.c0 (sources.point coordinate)
          _ = ((ProductNifsCodec.runningCodecFor rowVariables fullShape).encode
                running).getD (pointIndex coordinate 0).val 0 := by
            symm
            simpa [pointIndex] using encoded
      · rw [runningCoordinate_point_index]
        calc
          ProductPiDecLinearCombination.fieldAt assignment canonical
                (layout.carrierColumn (pointIndex coordinate 1)) =
              (ofProjection (KFixedPhaseSumCheck.decodeCarried assignment
                (point coordinate))).c1 := fields.2
          _ = (running.point.coordinates.getD coordinate.val K.zero).c1 :=
            congrArg K.c1 (sources.point coordinate)
          _ = ((ProductNifsCodec.runningCodecFor rowVariables fullShape).encode
                running).getD (pointIndex coordinate 1).val 0 := by
            symm
            simpa [pointIndex] using encoded
  | commitment child component row lane =>
      rw [runningCoordinate_commitment_index,
        valid.childCommitment child component row lane]
      calc
        ProductPiDecLinearCombination.fieldAt assignment canonical
              ((piDec.childBundle child).column
                (componentAt component) row lane) =
            running.commitments child (componentAt component) row lane :=
          sources.commitment child component row lane
        _ = ((ProductNifsCodec.runningCodecFor rowVariables fullShape).encode
              running).getD
                (commitmentIndex (rowVariables := rowVariables)
                  child component row lane).val 0 := by
          symm
          simpa [commitmentIndex] using
            (ProductNifsRunningCoordinatesFor.runningCodecFor_commitment_getD
              running child (componentAt component) row lane)
  | publicInput child column =>
      rw [runningCoordinate_public_index contract,
        ← valid.childPublic child column]
      have split := ProductPiDecPublicSplitRows.rows_sound canonical one
        (public_rows_hold holds) column child
      calc
        ProductPiDecLinearCombination.fieldAt assignment canonical
              (layout.publicSplit.childColumn child column) =
            PiDECAlgebra.Radix.splitScalar
              (ProductPiDecLinearCombination.fieldAt assignment canonical
                (layout.publicSplit.parentColumn column)) child := split
        _ = PiDECAlgebra.Radix.splitScalar
              (parentInput (fullPublicColumn contract column)) child := by
          rw [valid.parentPublic column, sources.parentCoordinate column]
        _ = running.publicInputs child
              (fullPublicColumn contract column) :=
          (sources.publicInput child column).symm
        _ = ((ProductNifsCodec.runningCodecFor rowVariables fullShape).encode
              running).getD (publicIndex contract child column).val 0 := by
          symm
          simpa [publicIndex] using
            (ProductNifsRunningCoordinatesFor.runningCodecFor_publicInput_getD
              running child (fullPublicColumn contract column))
  | evaluation child matrix lane limb =>
      rw [runningCoordinate_evaluation_index contract,
        valid.childEvaluation child matrix lane limb]
      have exactEvaluation := sources.evaluation child matrix lane
      have encoded :=
        ProductNifsRunningCoordinatesFor.runningCodecFor_evaluation_getD
          running child matrix lane limb
      fin_cases limb
      · calc
          ProductPiDecLinearCombination.fieldAt assignment canonical
                ((piDec.childEvaluation child).column matrix lane 0) =
              (ProductPiDecTypedBridgeFor.decodeEvaluation rowVariables
                (piDec.childEvaluation child) assignment canonical
                matrix lane).c0 := rfl
          _ = (running.evaluations child matrix lane).c0 :=
            congrArg K.c0 exactEvaluation
          _ = ((ProductNifsCodec.runningCodecFor rowVariables fullShape).encode
                running).getD
                  (evaluationIndex contract child matrix lane 0).val 0 := by
            symm
            simpa [evaluationIndex] using encoded
      · calc
          ProductPiDecLinearCombination.fieldAt assignment canonical
                ((piDec.childEvaluation child).column matrix lane 1) =
              (ProductPiDecTypedBridgeFor.decodeEvaluation rowVariables
                (piDec.childEvaluation child) assignment canonical
                matrix lane).c1 := rfl
          _ = (running.evaluations child matrix lane).c1 :=
            congrArg K.c1 exactEvaluation
          _ = ((ProductNifsCodec.runningCodecFor rowVariables fullShape).encode
                running).getD
                  (evaluationIndex contract child matrix lane 1).val 0 := by
            symm
            simpa [evaluationIndex] using encoded

/-! ## Complete upstream derivation -/

/-- The exact row-visible SumCheck point that the NIFS output must carry. -/
noncomputable def verifierPoint
    (candidate : Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.Id)
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
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables) :
    Fin rowVariables -> KMul.Carried :=
  ProductPiCcsTranscriptRowsFor.pointAt
    (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId config
      artifact running fresh wires)

/-- The verifier-computed PiRLC parent public input. -/
noncomputable def verifierParentInput
    (candidate : Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.Id)
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
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) :
    PublicInput (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
      publicFits) :=
  (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId config
    artifact).parent running fresh proof |>.publicInput

/-- The output section owns its emitted rows and only static column aliases. -/
structure SectionRows
    (candidate : Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.Id)
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
    (algebra : ProductPiRlcAlgebraRows.Layout)
    (piDec : ProductPiDecRows.Layout)
    (assignment : Nat -> Nat) where
  layout : Layout rowVariables
  valid : layout.Valid
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits) algebra piDec
  satisfied : Satisfies
    (rows layout
      (verifierPoint candidate statementId config artifact running fresh wires))
    assignment

/-- PiCCS, sampler, PiRLC, and PiDEC placement facts derive the narrow source
interface. No source field is supplied independently of those rows. -/
theorem sources_of_nifs_rows
    (candidate : Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.Id)
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
    (samplerBase : Nat) (algebra : ProductPiRlcAlgebraRows.Layout)
    (piDec : ProductPiDecRows.Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (piCcsPlacement : ProductionProductPiCcsTypedBridgeFor.Placement candidate
      statementId config artifact running fresh proof wires assignment)
    (piCcsRows : Satisfies
      (ProductPiCcsTranscriptRowsFor.rows
        (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
          config artifact running fresh wires)) assignment)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (classificationRows : ProductPiRlcCandidateClassificationRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (selectorRows : ProductPiRlcFirstAcceptedBatchRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (algebraRows : Satisfies (ProductPiRlcAlgebraRows.rows algebra) assignment)
    (placement : ProductionProductNifsPaperRowsSoundFor.Placement candidate
      statementId config artifact running fresh proof wires samplerBase algebra
      piDec assignment canonical)
    (outputSection : SectionRows candidate statementId config artifact running fresh
      wires algebra piDec assignment) :
    Sources
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits)
      outputSection.layout
      (verifierPoint candidate statementId config artifact running fresh wires)
      algebra piDec assignment canonical
      (verifierParentInput candidate statementId config artifact running fresh
        proof)
      ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
        config artifact).output running fresh proof) := by
  let selected := ProductionProductPiCcsTypedBridgeFor.paperKey candidate
    statementId config artifact
  let input := ProductionProductPiCcsTypedBridgeFor.rowInput candidate
    statementId config artifact running fresh wires
  have coins := ProductionProductPiCcsTypedBridgeFor.decodedCoins_eq_executionCoins
    candidate statementId config artifact running fresh proof wires assignment
    canonical one piCcsPlacement piCcsRows
  have parentFields :=
    ProductionProductPiRlcParentBridgeFor.parentFields_of_rows candidate
      statementId config artifact running fresh proof wires samplerBase algebra
      assignment canonical one piCcsPlacement piCcsRows transcriptRows
      classificationRows selectorRows algebraRows placement.piRlc
  refine
    { point := ?_
      parentCoordinate := ?_
      publicInput := ?_
      commitment := ?_
      evaluation := ?_ }
  · intro coordinate
    have pointCoordinates := coins.2.2
    have selectedCoordinate := congrArg
      (fun values => values.getD coordinate.val K.zero) pointCoordinates
    have decodedCoordinate :
        ofProjection (KFixedPhaseSumCheck.decodeCarried assignment
          (verifierPoint candidate statementId config artifact running fresh
            wires coordinate)) =
          (KPiCcsOccurrence.decodedPoint
            (ProductPiCcsTranscriptRowsFor.occurrenceInput input) assignment
            ).coordinates.getD coordinate.val K.zero := by
      simp [verifierPoint, input, KPiCcsOccurrence.decodedPoint,
        KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedPoint,
        KPiCcsTerminal.alphaEqualityInput, KPointEquality.decodedLeft,
        KPointEquality.indices, KPointEquality.decoded,
        ProductPiCcsTranscriptRowsFor.occurrenceInput,
        ProductPiCcsTranscriptRowsFor.pointAt]
    have outputPoint :
        (selected.output running fresh proof).point =
          (selected.piCcsExecution running fresh proof).coins.roundPoint := by
      exact (Key.output_point selected running fresh proof).trans
        (Key.parent_point selected running fresh proof)
    exact decodedCoordinate.trans
      (selectedCoordinate.trans
        (congrArg (fun point => point.coordinates.getD coordinate.val K.zero)
          outputPoint).symm)
  · intro column
    have selectedCoordinate := congrArg
      (fun value => value (fullPublicColumn
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits) column)) parentFields.2.1
    change
      ProductPiDecLinearCombination.fieldAt assignment canonical
          (algebra.outputPublic (publicBlock column) (publicLane column)) =
        (selected.parent running fresh proof).publicInput
          (fullPublicColumn
            (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
              publicFits) column) at selectedCoordinate
    simpa only [verifierParentInput, selected] using selectedCoordinate
  · intro child column
    let targetColumn := fullPublicColumn
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) column
    let castChild := Fin.cast selected.runningCount_eq_outputCount child
    have outputSplit := congrFun
      (Key.output_publicInput selected running fresh proof child) targetColumn
    have selectedSplit := congrArg
      (fun split => split.split
        (selected.parent running fresh proof).publicInput castChild targetColumn)
      (ProductionProductPiCcsTypedBridgeFor.paperKey_piDecPublicInputSplit
        candidate statementId config artifact)
    have coordinateSplit :=
      ProductPaperAlgebraFor.publicInputSplit_coordinate config
        (selected.parent running fresh proof).publicInput castChild targetColumn
    have castChild_eq : castChild = child := by
      apply Fin.ext
      rfl
    calc
      (selected.output running fresh proof).publicInputs child targetColumn =
          selected.piDecPublicInputSplit.split
            (selected.parent running fresh proof).publicInput castChild
            targetColumn := outputSplit
      _ = (ProductPaperAlgebraFor.publicInputSplit config).split
            (selected.parent running fresh proof).publicInput castChild
            targetColumn := selectedSplit
      _ = PiDECAlgebra.Radix.splitScalar
            ((selected.parent running fresh proof).publicInput targetColumn)
            castChild := coordinateSplit
      _ = PiDECAlgebra.Radix.splitScalar
            ((selected.parent running fresh proof).publicInput targetColumn)
            child := congrArg
              (PiDECAlgebra.Radix.splitScalar
                ((selected.parent running fresh proof).publicInput targetColumn))
              castChild_eq
  · intro child component row lane
    have linked := congrArg
      (fun bundle => bundle (componentAt component) row lane)
      (placement.piDec.childBundle child)
    have outputExact :
        (selected.output running fresh proof).commitments child
            (componentAt component) row lane =
          ((selected.piDecAttempt running fresh proof).messages child
            ).commitment (componentAt component) row lane := by
      exact congrArg
        (fun bundle => bundle (componentAt component) row lane)
        (Key.output_commitment selected running fresh proof child)
    change
      ((selected.piDecAttempt running fresh proof).messages child
          ).commitment (componentAt component) row lane =
        ProductPiDecLinearCombination.fieldAt assignment canonical
          ((piDec.childBundle child).column
            (componentAt component) row lane) at linked
    exact linked.symm.trans outputExact.symm
  · intro child matrix lane
    have linked := placement.piDec.childEvaluation child
    have messageExact :
        ((selected.piDecAttempt running fresh proof).messages child
          ).evaluations =
          #[proof.piDecEvaluations child] := rfl
    have decodedExact :
        ProductPiDecTypedBridgeFor.decodeEvaluation rowVariables
            (piDec.childEvaluation child) assignment canonical =
          proof.piDecEvaluations child := by
      have singletonExact := messageExact.symm.trans linked
      simpa using singletonExact.symm
    have outputExact :
        (selected.output running fresh proof).evaluations child =
          proof.piDecEvaluations child :=
      Key.output_evaluation selected running fresh proof child
    exact congrFun (congrFun (decodedExact.trans outputExact.symm) matrix) lane

/-- Complete upstream rows plus output rows place every field of the exact
verifier-computed NIFS output. -/
theorem section_rows_sound
    (candidate : Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.Id)
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
    (samplerBase : Nat) (algebra : ProductPiRlcAlgebraRows.Layout)
    (piDec : ProductPiDecRows.Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (piCcsPlacement : ProductionProductPiCcsTypedBridgeFor.Placement candidate
      statementId config artifact running fresh proof wires assignment)
    (piCcsRows : Satisfies
      (ProductPiCcsTranscriptRowsFor.rows
        (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
          config artifact running fresh wires)) assignment)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (classificationRows : ProductPiRlcCandidateClassificationRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (selectorRows : ProductPiRlcFirstAcceptedBatchRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (algebraRows : Satisfies (ProductPiRlcAlgebraRows.rows algebra) assignment)
    (placement : ProductionProductNifsPaperRowsSoundFor.Placement candidate
      statementId config artifact running fresh proof wires samplerBase algebra
      piDec assignment canonical)
    (outputSection : SectionRows candidate statementId config artifact running fresh
      wires algebra piDec assignment) :
    Placed outputSection.layout assignment canonical
      ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
        config artifact).output running fresh proof) := by
  exact rows_sound outputSection.valid
    (sources_of_nifs_rows candidate statementId config artifact running fresh
      proof wires samplerBase algebra piDec assignment canonical one
      piCcsPlacement piCcsRows transcriptRows classificationRows selectorRows
      algebraRows placement outputSection)
    one outputSection.satisfied

end Nightstream.Implementation.NebulaV2.ProductionProductNifsOutputRowsFor
