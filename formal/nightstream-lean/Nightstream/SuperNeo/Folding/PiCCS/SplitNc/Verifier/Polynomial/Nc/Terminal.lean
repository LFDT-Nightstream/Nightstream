import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Types
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction

/-!
Verifier-output terminal binding for the independent Split-NC norm polynomial.

Protocol: SuperNeo `Pi_CCS`, split NC branch.
Phase: bind the raw `yZcol` output to the terminal SumCheck point.
Constraint family: active-lane output projection, padded-lane interpolation,
source mixing, and terminal equality; this file emits no rows.

Owns: a zero-extended typed lane table built from the verifier-visible
`yZcol` message; its evaluation at the verifier-derived lane point; the exact
bridge from source-bound claims to `SourceProjection.sourceValueAt`; and one
message terminal formula for every explicitly named gamma convention.

Does not own: derivation of the terminal column/lane point, transcript or
SumCheck checking, the authority proof that establishes source binding,
mixing-root probability, Rust, R1CS, row emission, row removal, or constraint
counts.

Emits constraints: no.

Authority boundary: `OutputMessage.yZcol` is untrusted data. The terminal may
use it only with `OutputClaims.YZcolBoundToSources` at the same verifier-owned
column point. Padded lanes are computed as zero and are never message fields.
The terminal preserves `.paperNc`, `.paperJointQ`, and `.splitV1` as explicit
arguments; no theorem in this file approves one convention over another.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.terminal.output.lane.active` | active lane leaves are the raw `yZcol[source,lane]` values | checked payload | `paddedYZcol` |
| `nifs.pi_ccs.nc.terminal.output.lane.padding` | lanes 54 through the padded lane domain are exactly zero | computed | `paddedYZcol` |
| `nifs.pi_ccs.nc.terminal.output.lane.mle` | the typed lane table is evaluated at the verifier-derived lane point | computed | `laneTable`, `valueAt` |
| `nifs.pi_ccs.nc.terminal.output.column` | canonical `yZcol` equals the source-derived column MLE | derived | `canonicalYZcol_eq_columnValueAt` |
| `nifs.pi_ccs.nc.terminal.output.binding` | source-bound output evaluation equals `SourceProjection.sourceValueAt` | checked then derived | `valueAt_eq_sourceValueAt_of_yZcolBoundToSources` |
| `nifs.pi_ccs.nc.terminal.range` | the strict cubic is applied after output interpolation | computed | `rangeAt` |
| `nifs.pi_ccs.nc.terminal.mixing` | source cubics use the selected named gamma schedule | computed | `mixedRangeAt` |
| `nifs.pi_ccs.nc.terminal.equality` | equality selectors and the mixed cubic reproduce semantic `qAtPoint` | checked then derived | `terminal_eq_qAtPoint_of_yZcolBoundToSources` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Terminal

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

/-! ## Finite-domain reindexing used by the column bridge -/

private theorem perm_of_nodup_mem_iff
    {Index : Type}
    [BEq Index] [LawfulBEq Index]
    {left right : List Index}
    (leftNodup : left.Nodup)
    (rightNodup : right.Nodup)
    (sameMembers : forall index, index ∈ left <-> index ∈ right) :
    left.Perm right := by
  rw [List.perm_iff_count]
  intro index
  rw [leftNodup.count, rightNodup.count]
  by_cases member : index ∈ left
  · have rightMember := (sameMembers index).mp member
    simp [member, rightMember]
  · have rightMember : index ∉ right := by
      exact fun present => member ((sameMembers index).mpr present)
    simp [member, rightMember]

private theorem finiteSum_eq_foldr (values : List K) :
    BooleanTable.finiteSum ops values = values.foldr ops.add ops.zero := by
  induction values with
  | nil => rfl
  | cons value values inductionHypothesis =>
      simp only [BooleanTable.finiteSum, List.foldr, inductionHypothesis]

private theorem finiteSum_eq_of_perm
    {left right : List K}
    (permutation : left.Perm right) :
    BooleanTable.finiteSum ops left = BooleanTable.finiteSum ops right := by
  rw [finiteSum_eq_foldr, finiteSum_eq_foldr]
  apply permutation.foldr_eq'
  intro left _ right _ tail
  calc
    ops.add right (ops.add left tail) =
        ops.add (ops.add right left) tail :=
      (laws.add_assoc _ _ _).symm
    _ = ops.add (ops.add left right) tail := by
      rw [laws.add_comm right left]
    _ = ops.add left (ops.add right tail) := laws.add_assoc _ _ _

private theorem sumMap_eq_of_perm
    {Index : Type}
    {left right : List Index}
    (permutation : left.Perm right)
    (value : Index -> K) :
    FiniteSumAlgebra.sumMap ops left value =
      FiniteSumAlgebra.sumMap ops right value := by
  unfold FiniteSumAlgebra.sumMap
  exact finiteSum_eq_of_perm (permutation.map value)

private theorem finiteSum_append (left right : List K) :
    BooleanTable.finiteSum ops (left ++ right) =
      K.add (BooleanTable.finiteSum ops left)
        (BooleanTable.finiteSum ops right) := by
  induction left with
  | nil => exact (laws.zero_add _).symm
  | cons value left inductionHypothesis =>
      simp only [List.cons_append, BooleanTable.finiteSum,
        inductionHypothesis]
      exact (laws.add_assoc _ _ _).symm

private theorem sumMap_append
    {Index : Type}
    (left right : List Index)
    (value : Index -> K) :
    FiniteSumAlgebra.sumMap ops (left ++ right) value =
      K.add (FiniteSumAlgebra.sumMap ops left value)
        (FiniteSumAlgebra.sumMap ops right value) := by
  unfold FiniteSumAlgebra.sumMap
  rw [List.map_append, finiteSum_append]

private theorem foldl_add_eq_sumMap
    {Index : Type}
    (indices : List Index)
    (value : Index -> K) :
    indices.foldl (fun accumulated index =>
        ops.add accumulated (value index)) ops.zero =
      FiniteSumAlgebra.sumMap ops indices value := by
  have withInitial : forall initial,
      indices.foldl (fun accumulated index =>
          ops.add accumulated (value index)) initial =
        ops.add initial (FiniteSumAlgebra.sumMap ops indices value) := by
    intro initial
    induction indices generalizing initial with
    | nil => exact (laws.add_zero initial).symm
    | cons index indices inductionHypothesis =>
        rw [List.foldl_cons, inductionHypothesis]
        exact laws.add_assoc initial (value index)
          (FiniteSumAlgebra.sumMap ops indices value)
  rw [withInitial ops.zero]
  exact laws.zero_add _

private theorem columnIndices_perm
    {domain : FlatNcDomain} :
    (BooleanVertex.all domain.columnVariables).map columnIndex
      |>.Perm (canonicalFinIndices domain.columnCount) := by
  apply perm_of_nodup_mem_iff
  · apply (BooleanVertex.all_nodup domain.columnVariables).map columnIndex
    intro left right different equal
    apply different
    calc
      left = columnVertex (columnIndex left) :=
        (columnVertex_columnIndex left).symm
      _ = columnVertex (columnIndex right) := by rw [equal]
      _ = right := columnVertex_columnIndex right
  · exact canonicalFinIndices_nodup domain.columnCount
  · intro index
    constructor
    · intro _
      exact List.mem_ofFn.mpr ⟨index, rfl⟩
    · intro _
      exact List.mem_map.mpr
        ⟨columnVertex index, BooleanVertex.mem_all _,
          columnIndex_columnVertex index⟩

private def liveColumn
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (column : Fin shape.carrierWidth) : Fin domain.columnCount :=
  domain.carrierColumn covers column

private def isLiveColumn
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (column : Fin domain.columnCount) : Bool :=
  decide (column.val < shape.carrierWidth)

private def liveColumns
    {shape : SemanticShape}
    {domain : FlatNcDomain} : List (Fin domain.columnCount) :=
  (canonicalFinIndices domain.columnCount).filter
    (isLiveColumn (shape := shape))

private def paddingColumns
    {shape : SemanticShape}
    {domain : FlatNcDomain} : List (Fin domain.columnCount) :=
  (canonicalFinIndices domain.columnCount).filter
    (fun column => !(isLiveColumn (shape := shape) column))

private theorem liveColumns_perm
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape) :
    (canonicalFinIndices shape.carrierWidth).map (liveColumn covers)
      |>.Perm (liveColumns (shape := shape) (domain := domain)) := by
  apply perm_of_nodup_mem_iff
  · apply (canonicalFinIndices_nodup shape.carrierWidth).map
      (liveColumn covers)
    intro left right different equal
    apply different
    apply Fin.ext
    simpa [liveColumn] using
      congrArg (fun index : Fin domain.columnCount => index.val) equal
  · exact (canonicalFinIndices_nodup domain.columnCount).filter _
  · intro column
    constructor
    · intro member
      rcases List.mem_map.mp member with ⟨live, _, rfl⟩
      apply List.mem_filter.mpr
      constructor
      · exact List.mem_ofFn.mpr ⟨liveColumn covers live, rfl⟩
      · simp [isLiveColumn, liveColumn]
    · intro member
      have parts := List.mem_filter.mp member
      have live : column.val < shape.carrierWidth := by
        simpa [isLiveColumn] using parts.2
      let sourceColumn : Fin shape.carrierWidth := ⟨column.val, live⟩
      apply List.mem_map.mpr
      refine ⟨sourceColumn, List.mem_ofFn.mpr ⟨sourceColumn, rfl⟩, ?_⟩
      exact Fin.ext rfl

private theorem fullColumns_partition
    {shape : SemanticShape}
    {domain : FlatNcDomain} :
    (liveColumns (shape := shape) (domain := domain) ++
      paddingColumns (shape := shape) (domain := domain)).Perm
        (canonicalFinIndices domain.columnCount) := by
  exact List.filter_append_perm
    (isLiveColumn (shape := shape))
    (canonicalFinIndices domain.columnCount)

private def fullColumnTerm
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (columnPoint : CubePoint K domain.columnVariables)
    (lane : Fin domain.laneCount)
    (column : Fin domain.columnCount) : K :=
  K.mul
    (K.embed (SourceProjection.paddedDiagonal covers data source column lane))
    (columnWeight columnPoint column)

private theorem fullColumnTerm_padding
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (columnPoint : CubePoint K domain.columnVariables)
    (lane : Fin domain.laneCount)
    (column : Fin domain.columnCount)
    (padding : shape.carrierWidth <= column.val) :
    fullColumnTerm covers data source columnPoint lane column = K.zero := by
  unfold fullColumnTerm
  rw [SourceProjection.paddedDiagonal_column_padding
    covers data source column lane padding]
  have embedded : K.embed (0 : F) = K.zero := ConcreteCarrier.embed_zero
  rw [embedded]
  change ops.mul ops.zero _ = ops.zero
  rw [laws.mul_comm, laws.mul_zero]

private theorem fullColumnTerm_live
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (columnPoint : CubePoint K domain.columnVariables)
    (lane : Fin ringDegree)
    (column : Fin shape.carrierWidth) :
    fullColumnTerm covers data source columnPoint
        (domain.phi81Lane covers lane) (liveColumn covers column) =
      yZcolTerm covers (data.assignment source) columnPoint lane column := by
  unfold fullColumnTerm liveColumn yZcolTerm
  rw [SourceProjection.paddedDiagonal_live]

private theorem columnValueAt_eq_canonicalYZcol
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (columnPoint : CubePoint K domain.columnVariables)
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree) :
    SourceProjection.columnValueAt covers data source columnPoint
        (laneVertex (domain.phi81Lane covers lane)) =
      canonicalYZcol covers data
        ({ rPrime := data.priorPoint, sPrime := columnPoint } :
          VerifierPoints shape domain)
        source lane := by
  let activeLane := domain.phi81Lane covers lane
  let term := fullColumnTerm covers data source columnPoint activeLane
  calc
    SourceProjection.columnValueAt covers data source columnPoint
        (laneVertex activeLane) =
      BooleanReproduction.equalityWeighted ops columnPoint (fun vertex =>
        K.embed (SourceProjection.paddedDiagonal covers data source
          (columnIndex vertex) activeLane)) := by
        unfold SourceProjection.columnValueAt SourceProjection.columnTable
        rw [laneIndex_laneVertex]
        exact (BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
          ops laws columnPoint _).symm
    _ = FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.columnVariables)
        (fun vertex => term (columnIndex vertex)) := by
      unfold BooleanReproduction.equalityWeighted term fullColumnTerm
      apply FiniteSumAlgebra.sumMap_congr
      intro vertex _
      have weightEq :
          BooleanVertex.equalityWeight ops vertex columnPoint =
            NumericBooleanDomain.tensorWeight ops
              (columnIndex vertex) columnPoint := by
        calc
          BooleanVertex.equalityWeight ops vertex columnPoint =
              BooleanVertex.equalityWeight ops
                (columnVertex (columnIndex vertex)) columnPoint := by
            rw [columnVertex_columnIndex]
          _ = NumericBooleanDomain.tensorWeight ops
                (columnIndex vertex) columnPoint :=
            (NumericBooleanDomain.tensorWeight_eq_equalityWeight
              ops (columnIndex vertex) columnPoint).symm
      rw [weightEq]
      rw [NumericBooleanDomain.tensorWeight_eq_testBitWeight ops
        (NumericBooleanDomain.WeightProductLaws.ofInterpolationEvaluationLaws
          laws)
        (columnIndex vertex) columnPoint]
      unfold columnWeight
      change K.mul _ _ = K.mul _ _
      exact laws.mul_comm _ _
    _ = FiniteSumAlgebra.sumMap ops
        ((BooleanVertex.all domain.columnVariables).map columnIndex) term := by
      simp [FiniteSumAlgebra.sumMap, Function.comp_def]
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices domain.columnCount) term :=
      sumMap_eq_of_perm columnIndices_perm term
    _ = FiniteSumAlgebra.sumMap ops
        (liveColumns (shape := shape) (domain := domain) ++
          paddingColumns (shape := shape) (domain := domain)) term :=
      sumMap_eq_of_perm fullColumns_partition.symm term
    _ = K.add
        (FiniteSumAlgebra.sumMap ops
          (liveColumns (shape := shape) (domain := domain)) term)
        (FiniteSumAlgebra.sumMap ops
          (paddingColumns (shape := shape) (domain := domain)) term) :=
      sumMap_append _ _ term
    _ = FiniteSumAlgebra.sumMap ops
        (liveColumns (shape := shape) (domain := domain)) term := by
      have paddingZero : FiniteSumAlgebra.sumMap ops
          (paddingColumns (shape := shape) (domain := domain)) term = K.zero := by
        calc
          _ = FiniteSumAlgebra.sumMap ops
              (paddingColumns (shape := shape) (domain := domain))
              (fun _ => K.zero) := by
            apply FiniteSumAlgebra.sumMap_congr
            intro column member
            have member' : column ∈
                (canonicalFinIndices domain.columnCount).filter
                  (fun candidate =>
                    !(isLiveColumn (shape := shape) candidate)) := by
              simpa [paddingColumns] using member
            have notLive := (List.mem_filter.mp member').2
            have padding : shape.carrierWidth <= column.val := by
              simp [isLiveColumn] at notLive
              omega
            exact fullColumnTerm_padding
              covers data source columnPoint activeLane column padding
          _ = K.zero := FiniteSumAlgebra.sumMap_zero ops laws _
      rw [paddingZero]
      change ops.add _ ops.zero = _
      exact laws.add_zero _
    _ = FiniteSumAlgebra.sumMap ops
        ((canonicalFinIndices shape.carrierWidth).map (liveColumn covers))
        term :=
      sumMap_eq_of_perm (liveColumns_perm covers).symm term
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.carrierWidth)
        (fun column =>
          yZcolTerm covers (data.assignment source) columnPoint lane column) := by
      unfold FiniteSumAlgebra.sumMap
      rw [List.map_map]
      apply congrArg (BooleanTable.finiteSum ops)
      apply List.map_congr_left
      intro column _
      exact fullColumnTerm_live covers data source columnPoint lane column
    _ = yZcolForAssignment covers (data.assignment source)
        columnPoint lane := by
      unfold yZcolForAssignment
      exact (foldl_add_eq_sumMap
        (canonicalFinIndices shape.carrierWidth)
        (fun column =>
          yZcolTerm covers (data.assignment source) columnPoint lane column)).symm
    _ = canonicalYZcol covers data
        ({ rPrime := data.priorPoint, sPrime := columnPoint } :
          VerifierPoints shape domain)
        source lane := rfl

/-- Canonical source-derived `yZcol` is exactly the corresponding leaf of the
independent source projection's lane table. This theorem is the anti-drift
bridge between the pre-existing output semantics and the new NC polynomial. -/
theorem canonicalYZcol_eq_columnValueAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (columnPoint : CubePoint K domain.columnVariables)
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree) :
    canonicalYZcol covers data
        ({ rPrime := data.priorPoint, sPrime := columnPoint } :
          VerifierPoints shape domain)
        source lane =
      SourceProjection.columnValueAt covers data source columnPoint
        (laneVertex (domain.phi81Lane covers lane)) :=
  (columnValueAt_eq_canonicalYZcol covers data columnPoint source lane).symm

/-! ## Verifier-output lane table and terminal -/

/-- Zero-extend the 54 active message lanes to the complete padded lane
domain. No padded lane is supplied by the prover. -/
def paddedYZcol
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount)
    (lane : Fin domain.laneCount) : K :=
  if live : lane.val < ringDegree then
    message.yZcol source ⟨lane.val, live⟩
  else
    K.zero

/-- An active Phi81 lane is read from the raw output without reindexing. -/
@[simp] theorem paddedYZcol_live
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree) :
    paddedYZcol (domain := domain) message source
        (domain.phi81Lane covers lane) =
      message.yZcol source lane := by
  simp [paddedYZcol, FlatNcDomain.phi81Lane]

/-- Every padded lane is computed as zero and is absent from the message. -/
theorem paddedYZcol_padding
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount)
    (lane : Fin domain.laneCount)
    (padding : ringDegree <= lane.val) :
    paddedYZcol message source lane = K.zero := by
  simp [paddedYZcol, Nat.not_lt.mpr padding]

/-- Typed padded lane table for one source at the verifier-derived column
point represented by the output message. -/
def laneTable
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount) : BooleanTable K domain.laneVariables :=
  BooleanTable.tabulate fun lane =>
    paddedYZcol message source (laneIndex lane)

/-- Evaluate the message's padded lane table at the verifier-derived terminal
lane point. -/
def valueAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount)
    (lanePoint : CubePoint K domain.laneVariables) : K :=
  (laneTable (domain := domain) message source).evaluate ops lanePoint

/-- Apply the strict-`b = 2` cubic only after interpolating the message lane
table. -/
def rangeAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount)
    (lanePoint : CubePoint K domain.laneVariables) : K :=
  let value := valueAt (domain := domain) message source lanePoint
  K.mul (K.mul (K.add value (K.embed 1)) value)
    (K.sub value (K.embed 1))

/-- Gamma compression of the output-derived source cubics. The convention is
an explicit argument and is never inferred from production behavior. -/
def mixedRangeAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : Mixing.GammaConvention)
    (message : OutputMessage shape)
    (coins : Mixing.Coins domain)
    (lanePoint : CubePoint K domain.laneVariables) : K :=
  FiniteSumAlgebra.sumMap ops
    (canonicalFinIndices shape.sourceCount) fun source =>
      SignedJointIdentity.gammaTerm ops coins.gamma
        (Mixing.sourceExponent shape convention source)
        (rangeAt (domain := domain) message source lanePoint)

/-- Raw-message NC terminal at the verifier-derived column/lane point. -/
def terminalFromMessage
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : Mixing.GammaConvention)
    (message : OutputMessage shape)
    (coins : Mixing.Coins domain)
    (point : Point domain) : K :=
  K.mul
    (K.mul
      (SumCheckTruthPath.pointEquality ops point.column coins.betaM)
      (SumCheckTruthPath.pointEquality ops point.lane coins.betaA))
    (mixedRangeAt convention message coins point.lane)

/-- Source binding at the verifier-owned column point makes the raw padded
lane MLE equal the independent nested source projection. In particular, the
theorem binds the value before applying the cubic; equality of cubic outputs
alone would not distinguish the valid roots zero and one. -/
theorem valueAt_eq_sourceValueAt_of_yZcolBoundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (point : Point domain)
    (message : OutputMessage shape)
    (bound : YZcolBoundToSources covers data
      ({ rPrime := data.priorPoint, sPrime := point.column } :
        VerifierPoints shape domain)
      message)
    (source : Fin shape.sourceCount) :
    valueAt (domain := domain) message source point.lane =
      SourceProjection.sourceValueAt covers data source point := by
  unfold valueAt laneTable SourceProjection.sourceValueAt
    SourceProjection.laneTableAtColumn
  rw [<- BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
    ops laws point.lane]
  rw [<- BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
    ops laws point.lane]
  unfold BooleanReproduction.equalityWeighted
  apply FiniteSumAlgebra.sumMap_congr
  intro lane _
  congr 1
  change paddedYZcol message source (laneIndex lane) =
    SourceProjection.columnValueAt covers data source point.column lane
  unfold paddedYZcol
  by_cases live : (laneIndex lane).val < ringDegree
  · rw [dif_pos live]
    let active : Fin ringDegree := ⟨(laneIndex lane).val, live⟩
    have claimBound := bound source active
    rw [claimBound]
    rw [canonicalYZcol_eq_columnValueAt covers data point.column source active]
    congr 1
    have activeLane : domain.phi81Lane covers active = laneIndex lane := by
      apply Fin.ext
      rfl
    rw [activeLane, laneVertex_laneIndex]
  · rw [dif_neg live]
    have padding : ringDegree <= (laneIndex lane).val := Nat.le_of_not_gt live
    -- The source lane table is zero for every column, hence its MLE is zero.
    symm
    unfold SourceProjection.columnValueAt SourceProjection.columnTable
    rw [<- BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
      ops laws point.column]
    unfold BooleanReproduction.equalityWeighted
    calc
      FiniteSumAlgebra.sumMap ops
          (BooleanVertex.all domain.columnVariables) (fun column =>
            ops.mul (BooleanVertex.equalityWeight ops column point.column)
              (K.embed (SourceProjection.paddedDiagonal covers data source
                (columnIndex column) (laneIndex lane)))) =
        FiniteSumAlgebra.sumMap ops
          (BooleanVertex.all domain.columnVariables) (fun _ => K.zero) := by
            apply FiniteSumAlgebra.sumMap_congr
            intro column _
            rw [SourceProjection.paddedDiagonal_lane_padding
              covers data source (columnIndex column) (laneIndex lane) padding]
            have embedded : K.embed (0 : F) = K.zero :=
              ConcreteCarrier.embed_zero
            rw [embedded]
            change ops.mul _ ops.zero = ops.zero
            exact laws.mul_zero _
      _ = K.zero := FiniteSumAlgebra.sumMap_zero ops laws _

private theorem rangeAt_eq_rangeValueAt_of_yZcolBoundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (point : Point domain)
    (message : OutputMessage shape)
    (bound : YZcolBoundToSources covers data
      ({ rPrime := data.priorPoint, sPrime := point.column } :
        VerifierPoints shape domain)
      message)
    (source : Fin shape.sourceCount) :
    rangeAt (domain := domain) message source point.lane =
      SourceProjection.rangeValueAt covers data source point := by
  unfold rangeAt SourceProjection.rangeValueAt
  rw [valueAt_eq_sourceValueAt_of_yZcolBoundToSources
    covers data point message bound source]

/-- Source-bound output mixing is exactly the independently defined semantic
mixing for the same named gamma convention. -/
theorem mixedRangeAt_eq_semantic_of_yZcolBoundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : Mixing.GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain)
    (message : OutputMessage shape)
    (bound : YZcolBoundToSources covers data
      ({ rPrime := data.priorPoint, sPrime := point.column } :
        VerifierPoints shape domain)
      message) :
    mixedRangeAt convention message coins point.lane =
      Mixing.mixedRangeAt convention covers data coins point := by
  unfold mixedRangeAt Mixing.mixedRangeAt
  apply FiniteSumAlgebra.sumMap_congr
  intro source _
  rw [rangeAt_eq_rangeValueAt_of_yZcolBoundToSources
    covers data point message bound source]

/-- Exact terminal anti-drift theorem: once every active `yZcol` is bound at
the same verifier-owned column point, the message terminal equals the
independent semantic NC polynomial. The statement is uniform over all three
gamma conventions. -/
theorem terminal_eq_qAtPoint_of_yZcolBoundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : Mixing.GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain)
    (message : OutputMessage shape)
    (bound : YZcolBoundToSources covers data
      ({ rPrime := data.priorPoint, sPrime := point.column } :
        VerifierPoints shape domain)
      message) :
    terminalFromMessage convention message coins point =
      Mixing.qAtPoint convention covers data coins point := by
  unfold terminalFromMessage Mixing.qAtPoint
  rw [mixedRangeAt_eq_semantic_of_yZcolBoundToSources
    convention covers data coins point message bound]

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Terminal
