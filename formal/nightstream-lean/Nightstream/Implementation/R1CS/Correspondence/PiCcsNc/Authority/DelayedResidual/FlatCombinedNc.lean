import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.ProductionRawChildren
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.MixingSoundness
import Nightstream.SuperNeo.SumCheck.FixedPhase
/-!
Contract: model the combined NC polynomial in a flat column-then-lane domain.
Assurance tier: model-level.
Owns: the raw-child delayed table, its nested flat-domain MLE, exact terminal
and Boolean-cube formulae, quartic coordinate slices, the fixed-270 model's
9+6 round profile, and the deterministic accepted-SumCheck decomposition.
Does not own: transcript scheduling or domain separation, parent padding-row
refinement, recursive-state continuity, commitment binding, generated rows,
Rust conformance, costs, or row-removal permission.
Emits constraints: no.
Authority boundary: the delayed table is computed only from `rawChildren`.
The fixed-production theorem below instantiates that argument with
`ProductionRawChildren.Fixed270.authoritativeRunningChildren data`; no output
message, `CeClaim.y_zcol`, digest, or caller-provided projection acceptance is
an input.
| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.nc.flat.delayed.source` | direct radix recomposition of raw child assignment diagonals | direct dataflow | `rawValueAt` |
| `pi_ccs.nc.flat.delayed.terminal` | terminal reads the same nested raw-table MLE | computed | `combinedAtPoint_eq_terminalRhs` |
| `pi_ccs.nc.flat.delayed.cube` | Boolean cube equals ordinary NC plus weighted old-point projection | derived | `combinedHypercubeSum_eq_ordinary_add_weightedProjection` |
| `pi_ccs.nc.flat.delayed.degree` | every column/lane slice has degree at most four | derived | `combinedAtPoint_column_quartic`, `combinedAtPoint_lane_quartic` |
| `pi_ccs.nc.flat.delayed.rounds.fixed270` | the 270-field public-carrier model uses 9 column and 6 lane rounds | computed | `fixed270_roundCount` |
| `pi_ccs.nc.flat.delayed.soundness` | acceptance yields NC truth and old-point binding or named roots/collision | security boundary | `accepted_implies_truth_and_oldPointRelation_or_badEvent` |
-/
namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.FlatCombinedNc
open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.MixingSoundness
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
open Nightstream.SuperNeo.SumCheck.Finite
private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws
private abbrev Polynomial := Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial K
private abbrev projectionOps :=
  Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.projectionOps
private abbrev flatShape (domain : FlatNcDomain) : PiCcsNc.Shape :=
  SourceRefinement.implementationShape domain
/-! ## Finite-domain reindexing -/
private theorem finiteSum_eq_foldr (values : List K) :
    BooleanTable.finiteSum ops values = values.foldr ops.add ops.zero := by
  induction values with
  | nil => rfl
  | cons value values inductionHypothesis =>
      simp only [BooleanTable.finiteSum, List.foldr, inductionHypothesis]
private theorem finiteSum_eq_of_perm
    {left right : List K} (permutation : left.Perm right) :
    BooleanTable.finiteSum ops left = BooleanTable.finiteSum ops right := by
  rw [finiteSum_eq_foldr, finiteSum_eq_foldr]
  apply permutation.foldr_eq'
  intro left _ right _ tail
  calc
    ops.add right (ops.add left tail) = ops.add (ops.add right left) tail :=
      (laws.add_assoc _ _ _).symm
    _ = ops.add (ops.add left right) tail := by rw [laws.add_comm right left]
    _ = ops.add left (ops.add right tail) := laws.add_assoc _ _ _
private theorem sumMap_eq_of_perm
    {Index : Type} {left right : List Index}
    (permutation : left.Perm right) (value : Index -> K) :
    FiniteSumAlgebra.sumMap ops left value =
      FiniteSumAlgebra.sumMap ops right value := by
  unfold FiniteSumAlgebra.sumMap
  exact finiteSum_eq_of_perm (permutation.map value)
private theorem perm_of_nodup_mem_iff
    {Index : Type} [BEq Index] [LawfulBEq Index]
    {left right : List Index}
    (leftNodup : left.Nodup) (rightNodup : right.Nodup)
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
private theorem columnIndices_perm {domain : FlatNcDomain} :
    ((BooleanVertex.all domain.columnVariables).map columnIndex).Perm
      (canonicalFinIndices domain.columnCount) := by
  apply perm_of_nodup_mem_iff
  · apply (BooleanVertex.all_nodup domain.columnVariables).map columnIndex
    intro left right different equal
    apply different
    calc
      left = columnVertex (columnIndex left) :=
        (columnVertex_columnIndex left).symm
      _ = columnVertex (columnIndex right) := by rw [equal]
      _ = right := columnVertex_columnIndex right
  · exact canonicalFinIndices_nodup _
  · intro column
    constructor
    · intro _
      exact List.mem_ofFn.mpr ⟨column, rfl⟩
    · intro _
      exact List.mem_map.mpr
        ⟨columnVertex column, BooleanVertex.mem_all _,
          columnIndex_columnVertex column⟩
private theorem laneIndices_perm {domain : FlatNcDomain} :
    ((BooleanVertex.all domain.laneVariables).map laneIndex).Perm
      (canonicalFinIndices domain.laneCount) := by
  apply perm_of_nodup_mem_iff
  · apply (BooleanVertex.all_nodup domain.laneVariables).map laneIndex
    intro left right different equal
    apply different
    calc
      left = laneVertex (laneIndex left) := (laneVertex_laneIndex left).symm
      _ = laneVertex (laneIndex right) := by rw [equal]
      _ = right := laneVertex_laneIndex right
  · exact canonicalFinIndices_nodup _
  · intro lane
    constructor
    · intro _
      exact List.mem_ofFn.mpr ⟨lane, rfl⟩
    · intro _
      exact List.mem_map.mpr
        ⟨laneVertex lane, BooleanVertex.mem_all _, laneIndex_laneVertex lane⟩
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
private theorem finiteSum_range_eq_sumRange
    (count : Nat) (term : Nat -> K) :
    BooleanTable.finiteSum ops ((List.range count).map term) =
      sumRange count term := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.map_append, finiteSum_append,
        inductionHypothesis, sumRange]
      simp only [List.map_singleton, BooleanTable.finiteSum]
      rw [laws.add_zero]
private theorem sumMap_canonical_eq_sumRange
    (count : Nat) (term : Nat -> K) :
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices count)
        (fun index => term index.val) =
      sumRange count term := by
  unfold FiniteSumAlgebra.sumMap
  calc
    BooleanTable.finiteSum ops
        ((canonicalFinIndices count).map (fun index => term index.val)) =
      BooleanTable.finiteSum ops ((List.range count).map term) := by
        congr 1
        simpa only [List.map_map, Function.comp_apply] using
          congrArg (List.map term) (canonicalFinIndices_values count)
    _ = sumRange count term := finiteSum_range_eq_sumRange count term
private theorem foldl_range_mul_eq_productRange
    (count : Nat) (term : Nat -> K) :
    (List.range count).foldl (fun accumulated index =>
        K.mul accumulated (term index)) K.one =
      productRange count term := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.foldl_append, inductionHypothesis]
      rfl
private theorem testBitWeight_eq_chi
    {variables : Nat} (index : Fin (2 ^ variables))
    (point : CubePoint K variables) :
    NumericBooleanDomain.testBitWeight ops point index =
      MixedPolynomial.chi point.coordinates index.val := by
  unfold NumericBooleanDomain.testBitWeight MixedPolynomial.chi
    MixedPolynomial.chiFactor
  rw [point.dimension]
  simp only [ConcreteCarrier.derived_sub_eq_concrete_sub]
  let factor : Nat -> K := fun bit =>
    if Nat.testBit index.val bit then
      point.coordinates.getD bit K.zero
    else
      K.sub K.one (point.coordinates.getD bit K.zero)
  calc
    _ =
      ((canonicalFinIndices variables).map
        (fun bit => factor bit.val)).foldl K.mul K.one := by
          rw [List.foldl_map]
          rfl
    _ = ((List.range variables).map factor).foldl K.mul K.one := by
      congr 1
      simpa only [List.map_map, Function.comp_apply] using
        congrArg (List.map factor) (canonicalFinIndices_values variables)
    _ = (List.range variables).foldl
        (fun accumulated bit => K.mul accumulated (factor bit)) K.one := by
      rw [List.foldl_map]
    _ = productRange variables factor :=
      foldl_range_mul_eq_productRange variables factor
private theorem equalityWeight_eq_chi
    {variables : Nat} (vertex : BooleanVertex variables)
    (point : CubePoint K variables) :
    vertex.equalityWeight ops point =
      MixedPolynomial.chi point.coordinates
        (NumericBooleanDomain.index vertex) := by
  let index : Fin (2 ^ variables) :=
    ⟨NumericBooleanDomain.index vertex,
      NumericBooleanDomain.index_lt_twoPow vertex⟩
  calc
    vertex.equalityWeight ops point =
        (NumericBooleanDomain.vertex variables index).equalityWeight ops point := by
      rw [NumericBooleanDomain.vertex_index]
    _ = NumericBooleanDomain.tensorWeight ops index point :=
      (NumericBooleanDomain.tensorWeight_eq_equalityWeight ops index point).symm
    _ = NumericBooleanDomain.testBitWeight ops point index :=
      NumericBooleanDomain.tensorWeight_eq_testBitWeight ops
        (NumericBooleanDomain.WeightProductLaws.ofInterpolationEvaluationLaws
          laws) index point
    _ = MixedPolynomial.chi point.coordinates index.val :=
      testBitWeight_eq_chi index point
/-! ## Authoritative raw-child flat MLE -/
/-- Column table of the radix-recomposed raw child diagonal at one Boolean
lane. -/
def rawColumnTable
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (lane : BooleanVertex domain.laneVariables) :
    BooleanTable K domain.columnVariables :=
  BooleanTable.tabulate fun column =>
    radixWeightedRawDiagonal radix rawChildren
      (columnIndex column).val (laneIndex lane).val
def rawColumnValueAt
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (column : CubePoint K domain.columnVariables)
    (lane : BooleanVertex domain.laneVariables) : K :=
  (rawColumnTable radix rawChildren lane).evaluate ops column

def rawLaneTableAtColumn
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (column : CubePoint K domain.columnVariables) :
    BooleanTable K domain.laneVariables :=
  BooleanTable.tabulate fun lane =>
    rawColumnValueAt radix rawChildren column lane

/-- Nested flat MLE in the production order: nine column coordinates, then
six Ajtai/Phi81 lane coordinates. -/
def rawValueAt
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (point : Point domain) : K :=
  (rawLaneTableAtColumn radix rawChildren point.column).evaluate ops point.lane

theorem rawValueAt_toCubePoint_eq_radixWeightedRawDiagonal
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (column : BooleanVertex domain.columnVariables)
    (lane : BooleanVertex domain.laneVariables) :
    rawValueAt radix rawChildren {
        column := column.toCubePoint ops
        lane := lane.toCubePoint ops } =
      radixWeightedRawDiagonal radix rawChildren
        (columnIndex column).val (laneIndex lane).val := by
  unfold rawValueAt rawLaneTableAtColumn
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt ops laws,
    BooleanTable.valueAt_tabulate]
  unfold rawColumnValueAt rawColumnTable
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt ops laws,
    BooleanTable.valueAt_tabulate]

/-- Numeric `chi` and the typed Boolean-table MLE give the same old-column
coefficient. This is the anti-drift bridge to `DelayedParentProjection`. -/
theorem rawColumnValueAt_eq_radixWeightedChildProjection
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (oldColumn : CubePoint K domain.columnVariables)
    (lane : Fin domain.laneCount) :
    rawColumnValueAt radix rawChildren oldColumn (laneVertex lane) =
      radixWeightedChildProjection (flatShape domain) radix rawChildren
        oldColumn.coordinates lane.val := by
  unfold rawColumnValueAt rawColumnTable
  rw [← BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
    ops laws oldColumn]
  unfold BooleanReproduction.equalityWeighted
  simp only [laneIndex_laneVertex]
  calc
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.columnVariables) (fun column =>
          K.mul (column.equalityWeight ops oldColumn)
            (radixWeightedRawDiagonal radix rawChildren
              (columnIndex column).val lane.val)) =
      FiniteSumAlgebra.sumMap ops
        ((BooleanVertex.all domain.columnVariables).map columnIndex)
        (fun column =>
          K.mul (radixWeightedRawDiagonal radix rawChildren
              column.val lane.val)
            (MixedPolynomial.chi oldColumn.coordinates column.val)) := by
        unfold FiniteSumAlgebra.sumMap
        rw [List.map_map]
        apply congrArg (BooleanTable.finiteSum ops)
        apply List.map_congr_left
        intro column _
        simp only [Function.comp_apply]
        rw [equalityWeight_eq_chi]
        simp only [columnIndex]
        exact laws.mul_comm _ _
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices domain.columnCount) (fun column =>
          K.mul (radixWeightedRawDiagonal radix rawChildren
              column.val lane.val)
            (MixedPolynomial.chi oldColumn.coordinates column.val)) :=
      sumMap_eq_of_perm columnIndices_perm _
    _ = sumRange domain.columnCount (fun column =>
          K.mul (radixWeightedRawDiagonal radix rawChildren column lane.val)
            (MixedPolynomial.chi oldColumn.coordinates column)) :=
      by simpa only using
        (sumMap_canonical_eq_sumRange domain.columnCount (fun column : Nat =>
          K.mul (radixWeightedRawDiagonal radix rawChildren column lane.val)
            (MixedPolynomial.chi oldColumn.coordinates column)))
    _ = radixWeightedChildProjection (flatShape domain) radix rawChildren
        oldColumn.coordinates lane.val := by
      simp only [radixWeightedChildProjection, flatShape,
        SourceRefinement.implementationShape_laneDomain,
        SourceRefinement.implementationShape_columnDomain, if_pos lane.isLt]

/-- The nested typed terminal is exactly the pre-existing flat
`radixCombinedRawZ` formula on every extension point. -/
theorem rawValueAt_eq_radixCombinedRawZ
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (point : Point domain) :
    rawValueAt radix rawChildren point =
      radixCombinedRawZ (flatShape domain) radix rawChildren
        point.column.coordinates point.lane.coordinates := by
  unfold rawValueAt rawLaneTableAtColumn
  rw [← BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
    ops laws point.lane]
  unfold BooleanReproduction.equalityWeighted
  calc
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.laneVariables) (fun lane =>
          K.mul (lane.equalityWeight ops point.lane)
            (rawColumnValueAt radix rawChildren point.column lane)) =
      FiniteSumAlgebra.sumMap ops
        ((BooleanVertex.all domain.laneVariables).map laneIndex) (fun lane =>
          K.mul
            (radixWeightedChildProjection (flatShape domain) radix rawChildren
              point.column.coordinates lane.val)
            (MixedPolynomial.chi point.lane.coordinates lane.val)) := by
        unfold FiniteSumAlgebra.sumMap
        rw [List.map_map]
        apply congrArg (BooleanTable.finiteSum ops)
        apply List.map_congr_left
        intro lane _
        simp only [Function.comp_apply]
        rw [equalityWeight_eq_chi]
        rw [← laneVertex_laneIndex lane]
        rw [rawColumnValueAt_eq_radixWeightedChildProjection]
        have indexRoundTrip :
            NumericBooleanDomain.index (laneVertex (laneIndex lane)) =
              (laneIndex lane).val :=
          congrArg Fin.val (laneIndex_laneVertex (laneIndex lane))
        rw [indexRoundTrip]
        simp only [laneIndex_laneVertex]
        exact laws.mul_comm _ _
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices domain.laneCount) (fun lane =>
          K.mul
            (radixWeightedChildProjection (flatShape domain) radix rawChildren
              point.column.coordinates lane.val)
            (MixedPolynomial.chi point.lane.coordinates lane.val)) :=
      sumMap_eq_of_perm laneIndices_perm _
    _ = sumRange domain.laneCount (fun lane =>
          K.mul
            (radixWeightedChildProjection (flatShape domain) radix rawChildren
              point.column.coordinates lane)
            (MixedPolynomial.chi point.lane.coordinates lane)) :=
      by simpa only using
        (sumMap_canonical_eq_sumRange domain.laneCount (fun lane : Nat =>
          K.mul (radixWeightedChildProjection (flatShape domain) radix rawChildren
            point.column.coordinates lane)
          (MixedPolynomial.chi point.lane.coordinates lane)))
    _ = radixCombinedRawZ (flatShape domain) radix rawChildren
        point.column.coordinates point.lane.coordinates := by
      rfl

private theorem targetPower_eq_powK (base : K) : forall exponent,
    TargetPolynomial.power ops.toOps base exponent = powK base exponent
  | 0 => rfl
  | exponent + 1 => by
      rw [TargetPolynomial.power, powK, targetPower_eq_powK]
      exact laws.mul_comm _ _

/-! ## Combined flat polynomial and exact cube -/

def betaPowerSelector
    {domain : FlatNcDomain} (producerBeta : K)
    (lane : CubePoint K domain.laneVariables) : K :=
  (BooleanTable.tabulate fun vertex =>
    TargetPolynomial.power ops.toOps producerBeta (laneIndex vertex).val
  ).evaluate ops lane

def delayedAtPoint
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (point : Point domain) : K :=
  K.mul batchWeight
    (K.mul (SumCheckTruthPath.pointEquality ops point.column oldColumn)
      (K.mul (betaPowerSelector producerBeta point.lane)
        (rawValueAt radix rawChildren point)))

/-- Current production convention (`splitV1`) plus the independent delayed
raw-child summand. -/
def combinedAtPoint
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (point : Point domain) : K :=
  K.add (Mixing.qAtPoint .splitV1 covers data coins point)
    (delayedAtPoint radix rawChildren producerBeta batchWeight oldColumn point)

def rawProjectionAtOldPoint
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (oldColumn : CubePoint K domain.columnVariables) : K :=
  FiniteSumAlgebra.sumMap ops
    (BooleanVertex.all domain.laneVariables) fun lane =>
      K.mul (TargetPolynomial.power ops.toOps producerBeta (laneIndex lane).val)
        (rawColumnValueAt radix rawChildren oldColumn lane)

private theorem rawValueAt_column_reproduce
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (oldColumn : CubePoint K domain.columnVariables)
    (lane : BooleanVertex domain.laneVariables) :
    BooleanReproduction.equalityWeighted ops oldColumn (fun column =>
        rawValueAt radix rawChildren {
          column := column.toCubePoint ops
          lane := lane.toCubePoint ops }) =
      rawValueAt radix rawChildren {
        column := oldColumn
        lane := lane.toCubePoint ops } := by
  calc
    BooleanReproduction.equalityWeighted ops oldColumn (fun column =>
        rawValueAt radix rawChildren {
          column := column.toCubePoint ops
          lane := lane.toCubePoint ops }) =
      BooleanReproduction.equalityWeighted ops oldColumn (fun column =>
        radixWeightedRawDiagonal radix rawChildren
          (columnIndex column).val (laneIndex lane).val) := by
            apply congrArg
            funext column
            exact rawValueAt_toCubePoint_eq_radixWeightedRawDiagonal
              radix rawChildren column lane
    _ = (rawColumnTable radix rawChildren lane).evaluate ops oldColumn :=
      BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
        ops laws oldColumn _
    _ = rawValueAt radix rawChildren {
        column := oldColumn
        lane := lane.toCubePoint ops } := by
      unfold rawValueAt rawLaneTableAtColumn
      rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt ops laws,
        BooleanTable.valueAt_tabulate]
      rfl

def delayedHypercubeSum
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables) : K :=
  FiniteSumAlgebra.sumMap ops
    (BooleanVertex.all domain.columnVariables) fun column =>
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.laneVariables) fun lane =>
          delayedAtPoint radix rawChildren producerBeta batchWeight oldColumn {
            column := column.toCubePoint ops
            lane := lane.toCubePoint ops }

/-- The flat delayed cube is the sampled weight times the direct raw-child
old-point scalar. No nonzero assumption is used. -/
theorem delayedHypercubeSum_eq_weightedProjection
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables) :
    delayedHypercubeSum radix rawChildren producerBeta batchWeight oldColumn =
      K.mul batchWeight
        (rawProjectionAtOldPoint radix rawChildren producerBeta oldColumn) := by
  let columns := BooleanVertex.all domain.columnVariables
  let lanes := BooleanVertex.all domain.laneVariables
  unfold delayedHypercubeSum delayedAtPoint
  calc
    FiniteSumAlgebra.sumMap ops columns (fun column =>
        FiniteSumAlgebra.sumMap ops lanes (fun lane =>
          K.mul batchWeight
            (K.mul
              (SumCheckTruthPath.pointEquality ops
                (column.toCubePoint ops) oldColumn)
              (K.mul (betaPowerSelector producerBeta (lane.toCubePoint ops))
                (rawValueAt radix rawChildren {
                  column := column.toCubePoint ops
                  lane := lane.toCubePoint ops }))))) =
      FiniteSumAlgebra.sumMap ops columns (fun column =>
        FiniteSumAlgebra.sumMap ops lanes (fun lane =>
          K.mul batchWeight
            (K.mul (column.equalityWeight ops oldColumn)
              (K.mul (TargetPolynomial.power ops.toOps producerBeta
                  (laneIndex lane).val)
                (rawValueAt radix rawChildren {
                  column := column.toCubePoint ops
                  lane := lane.toCubePoint ops }))))) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro column _
        apply FiniteSumAlgebra.sumMap_congr
        intro lane _
        rw [SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight
          ops laws]
        unfold betaPowerSelector
        rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt ops laws,
          BooleanTable.valueAt_tabulate]
    _ = K.mul batchWeight
        (FiniteSumAlgebra.sumMap ops columns (fun column =>
          FiniteSumAlgebra.sumMap ops lanes (fun lane =>
            K.mul (column.equalityWeight ops oldColumn)
              (K.mul (TargetPolynomial.power ops.toOps producerBeta
                  (laneIndex lane).val)
                (rawValueAt radix rawChildren {
                  column := column.toCubePoint ops
                  lane := lane.toCubePoint ops }))))) := by
      calc
        _ = FiniteSumAlgebra.sumMap ops columns (fun column =>
            K.mul batchWeight
              (FiniteSumAlgebra.sumMap ops lanes (fun lane =>
                K.mul (column.equalityWeight ops oldColumn)
                  (K.mul (TargetPolynomial.power ops.toOps producerBeta
                      (laneIndex lane).val)
                    (rawValueAt radix rawChildren {
                      column := column.toCubePoint ops
                      lane := lane.toCubePoint ops }))))) := by
              apply FiniteSumAlgebra.sumMap_congr
              intro column _
              exact FiniteSumAlgebra.sumMap_mul_left
                ops laws batchWeight lanes _
        _ = _ := FiniteSumAlgebra.sumMap_mul_left
          ops laws batchWeight columns _
    _ = K.mul batchWeight
        (FiniteSumAlgebra.sumMap ops lanes (fun lane =>
          FiniteSumAlgebra.sumMap ops columns (fun column =>
            K.mul (column.equalityWeight ops oldColumn)
              (K.mul (TargetPolynomial.power ops.toOps producerBeta
                  (laneIndex lane).val)
                (rawValueAt radix rawChildren {
                  column := column.toCubePoint ops
                  lane := lane.toCubePoint ops }))))) := by
      apply congrArg (K.mul batchWeight)
      exact FiniteSumAlgebra.sumMap_swap ops laws columns lanes _
    _ = K.mul batchWeight
        (FiniteSumAlgebra.sumMap ops lanes (fun lane =>
          K.mul (TargetPolynomial.power ops.toOps producerBeta
              (laneIndex lane).val)
            (BooleanReproduction.equalityWeighted ops oldColumn (fun column =>
              rawValueAt radix rawChildren {
                column := column.toCubePoint ops
                lane := lane.toCubePoint ops })))) := by
      apply congrArg (K.mul batchWeight)
      apply FiniteSumAlgebra.sumMap_congr
      intro lane _
      calc
        FiniteSumAlgebra.sumMap ops columns (fun column =>
            K.mul (column.equalityWeight ops oldColumn)
              (K.mul (TargetPolynomial.power ops.toOps producerBeta
                  (laneIndex lane).val)
                (rawValueAt radix rawChildren {
                  column := column.toCubePoint ops
                  lane := lane.toCubePoint ops }))) =
          FiniteSumAlgebra.sumMap ops columns (fun column =>
            K.mul (TargetPolynomial.power ops.toOps producerBeta
                (laneIndex lane).val)
              (K.mul (column.equalityWeight ops oldColumn)
                (rawValueAt radix rawChildren {
                  column := column.toCubePoint ops
                  lane := lane.toCubePoint ops }))) := by
            apply FiniteSumAlgebra.sumMap_congr
            intro column _
            let a := column.equalityWeight ops oldColumn
            let b := TargetPolynomial.power ops.toOps producerBeta
              (laneIndex lane).val
            let c := rawValueAt radix rawChildren {
              column := column.toCubePoint ops
              lane := lane.toCubePoint ops }
            change K.mul a (K.mul b c) = K.mul b (K.mul a c)
            calc
              _ = K.mul (K.mul a b) c := (laws.mul_assoc _ _ _).symm
              _ = K.mul (K.mul b a) c :=
                congrArg (fun value => K.mul value c) (laws.mul_comm _ _)
              _ = _ := laws.mul_assoc _ _ _
        _ = K.mul (TargetPolynomial.power ops.toOps producerBeta
              (laneIndex lane).val)
            (FiniteSumAlgebra.sumMap ops columns (fun column =>
              K.mul (column.equalityWeight ops oldColumn)
                (rawValueAt radix rawChildren {
                  column := column.toCubePoint ops
                  lane := lane.toCubePoint ops }))) :=
          FiniteSumAlgebra.sumMap_mul_left ops laws _ columns _
        _ = _ := rfl
    _ = K.mul batchWeight
        (rawProjectionAtOldPoint radix rawChildren producerBeta oldColumn) := by
      unfold rawProjectionAtOldPoint
      apply congrArg (K.mul batchWeight)
      apply FiniteSumAlgebra.sumMap_congr
      intro lane _
      rw [rawValueAt_column_reproduce]
      unfold rawValueAt rawLaneTableAtColumn
      rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt ops laws,
        BooleanTable.valueAt_tabulate]

def combinedHypercubeSum
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables) : K :=
  FiniteSumAlgebra.sumMap ops
    (BooleanVertex.all domain.columnVariables) fun column =>
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.laneVariables) fun lane =>
          combinedAtPoint covers data coins radix rawChildren producerBeta
            batchWeight oldColumn {
              column := column.toCubePoint ops
              lane := lane.toCubePoint ops }

theorem combinedHypercubeSum_eq_ordinary_add_weightedProjection
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables) :
    combinedHypercubeSum covers data coins radix rawChildren producerBeta
        batchWeight oldColumn =
      K.add (InitialSum.hypercubeSum .splitV1 covers data coins)
        (K.mul batchWeight
          (rawProjectionAtOldPoint radix rawChildren producerBeta oldColumn)) := by
  unfold combinedHypercubeSum combinedAtPoint
  calc
    _ = FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.columnVariables) (fun column =>
          K.add
            (FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all domain.laneVariables) (fun lane =>
                Mixing.qAtPoint .splitV1 covers data coins {
                  column := column.toCubePoint ops
                  lane := lane.toCubePoint ops }))
            (FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all domain.laneVariables) (fun lane =>
                delayedAtPoint radix rawChildren producerBeta batchWeight
                  oldColumn {
                    column := column.toCubePoint ops
                    lane := lane.toCubePoint ops }))) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro column _
        exact FiniteSumAlgebra.sumMap_add ops laws
          (BooleanVertex.all domain.laneVariables) _ _
    _ = K.add (InitialSum.hypercubeSum .splitV1 covers data coins)
        (delayedHypercubeSum radix rawChildren producerBeta batchWeight
          oldColumn) := by
      exact FiniteSumAlgebra.sumMap_add ops laws
        (BooleanVertex.all domain.columnVariables) _ _
    _ = _ := by rw [delayedHypercubeSum_eq_weightedProjection]

/-! ## Exact terminal and zero weight -/

def delayedTerminalRhs
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (point : Point domain) : K :=
  K.mul batchWeight
    (K.mul (SumCheckTruthPath.pointEquality ops point.column oldColumn)
      (K.mul (betaPowerSelector producerBeta point.lane)
        (rawValueAt radix rawChildren point)))

theorem delayedTerminalRhs_uses_flatRawZ
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (point : Point domain) :
    delayedTerminalRhs radix rawChildren producerBeta batchWeight oldColumn
        point =
      K.mul batchWeight
        (K.mul (SumCheckTruthPath.pointEquality ops point.column oldColumn)
          (K.mul (betaPowerSelector producerBeta point.lane)
            (radixCombinedRawZ (flatShape domain) radix rawChildren
              point.column.coordinates point.lane.coordinates))) := by
  unfold delayedTerminalRhs
  rw [rawValueAt_eq_radixCombinedRawZ]

def combinedTerminalRhs
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (point : Point domain) : K :=
  K.add (Mixing.qAtPoint .splitV1 covers data coins point)
    (delayedTerminalRhs radix rawChildren producerBeta batchWeight
      oldColumn point)

theorem combinedAtPoint_eq_terminalRhs
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (point : Point domain) :
    combinedAtPoint covers data coins radix rawChildren producerBeta
        batchWeight oldColumn point =
      combinedTerminalRhs covers data coins radix rawChildren producerBeta
        batchWeight oldColumn point := by
  rfl

theorem delayedAtPoint_eq_zero_of_batchWeight_eq_zero
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables) (point : Point domain)
    (batchWeightZero : batchWeight = K.zero) :
    delayedAtPoint radix rawChildren producerBeta batchWeight oldColumn point =
      K.zero := by
  subst batchWeight
  unfold delayedAtPoint
  change K.mul K.zero _ = K.zero
  calc
    _ = K.mul _ K.zero := laws.mul_comm _ _
    _ = K.zero := laws.mul_zero _

theorem combinedAtPoint_eq_ordinary_of_batchWeight_eq_zero
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables) (point : Point domain)
    (batchWeightZero : batchWeight = K.zero) :
    combinedAtPoint covers data coins radix rawChildren producerBeta
        batchWeight oldColumn point =
      Mixing.qAtPoint .splitV1 covers data coins point := by
  unfold combinedAtPoint
  rw [delayedAtPoint_eq_zero_of_batchWeight_eq_zero
    radix rawChildren producerBeta batchWeight oldColumn point batchWeightZero]
  exact laws.add_zero _

/-! The raw scalar is definitionally independent of a carried child sidecar. -/
theorem rawProjectionAtOldPoint_eq_compactOldPointEvaluation
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (oldColumn : CubePoint K domain.columnVariables)
    (lanesCoverRing : ringDegree <= domain.laneCount) :
    rawProjectionAtOldPoint radix rawChildren producerBeta oldColumn =
      compactOldPointEvaluation (flatShape domain) radix rawChildren
        producerBeta oldColumn.coordinates := by
  unfold rawProjectionAtOldPoint
  calc
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.laneVariables) (fun lane =>
          K.mul (TargetPolynomial.power ops.toOps producerBeta
              (laneIndex lane).val)
            (rawColumnValueAt radix rawChildren oldColumn lane)) =
      FiniteSumAlgebra.sumMap ops
        ((BooleanVertex.all domain.laneVariables).map laneIndex) (fun lane =>
          K.mul (TargetPolynomial.power ops.toOps producerBeta lane.val)
            (rawColumnValueAt radix rawChildren oldColumn
              (laneVertex lane))) := by
        unfold FiniteSumAlgebra.sumMap
        rw [List.map_map]
        apply congrArg (BooleanTable.finiteSum ops)
        apply List.map_congr_left
        intro lane _
        simp only [Function.comp_apply, laneVertex_laneIndex]
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices domain.laneCount) (fun lane =>
          K.mul (TargetPolynomial.power ops.toOps producerBeta lane.val)
            (rawColumnValueAt radix rawChildren oldColumn
              (laneVertex lane))) :=
      sumMap_eq_of_perm laneIndices_perm _
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices domain.laneCount) (fun lane =>
          K.mul (powK producerBeta lane.val)
            (radixWeightedChildProjection (flatShape domain) radix rawChildren
              oldColumn.coordinates lane.val)) := by
      apply FiniteSumAlgebra.sumMap_congr
      intro lane _
      rw [targetPower_eq_powK]
      rw [rawColumnValueAt_eq_radixWeightedChildProjection
        radix rawChildren oldColumn lane]
    _ = sumRange domain.laneCount (fun lane =>
          K.mul (powK producerBeta lane)
            (radixWeightedChildProjection (flatShape domain) radix rawChildren
              oldColumn.coordinates lane)) :=
      by simpa only using
        (sumMap_canonical_eq_sumRange domain.laneCount (fun lane : Nat =>
          K.mul (powK producerBeta lane)
            (radixWeightedChildProjection (flatShape domain) radix rawChildren
              oldColumn.coordinates lane)))
    _ = sumRange domain.laneCount (fun lane =>
          K.mul
            (radixWeightedChildProjection (flatShape domain) radix rawChildren
              oldColumn.coordinates lane)
            (powK producerBeta lane)) := by
      apply sumRange_congr
      intro lane _
      exact laws.mul_comm _ _
    _ = rawProjectionAtProducerBeta (flatShape domain) radix rawChildren
        producerBeta oldColumn.coordinates :=
      (rawProjectionAtProducerBeta_eq_yZcolEvaluation
        (flatShape domain) radix rawChildren producerBeta
          oldColumn.coordinates).symm
    _ = activeRawProjectionAtProducerBeta (flatShape domain) radix rawChildren
        producerBeta oldColumn.coordinates :=
      rawProjectionAtProducerBeta_eq_active (flatShape domain) radix rawChildren
        producerBeta oldColumn.coordinates lanesCoverRing
    _ = compactOldPointEvaluation (flatShape domain) radix rawChildren
        producerBeta oldColumn.coordinates :=
      (compactOldPointEvaluation_eq_active (flatShape domain) radix rawChildren
        producerBeta oldColumn.coordinates).symm

/-! ## Per-axis degree and fixed-width rounds -/

private theorem rawColumnValueAt_column_affine
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (lane : BooleanVertex domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    DegreeSupport.Represents 1 fun point =>
      rawColumnValueAt radix rawChildren
        (cubeSlice before after length point) lane := by
  unfold rawColumnValueAt BooleanTable.evaluate
  exact evaluateCoordinates_affine
    (rawColumnTable radix rawChildren lane) before after length

private theorem rawValueAt_column_affine
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    DegreeSupport.Represents 1 fun point =>
      rawValueAt radix rawChildren {
        column := cubeSlice before after length point
        lane := lane } := by
  have represented := polynomial_sum_exists
    (BooleanVertex.all domain.laneVariables)
    (fun vertex => vertex.equalityWeight ops lane)
    (fun vertex point => rawColumnValueAt radix rawChildren
      (cubeSlice before after length point) vertex)
    (by
      intro vertex _
      exact rawColumnValueAt_column_affine
        radix rawChildren vertex before after length)
  rcases represented with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  unfold rawValueAt rawLaneTableAtColumn
  exact BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
    ops laws lane _

private theorem rawValueAt_lane_affine
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (column : CubePoint K domain.columnVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    DegreeSupport.Represents 1 fun point =>
      rawValueAt radix rawChildren {
        column := column
        lane := cubeSlice before after length point } := by
  unfold rawValueAt BooleanTable.evaluate
  exact evaluateCoordinates_affine
    (rawLaneTableAtColumn radix rawChildren column) before after length

private theorem pointEquality_column_affine
    {domain : FlatNcDomain}
    (oldColumn : CubePoint K domain.columnVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    DegreeSupport.Represents 1 fun point =>
      SumCheckTruthPath.pointEquality ops
        (cubeSlice before after length point) oldColumn := by
  unfold SumCheckTruthPath.pointEquality
  apply pointEqualityCoordinates_affine
  rw [oldColumn.dimension]
  exact length

private theorem betaPowerSelector_lane_affine
    {domain : FlatNcDomain} (producerBeta : K)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    DegreeSupport.Represents 1 fun point =>
      betaPowerSelector producerBeta
        (cubeSlice before after length point) := by
  unfold betaPowerSelector BooleanTable.evaluate
  exact evaluateCoordinates_affine
    (BooleanTable.tabulate fun vertex =>
      TargetPolynomial.power ops.toOps producerBeta (laneIndex vertex).val)
    before after length

private theorem delayedAtPoint_column_quadratic
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    DegreeSupport.Represents 2 fun point =>
      delayedAtPoint radix rawChildren producerBeta batchWeight oldColumn {
        column := cubeSlice before after length point
        lane := lane } := by
  have selector := pointEquality_column_affine oldColumn before after length
  have raw := rawValueAt_column_affine
    radix rawChildren lane before after length
  simpa [delayedAtPoint] using
    DegreeSupport.Represents.scale batchWeight
      (DegreeSupport.Represents.mul selector
        (DegreeSupport.Represents.scale
          (betaPowerSelector producerBeta lane) raw))

private theorem delayedAtPoint_lane_quadratic
    {domain : FlatNcDomain} (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn column : CubePoint K domain.columnVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    DegreeSupport.Represents 2 fun point =>
      delayedAtPoint radix rawChildren producerBeta batchWeight oldColumn {
        column := column
        lane := cubeSlice before after length point } := by
  have selector := betaPowerSelector_lane_affine
    producerBeta before after length
  have raw := rawValueAt_lane_affine
    radix rawChildren column before after length
  let columnSelector :=
    SumCheckTruthPath.pointEquality ops column oldColumn
  simpa [delayedAtPoint, columnSelector] using
    DegreeSupport.Represents.scale batchWeight
      (DegreeSupport.Represents.scale columnSelector
        (DegreeSupport.Represents.mul selector raw))

theorem combinedAtPoint_column_quartic
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.columnVariables) :
    RepresentsAtMostFour fun point =>
      combinedAtPoint covers data coins radix rawChildren producerBeta
        batchWeight oldColumn {
          column := cubeSlice before after length point
          lane := lane } := by
  have ordinary : RepresentsAtMostFour fun point =>
      Mixing.qAtPoint .splitV1 covers data coins {
        column := cubeSlice before after length point
        lane := lane } :=
    qAtPoint_column_quartic .splitV1 covers data coins lane
      before after length
  have delayed := delayedAtPoint_column_quadratic radix rawChildren
    producerBeta batchWeight oldColumn lane before after length
  simpa [combinedAtPoint] using DegreeSupport.Represents.add ordinary
    (DegreeSupport.Represents.widen (by decide) delayed)

theorem combinedAtPoint_lane_quartic
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn column : CubePoint K domain.columnVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    RepresentsAtMostFour fun point =>
      combinedAtPoint covers data coins radix rawChildren producerBeta
        batchWeight oldColumn {
          column := column
          lane := cubeSlice before after length point } := by
  have ordinary : RepresentsAtMostFour fun point =>
      Mixing.qAtPoint .splitV1 covers data coins {
        column := column
        lane := cubeSlice before after length point } :=
    qAtPoint_lane_quartic .splitV1 covers data coins column
      before after length
  have delayed := delayedAtPoint_lane_quadratic radix rawChildren
    producerBeta batchWeight oldColumn column before after length
  simpa [combinedAtPoint] using DegreeSupport.Represents.add ordinary
    (DegreeSupport.Represents.widen (by decide) delayed)

/-- Fail-closed list evaluator used by the generic fixed-phase checker. -/
def sumcheckPolynomial
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (coordinates : List K) : K :=
  if length : coordinates.length =
      domain.columnVariables + domain.laneVariables then
    combinedAtPoint covers data coins radix rawChildren producerBeta
      batchWeight oldColumn (Point.ofCoordinates coordinates length)
  else
    K.zero

theorem sumcheckPolynomial_coordinates_eq_combinedAtPoint
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables) (point : Point domain) :
    sumcheckPolynomial covers data coins radix rawChildren producerBeta
        batchWeight oldColumn point.coordinates =
      combinedAtPoint covers data coins radix rawChildren producerBeta
        batchWeight oldColumn point := by
  unfold sumcheckPolynomial
  rw [dif_pos point.coordinates_length, Point.ofCoordinates_coordinates]

private theorem cubePoint_eq_of_coordinates_eq
    {variables : Nat} (left right : CubePoint K variables)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

private theorem ofCoordinates_eq_columnSlice
    {domain : FlatNcDomain} (before after : List K)
    (beforeColumn : before.length < domain.columnVariables)
    (totalLength : before.length + 1 + after.length =
      domain.columnVariables + domain.laneVariables) (point : K) :
    let columnAfter :=
      after.take (domain.columnVariables - before.length - 1)
    let laneCoordinates :=
      after.drop (domain.columnVariables - before.length - 1)
    let columnLength : before.length + 1 + columnAfter.length =
        domain.columnVariables := by
      dsimp only [columnAfter]
      rw [List.length_take]
      omega
    let laneLength : laneCoordinates.length = domain.laneVariables := by
      dsimp only [laneCoordinates]
      rw [List.length_drop]
      omega
    Point.ofCoordinates (before ++ point :: after) (by simp; omega) = {
      column := cubeSlice before columnAfter columnLength point
      lane := { coordinates := laneCoordinates, dimension := laneLength } } := by
  dsimp only
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates cubeSlice
    simp only
    rw [List.take_append,
      List.take_of_length_le (Nat.le_of_lt beforeColumn)]
    have remainingSucc : domain.columnVariables - before.length =
        (domain.columnVariables - before.length - 1) + 1 := by omega
    rw [remainingSucc]
    rfl
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates cubeSlice
    simp only
    rw [List.drop_append,
      List.drop_eq_nil_of_le (Nat.le_of_lt beforeColumn)]
    have remainingSucc : domain.columnVariables - before.length =
        (domain.columnVariables - before.length - 1) + 1 := by omega
    rw [remainingSucc]
    rfl

private theorem ofCoordinates_eq_laneSlice
    {domain : FlatNcDomain} (before after : List K)
    (columnBefore : domain.columnVariables <= before.length)
    (totalLength : before.length + 1 + after.length =
      domain.columnVariables + domain.laneVariables) (point : K) :
    let columnCoordinates := before.take domain.columnVariables
    let laneBefore := before.drop domain.columnVariables
    let columnLength : columnCoordinates.length = domain.columnVariables := by
      dsimp only [columnCoordinates]
      rw [List.length_take]
      omega
    let laneLength : laneBefore.length + 1 + after.length =
        domain.laneVariables := by
      dsimp only [laneBefore]
      rw [List.length_drop]
      omega
    Point.ofCoordinates (before ++ point :: after) (by simp; omega) = {
      column := { coordinates := columnCoordinates, dimension := columnLength }
      lane := cubeSlice laneBefore after laneLength point } := by
  dsimp only
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates
    simp only
    exact List.take_append_of_le_length columnBefore
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates cubeSlice
    simp only
    exact List.drop_append_of_le_length columnBefore

theorem sumcheckPolynomial_slice_quartic
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (before after : List K)
    (length : before.length + 1 + after.length =
      domain.columnVariables + domain.laneVariables) :
    RepresentsAtMostFour fun point =>
      sumcheckPolynomial covers data coins radix rawChildren producerBeta
        batchWeight oldColumn (before ++ point :: after) := by
  by_cases beforeColumn : before.length < domain.columnVariables
  · let columnAfter :=
      after.take (domain.columnVariables - before.length - 1)
    let laneCoordinates :=
      after.drop (domain.columnVariables - before.length - 1)
    have columnLength : before.length + 1 + columnAfter.length =
        domain.columnVariables := by
      dsimp only [columnAfter]
      rw [List.length_take]
      omega
    have laneLength : laneCoordinates.length = domain.laneVariables := by
      dsimp only [laneCoordinates]
      rw [List.length_drop]
      omega
    let lane : CubePoint K domain.laneVariables :=
      { coordinates := laneCoordinates, dimension := laneLength }
    rcases combinedAtPoint_column_quartic covers data coins radix rawChildren
      producerBeta batchWeight oldColumn lane before columnAfter columnLength
      with ⟨polynomial, represents⟩
    refine ⟨polynomial, ?_⟩
    intro point
    have arityAt : (before ++ point :: after).length =
        domain.columnVariables + domain.laneVariables := by
      simp only [List.length_append, List.length_cons]
      omega
    simp only [sumcheckPolynomial, dif_pos arityAt]
    rw [ofCoordinates_eq_columnSlice before after beforeColumn length]
    exact represents point
  · have columnBefore : domain.columnVariables <= before.length :=
      Nat.le_of_not_gt beforeColumn
    let columnCoordinates := before.take domain.columnVariables
    let laneBefore := before.drop domain.columnVariables
    have columnLength : columnCoordinates.length = domain.columnVariables := by
      dsimp only [columnCoordinates]
      rw [List.length_take]
      omega
    have laneLength : laneBefore.length + 1 + after.length =
        domain.laneVariables := by
      dsimp only [laneBefore]
      rw [List.length_drop]
      omega
    let column : CubePoint K domain.columnVariables :=
      { coordinates := columnCoordinates, dimension := columnLength }
    rcases combinedAtPoint_lane_quartic covers data coins radix rawChildren
      producerBeta batchWeight oldColumn column laneBefore after laneLength
      with ⟨polynomial, represents⟩
    refine ⟨polynomial, ?_⟩
    intro point
    have arityAt : (before ++ point :: after).length =
        domain.columnVariables + domain.laneVariables := by
      simp only [List.length_append, List.length_cons]
      omega
    simp only [sumcheckPolynomial, dif_pos arityAt]
    rw [ofCoordinates_eq_laneSlice before after columnBefore length]
    exact represents point

theorem expectedRound_quartic
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (fixed : List K) (remaining : Nat)
    (length : fixed.length + 1 + remaining =
      domain.columnVariables + domain.laneVariables) :
    RepresentsAtMostFour fun point =>
      HypercubeTruth.sumCompletions ops.toOps
        (sumcheckPolynomial covers data coins radix rawChildren producerBeta
          batchWeight oldColumn) (fixed ++ [point]) remaining := by
  apply DegreeSupport.sumCompletions_represents
  intro vertex
  have suffixLength :
      (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex).length =
        remaining :=
    SumCheckTruthPath.VertexEncoding.fieldCoordinates_length ops vertex
  rcases sumcheckPolynomial_slice_quartic covers data coins radix rawChildren
    producerBeta batchWeight oldColumn fixed
      (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex) (by
        rw [suffixLength]
        exact length) with ⟨polynomial, represents⟩
  exact ⟨polynomial, fun point => by
    simpa only [List.append_assoc, List.singleton_append] using
      represents point⟩

private theorem expectedPolynomialsFrom_representable
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (fixed challenges : List K)
    (arity : fixed.length + challenges.length =
      domain.columnVariables + domain.laneVariables) :
    ∀ expected ∈ HypercubeTruth.expectedPolynomialsFrom ops.toOps
      (sumcheckPolynomial covers data coins radix rawChildren producerBeta
        batchWeight oldColumn) fixed challenges,
      exists polynomial : Polynomial ncSumcheckDegreeBound,
        FixedPhase.Represents ops.toOps polynomial expected := by
  induction challenges generalizing fixed with
  | nil => simp [HypercubeTruth.expectedPolynomialsFrom]
  | cons challenge challenges inductionHypothesis =>
      intro expected expectedIn
      simp only [HypercubeTruth.expectedPolynomialsFrom,
        List.mem_cons] at expectedIn
      rcases expectedIn with rfl | expectedIn
      · rcases expectedRound_quartic covers data coins radix rawChildren
          producerBeta batchWeight oldColumn fixed challenges.length (by
            simp only [List.length_cons] at arity
            omega) with ⟨polynomial, represents⟩
        exact ⟨polynomial, represents⟩
      · exact inductionHypothesis (fixed := fixed ++ [challenge]) (by
          simp only [List.length_cons] at arity
          simp only [List.length_append, List.length_singleton]
          omega) expected expectedIn

theorem expectedRoundsRepresentable
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables) (point : Point domain) :
    FixedPhase.ExpectedRoundsRepresentable ops.toOps
      (sumcheckPolynomial covers data coins radix rawChildren producerBeta
        batchWeight oldColumn) ncSumcheckDegreeBound point.coordinates := by
  intro expected expectedIn
  exact expectedPolynomialsFrom_representable covers data coins radix
    rawChildren producerBeta batchWeight oldColumn [] point.coordinates
    (by simpa using point.coordinates_length) expected (by
      simpa [FixedPhase.expectedRounds, HypercubeTruth.expectedPolynomials]
        using expectedIn)

/-- The fixed 270-field public-carrier model has exactly nine scalar-column
rounds followed by six Ajtai/Phi81-lane rounds. This is not the round count of
an arbitrary full production witness. -/
theorem fixed270_roundCount :
    Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain.columnVariables +
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain.laneVariables = 15 :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_variableCount

theorem fixed270Accepted_rounds_length
    (q : List K -> K) (initial : K)
    (point : Point
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain)
    (certificate : FixedPhase.Certificate K ncSumcheckDegreeBound)
    (accepted : FixedPhase.Accepted ops.toOps q initial point.coordinates
      certificate) :
    certificate.rounds.length = 15 := by
  calc
    certificate.rounds.length = point.coordinates.length :=
      FixedPhase.Chain.rounds_length_eq_challenges_length ops.toOps initial
        (q point.coordinates) certificate.rounds point.coordinates accepted
    _ = 15 := by rw [point.coordinates_length, fixed270_roundCount]

/-! ## Fixed-phase acceptance -/

private theorem sumcheckHypercubeSum_eq_combinedHypercubeSum
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables) :
    HypercubeTruth.sumCompletions ops.toOps
        (sumcheckPolynomial covers data coins radix rawChildren producerBeta
          batchWeight oldColumn) []
        (domain.columnVariables + domain.laneVariables) =
      combinedHypercubeSum covers data coins radix rawChildren producerBeta
        batchWeight oldColumn := by
  rw [HypercubeTruth.sumCompletions_add]
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  unfold combinedHypercubeSum FiniteSumAlgebra.sumMap
  simp only [List.nil_append]
  congr 1
  apply List.map_congr_left
  intro column _
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  congr 1
  apply List.map_congr_left
  intro lane _
  exact sumcheckPolynomial_coordinates_eq_combinedAtPoint
    covers data coins radix rawChildren producerBeta batchWeight oldColumn {
      column := column.toCubePoint ops
      lane := lane.toCubePoint ops }

theorem semanticInitial_eq_ordinary_add_weightedProjection
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K)
    (oldColumn : CubePoint K domain.columnVariables) (point : Point domain) :
    FixedPhase.semanticInitial ops.toOps
        (sumcheckPolynomial covers data coins radix rawChildren producerBeta
          batchWeight oldColumn) point.coordinates.length =
      K.add (InitialSum.mixedResidualAtBeta .splitV1 covers data coins)
        (K.mul batchWeight
          (rawProjectionAtOldPoint radix rawChildren producerBeta oldColumn)) := by
  unfold FixedPhase.semanticInitial
  rw [point.coordinates_length]
  rw [sumcheckHypercubeSum_eq_combinedHypercubeSum]
  rw [combinedHypercubeSum_eq_ordinary_add_weightedProjection]
  rw [InitialSum.hypercubeSum_eq_mixedResidualAtBeta]

/-- A non-exact degree-one residual-weight comparison that nevertheless
vanishes at `batchWeight`. This definition deliberately retains the
`batchWeight = 0` degeneration. -/
def ResidualWeightRoot
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight parentProjection : K)
    (oldColumn : CubePoint K domain.columnVariables) : Prop :=
  K.mul batchWeight parentProjection =
      K.add (InitialSum.mixedResidualAtBeta .splitV1 covers data coins)
        (K.mul batchWeight
          (rawProjectionAtOldPoint radix rawChildren producerBeta oldColumn)) ∧
    ¬ (InitialSum.mixedResidualAtBeta .splitV1 covers data coins = K.zero ∧
      parentProjection =
        rawProjectionAtOldPoint radix rawChildren producerBeta oldColumn)

theorem residualWeightRoot_of_zero_weight_and_parent_mismatch
    {shape : SemanticShape} {domain : FlatNcDomain}
    (covers : domain.Covers shape) (data : Data shape)
    (coins : Mixing.Coins domain)
    (radix : F) (rawChildren : List (List F))
    (producerBeta parentProjection : K)
    (oldColumn : CubePoint K domain.columnVariables)
    (ordinaryZero : InitialSum.mixedResidualAtBeta .splitV1 covers data coins =
      K.zero)
    (parentMismatch : parentProjection ≠
      rawProjectionAtOldPoint radix rawChildren producerBeta oldColumn) :
    ResidualWeightRoot covers data coins radix rawChildren producerBeta
      K.zero parentProjection oldColumn := by
  constructor
  · rw [ordinaryZero]
    change K.mul K.zero parentProjection =
      K.add K.zero (K.mul K.zero _)
    have zeroMul (value : K) : K.mul K.zero value = K.zero := by
      calc
        _ = K.mul value K.zero := laws.mul_comm _ _
        _ = K.zero := laws.mul_zero _
    rw [zeroMul, zeroMul]
    exact (laws.zero_add K.zero).symm
  · exact fun exactParts => parentMismatch exactParts.2

/-- Accepted fixed-production flat SumCheck derives ordinary NC truth and the
raw-child old-point relation, unless one named selector/gamma/root/collision
event occurs. `parentPadding` is the still-external generated-row fact for the
ten lanes excluded from the degree-53 producer identity. -/
theorem accepted_implies_truth_and_oldPointRelation_or_badEvent
    (baseNoZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (sevenNonresidue : ConcreteCarrier.SevenProjectiveNonresidue)
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Data
      (ProductionRawChildren.Fixed270.semanticShape rowVariables freshCount
        runningCount matrixCount))
    (coins : Mixing.Coins ProductionRawChildren.Fixed270.domain)
    (radix : F) (parent : DelayedParent)
    (pointLength : parent.sCol.length =
      ProductionRawChildren.Fixed270.domain.columnVariables)
    (parentPadding : forall lane, ringDegree <= lane ->
      lane < ProductionRawChildren.Fixed270.implementationShape.laneDomain ->
      parent.yZcol lane = K.zero)
    (producerBeta batchWeight : K)
    (point : Point ProductionRawChildren.Fixed270.domain)
    (certificate : FixedPhase.Certificate K ncSumcheckDegreeBound)
    (challengeSetSize : Nat)
    (accepted : FixedPhase.Accepted ops.toOps
      (sumcheckPolynomial
        (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
          rowVariables freshCount runningCount matrixCount)
        data coins radix
        (ProductionRawChildren.Fixed270.authoritativeRunningChildren data)
        producerBeta batchWeight
        { coordinates := parent.sCol, dimension := pointLength })
      (K.mul batchWeight
        (Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
          (delayedParentActiveCoefficients parent) producerBeta))
      point.coordinates certificate) :
    (Semantics.Nc.Truth data ∧
      OldPointSumcheckRelation ProductionRawChildren.Fixed270.implementationShape
        radix parent
        (ProductionRawChildren.Fixed270.authoritativeRunningChildren data)) ∨
    SelectorRoot
      (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
        rowVariables freshCount runningCount matrixCount) data coins ∨
    GammaRoot
      (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
        rowVariables freshCount runningCount matrixCount) data coins ∨
    SplitV1GammaZero
      (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
        rowVariables freshCount runningCount matrixCount) data coins ∨
    ResidualWeightRoot
      (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
        rowVariables freshCount runningCount matrixCount) data coins radix
      (ProductionRawChildren.Fixed270.authoritativeRunningChildren data)
      producerBeta batchWeight
      (Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
        (delayedParentActiveCoefficients parent) producerBeta)
      { coordinates := parent.sCol, dimension := pointLength } ∨
    Nightstream.SuperNeo.ProjectionCheck.BadRoot projectionOps
      (projectionIdentity ProductionRawChildren.Fixed270.implementationShape
        radix (ProductionRawChildren.Fixed270.authoritativeRunningChildren data)
        parent.sCol (delayedParentActiveCoefficients parent) producerBeta) ∨
    exists round, FixedPhase.BadChallenge ops.toOps
      (sumcheckPolynomial
        (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
          rowVariables freshCount runningCount matrixCount)
        data coins radix
        (ProductionRawChildren.Fixed270.authoritativeRunningChildren data)
        producerBeta batchWeight
        { coordinates := parent.sCol, dimension := pointLength })
      ncSumcheckDegreeBound challengeSetSize
      (K.mul batchWeight
        (Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
          (delayedParentActiveCoefficients parent) producerBeta))
      point.coordinates certificate round := by
  let covers :=
    Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
      rowVariables freshCount runningCount matrixCount
  let rawChildren :=
    ProductionRawChildren.Fixed270.authoritativeRunningChildren data
  let oldColumn : CubePoint K ProductionRawChildren.Fixed270.domain.columnVariables :=
    { coordinates := parent.sCol, dimension := pointLength }
  let parentProjection := Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
    (delayedParentActiveCoefficients parent) producerBeta
  by_cases claimTrue : K.mul batchWeight parentProjection =
      FixedPhase.semanticInitial ops.toOps
        (sumcheckPolynomial covers data coins radix rawChildren producerBeta
          batchWeight oldColumn) point.coordinates.length
  · have weightedEquation : K.mul batchWeight parentProjection =
        K.add (InitialSum.mixedResidualAtBeta .splitV1 covers data coins)
          (K.mul batchWeight
            (rawProjectionAtOldPoint radix rawChildren producerBeta oldColumn)) := by
      rw [claimTrue]
      exact semanticInitial_eq_ordinary_add_weightedProjection covers data coins
        radix rawChildren producerBeta batchWeight oldColumn point
    by_cases exactParts :
        InitialSum.mixedResidualAtBeta .splitV1 covers data coins = K.zero ∧
          parentProjection = rawProjectionAtOldPoint radix rawChildren
            producerBeta oldColumn
    · rcases (splitV1_mixedResidualAtBeta_eq_zero_iff_truth_or_selectorRoot_or_gammaRoot_or_gammaZero
        baseNoZeroDivisors sevenNonresidue covers data coins).1 exactParts.1 with
        truth | selectorRoot | gammaRoot | gammaZero
      · have projectionAccepted : Nightstream.SuperNeo.ProjectionCheck.Accepted
            projectionOps (projectionIdentity
              ProductionRawChildren.Fixed270.implementationShape radix
              rawChildren parent.sCol (delayedParentActiveCoefficients parent)
              producerBeta) := by
          constructor
          · exact projectionIdentity_wellFormed _ _ _ _ _ _
              (delayedParentActiveCoefficients_length parent)
          · change parentProjection = compactOldPointEvaluation
              ProductionRawChildren.Fixed270.implementationShape radix
              rawChildren producerBeta parent.sCol
            rw [exactParts.2]
            exact rawProjectionAtOldPoint_eq_compactOldPointEvaluation
              radix rawChildren producerBeta oldColumn covers.2
        rcases ProductionRawChildren.Fixed270.acceptedProjectionIdentity_implies_oldPointRelation_or_badRoot
          data radix parent producerBeta pointLength parentPadding
          projectionAccepted with relation | badRoot
        · exact Or.inl ⟨truth, relation⟩
        · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl badRoot)))))
      · exact Or.inr (Or.inl selectorRoot)
      · exact Or.inr (Or.inr (Or.inl gammaRoot))
      · exact Or.inr (Or.inr (Or.inr (Or.inl gammaZero)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr
        (Or.inl ⟨weightedEquation, exactParts⟩))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
      (FixedPhase.false_acceptance_implies_bad_challenge ops.toOps
        (sumcheckPolynomial covers data coins radix rawChildren producerBeta
          batchWeight oldColumn) challengeSetSize
        (K.mul batchWeight parentProjection) point.coordinates certificate
        (expectedRoundsRepresentable covers data coins radix rawChildren
          producerBeta batchWeight oldColumn point) accepted claimTrue))))))

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.FlatCombinedNc
