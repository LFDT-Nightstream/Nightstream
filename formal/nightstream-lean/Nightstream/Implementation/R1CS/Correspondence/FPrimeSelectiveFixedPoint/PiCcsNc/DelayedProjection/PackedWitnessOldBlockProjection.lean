import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitness
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.PiDEC
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction

/-!
Native packed-witness projection refinement for the fixed production profile.

Assurance tier: model-level until the generated execution audit instantiates
the concrete witness matrices, old block point, and projected values.

Owns: the exact loop implemented by
`project_raw_witnesses_at_block_point`: 211,797 packed blocks in increasing
order, little-endian tensor weights at the 19-coordinate old block point,
54 witness-backed lanes, and ten verifier-computed zero lanes. It also proves
that child-major radix recomposition of those projections is the independent
packed projection of the complete PiDEC-recomposed assignment.

Does not own: capture of `RunningInstance.witnesses`, generated fixture
values, terminal acceptance, commitment binding, transcript scheduling,
`y_ring`, costs, or row-removal permission.

Emits constraints: none; executable-algorithm correspondence only.

| Stable stage path | Obligation | Authority class | Rust owner |
|---|---|---|---|
| `f_prime.pi_ccs_nc.raw_old_block.weights` | use the production little-endian tensor weight for every packed block | computed | `tensor_point_parallel` |
| `f_prime.pi_ccs_nc.raw_old_block.active` | project every actual `WitnessMat` row over all 211,797 packed columns | direct dataflow | `project_raw_witnesses_at_block_point` |
| `f_prime.pi_ccs_nc.raw_old_block.padding` | lanes 54 through 63 are verifier-computed zero | computed | execution-audit exporter |
| `f_prime.pi_ccs_nc.raw_old_block.radix` | recompose fourteen ordered children with production powers of two | derived | `radix_recompose_raw_witnesses_at_block_point` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessOldBlockProjection

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open BlockNcDomain
open PackedWitness

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws
private abbrev productionShape := ProductionDomain.semanticShape
private abbrev productionDomain := PiCcsDomains.production.nc
private abbrev productionCovers := ProductionDomain.blockLaneDomain_covers

/-- One term of the literal native packed-column loop. The multiplication
order matches Rust's `K::from(digit) * weights[block]`. -/
def nativeProjectionTerm
    (witness : Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables)
    (lane : Fin ringDegree)
    (block : Fin (Phi81ColumnLayout.blockCount
      productionShape.carrierWidth)) : K :=
  K.mul
    (K.embed (witness lane (rustBlockOfSemantic block)))
    (NumericBooleanDomain.testBitWeight ops point
      (BlockNcDomain.carrierBlock productionCovers block))

/-- The exact active-lane loop used by the native old-block projection.
Only live packed columns are traversed; the Boolean-domain suffix is
verifier-computed zero. -/
def nativeProjectedLane
    (witness : Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables)
    (lane : Fin ringDegree) : K :=
  (canonicalFinIndices
      (Phi81ColumnLayout.blockCount productionShape.carrierWidth)).foldl
    (fun accumulated block =>
      K.add accumulated (nativeProjectionTerm witness point lane block))
    K.zero

/-- All 54 values returned for one native raw child. -/
def nativeProjectedChild
    (witness : Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables) : RingK :=
  fun lane => nativeProjectedLane witness point lane

/-- The 64-lane execution-audit view: the 54 native values followed by ten
computed zeros. No padded lane reads a witness cell. -/
def paddedProjectedChild
    (witness : Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables)
    (lane : Fin productionDomain.laneCount) : K :=
  if live : lane.val < ringDegree then
    nativeProjectedLane witness point ⟨lane.val, live⟩
  else
    K.zero

@[simp] theorem paddedProjectedChild_active
    (witness : Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables)
    (lane : Fin ringDegree) :
    paddedProjectedChild witness point
        (BlockNcDomain.phi81Lane productionCovers lane) =
      nativeProjectedChild witness point lane := by
  simp [paddedProjectedChild, nativeProjectedChild]

theorem paddedProjectedChild_padding
    (witness : Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables)
    (lane : Fin productionDomain.laneCount)
    (padding : ringDegree <= lane.val) :
    paddedProjectedChild witness point lane = K.zero := by
  simp [paddedProjectedChild, Nat.not_lt.mpr padding]

/-! ## Finite-sum and index bridges -/

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
    ops.add right (ops.add left tail) = ops.add (ops.add right left) tail :=
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
    indices.foldl
        (fun accumulated index => ops.add accumulated (value index))
        ops.zero =
      FiniteSumAlgebra.sumMap ops indices value := by
  have withInitial : forall initial,
      indices.foldl
          (fun accumulated index => ops.add accumulated (value index))
          initial =
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

private theorem blockIndices_perm :
    (BooleanVertex.all productionDomain.blockVariables).map blockIndex
      |>.Perm (canonicalFinIndices productionDomain.blockCount) := by
  apply perm_of_nodup_mem_iff
  · apply (BooleanVertex.all_nodup productionDomain.blockVariables).map
      blockIndex
    intro left right different equal
    apply different
    calc
      left = blockVertex (blockIndex left) :=
        (blockVertex_blockIndex left).symm
      _ = blockVertex (blockIndex right) := by rw [equal]
      _ = right := blockVertex_blockIndex right
  · exact canonicalFinIndices_nodup productionDomain.blockCount
  · intro block
    constructor
    · intro _
      exact List.mem_ofFn.mpr ⟨block, rfl⟩
    · intro _
      exact List.mem_map.mpr
        ⟨blockVertex block, BooleanVertex.mem_all _,
          blockIndex_blockVertex block⟩

private def isLiveBlock (block : Fin productionDomain.blockCount) : Bool :=
  decide (block.val <
    Phi81ColumnLayout.blockCount productionShape.carrierWidth)

private def liveBlocks : List (Fin productionDomain.blockCount) :=
  (canonicalFinIndices productionDomain.blockCount).filter isLiveBlock

private def paddingBlocks : List (Fin productionDomain.blockCount) :=
  (canonicalFinIndices productionDomain.blockCount).filter
    (fun block => !(isLiveBlock block))

private theorem liveBlocks_perm :
    (canonicalFinIndices
        (Phi81ColumnLayout.blockCount productionShape.carrierWidth)).map
        (BlockNcDomain.carrierBlock productionCovers)
      |>.Perm liveBlocks := by
  apply perm_of_nodup_mem_iff
  · apply (canonicalFinIndices_nodup
      (Phi81ColumnLayout.blockCount productionShape.carrierWidth)).map
      (BlockNcDomain.carrierBlock productionCovers)
    intro left right different equal
    apply different
    apply Fin.ext
    simpa using congrArg
      (fun block : Fin productionDomain.blockCount => block.val) equal
  · exact (canonicalFinIndices_nodup _).filter _
  · intro block
    constructor
    · intro member
      rcases List.mem_map.mp member with ⟨live, _, rfl⟩
      apply List.mem_filter.mpr
      exact ⟨List.mem_ofFn.mpr
        ⟨BlockNcDomain.carrierBlock productionCovers live, rfl⟩,
        by
          simp only [isLiveBlock, decide_eq_true_eq,
            BlockNcDomain.carrierBlock_val]
          simpa only [ProductionDomain.semanticShape_carrierWidth] using
            live.isLt⟩
    · intro member
      have parts := List.mem_filter.mp member
      have live : block.val <
          Phi81ColumnLayout.blockCount productionShape.carrierWidth := by
        simpa [isLiveBlock] using parts.2
      let sourceBlock : Fin
          (Phi81ColumnLayout.blockCount productionShape.carrierWidth) :=
        ⟨block.val, live⟩
      apply List.mem_map.mpr
      refine ⟨sourceBlock, List.mem_ofFn.mpr ⟨sourceBlock, rfl⟩, ?_⟩
      exact Fin.ext rfl

private theorem fullBlocks_partition :
    (liveBlocks ++ paddingBlocks).Perm
      (canonicalFinIndices productionDomain.blockCount) := by
  exact List.filter_append_perm isLiveBlock
    (canonicalFinIndices productionDomain.blockCount)

private def fullBlockTerm
    (witness : Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables)
    (lane : Fin ringDegree)
    (block : Fin productionDomain.blockCount) : K :=
  K.mul
    (PackedBlockAction.blockRows (unpack witness)
      (blockVertex block) lane)
    (NumericBooleanDomain.testBitWeight ops point block)

private theorem fullBlockTerm_padding
    (witness : Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables)
    (lane : Fin ringDegree)
    (block : Fin productionDomain.blockCount)
    (padding : Phi81ColumnLayout.blockCount
      productionShape.carrierWidth <= block.val) :
    fullBlockTerm witness point lane block = K.zero := by
  unfold fullBlockTerm PackedBlockAction.blockRows
  simp only [blockIndex_blockVertex]
  rw [dif_neg (Nat.not_lt.mpr padding)]
  change K.mul K.zero _ = K.zero
  calc
    _ = K.mul _ K.zero := laws.mul_comm _ _
    _ = K.zero := laws.mul_zero _

private theorem fullBlockTerm_live
    (witness : Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables)
    (lane : Fin ringDegree)
    (block : Fin (Phi81ColumnLayout.blockCount
      productionShape.carrierWidth)) :
    fullBlockTerm witness point lane
        (BlockNcDomain.carrierBlock productionCovers block) =
      nativeProjectionTerm witness point lane block := by
  have aligned : CoordinatesAligned witness (unpack witness) :=
    (coordinatesAligned_iff_unpack_eq witness (unpack witness)).2 rfl
  have cell :
      unpack witness (Phi81CarrierLayout.carrierColumn block lane) =
        witness lane (rustBlockOfSemantic block) := by
    simpa using
      (aligned lane (rustBlockOfSemantic block)).symm
  unfold fullBlockTerm nativeProjectionTerm PackedBlockAction.blockRows
  simp only [blockIndex_blockVertex, BlockNcDomain.carrierBlock_val]
  rw [dif_pos block.isLt]
  change K.mul
      (K.embed (unpack witness
        (Phi81CarrierLayout.carrierColumn block lane))) _ =
    K.mul (K.embed (witness lane (rustBlockOfSemantic block))) _
  rw [cell]

/-- The literal native packed-column loop is exactly the independent packed
projection for every 54-lane full witness and every verifier-owned production
block point. -/
theorem nativeProjectedLane_eq_packedYZcol
    (witness : Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables)
    (lane : Fin ringDegree) :
    nativeProjectedLane witness point lane =
      PackedBlockAction.packedYZcol productionCovers (unpack witness)
        point lane := by
  symm
  unfold PackedBlockAction.packedYZcol RingKAction.evaluateRows
  calc
    (BooleanTable.tabulate fun vertex =>
        PackedBlockAction.blockRows (unpack witness) vertex lane).evaluate
          ops point =
      BooleanReproduction.equalityWeighted ops point (fun vertex =>
        PackedBlockAction.blockRows (unpack witness) vertex lane) :=
      (BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
        ops laws point _).symm
    _ = FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all productionDomain.blockVariables)
        (fun vertex => fullBlockTerm witness point lane
          (blockIndex vertex)) := by
      unfold BooleanReproduction.equalityWeighted
      apply FiniteSumAlgebra.sumMap_congr
      intro vertex _
      unfold fullBlockTerm
      rw [blockVertex_blockIndex]
      have weightEq :
          BooleanVertex.equalityWeight ops vertex point =
            NumericBooleanDomain.testBitWeight ops point
              (blockIndex vertex) := by
        calc
          BooleanVertex.equalityWeight ops vertex point =
              BooleanVertex.equalityWeight ops
                (blockVertex (blockIndex vertex)) point := by
            rw [blockVertex_blockIndex]
          _ = NumericBooleanDomain.tensorWeight ops
                (blockIndex vertex) point :=
            (NumericBooleanDomain.tensorWeight_eq_equalityWeight
              ops (blockIndex vertex) point).symm
          _ = NumericBooleanDomain.testBitWeight ops point
                (blockIndex vertex) :=
            NumericBooleanDomain.tensorWeight_eq_testBitWeight ops
              (NumericBooleanDomain.WeightProductLaws.ofInterpolationEvaluationLaws
                laws)
              (blockIndex vertex) point
      rw [weightEq]
      exact laws.mul_comm _ _
    _ = FiniteSumAlgebra.sumMap ops
        ((BooleanVertex.all productionDomain.blockVariables).map blockIndex)
        (fullBlockTerm witness point lane) := by
      simp [FiniteSumAlgebra.sumMap, Function.comp_def]
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices productionDomain.blockCount)
        (fullBlockTerm witness point lane) :=
      sumMap_eq_of_perm blockIndices_perm _
    _ = FiniteSumAlgebra.sumMap ops (liveBlocks ++ paddingBlocks)
        (fullBlockTerm witness point lane) :=
      sumMap_eq_of_perm fullBlocks_partition.symm _
    _ = K.add
        (FiniteSumAlgebra.sumMap ops liveBlocks
          (fullBlockTerm witness point lane))
        (FiniteSumAlgebra.sumMap ops paddingBlocks
          (fullBlockTerm witness point lane)) :=
      sumMap_append _ _ _
    _ = FiniteSumAlgebra.sumMap ops liveBlocks
        (fullBlockTerm witness point lane) := by
      have paddingZero :
          FiniteSumAlgebra.sumMap ops paddingBlocks
              (fullBlockTerm witness point lane) = K.zero := by
        calc
          _ = FiniteSumAlgebra.sumMap ops paddingBlocks
              (fun _ => K.zero) := by
            apply FiniteSumAlgebra.sumMap_congr
            intro block member
            have parts := List.mem_filter.mp member
            have padding : Phi81ColumnLayout.blockCount
                productionShape.carrierWidth <= block.val := by
              simpa [paddingBlocks, isLiveBlock] using parts.2
            exact fullBlockTerm_padding witness point lane block padding
          _ = K.zero := FiniteSumAlgebra.sumMap_zero ops laws _
      rw [paddingZero]
      exact laws.add_zero _
    _ = FiniteSumAlgebra.sumMap ops
        ((canonicalFinIndices
          (Phi81ColumnLayout.blockCount productionShape.carrierWidth)).map
            (BlockNcDomain.carrierBlock productionCovers))
        (fullBlockTerm witness point lane) :=
      sumMap_eq_of_perm liveBlocks_perm.symm _
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices
          (Phi81ColumnLayout.blockCount productionShape.carrierWidth))
        (nativeProjectionTerm witness point lane) := by
      unfold FiniteSumAlgebra.sumMap
      rw [List.map_map]
      apply congrArg (BooleanTable.finiteSum ops)
      apply List.map_congr_left
      intro block _
      exact fullBlockTerm_live witness point lane block
    _ = nativeProjectedLane witness point lane := by
      unfold nativeProjectedLane
      exact (foldl_add_eq_sumMap
        (canonicalFinIndices
          (Phi81ColumnLayout.blockCount productionShape.carrierWidth))
        (nativeProjectionTerm witness point lane)).symm

theorem nativeProjectedChild_eq_packedYZcol
    (witness : Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables) :
    nativeProjectedChild witness point =
      PackedBlockAction.packedYZcol productionCovers (unpack witness) point := by
  funext lane
  exact nativeProjectedLane_eq_packedYZcol witness point lane

/-- Child-major native recomposition with the production PiDEC weights. -/
def nativeRadixRecomposition
    (witnesses : Fin productionGlobalParams.k -> Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables) : RingK :=
  BaseLinear.combineEvaluations PiDEC.radixWeight fun child =>
    nativeProjectedChild (witnesses child) point

/-- The native child projections and their ordered powers-of-two
recomposition equal the packed projection of the complete authoritative raw
PiDEC assignment. -/
theorem nativeRadixRecomposition_eq_packedYZcol
    (witnesses : Fin productionGlobalParams.k -> Matrix productionShape)
    (point : CubePoint K productionDomain.blockVariables) :
    nativeRadixRecomposition witnesses point =
      PackedBlockAction.packedYZcol productionCovers
        (PiDEC.Raw.recomposeAssignment fun child =>
          unpack (witnesses child)) point := by
  calc
    nativeRadixRecomposition witnesses point =
      BaseLinear.combineEvaluations PiDEC.radixWeight (fun child =>
        PackedBlockAction.packedYZcol productionCovers
          (unpack (witnesses child)) point) := by
        unfold nativeRadixRecomposition
        apply congrArg (BaseLinear.combineEvaluations PiDEC.radixWeight)
        funext child
        exact nativeProjectedChild_eq_packedYZcol (witnesses child) point
    _ = PackedBlockAction.packedYZcol productionCovers
        (PiDEC.Raw.recomposeAssignment fun child =>
          unpack (witnesses child)) point :=
      (PackedBlockAction.PiDEC.packedYZcol_piDecRecompose productionCovers
        (fun child => unpack (witnesses child)) point).symm

/-- Fixed production cardinalities owned by this native refinement leaf. -/
theorem production_projection_cardinalities :
    productionDomain.blockVariables = 19 /\
      Phi81ColumnLayout.blockCount productionShape.carrierWidth = 211797 /\
      ringDegree = 54 /\
      productionDomain.laneCount - ringDegree = 10 /\
      productionGlobalParams.k = 14 := by
  decide

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessOldBlockProjection
