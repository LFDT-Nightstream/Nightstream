import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.PiDEC
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.OutputBridge
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial

/-!
Production specialization of the combined-NC delayed scalar.

Assurance tier: model-level registered production deviation.

Owns: the exact production radix weights for the fourteen authoritative raw
running assignments and equality between the combined-NC old-block scalar and
the canonical degree-53 projection of their recomposed packed block value.

Does not own: output-message authority, SumCheck acceptance, transcript
sampling, one-fold continuity, commitment binding, Rust/R1CS rows, costs, or
row-removal permission.

Emits constraints: none.

Authority boundary: every child value is read from
`Sources.Data.runningAssignments` through `DelayedRawChildren`. The ten padded
Boolean lanes are derived zeros; the scalar retains exactly the 54 active
`RingK` lanes. No output claim or prover-carried `y_zcol` sidecar occurs.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.delayed_running.production_weights` | map the semantic running order bijectively to all fourteen fixed PiDEC radix weights | computed | `productionWeights` |
| `nifs.pi_ccs.nc.delayed_running.production_projection` | the padded combined-NC scalar is the degree-53 projection of the recomposed authoritative raw assignments | derived | `authoritativeRunningProjection_eq_projectedRawRecomposition` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionProjection

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Every semantic running slot receives the production PiDEC weight at the
unique aligned fixed-active child index. -/
def productionWeights
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows) :
    RunningWeights shape :=
  fun running =>
    PiDEC.radixWeight (context.alignment.productRunningIndex running)

/-! ## Small finite-algebra bridges -/

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

private theorem combineEvaluations_apply_eq_sumMap
    {count : Nat}
    (weights : Fin count -> F)
    (values : Fin count -> RingK)
    (lane : Fin ringDegree) :
    BaseLinear.combineEvaluations weights values lane =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices count) fun index =>
        K.mul (K.embed (weights index)) (values index lane) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      have indicesSucc :
          canonicalFinIndices (count + 1) =
            0 :: (canonicalFinIndices count).map Fin.succ := by
        unfold canonicalFinIndices
        rw [List.ofFn_succ]
        congr 1
        simp [Function.comp_def]
      rw [indicesSucc]
      change K.add
          (K.mul (K.embed (weights 0)) (values 0 lane))
          (BaseLinear.combineEvaluations
            (fun index => weights index.succ)
            (fun index => values index.succ) lane) =
        K.add
          (K.mul (K.embed (weights 0)) (values 0 lane))
          (FiniteSumAlgebra.sumMap ops
            ((canonicalFinIndices count).map Fin.succ) fun index =>
              K.mul (K.embed (weights index)) (values index lane))
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => values index.succ)]
      apply congrArg (K.add (K.mul (K.embed (weights 0)) (values 0 lane)))
      unfold FiniteSumAlgebra.sumMap
      rw [List.map_map]
      apply congrArg (BooleanTable.finiteSum ops)
      apply List.map_congr_left
      intro index _
      rfl

private theorem projectionEval_eq_messageEval
    (values : List K)
    (producerBeta : K) :
    ProjectionCheck.eval DelayedPackedProjection.projectionOps values
        producerBeta =
      SumCheck.Finite.Message.evaluateCoefficients ops.toOps producerBeta
        values := by
  induction values with
  | nil => rfl
  | cons value values inductionHypothesis =>
      change K.add value
          (K.mul producerBeta
            (ProjectionCheck.eval
              DelayedPackedProjection.projectionOps values producerBeta)) =
        K.add value
          (K.mul producerBeta
            (SumCheck.Finite.Message.evaluateCoefficients ops.toOps
              producerBeta values))
      rw [inductionHypothesis]

private theorem projectedValue_eq_activeLaneSum
    (value : RingK)
    (producerBeta : K) :
    DelayedPackedProjection.projectedValue value producerBeta =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree) fun lane =>
        K.mul
          (TargetPolynomial.power ops.toOps producerBeta lane.val)
          (value lane) := by
  unfold DelayedPackedProjection.projectedValue
    DelayedPackedProjection.coefficients
  rw [projectionEval_eq_messageEval]
  have canonical :
      List.ofFn value = (canonicalFinIndices ringDegree).map value := by
    simp [canonicalFinIndices]
  rw [canonical]
  simpa [SignedJointIdentity.gammaTerm] using
    (SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
      ops laws producerBeta ringDegree value)

/-! ## Exact lane and running-index partitions -/

private theorem laneIndices_perm :
    (BooleanVertex.all PiCcsDomains.production.nc.laneVariables).map
        BlockNcDomain.laneIndex
      |>.Perm
        (canonicalFinIndices PiCcsDomains.production.nc.laneCount) := by
  apply perm_of_nodup_mem_iff
  · apply
      (BooleanVertex.all_nodup
        PiCcsDomains.production.nc.laneVariables).map
        BlockNcDomain.laneIndex
    intro left right different equal
    apply different
    calc
      left = BlockNcDomain.laneVertex (BlockNcDomain.laneIndex left) :=
        (BlockNcDomain.laneVertex_laneIndex left).symm
      _ = BlockNcDomain.laneVertex (BlockNcDomain.laneIndex right) := by
        rw [equal]
      _ = right := BlockNcDomain.laneVertex_laneIndex right
  · exact canonicalFinIndices_nodup _
  · intro lane
    constructor
    · intro _
      exact List.mem_ofFn.mpr ⟨lane, rfl⟩
    · intro _
      exact List.mem_map.mpr
        ⟨BlockNcDomain.laneVertex lane, BooleanVertex.mem_all _,
          BlockNcDomain.laneIndex_laneVertex lane⟩

private theorem sumMap_vertices_eq_numeric
    (value :
      BooleanVertex PiCcsDomains.production.nc.laneVariables -> K) :
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all PiCcsDomains.production.nc.laneVariables) value =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices PiCcsDomains.production.nc.laneCount)
        (fun lane => value (BlockNcDomain.laneVertex lane)) := by
  calc
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all PiCcsDomains.production.nc.laneVariables) value =
      FiniteSumAlgebra.sumMap ops
        ((BooleanVertex.all
          PiCcsDomains.production.nc.laneVariables).map
            BlockNcDomain.laneIndex)
        (fun lane => value (BlockNcDomain.laneVertex lane)) := by
          unfold FiniteSumAlgebra.sumMap
          apply congrArg (BooleanTable.finiteSum ops)
          rw [List.map_map]
          apply List.map_congr_left
          intro lane _
          simp only [Function.comp_apply]
          rw [BlockNcDomain.laneVertex_laneIndex]
    _ = _ := sumMap_eq_of_perm laneIndices_perm _

private def isLiveLane
    (lane : Fin PiCcsDomains.production.nc.laneCount) : Bool :=
  decide (lane.val < ringDegree)

private def liveLanes :
    List (Fin PiCcsDomains.production.nc.laneCount) :=
  (canonicalFinIndices PiCcsDomains.production.nc.laneCount).filter isLiveLane

private def paddingLanes :
    List (Fin PiCcsDomains.production.nc.laneCount) :=
  (canonicalFinIndices PiCcsDomains.production.nc.laneCount).filter
    (fun lane => !(isLiveLane lane))

private theorem liveLanes_perm
    (covers : PiCcsDomains.production.nc.Covers shape) :
    (canonicalFinIndices ringDegree).map
        (PiCcsDomains.production.nc.phi81Lane covers)
      |>.Perm liveLanes := by
  apply perm_of_nodup_mem_iff
  · apply (canonicalFinIndices_nodup ringDegree).map
      (PiCcsDomains.production.nc.phi81Lane covers)
    intro left right different equal
    apply different
    apply Fin.ext
    simpa using congrArg
      (fun index : Fin PiCcsDomains.production.nc.laneCount => index.val) equal
  · exact (canonicalFinIndices_nodup _).filter _
  · intro lane
    constructor
    · intro member
      rcases List.mem_map.mp member with ⟨live, _, rfl⟩
      apply List.mem_filter.mpr
      constructor
      · exact List.mem_ofFn.mpr
          ⟨PiCcsDomains.production.nc.phi81Lane covers live, rfl⟩
      · simp [isLiveLane]
    · intro member
      have parts := List.mem_filter.mp member
      have live : lane.val < ringDegree := by
        simpa [isLiveLane] using parts.2
      let active : Fin ringDegree := ⟨lane.val, live⟩
      apply List.mem_map.mpr
      refine ⟨active, List.mem_ofFn.mpr ⟨active, rfl⟩, ?_⟩
      exact Fin.ext rfl

private theorem fullLanePartition :
    (liveLanes ++ paddingLanes).Perm
      (canonicalFinIndices PiCcsDomains.production.nc.laneCount) := by
  exact List.filter_append_perm isLiveLane
    (canonicalFinIndices PiCcsDomains.production.nc.laneCount)

private theorem sourceValueAt_booleanLane
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : CubePoint K PiCcsDomains.production.nc.blockVariables)
    (lane : Fin PiCcsDomains.production.nc.laneCount) :
    SourceProjection.sourceValueAt covers data source {
        block := block
        lane := (BlockNcDomain.laneVertex lane).toCubePoint ops } =
      SourceProjection.blockValueAt covers data source block
        (BlockNcDomain.laneVertex lane) := by
  unfold SourceProjection.sourceValueAt SourceProjection.laneTableAtBlock
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt ops laws]
  rw [BooleanTable.valueAt_tabulate]

private theorem authoritativeRunningValueAt_padding
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (producerBlock : CubePoint K PiCcsDomains.production.nc.blockVariables)
    (lane : Fin PiCcsDomains.production.nc.laneCount)
    (padding : ringDegree <= lane.val) :
    authoritativeRunningValueAt context.covers data
        (productionWeights context) {
          block := producerBlock
          lane := (BlockNcDomain.laneVertex lane).toCubePoint ops } =
      K.zero := by
  unfold authoritativeRunningValueAt
  calc
    FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.runningCount) (fun running =>
          K.mul (K.embed (productionWeights context running))
            (SourceProjection.sourceValueAt context.covers data
              (Data.runningIndex running) {
                block := producerBlock
                lane := (BlockNcDomain.laneVertex lane).toCubePoint ops })) =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.runningCount) (fun _ => K.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro running _
          rw [sourceValueAt_booleanLane]
          rw [SourceProjection.blockValueAt_lane_padding
            context.covers data (Data.runningIndex running)
            producerBlock lane padding]
          exact laws.mul_zero _
    _ = K.zero := FiniteSumAlgebra.sumMap_zero ops laws _

private theorem runningIndices_perm
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows) :
    (canonicalFinIndices shape.runningCount).map
        context.alignment.productRunningIndex
      |>.Perm (canonicalFinIndices productionGlobalParams.k) := by
  apply perm_of_nodup_mem_iff
  · apply (canonicalFinIndices_nodup shape.runningCount).map
      context.alignment.productRunningIndex
    intro left right different equal
    apply different
    calc
      left = context.alignment.semanticRunningIndex
          (context.alignment.productRunningIndex left) :=
        (SourceAlignment.semanticRunningIndex_productRunningIndex
          context.alignment left).symm
      _ = context.alignment.semanticRunningIndex
          (context.alignment.productRunningIndex right) := by rw [equal]
      _ = right :=
        SourceAlignment.semanticRunningIndex_productRunningIndex
          context.alignment right
  · exact canonicalFinIndices_nodup _
  · intro child
    constructor
    · intro _
      exact List.mem_ofFn.mpr ⟨child, rfl⟩
    · intro _
      exact List.mem_map.mpr
        ⟨context.alignment.semanticRunningIndex child,
          List.mem_ofFn.mpr
            ⟨context.alignment.semanticRunningIndex child, rfl⟩,
          SourceAlignment.productRunningIndex_semanticRunningIndex
            context.alignment child⟩

private theorem authoritativeRunningValueAt_active
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (producerBlock : CubePoint K PiCcsDomains.production.nc.blockVariables)
    (lane : Fin ringDegree) :
    authoritativeRunningValueAt context.covers data
        (productionWeights context) {
          block := producerBlock
          lane := (BlockNcDomain.laneVertex
            (PiCcsDomains.production.nc.phi81Lane context.covers lane)
              ).toCubePoint ops } =
      PackedBlockAction.packedYZcol context.covers
        (PiDEC.Raw.recomposeAssignment
          (DelayedRawChildren.rawRunningAssignments context data))
        producerBlock lane := by
  rw [PackedBlockAction.PiDEC.packedYZcol_piDecRecompose]
  rw [combineEvaluations_apply_eq_sumMap]
  unfold authoritativeRunningValueAt
  calc
    FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.runningCount) (fun running =>
          K.mul (K.embed (productionWeights context running))
            (SourceProjection.sourceValueAt context.covers data
              (Data.runningIndex running) {
                block := producerBlock
                lane := (BlockNcDomain.laneVertex
                  (PiCcsDomains.production.nc.phi81Lane context.covers lane)
                    ).toCubePoint ops })) =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.runningCount) (fun running =>
          K.mul (K.embed (productionWeights context running))
            (PackedBlockAction.packedYZcol context.covers
              (data.runningAssignments running) producerBlock lane)) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro running _
          apply congrArg (K.mul (K.embed (productionWeights context running)))
          calc
            SourceProjection.sourceValueAt context.covers data
                (Data.runningIndex running) {
                  block := producerBlock
                  lane := (BlockNcDomain.laneVertex
                    (PiCcsDomains.production.nc.phi81Lane context.covers lane)
                      ).toCubePoint ops } =
              SourceProjection.blockValueAt context.covers data
                (Data.runningIndex running) producerBlock
                (BlockNcDomain.laneVertex
                  (PiCcsDomains.production.nc.phi81Lane context.covers lane)) :=
                sourceValueAt_booleanLane context.covers data
                  (Data.runningIndex running) producerBlock _
            _ = PackedBlockAction.packedYZcol context.covers
                (data.assignment (Data.runningIndex running)) producerBlock
                lane :=
              (OutputBridge.packedYZcol_lane_eq_blockValueAt
                context.covers data (Data.runningIndex running)
                producerBlock lane).symm
            _ = PackedBlockAction.packedYZcol context.covers
                (data.runningAssignments running) producerBlock lane := by
              rw [data.assignment_runningIndex]
    _ = FiniteSumAlgebra.sumMap ops
        ((canonicalFinIndices shape.runningCount).map
          context.alignment.productRunningIndex) (fun child =>
          K.mul (K.embed (PiDEC.radixWeight child))
            (PackedBlockAction.packedYZcol context.covers
              (DelayedRawChildren.rawRunningAssignments context data child)
              producerBlock lane)) := by
          unfold FiniteSumAlgebra.sumMap
          apply congrArg (BooleanTable.finiteSum ops)
          rw [List.map_map]
          apply List.map_congr_left
          intro running _
          simp [productionWeights, DelayedRawChildren.rawRunningAssignments,
            DelayedRawChildren.rawRunningAssignment]
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices productionGlobalParams.k) (fun child =>
          K.mul (K.embed (PiDEC.radixWeight child))
            (PackedBlockAction.packedYZcol context.covers
              (DelayedRawChildren.rawRunningAssignments context data child)
              producerBlock lane)) :=
        sumMap_eq_of_perm (runningIndices_perm context) _

/-! ## Active 54-lane projection -/

private def fullLaneTerm
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (producerBeta : K)
    (oldBlock : CubePoint K PiCcsDomains.production.nc.blockVariables)
    (lane : Fin PiCcsDomains.production.nc.laneCount) : K :=
  K.mul
    (TargetPolynomial.power ops.toOps producerBeta lane.val)
    (authoritativeRunningValueAt context.covers data
      (productionWeights context) {
        block := oldBlock
        lane := (BlockNcDomain.laneVertex lane).toCubePoint ops })

private theorem fullLaneTerm_padding
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (producerBeta : K)
    (oldBlock : CubePoint K PiCcsDomains.production.nc.blockVariables)
    (lane : Fin PiCcsDomains.production.nc.laneCount)
    (padding : ringDegree <= lane.val) :
    fullLaneTerm context data producerBeta oldBlock lane = K.zero := by
  unfold fullLaneTerm
  rw [authoritativeRunningValueAt_padding
    context data oldBlock lane padding]
  exact laws.mul_zero _

private theorem paddingLaneSum_eq_zero
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (producerBeta : K)
    (oldBlock : CubePoint K PiCcsDomains.production.nc.blockVariables) :
    FiniteSumAlgebra.sumMap ops paddingLanes
        (fullLaneTerm context data producerBeta oldBlock) = K.zero := by
  calc
    FiniteSumAlgebra.sumMap ops paddingLanes
        (fullLaneTerm context data producerBeta oldBlock) =
      FiniteSumAlgebra.sumMap ops paddingLanes (fun _ => K.zero) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro lane member
        have parts := List.mem_filter.mp member
        have padding : ringDegree <= lane.val := by
          apply Nat.le_of_not_gt
          simpa [isLiveLane] using parts.2
        exact fullLaneTerm_padding context data producerBeta oldBlock
          lane padding
    _ = K.zero := FiniteSumAlgebra.sumMap_zero ops laws _

private theorem authoritativeRunningProjection_eq_activeLaneSum
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (producerBeta : K)
    (oldBlock : CubePoint K PiCcsDomains.production.nc.blockVariables) :
    authoritativeRunningProjection context.covers data
        (productionWeights context) producerBeta oldBlock =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree) fun lane =>
        K.mul (TargetPolynomial.power ops.toOps producerBeta lane.val)
          (authoritativeRunningValueAt context.covers data
            (productionWeights context) {
              block := oldBlock
              lane := (BlockNcDomain.laneVertex
                (PiCcsDomains.production.nc.phi81Lane context.covers lane)
                  ).toCubePoint ops }) := by
  unfold authoritativeRunningProjection
  calc
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all PiCcsDomains.production.nc.laneVariables)
        (fun lane =>
          K.mul (TargetPolynomial.power ops.toOps producerBeta
              (BlockNcDomain.laneIndex lane).val)
            (authoritativeRunningValueAt context.covers data
              (productionWeights context) {
                block := oldBlock
                lane := lane.toCubePoint ops })) =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices PiCcsDomains.production.nc.laneCount)
        (fullLaneTerm context data producerBeta oldBlock) := by
          rw [sumMap_vertices_eq_numeric]
          apply FiniteSumAlgebra.sumMap_congr
          intro lane _
          simp [fullLaneTerm]
    _ = FiniteSumAlgebra.sumMap ops (liveLanes ++ paddingLanes)
        (fullLaneTerm context data producerBeta oldBlock) :=
      (sumMap_eq_of_perm fullLanePartition _).symm
    _ = K.add
        (FiniteSumAlgebra.sumMap ops liveLanes
          (fullLaneTerm context data producerBeta oldBlock))
        (FiniteSumAlgebra.sumMap ops paddingLanes
          (fullLaneTerm context data producerBeta oldBlock)) :=
      sumMap_append liveLanes paddingLanes _
    _ = FiniteSumAlgebra.sumMap ops liveLanes
        (fullLaneTerm context data producerBeta oldBlock) := by
      rw [paddingLaneSum_eq_zero]
      exact laws.add_zero _
    _ = FiniteSumAlgebra.sumMap ops
        ((canonicalFinIndices ringDegree).map
          (PiCcsDomains.production.nc.phi81Lane context.covers))
        (fullLaneTerm context data producerBeta oldBlock) :=
      (sumMap_eq_of_perm (liveLanes_perm context.covers) _).symm
    _ = FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane =>
          K.mul (TargetPolynomial.power ops.toOps producerBeta lane.val)
            (authoritativeRunningValueAt context.covers data
              (productionWeights context) {
                block := oldBlock
                lane := (BlockNcDomain.laneVertex
                  (PiCcsDomains.production.nc.phi81Lane context.covers lane)
                    ).toCubePoint ops })) := by
      unfold FiniteSumAlgebra.sumMap
      apply congrArg (BooleanTable.finiteSum ops)
      rw [List.map_map]
      apply List.map_congr_left
      intro lane _
      simp [fullLaneTerm]

/-- Exact production scalar bridge. The left side traverses the padded
block-lane source table, but its ten padding lanes are proved zero. The right
side recomposes all fourteen raw running assignments with the verifier-fixed
PiDEC weights and projects exactly its 54 active coefficients. -/
theorem authoritativeRunningProjection_eq_projectedRawRecomposition
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (producerBeta : K)
    (oldBlock : CubePoint K PiCcsDomains.production.nc.blockVariables) :
    authoritativeRunningProjection context.covers data
        (productionWeights context) producerBeta oldBlock =
      DelayedPackedProjection.projectedValue
        (PackedBlockAction.packedYZcol context.covers
          (PiDEC.Raw.recomposeAssignment
            (DelayedRawChildren.rawRunningAssignments context data))
          oldBlock)
        producerBeta := by
  rw [authoritativeRunningProjection_eq_activeLaneSum]
  rw [projectedValue_eq_activeLaneSum]
  apply FiniteSumAlgebra.sumMap_congr
  intro lane _
  apply congrArg
    (K.mul (TargetPolynomial.power ops.toOps producerBeta lane.val))
  exact authoritativeRunningValueAt_active context data oldBlock lane

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionProjection
