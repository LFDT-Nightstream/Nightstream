import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath

/-!
Source-derived polynomial for canonical Split-NC block×lane checking.

Assurance tier: model-level.

Owns: the padded block/lane table derived from authoritative assignments;
zero block and lane tails; canonical block-then-lane multilinear evaluation;
the strict-`b = 2` cubic; and equivalence of its Boolean restriction with
independent full-carrier NC truth.

Does not own: mixing, an initial claim, SumCheck messages, transcript coins,
`yZcol`, terminal binding, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: every live leaf is computed from
`Sources.Data.assignment`. Blocks after the complete Phi81 carrier and lanes
after coefficient 53 are computed as zero; neither tail is prover supplied.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.source.live` | each live block/lane leaf is the authoritative carrier coefficient | computed | `paddedValue_live` |
| `nifs.pi_ccs.nc.block_lane.source.padding.block` | padded blocks are zero | computed | `paddedValue_block_padding` |
| `nifs.pi_ccs.nc.block_lane.source.padding.lane` | padded lanes are zero | computed | `paddedValue_lane_padding` |
| `nifs.pi_ccs.nc.block_lane.source.mle` | block interpolation precedes lane interpolation | computed | `sourceValueAt` |
| `nifs.pi_ccs.nc.block_lane.source.mle.lane_padding` | every padded lane has zero block MLE at arbitrary block points | derived | `blockValueAt_lane_padding` |
| `nifs.pi_ccs.nc.block_lane.source.boolean` | the MLE restricts to the exact padded table | derived | `sourceValueAt_toCubePoint_eq_embed_paddedValue` |
| `nifs.pi_ccs.nc.block_lane.range` | the Boolean restriction is the embedded semantic cubic | derived | `rangeValueAt_toCubePoint_eq_embed_cubicResidual` |
| `nifs.pi_ccs.nc.block_lane.range.complete` | independent NC truth zeros the entire padded table | derived | `booleanResidualsZero_of_truth` |
| `nifs.pi_ccs.nc.block_lane.range.sound` | zero padded cubics imply independent NC truth | checked | `truth_of_booleanResidualsZero` |
| `nifs.pi_ccs.nc.block_lane.range.exact` | padded block×lane cubics are equivalent to independent NC truth | derived | `booleanResidualsZero_iff_truth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.SourceProjection

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open BlockNcDomain

/-- Complete padded block×lane table. Live cells read the authoritative
assignment; both Boolean suffixes are canonical zero. -/
def paddedValue
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (_covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : Fin domain.blockCount)
    (lane : Fin domain.laneCount) : F :=
  if blockLive : block.val <
      Phi81ColumnLayout.blockCount shape.carrierWidth then
    if laneLive : lane.val < ringDegree then
      Semantics.Nc.BlockLane.value (data.assignment source)
        ⟨block.val, blockLive⟩ ⟨lane.val, laneLive⟩
    else
      0
  else
    0

/-- Embedding a live block and lane preserves the exact authoritative value. -/
theorem paddedValue_live
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) :
    paddedValue covers data source
        (domain.carrierBlock covers block)
        (domain.phi81Lane covers lane) =
      Semantics.Nc.BlockLane.value
        (data.assignment source) block lane := by
  simp [paddedValue]

/-- Every padded block after the complete carrier is zero in every lane. -/
theorem paddedValue_block_padding
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : Fin domain.blockCount)
    (lane : Fin domain.laneCount)
    (padding :
      Phi81ColumnLayout.blockCount shape.carrierWidth <= block.val) :
    paddedValue covers data source block lane = 0 := by
  simp [paddedValue, Nat.not_lt.mpr padding]

/-- Every padded lane after the 54 active coefficients is zero in every block. -/
theorem paddedValue_lane_padding
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : Fin domain.blockCount)
    (lane : Fin domain.laneCount)
    (padding : ringDegree <= lane.val) :
    paddedValue covers data source block lane = 0 := by
  simp [paddedValue, Nat.not_lt.mpr padding]

/-- Canonical block table for one Boolean lane. -/
def blockTable
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (lane : BooleanVertex domain.laneVariables) :
    BooleanTable K domain.blockVariables :=
  BooleanTable.tabulate fun block =>
    K.embed <| paddedValue covers data source
      (blockIndex block) (laneIndex lane)

/-- Evaluate the block table before performing lane interpolation. -/
def blockValueAt
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : CubePoint K domain.blockVariables)
    (lane : BooleanVertex domain.laneVariables) : K :=
  (blockTable covers data source lane).evaluate
    ConcreteCarrier.extensionOps block

/-- A padded Boolean lane has zero block MLE at every, including non-Boolean,
block point. -/
theorem blockValueAt_lane_padding
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : CubePoint K domain.blockVariables)
    (lane : Fin domain.laneCount)
    (padding : ringDegree ≤ lane.val) :
    blockValueAt covers data source block (laneVertex lane) = K.zero := by
  unfold blockValueAt blockTable
  have tableZero :
      BooleanTable.tabulate (fun vertex =>
          K.embed (paddedValue covers data source
            (blockIndex vertex) (laneIndex (laneVertex lane)))) =
        BooleanTable.tabulate
          (fun _ : BooleanVertex domain.blockVariables => K.zero) := by
    apply congrArg BooleanTable.tabulate
    funext vertex
    rw [laneIndex_laneVertex]
    rw [paddedValue_lane_padding covers data source
      (blockIndex vertex) lane padding]
    exact ConcreteCarrier.embed_zero
  rw [tableZero]
  exact BooleanReproduction.evaluate_tabulate_constant
    ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws K.zero block

/-- Lane table whose leaves are already-evaluated block MLEs. -/
def laneTableAtBlock
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : CubePoint K domain.blockVariables) :
    BooleanTable K domain.laneVariables :=
  BooleanTable.tabulate fun lane =>
    blockValueAt covers data source block lane

/-- Nested canonical MLE of the complete padded source table. -/
def sourceValueAt
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (point : Point domain) : K :=
  (laneTableAtBlock covers data source point.block).evaluate
    ConcreteCarrier.extensionOps point.lane

/-- At Boolean block and lane points, the nested MLE returns the exact leaf. -/
theorem sourceValueAt_toCubePoint_eq_embed_paddedValue
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : BooleanVertex domain.blockVariables)
    (lane : BooleanVertex domain.laneVariables) :
    sourceValueAt covers data source {
        block := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps block
        lane := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps lane } =
      K.embed (paddedValue covers data source
        (blockIndex block) (laneIndex lane)) := by
  unfold sourceValueAt laneTableAtBlock
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt
    ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws]
  rw [BooleanTable.valueAt_tabulate]
  unfold blockValueAt blockTable
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt
    ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws]
  rw [BooleanTable.valueAt_tabulate]

/-- Numeric padded indices use the same Boolean restriction. -/
theorem sourceValueAt_booleanPoint_eq_embed_paddedValue
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : Fin domain.blockCount)
    (lane : Fin domain.laneCount) :
    sourceValueAt covers data source (booleanPoint block lane) =
      K.embed (paddedValue covers data source block lane) := by
  simpa [booleanPoint] using
    sourceValueAt_toCubePoint_eq_embed_paddedValue
      covers data source (blockVertex block) (laneVertex lane)

/-- A live Boolean point evaluates to the authoritative carrier coefficient. -/
theorem sourceValueAt_live
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) :
    sourceValueAt covers data source
        (booleanPoint (domain.carrierBlock covers block)
          (domain.phi81Lane covers lane)) =
      K.embed (Semantics.Nc.BlockLane.value
        (data.assignment source) block lane) := by
  rw [sourceValueAt_booleanPoint_eq_embed_paddedValue]
  rw [paddedValue_live]

/-- A padded block evaluates to extension zero. -/
theorem sourceValueAt_block_padding
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : Fin domain.blockCount)
    (lane : Fin domain.laneCount)
    (padding :
      Phi81ColumnLayout.blockCount shape.carrierWidth <= block.val) :
    sourceValueAt covers data source (booleanPoint block lane) = K.zero := by
  rw [sourceValueAt_booleanPoint_eq_embed_paddedValue]
  rw [paddedValue_block_padding covers data source block lane padding]
  exact ConcreteCarrier.embed_zero

/-- A padded lane evaluates to extension zero. -/
theorem sourceValueAt_lane_padding
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : Fin domain.blockCount)
    (lane : Fin domain.laneCount)
    (padding : ringDegree <= lane.val) :
    sourceValueAt covers data source (booleanPoint block lane) = K.zero := by
  rw [sourceValueAt_booleanPoint_eq_embed_paddedValue]
  rw [paddedValue_lane_padding covers data source block lane padding]
  exact ConcreteCarrier.embed_zero

/-- Strict-`b = 2` cubic evaluated after block×lane interpolation. -/
def rangeValueAt
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (point : Point domain) : K :=
  let value := sourceValueAt covers data source point
  K.mul (K.mul (K.add value (K.embed 1)) value)
    (K.sub value (K.embed 1))

/-- On the Boolean product cube, the extension cubic is the embedding of the
independently defined base-field residual. -/
theorem rangeValueAt_toCubePoint_eq_embed_cubicResidual
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : BooleanVertex domain.blockVariables)
    (lane : BooleanVertex domain.laneVariables) :
    rangeValueAt covers data source {
        block := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps block
        lane := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps lane } =
      K.embed (NormRange.cubicResidual
        (paddedValue covers data source
          (blockIndex block) (laneIndex lane))) := by
  unfold rangeValueAt
  rw [sourceValueAt_toCubePoint_eq_embed_paddedValue]
  exact NormRange.embed_cubicResidual _

/-- Numeric padded indices satisfy the same cubic restriction. -/
theorem rangeValueAt_booleanPoint_eq_embed_cubicResidual
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : Fin domain.blockCount)
    (lane : Fin domain.laneCount) :
    rangeValueAt covers data source (booleanPoint block lane) =
      K.embed (NormRange.cubicResidual
        (paddedValue covers data source block lane)) := by
  simpa [booleanPoint] using
    rangeValueAt_toCubePoint_eq_embed_cubicResidual
      covers data source (blockVertex block) (laneVertex lane)

/-- At a live point the cubic is exactly the authoritative block×lane cubic. -/
theorem rangeValueAt_live
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) :
    rangeValueAt covers data source
        (booleanPoint (domain.carrierBlock covers block)
          (domain.phi81Lane covers lane)) =
      K.embed (NormRange.cubicResidual
        (Semantics.Nc.BlockLane.value
          (data.assignment source) block lane)) := by
  rw [rangeValueAt_booleanPoint_eq_embed_cubicResidual]
  rw [paddedValue_live]

/-- Every cubic vanishes on the exact padded Boolean product domain. -/
def BooleanResidualsZero
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape) : Prop :=
  forall source block lane,
    rangeValueAt covers data source (booleanPoint block lane) = K.zero

/-- Independent NC truth zeros every live and padded block×lane cubic. -/
theorem booleanResidualsZero_of_truth
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape) :
    Semantics.Nc.Truth data -> BooleanResidualsZero covers data := by
  intro truth source block lane
  rw [rangeValueAt_booleanPoint_eq_embed_cubicResidual]
  have residualZero :
      NormRange.cubicResidual
          (paddedValue covers data source block lane) = 0 := by
    by_cases blockLive : block.val <
        Phi81ColumnLayout.blockCount shape.carrierWidth
    · by_cases laneLive : lane.val < ringDegree
      · simpa [paddedValue, blockLive, laneLive] using
          Semantics.Nc.BlockLane.residualsZero_of_truth data truth source
            ⟨block.val, blockLive⟩ ⟨lane.val, laneLive⟩
      · rw [paddedValue_lane_padding covers data source block lane
          (Nat.le_of_not_gt laneLive)]
        rfl
    · rw [paddedValue_block_padding covers data source block lane
          (Nat.le_of_not_gt blockLive)]
      rfl
  rw [residualZero]
  exact ConcreteCarrier.embed_zero

/-- Zero cubics on the complete padded product imply independent NC truth. -/
theorem truth_of_booleanResidualsZero
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape) :
    BooleanResidualsZero covers data -> Semantics.Nc.Truth data := by
  intro booleanResiduals
  apply Semantics.Nc.BlockLane.truth_of_residualsZero
    noZeroDivisors data
  intro source block lane
  have accepted := booleanResiduals source
    (domain.carrierBlock covers block) (domain.phi81Lane covers lane)
  rw [rangeValueAt_live covers data source block lane] at accepted
  have baseComponent := congrArg K.c0 accepted
  simpa only [K.embed, K.zero] using baseComponent

/-- The padded block×lane cubic relation is sound and complete for the same
independent full-carrier NC truth. The premise is used only for soundness. -/
theorem booleanResidualsZero_iff_truth
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape) :
    BooleanResidualsZero covers data <-> Semantics.Nc.Truth data := by
  exact ⟨truth_of_booleanResidualsZero noZeroDivisors covers data,
    booleanResidualsZero_of_truth covers data⟩

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.SourceProjection
