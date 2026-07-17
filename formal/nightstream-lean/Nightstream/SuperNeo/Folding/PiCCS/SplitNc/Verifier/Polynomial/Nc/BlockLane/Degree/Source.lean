import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree.Parameters
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing

/-!
Source-slice degree bounds for the canonical Split-NC block×lane polynomial.

Assurance tier: model-level.

Owns: affine coordinate slices of the nested source MLE, their exact
strict-`b = 2` cubic images, and preservation of that cubic ceiling under the
paper-relative gamma source sum.

Does not own: equality selectors, the final quartic polynomial, SumCheck
messages, transcript derivation, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: all coefficient representations are constructed from
the source-derived Boolean tables and verifier coins. No prover-supplied
degree, coefficient list, or polynomial callback is consumed.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.degree.source.block` | one block coordinate of the nested source MLE is affine | derived | `sourceValueAt_block_affine` |
| `nifs.pi_ccs.nc.block_lane.degree.source.lane` | one lane coordinate of the nested source MLE is affine | derived | `sourceValueAt_lane_affine` |
| `nifs.pi_ccs.nc.block_lane.degree.range.block` | strict range maps an affine block slice to a cubic | derived | `rangeValueAt_block_cubic` |
| `nifs.pi_ccs.nc.block_lane.degree.range.lane` | strict range maps an affine lane slice to a cubic | derived | `rangeValueAt_lane_cubic` |
| `nifs.pi_ccs.nc.block_lane.degree.mix.block` | source compression preserves the block cubic ceiling | derived | `mixedRangeAt_block_cubic` |
| `nifs.pi_ccs.nc.block_lane.degree.mix.lane` | source compression preserves the lane cubic ceiling | derived | `mixedRangeAt_lane_cubic` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Source

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws
private abbrev Polynomial := SumCheck.Finite.FixedPolynomial K

/-- One coordinate of the block MLE at a fixed Boolean lane is affine. -/
theorem blockValueAt_affine
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (lane : BooleanVertex domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    ∃ polynomial : Polynomial 1, ∀ point,
      polynomial.evaluate ops.toOps point =
        SourceProjection.blockValueAt covers data source
          (cubeSlice before after length point) lane := by
  unfold SourceProjection.blockValueAt BooleanTable.evaluate
  exact evaluateCoordinates_affine
    (SourceProjection.blockTable covers data source lane)
    before after length

/-- Every block coordinate of the complete nested source MLE is affine. -/
theorem sourceValueAt_block_affine
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    ∃ polynomial : Polynomial 1, ∀ point,
      polynomial.evaluate ops.toOps point =
        SourceProjection.sourceValueAt covers data source {
          block := cubeSlice before after length point
          lane := lane } := by
  have represented := polynomial_sum_exists
    (BooleanVertex.all domain.laneVariables)
    (fun vertex => vertex.equalityWeight ops lane)
    (fun vertex point => SourceProjection.blockValueAt covers data source
      (cubeSlice before after length point) vertex)
    (by
      intro vertex _
      exact blockValueAt_affine covers data source vertex
        before after length)
  rcases represented with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  unfold SourceProjection.sourceValueAt SourceProjection.laneTableAtBlock
  rw [← BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
    ops laws]
  rfl

/-- Every lane coordinate of the complete nested source MLE is affine. -/
theorem sourceValueAt_lane_affine
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : CubePoint K domain.blockVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ polynomial : Polynomial 1, ∀ point,
      polynomial.evaluate ops.toOps point =
        SourceProjection.sourceValueAt covers data source {
          block := block
          lane := cubeSlice before after length point } := by
  unfold SourceProjection.sourceValueAt BooleanTable.evaluate
  exact evaluateCoordinates_affine
    (SourceProjection.laneTableAtBlock covers data source block)
    before after length

/-- Strict-`b = 2` turns an affine block slice into a cubic. -/
theorem rangeValueAt_block_cubic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    ∃ polynomial : Polynomial 3, ∀ point,
      polynomial.evaluate ops.toOps point =
        SourceProjection.rangeValueAt covers data source {
          block := cubeSlice before after length point
          lane := lane } := by
  rcases sourceValueAt_block_affine covers data source lane
    before after length with ⟨sourcePolynomial, sourceRepresents⟩
  refine ⟨strictRangeOfAffine sourcePolynomial, ?_⟩
  intro point
  unfold SourceProjection.rangeValueAt
  rw [evaluate_strictRangeOfAffine, sourceRepresents]
  rw [ConcreteCarrier.derived_sub_eq_concrete_sub]
  rfl

/-- Strict-`b = 2` turns an affine lane slice into a cubic. -/
theorem rangeValueAt_lane_cubic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : CubePoint K domain.blockVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ polynomial : Polynomial 3, ∀ point,
      polynomial.evaluate ops.toOps point =
        SourceProjection.rangeValueAt covers data source {
          block := block
          lane := cubeSlice before after length point } := by
  rcases sourceValueAt_lane_affine covers data source block
    before after length with ⟨sourcePolynomial, sourceRepresents⟩
  refine ⟨strictRangeOfAffine sourcePolynomial, ?_⟩
  intro point
  unfold SourceProjection.rangeValueAt
  rw [evaluate_strictRangeOfAffine, sourceRepresents]
  rw [ConcreteCarrier.derived_sub_eq_concrete_sub]
  rfl

/-- Gamma compression preserves the cubic block-coordinate ceiling. -/
theorem mixedRangeAt_block_cubic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    ∃ polynomial : Polynomial 3, ∀ point,
      polynomial.evaluate ops.toOps point =
        Mixing.mixedRangeAt covers data coins {
          block := cubeSlice before after length point
          lane := lane } := by
  unfold Mixing.mixedRangeAt
  apply polynomial_sum_exists
  intro source _
  exact rangeValueAt_block_cubic covers data source lane
    before after length

/-- Gamma compression preserves the cubic lane-coordinate ceiling. -/
theorem mixedRangeAt_lane_cubic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (block : CubePoint K domain.blockVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ polynomial : Polynomial 3, ∀ point,
      polynomial.evaluate ops.toOps point =
        Mixing.mixedRangeAt covers data coins {
          block := block
          lane := cubeSlice before after length point } := by
  unfold Mixing.mixedRangeAt
  apply polynomial_sum_exists
  intro source _
  exact rangeValueAt_lane_cubic covers data source block
    before after length

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Source
