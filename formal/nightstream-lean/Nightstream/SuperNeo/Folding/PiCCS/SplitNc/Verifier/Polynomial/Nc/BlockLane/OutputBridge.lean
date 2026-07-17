import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.SourceProjection
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction

/-!
Anti-drift bridge from packed `y_zcol` rows to the block×lane source MLE.

Assurance tier: model-level.

Owns: equality between each active lane of the canonical packed block
projection and the corresponding block MLE in the independent NC source
polynomial.

Does not own: a prover message, commitment binding, lane interpolation, the
terminal cubic, SumCheck, transcript derivation, Rust, R1CS, costs, or row
removal.

Emits constraints: no.

Authority boundary: both sides consume the same explicit authoritative
assignment and verifier-owned block point. This is an anti-drift theorem, not
evidence that any externally supplied `y_zcol` opens a commitment.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.output_bridge.lane` | packed output lane equals the source polynomial's block MLE at that lane | derived | `packedYZcol_lane_eq_blockValueAt` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.OutputBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open BlockNcDomain

/-- Each active lane of the packed block projection is exactly the block MLE
used by the source-derived NC polynomial. Both sides share the domain-owned
little-endian block index. -/
theorem packedYZcol_lane_eq_blockValueAt
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (blockPoint : CubePoint K domain.blockVariables)
    (lane : Fin ringDegree) :
    PackedBlockAction.packedYZcol covers
        (data.assignment source) blockPoint lane =
      SourceProjection.blockValueAt covers data source blockPoint
        (laneVertex (domain.phi81Lane covers lane)) := by
  unfold PackedBlockAction.packedYZcol RingKAction.evaluateRows
    SourceProjection.blockValueAt SourceProjection.blockTable
  rw [laneIndex_laneVertex]
  congr 1
  apply congrArg BooleanTable.tabulate
  funext vertex
  by_cases live :
      (blockIndex vertex).val <
        Phi81ColumnLayout.blockCount shape.carrierWidth
  · simp only [PackedBlockAction.blockRows, live, dif_pos]
    simp [SourceProjection.paddedValue, live,
      RingKAction.embedChallenge, CarrierAction.assignmentBlock,
      CarrierAction.carrierColumn,
      Semantics.Nc.BlockLane.value,
      Semantics.Nc.BlockLane.carrierColumn]
    apply congrArg K.embed
    apply congrArg (data.assignment source)
    apply congrArg (fun block =>
      Phi81CarrierLayout.carrierColumn block lane)
    apply Fin.ext
    rfl
  · simp only [PackedBlockAction.blockRows, live]
    rw [SourceProjection.paddedValue_block_padding covers data source
      (blockIndex vertex) (domain.phi81Lane covers lane)
      (Nat.le_of_not_gt live)]
    exact ConcreteCarrier.embed_zero.symm

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.OutputBridge
