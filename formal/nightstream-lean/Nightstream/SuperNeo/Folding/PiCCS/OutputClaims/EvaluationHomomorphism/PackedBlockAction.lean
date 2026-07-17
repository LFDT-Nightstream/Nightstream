import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.Embedding
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane

/-!
Contract: prove that the canonical block-projected NC evaluation commutes
with the exact Phi81 `RingF` action used by Π_RLC.

Owns: the semantic complete-assignment carrier, zero-padded block rows, their
block-domain MLE, pointwise action compatibility, and the resulting packed
`y_zcol` action theorem.

Does not own: lane SumCheck semantics, augmented CE membership, Π_RLC source
combination, Π_DEC recomposition, transcript timing, commitments, Rust, R1CS,
costs, or row removal.

Emits constraints: no.

Authority boundary: every live row is derived from one authoritative complete
assignment block. Boolean block padding is definitionally `ringKZero`.
`packedYZcol` requires explicit block/lane coverage, but this file proves only
the model-level action law; it does not bind a prover message to a commitment.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.output.assignment` | packed output consumes the independent full-carrier assignment directly | direct dataflow | `SemanticAssignment` |
| `nifs.pi_ccs.nc.block_lane.output.rows.live` | each live block row is the coefficientwise embedding of one authoritative 54-lane assignment block | computed | `blockRows` |
| `nifs.pi_ccs.nc.block_lane.output.rows.padding` | padded block rows are canonical zero | computed | `blockRows` |
| `nifs.pi_ccs.nc.block_lane.output.projection` | block MLE yields one 54-lane `RingK` evaluation | computed | `packedYZcol` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.action` | packed block projection commutes with the exact `RingF` assignment action | derived | `blockRows_act`, `packedYZcol_ringAction` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open BlockNcDomain

/-- The independent complete assignment used by Split-NC semantics. No
relation shape or invented public width is needed to state this projection. -/
abbrev SemanticAssignment (shape : SemanticShape) :=
  PaperLinearAlgebra.Assignment F shape.carrierWidth

/-- One coefficientwise-embedded assignment block at every live Boolean row;
the entire padded block suffix is canonical zero. -/
def blockRows
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (assignment : SemanticAssignment shape) :
    RingKAction.Rows domain.blockVariables :=
  fun vertex =>
    let padded := blockIndex vertex
    if live : padded.val < Phi81ColumnLayout.blockCount shape.carrierWidth then
      RingKAction.embedChallenge
        (CarrierAction.assignmentBlock assignment ⟨padded.val, live⟩)
    else
      ringKZero

/-- Evaluate the canonical block rows at the verifier-owned block point.
Coverage is explicit so an undersized Boolean domain cannot be presented as a
complete NC projection. -/
def packedYZcol
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (_covers : domain.Covers shape)
    (assignment : SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables) : RingK :=
  RingKAction.evaluateRows (blockRows assignment) point

/-- Acting on the authoritative assignment acts on every live packed block;
the padded zero suffix remains zero. -/
theorem blockRows_act
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (challenge : RingF)
    (assignment : SemanticAssignment shape) :
    blockRows (domain := domain) (CarrierAction.act challenge assignment) =
      RingKAction.actRows (RingKAction.embedChallenge challenge)
        (blockRows (domain := domain) assignment) := by
  funext vertex
  by_cases live :
      (blockIndex vertex).val <
        Phi81ColumnLayout.blockCount shape.carrierWidth
  · simp only [blockRows, live, dif_pos, RingKAction.actRows]
    rw [CarrierAction.assignmentBlock_act,
      Embedding.embedChallenge_ringFMul]
  · simp only [blockRows, live, RingKAction.actRows]
    exact (RingKAction.ringKMul_right_zero _).symm

/-- The packed block projection is equivariant under the exact `RingF` action.
Unlike the current flat-column projection, all 54 lanes use the same block
weight, so ring multiplication can move across the block MLE. -/
theorem packedYZcol_ringAction
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (challenge : RingF)
    (assignment : SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables) :
    packedYZcol covers (CarrierAction.act challenge assignment) point =
      ringKMul (RingKAction.embedChallenge challenge)
        (packedYZcol covers assignment point) := by
  unfold packedYZcol
  calc
    RingKAction.evaluateRows
        (blockRows (domain := domain)
          (CarrierAction.act challenge assignment)) point =
      RingKAction.evaluateRows
        (RingKAction.actRows (RingKAction.embedChallenge challenge)
          (blockRows (domain := domain) assignment)) point := by
        rw [blockRows_act (domain := domain)]
    _ = ringKMul (RingKAction.embedChallenge challenge)
          (RingKAction.evaluateRows
            (blockRows (domain := domain) assignment) point) :=
      RingKAction.evaluateRows_embeddedChallenge_action
        challenge (blockRows (domain := domain) assignment) point

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction
