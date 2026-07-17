import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction

/-!
Contract: expose the shared base-field linearity leaves for packed block
projection. Π_RLC uses zero/addition; Π_DEC additionally uses scaling.

Owns: exact row-level and evaluated zero, addition, and embedded base-field
scaling laws for `packedYZcol`.

Does not own: `RingF` action, finite Π_RLC combination, Π_DEC radix weights,
augmented CE membership, transcripts, commitments, Rust, R1CS, costs, or row
removal.

Emits constraints: no.

Authority boundary: every row is derived from the same explicit complete
assignment. These are algebraic transport laws, not a proof that a packed
evaluation is bound to a commitment or verifier transcript.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.shared.packed_y_zcol.linear.zero` | zero assignment projects to zero | derived | `packedYZcol_zero` |
| `nifs.shared.packed_y_zcol.linear.add` | assignment addition projects to `RingK` addition | derived | `packedYZcol_add` |
| `nifs.shared.packed_y_zcol.linear.scale` | base-field assignment scaling projects to embedded-`F` scaling | derived | `packedYZcol_scale` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.Linear

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open BlockNcDomain
open PackedBlockAction

/-- Canonical zero assignment rows are exactly the canonical zero row table. -/
theorem blockRows_zero
    {shape : SemanticShape}
    {domain : BlockNcDomain} :
    blockRows (domain := domain)
        (BaseLinear.Raw.assignmentZero (columns := shape.carrierWidth)) =
      RingKAction.zeroRows := by
  funext vertex
  by_cases live :
      (blockIndex vertex).val <
        Phi81ColumnLayout.blockCount shape.carrierWidth
  · simp only [blockRows, live, dif_pos, RingKAction.zeroRows]
    funext lane
    rfl
  · simp [blockRows, live, RingKAction.zeroRows]

/-- Pointwise assignment addition is exactly row-wise `RingK` addition. -/
theorem blockRows_add
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (left right : SemanticAssignment shape) :
    blockRows (domain := domain)
        (BaseLinear.Raw.assignmentAdd left right) =
      RingKAction.addRows
        (blockRows (domain := domain) left)
        (blockRows (domain := domain) right) := by
  funext vertex
  by_cases live :
      (blockIndex vertex).val <
        Phi81ColumnLayout.blockCount shape.carrierWidth
  · simp only [blockRows, live, dif_pos, RingKAction.addRows]
    funext lane
    rfl
  · simp only [blockRows, live, RingKAction.addRows]
    funext lane
    rfl

/-- Base-field assignment scaling is exactly row-wise embedded-`F` scaling. -/
theorem blockRows_scale
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (scalar : F)
    (assignment : SemanticAssignment shape) :
    blockRows (domain := domain)
        (BaseLinear.Raw.assignmentScale scalar assignment) =
      RingKAction.scaleRows (K.embed scalar)
        (blockRows (domain := domain) assignment) := by
  funext vertex
  by_cases live :
      (blockIndex vertex).val <
        Phi81ColumnLayout.blockCount shape.carrierWidth
  · simp only [blockRows, live, dif_pos, RingKAction.scaleRows]
    funext lane
    simpa only [ConcreteCarrier.baseOps, ConcreteCarrier.extensionOps] using
      (ConcreteCarrier.embed_mul scalar
        (assignment (SplitNc.Semantics.Nc.BlockLane.carrierColumn
          ⟨(blockIndex vertex).val, live⟩ lane)))
  · simp only [blockRows, live, RingKAction.scaleRows]
    funext lane
    exact (ConcreteCarrier.extensionLaws.mul_zero (K.embed scalar)).symm

/-- The packed block evaluation of the zero assignment is zero. -/
theorem packedYZcol_zero
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (point : CubePoint K domain.blockVariables) :
    packedYZcol covers
        (BaseLinear.Raw.assignmentZero (columns := shape.carrierWidth)) point =
      ringKZero := by
  unfold packedYZcol
  rw [blockRows_zero, RingKAction.evaluateRows_zero]

/-- Packed block evaluation preserves pointwise assignment addition. -/
theorem packedYZcol_add
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (left right : SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables) :
    packedYZcol covers (BaseLinear.Raw.assignmentAdd left right) point =
      ringKAdd (packedYZcol covers left point)
        (packedYZcol covers right point) := by
  unfold packedYZcol
  rw [blockRows_add, RingKAction.evaluateRows_add]

/-- Packed block evaluation preserves base-field scaling. -/
theorem packedYZcol_scale
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (scalar : F)
    (assignment : SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables) :
    packedYZcol covers (BaseLinear.Raw.assignmentScale scalar assignment) point =
      RingKAction.scale (K.embed scalar)
        (packedYZcol covers assignment point) := by
  unfold packedYZcol
  rw [blockRows_scale, RingKAction.evaluateRows_scale]

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.Linear
