import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Sampling

/-!
Paper `Pi_CCS` output-point ownership.

Owns: the exact identification of the new CE point `r'` with the complete
ordered SumCheck challenge vector and the shared-point requirement for every
one of the `K+k` outputs.

Does not own: the SumCheck transcript, challenge generation, the terminal
identity `v = Q(r')`, output evaluation correctness, commitments, concrete CE
serialization, Fiat--Shamir, Rust, or R1CS.

Emits constraints: no.

Authority boundary: an output point is not accepted because it has the right
dimension. `BoundOutputs` requires each output's point to equal the
verifier-owned `roundChallenges` value exactly.

| Paper object | Typed owner | Enforced equality |
|---|---|---|
| SumCheck point `r'` | `SumCheckConclusion.roundChallenges` | its coordinates are the ordered round challenges |
| one CE output point | `OutputAt.point` | equals `roundChallenges` |
| full CE product | `BoundOutputs` | every one of `K+k` output points equals the same `roundChallenges` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField uPayload

/-- The verifier-owned result of the paper's one joint SumCheck. The challenge
vector itself is the new point `r'`; there is no independent point field to
drift from it. -/
structure SumCheckConclusion (Field : Type uField) (shape : Shape) where
  roundChallenges : CubePoint Field shape.cubeVariables
  terminalClaim : Field

/-- One public output payload paired with the point it claims to use. -/
structure OutputAt
    (Field : Type uField)
    (shape : Shape)
    (Payload : Type uPayload) where
  point : CubePoint Field shape.cubeVariables
  payload : Payload

/-- The paper's `K+k` output product before point binding is checked. -/
structure OutputProduct
    (Field : Type uField)
    (shape : Shape)
    (Payload : Type uPayload) where
  outputs : List (OutputAt Field shape Payload)
  outputCount : outputs.length = shape.sourceCount

/-- Exact verifier-owned point binding for the full output product. -/
structure BoundOutputs
    {Field : Type uField}
    {shape : Shape}
    {Payload : Type uPayload}
    (conclusion : SumCheckConclusion Field shape)
    (product : OutputProduct Field shape Payload) : Prop where
  pointBinding : forall output,
    output ∈ product.outputs -> output.point = conclusion.roundChallenges

/-- Every output point is the ordered SumCheck challenge vector, not merely a
same-length prover-selected point. -/
theorem BoundOutputs.outputPoint_eq_roundChallenges
    {Field : Type uField}
    {shape : Shape}
    {Payload : Type uPayload}
    {conclusion : SumCheckConclusion Field shape}
    {product : OutputProduct Field shape Payload}
    (bound : BoundOutputs conclusion product)
    (output : OutputAt Field shape Payload)
    (member : output ∈ product.outputs) :
    output.point = conclusion.roundChallenges :=
  bound.pointBinding output member

/-- Two outputs accepted under one conclusion necessarily use the same point. -/
theorem BoundOutputs.outputs_share_point
    {Field : Type uField}
    {shape : Shape}
    {Payload : Type uPayload}
    {conclusion : SumCheckConclusion Field shape}
    {product : OutputProduct Field shape Payload}
    (bound : BoundOutputs conclusion product)
    (left right : OutputAt Field shape Payload)
    (leftMember : left ∈ product.outputs)
    (rightMember : right ∈ product.outputs) :
    left.point = right.point := by
  exact (bound.pointBinding left leftMember).trans
    (bound.pointBinding right rightMember).symm

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
