import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Finite

/-!
Curated production norm-growth surface for the typed Phi81 `PiRLC.Algebra`.

Protocol: SuperNeo Definition 14 and `Pi_RLC`.
Phase: valid challenge action through the complete finite batch.
Constraint family: semantic norm only; this parent emits no rows.

Owns: the dependency and audit boundary for the concrete norm-growth field.

Does not own: transcript validity, other algebra fields, Rust/R1CS refinement,
row removal, or constraint counts.

Emits constraints: no.

Authority boundary: the exported theorem is derived from centered Goldilocks
arithmetic, the executable Phi81 multiplication support, and verifier-owned
production parameters. Existing circuits and measured counts are not inputs.

| Child path | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `centered` | cyclic-distance triangle and five-symbol multiplier bound | no | `Norm.Centered` |
| `product` | exact raw/reduction support and `216` expansion bound | no | `Norm.Product` |
| `finite` | canonical source fold and Definition-14 production bound | no | `Norm.Finite` |
| `algebra` | exact `PiRLC.Algebra.norm_growth` field shape | no | `Norm.Finite.relation_norm_growth` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm

export Centered
  (centeredMagnitude_add_le centeredMagnitude_sub_le
    embedCoefficient_mul_le_two)

export Product
  (supportActive supportCount totalSupport totalSupport_le_two_degrees
    rawMulCoeffF_le_support ringFMul_le_expansion)

export Finite
  (act_coordinate_le_expansion combineAssignments_le production_total_bound
    relation_norm_growth)

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm
