import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm
import tests.Axioms.Support

/-!
Fail-closed dependency gate for concrete Phi81 `PiRLC` norm growth.

| Stage path | Guarded theorem |
|---|---|
| `nifs.pi_rlc.verify.norm_growth.centered.triangle` | centered Goldilocks addition triangle |
| `nifs.pi_rlc.verify.norm_growth.product.reduction.support` | exact finite executable support census |
| `nifs.pi_rlc.verify.norm_growth.product.expansion` | production `216` quotient-ring bound |
| `nifs.pi_rlc.verify.norm_growth.algebra` | exact concrete algebra-field theorem |
-/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_add_le' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_add_le

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Product.totalSupport_le_two_degrees' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Product.totalSupport_le_two_degrees

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Product.ringFMul_le_expansion' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Product.ringFMul_le_expansion

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Finite.relation_norm_growth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Finite.relation_norm_growth
