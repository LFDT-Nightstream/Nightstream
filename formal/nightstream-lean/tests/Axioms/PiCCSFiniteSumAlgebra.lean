import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for shared finite-sum algebra.

Owns: dependency expectations for congruence, additive/subtractive
linearity, scalar distribution, and finite product-domain reordering.

Does not own: protocol index families, Boolean selectors, residual formulas,
SumCheck soundness, Rust, R1CS, or constraint counts.

| Audited theorem | Model-level guarantee |
|---|---|
| `sumMap_congr` | pointwise equality on the explicit index list preserves its sum |
| `sumMap_add` | addition distributes through the explicit finite sum |
| `sumMap_mul_left` | a common left factor moves outside the explicit finite sum |
| `sumMap_sub` | subtraction distributes through the explicit finite sum |
| `sub_eq_zero_iff` | explicit subtraction vanishes exactly when both operands agree |
| `sumMap_swap` | nested finite sums may exchange their explicit traversal domains |
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra.sumMap_congr' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms sumMap_congr

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra.sumMap_add' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms sumMap_add

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra.sumMap_mul_left' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms sumMap_mul_left

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra.sumMap_sub' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms sumMap_sub

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra.sub_eq_zero_iff' does not depend on any axioms -/
#guard_msgs in
#audit_axioms sub_eq_zero_iff

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra.sumMap_swap' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms sumMap_swap
