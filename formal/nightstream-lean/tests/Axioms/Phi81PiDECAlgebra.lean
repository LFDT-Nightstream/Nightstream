import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the first typed Phi81 `PiDEC.Algebra` slice.

| Stage path | Guarded theorem |
|---|---|
| `nifs.pi_dec.verify.radix.parameters` | exact production parameter tuple |
| `nifs.pi_dec.verify.radix.scalar.digits` | each computed binary digit is strict-`2` |
| `nifs.pi_dec.verify.radix.recompose` | total scalar and complete-assignment recomposition |
| `nifs.pi_dec.verify.radix.split_norm` | bounded parents produce bounded children |
| `nifs.pi_dec.verify.radix.recompose_norm` | bounded children produce a bounded parent |
-/

/-! Parameter and digit arithmetic are constructive. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.production_parameters' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.production_parameters

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.magnitudeDigit_lt_two' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.magnitudeDigit_lt_two

/-! The exported relation-level laws use only Lean's standard proposition,
choice, and quotient soundness axioms. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.splitScalar_recompose' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.splitScalar_recompose

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.split_recompose' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.split_recompose

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.split_norm' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.split_norm

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.recompose_norm' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.recompose_norm
