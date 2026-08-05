import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for the shared `Pi_CCS` numeric Boolean
domain.

Owns: the dependency expectation for the exact bridge between recursive
least-significant-bit tensor weights and the preserved `Nat.testBit` fold.

Does not own: output-claim semantics, production arithmetic, Rust,
R1CS, or constraint counts.

| Audited theorem | Model-level guarantee |
|---|---|
| `tensorWeight_eq_testBitWeight` | both numeric weight implementations coincide for every bounded index and dimension-checked point |
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain.tensorWeight_eq_testBitWeight' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain.tensorWeight_eq_testBitWeight
