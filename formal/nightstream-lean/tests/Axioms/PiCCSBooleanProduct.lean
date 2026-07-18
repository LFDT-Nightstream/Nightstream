import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanProduct
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for structured Boolean product domains.

Owns: dependency expectations for little-endian prefix placement, the zero
prefix, and exact Boolean-prefix specialization of a tabulated MLE.

Does not own: fixed lane/block dimensions, matrix semantics, Phi81 algebra,
Rust, R1CS, row removal, or constraint counts.

| Audited theorem | Model-level guarantee |
|---|---|
| `fieldCoordinates_withLowPrefix` | typed product coordinates are prefix then suffix |
| `index_withLowPrefix` | the prefix occupies the low numeric bits |
| `index_zeros` | the canonical zero prefix has numeric index zero |
| `evaluate_tabulate_booleanPrefix` | a Boolean prefix restricts the product MLE exactly |
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanVertex.fieldCoordinates_withLowPrefix' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanVertex.fieldCoordinates_withLowPrefix

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain.index_withLowPrefix' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain.index_withLowPrefix

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain.index_zeros' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain.index_zeros

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable.evaluate_tabulate_booleanPrefix' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable.evaluate_tabulate_booleanPrefix
