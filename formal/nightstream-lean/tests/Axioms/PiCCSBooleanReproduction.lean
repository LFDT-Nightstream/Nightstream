import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for canonical Boolean reproduction.

Owns: dependency expectations for the neutral Bool-to-field point, exact
Boolean Kronecker weights, partition of unity, canonical table reproduction,
and numeric tensor-selector reproduction.

Does not own: protocol table construction, numeric-to-field encoding,
SumCheck, Rust, R1CS, or constraint counts.

| Audited theorem | Model-level guarantee |
|---|---|
| `BooleanVertex.toCubePoint_coordinates` | field coordinates are exactly the neutral Bool-to-field serialization |
| `equalityWeight_toCubePoint` | equality weight at a Boolean point is the exact Kronecker selector |
| `equalityWeight_sum_eq_one` | canonical Boolean equality weights sum to one |
| `equalityWeighted_tabulate_eq_evaluate` | equality-weighted leaves reproduce the canonical table MLE |
| `equalityWeighted_sumMap` | equality weighting commutes with an explicit weighted family sum |
| `equalityWeighted_tensorWeight_eq_tensorWeight` | equality weighting reproduces the shared numeric tensor selector |
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanVertex.toCubePoint_coordinates' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanVertex.toCubePoint_coordinates

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction.equalityWeight_toCubePoint' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms equalityWeight_toCubePoint

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction.equalityWeight_sum_eq_one' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms equalityWeight_sum_eq_one

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction.equalityWeighted_tabulate_eq_evaluate' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms equalityWeighted_tabulate_eq_evaluate

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction.equalityWeighted_sumMap' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms equalityWeighted_sumMap

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction.equalityWeighted_tensorWeight_eq_tensorWeight' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms equalityWeighted_tensorWeight_eq_tensorWeight
