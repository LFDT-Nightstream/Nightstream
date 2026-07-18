import Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution
import tests.Axioms.Support

/-! Fail-closed dependency gate for the production child substitution. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.witness' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.witness

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.no_parentOnlyHandle_binds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.no_parentOnlyHandle_binds
