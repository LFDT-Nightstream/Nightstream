import Nightstream.SuperNeo.SumCheck.FixedPhase.Sequential
import tests.Axioms.Support

/-! Fail-closed dependency gate for sequential honest SumCheck construction. -/

/--
info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Sequential.ofFn_functionOfExactList' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Sequential.ofFn_functionOfExactList

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Sequential.exists_honest_run' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Sequential.exists_honest_run
