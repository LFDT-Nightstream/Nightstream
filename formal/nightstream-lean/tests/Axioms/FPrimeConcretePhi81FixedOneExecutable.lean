import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical
import tests.Axioms.Support

/-! Fail-closed dependency gate for the executable canonical fixed-one F-prime verifier. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.nifsContext_materialize' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.nifsContext_materialize

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.check_eq_true_iff_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.check_eq_true_iff_accepted

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.run_eq_some_iff_physicalChecks' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.run_eq_some_iff_physicalChecks

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.run_sound_of_closure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.run_sound_of_closure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.exists_run_and_holds_or_samplerShortfall' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.exists_run_and_holds_or_samplerShortfall
