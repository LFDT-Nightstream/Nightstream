import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.PriorLink.ParentOnlyAuthority
import tests.Axioms.Support

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.PriorLink.ParentOnlyAuthority

/-! Fail-closed dependency gate for the parent-only F-prime authority witness. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.PriorLink.ParentOnlyAuthority.Substitution.no_parentOnlyAccumulator_binds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Substitution.no_parentOnlyAccumulator_binds

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.PriorLink.ParentOnlyAuthority.Substitution.rightIncomingAccepted_but_notCanonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Substitution.rightIncomingAccepted_but_notCanonical

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.PriorLink.ParentOnlyAuthority.Substitution.xOut_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Substitution.xOut_eq
