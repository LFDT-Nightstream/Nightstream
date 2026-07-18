import Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne.Minimality
import tests.Axioms.Support

/-! Fail-closed dependency gate for the one-slot base obligation plan. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.exact' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.exact

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne.exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne.exact

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne.Minimality.iterationZero_necessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne.Minimality.iterationZero_necessary

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne.Minimality.initialState_necessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne.Minimality.initialState_necessary

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne.Minimality.inclusionMinimalSound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne.Minimality.inclusionMinimalSound
