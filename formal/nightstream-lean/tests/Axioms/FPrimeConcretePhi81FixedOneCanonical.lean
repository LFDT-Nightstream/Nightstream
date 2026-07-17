import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical
import tests.Axioms.Support

/-!
Fail-closed dependency guard for the canonical fixed-one active carrier.
-/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical.Input.toActive_erase_of_authority' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical.Input.toActive_erase_of_authority

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical.obligations_iff_active' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical.obligations_iff_active

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical.holds_iff_active' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical.holds_iff_active

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical.holds_projection_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical.holds_projection_iff
