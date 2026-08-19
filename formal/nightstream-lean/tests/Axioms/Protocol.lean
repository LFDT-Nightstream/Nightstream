import Nightstream.HyperNova
import Nightstream.Protocol
import tests.Axioms.Support

/-!
Fail-closed protocol axioms gate. Every expectation is checked when this
module is built; the aggregate entrypoint imports all ownership groups.
-/

/-- info: 'Nightstream.Protocol.FPrime.XOut.xOut_binding_or_collision' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.XOut.xOut_binding_or_collision

/-- info: 'Nightstream.HyperNova.Construction2.Default.emptyRunning_realizes_default' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.Default.emptyRunning_realizes_default

/-- info: 'Nightstream.Protocol.FPrime.Step.check_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.check_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.check_eq_true_iff_holds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.check_eq_true_iff_holds

/-- info: 'Nightstream.Protocol.FPrime.Step.holds_iff_local_and_outgoing' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.holds_iff_local_and_outgoing

/-- info: 'Nightstream.Protocol.FPrime.Step.checkLocal_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.checkLocal_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeBaseLocal_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.fPrimeBaseLocal_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeRecursiveLocal_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.fPrimeRecursiveLocal_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.closeLocal' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.closeLocal

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeBase_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.fPrimeBase_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeRecursive_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.fPrimeRecursive_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.next_state_pinned' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.next_state_pinned

/-- info: 'Nightstream.Protocol.FPrime.Step.holds_advance_facts' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.holds_advance_facts

/-- info: 'Nightstream.Protocol.TerminalCE.terminalCE_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.TerminalCE.terminalCE_sound

/-- info: 'Nightstream.Protocol.TerminalCE.terminalCE_complete' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.TerminalCE.terminalCE_complete
