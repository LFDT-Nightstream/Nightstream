import Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the fixed-active carrier necessity results.

The guarded surface instantiates the audited paper verifier at the complete
270-coordinate carrier and proves that hypothetical 257-coordinate pins of the
carrier-polymorphic NIFS, fixed-one, and Construction-2 types are lossy. It does
not assert that the repository currently selects such a pin.
-/

/-- info: 'Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.exactPaperVerifier_soundAndCompleteModulo' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.exactPaperVerifier_soundAndCompleteModulo

/-- info: 'Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.eraseRunning_zero_eq_tail' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.eraseRunning_zero_eq_tail

/-- info: 'Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.zeroPublicRunning_ne_tailMutatedRunning' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.zeroPublicRunning_ne_tailMutatedRunning

/-- info: 'Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.no_exact_paperNifs_running_decoder' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.no_exact_paperNifs_running_decoder

/-- info: 'Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.no_exact_fixedOne_fprime_decoder' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.no_exact_fixedOne_fprime_decoder

/-- info: 'Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.no_exact_construction2_fprime_decoder' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction.no_exact_construction2_fprime_decoder
