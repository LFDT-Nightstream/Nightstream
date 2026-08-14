import tests.NebulaProgramBinding
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guards for the exact Nebula program binding.
-/

/-- info: 'Nightstream.Implementation.R1CS.NebulaProgramBindingSound.program_binding_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.NebulaProgramBindingSound.program_binding_sound
