import Nightstream.Assurance.NebulaV2.PiRlcSamplerSecurity
import tests.Axioms.Support

set_option autoImplicit false

/-- info: 'Nightstream.Assurance.NebulaV2.PiRlcSamplerSecurity.exact_schedule' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.PiRlcSamplerSecurity.exact_schedule

/-- info: 'Nightstream.Assurance.NebulaV2.PiRlcSamplerSecurity.lifetimeAbortBound_nonnegative' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.PiRlcSamplerSecurity.lifetimeAbortBound_nonnegative

/-- info: 'Nightstream.Assurance.NebulaV2.PiRlcSamplerSecurity.lifetimeAbortBound_le_166' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.PiRlcSamplerSecurity.lifetimeAbortBound_le_166

/-- info: 'Nightstream.Assurance.NebulaV2.PiRlcSamplerSecurity.dyadic_167_lt_lifetimeAbortBound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.PiRlcSamplerSecurity.dyadic_167_lt_lifetimeAbortBound
