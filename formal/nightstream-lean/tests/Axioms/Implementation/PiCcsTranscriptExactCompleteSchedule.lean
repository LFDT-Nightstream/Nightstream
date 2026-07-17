import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.CompleteSchedule
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the exact carrier projection and complete
FE-to-NC transcript schedule.

| Stage path | Guarded obligation | Emits constraints? |
|---|---|---|
| `nifs.pi_ccs.exact.carrier` | semantic certificates round-trip through exact physical storage | no |
| `nifs.pi_ccs.exact.complete.shape` | exact carrier typing implies schedule shape | no |
| `nifs.pi_ccs.exact.complete.sumcheck` | FE successor is the NC predecessor | no |
| `nifs.pi_ccs.exact.schedule.nc.cursor` | positive exact NC replay computes cursor zero | no |
| `nifs.pi_ccs.exact.complete.catchup` | terminal state and digest use the exact NC successor | no |
-/

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.ExactRoundProjection.toFunction_ofFunction' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.ExactRoundProjection.toFunction_ofFunction

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Carrier.toFeCertificate_ofProtocolCertificate' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Carrier.toFeCertificate_ofProtocolCertificate

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Carrier.toNcCertificate_ofProtocolCertificate' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Carrier.toNcCertificate_ofProtocolCertificate

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule.scheduleInput_wellShaped' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule.scheduleInput_wellShaped

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule.run_challenges' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule.run_challenges

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule.run_sumcheck_eq_exact' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule.run_sumcheck_eq_exact

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule.run_afterNc_absorbed_zero' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule.run_afterNc_absorbed_zero

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule.run_afterNc_eq_exact' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule.run_afterNc_eq_exact

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule.run_catchup_joint' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule.run_catchup_joint
