import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Coins
import tests.Axioms.Support

/-!
Fail-closed dependency gate for exact challenge dimensions and typed semantic
coin projection.

| Stage path | Guarded obligation | Emits constraints? |
|---|---|---|
| `nifs.pi_ccs.transcript.squeeze` | requested field and extension counts are exact | no |
| `nifs.pi_ccs.transcript.challenge.shape` | every concrete challenge has its verifier-owned dimension | no |
| `nifs.pi_ccs.transcript.coins.shared` | FE and NC share one `betaA`/`gamma` authority | no |
-/

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.count_le_four_mul_blocksFor' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.count_le_four_mul_blocksFor

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.squeezeN_fields_length' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.squeezeN_fields_length

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.pairFields_length_of_length_eq_two_mul' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.pairFields_length_of_length_eq_two_mul

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.pairFields_squeezeN_even_length' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.pairFields_squeezeN_even_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_alpha_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_alpha_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_betaA_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_betaA_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_betaR_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_betaR_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_betaM_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_betaM_length

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Coins.ncCoins_betaA_eq_feCoins_betaA' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Coins.ncCoins_betaA_eq_feCoins_betaA

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Coins.ncCoins_gamma_eq_feCoins_gamma' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Coins.ncCoins_gamma_eq_feCoins_gamma

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Coins.run_state' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Coins.run_state
