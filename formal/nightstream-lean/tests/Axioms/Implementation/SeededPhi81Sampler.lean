import Nightstream.Implementation
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for seeded Phi81 sampling and the PiCCS output-
digest SIS sampler bridge.

Owns: dependency guards for unbounded accepted-symbol sampling, rejection repair,
vector and schedule assembly, fast-to-pure ChaCha refinement, the explicit
initial-acceptance boundary, and verifier-owned SIS seed derivation.

Does not own: sampler semantics, production row lowering, transcript authority,
or proof that the production no-rejection premises hold.

| Constraint-tree branch | Guarded mathematical obligation | Emits constraints? |
|---|---|---|
| `pi_ccs.output_digest.sis.sampler.core` | first-accepted and repair semantics | no |
| `pi_ccs.output_digest.sis.sampler.schedule` | vector/output schedule semantics | no |
| `pi_ccs.output_digest.sis.sampler.fast_refinement` | fast ChaCha equals pure sampling conditionally | no |
| `pi_ccs.output_digest.sis.seed_binding` | verifier-owned primary/compression seeds | no |
-/

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.nextAccepted_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.nextAccepted_sound

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.FirstAccepted.unique' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.FirstAccepted.unique

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.FirstAccepted.exists_fuel' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.FirstAccepted.exists_fuel

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81SamplerRefinement.scheduleBaseRotations_eq_pure' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81SamplerRefinement.scheduleBaseRotations_eq_pure

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81SamplerRefinement.blockBaseRotations_eq_pure' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81SamplerRefinement.blockBaseRotations_eq_pure

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.nextAccepted_mono' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.nextAccepted_mono

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.repairRejected_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.repairRejected_sound

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.Repairs.exists_fuel' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.Repairs.exists_fuel

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.Repairs.values_canonical' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.Repairs.values_canonical

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.sampleVector_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.sampleVector_sound

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.SampleVector.exists_fuel' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.SampleVector.exists_fuel

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.sampleVectors_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.sampleVectors_sound

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.SamplesVectors.length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.SamplesVectors.length

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.sampleOutput_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.sampleOutput_sound

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.sampleScheduleOutputs_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.sampleScheduleOutputs_sound

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.Schedule.baseRotations_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.Schedule.baseRotations_sound

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.SamplesSchedule.vectors_canonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.SamplesSchedule.vectors_canonical

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81.Block.Valid.baseRotations_success' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81.Block.Valid.baseRotations_success

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.SeededPhi81.Block.Valid.refines_pureSampling' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.SeededPhi81.Block.Valid.refines_pureSampling

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.PureBlockSampling.vectors_canonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.PureBlockSampling.vectors_canonical

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.SeededPhi81.Block.Valid.baseRotations_canonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.SeededPhi81.Block.Valid.baseRotations_canonical

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.runInitiallyAcceptedCheck_eq_true_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.runInitiallyAcceptedCheck_eq_true_iff

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.InitialRunAccepted.append' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.InitialRunAccepted.append

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81Sampler.sampleVectors_exists_of_initiallyAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81Sampler.sampleVectors_exists_of_initiallyAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.primarySeed0_derived' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.primarySeed0_derived

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.primarySeed1_derived' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.primarySeed1_derived

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.compressionSeed0_derived' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.compressionSeed0_derived

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.primaryFastSuccess' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.primaryFastSuccess

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.compressionFastSuccess' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.compressionFastSuccess

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.primaryBlock_pureSampling' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.primaryBlock_pureSampling

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.compressionBlock_pureSampling' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding.compressionBlock_pureSampling
