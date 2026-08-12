import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.PostPiCcsBridge
import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.SamplerResponseSound
import tests.Axioms.Support

/-! Dependency audit for the exact V2 full-field PiRLC sampler rows. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptCursor.afterFullOutput_absorbed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptCursor.afterFullOutput_absorbed

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcPostPiCcsBridge.rows_imply_candidate_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcPostPiCcsBridge.rows_imply_candidate_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcCandidateClassificationSound.all_candidates_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcCandidateClassificationSound.all_candidates_sound

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedSound.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedSound.sound

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchSound.sampleCoefficient_eq_some_output' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchSound.sampleCoefficient_eq_some_output

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcSamplerResponseSound.sampler_available' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcSamplerResponseSound.sampler_available

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcSamplerResponseSound.samplerSucceeded_eq_true' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcSamplerResponseSound.samplerSucceeded_eq_true

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcSamplerResponseSound.output_eq_scalarResponse' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcSamplerResponseSound.output_eq_scalarResponse
