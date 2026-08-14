import Nightstream.Implementation.Nebula.NIFS.PiRLC.PostPiCcsBridge
import Nightstream.Implementation.Nebula.NIFS.PiRLC.SamplerResponseSound
import tests.Axioms.Support

/-! Dependency audit for the exact V2 full-field PiRLC sampler rows. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiCcsTranscriptCursor.afterFullOutput_absorbed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiCcsTranscriptCursor.afterFullOutput_absorbed

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcPostPiCcsBridge.rows_imply_candidate_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcPostPiCcsBridge.rows_imply_candidate_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcCandidateClassificationSound.all_candidates_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcCandidateClassificationSound.all_candidates_sound

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedSound.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedSound.sound

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedBatchSound.sampleCoefficient_eq_some_output' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedBatchSound.sampleCoefficient_eq_some_output

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound.sampler_available' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound.sampler_available

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound.samplerSucceeded_eq_true' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound.samplerSucceeded_eq_true

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound.output_eq_scalarResponse' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound.output_eq_scalarResponse
