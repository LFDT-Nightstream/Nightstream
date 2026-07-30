import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the selected paper
`Pi_CCS`-to-`Pi_RLC` transcript handoff.
-/

namespace NightstreamTests.Axioms.FPrimePaperNifsPiRlcHandoff

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptCursor.outputFields_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptCursor.outputFields_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptCursor.replay_afterOutput_absorbed' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptCursor.replay_afterOutput_absorbed

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff.selected_outputFields_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff.selected_outputFields_length

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff.selected_afterOutput_absorbed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff.selected_afterOutput_absorbed

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff.samplerInitialBuilder_eq_handoff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff.samplerInitialBuilder_eq_handoff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff.decoded_samplerInitialBuilder_eq_valueReplay' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff.decoded_samplerInitialBuilder_eq_valueReplay

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff.rows_bind_sampler_to_piCcs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff.rows_bind_sampler_to_piCcs

end NightstreamTests.Axioms.FPrimePaperNifsPiRlcHandoff
