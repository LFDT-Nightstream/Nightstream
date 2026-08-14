import Nightstream.Implementation.Nebula.NIFS.PiRLC.TranscriptSemantics
import tests.Axioms.Support

/-! Dependency audit for the exact V2 PiRLC candidate transcript. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcTranscriptRows.CandidateIndex.flat_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcTranscriptRows.CandidateIndex.flat_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcTranscriptRows.candidate_windows_disjoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcTranscriptRows.candidate_windows_disjoint

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcTranscriptSemantics.candidate_rows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcTranscriptSemantics.candidate_rows_sound

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcTranscriptSemantics.all_candidates_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcTranscriptSemantics.all_candidates_sound
