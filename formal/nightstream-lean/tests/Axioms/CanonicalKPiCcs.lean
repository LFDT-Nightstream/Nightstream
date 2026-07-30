import Nightstream.Implementation.R1CS.Canonical.KPiCcsEventBinding
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKPiCcs

open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMulChain.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KMulChain.rows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPointEquality.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPointEquality.rows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSparsePolynomial.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KSparsePolynomial.rows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KStrictNorm.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KStrictNorm.rows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiCcsTerminal.rows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPiCcsTerminal.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiCcsTerminal.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPiCcsTerminal.rows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiCcsOccurrence.rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiCcsOccurrence.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiCcsOccurrence.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPiCcsOccurrence.rows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiCcsOccurrence.rows_imply_tableTruth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiCcsOccurrence.rows_imply_tableTruth_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptSemantics.rows_replay_semantics' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiCcsTranscriptSemantics.rows_replay_semantics

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiCcsPaperFiatShamir.derive_agrees' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPiCcsPaperFiatShamir.derive_agrees

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiCcsPaperFiatShamir.rows_derive_paper_schedule' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiCcsPaperFiatShamir.rows_derive_paper_schedule

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiCcsEventBinding.rows_imply_tableTruth_or_paperBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiCcsEventBinding.rows_imply_tableTruth_or_paperBadEvent

end NightstreamTests.Axioms.CanonicalKPiCcs
