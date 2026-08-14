import Nightstream.Implementation.Nebula.Commitment.Compact.AjtaiStageRows
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.CompactAjtaiStageRows.unitDigitResidue_signedDigit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactAjtaiStageRows.unitDigitResidue_signedDigit

/-- info: 'Nightstream.Implementation.Nebula.CompactAjtaiStageRows.coefficient_eq_seeded_phi81_basis_action' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactAjtaiStageRows.coefficient_eq_seeded_phi81_basis_action

/-- info: 'Nightstream.Implementation.Nebula.CompactAjtaiStageRows.sourceDigit_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactAjtaiStageRows.sourceDigit_exact

/-- info: 'Nightstream.Implementation.Nebula.CompactAjtaiStageRows.output_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactAjtaiStageRows.output_exact
