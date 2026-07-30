import Nightstream.Implementation.R1CS.Canonical.KBooleanMleCarriedPadded
import Nightstream.Implementation.R1CS.Canonical.KBooleanMleOwnership
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKBooleanMle

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBooleanMle.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KBooleanMle.rows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBooleanMleSemantics.rows_compute_evaluate' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KBooleanMleSemantics.rows_compute_evaluate

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBooleanMlePadded.semanticTable_evaluate' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KBooleanMlePadded.semanticTable_evaluate

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBooleanMleCarriedPadded.rows_compute_paddedLaneEvaluation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KBooleanMleCarriedPadded.rows_compute_paddedLaneEvaluation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBooleanMleHonest.witness_satisfies' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KBooleanMleHonest.witness_satisfies

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBooleanMleOwnership.ownership_is_positional' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KBooleanMleOwnership.ownership_is_positional

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBooleanMleOwnership.rows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KBooleanMleOwnership.rows_conservation

end NightstreamTests.Axioms.CanonicalKBooleanMle
