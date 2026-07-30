import Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe
import tests.Axioms.Support

/-!
Fail-closed axiom guard for CommitmentMixerRecipe.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalCommitmentMixerRecipe

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.mixerRows_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.mixerRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.mixerColumns_nodup' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.mixerColumns_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.mixerRows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.mixerRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.mixerRows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.mixerRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.mixerRows_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.mixerRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.mixerCost_rows' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.mixerCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.mixerCost_columns' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.mixerCost_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.mixing_alone_does_not_bind' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.mixing_alone_does_not_bind

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.parent_determines_coordinate' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.parent_determines_coordinate

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.mixerRow_parent' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.mixerRow_parent

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.mixerRows_owned' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.mixerRows_owned


/-- info: 'Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe.mixerColumns_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CommitmentMixerRecipe.mixerColumns_length

end NightstreamTests.Axioms.CanonicalCommitmentMixerRecipe
