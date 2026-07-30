import Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe
import tests.Axioms.Support

/-!
Fail-closed axiom guard for FoldDigestRecipe.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalFoldDigestRecipe

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.carried_is_canonical' does not depend on any axioms -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.carried_is_canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.no_canonicality_violation' does not depend on any axioms -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.no_canonicality_violation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.digestRows_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.digestRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.digestColumns_nodup' does not depend on any axioms -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.digestColumns_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.digestRows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.digestRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.digestRows_honest' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.digestRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.digestRows_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.digestRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.digestCost_rows' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.digestCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.digestCost_columns' does not depend on any axioms -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.digestCost_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.canonicalityCost_rows' does not depend on any axioms -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.canonicalityCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.digestRows_claim_is_lane_equality' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.digestRows_claim_is_lane_equality

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.equalityRow_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.equalityRow_injective

/-- info: 'Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe.digestRows_owned' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms FoldDigestRecipe.digestRows_owned

end NightstreamTests.Axioms.CanonicalFoldDigestRecipe
