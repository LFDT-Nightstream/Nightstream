import Nightstream.Implementation.R1CS.Canonical.KRecomposition
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the radix-`b` recomposition recipe.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalKRecomposition

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.scaled_term_mod' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KRecomposition.scaled_term_mod

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.add_inner_mod' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KRecomposition.add_inner_mod

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.mul_inner_mod' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KRecomposition.mul_inner_mod

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.rawSum_scaleTerms' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.rawSum_scaleTerms

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.lcEval_scaleTerms' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.lcEval_scaleTerms

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.lcEval_recomposeComb' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.lcEval_recomposeComb

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.powerSumFrom_eq_hornerValue' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KRecomposition.powerSumFrom_eq_hornerValue

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.powerSum_one' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KRecomposition.powerSum_one

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.carriedValue_recompose' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.carriedValue_recompose

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionRows_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionColumns_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionColumns_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionColumns_nodup' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionColumns_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionRows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionRows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.mentions_recomposeComb' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.mentions_recomposeComb

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionRows_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionCost_rows' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionCost_columns' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionCost_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionsRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionsRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionsRows_length_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionsRows_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionsRows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionsRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionsRows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionsRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionsCost_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionsCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.witnessChecks_differ' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KRecomposition.witnessChecks_differ

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionRows_owner_not_unique' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionRows_owner_not_unique


/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionsRows_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionsRows_conservation


/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.flatMap_getD_range' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.flatMap_getD_range

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.recompositionsRows_eq_map_owners' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.recompositionsRows_eq_map_owners

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.owners_nodup' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.owners_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRecomposition.ownership_is_positional' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRecomposition.ownership_is_positional

end NightstreamTests.Axioms.CanonicalKRecomposition
