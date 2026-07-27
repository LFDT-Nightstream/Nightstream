import Nightstream.Implementation.R1CS.Canonical.LinCombNormal
import tests.Axioms.Support

/-!
Fail-closed dependency gate for combination aggregation, support algebra
and the field-canonical normal form.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalLinCombNormal

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.lcEval_normalize' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.lcEval_normalize

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.rawSum_normalize' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.rawSum_normalize

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.rawSum_insertTerm' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.rawSum_insertTerm

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.rawSum_cons' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.rawSum_cons

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.rawSum_append' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.rawSum_append

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.rawSum_flatMap' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.rawSum_flatMap

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.normalize_length_le' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.normalize_length_le

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.insertTerm_length_le' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.insertTerm_length_le

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.insertTerm_length_of_fresh' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms LinCombNormal.insertTerm_length_of_fresh

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.insertTerm_nodup' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms LinCombNormal.insertTerm_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.normalize_nodup' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms LinCombNormal.normalize_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.normalize_length_of_nodup' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms LinCombNormal.normalize_length_of_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.mentions_single' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms LinCombNormal.mentions_single

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.mentions_map_scale' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.mentions_map_scale

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.mentions_append' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.mentions_append

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.mentions_insertTerm' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms LinCombNormal.mentions_insertTerm

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.mentions_normalize' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms LinCombNormal.mentions_normalize

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.nodup_map' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.nodup_map

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.nodup_length_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.nodup_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.normalize_length_eq_witness' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.normalize_length_eq_witness

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.rawSum_filterMap_reduceTerm' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.rawSum_filterMap_reduceTerm

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.lcEval_fieldNormalize' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.lcEval_fieldNormalize

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.fieldNormalize_canonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.fieldNormalize_canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.fieldNormalize_nonzero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.fieldNormalize_nonzero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.fieldNormalize_length_le' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms LinCombNormal.fieldNormalize_length_le

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.mentions_fieldNormalize_subset' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.mentions_fieldNormalize_subset

/-! Coefficient survival: the structural half of no-cancellation. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.insertTerm_entries' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms LinCombNormal.insertTerm_entries

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.normalize_entries_of_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms LinCombNormal.normalize_entries_of_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.filterMap_reduceTerm_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms LinCombNormal.filterMap_reduceTerm_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.LinCombNormal.fieldNormalize_length_of_nonzero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms LinCombNormal.fieldNormalize_length_of_nonzero

end NightstreamTests.Axioms.CanonicalLinCombNormal
