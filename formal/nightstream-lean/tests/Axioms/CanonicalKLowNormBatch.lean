import Nightstream.Implementation.R1CS.Canonical.KLowNormBatch
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the batched low-norm check.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalKLowNormBatch

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchRows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchRows_length_eq' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchRows_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchColumns_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchColumns_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.WellFormed.tail' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KLowNormBatch.WellFormed.tail

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchRows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchWitness_off_columns' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchWitness_off_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.lcEval_batchWitness' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.lcEval_batchWitness

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchWitness_at_column' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchWitness_at_column

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchRows_honest' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchRows_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchCost_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchCost_columns' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchCost_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.canonicalDigits_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KLowNormBatch.canonicalDigits_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.canonicalDigits_column_gt' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KLowNormBatch.canonicalDigits_column_gt

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.canonicalDigits_column_le' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.canonicalDigits_column_le

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.canonicalDigits_nodup' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.canonicalDigits_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.canonicalDigits_wellFormed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.canonicalDigits_wellFormed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.lowNormRows_determines_column' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KLowNormBatch.lowNormRows_determines_column

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.column_determines_digit' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.column_determines_digit

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchRows_owner_unique' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchRows_owner_unique

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchRows_owned' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchRows_owned


/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNormBatch.batchRows_use_columns' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNormBatch.batchRows_use_columns

end NightstreamTests.Axioms.CanonicalKLowNormBatch
