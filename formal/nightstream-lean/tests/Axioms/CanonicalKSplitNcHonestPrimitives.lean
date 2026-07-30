import Nightstream.Implementation.R1CS.Canonical.KBooleanMleSequentialHonest
import Nightstream.Implementation.R1CS.Canonical.KPointEqualityHonest
import Nightstream.Implementation.R1CS.Canonical.KSparsePolynomialHonest
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKSplitNcHonestPrimitives

open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPointEqualityHonest.witness_off_block' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPointEqualityHonest.witness_off_block

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPointEqualityHonest.rows_honest' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPointEqualityHonest.rows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSparsePolynomialHonest.witness_off_block' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSparsePolynomialHonest.witness_off_block

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSparsePolynomialHonest.rows_honest' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KSparsePolynomialHonest.rows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBooleanMleSequentialHonest.rowsFrom_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KBooleanMleSequentialHonest.rowsFrom_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBooleanMleSequentialHonest.rowsFrom_below_end' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KBooleanMleSequentialHonest.rowsFrom_below_end

end NightstreamTests.Axioms.CanonicalKSplitNcHonestPrimitives
