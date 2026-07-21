import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictCanonicalX
import tests.Axioms.Support

/-! Fail-closed dependency gate for the model-level canonical-X R1CS family. -/

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictCanonicalX.rows_force_splitScalar' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictCanonicalX.rows_force_splitScalar

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictCanonicalX.materializedSign_complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictCanonicalX.materializedSign_complete

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictCanonicalX.honest_complete_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictCanonicalX.honest_complete_rows

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictCanonicalX.exact_saving' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictCanonicalX.exact_saving
