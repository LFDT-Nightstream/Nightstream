import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictReducedY
import tests.Axioms.Support

/-! Fail-closed dependency gate for the model-level strict-PiDEC y reduction. -/

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictReducedY.reducedAccepted_iff_full' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictReducedY.reducedAccepted_iff_full

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictReducedY.reducedFamily_satisfies_iff_fullFamily' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictReducedY.reducedFamily_satisfies_iff_fullFamily

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictReducedY.fullFamily_row_count' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictReducedY.fullFamily_row_count
