import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictSound
import Nightstream.Implementation.R1CS.Correspondence.PiDecStrict.ShapeNecessity
import tests.Axioms.Support

/-! Fail-closed dependency gate for exact strict-PiDEC row soundness. -/

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictSound.rows_sound' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictSound.rows_sound

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictShapeNecessity.rows_alone_do_not_imply_strict_acceptance' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictShapeNecessity.rows_alone_do_not_imply_strict_acceptance
