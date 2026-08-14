import Nightstream.Implementation.Nebula.NIFS.PiRLC.AlgebraSound
import tests.Axioms.Support

/-! Dependency audit for the exact aggregate V2 PiRLC algebra rows. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.family_windows_disjoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.family_windows_disjoint

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.family_satisfies' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.family_satisfies

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound.publicBlock_publicOfRings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound.publicBlock_publicOfRings

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound.typedEquations_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound.typedEquations_of_rows
