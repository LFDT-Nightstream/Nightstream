import Nightstream.Implementation.Nebula.NIFS.PiRLC.AlgebraSoundFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductPiRlcAlgebraSoundFor

/-! Dependency gate for exponent-indexed PiRLC algebra soundness. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSoundFor.typedEquations_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSoundFor.typedEquations_of_rows

end tests.Axioms.NebulaProductPiRlcAlgebraSoundFor
