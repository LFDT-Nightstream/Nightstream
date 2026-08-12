import Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSoundFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductPiRlcAlgebraSoundFor

/-! Dependency gate for exponent-indexed PiRLC algebra soundness. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSoundFor.typedEquations_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSoundFor.typedEquations_of_rows

end tests.Axioms.NebulaV2ProductPiRlcAlgebraSoundFor
