import Nightstream.Implementation.Nebula.Production.NIFS.Core.NifsPaperRowsSoundFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionProductNifsPaperRowsSoundFor

/-! Dependency gate for exponent-indexed production NIFS section soundness. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionProductNifsPaperRowsSoundFor.rows_imply_exact_result' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionProductNifsPaperRowsSoundFor.rows_imply_exact_result

end tests.Axioms.NebulaProductionProductNifsPaperRowsSoundFor
