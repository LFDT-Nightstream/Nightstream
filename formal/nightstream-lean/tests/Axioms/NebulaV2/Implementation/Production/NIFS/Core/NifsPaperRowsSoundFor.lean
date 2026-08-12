import Nightstream.Implementation.NebulaV2.Production.NIFS.Core.NifsPaperRowsSoundFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionProductNifsPaperRowsSoundFor

/-! Dependency gate for exponent-indexed production NIFS section soundness. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsPaperRowsSoundFor.rows_imply_exact_result' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsPaperRowsSoundFor.rows_imply_exact_result

end tests.Axioms.NebulaV2ProductionProductNifsPaperRowsSoundFor
