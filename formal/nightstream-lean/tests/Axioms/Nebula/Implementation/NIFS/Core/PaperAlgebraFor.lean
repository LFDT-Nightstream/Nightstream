import Nightstream.Implementation.Nebula.NIFS.Core.PaperAlgebraFor
import tests.Axioms.Support

/-! Dependency gate for the exponent-indexed product paper algebra. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaProductPaperAlgebraFor

open Nightstream.Implementation.Nebula.ProductPaperAlgebraFor

#audit_axioms ambientAgreement
#audit_axioms evaluations_combine
#audit_axioms evaluations_recompose

end tests.Axioms.NebulaProductPaperAlgebraFor
