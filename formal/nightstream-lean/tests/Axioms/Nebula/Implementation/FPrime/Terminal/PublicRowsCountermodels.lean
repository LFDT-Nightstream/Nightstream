import tests.Nebula.Implementation.FPrime.Terminal.PublicRowsCountermodels
import tests.Axioms.Support

/-! Dependency gate for the omitted terminal-public-link countermodel. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaTerminalPublicRowsCountermodels

open tests.NebulaTerminalPublicRowsCountermodels

/-- info: 'tests.NebulaTerminalPublicRowsCountermodels.omitted_terminal_link_allows_public_mismatch' does not depend on any axioms -/
#guard_msgs in
#audit_axioms omitted_terminal_link_allows_public_mismatch

end tests.Axioms.NebulaTerminalPublicRowsCountermodels
