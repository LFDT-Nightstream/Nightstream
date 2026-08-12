import tests.NebulaV2.Implementation.FPrime.Terminal.PublicRowsCountermodels
import tests.Axioms.Support

/-! Dependency gate for the omitted terminal-public-link countermodel. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaV2TerminalPublicRowsCountermodels

open tests.NebulaV2TerminalPublicRowsCountermodels

/-- info: 'tests.NebulaV2TerminalPublicRowsCountermodels.omitted_terminal_link_allows_public_mismatch' does not depend on any axioms -/
#guard_msgs in
#audit_axioms omitted_terminal_link_allows_public_mismatch

end tests.Axioms.NebulaV2TerminalPublicRowsCountermodels
