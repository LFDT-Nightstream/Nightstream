import tests.Nebula.Implementation.FPrime.Terminal.StatementRowsCountermodels
import tests.Axioms.Support

/-! Dependency gate for terminal statement-row countermodels. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaTerminalStatementRowsCountermodels

open tests.NebulaTerminalStatementRowsCountermodels

/-- info: 'tests.NebulaTerminalStatementRowsCountermodels.omitted_recomposition_allows_public_field_substitution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms omitted_recomposition_allows_public_field_substitution

/-- info: 'tests.NebulaTerminalStatementRowsCountermodels.modulus_word_aliases_zero_without_integer_bound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms modulus_word_aliases_zero_without_integer_bound

end tests.Axioms.NebulaTerminalStatementRowsCountermodels
