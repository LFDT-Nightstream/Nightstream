import tests.NebulaV2TerminalStatementRowsCountermodels
import tests.Axioms.Support

/-! Dependency gate for terminal statement-row countermodels. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaV2TerminalStatementRowsCountermodels

open tests.NebulaV2TerminalStatementRowsCountermodels

/-- info: 'tests.NebulaV2TerminalStatementRowsCountermodels.omitted_recomposition_allows_public_field_substitution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms omitted_recomposition_allows_public_field_substitution

/-- info: 'tests.NebulaV2TerminalStatementRowsCountermodels.modulus_word_aliases_zero_without_integer_bound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms modulus_word_aliases_zero_without_integer_bound

end tests.Axioms.NebulaV2TerminalStatementRowsCountermodels
