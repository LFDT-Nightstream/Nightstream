import tests.Axioms.Support
import tests.NebulaV2.Implementation.Production.Application.AuthorityCountermodels

namespace tests.Axioms.NebulaV2ProductionApplicationAuthorityCountermodels

open tests.NebulaV2ProductionApplicationAuthorityCountermodels

/-- info: 'tests.NebulaV2ProductionApplicationAuthorityCountermodels.output_placement_accepts_unlinked_accesses' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms output_placement_accepts_unlinked_accesses

/-- info: 'tests.NebulaV2ProductionApplicationAuthorityCountermodels.same_row_shape_does_not_imply_selected_machine_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms same_row_shape_does_not_imply_selected_machine_transition

/-- info: 'tests.NebulaV2ProductionApplicationAuthorityCountermodels.universal_supplement_is_vacuous' does not depend on any axioms -/
#guard_msgs in
#audit_axioms universal_supplement_is_vacuous

end tests.Axioms.NebulaV2ProductionApplicationAuthorityCountermodels
