import Nightstream.Implementation.NebulaV2.Production.Artifact.RelationDimensions
import tests.Axioms.Support

/-! Dependency gate for the shared augmented-relation dimension authority. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionRelationDimensions

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionRelationDimensions

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.relationRowVariables_minimum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.relationRowVariables_minimum

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.nifsShape_ne_reference25' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.nifsShape_ne_reference25

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.nifsPublicFrameFields_exceed_reference' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.nifsPublicFrameFields_exceed_reference

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.selected_exponent_core_included' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.selected_exponent_core_included

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.selected_exponent_core_count_fits' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.selected_exponent_core_count_fits

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.selected_exponent_core_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.selected_exponent_core_satisfied

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.terminalProgram_fold' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminalProgram_fold

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.terminal_program_included' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminal_program_included

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.terminal_program_count_fits' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminal_program_count_fits

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.terminal_program_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminal_program_satisfied

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.terminal_rows_and_columns_fit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminal_rows_and_columns_fit

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.Artifact.terminal_columns_scoped' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminal_columns_scoped

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRecursiveCoreManifestFor.length_only_accepts_zero_row_substitution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms
  ProductionRecursiveCoreManifestFor.length_only_accepts_zero_row_substitution

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.terminalLengthOnly_accepts_zero_row_substitution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms terminalLengthOnly_accepts_zero_row_substitution

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.splitExponentFit_accepts_incompatible_e1' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms splitExponentFit_accepts_incompatible_e1

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRelationDimensions.terminalRowCapacityOnly_does_not_imply_rectangularCapacity' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms terminalRowCapacityOnly_does_not_imply_rectangularCapacity

end tests.Axioms.NebulaV2ProductionRelationDimensions
