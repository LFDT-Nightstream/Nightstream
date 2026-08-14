import Nightstream.Implementation.Nebula.Production.Artifact.RelationDimensions
import tests.Axioms.Support

/-! Dependency gate for the shared augmented-relation dimension authority. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionRelationDimensions

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionRelationDimensions

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.relationRowVariables_minimum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.relationRowVariables_minimum

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.nifsShape_ne_reference25' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.nifsShape_ne_reference25

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.nifsPublicFrameFields_exceed_reference' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.nifsPublicFrameFields_exceed_reference

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.selected_exponent_core_included' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.selected_exponent_core_included

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.selected_exponent_core_count_fits' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.selected_exponent_core_count_fits

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.selected_exponent_core_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.selected_exponent_core_satisfied

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.terminalProgram_fold' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminalProgram_fold

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.terminal_program_included' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminal_program_included

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.terminal_program_count_fits' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminal_program_count_fits

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.terminal_program_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminal_program_satisfied

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.terminal_rows_and_columns_fit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminal_rows_and_columns_fit

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.Artifact.terminal_columns_scoped' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminal_columns_scoped

/-- info: 'Nightstream.Implementation.Nebula.ProductionRecursiveCoreManifestFor.length_only_accepts_zero_row_substitution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms
  ProductionRecursiveCoreManifestFor.length_only_accepts_zero_row_substitution

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.terminalLengthOnly_accepts_zero_row_substitution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms terminalLengthOnly_accepts_zero_row_substitution

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.splitExponentFit_accepts_incompatible_e1' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms splitExponentFit_accepts_incompatible_e1

/-- info: 'Nightstream.Implementation.Nebula.ProductionRelationDimensions.terminalRowCapacityOnly_does_not_imply_rectangularCapacity' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms terminalRowCapacityOnly_does_not_imply_rectangularCapacity

end tests.Axioms.NebulaProductionRelationDimensions
