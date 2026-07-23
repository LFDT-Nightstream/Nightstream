import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.PhysicalAgreement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RetainedCompilerRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.RetainedChecks

/-!
Conditional satisfaction of the literal production combined-NC source rows.

Owns: kernel-only transport from literal selected-row satisfaction through
the exact 52 retained checks and the independently reconstructed source
program to all 8,021 generated source rows.

Does not own: visible-column agreement, selector-one or constant-one
enforcement, protocol acceptance, transcript authority, raw-child authority,
commitment binding, costs, or permission to remove rows. The visible-column
agreement premise is the precise contract discharged by `VisibleAgreement`.

Assurance tier: artifact-checked for the fixed generated production profile
once this leaf validates.
-/

/-!
Emits constraints: none; this module interprets satisfaction of existing source rows.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_satisfaction` | Relate decoded sparse-row satisfaction to the materialized source equations. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceSatisfaction

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-! ## Generic exact-row transport -/

/-- Forward satisfaction transport along lockstep sparse-row permutation.
This theorem is structural: specializing it to the 52 retained rows does not
evaluate either concrete row list. -/
private theorem satisfies_of_lockstep
    {source checked : List Row} {assignment : Nat → Nat}
    (lockstep : RowsPermutationEquivalentList source checked)
    (sourceSatisfies : Satisfies source assignment) :
    Satisfies checked assignment := by
  induction source generalizing checked with
  | nil =>
      cases checked with
      | nil =>
          intro row member
          simp at member
      | cons _ _ => simp [RowsPermutationEquivalentList] at lockstep
  | cons sourceRow sourceRows inductionHypothesis =>
      cases checked with
      | nil => simp [RowsPermutationEquivalentList] at lockstep
      | cons checkedRow checkedRows =>
          rcases lockstep with ⟨headEquivalent, tailEquivalent⟩
          intro row member
          simp only [List.mem_cons] at member
          rcases member with rfl | tailMember
          · exact rowHolds_of_permutationEquivalent headEquivalent
              (sourceSatisfies sourceRow (by simp))
          · apply inductionHypothesis tailEquivalent
            · intro candidate candidateMember
              exact sourceSatisfies candidate (by simp [candidateMember])
            · exact tailMember

/-! ## Exact reconstruction identity -/

/-- `SourceProgram` and `SourceExecution` execute the same exact definition
projection of `StageProgram.instructions`; neither side carries a separate
assignment or source-satisfaction premise. -/
theorem sourceProgramReconstruct_eq_sourceExecution
    (seed : Nat → Nat) :
    SourceProgram.reconstruct seed StageProgram.instructions =
      SourceExecution.reconstruct seed := by
  rfl

theorem sourceProgramReconstruct_eq_reconstructedAssignment
    (assignment : Nat → Nat) :
    SourceProgram.reconstruct
        (SourceAssignment.compilerAssignment assignment)
        StageProgram.instructions =
      PhysicalAgreement.reconstructedAssignment assignment := by
  rw [sourceProgramReconstruct_eq_sourceExecution]
  rfl

/-! ## Selected rows to all literal source rows -/

set_option maxRecDepth 100000 in
private theorem retainedChecksSatisfy_onCompilerAssignment
    {assignment : Nat → Nat}
    (selectedRows :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    Satisfies (checks StageProgram.instructions)
      (SourceAssignment.compilerAssignment assignment) := by
  apply satisfies_of_lockstep SourceDisposition.retainedChecks_lockstep
  exact
    RetainedCompilerRows.generatedEmittedRowsSatisfy_implies_retainedRawRowsSatisfy
      selectedRows selectorOne

private theorem retainedChecksHold_onReconstruction
    {assignment : Nat → Nat}
    (selectedRows :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1)
    (visibleAgreement :
      AgreeOn (PhysicalAgreement.reconstructedAssignment assignment)
        (SourceAssignment.compilerAssignment assignment)
        SourceDisposition.visibleDefinitionColumns) :
    ChecksHold (SourceAssignment.compilerAssignment assignment)
      StageProgram.instructions := by
  have compilerChecks :=
    retainedChecksSatisfy_onCompilerAssignment selectedRows selectorOne
  change Satisfies (checks StageProgram.instructions)
    (PhysicalAgreement.reconstructedAssignment assignment)
  intro row member
  apply (rowHolds_agree visibleAgreement row
    (SourceDisposition.retainedChecks_referencesOnly row member)).mpr
  exact compilerChecks row member

/-- Literal selected-row satisfaction plus the explicit visible-agreement,
selector, and constant-one boundaries establishes every literal generated
source row on the independent source reconstruction. No check truth or source
satisfaction proposition is accepted as a premise. -/
theorem generatedEmittedRowsSatisfy_implies_generatedSourceRowsSatisfy
    {assignment : Nat → Nat}
    (selectedRows :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1)
    (constantOne : assignment 0 = 1)
    (visibleAgreement :
      AgreeOn (PhysicalAgreement.reconstructedAssignment assignment)
        (SourceAssignment.compilerAssignment assignment)
        SourceDisposition.visibleDefinitionColumns) :
    Satisfies SourceProgram.generatedRows
      (PhysicalAgreement.reconstructedAssignment assignment) := by
  have checksHold := retainedChecksHold_onReconstruction selectedRows
    selectorOne visibleAgreement
  have sourceRows := SourceProgram.reconstruct_satisfies_generatedRows
    StageProgram.sourceRows_exact SourceExecution.stageProgramWellFormed
    StageProgram.definitions_canonical
    (SourceAssignment.compilerAssignmentCanonical assignment)
    SourceExecution.constantOne_mem_inputColumns
    (PhysicalAgreement.compiler_constantOne constantOne) checksHold
  rw [sourceProgramReconstruct_eq_reconstructedAssignment] at sourceRows
  exact sourceRows

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceSatisfaction
