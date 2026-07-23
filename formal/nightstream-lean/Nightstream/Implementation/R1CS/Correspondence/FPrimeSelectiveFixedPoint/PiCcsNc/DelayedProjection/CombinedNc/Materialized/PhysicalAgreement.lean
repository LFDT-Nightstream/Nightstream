import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.AssignmentAgreement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.InputBoundary

/-!
Concrete source/compiler agreement for physical combined-NC definitions.

Owns: agreement between the deterministic materialized source execution and
the exact generated compiler assignment on the source input boundary and all
748 physical compiler-linear outputs.

Does not own: the 941 rewrite-terminal pivots, retained-check transfer,
selected-row satisfaction, protocol acceptance, transcript order,
commitment binding, costs, or row removal.

No executable certificate occurs here.  The proof consumes the exact
artifact-backed definition membership and input-boundary inclusion from
`SourceDisposition`, then applies the two already validated interpreters.
-/

/-!
Emits constraints: none; this module proves equality with already-emitted physical rows.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.physical_agreement` | Relate logical selected-row indices to exact physical row intervals and coefficients. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.PhysicalAgreement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

def reconstructedAssignment (assignment : Nat → Nat) : Nat → Nat :=
  SourceExecution.reconstruct
    (SourceAssignment.compilerAssignment assignment)

theorem reconstructed_canonical (assignment : Nat → Nat) :
    ∀ column, reconstructedAssignment assignment column < goldilocksP := by
  exact SourceExecution.reconstruct_canonical
    (SourceAssignment.compilerAssignmentCanonical assignment)

theorem compiler_constantOne {assignment : Nat → Nat}
    (constantOne : assignment 0 = 1) :
    SourceAssignment.compilerAssignment assignment 0 = 1 := by
  have preserved := Program.run_preserves_known
    CompilerExecution.compilerProgramWellFormed
    (SourceAssignment.retainedSeed assignment) 0 (by
      simp [CompilerExecution.retainedColumns])
  exact preserved.trans
    (SourceAssignment.retainedSeedConstantOne constantOne)

theorem reconstructed_constantOne {assignment : Nat → Nat}
    (constantOne : assignment 0 = 1) :
    reconstructedAssignment assignment 0 = 1 := by
  exact SourceExecution.reconstruct_preserves_constantOne
    (compiler_constantOne constantOne)

theorem inputAgreement (assignment : Nat → Nat) :
    AgreeOn (reconstructedAssignment assignment)
      (SourceAssignment.compilerAssignment assignment)
      SourceExecution.inputColumns := by
  exact SourceExecution.reconstruct_preserves_inputColumns
    (SourceAssignment.compilerAssignment assignment)

theorem physicalInputAgreement (assignment : Nat → Nat) :
    AgreeOn (reconstructedAssignment assignment)
      (SourceAssignment.compilerAssignment assignment)
      SourceDisposition.terminalPivotColumns →
    AgreeOn (reconstructedAssignment assignment)
      (SourceAssignment.compilerAssignment assignment)
      CompilerExecution.physicalInputColumns := by
  intro pivotAgreement column member
  rcases SourceDisposition.physicalInputColumns_subset_sourceInputOrPivots
      column member with sourceInput | pivot
  · exact inputAgreement assignment column sourceInput
  · exact pivotAgreement column pivot

/-- Every exact physical compiler-linear output has the same value in the
materialized source execution and generated compiler execution. -/
theorem physicalDefinitionOutputAgreement (assignment : Nat → Nat) :
    assignment 0 = 1 →
    AgreeOn (reconstructedAssignment assignment)
      (SourceAssignment.compilerAssignment assignment)
      SourceDisposition.terminalPivotColumns →
    AgreeOn (reconstructedAssignment assignment)
      (SourceAssignment.compilerAssignment assignment)
      SourceDisposition.physicalDefinitionOutputs := by
  intro constantOne pivotAgreement
  intro column member
  rcases List.mem_map.mp member with
    ⟨definition, definitionMember, outputEqual⟩
  subst column
  rcases SourceDisposition.physicalDefinitions_refine_source definition
      definitionMember with
    ⟨source, sourceMember, sourceOutput, rowEquivalent, physicalCanonical⟩
  have sourceProjectionMember : source ∈
      definitions StageProgram.instructions := by
    rw [← SourceExecution.sourceDefinitions_eq_stageProjection]
    exact sourceMember
  have sourceHolds := SourceExecution.reconstruct_definitionsHold
    (SourceAssignment.compilerAssignment assignment)
    source sourceProjectionMember
  have sourceCanonical :=
    StageProgram.definitions_canonical source sourceProjectionMember
  have sourceBuilderHolds :
      RowHolds (reconstructedAssignment assignment) source.builderRow := by
    exact Program.builderDefinition_complete
      (reconstructed_canonical assignment)
      (reconstructed_constantOne constantOne)
      source sourceCanonical sourceHolds
  have physicalBuilderHolds :
      RowHolds (reconstructedAssignment assignment) definition.builderRow :=
    ProjectionIndexedRows.rowHolds_of_permutationEquivalent rowEquivalent
      sourceBuilderHolds
  have reconstructedPhysicalHolds :
      Definition.Holds (reconstructedAssignment assignment) definition := by
    exact Program.builderDefinition_sound
      (reconstructed_canonical assignment)
      (reconstructed_constantOne constantOne)
      definition physicalCanonical physicalBuilderHolds
  have compilerHolds :=
    CompilerExecution.compilerAssignment_definitionsHold assignment
      definition (by
        rw [← CompilerExecution.compilerDefinitionPhases_exact]
        exact List.mem_append_right _ definitionMember)
  exact AssignmentAgreement.definitionOutput_eq_of_holds
    (physicalInputAgreement assignment pivotAgreement)
    (CompilerExecution.physicalDefinitionsIndependentValid.referencesOnly
      definition definitionMember)
    reconstructedPhysicalHolds compilerHolds

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.PhysicalAgreement
