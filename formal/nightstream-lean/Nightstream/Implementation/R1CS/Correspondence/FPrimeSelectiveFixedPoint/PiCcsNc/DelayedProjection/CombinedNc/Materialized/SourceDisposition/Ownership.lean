import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.Pivots

/-!
Exhaustive and disjoint source-definition ownership.

Owns: the hidden/physical/pivot partition, exact 6,280/748/941 cardinality
accounting, and the source-definition induction eliminator.

Does not own: selected-row satisfaction, protocol acceptance, transcript
authority, commitment binding, costs, or permission to remove rows.

Assurance tier: artifact-checked for the fixed generated production profile
once this leaf validates.
-/

/-!
Emits constraints: none; this module proves source-row ownership classification.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_disposition.ownership` | Prove exhaustive and exclusive ownership of materialized source definitions. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-! ## Exhaustive and disjoint definition ownership -/

inductive DefinitionOwner where
  | hiddenTrace
  | physicalCompilerLinear
  | rewriteTerminalPivot
deriving DecidableEq, Repr

def DefinitionOwnedBy (definition : Definition) : DefinitionOwner → Prop
  | .hiddenTrace =>
      definition ∈ SourceExecution.sourceDefinitions ∧
      definition.output ∉ physicalDefinitionOutputs ∧
      definition.output ∉ terminalPivotColumns
  | .physicalCompilerLinear =>
      definition ∈ SourceExecution.sourceDefinitions ∧
      definition.output ∈ physicalDefinitionOutputs
  | .rewriteTerminalPivot =>
      definition ∈ SourceExecution.sourceDefinitions ∧
      definition.output ∈ terminalPivotColumns

theorem sourceDefinition_has_unique_owner
    {definition : Definition}
    (member : definition ∈ SourceExecution.sourceDefinitions) :
    ∃ owner, DefinitionOwnedBy definition owner ∧
      ∀ other, DefinitionOwnedBy definition other → other = owner := by
  by_cases physical : definition.output ∈ physicalDefinitionOutputs
  · refine ⟨.physicalCompilerLinear, ⟨member, physical⟩, ?_⟩
    intro other owned
    cases other with
    | hiddenTrace => exact False.elim (owned.2.1 physical)
    | physicalCompilerLinear => rfl
    | rewriteTerminalPivot =>
        exact False.elim
          (terminalPivotColumns_disjoint_physical _ owned.2 physical)
  · by_cases pivot : definition.output ∈ terminalPivotColumns
    · refine ⟨.rewriteTerminalPivot, ⟨member, pivot⟩, ?_⟩
      intro other owned
      cases other with
      | hiddenTrace => exact False.elim (owned.2.2 pivot)
      | physicalCompilerLinear => exact False.elim (physical owned.2)
      | rewriteTerminalPivot => rfl
    · refine ⟨.hiddenTrace, ⟨member, physical, pivot⟩, ?_⟩
      intro other owned
      cases other with
      | hiddenTrace => rfl
      | physicalCompilerLinear => exact False.elim (physical owned.2)
      | rewriteTerminalPivot => exact False.elim (pivot owned.2)

/-! ## Exact owner cardinalities without concrete-list normalization

The semantic partition is `sourceDefinition_has_unique_owner`. Cardinalities
are accounted for on the already certified injective output lists: physical
outputs are distinct source outputs, terminal pivots are distinct source
outputs, and those two lists are disjoint. This avoids specializing a
recursive `filter` proof to the literal 7,969-definition schedule.
-/

/-- Exact cardinality accounting for the three unique owner classes. The
hidden count is the complement of the two disjoint, injective output-owner
lists inside the exact source-definition count. -/
theorem sourceDefinition_owner_cardinalities :
    SourceExecution.sourceDefinitions.length = 7969 ∧
    physicalDefinitionOutputs.length = 748 ∧
    terminalPivotColumns.length = 941 ∧
    SourceExecution.sourceDefinitions.length -
        physicalDefinitionOutputs.length - terminalPivotColumns.length =
      6280 := by
  rw [SourceExecution.sourceDefinition_count,
    physicalDefinitionOutput_count, terminalPivotColumn_count]
  decide

/-- Elimination principle for downstream source-definition induction.  Every
case carries the exact owner predicate, and no owner may be supplied twice. -/
theorem definitions_elim
    {Property : Definition → Prop}
    (hidden : ∀ definition,
      DefinitionOwnedBy definition .hiddenTrace → Property definition)
    (physical : ∀ definition,
      DefinitionOwnedBy definition .physicalCompilerLinear →
        Property definition)
    (pivot : ∀ definition,
      DefinitionOwnedBy definition .rewriteTerminalPivot →
        Property definition) :
    ∀ definition ∈ definitions StageProgram.instructions,
      Property definition := by
  intro definition member
  rw [← SourceExecution.sourceDefinitions_eq_stageProjection] at member
  rcases sourceDefinition_has_unique_owner member with ⟨owner, owned, unique⟩
  cases owner with
  | hiddenTrace => exact hidden definition owned
  | physicalCompilerLinear => exact physical definition owned
  | rewriteTerminalPivot => exact pivot definition owned

/-- A source definition classified as physical has a unique generated
physical compiler owner at that output, with exact coefficient-level builder
row agreement modulo sparse-term permutation. -/
theorem physical_owner_exact
    {definition : Definition}
    (owned : DefinitionOwnedBy definition .physicalCompilerLinear) :
    ∃ physical,
      physical ∈ CompilerExecution.physicalDefinitions ∧
      physical.output = definition.output ∧
      RowsPermutationEquivalent definition.builderRow physical.builderRow ∧
      physical.Canonical := by
  rcases List.mem_map.mp owned.2 with
    ⟨physical, physicalMember, outputEqual⟩
  rcases physicalDefinitions_refine_source physical physicalMember with
    ⟨source, sourceMember, sourceOutput, rowEquivalent, canonical⟩
  have exact : definition = source := sourceDefinition_output_injective
    owned.1 sourceMember (outputEqual.symm.trans sourceOutput.symm)
  subst source
  exact ⟨physical, physicalMember, outputEqual, rowEquivalent, canonical⟩

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition
