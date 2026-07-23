import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.CompilerExecution

/-!
Exact source ownership of the physical compiler-linear definitions.

Owns: source-output injectivity, exact coefficient-level source-row refinement,
distinctness, and exact cardinality of the 748 physical compiler definitions.

Does not own: selected-row satisfaction, protocol acceptance, transcript
authority, commitment binding, costs, or permission to remove rows.

Assurance tier: artifact-checked for the fixed generated production profile
once this leaf validates.
-/

/-!
Emits constraints: none; this module classifies existing physical source definitions.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_disposition.physical` | Match each physical source definition to its exact generated source row. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-! ## Kernel injectivity of the source definition schedule -/

private theorem executionLower_lt_allOutputs
    {lower : Nat} {values : List Definition}
    (valid : SourceExecution.ExecutionValid (some lower) values) :
    ∀ definition ∈ values, lower < definition.output := by
  induction values generalizing lower with
  | nil => simp
  | cons head tail inductionHypothesis =>
      cases valid with
      | cons previousLt outputInSource referencesInSource referencesEarlier rest =>
          intro definition member
          simp only [List.mem_cons] at member
          rcases member with rfl | member
          · simpa [SourceExecution.PreviousOutputLt] using previousLt
          · have lowerHead : lower < head.output := by
              simpa [SourceExecution.PreviousOutputLt] using previousLt
            exact Nat.lt_trans lowerHead
              (inductionHypothesis rest definition member)

private theorem executionOutputs_nodup
    {previous : Option Nat} {values : List Definition}
    (valid : SourceExecution.ExecutionValid previous values) :
    (values.map Definition.output).Nodup := by
  induction values generalizing previous with
  | nil => exact List.nodup_nil
  | cons head tail inductionHypothesis =>
      cases valid with
      | cons previousLt outputInSource referencesInSource referencesEarlier rest =>
          rw [List.map_cons, List.nodup_cons]
          constructor
          · intro member
            rcases List.mem_map.mp member with
              ⟨future, futureMember, outputEqual⟩
            have later := executionLower_lt_allOutputs rest future futureMember
            omega
          · exact inductionHypothesis rest

/-- Every materialized source definition has a distinct output column.  This
is a kernel consequence of the already bounded strict-SSA certificate. -/
theorem sourceDefinitionOutputs_nodup :
    SourceExecution.definitionOutputs.Nodup := by
  unfold SourceExecution.definitionOutputs
  exact executionOutputs_nodup SourceExecution.sourceDefinitionsExecutionValid

private theorem executionOutput_injective
    {previous : Option Nat} {values : List Definition}
    (valid : SourceExecution.ExecutionValid previous values) :
    ∀ {left right : Definition}, left ∈ values → right ∈ values →
      left.output = right.output → left = right := by
  induction values generalizing previous with
  | nil => simp
  | cons head tail inductionHypothesis =>
      cases valid with
      | cons previousLt outputInSource referencesInSource referencesEarlier rest =>
          intro left right leftMember rightMember outputsEqual
          simp only [List.mem_cons] at leftMember rightMember
          rcases leftMember with rfl | leftMember
          · rcases rightMember with rfl | rightMember
            · rfl
            · have later := executionLower_lt_allOutputs rest right rightMember
              omega
          · rcases rightMember with rfl | rightMember
            · have later := executionLower_lt_allOutputs rest left leftMember
              omega
            · exact inductionHypothesis rest leftMember rightMember outputsEqual

/-- Source outputs identify literal source definitions because the exact
materialized schedule is strict SSA. -/
theorem sourceDefinition_output_injective
    {left right : Definition}
    (leftMember : left ∈ SourceExecution.sourceDefinitions)
    (rightMember : right ∈ SourceExecution.sourceDefinitions)
    (outputsEqual : left.output = right.output) :
    left = right :=
  executionOutput_injective SourceExecution.sourceDefinitionsExecutionValid
    leftMember rightMember outputsEqual

/-! ## Exact physical compiler-linear subset -/

def physicalDefinitionOutputs : List Nat :=
  CompilerExecution.physicalDefinitions.map Definition.output

structure PhysicalMembershipShape where
  output : Nat
  sourceRowEquivalent : Bool
deriving DecidableEq, Repr

/-- First source definition with the requested output.  Searching on the Nat
key avoids comparing a physical definition against all 7,969 full source
definitions. -/
def sourceDefinitionAtOutput? (output : Nat) : Option Definition :=
  SourceExecution.sourceDefinitions.find? fun source =>
    decide (source.output = output)

def physicalMembershipShape (definition : Definition) :
    PhysicalMembershipShape :=
  { output := definition.output
    sourceRowEquivalent :=
      match sourceDefinitionAtOutput? definition.output with
      | none => false
      | some source => decide
          (RowsPermutationEquivalent source.builderRow definition.builderRow ∧
            definition.Canonical) }

def physicalMembershipCheck (values : List PhysicalMembershipShape) : Bool :=
  values.all PhysicalMembershipShape.sourceRowEquivalent

private theorem find?_eq_some_mem_and_matches {α : Type}
    (predicate : α → Bool) :
    ∀ (values : List α) {found : α},
      values.find? predicate = some found → found ∈ values := by
  intro values
  induction values with
  | nil => simp [List.find?]
  | cons head tail inductionHypothesis =>
      intro found lookup
      cases test : predicate head with
      | false =>
          exact List.mem_cons_of_mem head
            (inductionHypothesis (by
              simpa [List.find?, test] using lookup))
      | true =>
          have equal : head = found := by
            simpa [List.find?, test] using lookup
          subst found
          exact List.mem_cons_self

private theorem find?_eq_some_matches {α : Type}
    (predicate : α → Bool) :
    ∀ (values : List α) {found : α},
      values.find? predicate = some found → predicate found = true := by
  intro values
  induction values with
  | nil => simp [List.find?]
  | cons head tail inductionHypothesis =>
      intro found lookup
      cases test : predicate head with
      | false =>
          exact inductionHypothesis (by
            simpa [List.find?, test] using lookup)
      | true =>
          have equal : head = found := by
            simpa [List.find?, test] using lookup
          simpa [← equal] using test

private theorem physicalRefinements_of_check
    {values : List Definition}
    (checked : physicalMembershipCheck
      (values.map physicalMembershipShape) = true) :
    ∀ definition ∈ values,
      ∃ source,
        source ∈ SourceExecution.sourceDefinitions ∧
        source.output = definition.output ∧
        RowsPermutationEquivalent source.builderRow definition.builderRow ∧
        definition.Canonical := by
  intro definition member
  have shapeMember : physicalMembershipShape definition ∈
      values.map physicalMembershipShape :=
    List.mem_map.mpr ⟨definition, member, rfl⟩
  have trueShape :=
    (List.all_eq_true.mp checked) (physicalMembershipShape definition)
      shapeMember
  cases lookup : sourceDefinitionAtOutput? definition.output with
  | none =>
      simp [physicalMembershipShape, lookup] at trueShape
  | some source =>
      have rowFacts :
          RowsPermutationEquivalent source.builderRow definition.builderRow ∧
            definition.Canonical :=
        of_decide_eq_true (by
          simpa [physicalMembershipShape, lookup] using trueShape)
      have sourceMember : source ∈ SourceExecution.sourceDefinitions := by
        apply find?_eq_some_mem_and_matches
          (fun candidate : Definition =>
            decide (candidate.output = definition.output))
        simpa [sourceDefinitionAtOutput?] using lookup
      have outputMatches : source.output = definition.output := by
        apply of_decide_eq_true
        apply find?_eq_some_matches
          (fun candidate : Definition =>
            decide (candidate.output = definition.output))
        simpa [sourceDefinitionAtOutput?] using lookup
      exact ⟨source, sourceMember, outputMatches, rowFacts⟩

/-! The seven subjects below are proof-free `(Nat, Bool)` records.  Their
exact cardinalities are 60, 128, 128, 128, 128, 128, and 48. -/

private theorem physicalChunk0Certificate :
    physicalMembershipCheck
      (CompilerExecution.physicalChunk0Definitions.map
        physicalMembershipShape) = true := by
  native_decide

private theorem physicalChunk1Certificate :
    physicalMembershipCheck
      (CompilerExecution.physicalChunk1Definitions.map
        physicalMembershipShape) = true := by
  native_decide

private theorem physicalChunk2Certificate :
    physicalMembershipCheck
      (CompilerExecution.physicalChunk2Definitions.map
        physicalMembershipShape) = true := by
  native_decide

private theorem physicalChunk3Certificate :
    physicalMembershipCheck
      (CompilerExecution.physicalChunk3Definitions.map
        physicalMembershipShape) = true := by
  native_decide

private theorem physicalChunk4Certificate :
    physicalMembershipCheck
      (CompilerExecution.physicalChunk4Definitions.map
        physicalMembershipShape) = true := by
  native_decide

private theorem physicalChunk5Certificate :
    physicalMembershipCheck
      (CompilerExecution.physicalChunk5Definitions.map
        physicalMembershipShape) = true := by
  native_decide

private theorem physicalChunk6Certificate :
    physicalMembershipCheck
      (CompilerExecution.physicalChunk6Definitions.map
        physicalMembershipShape) = true := by
  native_decide

/-- Every physical compiler definition has one exact source definition at the
same output whose builder row is coefficient-identical modulo sparse-term
permutation. Output equality alone is not semantic authority. -/
theorem physicalDefinitions_refine_source :
    ∀ definition ∈ CompilerExecution.physicalDefinitions,
      ∃ source,
        source ∈ SourceExecution.sourceDefinitions ∧
        source.output = definition.output ∧
        RowsPermutationEquivalent source.builderRow definition.builderRow ∧
        definition.Canonical := by
  intro definition member
  simp only [CompilerExecution.physicalDefinitions,
    List.mem_append] at member
  rcases member with member | member | member | member | member | member | member
  · exact physicalRefinements_of_check physicalChunk0Certificate definition member
  · exact physicalRefinements_of_check physicalChunk1Certificate definition member
  · exact physicalRefinements_of_check physicalChunk2Certificate definition member
  · exact physicalRefinements_of_check physicalChunk3Certificate definition member
  · exact physicalRefinements_of_check physicalChunk4Certificate definition member
  · exact physicalRefinements_of_check physicalChunk5Certificate definition member
  · exact physicalRefinements_of_check physicalChunk6Certificate definition member

private theorem independentLower_lt_allOutputs
    {known : List Nat} {lower : Nat} {values : List Definition}
    (valid : CompilerExecution.IndependentValid known (some lower) values) :
    ∀ definition ∈ values, lower < definition.output := by
  induction values generalizing lower with
  | nil => simp
  | cons head tail inductionHypothesis =>
      cases valid with
      | cons previousLt outputInSource outputFresh referencesKnown rest =>
          intro definition member
          simp only [List.mem_cons] at member
          rcases member with rfl | member
          · simpa [SourceExecution.PreviousOutputLt] using previousLt
          · have lowerHead : lower < head.output := by
              simpa [SourceExecution.PreviousOutputLt] using previousLt
            exact Nat.lt_trans lowerHead
              (inductionHypothesis rest definition member)

private theorem independentOutputs_nodup
    {known : List Nat} {previous : Option Nat} {values : List Definition}
    (valid : CompilerExecution.IndependentValid known previous values) :
    (values.map Definition.output).Nodup := by
  induction values generalizing previous with
  | nil => exact List.nodup_nil
  | cons head tail inductionHypothesis =>
      cases valid with
      | cons previousLt outputInSource outputFresh referencesKnown rest =>
          rw [List.map_cons, List.nodup_cons]
          constructor
          · intro member
            rcases List.mem_map.mp member with
              ⟨future, futureMember, outputEqual⟩
            have later := independentLower_lt_allOutputs rest future futureMember
            omega
          · exact inductionHypothesis rest

theorem physicalDefinitionOutputs_nodup : physicalDefinitionOutputs.Nodup := by
  unfold physicalDefinitionOutputs
  exact independentOutputs_nodup
    CompilerExecution.physicalDefinitionsIndependentValid

theorem physicalDefinitionOutput_count :
    physicalDefinitionOutputs.length = 748 := by
  simpa [physicalDefinitionOutputs] using
    CompilerExecution.physicalDefinition_count

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition
