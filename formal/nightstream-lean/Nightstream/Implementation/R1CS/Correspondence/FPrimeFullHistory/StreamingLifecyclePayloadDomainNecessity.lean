import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSemanticLink
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingHashRecipeBlockProgram

/-!
Contract: exact omission counterexample for the lifecycle payload-domain
family.

The selected family owns the two payload Boolean slices. When it is absent,
the retained constant, Poseidon2 trace, and outer semantic-link rows accept a
deterministic assignment whose first before-payload value is two. The
independent `SemanticLink` target rejects that assignment.

This file does not claim that any row is redundant. It proves that the
complete payload-domain family must be retained.

Assurance tier: Rust-conformant for the exact lifecycle semantic-link artifact.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecyclePayloadDomainNecessity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingHashRecipeProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingHashRecipeBlockProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSemanticLink
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSemanticLink

def beforeRecipe : HashRecipe := rawArtifact.hashRecipe .before
def afterRecipe : HashRecipe := rawArtifact.hashRecipe .after

def beforeDefinitions : List Definition := definitions beforeRecipe
def afterDefinitions : List Definition := definitions afterRecipe

def sourceColumns : List Nat :=
  initialColumns beforeRecipe ++
    (afterRecipe.localColumns ++ afterRecipe.payloadColumns)

def coreDefinitions : List Definition :=
  beforeDefinitions ++ afterDefinitions

def coreKnown : List Nat :=
  knownAfter sourceColumns coreDefinitions

def equalityDefinitions
    (artifact :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact) :
    List Definition :=
  (List.range digestFields).flatMap fun lane =>
    [{ output := (artifact.semanticColumns .before).getD lane 0,
       rhs := .linear [((artifact.hashOutputColumns .before).getD lane 0, 1)] },
     { output := (artifact.semanticColumns .after).getD lane 0,
       rhs := .linear [((artifact.hashOutputColumns .after).getD lane 0, 1)] }]

def retainedDefinitions : List Definition :=
  coreDefinitions ++ equalityDefinitions rawArtifact

def retainedRows : List Row :=
  (rawArtifact.programPieces.drop 1).flatten

def sourceAssignment (column : Nat) : Nat :=
  if column = 0 then 1 else if column = 17 then 2 else 0

def omissionAssignment : Nat → Nat :=
  run sourceAssignment retainedDefinitions

private theorem getD_mem_of_lt
    {values : List Nat} {index : Nat} (inBounds : index < values.length) :
    values.getD index 0 ∈ values := by
  rw [← List.getElem_eq_getD (h := inBounds) 0]
  exact List.getElem_mem _

theorem equalityDefinitionRows
    (artifact :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact) :
    (equalityDefinitions artifact).map Definition.builderRow =
      artifact.equalityRows := by
  simp [equalityDefinitions,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact.equalityRows,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact.semanticColumns,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact.hashOutputColumns,
    Definition.builderRow, List.map_flatMap]

private theorem beforeInputsBelowStart : ∀ column ∈
    beforeRecipe.localColumns ++ beforeRecipe.payloadColumns,
    column < beforeRecipe.constantStartColumn := by
  intro column member
  change column ∈ [9, 10, 11, 12] ++
    List.range' 17 payloadFields at member
  change column < 4355
  rw [List.mem_append] at member
  rcases member with localMember | payloadMember
  · simp at localMember
    omega
  · rcases List.mem_range'.mp payloadMember with
      ⟨offset, offsetLt, rfl⟩
    unfold payloadFields at offsetLt
    omega

private theorem afterInputsBelowStart : ∀ column ∈
    afterRecipe.localColumns ++ afterRecipe.payloadColumns,
    column < afterRecipe.constantStartColumn := by
  intro column member
  change column ∈ [13, 14, 15, 16] ++
    List.range' 2186 payloadFields at member
  change column < 334752
  rw [List.mem_append] at member
  rcases member with localMember | payloadMember
  · simp at localMember
    omega
  · rcases List.mem_range'.mp payloadMember with
      ⟨offset, offsetLt, rfl⟩
    unfold payloadFields at offsetLt
    omega

private theorem sourceColumns_below_beforeStart : ∀ column ∈ sourceColumns,
    column < beforeRecipe.constantStartColumn := by
  intro column member
  change column ∈
    (0 :: ([9, 10, 11, 12] ++ List.range' 17 payloadFields)) ++
      ([13, 14, 15, 16] ++ List.range' 2186 payloadFields) at member
  change column < 4355
  rw [List.mem_append] at member
  rcases member with beforeMember | afterMember
  · rw [List.mem_cons] at beforeMember
    rcases beforeMember with rfl | beforeMember
    · omega
    · rw [List.mem_append] at beforeMember
      rcases beforeMember with localMember | payloadMember
      · simp at localMember
        omega
      · rcases List.mem_range'.mp payloadMember with
          ⟨offset, offsetLt, rfl⟩
        unfold payloadFields at offsetLt
        omega
  · rw [List.mem_append] at afterMember
    rcases afterMember with localMember | payloadMember
    · simp at localMember
      omega
    · rcases List.mem_range'.mp payloadMember with
        ⟨offset, offsetLt, rfl⟩
      unfold payloadFields at offsetLt
      omega

private theorem afterInitialIncluded : ∀ column ∈ initialColumns afterRecipe,
    column ∈ sourceColumns := by
  intro column member
  rw [initialColumns, List.mem_cons] at member
  rcases member with rfl | inputMember
  · simp [sourceColumns, initialColumns]
  · exact List.mem_append_right _ inputMember

private theorem beforeDefinitions_wellFormed :
    WellFormed sourceColumns beforeDefinitions := by
  apply wellFormed_weaken
      (definitions_wellFormed beforeRecipe (input_length .before)
        (by change 0 < 4355; omega) beforeInputsBelowStart)
  · intro column member
    exact List.mem_append_left _ member
  · intro definition definitionMember outputMember
    have outputGe :=
      (definition_output_bounds beforeRecipe (input_length .before)
        definitionMember).1
    have outputLt :=
      sourceColumns_below_beforeStart definition.output outputMember
    omega

private theorem beforeDefinition_output_lt_afterStart
    {definition : Definition} (member : definition ∈ beforeDefinitions) :
    definition.output < afterRecipe.constantStartColumn := by
  have upper :=
    (definition_output_bounds beforeRecipe (input_length .before) member).2
  change definition.output < 334752
  change definition.output < 4355 + 11 + hashTraceRows at upper
  norm_num [hashTraceRows, absorbRounds, hashInputFields,
    hashConstantFields, domainFields, payloadFields, absorbRoundRows,
    permutationRows] at upper
  exact upper

private theorem knownAfterBefore_below_afterStart : ∀ column ∈
    knownAfter sourceColumns beforeDefinitions,
    column < afterRecipe.constantStartColumn := by
  apply knownAfter_below
  · intro column member
    have below := sourceColumns_below_beforeStart column member
    change column < 334752
    change column < 4355 at below
    omega
  · intro definition member
    exact beforeDefinition_output_lt_afterStart member

private theorem afterDefinitions_wellFormed :
    WellFormed (knownAfter sourceColumns beforeDefinitions)
      afterDefinitions := by
  apply wellFormed_weaken
      (definitions_wellFormed afterRecipe (input_length .after)
        (by change 0 < 334752; omega) afterInputsBelowStart)
  · intro column member
    exact mem_knownAfter (afterInitialIncluded column member)
  · intro definition definitionMember outputMember
    have outputGe :=
      (definition_output_bounds afterRecipe (input_length .after)
        definitionMember).1
    have outputLt := knownAfterBefore_below_afterStart
      definition.output outputMember
    omega

private theorem coreDefinitions_wellFormed :
    WellFormed sourceColumns coreDefinitions := by
  rw [coreDefinitions]
  exact wellFormed_append beforeDefinitions_wellFormed
    afterDefinitions_wellFormed

private theorem coreOutputColumn_known
    (side : StateSide) {column : Nat}
    (member : column ∈ rawArtifact.hashOutputColumns side) :
    column ∈ coreKnown := by
  cases side with
  | before =>
      have localKnown := outputColumns_known beforeRecipe
        (input_length .before) beforeInputsBelowStart (output_exact .before)
        column member
      have enlargedKnown : column ∈
          knownAfter sourceColumns beforeDefinitions :=
        knownAfter_mono (fun current currentMember =>
          List.mem_append_left _ currentMember) column localKnown
      rw [coreKnown, coreDefinitions, knownAfter_append]
      exact mem_knownAfter enlargedKnown
  | after =>
      have localKnown := outputColumns_known afterRecipe
        (input_length .after) afterInputsBelowStart (output_exact .after)
        column member
      have enlargedKnown : column ∈
          knownAfter (knownAfter sourceColumns beforeDefinitions)
            afterDefinitions :=
        knownAfter_mono (fun current currentMember =>
          mem_knownAfter (afterInitialIncluded current currentMember))
          column localKnown
      simpa [coreKnown, coreDefinitions, knownAfter_append] using enlargedKnown

private theorem sourceColumn_zero_or_ge_nine
    {column : Nat} (member : column ∈ sourceColumns) :
    column = 0 ∨ 9 ≤ column := by
  change column ∈
    (0 :: ([9, 10, 11, 12] ++ List.range' 17 payloadFields)) ++
      ([13, 14, 15, 16] ++ List.range' 2186 payloadFields) at member
  rw [List.mem_append] at member
  rcases member with beforeMember | afterMember
  · rw [List.mem_cons] at beforeMember
    rcases beforeMember with rfl | beforeMember
    · exact Or.inl rfl
    · right
      rw [List.mem_append] at beforeMember
      rcases beforeMember with localMember | payloadMember
      · simp at localMember
        omega
      · rcases List.mem_range'.mp payloadMember with
          ⟨offset, _offsetLt, rfl⟩
        omega
  · right
    rw [List.mem_append] at afterMember
    rcases afterMember with localMember | payloadMember
    · simp at localMember
      omega
    · rcases List.mem_range'.mp payloadMember with
        ⟨offset, _offsetLt, rfl⟩
      omega

private theorem coreKnown_zero_or_ge_nine
    {column : Nat} (member : column ∈ coreKnown) :
    column = 0 ∨ 9 ≤ column := by
  rcases mem_knownAfter_cases member with sourceMember |
      ⟨definition, definitionMember, outputExact⟩
  · exact sourceColumn_zero_or_ge_nine sourceMember
  · right
    rw [coreDefinitions, List.mem_append] at definitionMember
    rcases definitionMember with beforeMember | afterMember
    · have lower :=
        (definition_output_bounds beforeRecipe (input_length .before)
          beforeMember).1
      change 4355 ≤ definition.output at lower
      omega
    · have lower :=
        (definition_output_bounds afterRecipe (input_length .after)
          afterMember).1
      change 334752 ≤ definition.output at lower
      omega

private theorem semanticColumn_between
    (side : StateSide) {column : Nat}
    (member : column ∈ rawArtifact.semanticColumns side) :
    1 ≤ column ∧ column ≤ 8 := by
  exact semanticColumns_between side member

private theorem equalityDefinitions_references :
    ∀ definition ∈ equalityDefinitions rawArtifact,
      ReferencesOnly coreKnown definition := by
  intro definition member
  rcases List.mem_flatMap.mp member with
    ⟨lane, laneMember, definitionMember⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false] at definitionMember
  rcases definitionMember with rfl | rfl
  · intro column referenceMember
    simp [Rhs.refs] at referenceMember
    subst column
    apply coreOutputColumn_known .before
    apply getD_mem_of_lt
    rw [hashOutputColumns_length]
    exact List.mem_range.mp laneMember
  · intro column referenceMember
    simp [Rhs.refs] at referenceMember
    subst column
    apply coreOutputColumn_known .after
    apply getD_mem_of_lt
    rw [hashOutputColumns_length]
    exact List.mem_range.mp laneMember

private theorem equalityDefinitions_outputsFresh :
    ∀ definition ∈ equalityDefinitions rawArtifact,
      definition.output ∉ coreKnown := by
  intro definition member outputKnown
  rcases List.mem_flatMap.mp member with
    ⟨lane, laneMember, definitionMember⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false] at definitionMember
  have laneLtBefore : lane <
      (rawArtifact.semanticColumns .before).length := by
    rw [semanticColumns_length]
    exact List.mem_range.mp laneMember
  have laneLtAfter : lane <
      (rawArtifact.semanticColumns .after).length := by
    rw [semanticColumns_length]
    exact List.mem_range.mp laneMember
  rcases definitionMember with rfl | rfl
  · change (rawArtifact.semanticColumns .before).getD lane 0 ∈
        coreKnown at outputKnown
    have knownShape := coreKnown_zero_or_ge_nine outputKnown
    have semanticMember :
        (rawArtifact.semanticColumns .before).getD lane 0 ∈
          rawArtifact.semanticColumns .before := getD_mem_of_lt laneLtBefore
    have bounds := semanticColumn_between .before semanticMember
    rcases knownShape with zero | high <;> omega
  · change (rawArtifact.semanticColumns .after).getD lane 0 ∈
        coreKnown at outputKnown
    have knownShape := coreKnown_zero_or_ge_nine outputKnown
    have semanticMember :
        (rawArtifact.semanticColumns .after).getD lane 0 ∈
          rawArtifact.semanticColumns .after := getD_mem_of_lt laneLtAfter
    have bounds := semanticColumn_between .after semanticMember
    rcases knownShape with zero | high <;> omega

private theorem equalityDefinitions_outputsNodup :
    ((equalityDefinitions rawArtifact).map Definition.output).Nodup := by
  simpa [equalityDefinitions, List.map_flatMap] using
    equalitySemanticColumns_nodup

private theorem equalityDefinitions_wellFormed :
    WellFormed coreKnown (equalityDefinitions rawArtifact) :=
  wellFormed_of_global_bounds equalityDefinitions_references
    equalityDefinitions_outputsFresh equalityDefinitions_outputsNodup

private theorem retainedDefinitions_wellFormed :
    WellFormed sourceColumns retainedDefinitions := by
  rw [retainedDefinitions]
  exact wellFormed_append coreDefinitions_wellFormed
    equalityDefinitions_wellFormed

private theorem equalityDefinitions_canonical :
    ∀ definition ∈ equalityDefinitions rawArtifact,
      definition.Canonical := by
  intro definition member
  rcases List.mem_flatMap.mp member with
    ⟨_lane, _laneMember, definitionMember⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false] at definitionMember
  rcases definitionMember with rfl | rfl <;>
    norm_num [Definition.Canonical, CanonicalTerms, goldilocksP]

private theorem retainedDefinitions_canonical :
    ∀ definition ∈ retainedDefinitions,
      definition.Canonical := by
  intro definition member
  rw [retainedDefinitions, List.mem_append] at member
  rcases member with coreMember | equalityMember
  · rw [coreDefinitions, List.mem_append] at coreMember
    rcases coreMember with beforeMember | afterMember
    · exact definitions_canonical beforeRecipe constantValues_canonical
        definition beforeMember
    · exact definitions_canonical afterRecipe constantValues_canonical
        definition afterMember
  · exact equalityDefinitions_canonical definition equalityMember

theorem retainedDefinitionRows :
    retainedDefinitions.map Definition.builderRow = retainedRows := by
  rw [retainedDefinitions, coreDefinitions, List.map_append,
    List.map_append, beforeDefinitions, afterDefinitions,
    definitionRows beforeRecipe,
    definitionRows afterRecipe, equalityDefinitionRows]
  simp [retainedRows,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact.programPieces,
    beforeRecipe, afterRecipe]

theorem sourceAssignment_canonical (column : Nat) :
    sourceAssignment column < goldilocksP := by
  by_cases zero : column = 0
  · simp [sourceAssignment, zero, goldilocksP]
  by_cases payload : column = 17
  · simp [sourceAssignment, zero, payload, goldilocksP]
  · simp [sourceAssignment, zero, payload, goldilocksP]

theorem sourceAssignment_one : sourceAssignment 0 = 1 := by
  simp [sourceAssignment]

theorem omissionAssignment_canonical (column : Nat) :
    omissionAssignment column < goldilocksP :=
  run_canonical sourceAssignment_canonical column

private theorem sourceColumn_preserved {column : Nat}
    (member : column ∈ sourceColumns) :
    omissionAssignment column = sourceAssignment column := by
  exact run_preserves_known retainedDefinitions_wellFormed sourceAssignment
    column member

theorem omissionAssignment_one : omissionAssignment 0 = 1 := by
  rw [sourceColumn_preserved]
  · exact sourceAssignment_one
  · simp [sourceColumns, initialColumns]

theorem omissionAssignment_payload17 : omissionAssignment 17 = 2 := by
  rw [sourceColumn_preserved]
  · simp [sourceAssignment]
  · change 17 ∈
      (0 :: ([9, 10, 11, 12] ++ List.range' 17 payloadFields)) ++
        ([13, 14, 15, 16] ++ List.range' 2186 payloadFields)
    apply List.mem_append_left
    simp [payloadFields]

theorem retainedRows_hold :
    Satisfies retainedRows omissionAssignment := by
  rw [← retainedDefinitionRows]
  exact run_satisfies_builder_rows retainedDefinitions_wellFormed
    sourceAssignment_canonical (by simp [sourceColumns, initialColumns])
    sourceAssignment_one retainedDefinitions_canonical

theorem payloadDomain_fails :
    ¬ SemanticLink rawArtifact omissionAssignment := by
  intro accepted
  have binary := accepted.payloadBinary .before 17 (by
    change 17 ∈ List.range' 17 payloadFields
    exact List.mem_range'.mpr ⟨0, by norm_num [payloadFields], by omega⟩)
  rw [omissionAssignment_payload17] at binary
  omega

/-- Lean-checked removal counterexample for the exact Rust payload-domain
artifact slice. The selected payload rows are absent; all retained semantic
link rows hold, but the complete typed target fails. -/
theorem exact_removal_counterexample :
    omissionAssignment 0 = 1 ∧
      (∀ column, omissionAssignment column < goldilocksP) ∧
      Satisfies retainedRows omissionAssignment ∧
      ¬ SemanticLink rawArtifact omissionAssignment :=
  ⟨omissionAssignment_one, omissionAssignment_canonical,
    retainedRows_hold, payloadDomain_fails⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecyclePayloadDomainNecessity
