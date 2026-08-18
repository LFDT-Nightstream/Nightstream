import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingHashRecipeProgram

/-!
Contract: structural checked-program completeness for one full HashRecipe
block, including its constant pins and formulaic Poseidon2 trace.

Does not own a concrete artifact, lifecycle semantics, or payload authority.

Assurance tier: model-level.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingHashRecipeBlockProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingHashRecipeProgram

def initialColumns (recipe : HashRecipe) : List Nat :=
  0 :: (recipe.localColumns ++ recipe.payloadColumns)

def constantDefinitionsFrom : Nat → List Nat → List Definition
  | _, [] => []
  | column, value :: values =>
      { output := column, rhs := .linear [(0, value)] } ::
        constantDefinitionsFrom (column + 1) values

def constantDefinitions (recipe : HashRecipe) : List Definition :=
  constantDefinitionsFrom recipe.constantStartColumn recipe.constantValues

private theorem constantDefinitionsFrom_rows
    (start : Nat) (values : List Nat) :
    (constantDefinitionsFrom start values).map Definition.builderRow =
      ((List.range' start values.length).zip values).map fun entry =>
        builderLinearRow entry.1 [(0, entry.2)] := by
  induction values generalizing start with
  | nil => rfl
  | cons value values inductionHypothesis =>
      simp [constantDefinitionsFrom, List.range'_succ,
        Definition.builderRow, inductionHypothesis]

theorem constantDefinitionRows (recipe : HashRecipe) :
    (constantDefinitions recipe).map Definition.builderRow =
      constantRows recipe := by
  exact constantDefinitionsFrom_rows recipe.constantStartColumn
    recipe.constantValues

private theorem constantDefinitionsFrom_output_bounds
    (start : Nat) (values : List Nat) {definition : Definition}
    (member : definition ∈ constantDefinitionsFrom start values) :
    start ≤ definition.output ∧
      definition.output < start + values.length := by
  induction values generalizing start with
  | nil => simp [constantDefinitionsFrom] at member
  | cons value values inductionHypothesis =>
      rw [constantDefinitionsFrom, List.mem_cons] at member
      rcases member with rfl | tailMember
      · simp
      · have bounds := inductionHypothesis (start := start + 1) tailMember
        simp only [List.length_cons]
        omega

theorem constantDefinition_output_bounds
    (recipe : HashRecipe) {definition : Definition}
    (member : definition ∈ constantDefinitions recipe) :
    recipe.constantStartColumn ≤ definition.output ∧
      definition.output < recipe.zeroColumn := by
  have bounds := constantDefinitionsFrom_output_bounds
    recipe.constantStartColumn recipe.constantValues member
  simpa [HashRecipe.zeroColumn, constantDefinitions] using bounds

private theorem constantDefinitionsFrom_columnsKnown
    (start : Nat) (values : List Nat) (known : List Nat) :
    ∀ column ∈ List.range' start values.length,
      column ∈ knownAfter known (constantDefinitionsFrom start values) := by
  induction values generalizing start known with
  | nil => simp
  | cons value values inductionHypothesis =>
      intro column member
      simp only [List.length_cons] at member
      rw [List.range'_succ, List.mem_cons] at member
      rcases member with rfl | tailMember
      · have headMember :
            ({ output := column, rhs := .linear [(0, value)] } : Definition) ∈
              constantDefinitionsFrom column (value :: values) := by
          simp [constantDefinitionsFrom]
        exact output_mem_knownAfter headMember
      · simp only [constantDefinitionsFrom, knownAfter]
        exact inductionHypothesis (start := start + 1)
          (known := start :: known) column tailMember

theorem constantColumns_known
    (recipe : HashRecipe) (known : List Nat) :
    ∀ column ∈ recipe.constantColumns,
      column ∈ knownAfter known (constantDefinitions recipe) := by
  simpa [HashRecipe.constantColumns, constantDefinitions] using
    constantDefinitionsFrom_columnsKnown recipe.constantStartColumn
      recipe.constantValues known

private theorem constantDefinitionsFrom_canonical
    (start : Nat) (values : List Nat)
    (valuesCanonical : ∀ value ∈ values,
      0 < value ∧ value < goldilocksP) :
    ∀ definition ∈ constantDefinitionsFrom start values,
      definition.Canonical := by
  induction values generalizing start with
  | nil => simp [constantDefinitionsFrom]
  | cons value values inductionHypothesis =>
      intro definition member
      rw [constantDefinitionsFrom, List.mem_cons] at member
      rcases member with rfl | tailMember
      · simpa [Definition.Canonical, CanonicalTerms] using
          valuesCanonical value (by simp)
      · apply inductionHypothesis (start := start + 1)
        · intro current currentMember
          exact valuesCanonical current (by simp [currentMember])
        · exact tailMember

theorem constantDefinitions_canonical
    (recipe : HashRecipe)
    (valuesCanonical : ∀ value ∈ recipe.constantValues,
      0 < value ∧ value < goldilocksP) :
    ∀ definition ∈ constantDefinitions recipe,
      definition.Canonical :=
  constantDefinitionsFrom_canonical recipe.constantStartColumn
    recipe.constantValues valuesCanonical

private theorem constantDefinitionsFrom_wellFormed
    (start : Nat) (values : List Nat) {known : List Nat}
    (zeroKnown : 0 ∈ known)
    (knownBelow : ∀ column ∈ known, column < start) :
    WellFormed known (constantDefinitionsFrom start values) := by
  induction values generalizing start known with
  | nil => exact .nil known
  | cons value values inductionHypothesis =>
      apply WellFormed.cons
      · intro column member
        simp [Rhs.refs] at member
        subst column
        exact zeroKnown
      · intro member
        exact (Nat.lt_irrefl start) (knownBelow start member)
      · apply inductionHypothesis (start := start + 1)
        · simp [zeroKnown]
        · intro column member
          rw [List.mem_cons] at member
          rcases member with rfl | priorMember
          · omega
          · have below := knownBelow column priorMember
            omega

theorem constantDefinitions_wellFormed
    (recipe : HashRecipe) {known : List Nat}
    (zeroKnown : 0 ∈ known)
    (knownBelow : ∀ column ∈ known,
      column < recipe.constantStartColumn) :
    WellFormed known (constantDefinitions recipe) :=
  constantDefinitionsFrom_wellFormed recipe.constantStartColumn
    recipe.constantValues zeroKnown knownBelow

def definitions (recipe : HashRecipe) : List Definition :=
  constantDefinitions recipe ++ traceDefinitions recipe.trace

theorem definitions_canonical
    (recipe : HashRecipe)
    (valuesCanonical : ∀ value ∈ recipe.constantValues,
      0 < value ∧ value < goldilocksP) :
    ∀ definition ∈ definitions recipe, definition.Canonical := by
  intro definition member
  rw [definitions, List.mem_append] at member
  rcases member with constantMember | traceMember
  · exact constantDefinitions_canonical recipe valuesCanonical
      definition constantMember
  · exact traceDefinitions_canonical recipe.trace definition traceMember

theorem definitionRows (recipe : HashRecipe) :
    (definitions recipe).map Definition.builderRow =
      constantRows recipe ++ recipe.trace.rows := by
  simp [definitions, constantDefinitionRows, traceDefinitionRows]

theorem definitions_wellFormed
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    (constantStartPositive : 0 < recipe.constantStartColumn)
    (inputsBelowStart : ∀ column ∈
      recipe.localColumns ++ recipe.payloadColumns,
      column < recipe.constantStartColumn) :
    WellFormed (initialColumns recipe) (definitions recipe) := by
  have initialBelow : ∀ column ∈ initialColumns recipe,
      column < recipe.constantStartColumn := by
    intro column member
    rw [initialColumns, List.mem_cons] at member
    rcases member with rfl | inputMember
    · exact constantStartPositive
    · exact inputsBelowStart column inputMember
  have constantsWellFormed := constantDefinitions_wellFormed recipe
    (known := initialColumns recipe) (by simp [initialColumns]) initialBelow
  have afterConstantsBelow : ∀ column ∈
      knownAfter (initialColumns recipe) (constantDefinitions recipe),
      column < recipe.zeroColumn := by
    apply knownAfter_below
    · intro column member
      have below := initialBelow column member
      simp [HashRecipe.zeroColumn]
      omega
    · intro definition member
      exact (constantDefinition_output_bounds recipe member).2
  have traceBase := hashRecipe_traceDefinitions_wellFormed recipe inputLength
    (by
      simp only [HashRecipe.zeroColumn]
      omega) (by
      intro column member
      rw [HashRecipe.inputColumns, List.mem_append] at member
      rcases member with constantsOrLocal | payloadMember
      · rw [List.mem_append] at constantsOrLocal
        rcases constantsOrLocal with constantMember | localMember
        · rw [HashRecipe.constantColumns] at constantMember
          rcases List.mem_range'.mp constantMember with
            ⟨offset, offsetLt, rfl⟩
          simp [HashRecipe.zeroColumn]
          omega
        · have below := inputsBelowStart column (by simp [localMember])
          simp [HashRecipe.zeroColumn]
          omega
      · have below := inputsBelowStart column (by simp [payloadMember])
        simp [HashRecipe.zeroColumn]
        omega)
  have traceWellFormed : WellFormed
      (knownAfter (initialColumns recipe) (constantDefinitions recipe))
      (traceDefinitions recipe.trace) := by
    apply wellFormed_weaken traceBase
    · intro column member
      rw [List.mem_cons] at member
      rcases member with rfl | inputMember
      · exact mem_knownAfter (by simp [initialColumns])
      · rw [HashRecipe.inputColumns, List.mem_append] at inputMember
        rcases inputMember with constantsOrLocal | payloadMember
        · rw [List.mem_append] at constantsOrLocal
          rcases constantsOrLocal with constantMember | localMember
          · exact constantColumns_known recipe (initialColumns recipe)
              column constantMember
          · exact mem_knownAfter (by simp [initialColumns, localMember])
        · exact mem_knownAfter (by simp [initialColumns, payloadMember])
    · intro definition member knownMember
      have outputGe :=
        (hashRecipe_traceDefinition_output_bounds recipe inputLength member).1
      have knownLt := afterConstantsBelow definition.output knownMember
      omega
  rw [definitions]
  exact wellFormed_append constantsWellFormed traceWellFormed

theorem definition_output_bounds
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {definition : Definition} (member : definition ∈ definitions recipe) :
    recipe.constantStartColumn ≤ definition.output ∧
      definition.output < recipe.zeroColumn + hashTraceRows := by
  rw [definitions, List.mem_append] at member
  rcases member with constantMember | traceMember
  · have bounds := constantDefinition_output_bounds recipe constantMember
    have traceRowsPositive : 0 < hashTraceRows := by
      unfold hashTraceRows
      omega
    exact ⟨bounds.1, Nat.lt_trans bounds.2 (by omega)⟩
  · have bounds := hashRecipe_traceDefinition_output_bounds recipe
      inputLength traceMember
    have startLeZero :
        recipe.constantStartColumn ≤ recipe.zeroColumn := by
      simp only [HashRecipe.zeroColumn]
      omega
    exact ⟨Nat.le_trans startLeZero bounds.1, bounds.2⟩

theorem outputColumns_known
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    (inputsBelowStart : ∀ column ∈
      recipe.localColumns ++ recipe.payloadColumns,
      column < recipe.constantStartColumn)
    (outputExact : recipe.outputColumns =
      (recipe.callOutputColumns absorbRounds).take 4) :
    ∀ column ∈ recipe.outputColumns,
      column ∈ knownAfter (initialColumns recipe) (definitions recipe) := by
  have inputsBelowZero : ∀ column ∈ recipe.inputColumns,
      column < recipe.zeroColumn := by
    intro column member
    rw [HashRecipe.inputColumns, List.mem_append] at member
    rcases member with constantsOrLocal | payloadMember
    · rw [List.mem_append] at constantsOrLocal
      rcases constantsOrLocal with constantMember | localMember
      · rw [HashRecipe.constantColumns] at constantMember
        rcases List.mem_range'.mp constantMember with
          ⟨offset, offsetLt, rfl⟩
        simp [HashRecipe.zeroColumn]
        omega
      · have below := inputsBelowStart column (by simp [localMember])
        simp [HashRecipe.zeroColumn]
        omega
    · have below := inputsBelowStart column (by simp [payloadMember])
      simp [HashRecipe.zeroColumn]
      omega
  have traceInputsIncluded : ∀ column ∈ 0 :: recipe.inputColumns,
      column ∈ knownAfter (initialColumns recipe)
        (constantDefinitions recipe) := by
    intro column member
    rw [List.mem_cons] at member
    rcases member with rfl | inputMember
    · exact mem_knownAfter (by simp [initialColumns])
    · rw [HashRecipe.inputColumns, List.mem_append] at inputMember
      rcases inputMember with constantsOrLocal | payloadMember
      · rw [List.mem_append] at constantsOrLocal
        rcases constantsOrLocal with constantMember | localMember
        · exact constantColumns_known recipe (initialColumns recipe)
            column constantMember
        · exact mem_knownAfter (by simp [initialColumns, localMember])
      · exact mem_knownAfter (by simp [initialColumns, payloadMember])
  intro column member
  have callMember : column ∈ recipe.callOutputColumns absorbRounds := by
    rw [outputExact] at member
    exact List.mem_of_mem_take member
  have traceKnown := hashRecipe_traceOutputColumns_known recipe inputLength
    inputsBelowZero column callMember
  rw [definitions, knownAfter_append]
  exact knownAfter_mono traceInputsIncluded column traceKnown

theorem interpret_satisfies
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    (constantStartPositive : 0 < recipe.constantStartColumn)
    (inputsBelowStart : ∀ column ∈
      recipe.localColumns ++ recipe.payloadColumns,
      column < recipe.constantStartColumn)
    (valuesCanonical : ∀ value ∈ recipe.constantValues,
      0 < value ∧ value < goldilocksP)
    (state : Nat → Nat)
    (stateCanonical : ∀ column, state column < goldilocksP)
    (constantOne : state 0 = 1) :
    Satisfies (constantRows recipe ++ recipe.trace.rows)
      (run state (definitions recipe)) := by
  rw [← definitionRows]
  exact run_satisfies_builder_rows
    (definitions_wellFormed recipe inputLength constantStartPositive
      inputsBelowStart)
    stateCanonical (by simp [initialColumns]) constantOne
    (definitions_canonical recipe valuesCanonical)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingHashRecipeBlockProgram
