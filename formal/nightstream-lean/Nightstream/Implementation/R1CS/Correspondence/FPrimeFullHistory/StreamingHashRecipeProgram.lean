import Mathlib.Data.List.Nodup
import Mathlib.Data.List.GetD
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelopeSchema
import Nightstream.Implementation.R1CS.Core.CheckedProgram

/-!
Contract: exact checked-program representation of one streaming hash recipe.

Owns the zero definition, absorb and padding definitions, the exact column
renaming of every Poseidon2 SSA definition, and their formulaic SSA
well-formedness. The resulting instruction rows are definitionally the compact
trace rows.

Does not own input authority, witness values, or lifecycle semantics.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingHashRecipeProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.Poseidon2Call
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact

def renameRhs (columnMap : Nat → Nat) : Rhs → Rhs
  | .linear terms => .linear (renameTerms columnMap terms)
  | .product left right =>
      .product (renameTerms columnMap left) (renameTerms columnMap right)

def renameDefinition
    (columnMap : Nat → Nat) (definition : Definition) : Definition where
  output := columnMap definition.output
  rhs := renameRhs columnMap definition.rhs

theorem renameRhs_refs
    (columnMap : Nat → Nat) (rhs : Rhs) :
    (renameRhs columnMap rhs).refs = rhs.refs.map columnMap := by
  cases rhs <;>
    simp [renameRhs, Rhs.refs, renameTerms, List.map_map,
      List.map_append, Function.comp_def]

theorem renameDefinition_canonical
    (columnMap : Nat → Nat) (definition : Definition) :
    (renameDefinition columnMap definition).Canonical ↔
      definition.Canonical := by
  cases definition with
  | mk output rhs =>
      cases rhs with
      | linear terms =>
          change CanonicalTerms (renameTerms columnMap terms) ↔
            CanonicalTerms terms
          constructor
          · intro mapped term member
            exact mapped (columnMap term.1, term.2)
              (by
                rw [renameTerms]
                exact List.mem_map.mpr ⟨term, member, rfl⟩)
          · intro original mappedTerm member
            rw [renameTerms] at member
            rcases List.mem_map.mp member with
              ⟨term, termMember, rfl⟩
            exact original term termMember
      | product left right =>
          simp [renameDefinition, renameRhs, Definition.Canonical]

private theorem referencesOnly_rename
    {known : List Nat} {definition : Definition}
    (references : ReferencesOnly known definition)
    (columnMap : Nat → Nat) :
    ReferencesOnly (known.map columnMap)
      (renameDefinition columnMap definition) := by
  intro column member
  rw [renameDefinition, renameRhs_refs] at member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  exact List.mem_map.mpr
    ⟨source, references source sourceMember, rfl⟩

private theorem mapped_output_fresh
    {known : List Nat} {output : Nat}
    (fresh : output ∉ known)
    {columnMap : Nat → Nat} (injective : Function.Injective columnMap) :
    columnMap output ∉ known.map columnMap := by
  intro member
  rcases List.mem_map.mp member with ⟨source, sourceMember, equal⟩
  exact fresh (by
    have : source = output := injective equal
    simpa [this] using sourceMember)

theorem wellFormed_rename
    {known : List Nat} {definitions : List Definition}
    (wellFormed : WellFormed known definitions)
    (columnMap : Nat → Nat) (injective : Function.Injective columnMap) :
    WellFormed (known.map columnMap)
      (definitions.map (renameDefinition columnMap)) := by
  induction wellFormed with
  | nil known => exact .nil (known.map columnMap)
  | @cons known head tail references fresh rest inductionHypothesis =>
      rw [List.map_cons]
      apply WellFormed.cons
      · exact referencesOnly_rename references columnMap
      · simpa [renameDefinition] using
          mapped_output_fresh fresh injective
      · simpa [renameDefinition] using inductionHypothesis

theorem wellFormed_outputs_fresh
    {known : List Nat} {definitions : List Definition}
    (wellFormed : WellFormed known definitions) :
    ∀ definition ∈ definitions, definition.output ∉ known := by
  induction wellFormed with
  | nil known => simp
  | @cons known head tail references fresh rest inductionHypothesis =>
      intro definition member
      rw [List.mem_cons] at member
      rcases member with rfl | tailMember
      · exact fresh
      · intro knownMember
        exact inductionHypothesis definition tailMember
          (List.mem_cons_of_mem head.output knownMember)

theorem wellFormed_rename_outputs
    {known : List Nat} {definitions : List Definition}
    (wellFormed : WellFormed known definitions)
    (columnMap : Nat → Nat)
    (outputsReflect : ∀ left ∈ definitions, ∀ right ∈ definitions,
      columnMap left.output = columnMap right.output →
        left.output = right.output)
    (outputsAvoidKnown : ∀ definition ∈ definitions,
      columnMap definition.output ∉ known.map columnMap) :
    WellFormed (known.map columnMap)
      (definitions.map (renameDefinition columnMap)) := by
  induction wellFormed with
  | nil known => exact .nil (known.map columnMap)
  | @cons known head tail references fresh rest inductionHypothesis =>
      rw [List.map_cons]
      apply WellFormed.cons
      · exact referencesOnly_rename references columnMap
      · simpa [renameDefinition] using
          outputsAvoidKnown head (by simp)
      · apply inductionHypothesis
        · intro left leftMember right rightMember equal
          exact outputsReflect left (by simp [leftMember]) right
            (by simp [rightMember]) equal
        · intro definition definitionMember member
          simp only [List.map_cons, List.mem_cons] at member
          rcases member with mappedHead | mappedKnown
          · have outputEqual := outputsReflect definition
              (by simp [definitionMember]) head (by simp) mappedHead
            have originallyFresh :=
              wellFormed_outputs_fresh rest definition definitionMember
            exact originallyFresh (by simp [outputEqual])
          · exact outputsAvoidKnown definition
              (by simp [definitionMember]) mappedKnown

theorem wellFormed_weaken
    {known larger : List Nat} {definitions : List Definition}
    (wellFormed : WellFormed known definitions)
    (knownIncluded : ∀ column ∈ known, column ∈ larger)
    (outputsFresh : ∀ definition ∈ definitions,
      definition.output ∉ larger) :
    WellFormed larger definitions := by
  induction wellFormed generalizing larger with
  | nil known => exact .nil larger
  | @cons known head tail references fresh rest inductionHypothesis =>
      apply WellFormed.cons
      · intro column member
        exact knownIncluded column (references column member)
      · exact outputsFresh head (by simp)
      · apply inductionHypothesis
        · intro column member
          rw [List.mem_cons] at member ⊢
          rcases member with rfl | knownMember
          · exact Or.inl rfl
          · exact Or.inr (knownIncluded column knownMember)
        · intro definition definitionMember
          have originallyFresh :=
            wellFormed_outputs_fresh rest definition definitionMember
          intro member
          rw [List.mem_cons] at member
          rcases member with equal | largerMember
          · exact originallyFresh (by simp [equal])
          · exact outputsFresh definition (by simp [definitionMember])
              largerMember

theorem wellFormed_append
    {known : List Nat} {left right : List Definition}
    (leftWellFormed : WellFormed known left)
    (rightWellFormed : WellFormed (knownAfter known left) right) :
    WellFormed known (left ++ right) := by
  induction leftWellFormed with
  | nil known => exact rightWellFormed
  | @cons known head tail references fresh rest inductionHypothesis =>
      exact .cons references fresh (inductionHypothesis rightWellFormed)

theorem output_mem_knownAfter
    {known : List Nat} {definitions : List Definition}
    {definition : Definition} (member : definition ∈ definitions) :
    definition.output ∈ knownAfter known definitions := by
  induction definitions generalizing known with
  | nil => exact (List.not_mem_nil member).elim
  | cons head tail inductionHypothesis =>
      rw [List.mem_cons] at member
      rcases member with rfl | tailMember
      · exact mem_knownAfter
          (known := definition.output :: known)
          (definitions := tail)
          List.mem_cons_self
      · exact inductionHypothesis (known := head.output :: known) tailMember

theorem mem_knownAfter_cases
    {known : List Nat} {definitions : List Definition} {column : Nat}
    (member : column ∈ knownAfter known definitions) :
    column ∈ known ∨
      ∃ definition ∈ definitions, column = definition.output := by
  induction definitions generalizing known with
  | nil => exact Or.inl member
  | cons head tail inductionHypothesis =>
      rcases inductionHypothesis
          (known := head.output :: known) member with
        knownMember | ⟨definition, definitionMember, outputExact⟩
      · rw [List.mem_cons] at knownMember
        rcases knownMember with headExact | priorMember
        · exact Or.inr ⟨head, by simp, headExact⟩
        · exact Or.inl priorMember
      · exact Or.inr
          ⟨definition, by simp [definitionMember], outputExact⟩

theorem knownAfter_below
    {known : List Nat} {definitions : List Definition} {limit : Nat}
    (knownBelow : ∀ column ∈ known, column < limit)
    (outputsBelow : ∀ definition ∈ definitions,
      definition.output < limit) :
    ∀ column ∈ knownAfter known definitions, column < limit := by
  induction definitions generalizing known with
  | nil => exact knownBelow
  | cons head tail inductionHypothesis =>
      apply inductionHypothesis
      · intro column member
        rw [List.mem_cons] at member
        rcases member with rfl | priorMember
        · exact outputsBelow head (by simp)
        · exact knownBelow column priorMember
      · intro definition definitionMember
        exact outputsBelow definition (by simp [definitionMember])

theorem knownAfter_append
    (known : List Nat) (left right : List Definition) :
    knownAfter known (left ++ right) =
      knownAfter (knownAfter known left) right := by
  induction left generalizing known with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      exact inductionHypothesis (known := head.output :: known)

theorem knownAfter_mono
    {smaller larger : List Nat} {definitions : List Definition}
    (included : ∀ column ∈ smaller, column ∈ larger) :
    ∀ column ∈ knownAfter smaller definitions,
      column ∈ knownAfter larger definitions := by
  induction definitions generalizing smaller larger with
  | nil => exact included
  | cons head tail inductionHypothesis =>
      apply inductionHypothesis
      intro column member
      rw [List.mem_cons] at member ⊢
      rcases member with rfl | priorMember
      · exact Or.inl rfl
      · exact Or.inr (included column priorMember)

theorem knownAfter_rename
    (known : List Nat) (definitions : List Definition)
    (columnMap : Nat → Nat) :
    knownAfter (known.map columnMap)
        (definitions.map (renameDefinition columnMap)) =
      (knownAfter known definitions).map columnMap := by
  induction definitions generalizing known with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change knownAfter
          (columnMap head.output :: known.map columnMap)
          (tail.map (renameDefinition columnMap)) =
        (knownAfter (head.output :: known) tail).map columnMap
      simpa only [List.map_cons] using
        inductionHypothesis (known := head.output :: known)

theorem wellFormed_of_global_bounds
    {known : List Nat} {definitions : List Definition}
    (references : ∀ definition ∈ definitions,
      ReferencesOnly known definition)
    (outputsFresh : ∀ definition ∈ definitions,
      definition.output ∉ known)
    (outputsNodup : (definitions.map Definition.output).Nodup) :
    WellFormed known definitions := by
  induction definitions generalizing known with
  | nil => exact .nil known
  | cons head tail inductionHypothesis =>
      rw [List.map_cons, List.nodup_cons] at outputsNodup
      rcases outputsNodup with ⟨headNotTail, tailNodup⟩
      apply WellFormed.cons
      · exact references head (by simp)
      · exact outputsFresh head (by simp)
      · apply inductionHypothesis
        · intro definition definitionMember column columnMember
          exact List.mem_cons_of_mem head.output
            (references definition (by simp [definitionMember])
              column columnMember)
        · intro definition definitionMember knownMember
          rw [List.mem_cons] at knownMember
          rcases knownMember with equal | priorMember
          · apply headNotTail
            have mappedMember : definition.output ∈
                tail.map Definition.output :=
              List.mem_map.mpr ⟨definition, definitionMember, rfl⟩
            simpa [equal] using mappedMember
          · exact outputsFresh definition
              (by simp [definitionMember]) priorMember
        · exact tailNodup

def callDefinitions (call : Call) : List Definition :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions.map
    (renameDefinition call.columnMap)

structure CallInputBounds (call : Call) : Prop where
  length : call.inputColumns.length = 8
  allocatedPositive : 0 < call.firstAllocatedColumn
  inputsBeforeAllocated : ∀ column ∈ call.inputColumns,
    column < call.firstAllocatedColumn

private theorem stateBeforeColumns_length
    (recipe : HashRecipe) (index : Nat) :
    (recipe.stateBeforeColumns index).length = 8 := by
  unfold HashRecipe.stateBeforeColumns HashRecipe.callOutputColumns
  split <;> simp

private theorem chunkColumns_length
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {index : Nat} (indexLt : index < absorbRounds) :
    (recipe.chunkColumns index).length = 4 := by
  have enough : 4 ≤ hashInputFields - 4 * index := by
    simp [hashInputFields, hashConstantFields, domainFields, digestFields,
      payloadFields, absorbRounds] at indexLt ⊢
    omega
  simp [HashRecipe.chunkColumns, List.length_take, List.length_drop,
    inputLength, Nat.min_eq_left enough]

private theorem definitionCount_le_four (index : Nat) :
    HashRecipe.definitionCount index ≤ 4 := by
  unfold HashRecipe.definitionCount
  split <;> omega

private theorem stateBeforeColumns_lt_roundColumnStart
    (recipe : HashRecipe) (index : Nat) :
    ∀ column ∈ recipe.stateBeforeColumns index,
      column < recipe.roundColumnStart index := by
  intro column member
  by_cases zero : index = 0
  · subst index
    simp [HashRecipe.stateBeforeColumns] at member
    subst column
    simp [HashRecipe.roundColumnStart]
  · rw [HashRecipe.stateBeforeColumns, if_neg zero] at member
    rcases List.mem_range'.mp member with ⟨offset, offsetLt, rfl⟩
    have countLe := definitionCount_le_four (index - 1)
    have predecessor : index - 1 + 1 = index := by omega
    simp only [HashRecipe.callFirstAllocatedColumn,
      HashRecipe.roundColumnStart]
    simp [absorbRoundRows, permutationRows] at *
    omega

theorem hashRecipe_callInputBounds
    (recipe : HashRecipe) (index : Nat) :
    CallInputBounds (recipe.call index) := by
  constructor
  · change (recipe.callInputColumns index).length = 8
    rw [HashRecipe.callInputColumns]
    by_cases absorb : index < absorbRounds
    · rw [if_pos absorb]
      simp [stateBeforeColumns_length]
    · rw [if_neg absorb]
      simp [stateBeforeColumns_length]
  · change 0 < recipe.callFirstAllocatedColumn index
    unfold HashRecipe.callFirstAllocatedColumn HashRecipe.roundColumnStart
    omega
  · intro column member
    change column ∈ recipe.callInputColumns index at member
    change column < recipe.callFirstAllocatedColumn index
    by_cases absorb : index < absorbRounds
    · rw [HashRecipe.callInputColumns, if_pos absorb,
        List.mem_append] at member
      rcases member with current | prior
      · rcases List.mem_range'.mp current with ⟨offset, offsetLt, rfl⟩
        simp [HashRecipe.callFirstAllocatedColumn,
          HashRecipe.definitionCount, absorb]
        omega
      · have priorMember := List.mem_of_mem_drop prior
        have priorLt := stateBeforeColumns_lt_roundColumnStart
          recipe index _ priorMember
        simp [HashRecipe.callFirstAllocatedColumn,
          HashRecipe.definitionCount, absorb]
        omega
    · rw [HashRecipe.callInputColumns, if_neg absorb,
        List.mem_cons] at member
      rcases member with rfl | prior
      · simp [HashRecipe.callFirstAllocatedColumn,
          HashRecipe.definitionCount, absorb]
      · have priorMember := List.mem_of_mem_drop prior
        have priorLt := stateBeforeColumns_lt_roundColumnStart
          recipe index _ priorMember
        simp [HashRecipe.callFirstAllocatedColumn,
          HashRecipe.definitionCount, absorb]
        omega

private theorem getD_mem
    {values : List Nat} {index : Nat} (inBounds : index < values.length) :
    values.getD index 0 ∈ values := by
  rw [← List.getElem_eq_getD (h := inBounds) 0]
  exact List.getElem_mem _

private theorem columnMap_input
    (call : Call) {column : Nat} (positive : 0 < column)
    (small : column < 9) :
    call.columnMap column =
      call.inputColumns.getD (column - 1) 0 := by
  simp [Call.columnMap, Nat.ne_of_gt positive, small]

private theorem columnMap_allocated
    (call : Call) {column : Nat} (notSmall : ¬ column < 9) :
    call.columnMap column =
      call.firstAllocatedColumn + (column - 9) := by
  have nonzero : column ≠ 0 := by omega
  simp [Call.columnMap, nonzero, notSmall]

private theorem permutationDefinition_output_ge_nine
    (definition : Definition)
    (member : definition ∈
      Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions) :
    9 ≤ definition.output := by
  have fresh := wellFormed_outputs_fresh
    Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions_wellFormed
    definition member
  simp [Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns]
    at fresh
  omega

private theorem permutationDefinition_output_lt_colCount
    (definition : Definition)
    (member : definition ∈
      Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions) :
    definition.output <
      Nightstream.Implementation.R1CS.Poseidon2Permutation.colCount := by
  have outputMember :=
    Nightstream.Implementation.R1CS.Poseidon2Permutation.definition_output_mem
      member
  rw [Nightstream.Implementation.R1CS.Poseidon2Permutation.definitionOutputColumns]
    at outputMember
  rcases List.mem_range'.mp outputMember with
    ⟨offset, offsetLt, outputExact⟩
  simp [Nightstream.Implementation.R1CS.Poseidon2Permutation.rowCount,
    Nightstream.Implementation.R1CS.Poseidon2Permutation.colCount] at *
  omega

private theorem call_outputs_reflect
    (call : Call) :
    ∀ left ∈ Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions,
      ∀ right ∈ Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions,
        call.columnMap left.output = call.columnMap right.output →
          left.output = right.output := by
  intro left leftMember right rightMember equal
  have leftGe := permutationDefinition_output_ge_nine left leftMember
  have rightGe := permutationDefinition_output_ge_nine right rightMember
  rw [columnMap_allocated call (by omega),
    columnMap_allocated call (by omega)] at equal
  omega

private theorem call_outputs_avoid_inputs
    (call : Call) (bounds : CallInputBounds call) :
    ∀ definition ∈
        Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions,
      call.columnMap definition.output ∉
        Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns.map
          call.columnMap := by
  intro definition definitionMember mappedMember
  rcases List.mem_map.mp mappedMember with
    ⟨input, inputMember, equal⟩
  have outputGe :=
    permutationDefinition_output_ge_nine definition definitionMember
  have inputLt : input < 9 := by
    simp [Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns]
      at inputMember
    omega
  have outputMapped :
      call.columnMap definition.output =
        call.firstAllocatedColumn + (definition.output - 9) :=
    columnMap_allocated call (column := definition.output) (by omega)
  by_cases inputZero : input = 0
  · subst input
    rw [call.columnMap_zero, outputMapped] at equal
    have allocatedPositive := bounds.allocatedPositive
    omega
  · have inputPositive := Nat.pos_of_ne_zero inputZero
    have inputIndex : input - 1 < call.inputColumns.length := by
      rw [bounds.length]
      omega
    have physicalMember := getD_mem inputIndex
    have physicalBelow :=
      bounds.inputsBeforeAllocated _ physicalMember
    have inputMapped :
        call.columnMap input =
          call.inputColumns.getD (input - 1) 0 :=
      columnMap_input call inputPositive inputLt
    rw [inputMapped, outputMapped] at equal
    omega

theorem callDefinitions_wellFormed
    (call : Call) (bounds : CallInputBounds call) :
    WellFormed
      (Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns.map
        call.columnMap)
      (callDefinitions call) := by
  simpa [callDefinitions] using wellFormed_rename_outputs
    Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions_wellFormed
    call.columnMap (call_outputs_reflect call)
    (call_outputs_avoid_inputs call bounds)

private theorem callDefinition_output_ge_firstAllocated
    (call : Call) {definition : Definition}
    (member : definition ∈ callDefinitions call) :
    call.firstAllocatedColumn ≤ definition.output := by
  rw [callDefinitions] at member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  have outputGe :=
    permutationDefinition_output_ge_nine source sourceMember
  change call.firstAllocatedColumn ≤ call.columnMap source.output
  rw [columnMap_allocated call (by omega)]
  omega

private theorem callDefinition_output_lt_allocatedEnd
    (call : Call) {definition : Definition}
    (member : definition ∈ callDefinitions call) :
    definition.output < call.firstAllocatedColumn + permutationRows := by
  rw [callDefinitions] at member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  have sourceGe :=
    permutationDefinition_output_ge_nine source sourceMember
  have sourceLt :=
    permutationDefinition_output_lt_colCount source sourceMember
  change call.columnMap source.output <
    call.firstAllocatedColumn + permutationRows
  rw [columnMap_allocated call (by omega)]
  simp [permutationRows,
    Nightstream.Implementation.R1CS.Poseidon2Permutation.colCount]
    at sourceLt ⊢
  omega

private theorem callDefinitions_outputsFresh
    (call : Call) {known : List Nat}
    (knownBelow : ∀ column ∈ known,
      column < call.firstAllocatedColumn) :
    ∀ definition ∈ callDefinitions call,
      definition.output ∉ known := by
  intro definition definitionMember member
  have outputGe :=
    callDefinition_output_ge_firstAllocated call definitionMember
  have outputLt := knownBelow definition.output member
  omega

private theorem callMappedInputsKnown
    (call : Call) (bounds : CallInputBounds call)
    {known : List Nat} (zeroKnown : 0 ∈ known)
    (inputsKnown : ∀ column ∈ call.inputColumns, column ∈ known) :
    ∀ column ∈
        Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns.map
          call.columnMap,
      column ∈ known := by
  intro column member
  rcases List.mem_map.mp member with
    ⟨source, sourceMember, rfl⟩
  have sourceLt : source < 9 := by
    simp [Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns]
      at sourceMember
    omega
  by_cases sourceZero : source = 0
  · subst source
    simpa using zeroKnown
  · apply inputsKnown
    rw [columnMap_input call (Nat.pos_of_ne_zero sourceZero) sourceLt]
    apply getD_mem
    rw [bounds.length]
    omega

private theorem callMappedKnownAfter
    (call : Call) {known : List Nat}
    (inputsKnown : ∀ column ∈
        Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns.map
          call.columnMap,
      column ∈ known)
    {column : Nat}
    (sourceKnown : column ∈ knownAfter
      Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns
      Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions) :
    call.columnMap column ∈ knownAfter known (callDefinitions call) := by
  have mappedKnown : call.columnMap column ∈ knownAfter
      (Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns.map
        call.columnMap)
      (Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions.map
        (renameDefinition call.columnMap)) := by
    rw [knownAfter_rename]
    exact List.mem_map.mpr ⟨column, sourceKnown, rfl⟩
  exact knownAfter_mono inputsKnown _ (by
    simpa [callDefinitions] using mappedKnown)

private theorem callOutputRangeKnown
    (call : Call) {known : List Nat}
    (inputsKnown : ∀ column ∈
        Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns.map
          call.columnMap,
      column ∈ known) :
    ∀ column ∈ List.range' (call.firstAllocatedColumn + 592) 8,
      column ∈ knownAfter known (callDefinitions call) := by
  intro column member
  rcases List.mem_range'.mp member with
    ⟨lane, laneLt, rfl⟩
  have sourceMember : 601 + lane ∈
      Nightstream.Implementation.R1CS.Poseidon2Permutation.outputColumns := by
    simp only [
      Nightstream.Implementation.R1CS.Poseidon2Permutation.outputColumns,
      List.mem_cons, List.not_mem_nil, or_false]
    omega
  have mapped := callMappedKnownAfter call inputsKnown
    (Nightstream.Implementation.R1CS.Poseidon2PermutationSound.outputs_known
      (601 + lane) sourceMember)
  convert mapped using 1
  rw [columnMap_allocated call (by omega)]
  omega

theorem callDefinitions_canonical (call : Call) :
    ∀ definition ∈ callDefinitions call, definition.Canonical := by
  intro definition member
  rcases List.mem_map.mp member with
    ⟨source, sourceMember, rfl⟩
  exact (renameDefinition_canonical call.columnMap source).mpr
    (Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions_canonical
      source sourceMember)

private theorem renameTerms_negateTerms
    (columnMap : Nat → Nat) (terms : List (Nat × Nat)) :
    renameTerms columnMap (negateTerms terms) =
      negateTerms (renameTerms columnMap terms) := by
  simp [renameTerms, negateTerms, List.map_map, Function.comp_def]

theorem renameDefinition_builderRow
    (columnMap : Nat → Nat) (mapsZero : columnMap 0 = 0)
    (definition : Definition) :
    (renameDefinition columnMap definition).builderRow =
      renameRow columnMap definition.builderRow := by
  cases definition with
  | mk output rhs =>
      cases rhs with
      | linear terms =>
          simp only [renameDefinition, renameRhs, Definition.builderRow,
            builderLinearRow, renameRow, Row.mk.injEq]
          constructor
          · rw [show
                renameTerms columnMap ((output, 1) :: negateTerms terms) =
                    (columnMap output, 1) ::
                      renameTerms columnMap (negateTerms terms) by rfl]
            exact congrArg (List.cons (columnMap output, 1))
              (renameTerms_negateTerms columnMap terms).symm
          · constructor
            · simp [renameTerms, mapsZero]
            · simp [renameTerms]
      | product left right =>
          simp [renameDefinition, renameRhs, Definition.builderRow,
            renameRow, renameTerms]

theorem callDefinitionRows (call : Call) :
    (callDefinitions call).map Definition.builderRow = call.rows := by
  rw [callDefinitions, Call.rows,
    Nightstream.Implementation.R1CS.Poseidon2Permutation.rows,
    List.map_map, List.map_map]
  apply List.map_congr_left
  intro definition _member
  exact renameDefinition_builderRow call.columnMap call.columnMap_zero definition

def transitionDefinitions (round : Round) : List Definition :=
  match round.kind with
  | .absorb chunkColumns =>
      (List.range chunkColumns.length).map fun lane =>
        { output := round.permutationInputColumns.getD lane 0
          rhs := .linear
            [(round.stateBeforeColumns.getD lane 0, 1),
             (chunkColumns.getD lane 0, 1)] }
  | .pad =>
      [{ output := round.permutationInputColumns.getD 0 0
         rhs := .linear
           [(round.stateBeforeColumns.getD 0 0, 1), (0, 1)] }]

theorem transitionDefinitions_canonical (round : Round) :
    ∀ definition ∈ transitionDefinitions round,
      definition.Canonical := by
  intro definition member
  cases kind : round.kind with
  | absorb chunkColumns =>
      simp only [transitionDefinitions, kind] at member
      rcases List.mem_map.mp member with ⟨lane, _laneMember, rfl⟩
      simp [Definition.Canonical, CanonicalTerms, goldilocksP]
  | pad =>
      simp only [transitionDefinitions, kind, List.mem_singleton] at member
      subst definition
      simp [Definition.Canonical, CanonicalTerms, goldilocksP]

theorem absorbTransitionOutputs
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {index : Nat} (indexLt : index < absorbRounds) :
    (transitionDefinitions (recipe.absorbRound index)).map
        Definition.output =
      List.range' (recipe.roundColumnStart index) 4 := by
  have chunkLength := chunkColumns_length recipe inputLength indexLt
  simp only [transitionDefinitions, HashRecipe.absorbRound, chunkLength,
    List.map_map, List.range'_eq_map_range]
  apply List.map_congr_left
  intro lane member
  have laneLt : lane < 4 := List.mem_range.mp member
  change (recipe.callInputColumns index).getD lane 0 =
    recipe.roundColumnStart index + lane
  rw [HashRecipe.callInputColumns, if_pos indexLt]
  rw [List.getD_append _ _ _ _ (by simpa using laneLt)]
  rw [← List.getElem_eq_getD (h := by simpa using laneLt) 0]
  simp

theorem padTransitionOutputs (recipe : HashRecipe) :
    (transitionDefinitions recipe.padRound).map Definition.output =
      [recipe.roundColumnStart absorbRounds] := by
  simp [transitionDefinitions, HashRecipe.padRound,
    HashRecipe.callInputColumns, absorbRounds]

theorem absorbTransitionWellFormed
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {index : Nat} (indexLt : index < absorbRounds)
    {known : List Nat}
    (inputsKnown : ∀ column ∈ recipe.inputColumns, column ∈ known)
    (stateKnown : ∀ column ∈ recipe.stateBeforeColumns index,
      column ∈ known)
    (knownBelow : ∀ column ∈ known,
      column < recipe.roundColumnStart index) :
    WellFormed known
      (transitionDefinitions (recipe.absorbRound index)) := by
  have chunkLength := chunkColumns_length recipe inputLength indexLt
  have stateLength := stateBeforeColumns_length recipe index
  apply wellFormed_of_global_bounds
  · intro definition definitionMember
    simp only [transitionDefinitions, HashRecipe.absorbRound,
      chunkLength] at definitionMember
    rcases List.mem_map.mp definitionMember with
      ⟨lane, laneMember, rfl⟩
    have laneLt : lane < 4 := List.mem_range.mp laneMember
    intro column columnMember
    simp only [Rhs.refs, List.map_cons, List.map_nil,
      List.mem_cons, List.not_mem_nil, or_false] at columnMember
    rcases columnMember with rfl | rfl
    · apply stateKnown
      apply getD_mem
      rw [stateLength]
      omega
    · have chunkMember :
          (recipe.chunkColumns index).getD lane 0 ∈
            recipe.chunkColumns index := by
        apply getD_mem
        rw [chunkLength]
        exact laneLt
      apply inputsKnown
      unfold HashRecipe.chunkColumns at chunkMember
      exact List.mem_of_mem_drop (List.mem_of_mem_take chunkMember)
  · intro definition definitionMember previousMember
    have outputMember : definition.output ∈
        (transitionDefinitions (recipe.absorbRound index)).map
          Definition.output :=
      List.mem_map.mpr ⟨definition, definitionMember, rfl⟩
    rw [absorbTransitionOutputs recipe inputLength indexLt] at outputMember
    rcases List.mem_range'.mp outputMember with
      ⟨offset, offsetLt, outputExact⟩
    have previousLt := knownBelow definition.output previousMember
    omega
  · rw [absorbTransitionOutputs recipe inputLength indexLt]
    exact List.nodup_range'

theorem padTransitionWellFormed
    (recipe : HashRecipe) {known : List Nat}
    (zeroKnown : 0 ∈ known)
    (stateKnown : ∀ column ∈ recipe.stateBeforeColumns absorbRounds,
      column ∈ known)
    (knownBelow : ∀ column ∈ known,
      column < recipe.roundColumnStart absorbRounds) :
    WellFormed known (transitionDefinitions recipe.padRound) := by
  have stateLength := stateBeforeColumns_length recipe absorbRounds
  apply wellFormed_of_global_bounds
  · intro definition definitionMember
    simp only [transitionDefinitions, HashRecipe.padRound,
      List.mem_singleton] at definitionMember
    subst definition
    intro column columnMember
    simp only [Rhs.refs, List.map_cons, List.map_nil,
      List.mem_cons, List.not_mem_nil, or_false] at columnMember
    rcases columnMember with rfl | rfl
    · apply stateKnown
      apply getD_mem
      rw [stateLength]
      omega
    · exact zeroKnown
  · intro definition definitionMember previousMember
    have outputMember : definition.output ∈
        (transitionDefinitions recipe.padRound).map Definition.output :=
      List.mem_map.mpr ⟨definition, definitionMember, rfl⟩
    rw [padTransitionOutputs recipe] at outputMember
    simp only [List.mem_singleton] at outputMember
    have previousLt := knownBelow definition.output previousMember
    omega
  · rw [padTransitionOutputs recipe]
    exact List.nodup_singleton _

theorem transitionDefinitionRows (round : Round) :
    (transitionDefinitions round).map Definition.builderRow =
      round.expectedDefinitionRows := by
  cases kind : round.kind <;>
    simp [transitionDefinitions, Round.expectedDefinitionRows, kind,
      Definition.builderRow]

def roundDefinitions (round : Round) : List Definition :=
  transitionDefinitions round ++ callDefinitions round.call

theorem roundDefinitions_canonical (round : Round) :
    ∀ definition ∈ roundDefinitions round,
      definition.Canonical := by
  intro definition member
  rw [roundDefinitions, List.mem_append] at member
  rcases member with transitionMember | callMember
  · exact transitionDefinitions_canonical round definition transitionMember
  · exact callDefinitions_canonical round.call definition callMember

theorem roundDefinitionRows (round : Round) :
    (roundDefinitions round).map Definition.builderRow = round.rows := by
  simp [roundDefinitions, Round.rows, transitionDefinitionRows,
    callDefinitionRows]

theorem roundDefinitions_wellFormed
    (round : Round) {known : List Nat}
    (transitionWellFormed :
      WellFormed known (transitionDefinitions round))
    (callBounds : CallInputBounds round.call)
    (callInputsKnown :
      ∀ column ∈
          Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns.map
            round.call.columnMap,
        column ∈ knownAfter known (transitionDefinitions round))
    (callOutputsFresh :
      ∀ definition ∈ callDefinitions round.call,
        definition.output ∉
          knownAfter known (transitionDefinitions round)) :
    WellFormed known (roundDefinitions round) := by
  rw [roundDefinitions]
  apply wellFormed_append transitionWellFormed
  exact wellFormed_weaken
    (callDefinitions_wellFormed round.call callBounds)
    callInputsKnown callOutputsFresh

private theorem absorbRoundCallInputsKnown
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {index : Nat} (indexLt : index < absorbRounds)
    {known : List Nat} (zeroKnown : 0 ∈ known)
    (stateKnown : ∀ column ∈ recipe.stateBeforeColumns index,
      column ∈ known) :
    ∀ column ∈
        Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns.map
          (recipe.call index).columnMap,
      column ∈
        knownAfter known
          (transitionDefinitions (recipe.absorbRound index)) := by
  apply callMappedInputsKnown _ (hashRecipe_callInputBounds recipe index)
  · exact mem_knownAfter zeroKnown
  · intro column member
    change column ∈ recipe.callInputColumns index at member
    rw [HashRecipe.callInputColumns, if_pos indexLt,
      List.mem_append] at member
    rcases member with transitionMember | priorMember
    · have outputMember : column ∈
          (transitionDefinitions (recipe.absorbRound index)).map
            Definition.output := by
        rw [absorbTransitionOutputs recipe inputLength indexLt]
        exact transitionMember
      rcases List.mem_map.mp outputMember with
        ⟨definition, definitionMember, rfl⟩
      exact output_mem_knownAfter definitionMember
    · apply mem_knownAfter
      exact stateKnown column (List.mem_of_mem_drop priorMember)

theorem absorbRoundDefinitions_wellFormed
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {index : Nat} (indexLt : index < absorbRounds)
    {known : List Nat} (zeroKnown : 0 ∈ known)
    (inputsKnown : ∀ column ∈ recipe.inputColumns, column ∈ known)
    (stateKnown : ∀ column ∈ recipe.stateBeforeColumns index,
      column ∈ known)
    (knownBelow : ∀ column ∈ known,
      column < recipe.roundColumnStart index) :
    WellFormed known (roundDefinitions (recipe.absorbRound index)) := by
  have transitionWellFormed := absorbTransitionWellFormed recipe
    inputLength indexLt inputsKnown stateKnown knownBelow
  apply roundDefinitions_wellFormed _ transitionWellFormed
      (hashRecipe_callInputBounds recipe index)
  · exact absorbRoundCallInputsKnown recipe inputLength indexLt
      zeroKnown stateKnown
  · apply callDefinitions_outputsFresh
    apply knownAfter_below
    · intro column member
      have columnLt := knownBelow column member
      change column < recipe.callFirstAllocatedColumn index
      simp [HashRecipe.callFirstAllocatedColumn,
        HashRecipe.definitionCount, indexLt]
      omega
    · intro definition definitionMember
      have outputMember : definition.output ∈
          (transitionDefinitions (recipe.absorbRound index)).map
            Definition.output :=
        List.mem_map.mpr ⟨definition, definitionMember, rfl⟩
      rw [absorbTransitionOutputs recipe inputLength indexLt] at outputMember
      rcases List.mem_range'.mp outputMember with
        ⟨offset, offsetLt, outputExact⟩
      change definition.output < recipe.callFirstAllocatedColumn index
      simp [HashRecipe.callFirstAllocatedColumn,
        HashRecipe.definitionCount, indexLt]
      omega

private theorem absorbRoundDefinitions_outputsBelow
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {index : Nat} (indexLt : index < absorbRounds) :
    ∀ definition ∈ roundDefinitions (recipe.absorbRound index),
      definition.output < recipe.roundColumnStart (index + 1) := by
  intro definition definitionMember
  rw [roundDefinitions, List.mem_append] at definitionMember
  rcases definitionMember with transitionMember | callMember
  · have outputMember : definition.output ∈
        (transitionDefinitions (recipe.absorbRound index)).map
          Definition.output :=
      List.mem_map.mpr ⟨definition, transitionMember, rfl⟩
    rw [absorbTransitionOutputs recipe inputLength indexLt] at outputMember
    rcases List.mem_range'.mp outputMember with
      ⟨offset, offsetLt, outputExact⟩
    simp [HashRecipe.roundColumnStart, absorbRoundRows,
      permutationRows] at *
    omega
  · have outputLt :=
      callDefinition_output_lt_allocatedEnd (recipe.call index) callMember
    change definition.output <
      recipe.callFirstAllocatedColumn index + permutationRows at outputLt
    simp [HashRecipe.callFirstAllocatedColumn,
      HashRecipe.definitionCount, indexLt,
      HashRecipe.roundColumnStart, absorbRoundRows, permutationRows]
      at outputLt ⊢
    omega

private theorem absorbRoundOutputColumnsKnown
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {index : Nat} (indexLt : index < absorbRounds)
    {known : List Nat} (zeroKnown : 0 ∈ known)
    (stateKnown : ∀ column ∈ recipe.stateBeforeColumns index,
      column ∈ known) :
    ∀ column ∈ recipe.callOutputColumns index,
      column ∈ knownAfter known
        (roundDefinitions (recipe.absorbRound index)) := by
  intro column member
  rw [roundDefinitions, knownAfter_append]
  apply callOutputRangeKnown (recipe.call index)
    (absorbRoundCallInputsKnown recipe inputLength indexLt
      zeroKnown stateKnown)
  change column ∈
    List.range' (recipe.callFirstAllocatedColumn index + 592) 8
  exact member

private def absorbPrefixDefinitions
    (recipe : HashRecipe) (count : Nat) : List Definition :=
  (List.range count).flatMap fun index =>
    roundDefinitions (recipe.absorbRound index)

private theorem absorbPrefixDefinitions_succ
    (recipe : HashRecipe) (count : Nat) :
    absorbPrefixDefinitions recipe (count + 1) =
      absorbPrefixDefinitions recipe count ++
        roundDefinitions (recipe.absorbRound count) := by
  simp [absorbPrefixDefinitions, List.range_succ]

private theorem roundColumnStart_mono
    (recipe : HashRecipe) {left right : Nat} (order : left ≤ right) :
    recipe.roundColumnStart left ≤ recipe.roundColumnStart right := by
  unfold HashRecipe.roundColumnStart
  exact Nat.add_le_add_left
    (Nat.mul_le_mul_right absorbRoundRows order)
    (recipe.zeroColumn + 1)

private theorem absorbPrefix_knownBelow
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {count : Nat} (countLe : count ≤ absorbRounds)
    {known : List Nat}
    (knownBelowZero : ∀ column ∈ known,
      column < recipe.roundColumnStart 0) :
    ∀ column ∈ knownAfter known
        (absorbPrefixDefinitions recipe count),
      column < recipe.roundColumnStart count := by
  intro column member
  rcases mem_knownAfter_cases member with
    priorMember | ⟨definition, definitionMember, outputExact⟩
  · have columnLt := knownBelowZero column priorMember
    exact Nat.lt_of_lt_of_le columnLt
      (roundColumnStart_mono recipe (Nat.zero_le count))
  · unfold absorbPrefixDefinitions at definitionMember
    rcases List.mem_flatMap.mp definitionMember with
      ⟨index, indexMember, roundMember⟩
    have indexLtCount : index < count := List.mem_range.mp indexMember
    have indexLtAbsorb : index < absorbRounds :=
      Nat.lt_of_lt_of_le indexLtCount countLe
    have outputLt := absorbRoundDefinitions_outputsBelow recipe
      inputLength indexLtAbsorb definition roundMember
    rw [outputExact]
    exact Nat.lt_of_lt_of_le outputLt
      (roundColumnStart_mono recipe (Nat.succ_le_iff.mpr indexLtCount))

private theorem absorbPrefix_wellFormed
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {count : Nat} (countLe : count ≤ absorbRounds)
    {known : List Nat} (zeroKnown : 0 ∈ known)
    (inputsKnown : ∀ column ∈ recipe.inputColumns, column ∈ known)
    (zeroColumnKnown : recipe.zeroColumn ∈ known)
    (knownBelow : ∀ column ∈ known,
      column < recipe.roundColumnStart 0) :
    WellFormed known (absorbPrefixDefinitions recipe count) ∧
      (∀ column ∈ recipe.stateBeforeColumns count,
        column ∈ knownAfter known
          (absorbPrefixDefinitions recipe count)) := by
  induction count with
  | zero =>
      constructor
      · exact .nil known
      · intro column member
        simp [HashRecipe.stateBeforeColumns] at member
        subst column
        simpa [absorbPrefixDefinitions] using zeroColumnKnown
  | succ count inductionHypothesis =>
      have indexLt : count < absorbRounds := by omega
      rcases inductionHypothesis (by omega) with
        ⟨priorWellFormed, priorStateKnown⟩
      have priorBelow := absorbPrefix_knownBelow recipe inputLength
        (count := count) (by omega) knownBelow
      let priorDefinitions := absorbPrefixDefinitions recipe count
      have roundWellFormed : WellFormed
          (knownAfter known priorDefinitions)
          (roundDefinitions (recipe.absorbRound count)) :=
        absorbRoundDefinitions_wellFormed recipe inputLength indexLt
          (mem_knownAfter zeroKnown)
          (fun column member => mem_knownAfter (inputsKnown column member))
          priorStateKnown priorBelow
      constructor
      · rw [absorbPrefixDefinitions_succ]
        exact wellFormed_append priorWellFormed roundWellFormed
      · intro column member
        rw [HashRecipe.stateBeforeColumns, if_neg (by omega)] at member
        have outputMember : column ∈ recipe.callOutputColumns count := by
          simpa using member
        have outputKnown : column ∈ knownAfter
            (knownAfter known (absorbPrefixDefinitions recipe count))
            (roundDefinitions (recipe.absorbRound count)) :=
          absorbRoundOutputColumnsKnown recipe inputLength indexLt
            (mem_knownAfter zeroKnown) priorStateKnown
            column outputMember
        have knownExact : knownAfter known
              (absorbPrefixDefinitions recipe (count + 1)) =
            knownAfter
              (knownAfter known (absorbPrefixDefinitions recipe count))
              (roundDefinitions (recipe.absorbRound count)) := by
          rw [absorbPrefixDefinitions_succ, knownAfter_append]
        rw [knownExact]
        exact outputKnown

theorem padRoundDefinitions_wellFormed
    (recipe : HashRecipe) {known : List Nat}
    (zeroKnown : 0 ∈ known)
    (stateKnown : ∀ column ∈ recipe.stateBeforeColumns absorbRounds,
      column ∈ known)
    (knownBelow : ∀ column ∈ known,
      column < recipe.roundColumnStart absorbRounds) :
    WellFormed known (roundDefinitions recipe.padRound) := by
  have transitionWellFormed :=
    padTransitionWellFormed recipe zeroKnown stateKnown knownBelow
  apply roundDefinitions_wellFormed _ transitionWellFormed
      (hashRecipe_callInputBounds recipe absorbRounds)
  · apply callMappedInputsKnown _
        (hashRecipe_callInputBounds recipe absorbRounds)
    · exact mem_knownAfter zeroKnown
    · intro column member
      change column ∈ recipe.callInputColumns absorbRounds at member
      rw [HashRecipe.callInputColumns, if_neg (Nat.lt_irrefl absorbRounds)]
        at member
      rcases List.mem_cons.mp member with transitionMember | priorMember
      · have outputMember : column ∈
            (transitionDefinitions recipe.padRound).map
              Definition.output := by
          rw [padTransitionOutputs recipe]
          simpa using transitionMember
        rcases List.mem_map.mp outputMember with
          ⟨definition, definitionMember, rfl⟩
        exact output_mem_knownAfter definitionMember
      · apply mem_knownAfter
        exact stateKnown column (List.mem_of_mem_drop priorMember)
  · apply callDefinitions_outputsFresh
    apply knownAfter_below
    · intro column member
      have columnLt := knownBelow column member
      change column < recipe.callFirstAllocatedColumn absorbRounds
      simp [HashRecipe.callFirstAllocatedColumn,
        HashRecipe.definitionCount]
      omega
    · intro definition definitionMember
      have outputMember : definition.output ∈
          (transitionDefinitions recipe.padRound).map Definition.output :=
        List.mem_map.mpr ⟨definition, definitionMember, rfl⟩
      rw [padTransitionOutputs recipe] at outputMember
      simp only [List.mem_singleton] at outputMember
      change definition.output <
        recipe.callFirstAllocatedColumn absorbRounds
      simp [HashRecipe.callFirstAllocatedColumn,
        HashRecipe.definitionCount]
      omega

def zeroDefinition (trace : Trace) : Definition where
  output := trace.zeroColumn
  rhs := .linear []

def traceDefinitions (trace : Trace) : List Definition :=
  zeroDefinition trace :: trace.rounds.flatMap roundDefinitions

theorem traceDefinitions_canonical (trace : Trace) :
    ∀ definition ∈ traceDefinitions trace,
      definition.Canonical := by
  intro definition member
  rw [traceDefinitions, List.mem_cons] at member
  rcases member with rfl | roundMember
  · simp [zeroDefinition, Definition.Canonical, CanonicalTerms]
  · rcases List.mem_flatMap.mp roundMember with
      ⟨round, _traceMember, definitionMember⟩
    exact roundDefinitions_canonical round definition definitionMember

private theorem hashRecipe_traceDefinitions_eq (recipe : HashRecipe) :
    traceDefinitions recipe.trace =
      [zeroDefinition recipe.trace] ++
        (absorbPrefixDefinitions recipe absorbRounds ++
          roundDefinitions recipe.padRound) := by
  simp only [traceDefinitions, HashRecipe.trace, HashRecipe.rounds,
    List.flatMap_append, List.flatMap_map, List.flatMap_cons,
    List.flatMap_nil, List.append_nil, absorbPrefixDefinitions,
    List.singleton_append]

private theorem absorbRoundDefinitions_outputsAtLeast
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {index : Nat} (indexLt : index < absorbRounds)
    {definition : Definition}
    (member : definition ∈ roundDefinitions (recipe.absorbRound index)) :
    recipe.roundColumnStart index ≤ definition.output := by
  rw [roundDefinitions, List.mem_append] at member
  rcases member with transitionMember | callMember
  · have outputMember : definition.output ∈
        (transitionDefinitions (recipe.absorbRound index)).map
          Definition.output :=
      List.mem_map.mpr ⟨definition, transitionMember, rfl⟩
    rw [absorbTransitionOutputs recipe inputLength indexLt] at outputMember
    rcases List.mem_range'.mp outputMember with
      ⟨offset, _offsetLt, outputExact⟩
    omega
  · have outputGe :=
      callDefinition_output_ge_firstAllocated (recipe.call index) callMember
    change recipe.callFirstAllocatedColumn index ≤ definition.output at outputGe
    simp [HashRecipe.callFirstAllocatedColumn,
      HashRecipe.definitionCount, indexLt] at outputGe
    omega

private theorem padRoundDefinitions_output_bounds
    (recipe : HashRecipe) {definition : Definition}
    (member : definition ∈ roundDefinitions recipe.padRound) :
    recipe.roundColumnStart absorbRounds ≤ definition.output ∧
      definition.output <
        recipe.roundColumnStart absorbRounds + 1 + permutationRows := by
  rw [roundDefinitions, List.mem_append] at member
  rcases member with transitionMember | callMember
  · have outputMember : definition.output ∈
        (transitionDefinitions recipe.padRound).map Definition.output :=
      List.mem_map.mpr ⟨definition, transitionMember, rfl⟩
    rw [padTransitionOutputs recipe] at outputMember
    simp only [List.mem_singleton] at outputMember
    rw [outputMember]
    constructor
    · exact Nat.le_refl _
    · omega
  · have outputGe :=
      callDefinition_output_ge_firstAllocated recipe.padRound.call callMember
    have outputLt :=
      callDefinition_output_lt_allocatedEnd recipe.padRound.call callMember
    change recipe.callFirstAllocatedColumn absorbRounds ≤
      definition.output at outputGe
    change definition.output <
      recipe.callFirstAllocatedColumn absorbRounds + permutationRows at outputLt
    simp [HashRecipe.callFirstAllocatedColumn,
      HashRecipe.definitionCount] at outputGe outputLt
    omega

/-- Every formulaic trace definition owns one column in the exact contiguous
trace-allocation interval. This is geometry only; it does not evaluate a
concrete generated trace. -/
theorem hashRecipe_traceDefinition_output_bounds
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    {definition : Definition}
    (member : definition ∈ traceDefinitions recipe.trace) :
    recipe.zeroColumn ≤ definition.output ∧
      definition.output < recipe.zeroColumn + hashTraceRows := by
  rw [hashRecipe_traceDefinitions_eq, List.mem_append] at member
  rcases member with zeroMember | roundsMember
  · simp only [List.mem_singleton] at zeroMember
    subst definition
    change recipe.zeroColumn ≤ recipe.zeroColumn ∧
      recipe.zeroColumn < recipe.zeroColumn + hashTraceRows
    constructor
    · exact Nat.le_refl _
    · have traceRowsPositive : 0 < hashTraceRows := by
        unfold hashTraceRows
        omega
      omega
  · rw [List.mem_append] at roundsMember
    rcases roundsMember with prefixMember | padMember
    · unfold absorbPrefixDefinitions at prefixMember
      rcases List.mem_flatMap.mp prefixMember with
        ⟨index, indexMember, roundMember⟩
      have indexLt : index < absorbRounds := List.mem_range.mp indexMember
      have outputGe := absorbRoundDefinitions_outputsAtLeast recipe
        inputLength indexLt roundMember
      have outputLt := absorbRoundDefinitions_outputsBelow recipe
        inputLength indexLt definition roundMember
      simp [HashRecipe.roundColumnStart, hashTraceRows,
        absorbRoundRows, permutationRows] at outputGe outputLt ⊢
      omega
    · have bounds := padRoundDefinitions_output_bounds recipe padMember
      have startGe : recipe.zeroColumn ≤
          recipe.roundColumnStart absorbRounds := by
        unfold HashRecipe.roundColumnStart
        omega
      have endExact :
          recipe.roundColumnStart absorbRounds + 1 + permutationRows =
            recipe.zeroColumn + hashTraceRows := by
        simp [HashRecipe.roundColumnStart, hashTraceRows,
          absorbRoundRows, permutationRows, Nat.add_assoc]
      exact ⟨Nat.le_trans startGe bounds.1,
        by simpa [endExact] using bounds.2⟩

theorem hashRecipe_traceDefinitions_wellFormed
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    (zeroPositive : 0 < recipe.zeroColumn)
    (inputsBelowZero : ∀ column ∈ recipe.inputColumns,
      column < recipe.zeroColumn) :
    WellFormed (0 :: recipe.inputColumns)
      (traceDefinitions recipe.trace) := by
  let initialKnown := 0 :: recipe.inputColumns
  let zero := zeroDefinition recipe.trace
  have zeroWellFormed : WellFormed initialKnown [zero] := by
    apply WellFormed.cons
    · intro column member
      simp [zero, zeroDefinition, Rhs.refs] at member
    · intro member
      change recipe.zeroColumn ∈ 0 :: recipe.inputColumns at member
      rw [List.mem_cons] at member
      rcases member with equal | inputMember
      · omega
      · have inputLt := inputsBelowZero recipe.zeroColumn inputMember
        omega
    · exact .nil _
  have zeroMember : recipe.zeroColumn ∈ knownAfter initialKnown [zero] := by
    have zeroOutputMember : zero.output ∈
        knownAfter initialKnown [zero] :=
      output_mem_knownAfter
        (known := initialKnown) (definitions := [zero])
        (definition := zero) (by simp)
    simpa [zero, zeroDefinition, HashRecipe.trace] using zeroOutputMember
  have initialBelow : ∀ column ∈ initialKnown,
      column < recipe.roundColumnStart 0 := by
    intro column member
    change column ∈ 0 :: recipe.inputColumns at member
    rw [List.mem_cons] at member
    rcases member with rfl | inputMember
    · simp [HashRecipe.roundColumnStart]
    · have inputLt := inputsBelowZero column inputMember
      simp [HashRecipe.roundColumnStart]
      omega
  have afterZeroBelow : ∀ column ∈ knownAfter initialKnown [zero],
      column < recipe.roundColumnStart 0 := by
    apply knownAfter_below initialBelow
    intro definition definitionMember
    simp only [List.mem_singleton] at definitionMember
    subst definition
    change recipe.zeroColumn < recipe.roundColumnStart 0
    simp [HashRecipe.roundColumnStart]
  have prefixProof := absorbPrefix_wellFormed recipe inputLength
    (count := absorbRounds) (by omega)
    (known := knownAfter initialKnown [zero])
    (mem_knownAfter (by simp [initialKnown]))
    (fun column member => mem_knownAfter (by
      simp [initialKnown, member]))
    zeroMember afterZeroBelow
  have prefixBelow := absorbPrefix_knownBelow recipe inputLength
    (count := absorbRounds) (by omega) afterZeroBelow
  have padWellFormed : WellFormed
      (knownAfter (knownAfter initialKnown [zero])
        (absorbPrefixDefinitions recipe absorbRounds))
      (roundDefinitions recipe.padRound) :=
    padRoundDefinitions_wellFormed recipe
      (mem_knownAfter (mem_knownAfter (by simp [initialKnown])))
      prefixProof.2 prefixBelow
  have roundsWellFormed := wellFormed_append prefixProof.1 padWellFormed
  have complete := wellFormed_append zeroWellFormed roundsWellFormed
  rw [hashRecipe_traceDefinitions_eq]
  exact complete

theorem hashRecipe_traceOutputColumns_known
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    (inputsBelowZero : ∀ column ∈ recipe.inputColumns,
      column < recipe.zeroColumn) :
    ∀ column ∈ recipe.callOutputColumns absorbRounds,
      column ∈ knownAfter (0 :: recipe.inputColumns)
        (traceDefinitions recipe.trace) := by
  let initialKnown := 0 :: recipe.inputColumns
  let zero := zeroDefinition recipe.trace
  have zeroMember : recipe.zeroColumn ∈
      knownAfter initialKnown [zero] := by
    have outputMember : zero.output ∈ knownAfter initialKnown [zero] :=
      output_mem_knownAfter (definition := zero) (by simp)
    simpa [zero, zeroDefinition, HashRecipe.trace] using outputMember
  have initialBelow : ∀ column ∈ initialKnown,
      column < recipe.roundColumnStart 0 := by
    intro column member
    change column ∈ 0 :: recipe.inputColumns at member
    rw [List.mem_cons] at member
    rcases member with rfl | inputMember
    · simp [HashRecipe.roundColumnStart]
    · have inputLt := inputsBelowZero column inputMember
      simp [HashRecipe.roundColumnStart]
      omega
  have afterZeroBelow : ∀ column ∈ knownAfter initialKnown [zero],
      column < recipe.roundColumnStart 0 := by
    apply knownAfter_below initialBelow
    intro definition definitionMember
    simp only [List.mem_singleton] at definitionMember
    subst definition
    change recipe.zeroColumn < recipe.roundColumnStart 0
    simp [zero, zeroDefinition, HashRecipe.trace,
      HashRecipe.roundColumnStart]
  have prefixProof := absorbPrefix_wellFormed recipe inputLength
    (count := absorbRounds) (by omega)
    (known := knownAfter initialKnown [zero])
    (mem_knownAfter (by simp [initialKnown]))
    (fun column member => mem_knownAfter (by
      simp [initialKnown, member]))
    zeroMember afterZeroBelow
  let prefixKnown := knownAfter (knownAfter initialKnown [zero])
    (absorbPrefixDefinitions recipe absorbRounds)
  have prefixZeroKnown : 0 ∈ prefixKnown := by
    apply mem_knownAfter
    apply mem_knownAfter
    simp [initialKnown]
  have padOutputKnown : ∀ column ∈
      recipe.callOutputColumns absorbRounds,
      column ∈ knownAfter prefixKnown
        (roundDefinitions recipe.padRound) := by
    intro column member
    rw [roundDefinitions, knownAfter_append]
    apply callOutputRangeKnown (recipe.call absorbRounds)
    · apply callMappedInputsKnown _
          (hashRecipe_callInputBounds recipe absorbRounds)
      · exact mem_knownAfter prefixZeroKnown
      · intro input inputMember
        change input ∈ recipe.callInputColumns absorbRounds at inputMember
        rw [HashRecipe.callInputColumns,
          if_neg (Nat.lt_irrefl absorbRounds)] at inputMember
        rcases List.mem_cons.mp inputMember with
          transitionMember | priorMember
        · have outputMember : input ∈
              (transitionDefinitions recipe.padRound).map
                Definition.output := by
            rw [padTransitionOutputs recipe]
            simpa using transitionMember
          rcases List.mem_map.mp outputMember with
            ⟨definition, definitionMember, rfl⟩
          exact output_mem_knownAfter definitionMember
        · apply mem_knownAfter
          exact prefixProof.2 input (List.mem_of_mem_drop priorMember)
    · change column ∈
        List.range' (recipe.callFirstAllocatedColumn absorbRounds + 592) 8
      exact member
  intro column member
  have outputKnown := padOutputKnown column member
  change column ∈ knownAfter initialKnown
    (traceDefinitions recipe.trace)
  rw [hashRecipe_traceDefinitions_eq, knownAfter_append,
    knownAfter_append]
  simpa [prefixKnown, zero] using outputKnown

theorem traceDefinitionRows (trace : Trace) :
    (traceDefinitions trace).map Definition.builderRow = trace.rows := by
  simp [traceDefinitions, Trace.rows, Trace.zeroDefinitionRows,
    zeroDefinition, Definition.builderRow, List.map_flatMap,
    roundDefinitionRows]

def traceInstructions (trace : Trace) : List Instruction :=
  (traceDefinitions trace).map .define

theorem instructionDefinitions_exact (trace : Trace) :
    CheckedProgram.definitions (traceInstructions trace) =
      traceDefinitions trace := by
  simp [traceInstructions, CheckedProgram.definitions]

theorem instructionRows_exact (trace : Trace) :
    CheckedProgram.rows (traceInstructions trace) = trace.rows := by
  simp only [traceInstructions, CheckedProgram.rows, List.map_map]
  change (traceDefinitions trace).map Definition.builderRow = trace.rows
  exact traceDefinitionRows trace

theorem traceInterpret_satisfies
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    (zeroPositive : 0 < recipe.zeroColumn)
    (inputsBelowZero : ∀ column ∈ recipe.inputColumns,
      column < recipe.zeroColumn)
    (state : Nat → Nat)
    (stateCanonical : ∀ column, state column < goldilocksP)
    (constantOne : state 0 = 1) :
    Satisfies recipe.trace.rows
      (CheckedProgram.interpret state (traceInstructions recipe.trace)) := by
  have satisfies :
      Satisfies recipe.trace.rows
        (run state (traceDefinitions recipe.trace)) := by
    rw [← traceDefinitionRows]
    exact run_satisfies_builder_rows
      (hashRecipe_traceDefinitions_wellFormed recipe inputLength
        zeroPositive inputsBelowZero)
      stateCanonical (by simp) constantOne
      (traceDefinitions_canonical recipe.trace)
  simpa only [CheckedProgram.interpret, instructionDefinitions_exact]
    using satisfies

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingHashRecipeProgram
