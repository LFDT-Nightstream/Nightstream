import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalCeCompiler

/-!
Soundness and completeness primitives for the direct terminal-CE compiler.

Theorems in this file consume exact rows and compute the semantic conclusion.
Generated layout or schedule data may establish row identity and shape only;
it never carries `ClaimHolds` or any component of that conclusion.
-/

namespace Nightstream.Implementation.R1CS.TerminalCeSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.TerminalCeCompiler
open Nightstream.Implementation.R1CS.ProjectionProgram

theorem checkCenteredUnit_eq_true_iff (value : Nat) :
    checkCenteredUnit value = true ↔ CenteredUnit value := by
  simp [checkCenteredUnit, CenteredUnit, or_assoc]

theorem checkNorm_eq_true_iff (layout : Layout) (assignment : Nat → Nat) :
    checkNorm layout assignment = true ↔ NormHolds layout assignment := by
  constructor
  · intro accepted column member
    exact (checkCenteredUnit_eq_true_iff _).1
      ((List.all_eq_true.mp accepted) column member)
  · intro holds
    apply List.all_eq_true.mpr
    intro column member
    exact (checkCenteredUnit_eq_true_iff _).2 (holds column member)

theorem instruction_holds
    {instructions : List Instruction} {assignment : Nat → Nat}
    (satisfies : Satisfies (CheckedProgram.rows instructions) assignment)
    {instruction : Instruction} (member : instruction ∈ instructions) :
    RowHolds assignment instruction.row := by
  exact satisfies instruction.row (List.mem_map.mpr ⟨instruction, member, rfl⟩)

theorem satisfies_append
    {left right : List Row} {assignment : Nat → Nat}
    (leftSatisfies : Satisfies left assignment)
    (rightSatisfies : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  rcases List.mem_append.mp member with member | member
  · exact leftSatisfies row member
  · exact rightSatisfies row member

theorem satisfies_left_of_append
    {left right : List Row} {assignment : Nat → Nat}
    (satisfies : Satisfies (left ++ right) assignment) :
    Satisfies left assignment := by
  intro row member
  exact satisfies row (List.mem_append_left right member)

theorem satisfies_right_of_append
    {left right : List Row} {assignment : Nat → Nat}
    (satisfies : Satisfies (left ++ right) assignment) :
    Satisfies right assignment := by
  intro row member
  exact satisfies row (List.mem_append_right left member)

/-- The two-row `b = 2` centered-alphabet gadget implies exactly
`{0, 1, -1}` under the explicit Euclid-prime boundary. -/
theorem centeredUnitInstructions_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (column output : Nat)
    (satisfies : Satisfies
      (CheckedProgram.rows (centeredUnitInstructions column output)) assignment) :
    CenteredUnit (assignment column) := by
  have definitionRow : RowHolds assignment
      (Instruction.define ⟨output,
        .product [(column, 1), (0, 1)] [(column, 1)]⟩).row := by
    apply instruction_holds satisfies
    simp [centeredUnitInstructions]
  have outputEq := builderDefinition_sound canonical one
    ⟨output, .product [(column, 1), (0, 1)] [(column, 1)]⟩
    (by trivial) definitionRow
  have checkRow : RowHolds assignment
      (Instruction.check ⟨[(output, 1)],
        [(column, 1), (0, goldilocksP - 1)], []⟩).row := by
    apply instruction_holds satisfies
    simp [centeredUnitInstructions]
  have outputLt := canonical output
  have valueLt := canonical column
  have outputEq' : assignment output =
      ((assignment column + 1) % goldilocksP) *
        (assignment column % goldilocksP) % goldilocksP := by
    simpa [Definition.Holds, Rhs.eval, lcEval, one] using outputEq
  have checkRow' : assignment output *
      ((assignment column + (goldilocksP - 1)) % goldilocksP) %
        goldilocksP = 0 := by
    simpa [Instruction.row, RowHolds, lcEval, one,
      Nat.mod_eq_of_lt outputLt] using checkRow
  rcases prime _ _ checkRow' with outputZero | minusOneZero
  · have productZero :
        ((assignment column + 1) % goldilocksP) *
            (assignment column % goldilocksP) % goldilocksP = 0 := by
      rw [← outputEq']
      simpa [Nat.mod_eq_of_lt outputLt] using outputZero
    rcases prime _ _ productZero with plusOneZero | valueZero
    · right; right
      simp only [goldilocksP] at plusOneZero valueLt ⊢
      omega
    · left
      simp only [goldilocksP] at valueZero valueLt ⊢
      omega
  · right; left
    simp only [goldilocksP] at minusOneZero valueLt ⊢
    omega

theorem normInstructionsFrom_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    ∀ columns output,
      Satisfies (CheckedProgram.rows (normInstructionsFrom output columns))
        assignment →
      ∀ column ∈ columns, CenteredUnit (assignment column) := by
  intro columns
  induction columns with
  | nil => simp
  | cons head tail inductionHypothesis =>
      intro output satisfies column member
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · apply centeredUnitInstructions_sound prime canonical one column output
        apply satisfies_left_of_append
        simpa [normInstructionsFrom, CheckedProgram.rows] using satisfies
      · apply inductionHypothesis (output + 1)
        · apply satisfies_right_of_append
          simpa [normInstructionsFrom, CheckedProgram.rows] using satisfies
        · exact member

theorem normInstructions_sound
    (prime : EuclidPrime goldilocksP)
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows (TerminalCeCompiler.normInstructions layout))
      assignment) :
    NormHolds layout assignment := by
  exact normInstructionsFrom_sound prime canonical one layout.witnessCols
    layout.normFirstAllocatedColumn satisfies

theorem centeredUnitCheck_complete
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (column output : Nat)
    (centered : CenteredUnit (assignment column))
    (definitionHolds : Definition.Holds assignment
      ⟨output, .product [(column, 1), (0, 1)] [(column, 1)]⟩) :
    RowHolds assignment
      ⟨[(output, 1)], [(column, 1), (0, goldilocksP - 1)], []⟩ := by
  rcases centered with valueZero | valueOne | valueMinusOne
  · have outputZero : assignment output = 0 := by
      simpa [Definition.Holds, Rhs.eval, lcEval, one, valueZero] using
        definitionHolds
    simp [RowHolds, lcEval, one, valueZero, outputZero]
  · have outputValue : assignment output = 2 := by
      simpa [Definition.Holds, Rhs.eval, lcEval, one, valueOne] using
        definitionHolds
    simp [RowHolds, lcEval, one, valueOne, outputValue, goldilocksP]
  · have outputZero : assignment output = 0 := by
      have definitionEq : assignment output =
          ((assignment column + 1) % goldilocksP) *
            (assignment column % goldilocksP) % goldilocksP := by
        simpa [Definition.Holds, Rhs.eval, lcEval, one] using definitionHolds
      rw [valueMinusOne] at definitionEq
      have rhsZero :
          (((goldilocksP - 1) + 1) % goldilocksP) *
              ((goldilocksP - 1) % goldilocksP) % goldilocksP = 0 := by
        native_decide
      exact definitionEq.trans rhsZero
    simp [RowHolds, lcEval, outputZero]

theorem normChecksFrom_complete
    {assignment : Nat → Nat}
    (one : assignment 0 = 1) :
    ∀ columns output,
      (∀ column ∈ columns, CenteredUnit (assignment column)) →
      (∀ definition ∈ definitions (normInstructionsFrom output columns),
        Definition.Holds assignment definition) →
      Satisfies (checks (normInstructionsFrom output columns)) assignment := by
  intro columns
  induction columns with
  | nil =>
      intro _ _ _
      intro row member
      change row ∈ ([] : List Row) at member
      simp at member
  | cons column columns inductionHypothesis =>
      intro output centered definitionsHold
      have firstDefinition : Definition.Holds assignment
          ⟨output, .product [(column, 1), (0, 1)] [(column, 1)]⟩ := by
        apply definitionsHold
        simp [normInstructionsFrom, centeredUnitInstructions, definitions]
      have firstCheck : Satisfies
          [⟨[(output, 1)], [(column, 1), (0, goldilocksP - 1)], []⟩]
          assignment := by
        intro row member
        simp only [List.mem_singleton] at member
        subst row
        exact centeredUnitCheck_complete one column output
          (centered column (by simp)) firstDefinition
      have tailDefinitions : ∀ definition ∈
          definitions (normInstructionsFrom (output + 1) columns),
          Definition.Holds assignment definition := by
        intro definition member
        apply definitionsHold definition
        simpa [normInstructionsFrom, centeredUnitInstructions, definitions]
          using List.mem_cons_of_mem
            ({ output := output,
               rhs := Rhs.product [(column, 1), (0, 1)] [(column, 1)] })
            member
      have tailChecks := inductionHypothesis (output + 1)
        (fun current currentMember => centered current (by simp [currentMember]))
        tailDefinitions
      simpa [normInstructionsFrom, centeredUnitInstructions, checks] using
        satisfies_append firstCheck tailChecks

theorem normChecks_complete
    {layout : Layout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (norm : NormHolds layout assignment)
    (definitionsHold : ∀ definition ∈
      definitions (TerminalCeCompiler.normInstructions layout),
      Definition.Holds assignment definition) :
    Satisfies (checks (TerminalCeCompiler.normInstructions layout)) assignment := by
  exact normChecksFrom_complete one layout.witnessCols
    layout.normFirstAllocatedColumn norm definitionsHold

theorem projection_check_outputs (layout : Layout) :
    (projectionChecks layout).map LinearOutputs.Check.output =
      publicOutputColumns layout := by
  simp [projectionChecks, publicOutputColumns, List.map_flatMap,
    List.map_map, Function.comp_def]

theorem projection_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (LinearOutputs.rows (projectionChecks layout)) assignment) :
    decodePublicInput layout assignment = projectedPublic layout assignment := by
  have equalities := LinearOutputs.rows_sound canonical one (checks :=
    projectionChecks layout) (by
      intro check member
      rcases List.mem_flatMap.mp member with ⟨column, _, checkMember⟩
      rcases List.mem_map.mp checkMember with ⟨row, _, rfl⟩
      split <;> simp [LinearOutputs.Check.Canonical, CanonicalTerms] <;>
        decide) satisfies
  have values :
      (projectionChecks layout).map
          (fun check => residue (assignment check.output)) =
        (projectionChecks layout).map
          (fun check => residue (check.expected assignment)) := by
    apply List.map_congr_left
    intro check member
    rw [equalities check member]
  unfold decodePublicInput valuesAt projectedPublic
  rw [← projection_check_outputs]
  simpa only [List.map_map, Function.comp_apply, fieldAt] using values

theorem instructionSlice_mem
    {instructions : List Instruction} {start finish : Nat}
    {instruction : Instruction}
    (member : instruction ∈ instructionSlice instructions start finish) :
    instruction ∈ instructions := by
  exact List.mem_of_mem_drop (List.mem_of_mem_take member)

theorem checks_slice_satisfy
    {instructions : List Instruction} {assignment : Nat → Nat}
    (satisfies : Satisfies (checks instructions) assignment)
    (start finish : Nat) :
    Satisfies (checks (instructionSlice instructions start finish)) assignment := by
  intro row rowMember
  rcases List.mem_filterMap.mp rowMember with
    ⟨instruction, instructionMember, mapped⟩
  cases instruction with
  | define definition => simp at mapped
  | check current =>
      simp only at mapped
      cases mapped
      apply satisfies row
      apply List.mem_filterMap.mpr
      exact ⟨.check row, instructionSlice_mem instructionMember, rfl⟩

theorem rows_slice_satisfy
    {instructions : List Instruction} {assignment : Nat → Nat}
    (satisfies : Satisfies (CheckedProgram.rows instructions) assignment)
    (start finish : Nat) :
    Satisfies
      (CheckedProgram.rows (instructionSlice instructions start finish))
      assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, member, rfl⟩
  exact satisfies instruction.row
    (List.mem_map.mpr ⟨instruction, instructionSlice_mem member, rfl⟩)

theorem valuesAt_outputs_eq_expected
    {checks : List LinearOutputs.Check}
    {outputs known : List Nat}
    {assignment final : Nat → Nat}
    (outputsEq : checks.map LinearOutputs.Check.output = outputs)
    (outputsKnown : ∀ output ∈ outputs, output ∈ known)
    (agreement : AgreeOn final assignment known)
    (equalities : ∀ check ∈ checks,
      final check.output = check.expected final) :
    valuesAt assignment outputs =
      checks.map fun check => residue (check.expected final) := by
  rw [← outputsEq]
  unfold valuesAt fieldAt
  simp only [List.map_map]
  apply List.map_congr_left
  intro check member
  change residue (assignment check.output) =
    residue (check.expected final)
  congr 1
  calc
    assignment check.output = final check.output :=
      (agreement check.output (outputsKnown check.output
        (outputsEq ▸ List.mem_map.mpr ⟨check, member, rfl⟩))).symm
    _ = check.expected final := equalities check member

universe u v w

theorem map_eq_map_pointwise
    {α : Type u} {β : Type v}
    {values : List α} {f g : α → β}
    (equal : values.map f = values.map g) :
    ∀ value ∈ values, f value = g value := by
  induction values with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.cons.injEq] at equal
      intro value member
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · exact equal.1
      · exact inductionHypothesis equal.2 value member

theorem map_aligned
    {α : Type u} {β : Type v} {γ : Type w}
    {left : List α} {right : List β} {f : α → γ} {g : β → γ}
    (lengths : left.length = right.length)
    (equal : left.map f = right.map g) :
    ∀ a b, (a, b) ∈ left.zip right → f a = g b := by
  induction left generalizing right with
  | nil => simp
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp at lengths
      | cons rightHead rightTail =>
          simp only [List.map_cons, List.cons.injEq] at equal
          have tailLengths : leftTail.length = rightTail.length := by
            simpa using lengths
          intro a b member
          simp only [List.zip_cons_cons, List.mem_cons] at member
          rcases member with pairEqual | member
          · cases pairEqual
            exact equal.1
          · exact inductionHypothesis tailLengths equal.2 a b member

/-- Converse of `valuesAt_outputs_eq_expected` for canonical assignments.
It is the completeness bridge from a decoded semantic equality back to every
linear output assertion. -/
theorem equalities_of_valuesAt_eq_expected
    {checks : List LinearOutputs.Check}
    {outputs known : List Nat}
    {state final : Nat → Nat}
    (outputsEq : checks.map LinearOutputs.Check.output = outputs)
    (outputsKnown : ∀ output ∈ outputs, output ∈ known)
    (agreement : AgreeOn final state known)
    (stateCanonical : ∀ column, state column < goldilocksP)
    (valuesEqual : valuesAt state outputs =
      checks.map fun check => residue (check.expected final)) :
    ∀ check ∈ checks, final check.output = check.expected final := by
  rw [← outputsEq] at valuesEqual
  unfold valuesAt fieldAt at valuesEqual
  simp only [List.map_map] at valuesEqual
  have pointwise := map_eq_map_pointwise valuesEqual
  intro check member
  have fieldEqual := pointwise check member
  have residueEqual : state check.output % goldilocksP =
      check.expected final % goldilocksP := by
    simpa [residue] using congrArg Fin.val fieldEqual
  have expectedLt : check.expected final < goldilocksP := by
    exact Nat.mod_lt _ (by decide)
  calc
    final check.output = state check.output :=
      agreement check.output (outputsKnown check.output
        (outputsEq ▸ List.mem_map.mpr ⟨check, member, rfl⟩))
    _ = state check.output % goldilocksP :=
      (Nat.mod_eq_of_lt (stateCanonical check.output)).symm
    _ = check.expected final % goldilocksP := residueEqual
    _ = check.expected final := Nat.mod_eq_of_lt expectedLt

theorem splitByLengths_map_flatten
    {α : Type u} {β : Type v}
    (rows : List (List α)) (f : α → β) :
    TerminalCeCompiler.Program.splitByLengths (rows.map List.length)
        (rows.flatten.map f) =
      rows.map (List.map f) := by
  induction rows with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.flatten_cons,
        TerminalCeCompiler.Program.splitByLengths]
      rw [List.map_append, List.take_append_of_le_length (by simp),
        List.drop_append_of_le_length (by simp)]
      have headLength : head.length = (head.map f).length := by simp
      rw [headLength, List.take_length, List.drop_length]
      simpa using congrArg (fun rows => (head.map f) :: rows)
        inductionHypothesis

/-- Pairing into the quadratic extension is injective when no odd trailing
base-field limb can be discarded. -/
theorem pairs_injective_of_even_length
    {left right : List ProjectionProgram.F}
    (lengths : left.length = right.length)
    (even : left.length % 2 = 0)
    (equal : pairs left = pairs right) :
    left = right := by
  cases left with
  | nil =>
      have rightEmpty : right = [] := List.eq_nil_of_length_eq_zero lengths.symm
      exact rightEmpty.symm
  | cons left0 leftTail =>
      cases leftTail with
      | nil => simp at even
      | cons left1 leftRest =>
          cases right with
          | nil => simp at lengths
          | cons right0 rightTail =>
              cases rightTail with
              | nil => simp at lengths
              | cons right1 rightRest =>
                  simp only [pairs, List.cons.injEq] at equal
                  have headEqual : K.mk left0 left1 = K.mk right0 right1 := equal.1
                  have tailLengths : leftRest.length = rightRest.length := by
                    simpa using lengths
                  have tailEven : leftRest.length % 2 = 0 := by
                    change (leftRest.length + 1 + 1) % 2 = 0 at even
                    rw [show leftRest.length + 1 + 1 = leftRest.length + 2 by
                      omega] at even
                    simpa [Nat.add_mod] using even
                  have tailEqual := pairs_injective_of_even_length
                    tailLengths tailEven equal.2
                  cases headEqual
                  simp [tailEqual]
termination_by left.length
decreasing_by
  simp_wf
  subst_vars
  exact Nat.lt_trans (Nat.lt_succ_self _) (Nat.lt_succ_self _)

theorem splitByLengths_lengths
    {α : Type u} (lengths : List Nat) (values : List α)
    (within : lengths.sum ≤ values.length) :
    (Program.splitByLengths lengths values).map List.length = lengths := by
  induction lengths generalizing values with
  | nil => rfl
  | cons length lengths inductionHypothesis =>
      simp only [List.sum_cons] at within
      simp only [Program.splitByLengths, List.map_cons, List.cons.injEq]
      constructor
      · rw [List.length_take]
        omega
      · apply inductionHypothesis
        rw [List.length_drop]
        omega

theorem splitByLengths_flatten
    {α : Type u} (lengths : List Nat) (values : List α)
    (exact : lengths.sum = values.length) :
    (Program.splitByLengths lengths values).flatten = values := by
  induction lengths generalizing values with
  | nil =>
      have valuesEmpty : values = [] := List.eq_nil_of_length_eq_zero exact.symm
      simp [valuesEmpty, Program.splitByLengths]
  | cons length lengths inductionHypothesis =>
      simp only [List.sum_cons] at exact
      simp only [Program.splitByLengths, List.flatten_cons]
      rw [inductionHypothesis]
      · exact List.take_append_drop length values
      · rw [List.length_drop]
        omega

theorem map_pairs_injective
    {left right : List (List ProjectionProgram.F)}
    (lengths : left.map List.length = right.map List.length)
    (even : ∀ row ∈ left, row.length % 2 = 0)
    (equal : left.map pairs = right.map pairs) :
    left = right := by
  induction left generalizing right with
  | nil =>
      have rightEmpty : right = [] := by
        cases right <;> simp_all
      exact rightEmpty.symm
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp at lengths
      | cons rightHead rightTail =>
          simp only [List.map_cons, List.cons.injEq] at lengths equal
          have headEven := even leftHead (by simp)
          have headEqual := pairs_injective_of_even_length lengths.1
            headEven equal.1
          have tailEven : ∀ row ∈ leftTail, row.length % 2 = 0 := by
            intro row member
            exact even row (by simp [member])
          have tailEqual := inductionHypothesis lengths.2 tailEven equal.2
          simp [headEqual, tailEqual]

theorem evaluationFields_eq_of_decoded
    {program : TerminalCeCompiler.Program} {state : Nat → Nat}
    (shape : ShapeValid program.layout)
    (outputsEq : program.evaluationChecks.map
      LinearOutputs.Check.output = program.layout.evaluationCols.flatten)
    (equal : program.expectedEvaluations state =
      decodeEvaluations program.layout state) :
    program.expectedFields state Program.evaluationChecks =
      valuesAt state program.layout.evaluationCols.flatten := by
  let lengths := program.layout.evaluationCols.map List.length
  let expectedRows := Program.splitByLengths lengths
    (program.expectedFields state Program.evaluationChecks)
  let actualRows := program.layout.evaluationCols.map
    (List.map (fieldAt state))
  have outputLengths := congrArg List.length outputsEq
  have expectedLength : lengths.sum =
      (program.expectedFields state Program.evaluationChecks).length := by
    simpa [lengths, Program.expectedFields] using outputLengths.symm
  have expectedRowLengths : expectedRows.map List.length = lengths := by
    exact splitByLengths_lengths lengths _ (Nat.le_of_eq expectedLength)
  have actualRowLengths : actualRows.map List.length = lengths := by
    simp [actualRows, lengths, List.map_map, Function.comp_def]
  have alignedLengths : actualRows.map List.length =
      expectedRows.map List.length := actualRowLengths.trans
    expectedRowLengths.symm
  have actualEven : ∀ row ∈ actualRows, row.length % 2 = 0 := by
    intro row member
    rcases List.mem_map.mp member with ⟨columns, columnsMember, rfl⟩
    simpa using shape.evaluationRowsEven columns columnsMember
  have pairedEqual : actualRows.map pairs = expectedRows.map pairs := by
    simpa [actualRows, expectedRows, decodeEvaluations,
      Program.expectedEvaluations, valuesAt] using equal.symm
  have rowsEqual : actualRows = expectedRows :=
    map_pairs_injective alignedLengths actualEven pairedEqual
  have expectedFlatten : expectedRows.flatten =
      program.expectedFields state Program.evaluationChecks := by
    exact splitByLengths_flatten lengths _ expectedLength
  have actualFlatten : actualRows.flatten =
      valuesAt state program.layout.evaluationCols.flatten := by
    simp [actualRows, valuesAt, List.map_flatten]
  calc
    program.expectedFields state Program.evaluationChecks =
        expectedRows.flatten := expectedFlatten.symm
    _ = actualRows.flatten := congrArg List.flatten rowsEqual.symm
    _ = valuesAt state program.layout.evaluationCols.flatten := actualFlatten

theorem ncFields_eq_of_decoded
    {program : TerminalCeCompiler.Program} {state : Nat → Nat}
    (shape : ShapeValid program.layout)
    (outputsEq : program.ncChecks.map LinearOutputs.Check.output =
      program.layout.ncEvaluationCols)
    (equal : program.expectedNcEvaluations state =
      (decodeSidecar program.layout state).evaluations) :
    program.expectedFields state Program.ncChecks =
      valuesAt state program.layout.ncEvaluationCols := by
  have outputLengths := congrArg List.length outputsEq
  have lengths :
      (program.expectedFields state Program.ncChecks).length =
        (valuesAt state program.layout.ncEvaluationCols).length := by
    simpa [Program.expectedFields, valuesAt] using outputLengths
  have even : (program.expectedFields state Program.ncChecks).length % 2 = 0 := by
    rw [lengths]
    simp only [valuesAt, List.length_map, shape.ncEvaluationSize]
    omega
  apply pairs_injective_of_even_length lengths even
  simpa [Program.expectedNcEvaluations, decodeSidecar] using equal

theorem projection_complete
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (equal : decodePublicInput layout assignment =
      projectedPublic layout assignment) :
    Satisfies (LinearOutputs.rows (projectionChecks layout)) assignment := by
  apply LinearOutputs.rows_complete canonical one
  · intro check member
    rcases List.mem_flatMap.mp member with ⟨column, _, checkMember⟩
    rcases List.mem_map.mp checkMember with ⟨row, _, rfl⟩
    split <;> simp [LinearOutputs.Check.Canonical, CanonicalTerms] <;>
      decide
  · apply equalities_of_valuesAt_eq_expected (projection_check_outputs layout)
      (known := publicOutputColumns layout)
    · intro output member
      exact member
    · intro _ _
      rfl
    · exact canonical
    · simpa [decodePublicInput, projectedPublic] using equal

/-- The two output-equality rows emitted per evaluation pin the carried
constant term to the first quadratic-extension coefficient of that
evaluation. -/
theorem constantTermChecks_sound
    {layout : Layout} {assignment : Nat → Nat}
    (shape : ShapeValid layout)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (LinearOutputs.rows (constantTermChecks layout)) assignment) :
    (decodeEvaluations layout assignment).map
        (fun evaluation => evaluation.headD K.zero) =
      decodeConstantTerms layout assignment := by
  have equalities := LinearOutputs.rows_sound canonical one
    (checks := constantTermChecks layout) (by
      intro check member
      rcases List.mem_flatMap.mp member with ⟨pair, _, checkMember⟩
      simp at checkMember
      rcases checkMember with rfl | rfl <;>
        simp [LinearOutputs.Check.Canonical, CanonicalTerms] <;> decide) satisfies
  have go : ∀ (constantTerms : List KColumns)
      (evaluations : List (List Nat)),
      constantTerms.length = evaluations.length →
      (∀ row ∈ evaluations, 2 ≤ row.length) →
      (∀ check : LinearOutputs.Check, check ∈
        ((constantTerms.zip evaluations).flatMap fun pair =>
          [⟨pair.1.c0, [(pair.2.getD 0 0, 1)],
              LinearOutputs.Orientation.forward⟩,
           ⟨pair.1.c1, [(pair.2.getD 1 0, 1)],
              LinearOutputs.Orientation.forward⟩]) →
        assignment check.output = check.expected assignment) →
      (evaluations.map fun row =>
          (pairs (valuesAt assignment row)).headD K.zero) =
        constantTerms.map (kAt assignment) := by
    intro constantTerms
    induction constantTerms with
    | nil =>
        intro evaluations lengths _ _
        have evaluationsEmpty : evaluations = [] := by
          exact List.eq_nil_of_length_eq_zero (by simpa using lengths.symm)
        simp [evaluationsEmpty]
    | cons constantTerm constantTerms inductionHypothesis =>
        intro evaluations lengths rowLengths checksEqual
        cases evaluations with
        | nil => simp at lengths
        | cons evaluation evaluations =>
            have evaluationLength := rowLengths evaluation (by simp)
            cases evaluation with
            | nil => simp at evaluationLength
            | cons c0 tail =>
                cases tail with
                | nil => simp at evaluationLength
                | cons c1 rest =>
                    have c0Equal := checksEqual
                      ⟨constantTerm.c0, [(c0, 1)],
                        LinearOutputs.Orientation.forward⟩ (by simp)
                    have c1Equal := checksEqual
                      ⟨constantTerm.c1, [(c1, 1)],
                        LinearOutputs.Orientation.forward⟩ (by simp)
                    have tailLengths : constantTerms.length = evaluations.length := by
                      simpa using lengths
                    have tailRows : ∀ row ∈ evaluations, 2 ≤ row.length := by
                      intro row member
                      exact rowLengths row (by simp [member])
                    have tailChecks : ∀ check : LinearOutputs.Check, check ∈
                        ((constantTerms.zip evaluations).flatMap fun pair =>
                          [⟨pair.1.c0, [(pair.2.getD 0 0, 1)],
                              LinearOutputs.Orientation.forward⟩,
                           ⟨pair.1.c1, [(pair.2.getD 1 0, 1)],
                              LinearOutputs.Orientation.forward⟩]) →
                        assignment check.output = check.expected assignment := by
                      intro check member
                      apply checksEqual check
                      simp only [List.zip_cons_cons, List.flatMap_cons,
                        List.mem_append]
                      exact Or.inr member
                    have tailEqual := inductionHypothesis evaluations tailLengths
                      tailRows tailChecks
                    simp only [List.map_cons, List.cons.injEq]
                    constructor
                    · rcases constantTerm with ⟨constantC0, constantC1⟩
                      have c0Nat : assignment constantC0 = assignment c0 := by
                        simpa [LinearOutputs.Check.expected, lcEval,
                          Nat.mod_eq_of_lt (canonical c0)] using c0Equal
                      have c1Nat : assignment constantC1 = assignment c1 := by
                        simpa [LinearOutputs.Check.expected, lcEval,
                          Nat.mod_eq_of_lt (canonical c1)] using c1Equal
                      change K.mk (residue (assignment c0))
                          (residue (assignment c1)) =
                        K.mk (residue (assignment constantC0))
                          (residue (assignment constantC1))
                      rw [c0Nat, c1Nat]
                    · exact tailEqual
  unfold decodeEvaluations decodeConstantTerms kValuesAt
  simpa only [List.map_map, Function.comp_apply] using
    go layout.constantTermCols layout.evaluationCols
      shape.constantTermSize shape.evaluationRowsNonempty
      (by simpa [constantTermChecks] using equalities)

theorem constantTermChecks_complete
    {layout : Layout} {assignment : Nat → Nat}
    (shape : ShapeValid layout)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (equal : (decodeEvaluations layout assignment).map
        (fun evaluation => evaluation.headD K.zero) =
      decodeConstantTerms layout assignment) :
    Satisfies (LinearOutputs.rows (constantTermChecks layout)) assignment := by
  have aligned : ∀ constantTerm evaluation,
      (constantTerm, evaluation) ∈
        layout.constantTermCols.zip layout.evaluationCols →
      kAt assignment constantTerm =
        (pairs (valuesAt assignment evaluation)).headD K.zero := by
    apply map_aligned shape.constantTermSize
    simpa [decodeEvaluations, decodeConstantTerms, kValuesAt,
      List.map_map, Function.comp_def] using equal.symm
  apply LinearOutputs.rows_complete canonical one
  · intro check member
    rcases List.mem_flatMap.mp member with ⟨pair, _, checkMember⟩
    simp at checkMember
    rcases checkMember with rfl | rfl <;>
      simp [LinearOutputs.Check.Canonical, CanonicalTerms] <;> decide
  · intro check member
    rcases List.mem_flatMap.mp member with
      ⟨⟨constantTerm, evaluation⟩, pairMember, checkMember⟩
    have evaluationLength := shape.evaluationRowsNonempty evaluation
      (List.of_mem_zip pairMember).2
    cases evaluation with
    | nil => simp at evaluationLength
    | cons c0 tail =>
        cases tail with
        | nil => simp at evaluationLength
        | cons c1 rest =>
            have pairEqual := aligned constantTerm (c0 :: c1 :: rest)
              pairMember
            have c0Field : residue (assignment constantTerm.c0) =
                residue (assignment c0) := by
              simpa [kAt, KColumns.value, baseAt, valuesAt, fieldAt, pairs]
                using congrArg K.c0 pairEqual
            have c1Field : residue (assignment constantTerm.c1) =
                residue (assignment c1) := by
              simpa [kAt, KColumns.value, baseAt, valuesAt, fieldAt, pairs]
                using congrArg K.c1 pairEqual
            have c0Nat : assignment constantTerm.c0 = assignment c0 := by
              simpa [residue, Nat.mod_eq_of_lt (canonical constantTerm.c0),
                Nat.mod_eq_of_lt (canonical c0)] using congrArg Fin.val c0Field
            have c1Nat : assignment constantTerm.c1 = assignment c1 := by
              simpa [residue, Nat.mod_eq_of_lt (canonical constantTerm.c1),
                Nat.mod_eq_of_lt (canonical c1)] using congrArg Fin.val c1Field
            simp at checkMember
            rcases checkMember with rfl | rfl
            · simpa [LinearOutputs.Check.expected, lcEval,
                Nat.mod_eq_of_lt (canonical c0)] using c0Nat
            · simpa [LinearOutputs.Check.expected, lcEval,
                Nat.mod_eq_of_lt (canonical c1)] using c1Nat

end Nightstream.Implementation.R1CS.TerminalCeSound
