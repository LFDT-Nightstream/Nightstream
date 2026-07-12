import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPiDecArtifact

/-!
Soundness and completeness primitives for the strict-PiDEC semantic compiler.

The artifact-specific theorem at the end starts from satisfaction of every
exported production row and concludes `PiDecStrictCompiler.Accepted`, whose
fields are independent recomposition, point, shape, alphabet, padding, and
digest equations.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.PiDecStrictCompiler

theorem canonicalTerms_zip_of_powers
    {columns powers : List Nat}
    (canonical : ∀ coefficient ∈ powers,
      0 < coefficient ∧ coefficient < goldilocksP) :
    CanonicalTerms (columns.zip powers) := by
  intro term member
  induction columns generalizing powers with
  | nil => simp at member
  | cons column tail inductionHypothesis =>
      cases powers with
      | nil => simp at member
      | cons coefficient rest =>
          simp only [List.zip_cons_cons, List.mem_cons] at member
          rcases member with rfl | member
          · exact canonical coefficient (by simp)
          · apply inductionHypothesis
            · intro candidate candidateMember
              exact canonical candidate (List.mem_cons_of_mem coefficient candidateMember)
            · exact member

theorem instruction_holds
    {instructions : List Instruction} {assignment : Nat → Nat}
    (satisfies : Satisfies (CheckedProgram.rows instructions) assignment)
    {instruction : Instruction} (member : instruction ∈ instructions) :
    RowHolds assignment instruction.row := by
  apply satisfies instruction.row
  exact List.mem_map.mpr ⟨instruction, member, rfl⟩

theorem group_satisfies
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment)
    {group : List Instruction} (member : group ∈ groups layout) :
    Satisfies (CheckedProgram.rows group) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  apply satisfies instruction.row
  apply List.mem_map.mpr
  refine ⟨instruction, ?_, rfl⟩
  exact List.mem_flatten.mpr ⟨group, member, instructionMember⟩

theorem equalityCheck_sound
    {assignment : Nat → Nat} (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) {lhs rhs : Nat}
    (holds : RowHolds assignment (equalityCheck lhs rhs).row) :
    assignment lhs = assignment rhs := by
  have canonicalTerms : CanonicalTerms [(rhs, 1)] := by
    simp [CanonicalTerms, goldilocksP]
  have decoded := builderLinearRow_sound canonical one lhs [(rhs, 1)]
    canonicalTerms (by
      simpa [equalityCheck, Instruction.row, builderLinearRow, negateTerms,
        negCoeff, goldilocksP] using holds)
  simpa [lcEval, Nat.mod_eq_of_lt (canonical rhs)] using decoded

theorem zeroCheck_sound
    {assignment : Nat → Nat} (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) {column : Nat}
    (holds : RowHolds assignment (zeroCheck column).row) :
    assignment column = 0 := by
  have decoded := builderLinearRow_sound canonical one column [] (by
    simp [CanonicalTerms])
  apply decoded
  simpa [zeroCheck, Instruction.row, builderLinearRow, negateTerms,
    lcEval] using holds

theorem recompositionCheck_sound
    {assignment : Nat → Nat} (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) {parent : Nat} {children powers : List Nat}
    (powerCanonical : ∀ coefficient ∈ powers,
      0 < coefficient ∧ coefficient < goldilocksP)
    (holds : RowHolds assignment (recompositionCheck parent children powers).row) :
    Recomposes assignment parent children powers := by
  apply builderLinearRow_sound canonical one parent (children.zip powers)
    (canonicalTerms_zip_of_powers powerCanonical)
  simpa [recompositionCheck, Instruction.row, builderLinearRow, negateTerms]
    using holds

theorem dataRecomposition_sound
    {assignment : Nat → Nat} (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) {parent : List Nat}
    {children : List (List Nat)} {powers : List Nat}
    (powerCanonical : ∀ coefficient ∈ powers,
      0 < coefficient ∧ coefficient < goldilocksP)
    (satisfies :
      Satisfies (CheckedProgram.rows (dataRecomposition parent children powers)) assignment) :
    AllRecompose assignment parent children powers := by
  intro lane laneLt
  apply recompositionCheck_sound canonical one powerCanonical
  apply instruction_holds satisfies
  apply List.mem_map.mpr
  exact ⟨lane, List.mem_range.mpr laneLt, rfl⟩

theorem equality_member_sound
    {instructions : List Instruction} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (CheckedProgram.rows instructions) assignment)
    {lhs rhs : Nat} (member : equalityCheck lhs rhs ∈ instructions) :
    assignment lhs = assignment rhs :=
  equalityCheck_sound canonical one (instruction_holds satisfies member)

theorem zero_member_sound
    {instructions : List Instruction} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (CheckedProgram.rows instructions) assignment)
    {column : Nat} (member : zeroCheck column ∈ instructions) :
    assignment column = 0 :=
  zeroCheck_sound canonical one (instruction_holds satisfies member)

theorem centeredUnitInstructions_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) (column output : Nat)
    (satisfies : Satisfies
      (CheckedProgram.rows (centeredUnitInstructions column output)) assignment) :
    CenteredUnit (assignment column) := by
  have definitionRow : RowHolds assignment
      (Instruction.define ⟨output,
        .product [(column, 1), (0, 1)] [(column, 1)]⟩).row := by
    apply instruction_holds satisfies
    simp [centeredUnitInstructions]
  have outputEq := builderDefinition_sound canonical one
    ⟨output, .product [(column, 1), (0, 1)] [(column, 1)]⟩ (by trivial)
    definitionRow
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

theorem alphabetFrom_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    ∀ (columns : List Nat) (output : Nat),
      Satisfies (CheckedProgram.rows (alphabetFrom output columns)) assignment →
      ∀ column ∈ columns, CenteredUnit (assignment column) := by
  intro columns
  induction columns with
  | nil => simp
  | cons head tail inductionHypothesis =>
      intro output satisfies column member
      simp only [List.mem_cons] at member
      rcases member with equal | member
      · subst column
        apply centeredUnitInstructions_sound prime canonical one head output
        intro row rowMember
        apply satisfies row
        simpa [alphabetFrom, CheckedProgram.rows] using
          List.mem_append_left
            (CheckedProgram.rows (alphabetFrom (output + 1) tail)) rowMember
      · apply inductionHypothesis (output + 1)
        · intro row rowMember
          apply satisfies row
          simpa [alphabetFrom, CheckedProgram.rows] using
            List.mem_append_right
              (CheckedProgram.rows (centeredUnitInstructions head output)) rowMember
        · exact member

theorem xRecomposition_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (powerCanonical : ∀ coefficient ∈
      radixPowers layout.radix layout.children.length,
      0 < coefficient ∧ coefficient < goldilocksP)
    (satisfies : Satisfies
      (CheckedProgram.rows (xRecompositionInstructions layout
        (radixPowers layout.radix layout.children.length))) assignment) :
    ∀ row column,
      row < layout.parent.xRows → column < activeColumns layout →
      Recomposes assignment (xColumn layout layout.parent row column)
        (layout.children.map fun child => xColumn layout child row column)
        (radixPowers layout.radix layout.children.length) := by
  intro row column rowLt columnLt
  apply recompositionCheck_sound canonical one powerCanonical
  apply instruction_holds satisfies
  apply List.mem_flatMap.mpr
  refine ⟨row, List.mem_range.mpr rowLt, ?_⟩
  apply List.mem_map.mpr
  exact ⟨column, List.mem_range.mpr columnLt, rfl⟩

theorem yRecomposition_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (powerCanonical : ∀ coefficient ∈
      radixPowers layout.radix layout.children.length,
      0 < coefficient ∧ coefficient < goldilocksP)
    (satisfies : Satisfies
      (CheckedProgram.rows (yRecompositionInstructions layout
        (radixPowers layout.radix layout.children.length))) assignment) :
    ∀ row lane,
      row < layout.parent.yRingCols.length →
      lane < (layout.parent.yRingCols.getD row []).length →
      Recomposes assignment
        ((layout.parent.yRingCols.getD row []).getD lane 0)
        (layout.children.map fun child =>
          (child.yRingCols.getD row []).getD lane 0)
        (radixPowers layout.radix layout.children.length) := by
  intro row lane rowLt laneLt
  apply recompositionCheck_sound canonical one powerCanonical
  apply instruction_holds satisfies
  apply List.mem_flatMap.mpr
  refine ⟨row, List.mem_range.mpr rowLt, ?_⟩
  apply List.mem_map.mpr
  exact ⟨lane, List.mem_range.mpr laneLt, rfl⟩

theorem shapeInstructions_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows (shapeInstructions layout)) assignment) :
    ∀ child ∈ layout.children,
      assignment layout.parent.commitment.dCol = assignment child.commitment.dCol ∧
      assignment layout.parent.commitment.kappaCol = assignment child.commitment.kappaCol ∧
      assignment layout.parent.xRowsCol = assignment child.xRowsCol ∧
      assignment layout.parent.xWidthCol = assignment child.xWidthCol ∧
      assignment layout.parent.mInCol = assignment child.mInCol := by
  intro child childMember
  have member (instruction : Instruction)
      (localMember : instruction ∈
        [equalityCheck layout.parent.commitment.dCol child.commitment.dCol,
         equalityCheck layout.parent.commitment.kappaCol child.commitment.kappaCol,
         equalityCheck layout.parent.xRowsCol child.xRowsCol,
         equalityCheck layout.parent.xWidthCol child.xWidthCol,
         equalityCheck layout.parent.mInCol child.mInCol]) :
      instruction ∈ shapeInstructions layout := by
    apply List.mem_flatMap.mpr
    exact ⟨child, childMember, localMember⟩
  refine ⟨
    equality_member_sound canonical one satisfies (member _ (by simp)),
    equality_member_sound canonical one satisfies (member _ (by simp)),
    equality_member_sound canonical one satisfies (member _ (by simp)),
    equality_member_sound canonical one satisfies (member _ (by simp)),
    equality_member_sound canonical one satisfies (member _ (by simp))⟩

theorem pairEqualityInstructions_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) (parent : List (Nat × Nat))
    (children : List (List (Nat × Nat)))
    (satisfies : Satisfies
      (CheckedProgram.rows (pairEqualityInstructions parent children)) assignment) :
    ∀ child ∈ children, EqualPairs assignment parent child := by
  intro child childMember pair pairMember
  have firstMember : equalityCheck pair.1.1 pair.2.1 ∈
      pairEqualityInstructions parent children := by
    apply List.mem_flatMap.mpr
    refine ⟨child, childMember, ?_⟩
    apply List.mem_flatMap.mpr
    exact ⟨pair, pairMember, by simp⟩
  have secondMember : equalityCheck pair.1.2 pair.2.2 ∈
      pairEqualityInstructions parent children := by
    apply List.mem_flatMap.mpr
    refine ⟨child, childMember, ?_⟩
    apply List.mem_flatMap.mpr
    exact ⟨pair, pairMember, by simp⟩
  exact ⟨equality_member_sound canonical one satisfies firstMember,
    equality_member_sound canonical one satisfies secondMember⟩

theorem inactiveInstructions_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows (inactiveInstructions layout)) assignment) :
    ∀ claim ∈ layout.parent :: layout.children,
      ∀ column ∈ unique (inactiveXColumns layout claim),
        assignment column = 0 := by
  intro claim claimMember column columnMember
  apply zero_member_sound canonical one satisfies
  apply List.mem_flatMap.mpr
  refine ⟨claim, claimMember, ?_⟩
  exact List.mem_map.mpr ⟨column, columnMember, rfl⟩

theorem ctInstructions_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows (ctInstructions layout)) assignment) :
    ∀ claim ∈ layout.parent :: layout.children,
      ∀ pair ∈ claim.ctCols.zip claim.yRingCols,
        assignment pair.1.1 = assignment (pair.2.getD 0 0) ∧
          assignment pair.1.2 = assignment (pair.2.getD 1 0) := by
  intro claim claimMember pair pairMember
  constructor <;> apply equality_member_sound canonical one satisfies <;>
    apply List.mem_flatMap.mpr <;>
    refine ⟨claim, claimMember, ?_⟩ <;>
    apply List.mem_flatMap.mpr <;>
    exact ⟨pair, pairMember, by simp⟩

theorem paddingInstructions_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows (paddingInstructions layout)) assignment) :
    ∀ claim ∈ layout.parent :: layout.children,
      ∀ row ∈ claim.yRingCols,
        ∀ column ∈ row.drop (layout.ringDimension * layout.extensionLimbs),
          assignment column = 0 := by
  intro claim claimMember row rowMember column columnMember
  apply zero_member_sound canonical one satisfies
  apply List.mem_flatMap.mpr
  refine ⟨claim, claimMember, ?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨row, rowMember, ?_⟩
  exact List.mem_map.mpr ⟨column, columnMember, rfl⟩

theorem foldDigestInstructions_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows (foldDigestInstructions layout)) assignment) :
    ∀ child ∈ layout.children,
      ∀ pair ∈ child.foldDigestCols.zip layout.parent.foldDigestCols,
        assignment pair.1 = assignment pair.2 := by
  intro child childMember pair pairMember
  apply equality_member_sound canonical one satisfies
  apply List.mem_flatMap.mpr
  refine ⟨child, childMember, ?_⟩
  exact List.mem_map.mpr ⟨pair, pairMember, rfl⟩

theorem equalityCheck_complete
    {assignment : Nat → Nat} (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) {lhs rhs : Nat}
    (equal : assignment lhs = assignment rhs) :
    RowHolds assignment (equalityCheck lhs rhs).row := by
  have rhsEq : assignment lhs = lcEval assignment [(rhs, 1)] := by
    simpa [lcEval, Nat.mod_eq_of_lt (canonical rhs)] using equal
  have row := builderLinearRow_complete one lhs [(rhs, 1)]
    (by simp [CanonicalTerms, goldilocksP]) rhsEq
  simpa [equalityCheck, Instruction.row, builderLinearRow, negateTerms,
    negCoeff, goldilocksP] using row

theorem zeroCheck_complete
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    {column : Nat} (zero : assignment column = 0) :
    RowHolds assignment (zeroCheck column).row := by
  have row := builderLinearRow_complete one column [] (by
    simp [CanonicalTerms]) (by simpa [lcEval] using zero)
  simpa [zeroCheck, Instruction.row, builderLinearRow, negateTerms] using row

theorem recompositionCheck_complete
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    {parent : Nat} {children powers : List Nat}
    (powerCanonical : ∀ coefficient ∈ powers,
      0 < coefficient ∧ coefficient < goldilocksP)
    (recomposes : Recomposes assignment parent children powers) :
    RowHolds assignment (recompositionCheck parent children powers).row := by
  have row := builderLinearRow_complete one parent (children.zip powers)
    (canonicalTerms_zip_of_powers powerCanonical) recomposes
  simpa [recompositionCheck, Instruction.row, builderLinearRow, negateTerms]
    using row

theorem dataRecomposition_complete
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    {parent : List Nat} {children : List (List Nat)} {powers : List Nat}
    (powerCanonical : ∀ coefficient ∈ powers,
      0 < coefficient ∧ coefficient < goldilocksP)
    (accepted : AllRecompose assignment parent children powers) :
    Satisfies (CheckedProgram.rows (dataRecomposition parent children powers))
      assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_map.mp instructionMember with ⟨lane, laneMember, rfl⟩
  apply recompositionCheck_complete one powerCanonical
  exact accepted lane (List.mem_range.mp laneMember)

theorem xRecomposition_complete
    {layout : Layout} {assignment : Nat → Nat} (one : assignment 0 = 1)
    (powerCanonical : ∀ coefficient ∈
      radixPowers layout.radix layout.children.length,
      0 < coefficient ∧ coefficient < goldilocksP)
    (accepted : ∀ row column,
      row < layout.parent.xRows → column < activeColumns layout →
      Recomposes assignment (xColumn layout layout.parent row column)
        (layout.children.map fun child => xColumn layout child row column)
        (radixPowers layout.radix layout.children.length)) :
    Satisfies (CheckedProgram.rows (xRecompositionInstructions layout
      (radixPowers layout.radix layout.children.length))) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨rowIndex, rowIndexMember, instructionMember⟩
  rcases List.mem_map.mp instructionMember with
    ⟨columnIndex, columnIndexMember, rfl⟩
  apply recompositionCheck_complete one powerCanonical
  exact accepted rowIndex columnIndex (List.mem_range.mp rowIndexMember)
    (List.mem_range.mp columnIndexMember)

theorem yRecomposition_complete
    {layout : Layout} {assignment : Nat → Nat} (one : assignment 0 = 1)
    (powerCanonical : ∀ coefficient ∈
      radixPowers layout.radix layout.children.length,
      0 < coefficient ∧ coefficient < goldilocksP)
    (accepted : ∀ row lane,
      row < layout.parent.yRingCols.length →
      lane < (layout.parent.yRingCols.getD row []).length →
      Recomposes assignment
        ((layout.parent.yRingCols.getD row []).getD lane 0)
        (layout.children.map fun child =>
          (child.yRingCols.getD row []).getD lane 0)
        (radixPowers layout.radix layout.children.length)) :
    Satisfies (CheckedProgram.rows (yRecompositionInstructions layout
      (radixPowers layout.radix layout.children.length))) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨rowIndex, rowIndexMember, instructionMember⟩
  rcases List.mem_map.mp instructionMember with
    ⟨lane, laneMember, rfl⟩
  apply recompositionCheck_complete one powerCanonical
  exact accepted rowIndex lane (List.mem_range.mp rowIndexMember)
    (List.mem_range.mp laneMember)

theorem centeredUnitCheck_complete
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (column output : Nat)
    (definitionHolds : Definition.Holds assignment
      ⟨output, .product [(column, 1), (0, 1)] [(column, 1)]⟩)
    (centered : CenteredUnit (assignment column)) :
    RowHolds assignment (centeredUnitCheckRow column output) := by
  rcases centered with zero | oneValue | minusOne
  · have outputZero : assignment output = 0 := by
      simpa [Definition.Holds, Rhs.eval, lcEval, one, zero] using
        definitionHolds
    change lcEval assignment [(output, 1)] *
      lcEval assignment [(column, 1), (0, goldilocksP - 1)] %
        goldilocksP = lcEval assignment []
    have outputLcZero : lcEval assignment [(output, 1)] = 0 := by
      simp [lcEval, outputZero]
    rw [outputLcZero]
    simp [lcEval]
  · change lcEval assignment [(output, 1)] *
      lcEval assignment [(column, 1), (0, goldilocksP - 1)] %
        goldilocksP = lcEval assignment []
    have factorZero :
        lcEval assignment [(column, 1), (0, goldilocksP - 1)] = 0 := by
      simp [lcEval, one, oneValue, goldilocksP]
    rw [factorZero]
    simp [lcEval]
  · have outputZero : assignment output = 0 := by
      simpa [Definition.Holds, Rhs.eval, lcEval, one, minusOne,
        goldilocksP] using definitionHolds
    change lcEval assignment [(output, 1)] *
      lcEval assignment [(column, 1), (0, goldilocksP - 1)] %
        goldilocksP = lcEval assignment []
    have outputLcZero : lcEval assignment [(output, 1)] = 0 := by
      simp [lcEval, outputZero]
    rw [outputLcZero]
    simp [lcEval]

theorem satisfies_append
    {left right : List Row} {assignment : Nat → Nat}
    (leftSatisfies : Satisfies left assignment)
    (rightSatisfies : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  rcases List.mem_append.mp member with member | member
  · exact leftSatisfies row member
  · exact rightSatisfies row member

theorem alphabetCheckRowsFrom_complete
    {assignment : Nat → Nat} (one : assignment 0 = 1) :
    ∀ (columns : List Nat) (output : Nat),
      (∀ column ∈ columns, CenteredUnit (assignment column)) →
      (∀ column ∈ columns, ∀ index,
        index < columns.length → columns.getD index 0 = column →
        Definition.Holds assignment
          ⟨output + index,
            .product [(column, 1), (0, 1)] [(column, 1)]⟩) →
      Satisfies (alphabetCheckRowsFrom output columns) assignment := by
  intro columns output centered definitions
  intro row rowMember
  induction columns generalizing output with
  | nil => simp [alphabetCheckRowsFrom] at rowMember
  | cons head tail inductionHypothesis =>
      simp only [alphabetCheckRowsFrom, List.mem_cons] at rowMember
      rcases rowMember with rfl | rowMember
      · apply centeredUnitCheck_complete one head output
        · simpa using definitions head (by simp) 0 (by simp) rfl
        · exact centered head (by simp)
      · apply inductionHypothesis (output + 1)
        · intro column member
          exact centered column (by simp [member])
        · intro column member index indexLt getD
          have originalIndexLt : index + 1 < (head :: tail).length := by
            simp only [List.length_cons]
            omega
          have originalGetD : (head :: tail).getD (index + 1) 0 = column := by
            simpa using getD
          have := definitions column (by simp [member]) (index + 1)
            originalIndexLt originalGetD
          have outputIndex : output + 1 + index = output + (index + 1) := by
            omega
          rw [outputIndex]
          exact this
        · exact rowMember

theorem shapeInstructions_complete
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : ∀ child ∈ layout.children,
      assignment layout.parent.commitment.dCol = assignment child.commitment.dCol ∧
      assignment layout.parent.commitment.kappaCol = assignment child.commitment.kappaCol ∧
      assignment layout.parent.xRowsCol = assignment child.xRowsCol ∧
      assignment layout.parent.xWidthCol = assignment child.xWidthCol ∧
      assignment layout.parent.mInCol = assignment child.mInCol) :
    Satisfies (CheckedProgram.rows (shapeInstructions layout)) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨child, childMember, instructionMember⟩
  have facts := accepted child childMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at instructionMember
  rcases instructionMember with rfl | rfl | rfl | rfl | rfl
  · exact equalityCheck_complete canonical one facts.1
  · exact equalityCheck_complete canonical one facts.2.1
  · exact equalityCheck_complete canonical one facts.2.2.1
  · exact equalityCheck_complete canonical one facts.2.2.2.1
  · exact equalityCheck_complete canonical one facts.2.2.2.2

theorem pairEqualityInstructions_complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) (parent : List (Nat × Nat))
    (children : List (List (Nat × Nat)))
    (accepted : ∀ child ∈ children, EqualPairs assignment parent child) :
    Satisfies (CheckedProgram.rows (pairEqualityInstructions parent children))
      assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨child, childMember, instructionMember⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨pair, pairMember, instructionMember⟩
  have equal := accepted child childMember pair pairMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at instructionMember
  rcases instructionMember with rfl | rfl
  · exact equalityCheck_complete canonical one equal.1
  · exact equalityCheck_complete canonical one equal.2

theorem inactiveInstructions_complete
    {layout : Layout} {assignment : Nat → Nat} (one : assignment 0 = 1)
    (accepted : ∀ claim ∈ layout.parent :: layout.children,
      ∀ column ∈ unique (inactiveXColumns layout claim),
        assignment column = 0) :
    Satisfies (CheckedProgram.rows (inactiveInstructions layout)) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨claim, claimMember, instructionMember⟩
  rcases List.mem_map.mp instructionMember with ⟨column, columnMember, rfl⟩
  exact zeroCheck_complete one (accepted claim claimMember column columnMember)

theorem ctInstructions_complete
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : ∀ claim ∈ layout.parent :: layout.children,
      ∀ pair ∈ claim.ctCols.zip claim.yRingCols,
        assignment pair.1.1 = assignment (pair.2.getD 0 0) ∧
          assignment pair.1.2 = assignment (pair.2.getD 1 0)) :
    Satisfies (CheckedProgram.rows (ctInstructions layout)) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨claim, claimMember, instructionMember⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨pair, pairMember, instructionMember⟩
  have equal := accepted claim claimMember pair pairMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at instructionMember
  rcases instructionMember with rfl | rfl
  · exact equalityCheck_complete canonical one equal.1
  · exact equalityCheck_complete canonical one equal.2

theorem paddingInstructions_complete
    {layout : Layout} {assignment : Nat → Nat} (one : assignment 0 = 1)
    (accepted : ∀ claim ∈ layout.parent :: layout.children,
      ∀ row ∈ claim.yRingCols,
        ∀ column ∈ row.drop (layout.ringDimension * layout.extensionLimbs),
          assignment column = 0) :
    Satisfies (CheckedProgram.rows (paddingInstructions layout)) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨claim, claimMember, instructionMember⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨sourceRow, sourceRowMember, instructionMember⟩
  rcases List.mem_map.mp instructionMember with ⟨column, columnMember, rfl⟩
  exact zeroCheck_complete one
    (accepted claim claimMember sourceRow sourceRowMember column columnMember)

theorem foldDigestInstructions_complete
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : ∀ child ∈ layout.children,
      ∀ pair ∈ child.foldDigestCols.zip layout.parent.foldDigestCols,
        assignment pair.1 = assignment pair.2) :
    Satisfies (CheckedProgram.rows (foldDigestInstructions layout)) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨child, childMember, instructionMember⟩
  rcases List.mem_map.mp instructionMember with ⟨pair, pairMember, rfl⟩
  exact equalityCheck_complete canonical one
    (accepted child childMember pair pairMember)

theorem alphabetFrom_definition_mem :
    ∀ (columns : List Nat) (output index : Nat),
      index < columns.length →
      Instruction.define
          ⟨output + index,
            .product [(columns.getD index 0, 1), (0, 1)]
              [(columns.getD index 0, 1)]⟩ ∈
        alphabetFrom output columns := by
  intro columns
  induction columns with
  | nil => simp
  | cons head tail inductionHypothesis =>
      intro output index indexLt
      cases index with
      | zero => simp [alphabetFrom, centeredUnitInstructions]
      | succ index =>
          have tailLt : index < tail.length := by
            simp only [List.length_cons] at indexLt
            omega
          have member := inductionHypothesis (output + 1) index tailLt
          have outputIndex : output + (index + 1) = output + 1 + index := by
            omega
          simpa [alphabetFrom, outputIndex] using
            List.mem_append_right (centeredUnitInstructions head output) member

theorem checkRows_complete_noAdv
    {layout : Layout} (valid : ShapeValid layout)
    (parentNoAdv : layout.parent.adv = none)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : Accepted layout assignment)
    (alphabetDefinitions :
      let columns := layout.children.flatMap (activeXColumns layout)
      ∀ column ∈ columns, ∀ index,
        index < columns.length → columns.getD index 0 = column →
        Definition.Holds assignment
          ⟨layout.firstAllocatedColumn + index,
            .product [(column, 1), (0, 1)] [(column, 1)]⟩) :
    Satisfies (checkRows layout) assignment := by
  let powers := radixPowers layout.radix layout.children.length
  let columns := layout.children.flatMap (activeXColumns layout)
  have dataSatisfies := dataRecomposition_complete one
    valid.powersCanonical accepted.commitment
  have dataAdvSatisfies : Satisfies
      (CheckedProgram.rows
        (dataRecomposition layout.parent.commitment.dataCols
          (layout.children.map (·.commitment.dataCols)) powers ++
         advInstructions layout.parent.adv (layout.children.map (·.adv)) powers))
      assignment := by
    simpa [powers, parentNoAdv, advInstructions, CheckedProgram.rows] using
      dataSatisfies
  have xSatisfies := xRecomposition_complete one valid.powersCanonical accepted.x
  have ySatisfies := yRecomposition_complete one valid.powersCanonical accepted.y
  have shapeSatisfies := shapeInstructions_complete canonical one accepted.shape
  have rSatisfies := pairEqualityInstructions_complete canonical one
    layout.parent.rCols (layout.children.map (·.rCols)) (by
      intro childPairs childPairsMember
      rcases List.mem_map.mp childPairsMember with ⟨child, childMember, rfl⟩
      exact accepted.sameR child childMember)
  have sColSatisfies := pairEqualityInstructions_complete canonical one
    layout.parent.sColCols (layout.children.map (·.sColCols)) (by
      intro childPairs childPairsMember
      rcases List.mem_map.mp childPairsMember with ⟨child, childMember, rfl⟩
      exact accepted.sameSCol child childMember)
  have inactiveSatisfies := inactiveInstructions_complete one accepted.inactiveZero
  have alphabetSatisfies : Satisfies (alphabetCheckRows layout) assignment := by
    apply alphabetCheckRowsFrom_complete one columns layout.firstAllocatedColumn
    · intro column columnMember
      rcases List.mem_flatMap.mp columnMember with
        ⟨child, childMember, columnMember⟩
      exact accepted.childCentered child childMember column columnMember
    · simpa [columns] using alphabetDefinitions
  have ctSatisfies := ctInstructions_complete canonical one accepted.ct
  have paddingSatisfies := paddingInstructions_complete one accepted.paddingZero
  have foldSatisfies := foldDigestInstructions_complete canonical one
    accepted.foldDigest
  simpa [checkRows, powers] using
    satisfies_append dataAdvSatisfies
      (satisfies_append xSatisfies
        (satisfies_append ySatisfies
          (satisfies_append shapeSatisfies
            (satisfies_append rSatisfies
              (satisfies_append sColSatisfies
                (satisfies_append inactiveSatisfies
                  (satisfies_append alphabetSatisfies
                    (satisfies_append ctSatisfies
                      (satisfies_append paddingSatisfies foldSatisfies)))))))))

private theorem satisfies_instruction_append_left
    {left right : List Instruction} {assignment : Nat → Nat}
    (satisfies : Satisfies
      (CheckedProgram.rows (left ++ right)) assignment) :
    Satisfies (CheckedProgram.rows left) assignment := by
  intro row member
  apply satisfies row
  simpa [CheckedProgram.rows] using
    List.mem_append_left (CheckedProgram.rows right) member

/-- Generic radix-two strict-PiDEC compiler soundness for a no-adv layout.
The Nebula/adv branch has its own compiler rows and will be discharged by the
same equation lemmas when an adv-enabled artifact is selected. -/
theorem compiler_sound_noAdv
    (prime : EuclidPrime goldilocksP)
    {layout : Layout} (valid : ShapeValid layout)
    (radixTwo : layout.radix = 2)
    (parentNoAdv : layout.parent.adv = none)
    (childrenNoAdv : ∀ child ∈ layout.children, child.adv = none)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    Accepted layout assignment := by
  let powers := radixPowers layout.radix layout.children.length
  let data := dataRecomposition layout.parent.commitment.dataCols
    (layout.children.map (·.commitment.dataCols)) powers
  let adv := advInstructions layout.parent.adv (layout.children.map (·.adv)) powers
  have group0 : data ++ adv ∈ groups layout := by
    simp [groups, data, adv, powers]
  have group1 : xRecompositionInstructions layout powers ∈ groups layout := by
    simp [groups, powers]
  have group2 : yRecompositionInstructions layout powers ∈ groups layout := by
    simp [groups, powers]
  have group3 : shapeInstructions layout ∈ groups layout := by
    simp [groups]
  have group4 : pairEqualityInstructions layout.parent.rCols
      (layout.children.map (·.rCols)) ∈ groups layout := by simp [groups]
  have group5 : pairEqualityInstructions layout.parent.sColCols
      (layout.children.map (·.sColCols)) ∈ groups layout := by simp [groups]
  have group6 : inactiveInstructions layout ∈ groups layout := by simp [groups]
  have group7 : alphabetInstructions layout ∈ groups layout := by simp [groups]
  have group8 : ctInstructions layout ∈ groups layout := by simp [groups]
  have group9 : paddingInstructions layout ∈ groups layout := by simp [groups]
  have group10 : foldDigestInstructions layout ∈ groups layout := by simp [groups]
  have satisfies0 := group_satisfies satisfies group0
  have satisfies1 := group_satisfies satisfies group1
  have satisfies2 := group_satisfies satisfies group2
  have satisfies3 := group_satisfies satisfies group3
  have satisfies4 := group_satisfies satisfies group4
  have satisfies5 := group_satisfies satisfies group5
  have satisfies6 := group_satisfies satisfies group6
  have satisfies7 := group_satisfies satisfies group7
  have satisfies8 := group_satisfies satisfies group8
  have satisfies9 := group_satisfies satisfies group9
  have satisfies10 := group_satisfies satisfies group10
  refine {
    radixTwo := radixTwo
    commitment := dataRecomposition_sound canonical one valid.powersCanonical
      (satisfies_instruction_append_left satisfies0)
    adv := ?_
    x := xRecomposition_sound canonical one valid.powersCanonical satisfies1
    y := yRecomposition_sound canonical one valid.powersCanonical satisfies2
    shape := shapeInstructions_sound canonical one satisfies3
    sameR := ?_
    sameSCol := ?_
    inactiveZero := inactiveInstructions_sound canonical one satisfies6
    childCentered := ?_
    ct := ctInstructions_sound canonical one satisfies8
    paddingZero := paddingInstructions_sound canonical one satisfies9
    foldDigest := foldDigestInstructions_sound canonical one satisfies10
  }
  · simp [AdvAccepted, parentNoAdv]
    intro child childMember
    exact childrenNoAdv child childMember
  · intro child childMember
    apply pairEqualityInstructions_sound canonical one layout.parent.rCols
      (layout.children.map (·.rCols)) satisfies4 child.rCols
    exact List.mem_map.mpr ⟨child, childMember, rfl⟩
  · intro child childMember
    apply pairEqualityInstructions_sound canonical one layout.parent.sColCols
      (layout.children.map (·.sColCols)) satisfies5 child.sColCols
    exact List.mem_map.mpr ⟨child, childMember, rfl⟩
  · intro child childMember column columnMember
    apply alphabetFrom_sound prime canonical one
      (layout.children.flatMap (activeXColumns layout))
      layout.firstAllocatedColumn
      (by simpa [alphabetInstructions] using satisfies7) column
    apply List.mem_flatMap.mpr
    exact ⟨child, childMember, columnMember⟩

namespace Exact

open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec

theorem shapeValid : ShapeValid layout where
  ringPositive := by native_decide
  powersCanonical := by native_decide
  commitmentLengths := by native_decide
  xShapes := by native_decide
  activeXLengths := by native_decide
  yShapes := by native_decide
  rShapes := by native_decide
  sColShapes := by native_decide
  ctShapes := by native_decide
  foldDigestShapes := by native_decide

/-- Artifact-level CIR-SOUND for strict PiDEC, shared by recursive and
terminal NIFS: every canonical satisfying assignment meets the independent
strict verifier predicate. -/
theorem sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies FPrimeFullHistoryPiDec.rows assignment) :
    Accepted layout assignment := by
  apply compiler_sound_noAdv prime shapeValid (by native_decide)
    (by native_decide) (by native_decide) canonical one
  simpa [PiDecStrictCompiler.rows, FPrimeFullHistoryPiDec.rows,
    instructions_match_compiler] using satisfies

private theorem mapped_sound
    (prime : EuclidPrime goldilocksP)
    (columnMap : List Nat)
    (mapsOne : Relabel.column columnMap 0 = 0)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (FPrimeFullHistoryPiDec.rows.map (Relabel.row columnMap)) assignment) :
    Accepted layout (Relabel.assignment columnMap assignment) := by
  apply sound prime (Relabel.canonical canonical)
    (Relabel.constantOne mapsOne one)
  exact (Relabel.satisfies_mapped_iff
    FPrimeFullHistoryPiDec.rows columnMap assignment).mp satisfies

/-- Exact strict-PiDEC meaning of the recursive NIFS row range. -/
theorem recursive_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies FPrimeFullHistoryPiDec.recursiveRows assignment) :
    Accepted layout
      (Relabel.assignment FPrimeFullHistoryPiDec.recursiveColumnMap assignment) := by
  exact mapped_sound prime _ FPrimeFullHistoryPiDec.recursive_map_one
    canonical one satisfies

/-- Exact strict-PiDEC meaning of the terminal-fold NIFS row range. -/
theorem terminal_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies FPrimeFullHistoryPiDec.terminalRows assignment) :
    Accepted layout
      (Relabel.assignment FPrimeFullHistoryPiDec.terminalColumnMap assignment) := by
  exact mapped_sound prime _ FPrimeFullHistoryPiDec.terminal_map_one
    canonical one satisfies

/-- Exact strict-PiDEC meaning of the direct terminal-CE row range. -/
theorem terminal_ce_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies FPrimeFullHistoryPiDec.terminalCeRows assignment) :
    Accepted layout
      (Relabel.assignment FPrimeFullHistoryPiDec.terminalCeColumnMap assignment) := by
  exact mapped_sound prime _ FPrimeFullHistoryPiDec.terminal_ce_map_one
    canonical one satisfies

/-- Honest strict-PiDEC compiler execution.  The witness carries a source
state and the actual interpreter result, never definition-row satisfaction. -/
structure ExecutionWitness (assignment : Nat → Nat) where
  source : Nat → Nat
  sourceCanonical : ∀ column, source column < goldilocksP
  sourceOne : source 0 = 1
  executed : interpret source FPrimeFullHistoryPiDec.instructions = assignment
  accepted : Accepted layout assignment

def nativeCheck (assignment : Nat → Nat) : Bool :=
  ((definitions FPrimeFullHistoryPiDec.instructions).all fun definition =>
    decide (Definition.Holds assignment definition)) &&
    PiDecStrictCompiler.check layout assignment

theorem nativeCheck_eq_true_iff (assignment : Nat → Nat) :
    nativeCheck assignment = true ↔
      (∀ definition ∈
          definitions FPrimeFullHistoryPiDec.instructions,
        Definition.Holds assignment definition) ∧
      Accepted layout assignment := by
  simp [nativeCheck, List.all_eq_true,
    Bool.and_eq_true, decide_eq_true_eq,
    PiDecStrictCompiler.check_eq_true_iff]

/-- Same-assignment CIR-COMPLETE for a native strict-PiDEC witness. -/
theorem native_complete
    {assignment : Nat → Nat}
    (witness : ExecutionWitness assignment) :
    Satisfies rows assignment := by
  have executed :
      run witness.source (definitions FPrimeFullHistoryPiDec.instructions) =
        assignment := by
    simpa [interpret] using witness.executed
  have canonical : ∀ column, assignment column < goldilocksP := by
    rw [← executed]
    exact run_canonical witness.sourceCanonical
  have preserves := run_preserves_known definitions_wellFormed witness.source
  have one : assignment 0 = 1 := by
    rw [← executed]
    exact (preserves 0 (by native_decide)).trans witness.sourceOne
  have definitionsHold := run_definitions_hold definitions_wellFormed
    witness.source
  rw [executed] at definitionsHold
  have alphabetDefinitions :
      let columns := layout.children.flatMap (activeXColumns layout)
      ∀ column ∈ columns, ∀ index,
        index < columns.length → columns.getD index 0 = column →
        Definition.Holds assignment
          ⟨layout.firstAllocatedColumn + index,
            .product [(column, 1), (0, 1)] [(column, 1)]⟩ := by
    dsimp only
    intro column columnMember index indexLt getD
    let columns := layout.children.flatMap (activeXColumns layout)
    let definition : Definition :=
      ⟨layout.firstAllocatedColumn + index,
        .product [(column, 1), (0, 1)] [(column, 1)]⟩
    have alphabetMember : Instruction.define definition ∈
        alphabetInstructions layout := by
      have generated := alphabetFrom_definition_mem columns
        layout.firstAllocatedColumn index indexLt
      have getD' : columns.getD index 0 = column := by
        simpa [columns] using getD
      rw [getD'] at generated
      simpa [alphabetInstructions, columns, definition] using generated
    have groupMember : alphabetInstructions layout ∈ groups layout := by
      simp [groups]
    have compilerMember : Instruction.define definition ∈
        PiDecStrictCompiler.instructions layout :=
      List.mem_flatten.mpr
        ⟨alphabetInstructions layout, groupMember, alphabetMember⟩
    have artifactMember : Instruction.define definition ∈
        FPrimeFullHistoryPiDec.instructions := by
      rw [instructions_match_compiler]
      exact compilerMember
    exact definitionsHold definition (by
      apply List.mem_filterMap.mpr
      exact ⟨Instruction.define definition, artifactMember, rfl⟩)
  have checkRowsSatisfy : Satisfies (checkRows layout) assignment :=
    checkRows_complete_noAdv shapeValid (by native_decide) canonical one
      witness.accepted alphabetDefinitions
  apply CheckedProgram.assignmentHolds_complete definitions_canonical
    canonical one
  exact {
    definitions := definitionsHold
    checks := by
      rw [checks_match_compiler]
      exact checkRowsSatisfy
  }

private theorem mapped_native_complete
    (columnMap : List Nat)
    {assignment : Nat → Nat}
    (witness : ExecutionWitness
      (Relabel.assignment columnMap assignment)) :
    Satisfies (rows.map (Relabel.row columnMap)) assignment := by
  apply (Relabel.satisfies_mapped_iff rows columnMap assignment).mpr
  exact native_complete witness

theorem recursive_native_complete
    {assignment : Nat → Nat}
    (witness : ExecutionWitness
      (Relabel.assignment recursiveColumnMap assignment)) :
    Satisfies recursiveRows assignment :=
  mapped_native_complete recursiveColumnMap witness

theorem terminal_native_complete
    {assignment : Nat → Nat}
    (witness : ExecutionWitness
      (Relabel.assignment terminalColumnMap assignment)) :
    Satisfies terminalRows assignment :=
  mapped_native_complete terminalColumnMap witness

/-- Artifact-level CIR-COMPLETE for strict PiDEC. The checked-program
interpreter fills every centered-alphabet product column; semantic acceptance
of the resulting decoded claims makes every retained verifier assertion true. -/
theorem complete
    {state : Nat → Nat}
    (stateCanonical : ∀ column, state column < goldilocksP)
    (one : state 0 = 1)
    (accepted : Accepted layout
      (interpret state FPrimeFullHistoryPiDec.instructions)) :
    Satisfies FPrimeFullHistoryPiDec.rows
      (interpret state FPrimeFullHistoryPiDec.instructions) := by
  let final := interpret state FPrimeFullHistoryPiDec.instructions
  have zeroInput : 0 ∈ FPrimeFullHistoryPiDec.inputColumns := by native_decide
  have preserves := run_preserves_known
    FPrimeFullHistoryPiDec.definitions_wellFormed state
  have finalCanonical : ∀ column, final column < goldilocksP := by
    exact run_canonical stateCanonical
  have finalOne : final 0 = 1 := by
    exact (preserves 0 zeroInput).trans one
  have definitionsHold := run_definitions_hold
    FPrimeFullHistoryPiDec.definitions_wellFormed state
  have alphabetDefinitions :
      let columns := layout.children.flatMap (activeXColumns layout)
      ∀ column ∈ columns, ∀ index,
        index < columns.length → columns.getD index 0 = column →
        Definition.Holds final
          ⟨layout.firstAllocatedColumn + index,
            .product [(column, 1), (0, 1)] [(column, 1)]⟩ := by
    dsimp only
    intro column columnMember index indexLt getD
    let columns := layout.children.flatMap (activeXColumns layout)
    let definition : Definition :=
      ⟨layout.firstAllocatedColumn + index,
        .product [(column, 1), (0, 1)] [(column, 1)]⟩
    have alphabetMember : Instruction.define definition ∈
        alphabetInstructions layout := by
      have generated := alphabetFrom_definition_mem columns
        layout.firstAllocatedColumn index indexLt
      have getD' : columns.getD index 0 = column := by
        simpa [columns] using getD
      rw [getD'] at generated
      simpa [alphabetInstructions, columns, definition] using generated
    have groupMember : alphabetInstructions layout ∈ groups layout := by
      simp [groups]
    have compilerMember : Instruction.define definition ∈
        PiDecStrictCompiler.instructions layout :=
      List.mem_flatten.mpr
        ⟨alphabetInstructions layout, groupMember, alphabetMember⟩
    have artifactMember : Instruction.define definition ∈
        FPrimeFullHistoryPiDec.instructions := by
      rw [instructions_match_compiler]
      exact compilerMember
    apply definitionsHold definition
    apply List.mem_filterMap.mpr
    exact ⟨Instruction.define definition, artifactMember, rfl⟩
  have checkRowsSatisfy : Satisfies (checkRows layout) final :=
    checkRows_complete_noAdv shapeValid (by native_decide) finalCanonical
      finalOne accepted alphabetDefinitions
  have checksHold : ChecksHold state FPrimeFullHistoryPiDec.instructions := by
    change Satisfies (checks FPrimeFullHistoryPiDec.instructions) final
    rw [checks_match_compiler]
    exact checkRowsSatisfy
  exact CheckedProgram.complete FPrimeFullHistoryPiDec.definitions_wellFormed
    FPrimeFullHistoryPiDec.definitions_canonical stateCanonical zeroInput one
    checksHold

end Exact

end Nightstream.Implementation.R1CS.PiDecStrictSound
