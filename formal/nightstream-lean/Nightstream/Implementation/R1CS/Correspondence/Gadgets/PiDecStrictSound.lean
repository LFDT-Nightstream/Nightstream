import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictCompiler
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalCeSound
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: exact row soundness for the strict PiDEC compiler.

Owns: derivation of the independent `PiDecStrictCompiler.Accepted` predicate
from the compiler's exact rows, a canonical Goldilocks assignment, and the
verifier-owned host-shape certificate.

Does not own: a V2 wire layout, typed paper-PiDEC refinement, Ajtai binding,
Rust row equality, or a cryptographic security claim.

Emits constraints: no; it proves the meaning of existing rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.PiDecStrictSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.Program

theorem instruction_holds
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment)
    {instruction : Instruction} (member : instruction ∈ instructions layout) :
    RowHolds assignment instruction.row := by
  exact satisfies instruction.row
    (List.mem_map.mpr ⟨instruction, member, rfl⟩)

theorem group_instruction_holds
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment)
    {group : List Instruction} (groupMember : group ∈ groups layout)
    {instruction : Instruction} (member : instruction ∈ group) :
    RowHolds assignment instruction.row := by
  apply instruction_holds satisfies
  exact List.mem_flatten.mpr ⟨group, groupMember, member⟩

theorem group_satisfied
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment)
    {group : List Instruction} (groupMember : group ∈ groups layout) :
    Satisfies (CheckedProgram.rows group) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨instruction, inGroup, rfl⟩
  exact satisfies instruction.row (List.mem_map.mpr
    ⟨instruction,
      List.mem_flatten.mpr ⟨group, groupMember, inGroup⟩, rfl⟩)

theorem equalityCheck_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (left right : Nat)
    (holds : RowHolds assignment (equalityCheck left right).row) :
    assignment left = assignment right := by
  have result := builderLinearRow_sound canonical one left [(right, 1)]
    (by simp [CanonicalTerms, goldilocksP])
    (by
      simpa [equalityCheck, Instruction.row, builderLinearRow, negateTerms,
        negCoeff] using holds)
  simpa [lcEval, Nat.mod_eq_of_lt (canonical right)] using result

theorem zeroCheck_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (column : Nat)
    (holds : RowHolds assignment (zeroCheck column).row) :
    assignment column = 0 := by
  simpa [zeroCheck, Instruction.row, RowHolds, lcEval, one,
    Nat.mod_eq_of_lt (canonical column)] using holds

theorem recompositionCheck_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parent : Nat) (children powers : List Nat)
    (powersCanonical : ∀ coefficient ∈ powers,
      0 < coefficient ∧ coefficient < goldilocksP)
    (holds : RowHolds assignment
      (recompositionCheck parent children powers).row) :
    Recomposes assignment parent children powers := by
  have termsCanonical : CanonicalTerms (children.zip powers) := by
    intro term member
    exact powersCanonical term.2 (List.of_mem_zip member).2
  have result := builderLinearRow_sound canonical one parent
    (children.zip powers) termsCanonical
    (by
      simpa [recompositionCheck, Instruction.row, builderLinearRow,
        negateTerms, Function.comp_def] using holds)
  exact result

theorem satisfies_of_instructions_subset
    {small large : List Instruction} {assignment : Nat → Nat}
    (subset : ∀ instruction ∈ small, instruction ∈ large)
    (satisfies : Satisfies (CheckedProgram.rows large) assignment) :
    Satisfies (CheckedProgram.rows small) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨instruction, inSmall, rfl⟩
  exact satisfies instruction.row
    (List.mem_map.mpr ⟨instruction, subset instruction inSmall, rfl⟩)

theorem dataRecomposition_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parent : List Nat) (children : List (List Nat)) (powers : List Nat)
    (powersCanonical : ∀ coefficient ∈ powers,
      0 < coefficient ∧ coefficient < goldilocksP)
    (satisfies : Satisfies
      (CheckedProgram.rows (dataRecomposition parent children powers))
      assignment) :
    AllRecompose assignment parent children powers := by
  intro lane laneLt
  apply recompositionCheck_sound canonical one _ _ _ powersCanonical
  apply Nightstream.Implementation.R1CS.TerminalCeSound.instruction_holds
    satisfies
  exact List.mem_map.mpr
    ⟨lane, List.mem_range.mpr laneLt, rfl⟩

theorem advCoordinateInstructions_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (powers : List Nat)
    (powersCanonical : ∀ coefficient ∈ powers,
      0 < coefficient ∧ coefficient < goldilocksP)
    (parent : CommitmentLayout) (children : List CommitmentLayout)
    (satisfies : Satisfies
      (CheckedProgram.rows
        (advCoordinateInstructions parent children powers)) assignment) :
    CommitmentAccepted assignment powers parent children := by
  constructor
  · intro child childMember
    constructor
    · apply equalityCheck_sound canonical one
      apply Nightstream.Implementation.R1CS.TerminalCeSound.instruction_holds
        satisfies
      apply List.mem_append_left
      exact List.mem_flatMap.mpr
        ⟨child, childMember, by simp⟩
    · apply equalityCheck_sound canonical one
      apply Nightstream.Implementation.R1CS.TerminalCeSound.instruction_holds
        satisfies
      apply List.mem_append_left
      exact List.mem_flatMap.mpr
        ⟨child, childMember, by simp⟩
  · apply dataRecomposition_sound canonical one _ _ powers powersCanonical
    apply satisfies_of_instructions_subset _ satisfies
    intro instruction member
    exact List.mem_append_right _ member

private theorem filterMap_map_some {α : Type} (values : List α) :
    (values.map some).filterMap id = values := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis => simp

private theorem all_some_has_concrete
    (children : List ClaimLayout)
    (present : ∀ child ∈ children, ∃ adv, child.adv = some adv) :
    ∃ concrete : List AdvLayout,
      children.map (·.adv) = concrete.map some := by
  induction children with
  | nil => exact ⟨[], rfl⟩
  | cons child tail inductionHypothesis =>
      rcases present child (by simp) with ⟨adv, childAdv⟩
      rcases inductionHypothesis (fun current member =>
        present current (by simp [member])) with ⟨concrete, tailAdv⟩
      exact ⟨adv :: concrete, by simp [childAdv, tailAdv]⟩

theorem advInstructions_sound
    {layout : Layout} {assignment : Nat → Nat}
    (valid : ShapeValid layout)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows
        (advInstructions layout.parent.adv
          (layout.children.map (·.adv))
          (radixPowers layout.radix layout.children.length))) assignment) :
    AdvAccepted assignment
      (radixPowers layout.radix layout.children.length)
      layout.parent.adv (layout.children.map (·.adv)) := by
  cases parentAdv : layout.parent.adv with
  | none =>
      have absent := valid.advPresence
      rw [parentAdv] at absent
      simp only [AdvAccepted]
      intro childAdv member
      rcases List.mem_map.mp member with ⟨child, childMember, rfl⟩
      exact absent child childMember
  | some parent =>
      have present : ∀ child ∈ layout.children,
          ∃ adv, child.adv = some adv := by
        simpa only [parentAdv] using valid.advPresence
      rcases all_some_has_concrete layout.children present with
        ⟨concrete, concreteEq⟩
      refine ⟨concrete, concreteEq, ?_, ?_, ?_⟩
      all_goals
        apply advCoordinateInstructions_sound canonical one _
          valid.powersCanonical
        apply satisfies_of_instructions_subset _ satisfies
        intro instruction member
        rw [parentAdv]
        rw [concreteEq]
        simp only [advInstructions, filterMap_map_some]
      · exact List.mem_append_left _ (List.mem_append_left _ member)
      · exact List.mem_append_left _ (List.mem_append_right _ member)
      · exact List.mem_append_right _ member

theorem xRecompositionInstructions_sound
    {layout : Layout} {assignment : Nat → Nat}
    (valid : ShapeValid layout)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows
        (xRecompositionInstructions layout
          (radixPowers layout.radix layout.children.length))) assignment) :
    ∀ row column,
      row < layout.parent.xRows → column < activeColumns layout →
      Recomposes assignment (xColumn layout layout.parent row column)
        (layout.children.map fun child => xColumn layout child row column)
        (radixPowers layout.radix layout.children.length) := by
  intro row column rowLt columnLt
  apply recompositionCheck_sound canonical one _ _ _ valid.powersCanonical
  apply Nightstream.Implementation.R1CS.TerminalCeSound.instruction_holds
    satisfies
  exact List.mem_flatMap.mpr
    ⟨row, List.mem_range.mpr rowLt,
      List.mem_map.mpr ⟨column, List.mem_range.mpr columnLt, rfl⟩⟩

theorem yRecompositionInstructions_sound
    {layout : Layout} {assignment : Nat → Nat}
    (valid : ShapeValid layout)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows
        (yRecompositionInstructions layout
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
  apply recompositionCheck_sound canonical one _ _ _ valid.powersCanonical
  apply Nightstream.Implementation.R1CS.TerminalCeSound.instruction_holds
    satisfies
  exact List.mem_flatMap.mpr
    ⟨row, List.mem_range.mpr rowLt,
      List.mem_map.mpr ⟨lane, List.mem_range.mpr laneLt, rfl⟩⟩

theorem shapeInstructions_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows (shapeInstructions layout)) assignment) :
    ∀ child ∈ layout.children,
      assignment layout.parent.commitment.dCol =
          assignment child.commitment.dCol ∧
      assignment layout.parent.commitment.kappaCol =
          assignment child.commitment.kappaCol ∧
      assignment layout.parent.xRowsCol = assignment child.xRowsCol ∧
      assignment layout.parent.xWidthCol = assignment child.xWidthCol ∧
      assignment layout.parent.mInCol = assignment child.mInCol := by
  intro child childMember
  have holds (instruction : Instruction)
      (member : instruction ∈
        [equalityCheck layout.parent.commitment.dCol child.commitment.dCol,
         equalityCheck layout.parent.commitment.kappaCol child.commitment.kappaCol,
         equalityCheck layout.parent.xRowsCol child.xRowsCol,
         equalityCheck layout.parent.xWidthCol child.xWidthCol,
         equalityCheck layout.parent.mInCol child.mInCol]) :
      RowHolds assignment instruction.row := by
    apply Nightstream.Implementation.R1CS.TerminalCeSound.instruction_holds
      satisfies
    exact List.mem_flatMap.mpr ⟨child, childMember, member⟩
  exact ⟨
    equalityCheck_sound canonical one _ _ (holds _ (by simp)),
    equalityCheck_sound canonical one _ _ (holds _ (by simp)),
    equalityCheck_sound canonical one _ _ (holds _ (by simp)),
    equalityCheck_sound canonical one _ _ (holds _ (by simp)),
    equalityCheck_sound canonical one _ _ (holds _ (by simp))⟩

theorem pairEqualityInstructions_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parent : List (Nat × Nat)) (children : List (List (Nat × Nat)))
    (satisfies : Satisfies
      (CheckedProgram.rows (pairEqualityInstructions parent children))
      assignment) :
    ∀ child ∈ children, EqualPairs assignment parent child := by
  intro child childMember pair pairMember
  have holds (instruction : Instruction)
      (member : instruction ∈
        [equalityCheck pair.1.1 pair.2.1,
         equalityCheck pair.1.2 pair.2.2]) :
      RowHolds assignment instruction.row := by
    apply Nightstream.Implementation.R1CS.TerminalCeSound.instruction_holds
      satisfies
    exact List.mem_flatMap.mpr
      ⟨child, childMember, List.mem_flatMap.mpr
        ⟨pair, pairMember, member⟩⟩
  exact ⟨
    equalityCheck_sound canonical one _ _ (holds _ (by simp)),
    equalityCheck_sound canonical one _ _ (holds _ (by simp))⟩

private theorem alphabetFrom_eq_normInstructionsFrom :
    ∀ output columns,
      alphabetFrom output columns =
        Nightstream.Implementation.R1CS.TerminalCeCompiler.normInstructionsFrom
          output columns
  | _, [] => rfl
  | output, column :: tail => by
      simp only [alphabetFrom,
        Nightstream.Implementation.R1CS.TerminalCeCompiler.normInstructionsFrom]
      rw [alphabetFrom_eq_normInstructionsFrom (output + 1) tail]
      rfl

theorem alphabetInstructions_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows (alphabetInstructions layout)) assignment) :
    ∀ child ∈ layout.children,
      ∀ column ∈ activeXColumns layout child,
        CenteredUnit (assignment column) := by
  have allColumns :
      ∀ column ∈ layout.children.flatMap (activeXColumns layout),
        Nightstream.Implementation.R1CS.TerminalCeCompiler.CenteredUnit
          (assignment column) := by
    have normalized := satisfies
    rw [alphabetInstructions, alphabetFrom_eq_normInstructionsFrom] at normalized
    apply Nightstream.Implementation.R1CS.TerminalCeSound.normInstructionsFrom_sound
      Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime
      canonical one
    exact normalized
  intro child childMember column columnMember
  have semantic := allColumns column
    (List.mem_flatMap.mpr ⟨child, childMember, columnMember⟩)
  simpa [CenteredUnit,
    Nightstream.Implementation.R1CS.TerminalCeCompiler.CenteredUnit]
    using semantic

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
  have holds (instruction : Instruction)
      (member : instruction ∈
        [equalityCheck pair.1.1 (pair.2.getD 0 0),
         equalityCheck pair.1.2 (pair.2.getD 1 0)]) :
      RowHolds assignment instruction.row := by
    apply Nightstream.Implementation.R1CS.TerminalCeSound.instruction_holds
      satisfies
    exact List.mem_flatMap.mpr
      ⟨claim, claimMember, List.mem_flatMap.mpr
        ⟨pair, pairMember, member⟩⟩
  exact ⟨
    equalityCheck_sound canonical one _ _ (holds _ (by simp)),
    equalityCheck_sound canonical one _ _ (holds _ (by simp))⟩

theorem paddingInstructions_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows (paddingInstructions layout)) assignment) :
    ∀ claim ∈ layout.parent :: layout.children,
      ∀ row ∈ claim.yRingCols,
      ∀ column ∈ row.drop
        (layout.ringDimension * layout.extensionLimbs),
        assignment column = 0 := by
  intro claim claimMember row rowMember column columnMember
  apply zeroCheck_sound canonical one
  apply Nightstream.Implementation.R1CS.TerminalCeSound.instruction_holds
    satisfies
  exact List.mem_flatMap.mpr
    ⟨claim, claimMember, List.mem_flatMap.mpr
      ⟨row, rowMember, List.mem_map.mpr ⟨column, columnMember, rfl⟩⟩⟩

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
  apply equalityCheck_sound canonical one
  apply Nightstream.Implementation.R1CS.TerminalCeSound.instruction_holds
    satisfies
  exact List.mem_flatMap.mpr
    ⟨child, childMember, List.mem_map.mpr ⟨pair, pairMember, rfl⟩⟩

/-- Exact strict-PiDEC row satisfaction implies the independently defined
claim equations. The conclusion is not a field of the shape certificate. -/
theorem rows_sound
    {layout : Layout} {assignment : Nat → Nat}
    (valid : ShapeValid layout)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    Accepted layout assignment := by
  let powers := radixPowers layout.radix layout.children.length
  let commitmentProgram :=
    dataRecomposition layout.parent.commitment.dataCols
      (layout.children.map (·.commitment.dataCols)) powers
  let advProgram :=
    advInstructions layout.parent.adv (layout.children.map (·.adv)) powers
  let firstGroup := commitmentProgram ++ advProgram
  have firstSatisfied : Satisfies
      (CheckedProgram.rows firstGroup) assignment := by
    apply group_satisfied satisfies
    simp [groups, powers, commitmentProgram, advProgram, firstGroup]
  have commitmentSatisfied : Satisfies
      (CheckedProgram.rows commitmentProgram) assignment := by
    apply satisfies_of_instructions_subset _ firstSatisfied
    intro instruction member
    exact List.mem_append_left _ member
  have advSatisfied : Satisfies
      (CheckedProgram.rows advProgram) assignment := by
    apply satisfies_of_instructions_subset _ firstSatisfied
    intro instruction member
    exact List.mem_append_right _ member
  have xSatisfied : Satisfies
      (CheckedProgram.rows (xRecompositionInstructions layout powers))
      assignment := by
    apply group_satisfied satisfies
    simp [groups, powers]
  have ySatisfied : Satisfies
      (CheckedProgram.rows (yRecompositionInstructions layout powers))
      assignment := by
    apply group_satisfied satisfies
    simp [groups, powers]
  have shapeSatisfied : Satisfies
      (CheckedProgram.rows (shapeInstructions layout)) assignment := by
    apply group_satisfied satisfies
    simp [groups]
  have pairsSatisfied : Satisfies
      (CheckedProgram.rows
        (pairEqualityInstructions layout.parent.rCols
          (layout.children.map (·.rCols)))) assignment := by
    apply group_satisfied satisfies
    simp [groups]
  have alphabetSatisfied : Satisfies
      (CheckedProgram.rows (alphabetInstructions layout)) assignment := by
    apply group_satisfied satisfies
    simp [groups]
  have ctSatisfied : Satisfies
      (CheckedProgram.rows (ctInstructions layout)) assignment := by
    apply group_satisfied satisfies
    simp [groups]
  have paddingSatisfied : Satisfies
      (CheckedProgram.rows (paddingInstructions layout)) assignment := by
    apply group_satisfied satisfies
    simp [groups]
  have digestSatisfied : Satisfies
      (CheckedProgram.rows (foldDigestInstructions layout)) assignment := by
    apply group_satisfied satisfies
    simp [groups]
  refine {
    radixTwo := valid.radixTwo
    commitment := ?_
    adv := ?_
    x := ?_
    y := ?_
    shape := ?_
    sameR := ?_
    childCentered := ?_
    ct := ?_
    paddingZero := ?_
    foldDigest := ?_ }
  · exact dataRecomposition_sound canonical one _ _ powers
      valid.powersCanonical commitmentSatisfied
  · exact advInstructions_sound valid canonical one advSatisfied
  · exact xRecompositionInstructions_sound valid canonical one xSatisfied
  · exact yRecompositionInstructions_sound valid canonical one ySatisfied
  · exact shapeInstructions_sound canonical one shapeSatisfied
  · intro child childMember
    exact pairEqualityInstructions_sound canonical one _ _ pairsSatisfied
      child.rCols (List.mem_map.mpr ⟨child, childMember, rfl⟩)
  · exact alphabetInstructions_sound canonical one alphabetSatisfied
  · exact ctInstructions_sound canonical one ctSatisfied
  · exact paddingInstructions_sound canonical one paddingSatisfied
  · exact foldDigestInstructions_sound canonical one digestSatisfied

/-- The executable strict-PiDEC checker accepts every canonical satisfying
row assignment with a valid verifier-owned shape. -/
theorem check_eq_true_of_rows
    {layout : Layout} {assignment : Nat → Nat}
    (valid : ShapeValid layout)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    check layout assignment = true :=
  (check_eq_true_iff layout assignment).2
    (rows_sound valid canonical one satisfies)

end Nightstream.Implementation.R1CS.PiDecStrictSound
