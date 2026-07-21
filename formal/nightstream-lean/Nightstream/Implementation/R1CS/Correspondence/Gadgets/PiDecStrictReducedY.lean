import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictSound

/-!
Contract: model-level elimination of strict-`PiDEC` y-recomposition rows whose
coordinates are already forced to zero by the retained padding family.

This file owns a reduced y-plus-padding subcompiler and proves it equivalent
to the current full y-plus-padding family.  It does not claim that a Rust
emitter or a generated artifact uses the reduced schedule; physical deletion
still requires a separate Rust/artifact conformance theorem.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictReducedY

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.PiDecStrictCompiler

/-- Number of semantic field limbs in one decoded extension-ring value. -/
def semanticYWidth (layout : Layout) : Nat :=
  layout.ringDimension * layout.extensionLimbs

/-- Recompose only the semantic prefix of every parent y row.  `min` makes
the compiler total on malformed host layouts; `ShapeValid` relates child and
parent row lengths for the equivalence theorem below. -/
def reducedYRecompositionInstructions
    (layout : Layout) (powers : List Nat) : List Instruction :=
  (List.range layout.parent.yRingCols.length).flatMap fun row =>
    let parent := layout.parent.yRingCols.getD row []
    (List.range (min (semanticYWidth layout) parent.length)).map fun lane =>
      recompositionCheck (parent.getD lane 0)
        (layout.children.map fun child =>
          (child.yRingCols.getD row []).getD lane 0) powers

/-- Semantic endpoint of the reduced y-recomposition family. -/
def ReducedYAccepted
    (layout : Layout) (assignment : Nat → Nat) : Prop :=
  ∀ row lane,
    row < layout.parent.yRingCols.length →
    lane < min (semanticYWidth layout)
      (layout.parent.yRingCols.getD row []).length →
    Recomposes assignment
      ((layout.parent.yRingCols.getD row []).getD lane 0)
      (layout.children.map fun child =>
        (child.yRingCols.getD row []).getD lane 0)
      (radixPowers layout.radix layout.children.length)

/-- Semantic endpoint of the retained padding-zero family. -/
def PaddingAccepted
    (layout : Layout) (assignment : Nat → Nat) : Prop :=
  ∀ claim ∈ layout.parent :: layout.children,
    ∀ row ∈ claim.yRingCols,
      ∀ column ∈ row.drop (semanticYWidth layout),
        assignment column = 0

/-- The reduced physical family keeps all padding rows. -/
def reducedFamilyInstructions
    (layout : Layout) (powers : List Nat) : List Instruction :=
  reducedYRecompositionInstructions layout powers ++ paddingInstructions layout

/-- The corresponding family in the current strict compiler. -/
def fullFamilyInstructions
    (layout : Layout) (powers : List Nat) : List Instruction :=
  yRecompositionInstructions layout powers ++ paddingInstructions layout

def reducedFamilyRows (layout : Layout) : List Row :=
  CheckedProgram.rows (reducedFamilyInstructions layout
    (radixPowers layout.radix layout.children.length))

def fullFamilyRows (layout : Layout) : List Row :=
  CheckedProgram.rows (fullFamilyInstructions layout
    (radixPowers layout.radix layout.children.length))

private theorem getD_mem_of_lt {Carrier : Type}
    (values : List Carrier) (default : Carrier) (index : Nat)
    (indexLt : index < values.length) : values.getD index default ∈ values := by
  rw [← List.getElem_eq_getD (l := values) (i := index) default]
  exact List.getElem_mem indexLt

private theorem getD_mem_drop_of_le {Carrier : Type}
    (values : List Carrier) (default : Carrier) (start index : Nat)
    (startLe : start ≤ index) (indexLt : index < values.length) :
    values.getD index default ∈ values.drop start := by
  induction start generalizing values index with
  | zero =>
      simpa using getD_mem_of_lt values default index indexLt
  | succ start inductionHypothesis =>
      cases values with
      | nil => simp at indexLt
      | cons head tail =>
          cases index with
          | zero => omega
          | succ index =>
              have tailLt : index < tail.length := by
                simpa using indexLt
              simp only [List.drop_succ_cons]
              simpa using inductionHypothesis tail index (by omega) tailLt

private theorem lcEval_map_zip_eq_zero
    {Carrier : Type} (assignment : Nat → Nat)
    (values : List Carrier) (column : Carrier → Nat) (powers : List Nat)
    (zero : ∀ value ∈ values, assignment (column value) = 0) :
    lcEval assignment ((values.map column).zip powers) = 0 := by
  rw [lcEval_eq_raw_mod]
  have rawZero :
      rawLcEval assignment ((values.map column).zip powers) = 0 := by
    induction values generalizing powers with
    | nil => simp [rawLcEval]
    | cons head tail inductionHypothesis =>
        cases powers with
        | nil => simp [rawLcEval]
        | cons power powers =>
            simp only [List.map_cons, List.zip_cons_cons, rawLcEval]
            rw [zero head (by simp)]
            simp only [Nat.mul_zero, Nat.zero_add]
            apply inductionHypothesis
            intro value valueMember
            exact zero value (by simp [valueMember])
  rw [rawZero]
  simp

/-- Full y acceptance trivially restricts to the semantic prefix. -/
theorem reducedY_of_fullY
    {layout : Layout} {assignment : Nat → Nat}
    (full : ∀ row lane,
      row < layout.parent.yRingCols.length →
      lane < (layout.parent.yRingCols.getD row []).length →
      Recomposes assignment
        ((layout.parent.yRingCols.getD row []).getD lane 0)
        (layout.children.map fun child =>
          (child.yRingCols.getD row []).getD lane 0)
        (radixPowers layout.radix layout.children.length)) :
    ReducedYAccepted layout assignment := by
  intro row lane rowLt laneLt
  apply full row lane rowLt
  exact Nat.lt_of_lt_of_le laneLt (Nat.min_le_right _ _)

/-- Padding zeroes determine every omitted y-recomposition equation. -/
theorem fullY_of_reducedY_and_padding
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat → Nat}
    (reduced : ReducedYAccepted layout assignment)
    (padding : PaddingAccepted layout assignment) :
    ∀ row lane,
      row < layout.parent.yRingCols.length →
      lane < (layout.parent.yRingCols.getD row []).length →
      Recomposes assignment
        ((layout.parent.yRingCols.getD row []).getD lane 0)
        (layout.children.map fun child =>
          (child.yRingCols.getD row []).getD lane 0)
        (radixPowers layout.radix layout.children.length) := by
  intro row lane rowLt laneLt
  by_cases active : lane < semanticYWidth layout
  · apply reduced row lane rowLt
    exact Nat.lt_min.mpr ⟨active, laneLt⟩
  · have widthLe : semanticYWidth layout ≤ lane := Nat.le_of_not_gt active
    have parentRowMember :
        layout.parent.yRingCols.getD row [] ∈
          layout.parent.yRingCols :=
      getD_mem_of_lt layout.parent.yRingCols [] row rowLt
    have parentColumnMember :
        (layout.parent.yRingCols.getD row []).getD lane 0 ∈
          (layout.parent.yRingCols.getD row []).drop
            (semanticYWidth layout) :=
      getD_mem_drop_of_le _ _ _ _ widthLe laneLt
    have parentZero :
        assignment ((layout.parent.yRingCols.getD row []).getD lane 0) = 0 :=
      padding layout.parent (by simp) _ parentRowMember _ parentColumnMember
    have childZero : ∀ child ∈ layout.children,
        assignment ((child.yRingCols.getD row []).getD lane 0) = 0 := by
      intro child childMember
      have childRowsLength := (valid.yShapes child childMember).1
      have childRowLt : row < child.yRingCols.length := by omega
      have childRowMember : child.yRingCols.getD row [] ∈ child.yRingCols :=
        getD_mem_of_lt child.yRingCols [] row childRowLt
      have childRowLength := (valid.yShapes child childMember).2 row rowLt
      have childLaneLt : lane < (child.yRingCols.getD row []).length := by
        omega
      have childColumnMember :
          (child.yRingCols.getD row []).getD lane 0 ∈
            (child.yRingCols.getD row []).drop (semanticYWidth layout) :=
        getD_mem_drop_of_le _ _ _ _ widthLe childLaneLt
      exact padding child (by simp [childMember]) _ childRowMember _
        childColumnMember
    unfold Recomposes
    rw [parentZero]
    exact (lcEval_map_zip_eq_zero assignment layout.children
      (fun child => (child.yRingCols.getD row []).getD lane 0)
      (radixPowers layout.radix layout.children.length) childZero).symm

/-- All strict acceptance fields except the old full-width y field, with the
reduced semantic-prefix field in its place. -/
structure ReducedAccepted (layout : Layout) (assignment : Nat → Nat) : Prop where
  radixTwo : layout.radix = 2
  commitment : AllRecompose assignment layout.parent.commitment.dataCols
    (layout.children.map (·.commitment.dataCols))
    (radixPowers layout.radix layout.children.length)
  adv : AdvAccepted assignment
    (radixPowers layout.radix layout.children.length)
    layout.parent.adv (layout.children.map (·.adv))
  x : ∀ row column,
    row < layout.parent.xRows → column < activeColumns layout →
    Recomposes assignment (xColumn layout layout.parent row column)
      (layout.children.map fun child => xColumn layout child row column)
      (radixPowers layout.radix layout.children.length)
  y : ReducedYAccepted layout assignment
  shape : ∀ child ∈ layout.children,
    assignment layout.parent.commitment.dCol = assignment child.commitment.dCol ∧
    assignment layout.parent.commitment.kappaCol = assignment child.commitment.kappaCol ∧
    assignment layout.parent.xRowsCol = assignment child.xRowsCol ∧
    assignment layout.parent.xWidthCol = assignment child.xWidthCol ∧
    assignment layout.parent.mInCol = assignment child.mInCol
  sameR : ∀ child ∈ layout.children,
    EqualPairs assignment layout.parent.rCols child.rCols
  sameSCol : ∀ child ∈ layout.children,
    EqualPairs assignment layout.parent.sColCols child.sColCols
  inactiveZero : ∀ claim ∈ layout.parent :: layout.children,
    ∀ column ∈ unique (inactiveXColumns layout claim), assignment column = 0
  childCentered : ∀ child ∈ layout.children,
    ∀ column ∈ activeXColumns layout child, CenteredUnit (assignment column)
  ct : ∀ claim ∈ layout.parent :: layout.children,
    ∀ pair ∈ claim.ctCols.zip claim.yRingCols,
      assignment pair.1.1 = assignment (pair.2.getD 0 0) ∧
        assignment pair.1.2 = assignment (pair.2.getD 1 0)
  paddingZero : PaddingAccepted layout assignment
  foldDigest : ∀ child ∈ layout.children,
    ∀ pair ∈ child.foldDigestCols.zip layout.parent.foldDigestCols,
      assignment pair.1 = assignment pair.2

theorem ReducedAccepted.ofFull
    {layout : Layout} {assignment : Nat → Nat}
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    ReducedAccepted layout assignment where
  radixTwo := accepted.radixTwo
  commitment := accepted.commitment
  adv := accepted.adv
  x := accepted.x
  y := reducedY_of_fullY accepted.y
  shape := accepted.shape
  sameR := accepted.sameR
  sameSCol := accepted.sameSCol
  inactiveZero := accepted.inactiveZero
  childCentered := accepted.childCentered
  ct := accepted.ct
  paddingZero := accepted.paddingZero
  foldDigest := accepted.foldDigest

theorem ReducedAccepted.toFull
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat → Nat} (accepted : ReducedAccepted layout assignment) :
    PiDecStrictCompiler.Accepted layout assignment where
  radixTwo := accepted.radixTwo
  commitment := accepted.commitment
  adv := accepted.adv
  x := accepted.x
  y := fullY_of_reducedY_and_padding valid accepted.y accepted.paddingZero
  shape := accepted.shape
  sameR := accepted.sameR
  sameSCol := accepted.sameSCol
  inactiveZero := accepted.inactiveZero
  childCentered := accepted.childCentered
  ct := accepted.ct
  paddingZero := accepted.paddingZero
  foldDigest := accepted.foldDigest

/-- Model-level semantic equivalence of current and reduced strict acceptance. -/
theorem reducedAccepted_iff_full
    {layout : Layout} (valid : ShapeValid layout) {assignment : Nat → Nat} :
    ReducedAccepted layout assignment ↔
      PiDecStrictCompiler.Accepted layout assignment :=
  ⟨ReducedAccepted.toFull valid, ReducedAccepted.ofFull⟩

theorem reducedYRecomposition_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (powerCanonical : ∀ coefficient ∈
      radixPowers layout.radix layout.children.length,
      0 < coefficient ∧ coefficient < goldilocksP)
    (satisfies : Satisfies
      (CheckedProgram.rows (reducedYRecompositionInstructions layout
        (radixPowers layout.radix layout.children.length))) assignment) :
    ReducedYAccepted layout assignment := by
  intro row lane rowLt laneLt
  apply PiDecStrictSound.recompositionCheck_sound canonical one powerCanonical
  apply PiDecStrictSound.instruction_holds satisfies
  apply List.mem_flatMap.mpr
  refine ⟨row, List.mem_range.mpr rowLt, ?_⟩
  apply List.mem_map.mpr
  exact ⟨lane, List.mem_range.mpr laneLt, rfl⟩

theorem reducedYRecomposition_complete
    {layout : Layout} {assignment : Nat → Nat} (one : assignment 0 = 1)
    (powerCanonical : ∀ coefficient ∈
      radixPowers layout.radix layout.children.length,
      0 < coefficient ∧ coefficient < goldilocksP)
    (accepted : ReducedYAccepted layout assignment) :
    Satisfies (CheckedProgram.rows (reducedYRecompositionInstructions layout
      (radixPowers layout.radix layout.children.length))) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_flatMap.mp instructionMember with
    ⟨rowIndex, rowIndexMember, instructionMember⟩
  rcases List.mem_map.mp instructionMember with
    ⟨lane, laneMember, rfl⟩
  apply PiDecStrictSound.recompositionCheck_complete one powerCanonical
  exact accepted rowIndex lane (List.mem_range.mp rowIndexMember)
    (List.mem_range.mp laneMember)

private theorem satisfies_append_left
    {left right : List Instruction} {assignment : Nat → Nat}
    (satisfies : Satisfies (CheckedProgram.rows (left ++ right)) assignment) :
    Satisfies (CheckedProgram.rows left) assignment := by
  intro row rowMember
  apply satisfies row
  simpa [CheckedProgram.rows] using
    List.mem_append_left (CheckedProgram.rows right) rowMember

private theorem satisfies_append_right
    {left right : List Instruction} {assignment : Nat → Nat}
    (satisfies : Satisfies (CheckedProgram.rows (left ++ right)) assignment) :
    Satisfies (CheckedProgram.rows right) assignment := by
  intro row rowMember
  apply satisfies row
  simpa [CheckedProgram.rows] using
    List.mem_append_right (CheckedProgram.rows left) rowMember

/-- Model-level soundness of the reduced y-plus-padding row family. -/
theorem reducedFamily_sound
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (reducedFamilyRows layout) assignment) :
    (∀ row lane,
      row < layout.parent.yRingCols.length →
      lane < (layout.parent.yRingCols.getD row []).length →
      Recomposes assignment
        ((layout.parent.yRingCols.getD row []).getD lane 0)
        (layout.children.map fun child =>
          (child.yRingCols.getD row []).getD lane 0)
        (radixPowers layout.radix layout.children.length)) ∧
      PaddingAccepted layout assignment := by
  have reducedSatisfies := satisfies_append_left
    (left := reducedYRecompositionInstructions layout
      (radixPowers layout.radix layout.children.length))
    (right := paddingInstructions layout) satisfies
  have paddingSatisfies := satisfies_append_right
    (left := reducedYRecompositionInstructions layout
      (radixPowers layout.radix layout.children.length))
    (right := paddingInstructions layout) satisfies
  have reduced := reducedYRecomposition_sound canonical one
    valid.powersCanonical reducedSatisfies
  have padding := PiDecStrictSound.paddingInstructions_sound canonical one
    paddingSatisfies
  exact ⟨fullY_of_reducedY_and_padding valid reduced padding, padding⟩

/-- Model-level completeness of the reduced y-plus-padding row family. -/
theorem reducedFamily_complete
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (full : ∀ row lane,
      row < layout.parent.yRingCols.length →
      lane < (layout.parent.yRingCols.getD row []).length →
      Recomposes assignment
        ((layout.parent.yRingCols.getD row []).getD lane 0)
        (layout.children.map fun child =>
          (child.yRingCols.getD row []).getD lane 0)
        (radixPowers layout.radix layout.children.length))
    (padding : PaddingAccepted layout assignment) :
    Satisfies (reducedFamilyRows layout) assignment := by
  have ySatisfies := reducedYRecomposition_complete one valid.powersCanonical
    (reducedY_of_fullY full)
  have paddingSatisfies :=
    PiDecStrictSound.paddingInstructions_complete one padding
  simpa [reducedFamilyRows, reducedFamilyInstructions, CheckedProgram.rows]
    using PiDecStrictSound.satisfies_append ySatisfies paddingSatisfies

private theorem fullFamily_sound
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (fullFamilyRows layout) assignment) :
    (∀ row lane,
      row < layout.parent.yRingCols.length →
      lane < (layout.parent.yRingCols.getD row []).length →
      Recomposes assignment
        ((layout.parent.yRingCols.getD row []).getD lane 0)
        (layout.children.map fun child =>
          (child.yRingCols.getD row []).getD lane 0)
        (radixPowers layout.radix layout.children.length)) ∧
      PaddingAccepted layout assignment := by
  have ySatisfies := satisfies_append_left
    (left := yRecompositionInstructions layout
      (radixPowers layout.radix layout.children.length))
    (right := paddingInstructions layout) satisfies
  have paddingSatisfies := satisfies_append_right
    (left := yRecompositionInstructions layout
      (radixPowers layout.radix layout.children.length))
    (right := paddingInstructions layout) satisfies
  exact ⟨PiDecStrictSound.yRecomposition_sound canonical one
      valid.powersCanonical ySatisfies,
    PiDecStrictSound.paddingInstructions_sound canonical one paddingSatisfies⟩

private theorem fullFamily_complete
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (full : ∀ row lane,
      row < layout.parent.yRingCols.length →
      lane < (layout.parent.yRingCols.getD row []).length →
      Recomposes assignment
        ((layout.parent.yRingCols.getD row []).getD lane 0)
        (layout.children.map fun child =>
          (child.yRingCols.getD row []).getD lane 0)
        (radixPowers layout.radix layout.children.length))
    (padding : PaddingAccepted layout assignment) :
    Satisfies (fullFamilyRows layout) assignment := by
  have ySatisfies := PiDecStrictSound.yRecomposition_complete one
    valid.powersCanonical full
  have paddingSatisfies :=
    PiDecStrictSound.paddingInstructions_complete one padding
  simpa [fullFamilyRows, fullFamilyInstructions, CheckedProgram.rows]
    using PiDecStrictSound.satisfies_append ySatisfies paddingSatisfies

/-- Kernel-level row equivalence: after retaining padding zeroes, deleting
all non-semantic y recomposition rows neither adds nor removes assignments. -/
theorem reducedFamily_satisfies_iff_fullFamily
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    Satisfies (reducedFamilyRows layout) assignment ↔
      Satisfies (fullFamilyRows layout) assignment := by
  constructor
  · intro reducedSatisfies
    rcases reducedFamily_sound valid canonical one reducedSatisfies with
      ⟨full, padding⟩
    exact fullFamily_complete valid one full padding
  · intro fullSatisfies
    rcases fullFamily_sound valid canonical one fullSatisfies with
      ⟨full, padding⟩
    exact reducedFamily_complete valid one full padding

/-! ## Exact generic row saving -/

/-- Explicit fixed-width shape needed to turn the generic prefix reduction
into a closed row-count formula. -/
structure UniformParentYWidth (layout : Layout) (width : Nat) : Prop where
  rowWidth : ∀ row, row < layout.parent.yRingCols.length →
    (layout.parent.yRingCols.getD row []).length = width
  semanticFits : semanticYWidth layout ≤ width

private theorem length_flatMap_range_constant
    {Carrier : Type} (count width : Nat) (body : Nat → List Carrier)
    (bodyLength : ∀ index, index < count → (body index).length = width) :
    ((List.range count).flatMap body).length = count * width := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append, List.length_append]
      simp only [List.flatMap_singleton]
      rw [inductionHypothesis (fun index indexLt =>
        bodyLength index (by omega))]
      rw [bodyLength count (by omega)]
      simp [Nat.succ_mul]

theorem fullYRecompositionInstruction_count
    {layout : Layout} {width : Nat}
    (shape : UniformParentYWidth layout width) :
    (yRecompositionInstructions layout
      (radixPowers layout.radix layout.children.length)).length =
        layout.parent.yRingCols.length * width := by
  apply length_flatMap_range_constant
  intro row rowLt
  simp only [List.length_map, List.length_range]
  exact shape.rowWidth row rowLt

theorem reducedYRecompositionInstruction_count
    {layout : Layout} {width : Nat}
    (shape : UniformParentYWidth layout width) :
    (reducedYRecompositionInstructions layout
      (radixPowers layout.radix layout.children.length)).length =
        layout.parent.yRingCols.length * semanticYWidth layout := by
  apply length_flatMap_range_constant
  intro row rowLt
  simp only [List.length_map, List.length_range]
  rw [shape.rowWidth row rowLt, Nat.min_eq_left shape.semanticFits]

/-- Exact number of rows omitted from the y-plus-padding family.  Padding
rows cancel because the reduced family retains them unchanged. -/
theorem fullFamily_row_count
    {layout : Layout} {width : Nat}
    (shape : UniformParentYWidth layout width) :
    (fullFamilyRows layout).length =
      (reducedFamilyRows layout).length +
        layout.parent.yRingCols.length *
          (width - semanticYWidth layout) := by
  simp only [fullFamilyRows, reducedFamilyRows, fullFamilyInstructions,
    reducedFamilyInstructions, CheckedProgram.rows, List.length_map,
    List.length_append]
  rw [fullYRecompositionInstruction_count shape,
    reducedYRecompositionInstruction_count shape]
  have semanticFits := shape.semanticFits
  have widthDecomposition :
      width = semanticYWidth layout + (width - semanticYWidth layout) := by
    omega
  have productDecomposition :
      layout.parent.yRingCols.length * width =
        layout.parent.yRingCols.length * semanticYWidth layout +
          layout.parent.yRingCols.length *
            (width - semanticYWidth layout) := by
    calc
      layout.parent.yRingCols.length * width =
          layout.parent.yRingCols.length *
            (semanticYWidth layout + (width - semanticYWidth layout)) :=
        congrArg (fun value => layout.parent.yRingCols.length * value)
          widthDecomposition
      _ = _ := Nat.mul_add _ _ _
  rw [productDecomposition]
  omega

end Nightstream.Implementation.R1CS.PiDecStrictReducedY
