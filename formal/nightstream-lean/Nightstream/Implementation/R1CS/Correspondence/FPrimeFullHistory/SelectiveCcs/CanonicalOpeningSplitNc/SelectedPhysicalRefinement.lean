import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningSplitNc

/-!
Contract: define the selected physical canonical-opening layout and refine its
Lean-owned rows to the 21-row canonicality relation.

Assurance tier: model-level.

Owns: contiguous 61-column opening frames, injective physical placement,
relocation of the 21 canonical equations, exact row support, Split-NC coverage,
and composition from block×lane NC residual rows to canonical Goldilocks
openings.

Does not own: generated layouts, Rust measurements, verifier soundness events,
or proof that an enclosing relation activates these rows.

Emits constraints: 21 equations per distinct opening. Each frame contains
41 digit columns followed by 20 retained-borrow columns.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement

set_option maxRecDepth 262144

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
open Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-- Physical width of one memoized opening. -/
def openingWidth : Nat :=
  digitCount + chunkBorrowCount

theorem openingWidth_eq :
    openingWidth = 61 := by
  rfl

/-- Offset of one owned coordinate in the contiguous opening region. -/
def coordinateOffset
    {openingCount : Nat} :
    Coordinate openingCount → Nat
  | .digit opening index =>
      opening.val * openingWidth + index.val
  | .borrow opening index =>
      opening.val * openingWidth + digitCount + index.val

theorem coordinateOffset_lt
    {openingCount : Nat}
    (coordinate : Coordinate openingCount) :
    coordinateOffset coordinate < openingCount * openingWidth := by
  cases coordinate with
  | digit opening index =>
      have openingLt := opening.isLt
      have indexLt := index.isLt
      change index.val < 41 at indexLt
      change
        opening.val * 61 + index.val <
          openingCount * 61
      omega
  | borrow opening index =>
      have openingLt := opening.isLt
      have indexLt := index.isLt
      change index.val < 20 at indexLt
      change
        opening.val * 61 + 41 + index.val <
          openingCount * 61
      omega

theorem coordinateOffset_injective
    {openingCount : Nat} :
    Function.Injective
      (coordinateOffset (openingCount := openingCount)) := by
  intro left right equal
  cases left with
  | digit leftOpening leftIndex =>
      cases right with
      | digit rightOpening rightIndex =>
          have leftOpeningLt := leftOpening.isLt
          have rightOpeningLt := rightOpening.isLt
          have leftIndexLt := leftIndex.isLt
          have rightIndexLt := rightIndex.isLt
          change leftIndex.val < 41 at leftIndexLt
          change rightIndex.val < 41 at rightIndexLt
          change
            leftOpening.val * 61 + leftIndex.val =
              rightOpening.val * 61 + rightIndex.val at equal
          have openingsEqual :
              leftOpening.val = rightOpening.val := by
            omega
          have indicesEqual :
              leftIndex.val = rightIndex.val := by
            omega
          cases Fin.ext openingsEqual
          cases Fin.ext indicesEqual
          rfl
      | borrow rightOpening rightIndex =>
          have leftOpeningLt := leftOpening.isLt
          have rightOpeningLt := rightOpening.isLt
          have leftIndexLt := leftIndex.isLt
          have rightIndexLt := rightIndex.isLt
          change leftIndex.val < 41 at leftIndexLt
          change rightIndex.val < 20 at rightIndexLt
          change
            leftOpening.val * 61 + leftIndex.val =
              rightOpening.val * 61 + 41 + rightIndex.val at equal
          omega
  | borrow leftOpening leftIndex =>
      cases right with
      | digit rightOpening rightIndex =>
          have leftOpeningLt := leftOpening.isLt
          have rightOpeningLt := rightOpening.isLt
          have leftIndexLt := leftIndex.isLt
          have rightIndexLt := rightIndex.isLt
          change leftIndex.val < 20 at leftIndexLt
          change rightIndex.val < 41 at rightIndexLt
          change
            leftOpening.val * 61 + 41 + leftIndex.val =
              rightOpening.val * 61 + rightIndex.val at equal
          omega
      | borrow rightOpening rightIndex =>
          have leftOpeningLt := leftOpening.isLt
          have rightOpeningLt := rightOpening.isLt
          have leftIndexLt := leftIndex.isLt
          have rightIndexLt := rightIndex.isLt
          change leftIndex.val < 20 at leftIndexLt
          change rightIndex.val < 20 at rightIndexLt
          change
            leftOpening.val * 61 + 41 + leftIndex.val =
              rightOpening.val * 61 + 41 + rightIndex.val at equal
          have openingsEqual :
              leftOpening.val = rightOpening.val := by
            omega
          have indicesEqual :
              leftIndex.val = rightIndex.val := by
            omega
          cases Fin.ext openingsEqual
          cases Fin.ext indicesEqual
          rfl

/-- Selected physical layout. Distinct openings occupy distinct 61-column
frames, so every Ajtai use may reference an existing opening without allocating
another frame. -/
def selectedLayout
    {columns openingCount : Nat}
    (base : Nat)
    (fits : base + openingCount * openingWidth ≤ columns) :
    ProductionLayout columns openingCount where
  column coordinate :=
    ⟨base + coordinateOffset coordinate, by
      have below := coordinateOffset_lt coordinate
      omega⟩
  injective := by
    intro left right equal
    apply coordinateOffset_injective
    have valuesEqual := congrArg Fin.val equal
    simp only at valuesEqual
    omega

@[simp] theorem selectedLayout_column_val
    {columns openingCount : Nat}
    (base : Nat)
    (fits : base + openingCount * openingWidth ≤ columns)
    (coordinate : Coordinate openingCount) :
    ((selectedLayout base fits).column coordinate).val =
      base + coordinateOffset coordinate := by
  rfl

@[simp] theorem selectedLayout_digit_column
    {columns openingCount : Nat}
    (base : Nat)
    (fits : base + openingCount * openingWidth ≤ columns)
    (opening : Fin openingCount) (index : Fin digitCount) :
    ((selectedLayout base fits).column (.digit opening index)).val =
      base + opening.val * 61 + index.val := by
  rw [selectedLayout_column_val]
  simp only [coordinateOffset, openingWidth_eq]
  omega

@[simp] theorem selectedLayout_borrow_column
    {columns openingCount : Nat}
    (base : Nat)
    (fits : base + openingCount * openingWidth ≤ columns)
    (opening : Fin openingCount) (index : Fin chunkBorrowCount) :
    ((selectedLayout base fits).column (.borrow opening index)).val =
      base + opening.val * 61 + 41 + index.val := by
  rw [selectedLayout_column_val]
  change
    base + (opening.val * 61 + 41 + index.val) =
      base + opening.val * 61 + 41 + index.val
  omega

/-- Read one source assignment through its logical physical columns. -/
def sourceAssignment
    {rows columns freshCount runningCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount) :
    Nat → Nat :=
  fun column =>
    if below : column < columns then
      (data.assignment source
        (Phi81CarrierLayout.embedLogical ⟨column, below⟩)).val
    else
      0

@[simp] theorem sourceAssignment_layoutColumn
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (coordinate : Coordinate openingCount) :
    sourceAssignment data source (layout.column coordinate).val =
      (coordinateValue layout data source coordinate).val := by
  simp [sourceAssignment, coordinateValue, carrierColumn,
    (layout.column coordinate).isLt]

/-- Relocate a local canonicality polynomial to one selected physical frame.
Columns outside the 41-digit/20-borrow interface become literal zero. -/
abbrev CanonicalPolynomial :=
  Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Polynomial

abbrev CanonicalEquation :=
  Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Equation

def relocatePolynomial
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (opening : Fin openingCount) :
    CanonicalPolynomial → CanonicalPolynomial
  | .constant value => .constant value
  | .variable column =>
      match localCoordinate? opening column with
      | some coordinate => .variable (layout.column coordinate).val
      | none => .constant 0
  | .add left right =>
      .add (relocatePolynomial layout opening left)
        (relocatePolynomial layout opening right)
  | .mul left right =>
      .mul (relocatePolynomial layout opening left)
        (relocatePolynomial layout opening right)

def relocateEquation
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (opening : Fin openingCount)
    (equation : CanonicalEquation) : CanonicalEquation where
  left := relocatePolynomial layout opening equation.left
  right := relocatePolynomial layout opening equation.right

/-- The selected physical 21-row program for one opening. -/
def emittedRows
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (opening : Fin openingCount) : List CanonicalEquation :=
  (List.range chunkCount).map fun chunk =>
    relocateEquation layout opening (chunkEquation chunk)

theorem emittedRows_length
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (opening : Fin openingCount) :
    (emittedRows layout opening).length = 21 := by
  simp [emittedRows, chunkCount]

/-- Satisfaction of the selected physical rows by one authoritative source. -/
def EmittedRowsHold
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) : Prop :=
  ∀ equation ∈ emittedRows layout opening,
    equation.Holds (sourceAssignment data source)

theorem eval_relocatePolynomial
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount)
    (polynomial : CanonicalPolynomial) :
    (relocatePolynomial layout opening polynomial).eval
        (sourceAssignment data source) =
      polynomial.eval (localAssignment layout data source opening) := by
  induction polynomial with
  | constant value =>
      rfl
  | «variable» column =>
      cases matchEq : localCoordinate? opening column with
      | none =>
          simp [relocatePolynomial, localAssignment, matchEq,
            Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Polynomial.eval]
      | some coordinate =>
          simp [relocatePolynomial, localAssignment, matchEq,
            Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Polynomial.eval,
            sourceAssignment_layoutColumn]
  | add left right leftInduction rightInduction =>
      simp [relocatePolynomial,
        Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Polynomial.eval,
        leftInduction, rightInduction]
  | mul left right leftInduction rightInduction =>
      simp [relocatePolynomial,
        Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Polynomial.eval,
        leftInduction, rightInduction]

theorem relocateEquation_holds_iff
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount)
    (equation : CanonicalEquation) :
    (relocateEquation layout opening equation).Holds
        (sourceAssignment data source) ↔
      equation.Holds (localAssignment layout data source opening) := by
  unfold Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Equation.Holds
    relocateEquation
  rw [eval_relocatePolynomial, eval_relocatePolynomial]

theorem emittedRowsHold_iff_canonicalRowsHold
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) :
    EmittedRowsHold layout data source opening ↔
      CanonicalRowsHold layout data source opening := by
  constructor
  · intro rowsHold chunk chunkLt
    apply (relocateEquation_holds_iff
      layout data source opening (chunkEquation chunk)).mp
    apply rowsHold
    exact List.mem_map.mpr
      ⟨chunk, List.mem_range.mpr chunkLt, rfl⟩
  · intro canonical equation member
    rcases List.mem_map.mp member with
      ⟨chunk, chunkMember, equationEq⟩
    subst equation
    apply (relocateEquation_holds_iff
      layout data source opening (chunkEquation chunk)).mpr
    exact canonical chunk (List.mem_range.mp chunkMember)

/-- Syntactic physical columns read by a polynomial. -/
def polynomialColumns : CanonicalPolynomial → List Nat
  | .constant _ => []
  | .variable column => [column]
  | .add left right =>
      polynomialColumns left ++ polynomialColumns right
  | .mul left right =>
      polynomialColumns left ++ polynomialColumns right

def equationColumns (equation : CanonicalEquation) : List Nat :=
  polynomialColumns equation.left ++ polynomialColumns equation.right

/-- The exact local interface owned by one canonical opening. -/
def requiredLocalColumns : List Nat :=
  ((List.range digitCount).map fun index =>
      ShiftedTernary.digitCols.getD index 0) ++
    ((List.range chunkBorrowCount).map fun index =>
      chunkBorrowColumnBase + index)

/-- All source columns read by the Lean-owned 21-row schedule. -/
def sourceRowColumns : List Nat :=
  chunkEquations.flatMap equationColumns

/-- Fail-closed support census: every required digit and borrow column occurs
in the source schedule. -/
theorem requiredLocalColumns_used :
    ∀ column ∈ requiredLocalColumns,
      column ∈ sourceRowColumns := by
  decide

/-- Fail-closed support census: the source schedule contains no column outside
the 41-digit/20-borrow interface. -/
theorem sourceRowColumns_owned :
    ∀ column ∈ sourceRowColumns,
      column ∈ requiredLocalColumns := by
  decide

theorem relocatePolynomial_columns_owned
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (opening : Fin openingCount)
    (polynomial : CanonicalPolynomial)
    (column : Nat)
    (member :
      column ∈ polynomialColumns
        (relocatePolynomial layout opening polynomial)) :
    ∃ coordinate : Coordinate openingCount,
      column = (layout.column coordinate).val := by
  induction polynomial with
  | constant value =>
      simp [relocatePolynomial, polynomialColumns] at member
  | «variable» localColumn =>
      cases matchEq : localCoordinate? opening localColumn with
      | none =>
          simp [relocatePolynomial, polynomialColumns, matchEq] at member
      | some coordinate =>
          simp [relocatePolynomial, polynomialColumns, matchEq] at member
          exact ⟨coordinate, member⟩
  | add left right leftInduction rightInduction =>
      simp only [relocatePolynomial, polynomialColumns,
        List.mem_append] at member
      rcases member with leftMember | rightMember
      · exact leftInduction leftMember
      · exact rightInduction rightMember
  | mul left right leftInduction rightInduction =>
      simp only [relocatePolynomial, polynomialColumns,
        List.mem_append] at member
      rcases member with leftMember | rightMember
      · exact leftInduction leftMember
      · exact rightInduction rightMember

/-- A recognized local source column remains present after physical
relocation. -/
theorem relocatePolynomial_preserves_column
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (opening : Fin openingCount)
    (polynomial : CanonicalPolynomial)
    (localColumn : Nat)
    (coordinate : Coordinate openingCount)
    (resolved :
      localCoordinate? opening localColumn = some coordinate)
    (member : localColumn ∈ polynomialColumns polynomial) :
    (layout.column coordinate).val ∈
      polynomialColumns
        (relocatePolynomial layout opening polynomial) := by
  induction polynomial with
  | constant value =>
      simp [polynomialColumns] at member
  | «variable» column =>
      simp [polynomialColumns] at member
      subst column
      simp [relocatePolynomial, polynomialColumns, resolved]
  | add left right leftInduction rightInduction =>
      simp only [relocatePolynomial, polynomialColumns,
        List.mem_append] at member ⊢
      rcases member with leftMember | rightMember
      · exact Or.inl (leftInduction leftMember)
      · exact Or.inr (rightInduction rightMember)
  | mul left right leftInduction rightInduction =>
      simp only [relocatePolynomial, polynomialColumns,
        List.mem_append] at member ⊢
      rcases member with leftMember | rightMember
      · exact Or.inl (leftInduction leftMember)
      · exact Or.inr (rightInduction rightMember)

theorem relocateEquation_preserves_column
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (opening : Fin openingCount)
    (equation : CanonicalEquation)
    (localColumn : Nat)
    (coordinate : Coordinate openingCount)
    (resolved :
      localCoordinate? opening localColumn = some coordinate)
    (member : localColumn ∈ equationColumns equation) :
    (layout.column coordinate).val ∈
      equationColumns (relocateEquation layout opening equation) := by
  simp only [equationColumns, relocateEquation,
    List.mem_append] at member ⊢
  rcases member with leftMember | rightMember
  · exact Or.inl
      (relocatePolynomial_preserves_column
        layout opening equation.left localColumn coordinate
        resolved leftMember)
  · exact Or.inr
      (relocatePolynomial_preserves_column
        layout opening equation.right localColumn coordinate
        resolved rightMember)

/-- Every one of the 41 selected digit columns occurs in the emitted physical
row program. -/
theorem selected_digit_column_used
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (opening : Fin openingCount)
    (index : Fin digitCount) :
    (layout.column (.digit opening index)).val ∈
      (emittedRows layout opening).flatMap equationColumns := by
  let localColumn :=
    ShiftedTernary.digitCols.getD index.val 0
  have required : localColumn ∈ requiredLocalColumns := by
    unfold requiredLocalColumns
    apply List.mem_append_left
    apply List.mem_map.mpr
    exact ⟨index.val, List.mem_range.mpr index.isLt, rfl⟩
  have used :=
    requiredLocalColumns_used localColumn required
  rcases List.mem_flatMap.mp used with
    ⟨equation, equationMember, columnMember⟩
  apply List.mem_flatMap.mpr
  refine ⟨relocateEquation layout opening equation, ?_, ?_⟩
  · unfold chunkEquations at equationMember
    unfold emittedRows
    rcases List.mem_map.mp equationMember with
      ⟨chunk, chunkMember, equationEq⟩
    subst equation
    exact List.mem_map.mpr
      ⟨chunk, chunkMember, rfl⟩
  · exact relocateEquation_preserves_column
      layout opening equation localColumn (.digit opening index)
      (localCoordinate?_digit opening index) columnMember

/-- Every one of the 20 selected retained-borrow columns occurs in the emitted
physical row program. -/
theorem selected_borrow_column_used
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (opening : Fin openingCount)
    (index : Fin chunkBorrowCount) :
    (layout.column (.borrow opening index)).val ∈
      (emittedRows layout opening).flatMap equationColumns := by
  let localColumn := chunkBorrowColumnBase + index.val
  have required : localColumn ∈ requiredLocalColumns := by
    unfold requiredLocalColumns
    apply List.mem_append_right
    apply List.mem_map.mpr
    exact ⟨index.val, List.mem_range.mpr index.isLt, rfl⟩
  have used :=
    requiredLocalColumns_used localColumn required
  rcases List.mem_flatMap.mp used with
    ⟨equation, equationMember, columnMember⟩
  apply List.mem_flatMap.mpr
  refine ⟨relocateEquation layout opening equation, ?_, ?_⟩
  · unfold chunkEquations at equationMember
    unfold emittedRows
    rcases List.mem_map.mp equationMember with
      ⟨chunk, chunkMember, equationEq⟩
    subst equation
    exact List.mem_map.mpr
      ⟨chunk, chunkMember, rfl⟩
  · exact relocateEquation_preserves_column
      layout opening equation localColumn (.borrow opening index)
      (localCoordinate?_borrow opening index) columnMember

/-- Every physical column read by every emitted row has one typed opening
owner. No unrelated source column can enter the canonical-opening program. -/
theorem emittedRows_columns_owned
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (opening : Fin openingCount)
    (equation : CanonicalEquation)
    (rowMember : equation ∈ emittedRows layout opening)
    (column : Nat)
    (columnMember : column ∈ equationColumns equation) :
    ∃ coordinate : Coordinate openingCount,
      column = (layout.column coordinate).val := by
  rcases List.mem_map.mp rowMember with
    ⟨chunk, chunkMember, equationEq⟩
  subst equation
  simp only [equationColumns, relocateEquation,
    List.mem_append] at columnMember
  rcases columnMember with leftMember | rightMember
  · exact relocatePolynomial_columns_owned
      layout opening _ column leftMember
  · exact relocatePolynomial_columns_owned
      layout opening _ column rightMember

/-- Selected Split-NC coverage for all 41 digit and 20 retained-borrow
coordinates of one physical opening. -/
theorem selectedSplitNc_covers_opening
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {rows columns freshCount runningCount openingCount base : Nat}
    {profile : RelationProfile.Profile rows columns}
    (fits : base + openingCount * openingWidth ≤ columns)
    (data : Data (ncShape profile freshCount runningCount))
    (splitNcRows : Semantics.Nc.BlockLane.ResidualsZero data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) :
    OpeningCoordinatesBoundTwo
      (selectedLayout base fits) data source opening := by
  exact splitNc_covers_opening
    (selectedLayout base fits) data
    (Semantics.Nc.BlockLane.truth_of_residualsZero
      noZeroDivisors data splitNcRows)
    source opening

/-- Every column read by an emitted row is covered by selected Split-NC at
`b = 2`. -/
theorem selectedSplitNc_covers_emittedColumn
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {rows columns freshCount runningCount openingCount base : Nat}
    {profile : RelationProfile.Profile rows columns}
    (fits : base + openingCount * openingWidth ≤ columns)
    (data : Data (ncShape profile freshCount runningCount))
    (splitNcRows : Semantics.Nc.BlockLane.ResidualsZero data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount)
    (equation : CanonicalEquation)
    (rowMember :
      equation ∈ emittedRows (selectedLayout base fits) opening)
    (column : Nat)
    (columnMember : column ∈ equationColumns equation) :
    ∃ coordinate : Coordinate openingCount,
      column = ((selectedLayout base fits).column coordinate).val ∧
        NormBoundTwo (sourceAssignment data source column) := by
  rcases emittedRows_columns_owned
      (selectedLayout base fits) opening equation rowMember
      column columnMember with
    ⟨coordinate, columnEq⟩
  refine ⟨coordinate, columnEq, ?_⟩
  subst column
  rw [sourceAssignment_layoutColumn]
  exact splitNc_covers_coordinate
    (selectedLayout base fits) data
    (Semantics.Nc.BlockLane.truth_of_residualsZero
      noZeroDivisors data splitNcRows)
    source coordinate

/-- Selected verifier residual rows derive the exact digit premise used by
the canonicality theorem. No digit-bound premise is accepted from a caller. -/
theorem selectedSplitNcRows_supply_digitNormBoundTwo
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {rows columns freshCount runningCount openingCount base : Nat}
    {profile : RelationProfile.Profile rows columns}
    (fits : base + openingCount * openingWidth ≤ columns)
    (data : Data (ncShape profile freshCount runningCount))
    (splitNcRows : Semantics.Nc.BlockLane.ResidualsZero data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) :
    DigitNormBoundTwo
      (localAssignment (selectedLayout base fits) data source opening) := by
  exact splitNc_supplies_digitNormBoundTwo
    (selectedLayout base fits) data
    (Semantics.Nc.BlockLane.truth_of_residualsZero
      noZeroDivisors data splitNcRows)
    source opening

/-- Headline selected-layout refinement. The Split-NC rows supply ternarity,
the emitted 21 physical rows supply canonicality, and the composed result is a
canonical Goldilocks opening. -/
theorem selectedPhysicalRows_encoded_lt_modulus
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {rows columns freshCount runningCount openingCount base : Nat}
    {profile : RelationProfile.Profile rows columns}
    (fits : base + openingCount * openingWidth ≤ columns)
    (data : Data (ncShape profile freshCount runningCount))
    (splitNcRows : Semantics.Nc.BlockLane.ResidualsZero data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount)
    (canonicalRows :
      EmittedRowsHold
        (selectedLayout base fits) data source opening) :
    lowValue
        (assignmentTritMod
          (localAssignment
            (selectedLayout base fits) data source opening))
        digitCount <
      goldilocksP := by
  exact chunkSchedule_encoded_lt_modulus
    (selectedSplitNcRows_supply_digitNormBoundTwo
      noZeroDivisors fits data splitNcRows source opening)
    ((emittedRowsHold_iff_canonicalRowsHold
      (selectedLayout base fits) data source opening).mp canonicalRows)

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement
