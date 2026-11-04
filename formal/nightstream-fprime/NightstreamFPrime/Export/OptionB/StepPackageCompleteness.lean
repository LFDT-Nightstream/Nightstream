import NightstreamFPrime.Export.OptionB.StepPackage
import NightstreamFPrime.Export.Generic.CircuitRelationCompleteness

/-! Completeness at the package level (CW-5.9). `slotRuns` gives a slot to every column from the
public columns to the last S-box column except the blocks' dead locals (`slotRuns_covers`), so
the package's slots cover its circuit's relation when the blocks pack (`covers_packageOf`). For a
package that Rust builds, such as one whose application region holds one native rows piece and
native Poseidon2 blocks, `package_complete` is the companion of
`CircuitRelationCompleteness.rowsZero_of_pieces`: when the pieces' rows hold in the honest
environment (a ring piece's rows with its honest quotients, `PieceRowsCompleteness.ringRows_complete`)
and every block's template constraints hold, the prover's assignment (`blockAssignment`) makes the
package's rows and S-box rows zero, and its public input encodes the honest digest. -/

namespace NightstreamFPrime.Export.OptionB.StepPackageCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.NativeBlock
open NightstreamFPrime.Export.Generic.NativeBlockCompleteness
open NightstreamFPrime.Export.Generic.SlotLayout
open NightstreamFPrime.Export.Generic.SlotLayoutCompleteness
open NightstreamFPrime.Export.OptionB.StepPackage
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-! ### The kind codes of `slotRuns` -/

/-- The column count of `slotRuns`: up to the last S-box column. -/
def size (pieces : List Piece) (next : Nat) : Nat :=
  CircuitRelation.sboxStart pieces next + 150 * (CircuitRelation.blocksOf pieces).length

/-- The kind code of each column, as `slotRuns` computes it: 3 for a dead local, 0 for a column
that a `bit` hint fills, 2 otherwise. -/
def codesOf (pieces : List Piece) (next : Nat) (batches : List WitnessBatch) : Array Nat :=
  batches.foldl (fun codes batch =>
      (batch.hints.zipIdx).foldl (fun codes (hint, index) => match hint with
        | .bit .. => codes.set! (batch.start + batch.recipes.length + index) 0
        | _ => codes) codes)
    ((CircuitRelation.blocksOf pieces).foldl (fun codes block =>
        (List.range 1080).foldl (fun codes offset => codes.set! (block.start + offset) 3) codes)
      (Array.replicate (size pieces next) (2 : Nat)))

/-- One column of the run fold of `slotRuns`. -/
def runStep (codes : Array Nat) (runs : List (Nat × Nat × Nat)) (column : Nat) :
    List (Nat × Nat × Nat) :=
  let code : Nat := codes.getD column 3
  if code = 3 then runs else
    match runs with
    | (start, count, kind) :: rest =>
        if kind = code ∧ start + count = column then (start, count + 1, kind) :: rest
        else (column, 1, code) :: runs
    | [] => [(column, 1, code)]

theorem slotRuns_eq (publicColumns : Nat) (pieces : List Piece) (next : Nat)
    (batches : List WitnessBatch) :
    slotRuns publicColumns pieces next batches =
      (((List.range (size pieces next)).drop publicColumns).foldl
        (runStep (codesOf pieces next batches)) []).reverse :=
  rfl

theorem foldl_invariant {α β : Type} (invariant : β → Prop) (step : β → α → β) :
    ∀ (items : List α) (initial : β), (∀ item ∈ items, ∀ value, invariant value →
      invariant (step value item)) → invariant initial → invariant (items.foldl step initial)
  | [], _, _, holds => holds
  | item :: rest, _, preserves, holds =>
      foldl_invariant invariant step rest _
        (fun other member => preserves other (List.mem_cons_of_mem _ member))
        (preserves item List.mem_cons_self _ holds)

/-- The codes keep their size, and a code 3 marks a dead local. -/
def CodesValid (pieces : List Piece) (next : Nat) (codes : Array Nat) : Prop :=
  codes.size = size pieces next ∧
    ∀ column, codes[column]? = some 3 → deadIn (CircuitRelation.blocksOf pieces) column = true

theorem codesValid_set {pieces : List Piece} {next : Nat} {codes : Array Nat}
    (valid : CodesValid pieces next codes) (index value : Nat)
    (allowed : value = 3 → deadIn (CircuitRelation.blocksOf pieces) index = true) :
    CodesValid pieces next (codes.set! index value) := by
  refine ⟨by simp [Array.set!, valid.1], fun column found => ?_⟩
  simp only [Array.set!, Array.getElem?_setIfInBounds] at found
  by_cases same : index = column
  · subst same
    by_cases inside : index < codes.size
    · simp only [inside, ↓reduceIte, Option.some.injEq] at found
      exact allowed found
    · simp [inside] at found
  · simp only [same, ↓reduceIte] at found
    exact valid.2 column found

theorem codesOf_valid (pieces : List Piece) (next : Nat) (batches : List WitnessBatch) :
    CodesValid pieces next (codesOf pieces next batches) := by
  unfold codesOf
  apply foldl_invariant (CodesValid pieces next)
  · intro batch _ codes valid
    apply foldl_invariant (CodesValid pieces next) _ _ _ _ valid
    rintro ⟨hint, index⟩ _ codes valid
    cases hint with
    | bit source position =>
        exact codesValid_set valid _ 0 (fun zero => absurd zero (by decide))
    | inverseOrZero source => exact valid
    | quotientFive source => exact valid
    | remainderFive source => exact valid
  · apply foldl_invariant (CodesValid pieces next)
    · intro block member codes valid
      apply foldl_invariant (CodesValid pieces next) _ _ _ _ valid
      intro offset inside codes valid
      apply codesValid_set valid _ 3
      intro _
      simp only [deadIn, List.any_eq_true, Block.dead, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨block, member, by omega, by have := List.mem_range.mp inside; omega⟩
    · refine ⟨by simp, fun column found => ?_⟩
      simp only [Array.getElem?_replicate] at found
      split at found <;> simp at found

theorem code_ne {pieces : List Piece} {next : Nat} {batches : List WitnessBatch} {column : Nat}
    (inside : column < size pieces next)
    (live : deadIn (CircuitRelation.blocksOf pieces) column = false) :
    (codesOf pieces next batches).getD column 3 ≠ 3 := by
  have valid := codesOf_valid pieces next batches
  intro three
  have bound : column < (codesOf pieces next batches).size := by rw [valid.1]; exact inside
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem bound, Option.getD_some] at three
  have dead := valid.2 column (by rw [Array.getElem?_eq_getElem bound, three])
  rw [live] at dead
  exact Bool.false_ne_true dead

/-! ### The runs of `slotRuns` -/

def Covered (runs : List (Nat × Nat × Nat)) (column : Nat) : Prop :=
  ∃ run ∈ runs, run.1 ≤ column ∧ column < run.1 + run.2.1

theorem runStep_mono (codes : Array Nat) (runs : List (Nat × Nat × Nat)) (column other : Nat)
    (covered : Covered runs other) : Covered (runStep codes runs column) other := by
  obtain ⟨run, member, low, high⟩ := covered
  unfold runStep
  dsimp only
  split
  · exact ⟨run, member, low, high⟩
  · match runs, member with
    | (start, count, kind) :: rest, member =>
        dsimp only
        split
        · rcases List.mem_cons.mp member with rfl | member
          · exact ⟨(start, count + 1, kind), List.mem_cons_self, low, by simp only at high ⊢; omega⟩
          · exact ⟨run, List.mem_cons_of_mem _ member, low, high⟩
        · exact ⟨run, List.mem_cons_of_mem _ member, low, high⟩

theorem runStep_covers (codes : Array Nat) (runs : List (Nat × Nat × Nat)) (column : Nat)
    (code : codes.getD column 3 ≠ 3) : Covered (runStep codes runs column) column := by
  unfold runStep
  dsimp only
  rw [ite_eq_right code]
  match runs with
  | (start, count, kind) :: rest =>
      dsimp only
      split
      · exact ⟨(start, count + 1, kind), List.mem_cons_self, by omega, by simp only; omega⟩
      · exact ⟨(column, 1, _), List.mem_cons_self, le_refl _, by simp⟩
  | [] => exact ⟨(column, 1, _), List.mem_cons_self, le_refl _, by simp⟩

theorem runFold_mono (codes : Array Nat) :
    ∀ (columns : List Nat) (runs : List (Nat × Nat × Nat)) (other : Nat),
      Covered runs other → Covered (columns.foldl (runStep codes) runs) other
  | [], _, _, covered => covered
  | column :: rest, runs, other, covered =>
      runFold_mono codes rest _ other (runStep_mono codes runs column other covered)

theorem runFold_covers (codes : Array Nat) :
    ∀ (columns : List Nat) (runs : List (Nat × Nat × Nat)) (column : Nat),
      column ∈ columns → codes.getD column 3 ≠ 3 →
        Covered (columns.foldl (runStep codes) runs) column
  | [], _, _, member, _ => by simp at member
  | first :: rest, runs, column, member, code => by
      rw [List.foldl_cons]
      rcases List.mem_cons.mp member with rfl | member
      · exact runFold_mono codes rest _ column (runStep_covers codes runs column code)
      · exact runFold_covers codes rest _ column member code

/-- `slotRuns` puts every live column from the public columns to the last S-box column in a run. -/
theorem slotRuns_covers (publicColumns : Nat) (pieces : List Piece) (next : Nat)
    (batches : List WitnessBatch) {column : Nat} (low : publicColumns ≤ column)
    (high : column < size pieces next)
    (live : deadIn (CircuitRelation.blocksOf pieces) column = false) :
    ∃ run ∈ slotRuns publicColumns pieces next batches, run.1 ≤ column ∧
      column < run.1 + run.2.1 := by
  rw [slotRuns_eq]
  have member : column ∈ (List.range (size pieces next)).drop publicColumns := by
    rw [List.range_eq_range', List.drop_range', List.mem_range'_1]
    omega
  obtain ⟨run, runMember, runLow, runHigh⟩ :=
    runFold_covers _ _ [] column member (code_ne high live)
  exact ⟨run, List.mem_reverse.mpr runMember, runLow, runHigh⟩

/-- When the blocks pack, the relation fits the width up to the last S-box column. -/
theorem size_le {publicColumns : Nat} {pieces : List Piece} {next width : Nat}
    (fits : CircuitRelation.Fits publicColumns pieces next width)
    (packed : CircuitRelation.packable publicColumns pieces = true) : size pieces next ≤ width := by
  have sboxColumns := (fits.blocks packed).sboxColumns
  unfold size
  rw [Nat.mul_comm]
  exact sboxColumns

/-- The slots of `slotRuns` cover the relation of the circuit's pieces when the blocks pack. -/
theorem covers_packageOf (pieces : List Piece) (next width : Nat) (batches : List WitnessBatch)
    (packed : CircuitRelation.packable CircuitRelation.publicColumns pieces = true)
    (fits : size pieces next ≤ width) :
    CircuitRelationCompleteness.Covers
      (runSlots width (slotRuns CircuitRelation.publicColumns pieces next batches))
      CircuitRelation.publicColumns pieces next := by
  intro column notPublic high live
  have endEq : CircuitRelationCompleteness.columnEnd CircuitRelation.publicColumns pieces next =
      size pieces next := by
    unfold CircuitRelationCompleteness.columnEnd size
    rw [ite_eq_left packed, Nat.mul_comm]
  have dead : deadIn (CircuitRelation.blocksOf pieces) column = false := by
    simpa [CircuitRelationCompleteness.deadColumn, packed] using live
  rw [endEq] at high
  obtain ⟨⟨start, count, kind⟩, member, low, runHigh⟩ :=
    slotRuns_covers CircuitRelation.publicColumns pieces next batches notPublic high dead
  exact ⟨_, runSlots_mem member low runHigh (by omega), rfl⟩

/-! ### Rust-built packages -/

/-- The prover's assignment of a package: the S-box columns filled from the honest
environment, then encoded in the package's slots. -/
def packageAssignment (package : Package) (honest : Env) :
    Assignment F (ProductionAssignment.logicalWidth (runSlots package.width package.slots)) :=
  blockAssignment (runSlots package.width package.slots) package.blocks package.sboxStart honest

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Package completeness, the companion of `CircuitRelationCompleteness.rowsZero_of_pieces` for a
package that Rust builds: when the pieces' rows hold in the honest environment, every block's
template constraints hold there, the rows read no dead local and read below the S-box columns,
the blocks read below them, and every live column from the public columns to the last S-box column
has a slot, the prover's assignment makes the package's rows and S-box rows zero. Its public input
is the honest one. -/
theorem package_complete (package : Package) (frame : Framed package.blocks)
    (avoid : (package.pieces.flatMap PackagePiece.relationRows).all
      (rowAvoids (deadIn package.blocks)) = true)
    (rowsBelow : ∀ row ∈ package.pieces.flatMap PackagePiece.relationRows,
      row.VarsBelow package.sboxStart)
    (blocksBelow : ∀ block ∈ package.blocks, ∀ expression ∈
      (Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()),
      expression.VarsBelow package.sboxStart)
    (publicBelow : LogicalPlan.publicCount ≤ package.sboxStart)
    (sboxFits : package.sboxStart + package.blocks.length * 150 ≤ package.width)
    (covered : ∀ column, LogicalPlan.publicCount ≤ column →
      column < package.sboxStart + package.blocks.length * 150 →
        deadIn package.blocks column = false →
          ∃ slot ∈ runSlots package.width package.slots, slot.source.val = column)
    (fits : LogicalPlan.Fits package.relationRows package.width)
    (rowCount_le : package.blocks.length * 150 ≤ 2 ^ cubeVariables)
    (honest : Env) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) honest = encHash digest)
    (rowsHold : R1CS.RowsHold honest (package.pieces.flatMap PackagePiece.relationRows))
    (templates : ∀ block ∈ package.blocks, ConstraintsHold honest
      ((Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()))) :
    (LogicalPlan.plan package.relationRows package.width fits
        (layout (runSlots package.width package.slots))).RowsZero
        (packageAssignment package honest) ∧
      (PoseidonSboxFamilyPlan.plan (CircuitRelation.sboxInterface
          (layout (runSlots package.width package.slots)) package.blocks package.sboxStart)
        rowCount_le).RowsZero (packageAssignment package honest) ∧
      LogicalPlan.publicInput (publicFits := publicFits) (LogicalPlan.sourceEnv
        (layout (runSlots package.width package.slots)) (packageAssignment package honest)) =
        encHash digest := by
  have one := CircuitRelationCompleteness.one_of_encoded (publicFits := publicFits) honest digest
    encoded
  obtain ⟨sboxRows, rows, publicEq, _⟩ := blocks_rows_complete
    (runSlots package.width package.slots) frame package.sboxStart avoid rowsBelow blocksBelow
    publicBelow sboxFits covered rowCount_le honest one rowsHold templates
  have oneEq : packageAssignment package honest
      (layout (runSlots package.width package.slots)).one = 1 := by
    unfold packageAssignment blockAssignment
    rw [encode_one, sboxFill_outside (Or.inl (lt_of_lt_of_le (by decide) publicBelow))]
    exact one
  refine ⟨(LogicalPlan.rowsZero_iff_rowsHold _ _ fits _ _ oneEq).mpr rows, sboxRows, ?_⟩
  rw [← encoded]
  funext column
  exact publicEq column.val column.isLt

end NightstreamFPrime.Export.OptionB.StepPackageCompleteness
