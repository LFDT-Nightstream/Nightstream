import NightstreamFPrime.Export.OptionB.StepPackage
import NightstreamFPrime.Export.Generic.CircuitRelationNorm

/-! Completeness at the package level (CW-5.9). `slotRuns` gives a slot to every column from the
public columns to the last S-box column except the blocks' dead locals (`slotRuns_covers`), so
the package's slots cover its circuit's relation when the blocks pack (`covers_packageOf`). For a
package that Rust builds, such as one whose application region holds one native rows piece and
native Poseidon2 blocks, `package_complete` is the companion of
`CircuitRelationCompleteness.rowsZero_of_pieces`: when the pieces' rows hold in the honest
environment (a ring piece's rows with its honest quotients, `PieceRowsCompleteness.ringRows_complete`)
and every block's template constraints hold, the prover's assignment (`blockAssignment`) makes the
package's rows and S-box rows zero, and its public input encodes the honest digest. When the
application is native, its rows and its blocks' template constraints, with the rest of the step,
suffice (`native_complete`).

A slot of `slotRuns` is a field slot, or a bit slot on the column of a `bit` hint
(`slotRuns_kind`). The slots are valid for an environment that holds each `bit` hint's value at
its column (`slotRuns_valid`, `NativeBlock.bitHint_valid`). -/

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

/-! ### The kinds of `slotRuns` -/

/-- `column` is the column of a `bit` hint of `batches`. -/
def BitColumn (batches : List WitnessBatch) (column : Nat) : Prop :=
  ∃ batch ∈ batches, ∃ index source bit, batch.hints[index]? = some (Hint.bit source bit) ∧
    column = batch.start + batch.recipes.length + index

/-- Every code is 2 or 3, or 0 on the column of a `bit` hint. -/
def CodesKinds (batches : List WitnessBatch) (codes : Array Nat) : Prop :=
  ∀ column code, codes[column]? = some code →
    code = 2 ∨ code = 3 ∨ (code = 0 ∧ BitColumn batches column)

theorem codesKinds_set {batches : List WitnessBatch} {codes : Array Nat}
    (kinds : CodesKinds batches codes) (index value : Nat)
    (allowed : value = 2 ∨ value = 3 ∨ (value = 0 ∧ BitColumn batches index)) :
    CodesKinds batches (codes.set! index value) := by
  intro column code found
  simp only [Array.set!, Array.getElem?_setIfInBounds] at found
  by_cases same : index = column
  · subst same
    by_cases inside : index < codes.size
    · simp only [inside, ↓reduceIte, Option.some.injEq] at found
      subst found
      exact allowed
    · simp [inside] at found
  · simp only [same, ↓reduceIte] at found
    exact kinds column code found

theorem codesOf_kinds (pieces : List Piece) (next : Nat) (batches : List WitnessBatch) :
    CodesKinds batches (codesOf pieces next batches) := by
  unfold codesOf
  apply foldl_invariant (CodesKinds batches)
  · intro batch batchMember codes kinds
    apply foldl_invariant (CodesKinds batches) _ _ _ _ kinds
    rintro ⟨hint, index⟩ member codes kinds
    cases hint with
    | bit source position =>
        exact codesKinds_set kinds _ 0 (Or.inr (Or.inr ⟨rfl, batch, batchMember, index, source,
          position, List.mem_zipIdx_iff_getElem?.mp member, rfl⟩))
    | inverseOrZero source => exact kinds
    | quotientFive source => exact kinds
    | remainderFive source => exact kinds
  · apply foldl_invariant (CodesKinds batches)
    · intro block _ codes kinds
      apply foldl_invariant (CodesKinds batches) _ _ _ _ kinds
      intro offset _ codes kinds
      exact codesKinds_set kinds _ 3 (Or.inr (Or.inl rfl))
    · intro column code found
      rw [Array.getElem?_replicate] at found
      split at found
      · exact Or.inl (Option.some.inj found).symm
      · exact absurd found (by simp)

/-- Each run's columns carry its kind code, which is not 3. -/
def RunsKinds (codes : Array Nat) (runs : List (Nat × Nat × Nat)) : Prop :=
  ∀ run ∈ runs, run.2.2 ≠ 3 ∧
    ∀ column, run.1 ≤ column → column < run.1 + run.2.1 → codes.getD column 3 = run.2.2

theorem runsKinds_cons {codes : Array Nat} {run : Nat × Nat × Nat}
    {runs : List (Nat × Nat × Nat)} (kind : run.2.2 ≠ 3)
    (columns : ∀ column, run.1 ≤ column → column < run.1 + run.2.1 →
      codes.getD column 3 = run.2.2)
    (rest : RunsKinds codes runs) : RunsKinds codes (run :: runs) := by
  intro other member
  rcases List.mem_cons.mp member with rfl | member
  · exact ⟨kind, columns⟩
  · exact rest other member

theorem runStep_kinds (codes : Array Nat) (runs : List (Nat × Nat × Nat)) (column : Nat)
    (kinds : RunsKinds codes runs) : RunsKinds codes (runStep codes runs column) := by
  have fresh : ∀ others : List (Nat × Nat × Nat), RunsKinds codes others →
      codes.getD column 3 ≠ 3 → RunsKinds codes ((column, 1, codes.getD column 3) :: others) :=
    fun others rest live => runsKinds_cons live (fun other low high => by
      simp only at low high
      rw [show other = column by omega]) rest
  unfold runStep
  dsimp only
  split
  · exact kinds
  · rename_i live
    match runs, kinds with
    | (start, count, kind) :: rest, kinds =>
        dsimp only
        split
        · rename_i extend
          obtain ⟨kindNe, columns⟩ := kinds _ List.mem_cons_self
          refine runsKinds_cons kindNe (fun other low high => ?_)
            (fun run member => kinds run (List.mem_cons_of_mem _ member))
          simp only at low high columns ⊢
          by_cases last : other < start + count
          · exact columns other low last
          · rw [show other = column by omega]
            exact extend.1.symm
        · exact fresh _ kinds live
    | [], _ => exact fresh [] (fun _ member => by simp at member) live

/-- A slot of `slotRuns` is a field slot, or a bit slot on the column of a `bit` hint. -/
theorem slotRuns_kind (publicColumns : Nat) (pieces : List Piece) (next width : Nat)
    (batches : List WitnessBatch) {slot : LowNormAssignment.Slot width}
    (member : slot ∈ runSlots width (slotRuns publicColumns pieces next batches)) :
    slot.kind = .field ∨ (slot.kind = .bit ∧ BitColumn batches slot.source.val) := by
  rw [slotRuns_eq] at member
  unfold runSlots at member
  obtain ⟨⟨start, count, kind⟩, runMember, slotMember⟩ := List.mem_flatMap.mp member
  obtain ⟨offset, offsetMember, found⟩ := List.mem_filterMap.mp slotMember
  have offsetLt := List.mem_range.mp offsetMember
  split at found
  · obtain rfl := Option.some.inj found
    have kinds := foldl_invariant (RunsKinds (codesOf pieces next batches))
      (runStep (codesOf pieces next batches))
      ((List.range (size pieces next)).drop publicColumns) []
      (fun column _ runs kinds => runStep_kinds _ runs column kinds)
      (fun _ member => by simp at member)
    obtain ⟨kindNe, columns⟩ := kinds _ (List.mem_reverse.mp runMember)
    have codeEq := columns (start + offset) (by simp only; omega) (by simp only; omega)
    simp only at kindNe codeEq ⊢
    rw [Array.getD_eq_getD_getElem?] at codeEq
    cases lookup : (codesOf pieces next batches)[start + offset]? with
    | none =>
        rw [lookup] at codeEq
        exact absurd codeEq.symm kindNe
    | some code =>
        rw [lookup, Option.getD_some] at codeEq
        subst codeEq
        rcases codesOf_kinds pieces next batches _ _ lookup with two | three | ⟨zero, bitColumn⟩
        · rw [two]
          exact Or.inl rfl
        · exact absurd three kindNe
        · rw [zero]
          exact Or.inr ⟨rfl, bitColumn⟩
  · exact absurd found (by simp)

/-- The package's slots are valid for an environment that holds each `bit` hint's value at its
column, below `next` (`NativeBlock.bitHint_valid`). -/
theorem slotRuns_valid (publicColumns : Nat) (pieces : List Piece) (next width : Nat)
    (batches : List WitnessBatch) (env : Env)
    (hints : ∀ batch ∈ batches, ∀ index source bit,
      batch.hints[index]? = some (Hint.bit source bit) →
        batch.start + batch.recipes.length + index < next ∧
          env (batch.start + batch.recipes.length + index) = Hint.eval env (.bit source bit)) :
    CircuitRelationNorm.SlotsValid (runSlots width (slotRuns publicColumns pieces next batches))
      next env := by
  intro slot member nonField
  rcases slotRuns_kind publicColumns pieces next width batches member with
    field | ⟨bit, batch, batchMember, index, source, position, found, columnEq⟩
  · exact absurd field nonField
  · obtain ⟨below, value⟩ := hints batch batchMember index source position found
    rw [columnEq, bit, value]
    exact ⟨below, bitHint_valid env source position⟩

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

/-! ### Packages with a native application -/

theorem native_blocks {package : Package} {nativeRows : List R1CS.Row}
    {nativeBlocks : List Block} {rest : List Piece}
    (piecesEq : package.pieces = PackagePiece.rows nativeRows ::
      ((nativeBlocks.map fun block => PackagePiece.block block.start block.lanes) ++
        packagePieces rest package.firstFresh)) :
    package.blocks = nativeBlocks ++ CircuitRelation.blocksOf rest := by
  have nativeEq : ∀ blocks : List Block, (blocks.map fun block =>
      PackagePiece.block block.start block.lanes).filterMap PackagePiece.block? = blocks := by
    intro blocks
    induction blocks with
    | nil => rfl
    | cons block others inductionHypothesis =>
        simp only [List.map_cons, List.filterMap_cons, PackagePiece.block?, inductionHypothesis]
  simp only [Package.blocks, piecesEq, List.filterMap_cons, PackagePiece.block?,
    List.filterMap_append, nativeEq, packagePieces, packageAux_eq, List.reverse_nil,
    List.nil_append, blocks_packageSpec]

theorem native_rows {package : Package} {nativeRows : List R1CS.Row}
    {nativeBlocks : List Block} {rest : List Piece}
    (piecesEq : package.pieces = PackagePiece.rows nativeRows ::
      ((nativeBlocks.map fun block => PackagePiece.block block.start block.lanes) ++
        packagePieces rest package.firstFresh)) :
    package.pieces.flatMap PackagePiece.relationRows = nativeRows ++
      (PieceRows.lowerPieces (PilotData.canonicalConstraints ()) (CircuitRelation.otherPieces rest)
        package.firstFresh).1 := by
  have nativeEq : (nativeBlocks.map fun block =>
      PackagePiece.block block.start block.lanes).flatMap PackagePiece.relationRows = [] := by
    simp [List.flatMap_map, PackagePiece.relationRows]
  simp only [piecesEq, List.flatMap_cons, List.flatMap_append, nativeEq, PackagePiece.relationRows,
    List.nil_append, packagePieces, packageAux_eq, List.reverse_nil, flatMap_packageSpec,
    PieceRows.lowerPieces, lowerPiecesAux_eq _ rfl]

/-- Completeness of a package whose application is native (`with_application` in Rust): one rows
piece and the native Poseidon2 blocks lead the pieces, and the rest are the package pieces of Lean
pieces with fresh columns from the first fresh column. It needs only that the native rows and the
native blocks' template constraints hold in the honest environment, and that the rest of the step
holds there: each Lean piece's constraints hold and read below the first fresh column. The prover
fills the rest's fresh columns (`PieceRowsCompleteness.fillPieces`): the quotient of each ring
product (`ringRows_complete`) and the products of each lowering (`lowerPieces_complete`). Then
`package_complete` gives zero rows and S-box rows and the honest public input. -/
theorem native_complete (package : Package) (nativeRows : List R1CS.Row)
    (nativeBlocks : List Block) (rest : List Piece)
    (piecesEq : package.pieces = PackagePiece.rows nativeRows ::
      ((nativeBlocks.map fun block => PackagePiece.block block.start block.lanes) ++
        packagePieces rest package.firstFresh))
    (frame : Framed package.blocks)
    (nativeAvoid : nativeRows.all (rowAvoids (deadIn package.blocks)) = true)
    (restAvoid : ∀ piece ∈ CircuitRelation.otherPieces rest,
      CircuitRelation.avoidsDead (deadIn package.blocks) piece = true)
    (nativeBelow : ∀ row ∈ nativeRows, row.VarsBelow package.firstFresh)
    (nativeBlocksBelow : ∀ block ∈ nativeBlocks, ∀ expression ∈
      (Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()),
      expression.VarsBelow package.firstFresh)
    (restValid : ∀ piece ∈ rest, piece.Valid)
    (restBelow : ∀ piece ∈ rest,
      ∀ expression ∈ piece.constraints (PilotData.canonicalConstraints ()),
        expression.VarsBelow package.firstFresh)
    (restEnd : CircuitRelation.sboxStart rest package.firstFresh ≤ package.sboxStart)
    (publicBelow : LogicalPlan.publicCount ≤ package.firstFresh)
    (sboxFits : package.sboxStart + package.blocks.length * 150 ≤ package.width)
    (covered : ∀ column, LogicalPlan.publicCount ≤ column →
      column < package.sboxStart + package.blocks.length * 150 →
        deadIn package.blocks column = false →
          ∃ slot ∈ runSlots package.width package.slots, slot.source.val = column)
    (fits : LogicalPlan.Fits package.relationRows package.width)
    (rowCount_le : package.blocks.length * 150 ≤ 2 ^ cubeVariables)
    (honest : Env) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) honest = encHash digest)
    (nativeHold : R1CS.RowsHold honest nativeRows)
    (nativeTemplates : ∀ block ∈ nativeBlocks, ConstraintsHold honest
      ((Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ())))
    (restHolds : ∀ piece ∈ rest,
      ConstraintsHold honest (piece.constraints (PilotData.canonicalConstraints ()))) :
    (LogicalPlan.plan package.relationRows package.width fits
        (layout (runSlots package.width package.slots))).RowsZero
        (packageAssignment package (PieceRowsCompleteness.fillPieces
          (PilotData.canonicalConstraints ()) honest (CircuitRelation.otherPieces rest)
          package.firstFresh)) ∧
      (PoseidonSboxFamilyPlan.plan (CircuitRelation.sboxInterface
          (layout (runSlots package.width package.slots)) package.blocks package.sboxStart)
        rowCount_le).RowsZero (packageAssignment package (PieceRowsCompleteness.fillPieces
          (PilotData.canonicalConstraints ()) honest (CircuitRelation.otherPieces rest)
          package.firstFresh)) ∧
      LogicalPlan.publicInput (publicFits := publicFits) (LogicalPlan.sourceEnv
        (layout (runSlots package.width package.slots)) (packageAssignment package
          (PieceRowsCompleteness.fillPieces (PilotData.canonicalConstraints ()) honest
            (CircuitRelation.otherPieces rest) package.firstFresh))) = encHash digest := by
  let canonical := PilotData.canonicalConstraints ()
  let filled := PieceRowsCompleteness.fillPieces canonical honest (CircuitRelation.otherPieces rest)
    package.firstFresh
  have blocksEq := native_blocks piecesEq
  have rowsEq := native_rows piecesEq
  have nextLe : package.firstFresh ≤ package.sboxStart :=
    le_trans (PieceRowsCompleteness.lowerPiecesAux_le canonical (CircuitRelation.otherPieces rest)
      package.firstFresh []) restEnd
  have othersValid : ∀ piece ∈ CircuitRelation.otherPieces rest, piece.Valid :=
    fun piece member => restValid piece (CircuitRelationCompleteness.otherPieces_mem member).1
  have othersBelow : ∀ piece ∈ CircuitRelation.otherPieces rest,
      ∀ expression ∈ piece.constraints canonical, expression.VarsBelow package.firstFresh :=
    fun piece member => restBelow piece (CircuitRelationCompleteness.otherPieces_mem member).1
  have othersHold : ∀ piece ∈ CircuitRelation.otherPieces rest,
      ConstraintsHold honest (piece.constraints canonical) :=
    fun piece member => restHolds piece (CircuitRelationCompleteness.otherPieces_mem member).1
  obtain ⟨loweredRows, agree⟩ := PieceRowsCompleteness.lowerPieces_complete canonical
    (CircuitRelation.otherPieces rest) package.firstFresh honest othersValid othersBelow othersHold
  have low : ∀ column, column < package.firstFresh → filled column = honest column :=
    fun column below => agree column (Or.inl below)
  have allBelow : ∀ block ∈ package.blocks, ∀ expression ∈
      (Piece.permutation block.start block.lanes).constraints canonical,
      expression.VarsBelow package.firstFresh := by
    intro block member
    rw [blocksEq] at member
    rcases List.mem_append.mp member with native | other
    · exact nativeBlocksBelow block native
    · exact restBelow _ (CircuitRelationCompleteness.blocksOf_mem other)
  refine package_complete package frame ?_ ?_
    (fun block member expression inside =>
      Expr.VarsBelow.mono _ (allBelow block member expression inside) nextLe)
    (le_trans publicBelow nextLe) sboxFits covered fits rowCount_le filled digest ?_ ?_ ?_
  · rw [rowsEq, List.all_append, Bool.and_eq_true]
    refine ⟨nativeAvoid, List.all_eq_true.mpr ?_⟩
    exact CircuitRelationCompleteness.lowerPiecesAux_avoid
      (fun column dead => dead_below allBelow dead) (CircuitRelation.otherPieces rest)
      package.firstFresh [] (le_refl _)
      (fun piece member => ⟨(CircuitRelationCompleteness.otherPieces_mem member).2,
        restAvoid piece member⟩)
      (fun _ member => by simp at member)
  · intro row member
    rw [rowsEq] at member
    rcases List.mem_append.mp member with native | lowered
    · exact R1CS.Row.VarsBelow.mono row (nativeBelow row native) nextLe
    · exact R1CS.Row.VarsBelow.mono row (PieceRowsCompleteness.lowerPieces_rows_below canonical _
        package.firstFresh othersBelow row lowered) restEnd
  · rw [← encoded]
    funext column
    exact low column.val (lt_of_lt_of_le column.isLt publicBelow)
  · rw [rowsEq]
    intro row member
    rcases List.mem_append.mp member with native | lowered
    · exact R1CS.rowsHold_of_agree_below nativeRows package.firstFresh honest filled nativeBelow
        low nativeHold row native
    · exact loweredRows row lowered
  · intro block member
    apply constraintsHold_of_agree_below honest filled _ package.firstFresh (allBelow block member)
      low
    rw [blocksEq] at member
    rcases List.mem_append.mp member with native | other
    · exact nativeTemplates block native
    · exact restHolds _ (CircuitRelationCompleteness.blocksOf_mem other)

end NightstreamFPrime.Export.OptionB.StepPackageCompleteness
