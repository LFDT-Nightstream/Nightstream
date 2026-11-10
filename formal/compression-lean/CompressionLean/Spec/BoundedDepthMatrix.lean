import CompressionLean.Spec.Runs
import CompressionLean.Spec.Checks
import NightstreamFPrime.Export.BoundedDepth.StepPackage
import NightstreamFPrime.Export.Generic.SlotLayout
import NightstreamFPrime.Lifecycle.Nebula.FactorWords
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import NightstreamFPrime.Spec.Nebula

/-! The matrices of the bounded-depth scheme's circuit keys (CW-6.5). A circuit key's relation is its circuit's rows
read through a layout at the key width, then 150 S-box rows for each Poseidon2 block
(`keyRelation`, `CircuitRelation.plan`). The layout is the step package's slot layout, widened to
the key width (`keyLayout`): a key with fewer columns has a witness that is zero past them.

`keyRunsFailure` compares Rust's runs of a layer-1 relation with these matrices row by row. It
finds the slot of each source column by a binary search over the slots' sources and confirms the
slot at the position it finds (`slotAt`), so its result does not rest on the search. -/

namespace CompressionLean.Spec.BoundedDepth

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.BoundedDepth
open CompressionLean.Setup (Push)

/-! ### The circuit key's relation -/

/-- A form over more logical columns, with the same entries. -/
def widenForm {L L' : ℕ} (le : L ≤ L') (form : SparseForm L) : SparseForm L' :=
  ⟨form.entries.map fun entry => ⟨Fin.castLE le entry.column, entry.coefficient⟩⟩

/-- A layout over more logical columns. -/
def widen {width L L' : ℕ} (layout : LogicalPlan.Layout width L) (le : L ≤ L') :
    LogicalPlan.Layout width L' where
  map := ⟨fun column => widenForm le (layout.map.form column)⟩
  publicFits := le_trans layout.publicFits le
  publicForm := fun column bound => by
    show widenForm le (layout.map.form column) = _
    rw [layout.publicForm column bound]
    rfl

/-- A step package's slots, from its runs (`SlotLayout.runSlots`). -/
def slotsOf (p : StepPackage.Package) : List (LowNormAssignment.Slot p.width) :=
  SlotLayout.runSlots p.width p.slots

/-- A step package's slot layout at the key width. -/
def keyLayout (p : StepPackage.Package)
    (fits : ProductionAssignment.logicalWidth (slotsOf p) ≤ StepRelation.keyWidth) :
    LogicalPlan.Layout p.width StepRelation.keyWidth :=
  widen (SlotLayout.layout (slotsOf p)) fits

/-- A bounded-depth circuit key's relation at the key width. -/
def keyRelation (pieces : List Pieces.Piece) (next width : ℕ)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns pieces next width)
    (layout : LogicalPlan.Layout width StepRelation.keyWidth) :
    Lifecycle.ProductionKey.LogicalRelation StepRelation.keyWidth StepRelation.keyPublicFits :=
  (CircuitRelation.plan CircuitRelation.publicColumns pieces next width fits layout).logicalRelation
    (publicFits := StepRelation.keyPublicFits) StepRelation.keyCubeFits

/-- The bounded-depth scheme's circuit key for a step package: the digest and the relation of the circuit whose
pieces the package holds. -/
def circuitKey (digest : Lifecycle.KeyDigest) (p : StepPackage.Package) (pieces : List Pieces.Piece) (next : ℕ)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns pieces next p.width)
    (logical : ProductionAssignment.logicalWidth (slotsOf p) ≤ StepRelation.keyWidth) :
    Lifecycle.BoundedDepth.CircuitKey StepRelation.keyWidth StepRelation.keyPublicFits :=
  ⟨digest, keyRelation pieces next p.width fits (keyLayout p logical)⟩

/-- The layer-1 setup that Rust builds from a key's runs, over the key's own columns. -/
def keySetup (seed : List ℕ) (key : Circuit.Layer1Key) (pushes : List Push) : Layer1.Setup Gl :=
  setupAt seed (keyColumns key) pushes

/-! ### The check -/

/-- The constant column, logical column 0 (`LogicalPlan.Layout.one`). -/
def oneColumn : Fin StepRelation.keyWidth := ⟨0, by decide⟩

/-- The slots as an array, and the logical columns before each slot, after the public columns
(`LowNormAssignment.prefixWidth`). -/
structure SlotTable (width : ℕ) where
  slots : Array (LowNormAssignment.Slot width)
  offsets : Array ℕ

def tableOf {width : ℕ} (slots : List (LowNormAssignment.Slot width)) : SlotTable width :=
  ⟨slots.toArray, ((slots.map LowNormAssignment.Slot.width).scanl (· + ·) 0).toArray⟩

/-- Each slot's source is below the next slot's. -/
def sourcesIncrease {width : ℕ} (slots : Array (LowNormAssignment.Slot width)) : Bool :=
  (List.range (slots.size - 1)).all fun k =>
    match slots[k]?, slots[k + 1]? with
    | some left, some right => decide (left.source.val < right.source.val)
    | _, _ => false

/-- The first slot whose source is at least `c`, by binary search. -/
def lowerBound {width : ℕ} (slots : Array (LowNormAssignment.Slot width)) (c : ℕ) : ℕ :=
  go 64 0 slots.size
where
  go : ℕ → ℕ → ℕ → ℕ
    | 0, lo, _ => lo
    | fuel + 1, lo, hi =>
        if lo < hi then
          match slots[(lo + hi) / 2]? with
          | some slot =>
              if slot.source.val < c then go fuel ((lo + hi) / 2 + 1) hi else go fuel lo ((lo + hi) / 2)
          | none => lo
        else lo

/-- The slot before position `k` reads a column below `c`. -/
def belowAt {width : ℕ} (slots : Array (LowNormAssignment.Slot width)) (c : ℕ) : ℕ → Bool
  | 0 => true
  | k + 1 => match slots[k]? with
    | some slot => decide (slot.source.val < c)
    | none => false

/-- The slot of source column `c`: `some (some k)` when slot `k` reads `c`, `some none` when the
slots around the search's position show that no slot reads `c` (their sources increase), and
`none` when the position confirms neither. -/
def slotAt {width : ℕ} (slots : Array (LowNormAssignment.Slot width)) (c : ℕ) : Option (Option ℕ) :=
  let k := lowerBound slots c
  match slots[k]? with
  | some slot =>
      if slot.source.val = c then some (some k)
      else if belowAt slots c k && decide (c < slot.source.val) then some none else none
  | none => if belowAt slots c k && decide (k = slots.size) then some none else none

/-- The form of a slot whose first coordinate is logical column `publicCount + offset`
(`RetainedSlot.form`). -/
def slotForm (offset count : ℕ) (fits : LogicalPlan.publicCount + offset + count ≤ StepRelation.keyWidth) :
    SparseForm StepRelation.keyWidth :=
  RetainedSlot.recomposeForms (List.ofFn fun coordinate : Fin count =>
    SparseForm.singleton ⟨LogicalPlan.publicCount + offset + coordinate.val, by omega⟩ 1)

/-- The form of source column `c` at the key width, or `none` when the table does not confirm
its slot. -/
def fastForm {width : ℕ} (table : SlotTable width) (c : ℕ) : Option (SparseForm StepRelation.keyWidth) :=
  if inPublic : c < LogicalPlan.publicCount then
    some (SparseForm.singleton ⟨c, Nat.lt_of_lt_of_le inPublic (by decide)⟩ 1)
  else
    match slotAt table.slots c with
    | some (some k) =>
        match table.slots[k]?, table.offsets[k]? with
        | some slot, some offset =>
            if fits : LogicalPlan.publicCount + offset + slot.width ≤ StepRelation.keyWidth then
              some (slotForm offset slot.width fits)
            else none
        | _, _ => none
    | some none => some .empty
    | none => none

/-- The source map of the fast forms; a column whose slot the table does not confirm reads the
empty form, and the check refuses a row that reads it. -/
def fastMap {width : ℕ} (table : SlotTable width) : SourceCompiler.SourceMap width StepRelation.keyWidth :=
  ⟨fun column => (fastForm table column.val).getD .empty⟩

/-- A source column's fast form, and the empty form beyond the source width (`Layout.form`). -/
def fastFormAt {width : ℕ} (table : SlotTable width) (column : ℕ) : SparseForm StepRelation.keyWidth :=
  if column < width then (fastForm table column).getD .empty else .empty

/-- `Layout.combination` with the fast forms. -/
def fastCombination {width : ℕ} (table : SlotTable width) (combination : R1CS.LinearCombination) :
    SparseForm StepRelation.keyWidth :=
  .add (.singleton oneColumn combination.constant)
    (combination.terms.foldr (fun term rest => .add (.scale term.2 (fastFormAt table term.1)) rest) .empty)

/-- Every column in `columns` has a confirmed fast form. -/
def confirmed {width : ℕ} (table : SlotTable width) (columns : List ℕ) : Bool :=
  columns.all fun column => (fastForm table column).isSome

def termColumns (combination : R1CS.LinearCombination) : List ℕ :=
  combination.terms.map (·.1)

instance {width : ℕ} (row : R1CS.Row) : Decidable (SourceCompiler.RowBounded width row) :=
  have := SourceCompiler.combinationBoundedDecidable width
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

/-- Two runs of one matrix over the same columns with the same ratio. -/
def sameRun (left right : Push) : Prop :=
  left.matrix = right.matrix ∧ left.start = right.start ∧ left.len = right.len ∧ left.ratio = right.ratio

instance (left right : Push) : Decidable (sameRun left right) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

/-- Runs by matrix, then start, length and ratio. -/
def runBefore (left right : Push) : Bool :=
  left.matrix < right.matrix || (left.matrix == right.matrix && (left.start < right.start ||
    (left.start == right.start && (left.len < right.len || (left.len == right.len && left.ratio.val ≤ right.ratio.val)))))

/-- Each run of equal neighbours as one run: Rust pushes one run for each term of a row, and a row
that reads a source column twice has two equal runs. -/
def mergeRuns (current : Push) : List Push → List Push
  | [] => [current]
  | next :: rest =>
      if sameRun current next then mergeRuns { current with initial := current.initial + next.initial } rest
      else current :: mergeRuns next rest

/-- A row's runs as the normal form of its entries reads them: in column order, equal runs merged,
and no run that is zero. -/
def sortedRuns (runs : List Push) : List Push :=
  (match runs.mergeSort runBefore with
    | [] => []
    | first :: rest => mergeRuns first rest).filter (·.initial ≠ 0)

/-- An ordinary row's first failing matrix, or why its forms do not evaluate. -/
def ordinaryFailure {width : ℕ} (table : SlotTable width) (row : R1CS.Row) (runs : List Push) :
    Option String :=
  if bounded : SourceCompiler.RowBounded width row then
    if confirmed table (termColumns row.a ++ termColumns row.b ++ termColumns row.c) then
      (rowFailure (SourceCompiler.compileRow (fastMap table) oneColumn row bounded).meaningfulForm
        (sortedRuns runs)).map
        fun j => s!"matrix {j}"
    else some "a column's slot is not confirmed"
  else some "a term is past the source width"

/-- Block `block`'s atoms (`PoseidonRows.Atoms`): its input lanes and its S-box outputs as forms. -/
def blockAtoms {width : ℕ} (table : SlotTable width) (block : Frame.Block) (sboxStart index : ℕ) :
    PoseidonRows.Atoms StepRelation.keyWidth where
  inputs := Vector.ofFn fun lane =>
    PoseidonRows.normalize (fastCombination table (invocationInputCombination block.invocation lane.val).toR1CS)
  sbox := Vector.ofFn fun row =>
    (fastFormAt table (CircuitRelation.sboxColumn sboxStart index row.val)).entries
  one := oneColumn

/-- The source columns that a block's atoms read. -/
def blockColumns (block : Frame.Block) (sboxStart index : ℕ) : List ℕ :=
  (List.range 16).flatMap (fun lane => termColumns (invocationInputCombination block.invocation lane).toR1CS) ++
    (List.range 150).map (CircuitRelation.sboxColumn sboxStart index)

/-- A block's first failing row and matrix, or why its forms do not evaluate. -/
def blockFailure {width : ℕ} (table : SlotTable width) (block : Frame.Block) (sboxStart index : ℕ)
    (runs : ℕ → List Push) : Option String :=
  if confirmed table ((blockColumns block sboxStart index).filter (· < width)) then
    let rows := PoseidonRows.atomRows.map (blockAtoms table block sboxStart index).row
    (List.range 150).findSome? fun r =>
      (rowFailure (rows.getD r ⟨.empty, .empty⟩).meaningfulForm (sortedRuns (runs r))).map
        fun j => s!"S-box row {r}, matrix {j}"
  else some "a column's slot is not confirmed"

/-- The package's rows and blocks fit the cube and the source width (`CircuitRelation.Fits`). -/
def packageFits (p : StepPackage.Package) : Bool :=
  decide (p.relationRows.length + p.blocks.length * 150 ≤ 2 ^ Lifecycle.cubeVariables) &&
    decide (LogicalPlan.rowsMax p.relationRows < p.width) &&
    decide (p.sboxStart + p.blocks.length * 150 ≤ p.width) &&
    p.blocks.all fun block => (List.range 16).all fun lane =>
      (invocationInputCombination block.invocation lane).toR1CS.terms.all fun term => decide (term.1 < p.width)

/-- The first row where Rust's runs differ from the circuit key's matrices: each ordinary row of the
package, then 150 S-box rows for each block, and no runs past them. -/
def keyRunsFailure (p : StepPackage.Package) (pushes : List Push) : Option String :=
  if ProductionAssignment.logicalWidth (slotsOf p) ≤ StepRelation.keyWidth then
    let table := tableOf (slotsOf p)
    if packageFits p = false then some "the package does not fit the cube and its width"
    else if sourcesIncrease table.slots then
      let rows := p.relationRows
      let blocks := p.blocks
      let total := rows.length + 150 * blocks.length
      match pushes.find? (total ≤ ·.row) with
      | some push => some s!"row {push.row}, matrix {push.matrix}: past the relation's {total} rows"
      | none =>
          let byRow := runsByRow total pushes
          let runs (i : ℕ) := (byRow.getD i []).reverse
          let ordinary := rows.zipIdx.map fun (row, i) =>
            Task.spawn fun _ => (ordinaryFailure table row (runs i)).map (s!"row {i}, " ++ ·)
          let sbox := blocks.zipIdx.map fun (block, k) =>
            Task.spawn fun _ => (blockFailure table block p.sboxStart k
              (fun r => runs (rows.length + 150 * k + r))).map (s!"block {k}, " ++ ·)
          (ordinary ++ sbox).findSome? Task.get
    else some "the slots' sources do not increase"
  else some "the package's logical width exceeds the key width"

end CompressionLean.Spec.BoundedDepth
