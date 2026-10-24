import CompressionLean.Spec.Parent
import CompressionLean.Spec.PoseidonRows
import CompressionLean.Production
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache
import NightstreamFPrime.Export.Stage1.PiDECProductRow

/-!
Owns the run check (`runsFailure`): the runs that Rust pushes give F′'s
matrices of the Poseidon2 hash chain, row by row. `runsCheck` runs it on the
runs that Rust writes, and `Checks/Runs.lean` proves that a pass gives
`RunsGive`.

- No run names a row past F′'s rows.
- For each row of F′'s matrix program and each matrix, the runs of that row
  and matrix, in push order, give F′'s entries in column order
  (`matchRuns`): each run gives the next `len` entries, `initial·ratio^t` at
  column `start + t`. F′'s entries are sorted by column, one per column, none
  zero (`normalEntries`).

F′'s rows come from fast evaluators, unit by unit (`unitRows`): the 150 rows
of a Poseidon2 invocation (`PoseidonRows.invocationRows?`), the 108 rows of a
Phi81 product (`productRows`), and one row of another block by F′'s `row?`
over the stored source rows (`PiDECCanonicalSourceCache.stored`). One task
checks each unit.
-/

namespace CompressionLean.Spec

open CompressionLean.Setup CompressionLean.Spec.PoseidonRows
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.MatrixProgram

/-! ## A row against its runs -/

/-- `es` starts with the `k` entries `v·r^t` at column `c + t`, `t < k`;
the entries after them. -/
def matchRun {n : ℕ} (r : Gl) : ℕ → ℕ → Gl → List (SparseEntry n) → Option (List (SparseEntry n))
  | _, 0, _, es => some es
  | c, k + 1, v, e :: es => if e.column.val = c ∧ gl e.coefficient = v then matchRun r (c + 1) k (v * r) es else none
  | _, _ + 1, _, [] => none

/-- The runs, one after another, give exactly the entries `es`. -/
def matchRuns {n : ℕ} : List Push → List (SparseEntry n) → Bool
  | [], es => es.isEmpty
  | p :: ps, es =>
      match matchRun p.ratio p.start p.len p.initial es with
      | some rest => matchRuns ps rest
      | none => false

/-- The entries are sorted by column, one per column, and none is zero. -/
def entriesNormal {n : ℕ} : List (SparseEntry n) → Bool
  | [] => true
  | [e] => e.coefficient != 0
  | e :: f :: rest => e.coefficient != 0 && e.column.val < f.column.val && entriesNormal (f :: rest)

/-- The entries of a form sorted by column, one per column, none zero. -/
def normalEntries {n : ℕ} (form : SparseForm n) : List (SparseEntry n) :=
  if entriesNormal form.entries then form.entries else PoseidonRows.normalize form

/-- The first matrix of a row whose runs do not give the row's entries. -/
def rowFailure {n : ℕ} (forms : RowForms n) (runs : List Push) : Option ℕ :=
  (List.finRange Spec.ProductionRelation.meaningfulPortCount).findSome? fun j =>
    if matchRuns (runs.filter (·.matrix = j.val)) (normalEntries (forms j)) then none else some j.val

/-! ## F′'s rows by unit -/

/-- The rows of a unit of a block: a Poseidon2 invocation, a Phi81 product,
or one row of another block. -/
def unitSize : MatrixProgram.Block → ℕ
  | .poseidon _ => 150
  | .phi81Product _ => 108
  | _ => 1

/-- The units of a block. -/
def unitCount : MatrixProgram.Block → ℕ
  | .poseidon block => block.invocationCount
  | .phi81Product block => Phi81Product.ringCount block.families
  | block => block.rowCount

/-- The 108 rows of Phi81 product `g` (`Phi81Product.Block.row?`), with its
interface loaded once. -/
def productRows (block : Phi81Product.Block) (n g : ℕ) : Option (Array (RowForms n)) := do
  let descriptor ← Phi81Product.ringDescriptor? block.families g
  let interface ← PiDECProductInterface.interface? block n descriptor
  pure ((Array.ofFn (Phi81ProductPlan.rowAt interface)).map (·.meaningfulForm))

/-- The rows of unit `u` of a block at logical width `n`; `sourceRow` gives
the source rows of an ordinary block. -/
def unitRows (n : ℕ) (sourceRow : ℕ → Option R1CS.Row) : MatrixProgram.Block → ℕ → Option (Array (RowForms n))
  | .poseidon block, u => (invocationRows? block n u).map fun rows => rows.map (·.meaningfulForm)
  | .phi81Product block, u => productRows block n u
  | block, u => (block.row? n sourceRow u).map fun forms => #[forms]

/-- The check of unit `u` of a block whose first row is `start`: each row of
the unit against its runs (`byRow`, the last pushed first). -/
def unitFailure (n : ℕ) (sourceRow : ℕ → Option R1CS.Row) (byRow : Array (List Push))
    (job : MatrixProgram.Block × ℕ × ℕ) : Option String :=
  let (block, start, u) := job
  let first := start + unitSize block * u
  match unitRows n sourceRow block u with
  | none => some s!"row {first}: F′'s row does not evaluate"
  | some rows =>
      if rows.size = unitSize block then
        (List.range (unitSize block)).findSome? fun r =>
          (rowFailure (rows.getD r fun _ => .empty) (byRow.getD (first + r) []).reverse).map
            fun j => s!"row {first + r}, matrix {j}"
      else some s!"row {first}: F′'s unit has {rows.size} rows"

/-- One job per unit of each block: the block, its first row and the unit. -/
def unitJobs : ℕ → List MatrixProgram.Block → List (MatrixProgram.Block × ℕ × ℕ)
  | _, [] => []
  | start, block :: rest =>
      (List.range (unitCount block)).map (fun u => (block, start, u)) ++ unitJobs (start + block.rowCount) rest

/-- The runs of each row below `rows`, the last pushed first. -/
def runsByRow (rows : ℕ) (pushes : List Push) : Array (List Push) :=
  pushes.foldl (fun byRow p => byRow.modify p.row (p :: ·)) (Array.replicate rows [])

/-- **The run check.** The first row and matrix where the runs do not give
F′'s matrices of the Poseidon2 hash chain: a run past F′'s rows, or a row of
F′'s matrix program whose runs do not give its entries. Each unit is checked
in its own task. -/
def runsFailure (pushes : List Push) : Option String :=
  let application := Poseidon2HashChainV1Package.application
  let program := PerApplicationMatrixProgram.matrixProgram application
  match pushes.find? (program.rowCount ≤ ·.row) with
  | some p => some s!"row {p.row}, matrix {p.matrix}: past F′'s {program.rowCount} rows"
  | none =>
      -- `Production.poseidon2Width` is an abbreviation: inside the tasks, each task would compute it.
      let width := Production.poseidon2Width
      let source := PiDECCanonicalSourceCache.stored application
      let byRow := runsByRow program.rowCount pushes
      let tasks := (unitJobs 0 program.blocks).map fun job =>
        Task.spawn fun _ => unitFailure width (source[·]?) byRow job
      tasks.findSome? Task.get

end CompressionLean.Spec
