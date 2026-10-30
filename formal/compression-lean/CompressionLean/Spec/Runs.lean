import CompressionLean.Spec.Parent
import CompressionLean.Spec.PoseidonRows
import CompressionLean.Production
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache
import NightstreamFPrime.Export.Stage1.PiDECProductRow

namespace CompressionLean.Spec

open CompressionLean.Setup CompressionLean.Spec.PoseidonRows
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.MatrixProgram

def matchRun {n : ℕ} (r : Gl) : ℕ → ℕ → Gl → List (SparseEntry n) → Option (List (SparseEntry n))
  | _, 0, _, es => some es
  | c, k + 1, v, e :: es => if e.column.val = c ∧ gl e.coefficient = v then matchRun r (c + 1) k (v * r) es else none
  | _, _ + 1, _, [] => none

def matchRuns {n : ℕ} : List Push → List (SparseEntry n) → Bool
  | [], es => es.isEmpty
  | p :: ps, es =>
      match matchRun p.ratio p.start p.len p.initial es with
      | some rest => matchRuns ps rest
      | none => false

def entriesNormal {n : ℕ} : List (SparseEntry n) → Bool
  | [] => true
  | [e] => e.coefficient != 0
  | e :: f :: rest => e.coefficient != 0 && e.column.val < f.column.val && entriesNormal (f :: rest)

def normalEntries {n : ℕ} (form : SparseForm n) : List (SparseEntry n) :=
  if entriesNormal form.entries then form.entries else PoseidonRows.normalize form

def rowFailure {n : ℕ} (forms : RowForms n) (runs : List Push) : Option ℕ :=
  (List.finRange Spec.ProductionRelation.meaningfulPortCount).findSome? fun j =>
    if matchRuns (runs.filter (·.matrix = j.val)) (normalEntries (forms j)) then none else some j.val

def unitSize : MatrixProgram.Block → ℕ
  | .poseidon _ => 150
  | .phi81Product _ => 108
  | _ => 1

def unitCount : MatrixProgram.Block → ℕ
  | .poseidon block => block.invocationCount
  | .phi81Product block => Phi81Product.ringCount block.families
  | block => block.rowCount

def productRows (block : Phi81Product.Block) (n g : ℕ) : Option (Array (RowForms n)) := do
  let descriptor ← Phi81Product.ringDescriptor? block.families g
  let interface ← PiDECProductInterface.interface? block n descriptor
  pure ((Array.ofFn (Phi81ProductPlan.rowAt interface)).map (·.meaningfulForm))

def unitRows (n : ℕ) (sourceRow : ℕ → Option R1CS.Row) : MatrixProgram.Block → ℕ → Option (Array (RowForms n))
  | .poseidon block, u => (invocationRows? block n u).map fun rows => rows.map (·.meaningfulForm)
  | .phi81Product block, u => productRows block n u
  | block, u => (block.row? n sourceRow u).map fun forms => #[forms]

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

def unitJobs : ℕ → List MatrixProgram.Block → List (MatrixProgram.Block × ℕ × ℕ)
  | _, [] => []
  | start, block :: rest =>
      (List.range (unitCount block)).map (fun u => (block, start, u)) ++ unitJobs (start + block.rowCount) rest

def runsByRow (rows : ℕ) (pushes : List Push) : Array (List Push) :=
  pushes.foldl (fun byRow p => byRow.modify p.row (p :: ·)) (Array.replicate rows [])

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
