import CompressionLean.Layer1.Model
import CompressionLean.Fields
import NightstreamFPrime.Spec.AjtaiSetupV1
import Mathlib.Data.List.Sort

/-!
Owns the Lean port of Rust's layer-1 setup of the F′ circuit: `build` is
`neo_spartan::Setup::build` (`lib.rs`, `key` and `oracle`) over
`Tables::from_rows` (`matrix.rs`), from the runs that
`MatrixRows::visit_rows` pushes (`Push`). `Final/RustSetup.lean` proves what
it builds.
-/

namespace CompressionLean.Setup

open NightstreamFPrime NightstreamFPrime.Spec

/-! ## The pushed runs -/

/-- One `MatrixRowSink::push_run`: the row, the matrix and the
`GeometricRowRun`. The run gives column `start + t`, `t < len`, the value
`initial·ratio^t`. -/
structure Push where
  row : ℕ
  matrix : ℕ
  start : ℕ
  len : ℕ
  initial : Gl
  ratio : Gl

namespace Push

/-- The padding run of the setup columns: all zero. -/
def pad : Push := ⟨0, 0, 0, 0, 0, 0⟩

/-- `Class::ratio`: the canonical ratio, and one for a run of one column. -/
def ratioWord (p : Push) : ℕ := if p.len = 1 then 1 else p.ratio.val

/-- `Class` as one number. Its order is the derived order of `(len, ratio)`. -/
def classCode (p : Push) : ℕ := p.len * 2 ^ 64 + p.ratioWord

end Push

/-! ## The port of `Tables::from_rows` and `key` -/

/-- Sorted distinct values: a `BTreeSet`, or a sort and a `dedup`. -/
def sortedSet (l : List ℕ) : List ℕ := l.dedup.insertionSort (· ≤ ·)

/-- `Collect` keeps the runs of each matrix in visit order, and `from_rows`
puts the matrices in order. -/
def runList (matrices : ℕ) (pushes : List Push) : List Push :=
  (List.range matrices).flatMap fun j => pushes.filter fun p => p.matrix = j

/-- The run classes, sorted. -/
def classesOf (runs : List Push) : List ℕ := sortedSet (runs.map Push.classCode)

/-- The slot key of a run: its class, lane and block in one word, class high.
Rust packs it in a `u64` as `class << 40 | lane << 32 | block`. The port is
that word while the blocks are below `2^32` and the classes at most `2^24`;
`from_rows` refuses wider keys. -/
def slotKey (classes : List ℕ) (p : Push) : ℕ :=
  classes.idxOf p.classCode * 2 ^ 40 + p.start % ringDegree * 2 ^ 32 + p.start / ringDegree

/-- The slots, sorted. -/
def keysOf (runs : List Push) : List ℕ := sortedSet (runs.map (slotKey (classesOf runs)))

/-- The start column of slot `σ`: its block times `D` plus its lane. -/
def slotStart (keys : List ℕ) (σ : ℕ) : ℕ :=
  keys.getD σ 0 % 2 ^ 32 * ringDegree + keys.getD σ 0 / 2 ^ 32 % 2 ^ 8

/-- The class of slot `σ`. -/
def slotClass (runs : List Push) (σ : ℕ) : ℕ :=
  (classesOf runs).getD ((keysOf runs).getD σ 0 / 2 ^ 40) 0

/-- Key row `i` on the cube (`oracle`): coefficient `l` of the seeded Ajtai
key element `(i, b)` at `2^6·b + l` for `l < 54` and `b < blocks`, and zero
elsewhere. -/
def seedKey (seed : List ℕ) (blocks i x : ℕ) : Gl :=
  if x % 64 < ringDegree ∧ x / 64 < blocks then (AjtaiSetupV1.wideCoefficientNat seed i (x / 64) (x % 64) : Gl)
  else 0

/-- The setup that `Setup::build` writes for the runs in setup order: the key
rows, the run columns (row, slot, coefficient, zero past the runs) and per
slot its block, lane, length and ratio. -/
def setupOfRuns (seed : List ℕ) (blocks : ℕ) (runs : List Push) : Layer1.Setup Gl where
  key := seedKey seed blocks
  runRow k := (runs.getD k Push.pad).row
  runMatrix k := (runs.getD k Push.pad).matrix
  runSlot k := if k < runs.length then (keysOf runs).idxOf (slotKey (classesOf runs) (runs.getD k Push.pad)) else 0
  runCoef k := (runs.getD k Push.pad).initial
  runs := runs.length
  slotBlock σ := slotStart (keysOf runs) σ / ringDegree
  slotLane σ := slotStart (keysOf runs) σ % ringDegree
  slotLength σ := slotClass runs σ / 2 ^ 64
  slotRatio σ := ((slotClass runs σ % 2 ^ 64 : ℕ) : Gl)
  slots := (keysOf runs).length

/-- The checks of `from_rows` and `key` on the columns and the runs: whole
blocks, a matrix, one to `MAX_MESSAGE_COLUMNS` blocks, and runs of a matrix
in range, no longer than a block, that end inside the columns. -/
def Accepts (columns matrices : ℕ) (pushes : List Push) : Prop :=
  columns % ringDegree = 0 ∧ 0 < matrices ∧ 0 < columns / ringDegree ∧ columns / ringDegree ≤ 4708530 ∧
    ∀ p ∈ pushes, p.matrix < matrices ∧ p.len ≤ ringDegree ∧ p.start + p.len ≤ columns

instance (columns matrices : ℕ) (pushes : List Push) : Decidable (Accepts columns matrices pushes) := by
  unfold Accepts
  infer_instance

/-- `Setup::build` over `columns` columns and `matrices` matrices: the setup,
or `none` where Rust stops with an error. -/
def build (seed : List ℕ) (columns matrices : ℕ) (pushes : List Push) : Option (Layer1.Setup Gl) :=
  if Accepts columns matrices pushes then some (setupOfRuns seed (columns / ringDegree) (runList matrices pushes))
  else none

end CompressionLean.Setup
