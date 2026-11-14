import NightstreamFPrime.Spec.Compression.Rust.Error
import Mathlib.Data.Nat.Log

/-! Rust `Layout::new` (`crates/ns-compress/src/layout.rs`, SPEC.md Algorithm 9, step 3) as a
function (`layoutNew`). The witness slices of the claims with a table come first, the larger first,
then the other witness slices, then the count slices of the tables in claim order. Each slice
starts at the next multiple of its size. `W` and `V` take the first row of the schedule file of
the key whose `W` holds every cell and whose `V` holds the inverses of the span.

Rust tests compare `layoutNew` with `Layout::new` on the same inputs
(`crates/ns-compress/tests/unit/layout.rs`). -/

namespace NightstreamFPrime.Spec.Compression.Rust

open NightstreamFPrime.Spec.Compression.Verifier (Slice)

/-- A claim of `Layout::new`: its ring blocks, and the bound of its table (`Norm::Table`) or
`none` (`Norm::Caller`). -/
structure Claim where
  blocks : ℕ
  table : Option ℕ

instance : Inhabited Claim := ⟨⟨0, none⟩⟩

/-- Rust `LANE_VARS`, the variables of the lanes of a ring block. -/
def laneVars : ℕ := 6

/-- Rust `ROW_VARS`, the variables of the rows of the key polynomial. -/
def rowVars : ℕ := 5

/-- Rust `LOOKUP_VARS`, the variables of the columns of `V`. -/
def lookupVars : ℕ := 2

/-- Rust `vars_for`: the variables of the least power of two at or above `cells`. -/
def varsFor (cells : ℕ) : ℕ := Nat.clog 2 cells

/-- One schedule file (Rust `akita::Schedules`): the variables of the key polynomial, and the
variables of `W` and of `V` of each row, in increasing order. -/
structure Schedule where
  keyVars : ℕ
  rows : List (ℕ × ℕ)

/-- The blocks of each key row that the key polynomial holds. -/
def Schedule.blocks (s : Schedule) : ℕ := 2 ^ (s.keyVars - laneVars - rowVars)

/-- Rust `akita::SCHEDULES`. A Rust test compares the two tables. -/
def schedules : List Schedule := [⟨20, [(16, 16)]⟩, ⟨30, [(26, 28), (27, 28)]⟩]

/-- Rust `layout::Cells`. -/
structure Cells where
  witness : Slice
  blocks : ℕ
  table : Option (ℕ × Slice)

/-- Rust `Layout`. -/
structure Placed where
  cells : List Cells
  keyVars : ℕ
  witnessVars : ℕ
  witnessLen : ℕ
  spanVars : ℕ
  spanLen : ℕ
  inverseVars : ℕ

/-- The errors of `Layout::new`, in the order in which it checks them. -/
inductive LayoutError where
  | noKey
  | emptyClaim
  | claimAboveKey
  | noTable
  | tableWide
  | noRow
  deriving DecidableEq, Repr

/-- The name of an error in the parity file. -/
def LayoutError.name : LayoutError → String
  | .noKey => "no_key"
  | .emptyClaim => "empty_claim"
  | .claimAboveKey => "claim_above_key"
  | .noTable => "no_table"
  | .tableWide => "table_wide"
  | .noRow => "no_row"

/-- The least multiple of `size` at or above `cursor` (Rust `usize::next_multiple_of`). -/
def nextMultiple (cursor size : ℕ) : ℕ := (cursor + size - 1) / size * size

/-- Rust `place`: `2^vars` cells at the next multiple of `2^vars` from `cursor`, and the cursor
after them. -/
def place (cursor vars : ℕ) : Slice × ℕ :=
  let offset := nextMultiple cursor (2 ^ vars)
  (⟨offset, vars⟩, offset + 2 ^ vars)

/-- Place each `(claim, vars)` in order from `cursor`: the slices and the cursor after them. -/
def placeAll : ℕ → List (ℕ × ℕ) → List (ℕ × Slice) × ℕ
  | cursor, [] => ([], cursor)
  | cursor, (claim, vars) :: rest =>
      ((claim, (place cursor vars).1) :: (placeAll (place cursor vars).2 rest).1,
        (placeAll (place cursor vars).2 rest).2)

/-- The variables of the witness slice of claim `i`. -/
def witnessVarsOf (claims : List Claim) (i : ℕ) : ℕ :=
  varsFor (claims.getD i default).blocks + laneVars

/-- The claims whose table is `table?`, the larger first and stable (Rust `sort_by_key`). -/
def groupOrder (claims : List Claim) (table? : Bool) : List ℕ :=
  ((List.range claims.length).filter fun i => (claims.getD i default).table.isSome == table?).mergeSort
    fun a b => decide (varsFor (claims.getD b default).blocks ≤ varsFor (claims.getD a default).blocks)

/-- The count slices: the claims with a table, in claim order, with the variables of their tables. -/
def countRequests (claims : List Claim) : List (ℕ × ℕ) :=
  (List.range claims.length).filterMap fun i =>
    (claims.getD i default).table.map fun bound => (i, varsFor (2 * bound + 1))

/-- The witness slices of the claims with a table, from cell 0, and the cursor after them: the end
of the span. -/
def tableSlices (claims : List Claim) : List (ℕ × Slice) × ℕ :=
  placeAll 0 ((groupOrder claims true).map fun i => (i, witnessVarsOf claims i))

/-- The witness slices of the other claims, after the span. -/
def otherSlices (claims : List Claim) : List (ℕ × Slice) × ℕ :=
  placeAll (tableSlices claims).2 ((groupOrder claims false).map fun i => (i, witnessVarsOf claims i))

/-- The count slices, after the witness slices, and the cursor after them: the cells of `W`. -/
def countSlices (claims : List Claim) : List (ℕ × Slice) × ℕ :=
  placeAll (otherSlices claims).2 (countRequests claims)

/-- Rust `Cells` of claim `i`. -/
def cellOf (claims : List Claim) (i : ℕ) : Cells where
  witness := (((tableSlices claims).1 ++ (otherSlices claims).1).lookup i).getD ⟨0, 0⟩
  blocks := (claims.getD i default).blocks
  table := (claims.getD i default).table.map fun bound =>
    (bound, ((countSlices claims).1.lookup i).getD ⟨0, 0⟩)

/-- A schedule row whose `W` holds every cell and whose `V` holds the inverses of the span. -/
def rowFits (claims : List Claim) (row : ℕ × ℕ) : Bool :=
  decide (varsFor (countSlices claims).2 ≤ row.1) &&
    decide (varsFor (tableSlices claims).2 + lookupVars ≤ row.2)

/-- The layout of `claims` with the key of `schedule` and the variables `row` of `W` and `V`. -/
def assemble (claims : List Claim) (schedule : Schedule) (row : ℕ × ℕ) : Placed where
  cells := (List.range claims.length).map (cellOf claims)
  keyVars := schedule.keyVars
  witnessVars := row.1
  witnessLen := (countSlices claims).2
  spanVars := varsFor (tableSlices claims).2
  spanLen := (tableSlices claims).2
  inverseVars := row.2

/-- Rust `Layout::new` for `claims` under a key polynomial of `keyBlocks` blocks. -/
def layoutNew (claims : List Claim) (keyBlocks : ℕ) : Except LayoutError Placed :=
  match schedules.find? (fun s => s.blocks == keyBlocks) with
  | none => .error .noKey
  | some schedule =>
    if claims.any (fun c => c.blocks == 0) then .error .emptyClaim
    else if claims.any (fun c => keyBlocks < c.blocks) then .error .claimAboveKey
    else if !claims.any (fun c => c.table.isSome) then .error .noTable
    else if claims.any (fun c => c.table.any fun bound => 65535 < 2 * bound + 1) then
      .error .tableWide
    else match schedule.rows.find? (rowFits claims) with
      | none => .error .noRow
      | some row => .ok (assemble claims schedule row)

/-- `Production.Layout` of a layout of the three production claims. -/
def Placed.production (l : Placed) : Production.Layout where
  witness c := (l.cells.getD c ⟨⟨0, 0⟩, 0, none⟩).witness
  counts c := ((l.cells.getD c ⟨⟨0, 0⟩, 0, none⟩).table.map Prod.snd).getD ⟨0, 0⟩
  witnessVars := l.witnessVars
  spanVars := l.spanVars
  keyBlockVars := l.keyVars - laneVars - rowVars

/-- The counts that Rust `error_bits` reads from a layout, with `m` public cells and `t`
matrices. -/
def Placed.counts (l : Placed) (publicCells matrices : ℕ) : RunCounts where
  tables := l.cells.filterMap fun c => c.table.map fun t => (c.witness.vars, t.1)
  claims := l.cells.length
  publicCells := publicCells
  matrices := matrices
  witnessVars := l.witnessVars
  keyBlockVars := l.keyVars - laneVars - rowVars

/-- The claims of `Production.params`: claim `c` has `blocks c` ring blocks, and slot `s` has the
table of its depth. -/
def paramsClaims (blocks : Fin 3 → ℕ) (depths : Fin 2 → ℕ) : List Claim :=
  [⟨blocks 0, some (Production.bound depths 0)⟩, ⟨blocks 1, some (Production.bound depths 1)⟩,
    ⟨blocks 2, none⟩]

/-- The production claims: slot 0 and slot 1 with the tables of their depths, then the fresh
instance of circuit `pc − 1`, for relations of `blocks 0` and `blocks 1` ring blocks (Rust
`BoundedDepth::compression_claims`). -/
def productionClaims (blocks : Fin 2 → ℕ) (pc : Fin 2) (depths : Fin 2 → ℕ) : List Claim :=
  paramsClaims ![blocks 0, blocks 1, blocks pc] depths

/-- Rust `key_blocks`: the blocks of the smallest key polynomial that holds `blocks` blocks. -/
def keyBlocksFor (blocks : ℕ) : Option ℕ :=
  (schedules.find? fun s => decide (blocks ≤ s.blocks)).map Schedule.blocks

end NightstreamFPrime.Spec.Compression.Rust
