import NightstreamFPrime.Export.OptionB.StepPackage

/-! Parity data for Rust's expansion of the step packages (CW-7.1). A circuit's plan
(`CircuitRelation.plan`) is one plan row per row of `Package.relationRows`, followed by 150 S-box
rows per block (`PoseidonSboxFamilyPlan.plan`). Rust expands the relation rows itself: each
package piece, the ring rows and the binding rows. `summary` gives their count and a checksum that
mixes every entry with its row, port and column, so a Rust test that recomputes the checksum from
its own expansion compares the two expansions entry by entry. Rust builds the S-box rows from the
shared formula library (`SharedFormulas.poseidonComponent`), as today's production relation does;
Lean does not expand them here, because `SparseLayer` does not merge terms and the partial rounds
grow each form sixteenfold. The probe packages check Rust's identity code against
`StepPackage.identityFast`. The witness parity (CW-7.2) is below. -/

namespace NightstreamFPrime.Export.OptionB.Parity

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Generic

/-- A field element that depends on the plan row, the port and the column. -/
def mix (unit column : Nat) : F :=
  let point : F := Poseidon2.ofNat (unit * 16777259 + column + 1)
  point * point * point + Poseidon2.ofNat unit

def formSum {width : Nat} (unit : Nat) (form : SparseForm width) (total : F) : F :=
  form.entries.foldl (fun sum entry => sum + entry.coefficient * mix unit entry.column.val) total

/-- An R1CS row's plan ports: `a`, `b` and `c` as `formOf` gives them. -/
def ordinarySum (width : Nat) (rows : List R1CS.Row) (total : F) : F :=
  (rows.zipIdx).foldl (fun sum (row, index) =>
    let sum := formSum (4 * index) (CircuitRelation.formOf (width := width) row.a) sum
    let sum := formSum (4 * index + 1) (CircuitRelation.formOf (width := width) row.b) sum
    formSum (4 * index + 2) (CircuitRelation.formOf (width := width) row.c) sum) total

/-! ### The witness parity (CW-7.2)

Rust fills a circuit's columns with the package's witness program and computes the native ranges
itself: a block's outputs by the permutation of its lanes, a ring piece's outputs by the ring
product. Lean fills them with every witness batch of the circuit, so the two compare when the shared
columns agree. The shared columns are fixed pseudo-random words; column 0 is the constant 1. A
block's 1,080 dead locals are left out, because Rust leaves them 0. -/

def sharedWord (column : Nat) : F :=
  if column = 0 then 1 else
    Poseidon2.ofNat ((column * 6364136223846793005 + 1442695040888963407) % goldilocksModulus)

def runRecipes (values : Array F) (start : Nat) : List Expr → Array F
  | [] => values
  | recipe :: rest =>
      let value := recipe.eval (fun column => values.getD column 0)
      runRecipes (values.set! start value) (start + 1) rest

def runHints (values : Array F) (start : Nat) : List Hint → Array F
  | [] => values
  | hint :: rest =>
      let value := hint.eval (fun column => values.getD column 0)
      runHints (values.set! start value) (start + 1) rest

/-- `WitnessEncoding.executeBatch` for each batch, over an array of `size` columns. -/
def runBatches (sharedEnd size : Nat) (batches : List WitnessBatch) : Array F :=
  batches.foldl (fun values batch =>
      runHints (runRecipes values batch.start batch.recipes)
        (batch.start + batch.recipes.length) batch.hints)
    (Array.ofFn (n := size) fun column =>
      if column.val < sharedEnd then sharedWord column.val else 0)

def deadMask (size : Nat) (blocks : List Frame.Block) : Array Bool :=
  blocks.foldl (fun mask block =>
      (List.range 1080).foldl (fun mask offset => mask.set! (block.start + offset) true) mask)
    (Array.replicate size false)

/-- Every live column's value times `mix 0 column`. -/
def witnessSum (values : Array F) (dead : Array Bool) : F :=
  (List.range values.size).foldl (fun sum column =>
    if dead.getD column false then sum else sum + values.getD column 0 * mix 0 column) 0

/-- The logical assignment's checksum (CW-5.11): the public columns, then each slot's digits at
its logical column (`ProductionAssignment.logicalAssignment`), each value times
`mix 1 column`. Only slots below `limit` add, because the parity witness stops at the first
fresh column; every slot moves the logical column. -/
def logicalSum (values : Array F) (runs : List (Nat × Nat × Nat)) (limit : Nat) : F :=
  let publicPart := (List.range 270).foldl (fun sum column =>
    sum + values.getD column 0 * mix 1 column) 0
  (runs.foldl (fun (sum, offset) (start, count, kind) =>
      (List.range count).foldl (fun (sum, offset) index =>
          let column := start + index
          let digits := LowNormSlot.encode (SlotLayout.kindOf kind) (values.getD column 0)
          if column < limit then
            digits.foldl (fun (sum, offset) digit => (sum + digit * mix 1 offset, offset + 1))
              (sum, offset)
          else (sum, offset + digits.length))
        (sum, offset))
    (publicPart, 270)).1

/-- A circuit whose blocks pack: its relation rows, its S-box rows, the checksum of its relation
rows, the end of its shared columns, the checksum of its witness, and the checksum of its logical
assignment. -/
def summary (width : Nat) (pieces : List Pieces.Piece) (next sharedEnd : Nat)
    (batches : List WitnessBatch) : Value :=
  let rows := CircuitRelation.relationRows CircuitRelation.publicColumns pieces next width
  let blocks := CircuitRelation.blocksOf pieces
  let values := runBatches sharedEnd next batches
  let witness := witnessSum values (deadMask next blocks)
  let logical := logicalSum values
    (StepPackage.slotRuns CircuitRelation.publicColumns pieces next batches) next
  .array [.atom rows.length, .atom (150 * blocks.length), .atom (ordinarySum width rows 0).val,
    .atom sharedEnd, .atom witness.val, .atom logical.val]

/-- Two small packages whose identity Lean computes (`StepPackage.identityFast`): a probe for
Rust's identity code. -/
def probeLight : StepPackage.Package :=
  ⟨0, 270, 16, 300, 400,
    [.rows [⟨⟨5, [(1, 7)]⟩, .one, .zero⟩], .block 2 [.var 3, .const 9]],
    [⟨301, [.add (.var 1) (.const 2)], [.inverseOrZero (.var 4)]⟩], [(270, 3, 2), (273, 1, 0)]⟩

def probeRefresh : StepPackage.Package :=
  ⟨1, 270, 32, 10, 20, [.ring 1 4 30 [.mul (.const 3) (.var 8)]], [], []⟩

/-- The parity file: its version, both circuits' summaries, and the probe packages with their
identity. -/
def parity (width : Nat) (light refresh : List Pieces.Piece × Nat × Nat × List WitnessBatch) :
    Value :=
  .array [.atom 3, summary width light.1 light.2.1 light.2.2.1 light.2.2.2,
    summary width refresh.1 refresh.2.1 refresh.2.2.1 refresh.2.2.2,
    StepPackage.identityValue probeLight probeRefresh,
    .array ((StepPackage.identityFast probeLight probeRefresh).map fun word => .atom word.val)]

end NightstreamFPrime.Export.OptionB.Parity
