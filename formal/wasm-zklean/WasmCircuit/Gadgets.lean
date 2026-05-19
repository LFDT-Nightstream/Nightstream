-- Per-gadget intermediate representation for the wasm zkVM constraint export.
--
-- This file owns:
--   * `Instr` — a structured trace tag, one constructor per Rust gadget.
--   * `instrToRows` — closed-form lowering from `Instr` to flat R1CS rows.
--     Single source of truth for the row structure of every gadget. The
--     cross-check theorem in `Generated.lean`
--     (`trace_matches_actual := by native_decide`) fails at build time if it
--     ever drifts from the Rust gadget.
--   * `instrToBuilder` — closed-form lowering from `Instr` to a `ZKBuilder Fq`
--     fragment over zkLean's DSL. Derived mechanically from `instrToRows`
--     through a generic `rowToBuilder`, so it inherits "single source of
--     truth" status from `instrToRows`.
--
-- Soundness lemmas (one per `Instr` ctor) live in `WasmCircuit.Bridge` and
-- are stated against zkLean's `semantics`. This file deliberately does *not*
-- introduce a parallel row-form satisfaction predicate — zkLean's
-- `ConstrainR1CS` evaluator already covers that surface, and reasoning
-- against `semantics` directly keeps the proof story consistent with the
-- wider zkLean ecosystem.
--
-- Coefficients are `Int`: the wasm zkVM emits small integer literals, so
-- structural row equality is decidable without committing to a concrete
-- field at the data layer (Goldilocks `Fq` is only needed for the zkLean
-- bridge, defined separately in `Field.lean`).

import zkLean.AST
import zkLean.Builder
import WasmCircuit.Field

namespace WasmCircuit.Gadgets

/-- Column index of the constant-one witness column. Hardcoded here (not pulled
    from the generated `WasmCircuit.Columns`) so this module is buildable on a
    clean checkout without first running the Rust exporter. The exporter emits
    a `constOneCol_pinned` sanity check in `Generated.lean` that fails at
    `lake build` time if the Rust `COL_ONE` ever leaves index 0. -/
def constOneCol : Nat := 0

/-- Integer coefficients lifted into the working field via `IntCast`. -/
abbrev Coeff : Type := Int

/-- A sparse linear combination over witness columns: `(column_index, coeff)` pairs. -/
abbrev SparseRow : Type := List (Nat × Coeff)

/-- An R1CS row `(A, B, C)` representing the constraint `(A·w) * (B·w) = (C·w)`. -/
abbrev Row : Type := SparseRow × SparseRow × SparseRow

/-- Gadget trace tags. One constructor per Rust gadget that the exporter
    instruments. `Raw` is the escape hatch for rows that have not yet been
    promoted to a structured constructor. -/
inductive Instr where
  | ZeroTest (value invWitness isZero : Nat) : Instr
  | Raw      (a b c : SparseRow) : Instr
  deriving DecidableEq, Repr

/-- Closed-form lowering of a single instruction to its R1CS rows.
    The `ZeroTest` case mirrors `push_zero_test_gadget` in
    `crates/neo-fold-next/src/wasm/gadgets.rs`:

      row 1: `value · invWitness = 1 − isZero`
      row 2: `value · isZero     = 0`

    where the literal `1` is encoded as a coefficient on `constOneCol`. -/
def instrToRows : Instr → List Row
  | .ZeroTest v inv iz =>
      [ ([(v, 1)], [(inv, 1)], [(constOneCol, 1), (iz, -1)]),
        ([(v, 1)], [(iz, 1)], []) ]
  | .Raw a b c => [(a, b, c)]

/-! ## zkLean bridge

    `instrToBuilder` lowers an `Instr` into a `ZKBuilder Fq PUnit` fragment
    over zkLean's free-monad DSL. The concrete Goldilocks field
    (`WasmCircuit.Fq`, defined in `Field.lean`) is load-bearing: zkLean's
    `ZKExpr` and `ZKBuilder` are field-parametric, but constraint *evaluation*
    via `semantics` needs a `ZKField` instance, only provided for `Fq`.

    `Instr.Raw` is the escape hatch: gadgets that have not yet been promoted
    to a structured constructor can be lowered row-by-row through this path. -/

open WasmCircuit (Fq)

/-- Lift an integer coefficient into a constant `ZKExpr Fq`. -/
def coeffExpr (c : Coeff) : ZKExpr Fq := ZKExpr.Field (c : Fq)

/-- Build a `ZKExpr` representing the dot product of a sparse row with the
    pre-allocated witness expression array `alloc`. -/
def sparseRowToExpr (alloc : Array (ZKExpr Fq)) : SparseRow → ZKExpr Fq
  | [] => 0
  | (idx, c) :: rest => coeffExpr c * alloc[idx]! + sparseRowToExpr alloc rest

/-- Lower a single R1CS row to a zkLean constraint. -/
def rowToBuilder (alloc : Array (ZKExpr Fq)) (row : Row) : ZKBuilder Fq PUnit :=
  let (a, b, c) := row
  ZKBuilder.constrainR1CS
    (sparseRowToExpr alloc a)
    (sparseRowToExpr alloc b)
    (sparseRowToExpr alloc c)

/-- Lower an `Instr` to its zkLean builder fragment, by lowering each row of
    `instrToRows i` through `rowToBuilder`. This shape makes `instrToRows`
    the single source of truth for the row structure of every `Instr`
    constructor — `instrToBuilder` cannot drift from it by construction. -/
def instrToBuilder (alloc : Array (ZKExpr Fq)) (i : Instr) : ZKBuilder Fq PUnit :=
  (instrToRows i).forM (rowToBuilder alloc)

/-- Allocate `n` fresh witnesses and collect them into an array indexable by
    column number. -/
def allocateColumns (n : Nat) : ZKBuilder Fq (Array (ZKExpr Fq)) := do
  let mut alloc : Array (ZKExpr Fq) := Array.mkEmpty n
  for _ in [0:n] do
    let w ← ZKBuilder.witness
    alloc := alloc.push w
  return alloc

/-- Build the full zkLean circuit for a list of `Instr`s over a witness of
    `width` columns. -/
def circuitOfInstructions (width : Nat) (instructions : List Instr) :
    ZKBuilder Fq PUnit := do
  let alloc ← allocateColumns width
  instructions.forM (instrToBuilder alloc)

end WasmCircuit.Gadgets
