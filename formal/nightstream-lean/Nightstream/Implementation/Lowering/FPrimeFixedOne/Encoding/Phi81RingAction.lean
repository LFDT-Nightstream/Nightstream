import Nightstream.Implementation.Lowering.Goldilocks.Rows
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Product

/-!
Contract: exact Lean-owned physical rows for a finite sum of Phi81
base-ring actions.

For `count` public pairs `(rho_i, value_i)`, this program enforces

```text
output = sum_i rho_i * value_i
```

in `F[X]/(X^54 + X^27 + 1)`.  It allocates one product coordinate for every
`i × 54 × 54` schoolbook term and emits one reduced output equation per
ring lane.  The construction is independent of Rust and is shared by the
commitment, public-input, and evaluation branches of the selected PiRLC
parent.

This file owns the row construction, exact cost, semantic soundness, honest
completion, positional row ownership, and support classification.  It does
not own call-frame placement, activation, codecs, or the selected NIFS
composition.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm

/-- A ring value carried by sparse physical linear combinations. -/
abbrev CarriedRing := Fin ringDegree → LinearCombination

/-- Semantic value decoded from a carried ring under one assignment. -/
def decoded
    (assignment : ColumnId → F)
    (value : CarriedRing) : RingF :=
  fun lane => (value lane).eval assignment

/-- Canonical head-first finite sum of ring products.  Its recursion matches
the concrete PiRLC commitment and public-input folds. -/
def combine :
    {count : Nat} →
      (Fin count → RingF) →
      (Fin count → RingF) →
      RingF
  | 0, _, _ => ringFZero
  | _ + 1, challenges, values =>
      ringFAdd
        (ringFMul (challenges 0) (values 0))
        (combine
          (fun index => challenges index.succ)
          (fun index => values index.succ))

/-- Exact auxiliary width: one field product for every source and ordered
pair of ring coefficients. -/
def productWidth (count : Nat) : Nat :=
  count * ringDegree * ringDegree

/-- Exact row count: one row per product plus one reduced equality per lane. -/
def rowCount (count : Nat) : Nat :=
  productWidth count + ringDegree

/-- Structural inputs to one occurrence.  Product columns are supplied
explicitly so later call-frame placement can compact sparse component
allocations without changing this program. -/
structure Frame (count : Nat) where
  owner : PhysicalOwner
  firstOrdinal : Nat
  one : ColumnId
  challenges : Fin count → CarriedRing
  values : Fin count → CarriedRing
  output : CarriedRing
  productColumn : Nat → Nat → Nat → ColumnId

/-- Stable row-major product offset. -/
def productOffset (source left right : Nat) : Nat :=
  (source * ringDegree + left) * ringDegree + right

/-- Exact ordered product allocation used by the row program. -/
def productIds {count : Nat} (frame : Frame count) : List ColumnId :=
  (List.range count).flatMap fun source =>
    (List.range ringDegree).flatMap fun left =>
      (List.range ringDegree).map fun right =>
        frame.productColumn source left right

/-- Every source product equation. -/
def productRow {count : Nat}
    (frame : Frame count) (source left right : Nat) : Row where
  a :=
    if sourceLt : source < count then
      if leftLt : left < ringDegree then
        frame.challenges ⟨source, sourceLt⟩ ⟨left, leftLt⟩
      else []
    else []
  b :=
    if sourceLt : source < count then
      if rightLt : right < ringDegree then
        frame.values ⟨source, sourceLt⟩ ⟨right, rightLt⟩
      else []
    else []
  c := singleton (frame.productColumn source left right) 1

/-- All schoolbook product equations, in source/left/right order. -/
def productRows {count : Nat} (frame : Frame count) : List Row :=
  (List.range count).flatMap fun source =>
    (List.range ringDegree).flatMap fun left =>
      (List.range ringDegree).map fun right =>
        productRow frame source left right

/-- Sparse terms of one raw convolution over an explicit index list. -/
def sourceTerms {count : Nat}
    (frame : Frame count) (source degree : Nat)
    (indices : List Nat) : LinearCombination :=
  indices.map fun left =>
    if Product.supportActive degree left then
      {
        column := frame.productColumn source left (degree - left)
        coefficient := 1
      }
    else
      { column := frame.one, coefficient := 0 }

/-- One raw convolution combination for one source and one unreduced degree.
Inactive indices use the visible constant wire with coefficient zero, so the
combination never mentions an unallocated product coordinate. -/
def sourceRawCombination {count : Nat}
  (frame : Frame count) (source degree : Nat) : LinearCombination :=
  sourceTerms frame source degree (List.range ringDegree)

/-- Drop the head source while preserving the physical identity of every
remaining product coordinate. -/
def tailFrame {count : Nat} (frame : Frame (count + 1)) : Frame count where
  owner := frame.owner
  firstOrdinal := frame.firstOrdinal
  one := frame.one
  challenges index := frame.challenges index.succ
  values index := frame.values index.succ
  output := frame.output
  productColumn source left right :=
    frame.productColumn (source + 1) left right

/-- Raw convolution summed over every source pair.  The recursion deliberately
matches `combine`: this makes the emitted reduction's agreement with the
protocol fold an induction theorem instead of a list-indexing assumption. -/
def rawCombination :
    {count : Nat} → Frame count → Nat → LinearCombination
  | 0, _, _ => []
  | _ + 1, frame, degree =>
      sourceRawCombination frame 0 degree ++
        rawCombination (tailFrame frame) degree

/-- Coefficient-wise negation without allocating a field value. -/
def negate (combination : LinearCombination) : LinearCombination :=
  combination.map fun term =>
    { term with coefficient := -term.coefficient }

/-- Exact Phi81 reduction for one output lane. -/
def reducedCombination {count : Nat}
    (frame : Frame count) (output : Nat) : LinearCombination :=
  let folded :=
    if output < ringMiddleDegree then output + ringDegree
    else output + ringMiddleDegree
  let twice :=
    if output + 81 ≤ 106 then rawCombination frame (output + 81)
    else []
  rawCombination frame output ++
    negate (rawCombination frame folded) ++ twice

/-- One reduced output equation. -/
def outputRow {count : Nat}
    (frame : Frame count) (output : Nat) : Row where
  a := reducedCombination frame output
  b := singleton frame.one 1
  c :=
    if outputLt : output < ringDegree then
      frame.output ⟨output, outputLt⟩
    else []

/-- All reduced output equations. -/
def outputRows {count : Nat} (frame : Frame count) : List Row :=
  (List.range ringDegree).map (outputRow frame)

/-- Complete raw row program. -/
def rawRows {count : Nat} (frame : Frame count) : List Row :=
  productRows frame ++ outputRows frame

/-- Assign one stable owner and contiguous ordinal to every raw occurrence. -/
def ownRows
    (owner : PhysicalOwner) (firstOrdinal : Nat) :
    List Row → List OwnedRow
  | [] => []
  | row :: tail =>
      {
        id := { owner := owner, ordinal := firstOrdinal }
        row := row
      } :: ownRows owner (firstOrdinal + 1) tail

/-- Complete owned row program. -/
def rows {count : Nat} (frame : Frame count) : List OwnedRow :=
  ownRows frame.owner frame.firstOrdinal (rawRows frame)

private theorem sum_map_const
    {α : Type} (items : List α) (value : Nat) :
    (items.map fun _ => value).sum = value * items.length := by
  induction items with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis, Nat.mul_succ]
      omega

@[simp] theorem productIds_length
    {count : Nat} (frame : Frame count) :
    (productIds frame).length = productWidth count := by
  simp only [productIds, List.length_flatMap, List.length_map,
    List.length_range]
  rw [show
      (List.map
        (fun _ =>
          (List.map (fun _ => ringDegree)
            (List.range ringDegree)).sum)
        (List.range count)).sum =
        (List.map (fun _ => ringDegree * ringDegree)
          (List.range count)).sum by
      apply congrArg List.sum
      apply List.map_congr_left
      intro source sourceMember
      exact sum_map_const (List.range ringDegree) ringDegree]
  rw [sum_map_const]
  simp [productWidth, Nat.mul_comm, Nat.mul_left_comm]

@[simp] theorem productRows_length
    {count : Nat} (frame : Frame count) :
    (productRows frame).length = productWidth count := by
  simp only [productRows, List.length_flatMap, List.length_map,
    List.length_range]
  rw [show
      (List.map
        (fun _ =>
          (List.map (fun _ => ringDegree)
            (List.range ringDegree)).sum)
        (List.range count)).sum =
        (List.map (fun _ => ringDegree * ringDegree)
          (List.range count)).sum by
      apply congrArg List.sum
      apply List.map_congr_left
      intro source sourceMember
      exact sum_map_const (List.range ringDegree) ringDegree]
  rw [sum_map_const]
  simp [productWidth, Nat.mul_comm, Nat.mul_left_comm]

@[simp] theorem outputRows_length
    {count : Nat} (frame : Frame count) :
    (outputRows frame).length = ringDegree := by
  simp [outputRows]

@[simp] theorem rawRows_length
    {count : Nat} (frame : Frame count) :
    (rawRows frame).length = rowCount count := by
  simp [rawRows, rowCount]

@[simp] theorem ownRows_length
    (owner : PhysicalOwner) (firstOrdinal : Nat) (source : List Row) :
    (ownRows owner firstOrdinal source).length = source.length := by
  induction source generalizing firstOrdinal with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [ownRows, inductionHypothesis]

@[simp] theorem rows_length
    {count : Nat} (frame : Frame count) :
    (rows frame).length = rowCount count := by
  simp [rows]

/-- The exact auxiliary column receipt. -/
def columns {count : Nat} (frame : Frame count) : List OwnedColumn :=
  (productIds frame).map fun id =>
    { id := id, ownership := .auxiliaryColumn }

@[simp] theorem columns_length
    {count : Nat} (frame : Frame count) :
    (columns frame).length = productWidth count := by
  simp [columns]

/-- Program-derived intrinsic cost. -/
def cost (count : Nat) : Cost where
  recurringRows := rowCount count
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := productWidth count

theorem rows_cost
    {count : Nat} (frame : Frame count) :
    (rows frame).length = (cost count).recurringRows := by
  simp [cost]

theorem columns_cost
    {count : Nat} (frame : Frame count) :
    (columns frame).length = (cost count).auxiliaryColumns := by
  simp [cost]

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction
