import Nightstream.Implementation.Lowering.Nebula.Compiler

/-!
Contract: proof-free serialization of the Lean-owned Nebula row program.

Assurance tier: model-level.

Owns: canonical numeric coefficients, all fifteen matrix images, stable row
identity, exact encode/decode round trips, and the selected row order.

Does not own: JSON, combined F-prime placement, a Rust decoder, witness
values, or a security reduction.

Emits constraints: none. It serializes the existing Nebula rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Nebula.Manifest

open Nightstream.Implementation.Lowering.Nebula
open Nightstream.SuperNeo.Concrete

/-- One proof-free sparse term. Coefficients use their canonical Goldilocks
representative. -/
structure Term where
  column : Nat
  coefficient : Nat
deriving DecidableEq, Repr

namespace Term

def ofRowTerm (term : Rows.Term) : Term where
  column := term.column
  coefficient := term.coefficient.val

def decode (term : Term) : Rows.Term where
  column := term.column
  coefficient :=
    ⟨term.coefficient % goldilocksModulus,
      Nat.mod_lt _ (by decide)⟩

@[simp] theorem decode_ofRowTerm (term : Rows.Term) :
    (ofRowTerm term).decode = term := by
  cases term with
  | mk column coefficient =>
      simp [ofRowTerm, decode, Nat.mod_eq_of_lt coefficient.isLt]

end Term

abbrev Combination := List Term

namespace Combination

def ofRows (combination : Rows.LinearCombination) : Combination :=
  combination.map Term.ofRowTerm

def decode (combination : Combination) : Rows.LinearCombination :=
  combination.map Term.decode

@[simp] theorem decode_ofRows (combination : Rows.LinearCombination) :
    (ofRows combination).decode = combination := by
  rw [show (ofRows combination).decode =
    combination.map (Term.decode ∘ Term.ofRowTerm) by
      simp [ofRows, decode, List.map_map]]
  induction combination with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      simp [inductionHypothesis]

end Combination

/-- Proof-free image vector in the exact semantic role order. -/
structure Images where
  bit : Combination
  productLeft : Combination
  productRight : Combination
  linearLeft : Combination
  linearRight : Combination
  output : Combination
  extensionA : Combination
  extensionB : Combination
  pad : Combination
  active : Combination
  fingerprintA : Combination
  fingerprintB : Combination
  valueA : Combination
  valueB : Combination
  value : Combination
deriving DecidableEq, Repr

namespace Images

def ofRows (images : Rows.Images) : Images where
  bit := Combination.ofRows images.bit
  productLeft := Combination.ofRows images.productLeft
  productRight := Combination.ofRows images.productRight
  linearLeft := Combination.ofRows images.linearLeft
  linearRight := Combination.ofRows images.linearRight
  output := Combination.ofRows images.output
  extensionA := Combination.ofRows images.extensionA
  extensionB := Combination.ofRows images.extensionB
  pad := Combination.ofRows images.pad
  active := Combination.ofRows images.active
  fingerprintA := Combination.ofRows images.fingerprintA
  fingerprintB := Combination.ofRows images.fingerprintB
  valueA := Combination.ofRows images.valueA
  valueB := Combination.ofRows images.valueB
  value := Combination.ofRows images.value

def decode (images : Images) : Rows.Images where
  bit := Combination.decode images.bit
  productLeft := Combination.decode images.productLeft
  productRight := Combination.decode images.productRight
  linearLeft := Combination.decode images.linearLeft
  linearRight := Combination.decode images.linearRight
  output := Combination.decode images.output
  extensionA := Combination.decode images.extensionA
  extensionB := Combination.decode images.extensionB
  pad := Combination.decode images.pad
  active := Combination.decode images.active
  fingerprintA := Combination.decode images.fingerprintA
  fingerprintB := Combination.decode images.fingerprintB
  valueA := Combination.decode images.valueA
  valueB := Combination.decode images.valueB
  value := Combination.decode images.value

@[simp] theorem decode_ofRows (images : Rows.Images) :
    (ofRows images).decode = images := by
  cases images
  simp [ofRows, decode]

end Images

/-- One proof-free row. `id.position` remains the physical row owner. -/
structure Row where
  id : Rows.RowId
  images : Images
deriving DecidableEq, Repr

namespace Row

def ofRows (row : Rows.Row) : Row where
  id := row.id
  images := Images.ofRows row.images

def decode (row : Row) : Rows.Row where
  id := row.id
  images := row.images.decode

@[simp] theorem decode_ofRows (row : Rows.Row) :
    (ofRows row).decode = row := by
  cases row
  simp [ofRows, decode]

end Row

/-- Exact proof-free Nebula program in compiler emission order. -/
structure Program where
  matrixCount : Nat
  strictDegreeBound : Nat
  columnCount : Nat
  publicEnd : Nat
  rows : List Row
deriving DecidableEq, Repr

def ofParams (params : Layout.Params) : Program where
  matrixCount := StepPolynomial.matrixCount
  strictDegreeBound := StepPolynomial.polynomial.degreeBound
  columnCount := params.columnCount
  publicEnd := params.publicEnd
  rows := (Compiler.rows params).map Row.ofRows

def Program.decode (program : Program) : List Rows.Row :=
  program.rows.map Row.decode

@[simp] theorem decode_ofParams (params : Layout.Params) :
    (ofParams params).decode = Compiler.rows params := by
  rw [show (ofParams params).decode =
    (Compiler.rows params).map (Row.decode ∘ Row.ofRows) by
      simp [ofParams, Program.decode, List.map_map]]
  induction Compiler.rows params with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      simp [inductionHypothesis]

@[simp] theorem matrixCount_ofParams (params : Layout.Params) :
    (ofParams params).matrixCount = 15 :=
  rfl

@[simp] theorem strictDegreeBound_ofParams (params : Layout.Params) :
    (ofParams params).strictDegreeBound = 5 :=
  rfl

@[simp] theorem rows_length_ofParams (params : Layout.Params) :
    (ofParams params).rows.length = params.rowCount := by
  simp [ofParams, Compiler.rows_length]

end Nightstream.Implementation.Lowering.Nebula.Manifest
