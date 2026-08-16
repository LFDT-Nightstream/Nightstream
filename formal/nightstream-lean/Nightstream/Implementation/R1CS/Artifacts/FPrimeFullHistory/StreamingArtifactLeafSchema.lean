import Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
import Nightstream.Implementation.R1CS.Core.Poseidon2Call

/-!
Contract: shared leaf schema for exact streaming R1CS artifacts.

Owns the compact canonical-call and indexed-row records, plus their local
geometry predicates. It owns no generated schedule, coordinate map, or
protocol relation.

Emits constraints: no. It describes existing constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical

structure CanonicalCall where
  rowStart : Nat
  rowEnd : Nat
  fieldColumn : Nat
  bitBase : Nat
  highFlagColumn : Nat
  inverseColumn : Nat
deriving DecidableEq, Repr, Inhabited

def CanonicalCall.layout (call : CanonicalCall) : CanonicalU64Recipe.Layout where
  base := call.bitBase
  input := [(call.fieldColumn, 1)]

def CanonicalCall.Valid (columnCount : Nat) (call : CanonicalCall) : Prop :=
  call.rowEnd = call.rowStart + 69 ∧
    0 < call.bitBase ∧
    call.highFlagColumn = call.bitBase + 64 ∧
    call.inverseColumn = call.bitBase + 65 ∧
    call.fieldColumn < columnCount ∧
    call.inverseColumn < columnCount

instance (columnCount : Nat) (call : CanonicalCall) :
    Decidable (call.Valid columnCount) := by
  unfold CanonicalCall.Valid
  infer_instance

def CanonicalCall.Satisfied (assignment : Nat → Nat)
    (call : CanonicalCall) : Prop :=
  Satisfies (CanonicalU64Recipe.rows call.layout) assignment

structure IndexedRow where
  index : Nat
  row : Row
deriving DecidableEq, Repr

def rowColumnsBelow (columnCount : Nat) (row : Row) : Prop :=
  (∀ term ∈ row.a, term.1 < columnCount) ∧
    (∀ term ∈ row.b, term.1 < columnCount) ∧
    ∀ term ∈ row.c, term.1 < columnCount

instance (columnCount : Nat) (row : Row) :
    Decidable (rowColumnsBelow columnCount row) := by
  unfold rowColumnsBelow
  infer_instance

def PoseidonCallValid (columnCount : Nat) (call : Poseidon2Call.Call) : Prop :=
  call.rowEnd = call.rowStart + 600 ∧
    call.inputColumns.length = 8 ∧
    (∀ column ∈ call.inputColumns, column < columnCount) ∧
    call.firstAllocatedColumn + 600 ≤ columnCount

instance (columnCount : Nat) (call : Poseidon2Call.Call) :
    Decidable (PoseidonCallValid columnCount call) := by
  unfold PoseidonCallValid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
