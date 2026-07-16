import Nightstream.SuperNeo.Concrete.Algebra

/-!
Assignment-free mathematical semantics for the SIS layer used by the
`Pi_CCS` output digest.

Assurance tier: executable protocol-primitive semantics. This file does not
import Rust-derived block metadata, R1CS rows, production columns, ChaCha
implementations, generated seeds, or digest claims. It specifies the
canonical centered word, row-major padding, and one abstract linear action.

Owns: the 41-coordinate centered radix-three encoding of a canonical
Goldilocks field; the 54-row message layout; zero padding; sparse-coordinate
evaluation; and the flattened output order.

Does not own: which coefficient map production uses; how public seeds expand
to that map; any R1CS lowering; Poseidon2; collision resistance; transcript
placement; row necessity; row removal; or cost totals.

Emits constraints: no.

Authority boundary: `LinearMap` is an explicit mathematical parameter. A
later refinement must identify the production seeded map with one such value,
and cryptographic use must state the binding assumption for that map. No
prover-supplied commitment or digest is an input to `apply`.

| Protocol | Phase | Mathematical branch | Definition | Exact obligation |
|---|---|---|---|---|
| `Pi_CCS` | output digest | canonical word | `canonicalDigit` | unique centered digit of `(field + shift) mod p` |
| `Pi_CCS` | output digest | message layout | `messageValue` | field-major words placed in a 54-row row-major matrix, then zero padded |
| `Pi_CCS` | output digest | SIS action | `coordinateTerms` | one coefficient-weighted sparse coordinate equation |
| `Pi_CCS` | output digest | commitment | `apply` | output-major, coefficient-minor flattened linear-map result |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.Semantics

open Nightstream.SuperNeo.Concrete

/-- Canonical balanced-ternary word width used by the binding map. -/
def digitCount : Nat := 41

/-- Additive shift converting centered digits `{-1,0,1}` to radix-three
digits `{0,1,2}`. -/
def shift : Nat := (3 ^ digitCount - 1) / 2

/-- Canonical field representative modulus. -/
def modulus : Nat := goldilocksModulus

/-- Assignment-free centered digit, represented canonically in Goldilocks. -/
def canonicalDigit (fieldValue index : Nat) : Nat :=
  match ((fieldValue + shift) % modulus) / 3 ^ index % 3 with
  | 0 => modulus - 1
  | 1 => 0
  | _ => 1

/-- The cyclotomic coefficient dimension `phi(81)`. -/
def dimension : Nat := 54

/-- An abstract public linear map. `coefficient output messageColumn
messageRow coordinate` is the canonical Goldilocks coefficient multiplying
one message cell in one output coordinate. -/
structure LinearMap where
  kappa : Nat
  messageCols : Nat
  coefficient : Nat -> Nat -> Nat -> Nat -> Nat

/-- Row-major location corresponding to one message matrix cell. -/
def messageIndex (map : LinearMap) (messageColumn messageRow : Nat) : Nat :=
  messageRow * map.messageCols + messageColumn

/-- Canonical padded value stored at one 54-row message cell. -/
def messageValue (map : LinearMap) (fields : List Nat)
    (messageColumn messageRow : Nat) : Nat :=
  let index := messageIndex map messageColumn messageRow
  if index < fields.length * digitCount then
    canonicalDigit (fields.getD (index / digitCount) 0) (index % digitCount)
  else
    0

/-- Sparse value/coefficient pairs for one output coordinate. Padding and
zero coefficients do not appear. -/
def coordinateTerms (map : LinearMap) (fields : List Nat)
    (output coordinate : Nat) : List (Nat × Nat) :=
  (List.range map.messageCols).flatMap fun messageColumn =>
    (List.range dimension).filterMap fun messageRow =>
      let index := messageIndex map messageColumn messageRow
      if index < fields.length * digitCount then
        let coefficient :=
          map.coefficient output messageColumn messageRow coordinate
        if coefficient = 0 then none
        else some (messageValue map fields messageColumn messageRow,
          coefficient)
      else
        none

/-- Canonical Goldilocks evaluation of sparse value/coefficient pairs. -/
def evalTerms (terms : List (Nat × Nat)) : Nat :=
  terms.foldl (fun total term => total + term.2 * term.1) 0 % modulus

def applyCoordinate (map : LinearMap) (fields : List Nat)
    (output coordinate : Nat) : Nat :=
  evalTerms (coordinateTerms map fields output coordinate)

/-- Output-major, coordinate-minor linear-map result. -/
def apply (map : LinearMap) (fields : List Nat) : List Nat :=
  (List.range map.kappa).flatMap fun output =>
    (List.range dimension).map fun coordinate =>
      applyCoordinate map fields output coordinate

theorem apply_length (map : LinearMap) (fields : List Nat) :
    (apply map fields).length = map.kappa * dimension := by
  simp [apply, List.map_const']

theorem applyCoordinate_canonical (map : LinearMap) (fields : List Nat)
    (output coordinate : Nat) :
    applyCoordinate map fields output coordinate < modulus := by
  unfold applyCoordinate evalTerms
  exact Nat.mod_lt _ (by decide)

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.Semantics
