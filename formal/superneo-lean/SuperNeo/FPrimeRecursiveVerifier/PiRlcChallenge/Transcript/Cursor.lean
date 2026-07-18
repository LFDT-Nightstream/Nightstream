import Mathlib.Tactic
import SuperNeo.Primitives.Field

/-!
Owns: width-8 sponge state, rate-4 cursor, raw field absorption, and `digest32`
cursor transition.

Does not own: digest-lane byte extraction, rejection sampling, the concrete
Poseidon2 round constants, or the incoming cursor's authority.

Emits constraints: no. This file states executable transcript semantics.

Authority boundary: callers supply both the permutation core and an already
authoritative incoming cursor; this module only determines the next cursor.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `absorbElem` | `Poseidon2Transcript::append_fields_raw` | One rate-lane overwrite with overflow permutation | Supplied `Poseidon2Core` | No — concrete Poseidon2/Rust refinement open |
| `appendFieldsRaw` | `Poseidon2Transcript::append_fields_raw` | Length header precedes the raw fields | Supplied cursor/core | No — concrete Poseidon2/Rust refinement open |
| `digestCursor` | `Poseidon2Transcript::digest32` | Domain gate `1` is absorbed before permutation | Supplied cursor/core | No — concrete Poseidon2/Rust refinement open |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-- Width-8 Goldilocks sponge state. -/
abbrev SpongeState := Fin 8 → F

/-- Explicit dependency on the concrete width-8 Poseidon2 permutation. -/
structure Poseidon2Core where
  permute : SpongeState → SpongeState

/-- Rate-4 sponge state plus its statically bounded absorb cursor. -/
structure SpongeCursor where
  state : SpongeState
  absorbed : Fin 5

def spongeRate : Nat := 4

def setLane
    (state : SpongeState) (index : Fin 8) (value : F) : SpongeState :=
  Function.update state index value

def permuteCursor
    (core : Poseidon2Core) (cursor : SpongeCursor) : SpongeCursor :=
  { state := core.permute cursor.state
    absorbed := ⟨0, by decide⟩ }

/-- Exact single-element overwrite semantics of the native sponge. -/
def absorbElem
    (core : Poseidon2Core) (cursor : SpongeCursor) (value : F) :
    SpongeCursor :=
  if hSpace : cursor.absorbed.val < spongeRate then
    { state := setLane cursor.state
        ⟨cursor.absorbed.val, Nat.lt_trans hSpace (by decide)⟩ value
      absorbed := ⟨cursor.absorbed.val + 1, by
        simp only [spongeRate] at hSpace
        omega⟩ }
  else
    { state := setLane (core.permute cursor.state) ⟨0, by decide⟩ value
      absorbed := ⟨1, by decide⟩ }

/--
Exact native slice behavior: filling the rate portion permutes immediately,
including when the final input fills the fourth slot.
-/
def absorbSlice (core : Poseidon2Core) :
    SpongeCursor → List F → SpongeCursor
  | cursor, [] =>
      if cursor.absorbed.val = spongeRate then
        permuteCursor core cursor
      else
        cursor
  | cursor, value :: values =>
      let afterValue := absorbElem core cursor value
      let ready :=
        if afterValue.absorbed.val = spongeRate then
          permuteCursor core afterValue
        else
          afterValue
      absorbSlice core ready values

/-- Native `append_fields_raw`: one length header followed by field values. -/
def appendFieldsRaw
    (core : Poseidon2Core) (cursor : SpongeCursor) (values : List F) :
    SpongeCursor :=
  absorbSlice core
    (absorbElem core cursor (F.ofNat values.length)) values

/-- Native `digest32` cursor transition before byte serialization. -/
def digestCursor
    (core : Poseidon2Core) (cursor : SpongeCursor) : SpongeCursor :=
  permuteCursor core (absorbElem core cursor (F.ofNat 1))

@[simp] theorem permuteCursor_absorbed
    (core : Poseidon2Core) (cursor : SpongeCursor) :
    (permuteCursor core cursor).absorbed.val = 0 := rfl

@[simp] theorem digestCursor_absorbed
    (core : Poseidon2Core) (cursor : SpongeCursor) :
    (digestCursor core cursor).absorbed.val = 0 := rfl

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
