import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCursor

/-!
Contract: expose the value-independent control state of the symbolic
Poseidon2 duplex.

Assurance tier: model-level canonical encoding.

Owns: preservation of permutation-entry count and cursor state across
builders that absorb equally long field lists.

Does not own: a protocol transcript, field order, physical placement, or
security.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexControl

open Nightstream.Implementation.R1CS.Canonical

/-- Two builders have the same row-count control state. Lane expressions and
physical column identities are deliberately absent. -/
structure Equivalent
    (left right : SymbolicDuplex.Builder) : Prop where
  entries : left.entries.length = right.entries.length
  absorbed : left.absorbed = right.absorbed

@[refl] theorem Equivalent.refl
    (builder : SymbolicDuplex.Builder) :
    Equivalent builder builder :=
  ⟨rfl, rfl⟩

theorem Equivalent.symm
    {left right : SymbolicDuplex.Builder}
    (equivalent : Equivalent left right) :
    Equivalent right left :=
  ⟨equivalent.entries.symm, equivalent.absorbed.symm⟩

theorem Equivalent.trans
    {first second third : SymbolicDuplex.Builder}
    (left : Equivalent first second)
    (right : Equivalent second third) :
    Equivalent first third :=
  ⟨left.entries.trans right.entries,
    left.absorbed.trans right.absorbed⟩

theorem permute
    (leftBase rightBase : Nat)
    {left right : SymbolicDuplex.Builder}
    (equivalent : Equivalent left right) :
    Equivalent
      (SymbolicDuplex.permute leftBase left)
      (SymbolicDuplex.permute rightBase right) := by
  constructor
  · rw [SymbolicDuplex.permute_entries_length,
      SymbolicDuplex.permute_entries_length]
    exact congrArg (fun count => count + 1) equivalent.entries
  · rfl

theorem guarded
    (leftBase rightBase : Nat)
    {left right : SymbolicDuplex.Builder}
    (equivalent : Equivalent left right) :
    Equivalent
      (SymbolicDuplex.guarded leftBase left)
      (SymbolicDuplex.guarded rightBase right) := by
  unfold SymbolicDuplex.guarded
  by_cases full : Poseidon2Sponge.rate ≤ left.absorbed
  · have rightFull : Poseidon2Sponge.rate ≤ right.absorbed := by
      simpa [equivalent.absorbed] using full
    simp only [if_pos full, if_pos rightFull]
    exact permute leftBase rightBase equivalent
  · have rightNotFull : ¬ Poseidon2Sponge.rate ≤ right.absorbed := by
      simpa [equivalent.absorbed] using full
    simp only [if_neg full, if_neg rightNotFull]
    exact equivalent

theorem absorb
    (leftBase rightBase : Nat)
    (leftValue rightValue : LinCombNormal.LinComb)
    {left right : SymbolicDuplex.Builder}
    (equivalent : Equivalent left right) :
    Equivalent
      (SymbolicDuplex.absorb leftBase leftValue left)
      (SymbolicDuplex.absorb rightBase rightValue right) := by
  let leftReady := SymbolicDuplex.guarded leftBase left
  let rightReady := SymbolicDuplex.guarded rightBase right
  have readyEquivalent : Equivalent leftReady rightReady :=
    guarded leftBase rightBase equivalent
  constructor
  · exact readyEquivalent.entries
  · change leftReady.absorbed + 1 = rightReady.absorbed + 1
    exact congrArg (fun value => value + 1) readyEquivalent.absorbed

theorem absorbMany
    (leftBase rightBase : Nat) :
    ∀ (leftValues rightValues : List LinCombNormal.LinComb)
      (left right : SymbolicDuplex.Builder),
      leftValues.length = rightValues.length →
      Equivalent left right →
      Equivalent
        (SymbolicDuplex.absorbMany leftBase leftValues left)
        (SymbolicDuplex.absorbMany rightBase rightValues right)
  | [], [], _, _, _, equivalent => equivalent
  | [], _ :: _, _, _, lengths, _ => by cases lengths
  | _ :: _, [], _, _, lengths, _ => by cases lengths
  | leftValue :: leftRest, rightValue :: rightRest, left, right,
      lengths, equivalent => by
      simp only [List.length_cons, Nat.succ.injEq] at lengths
      exact absorbMany leftBase rightBase leftRest rightRest
        (SymbolicDuplex.absorb leftBase leftValue left)
        (SymbolicDuplex.absorb rightBase rightValue right)
        lengths
        (absorb leftBase rightBase leftValue rightValue equivalent)

theorem gate
    (leftBase rightBase : Nat)
    {left right : SymbolicDuplex.Builder}
    (equivalent : Equivalent left right) :
    Equivalent
      (SymbolicDuplex.gate leftBase left)
      (SymbolicDuplex.gate rightBase right) := by
  unfold SymbolicDuplex.gate
  apply permute
  exact absorb leftBase rightBase SymbolicDuplex.one SymbolicDuplex.one
    equivalent

end Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexControl
