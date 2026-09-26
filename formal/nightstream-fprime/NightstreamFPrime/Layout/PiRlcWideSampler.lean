import NightstreamFPrime.Gadgets.Sampling.WideReduction.Program
import NightstreamFPrime.Layout.LowNormSlot

/-! Retained values of one wide sampler. The four canonical children keep
64 bit slots and two field slots each. The 353 result bits also use one
coordinate each. Temporary helper hints have no retained slot.

This component layout does not select a Stage 1 package or matrix plan. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling.WideReduction

abbrev Slot := (Fin 4 × Fin 66) ⊕ Fin 353

def source : Slot → Nat
  | .inl (lane, index) => 66 * lane.val + index.val
  | .inr index => 264 + index.val

def childKind (index : Fin 66) : LowNormSlot.Kind :=
  if index.val < 64 then .bit else .field

def kind : Slot → LowNormSlot.Kind
  | .inl (_, index) => childKind index
  | .inr _ => .bit

theorem source_bound (slot : Slot) : source slot < privateCount := by
  change source slot < 617
  cases slot with
  | inl pair =>
      have lane := pair.1.isLt
      have index := pair.2.isLt
      change 66 * pair.1.val + pair.2.val < 617
      omega
  | inr index =>
      have bound := index.isLt
      change 264 + index.val < 617
      omega

theorem source_injective : Function.Injective source := by
  intro left right same
  cases left with
  | inl a =>
      cases right with
      | inl b =>
          have a0 := a.1.isLt
          have a1 := a.2.isLt
          have b0 := b.1.isLt
          have b1 := b.2.isLt
          change 66 * a.1.val + a.2.val = 66 * b.1.val + b.2.val at same
          have pair : a = b := Prod.ext (Fin.ext (by omega)) (Fin.ext (by omega))
          exact congrArg Sum.inl pair
      | inr b =>
          have a0 := a.1.isLt
          have a1 := a.2.isLt
          change 66 * a.1.val + a.2.val = 264 + b.val at same
          omega
  | inr a =>
      cases right with
      | inl b =>
          have b0 := b.1.isLt
          have b1 := b.2.isLt
          change 264 + a.val = 66 * b.1.val + b.2.val at same
          omega
      | inr b =>
          exact congrArg Sum.inr (Fin.ext (by change 264 + a.val = 264 + b.val at same; omega))

theorem source_covers (column : Nat) (bound : column < privateCount) :
    ∃ slot : Slot, source slot = column := by
  change column < 617 at bound
  by_cases child : column < 264
  · refine ⟨.inl (⟨column / 66, by omega⟩, ⟨column % 66, Nat.mod_lt _ (by decide)⟩), ?_⟩
    exact Nat.div_add_mod column 66
  · exact ⟨.inr ⟨column - 264, by omega⟩, by change 264 + (column - 264) = column; omega⟩

private theorem bit_valid (value : F) (bound : value.val ≤ 1) : LowNormSlot.Valid .bit value := by
  have bit : value.val = 0 ∨ value.val = 1 := by omega
  rcases bit with zero | one
  · exact Or.inl (Fin.ext zero)
  · exact Or.inr (Fin.ext one)

theorem kind_valid (interface : Interface) (hints : Nat → List Hint) (env : Env)
    (offset : Nat) (inputs : Assumptions interface offset)
    (rows : holds env (operations interface hints offset)) (slot : Slot) :
    LowNormSlot.Valid (kind slot) (env (offset + source slot)) := by
  cases slot with
  | inl pair =>
      rcases pair with ⟨lane, index⟩
      unfold kind childKind
      dsimp only
      split_ifs with bit
      · apply bit_valid
        apply retained_bit_le_one interface hints env offset _ inputs rows (source_bound (.inl (lane, index)))
        intro _
        have bounded := index.isLt
        change (66 * lane.val + index.val) % 66 < 64
        omega
      · trivial
  | inr index =>
      apply bit_valid
      apply retained_bit_le_one interface hints env offset _ inputs rows (source_bound (.inr index))
      intro impossible
      change 264 + index.val < 264 at impossible
      omega

def coordinateCount : Nat := ∑ slot : Slot, (kind slot).width

theorem child_coordinateCount : (∑ index : Fin 66, (childKind index).width) = 146 := by
  change (∑ index : Fin (64 + 2), (childKind index).width) = _
  rw [Fin.sum_univ_add]
  have first : (fun index : Fin 64 => (childKind (index.castAdd 2)).width) = fun _ => 1 := by
    funext index
    unfold childKind
    change (if index.val < 64 then LowNormSlot.Kind.bit else .field).width = 1
    rw [if_pos index.isLt]
    rfl
  have second : (fun index : Fin 2 => (childKind (index.natAdd 64)).width) = fun _ => 41 := by
    funext index
    unfold childKind
    rw [if_neg (by change ¬64 + index.val < 64; omega)]
    rfl
  rw [first, second]
  simp

theorem coordinateCount_eq : coordinateCount = 937 := by
  unfold coordinateCount
  rw [Fintype.sum_sum_type, Fintype.sum_prod_type]
  simp only [kind]
  simp_rw [child_coordinateCount]
  simp [LowNormSlot.Kind.width]

theorem encode_norm (interface : Interface) (hints : Nat → List Hint) (env : Env)
    (offset : Nat) (inputs : Assumptions interface offset)
    (rows : holds env (operations interface hints offset)) (slot : Slot) :
    normBounded 2 (LowNormSlot.encode (kind slot) (env (offset + source slot))) :=
  LowNormSlot.encode_norm _ _ (kind_valid interface hints env offset inputs rows slot)

theorem helper_excluded (start : Nat) (slot : Slot) :
    start + HintProgram.helperCount ≤ Program.coreOffset start + source slot := by
  unfold Program.coreOffset
  omega

end NightstreamFPrime.Layout.PiRlcWideSampler
