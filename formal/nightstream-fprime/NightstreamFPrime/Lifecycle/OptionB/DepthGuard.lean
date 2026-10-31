import NightstreamFPrime.Gadgets.Range.CanonicalU64
import NightstreamFPrime.Lifecycle.OptionB.Step

/-! The light circuit's depth guard and the depth update of `stepHolds_iff` (CW-2.4b). The cap
is `Schedule.productionLightFoldCap = 64 = 2 ^ 6`, so the guard decomposes the slot's depth word
with `CanonicalU64` (67 rows) and forces bits 6 to 63 to zero (58 rows): the word is then below
64. The next depth word is the depth word plus one after a light fold and zero after a refresh;
both are affine, so they cost no rows. -/

namespace NightstreamFPrime.Lifecycle.OptionB.DepthGuard

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Spec.Folding.Schedule (Step nextDepth productionLightFoldCap)

/-- The high-bit rows: bits 6 to 63 of the decomposition at `offset` are zero. -/
def HighBitsZero (offset : Nat) (env : Env) : Prop :=
  ∀ index, 6 ≤ index → index < CanonicalU64.bitCount → env (offset + index) = 0

theorem weightedValue_zero (env : Env) (offset start : Nat) :
    ∀ count, (∀ index, index < count → CanonicalU64.bitValue env offset (start + index) = 0) →
      CanonicalU64.weightedValue env offset start count = 0
  | 0, _ => rfl
  | count + 1, zero => by
      have prior := weightedValue_zero env offset start count
        (fun index bound => zero index (by omega))
      simp only [CanonicalU64.weightedValue, List.range_succ, List.foldl_append,
        List.foldl_cons, List.foldl_nil] at prior ⊢
      rw [prior, zero count (by omega)]
      rfl

/-- The guard rows put the depth word below the cap. -/
theorem word_lt_cap (interface : CanonicalU64.Interface) (env : Env) (offset : Nat)
    (spec : CanonicalU64.SpecHolds interface offset env) (high : HighBitsZero offset env) :
    ((interface.source offset).eval env).val < productionLightFoldCap := by
  have window := CanonicalU64.windowValue_eq interface env offset 6 58 spec (by decide)
  have zero : CanonicalU64.weightedValue env offset 6 58 = 0 :=
    weightedValue_zero env offset 6 58 fun index bound => by
      simp only [CanonicalU64.bitValue]
      rw [high (6 + index) (by omega) (by simp [CanonicalU64.bitCount]; omega)]
      rfl
  have below : ((interface.source offset).eval env).val < 2 ^ 64 := by
    have bounded := ((interface.source offset).eval env).isLt
    have modulus : goldilocksModulus < 2 ^ 64 := by decide
    omega
  rw [zero] at window
  have quotient : ((interface.source offset).eval env).val / 2 ^ 6 < 2 ^ 58 := by
    rw [Nat.div_lt_iff_lt_mul (by decide)]
    omega
  rw [Nat.mod_eq_of_lt quotient] at window
  simp only [productionLightFoldCap]
  omega

/-- After a light fold the depth word grows by one; after a refresh it is zero. -/
theorem next_word (depth : Nat) (step : Step) :
    natWord (nextDepth depth step) =
      match step with
      | .light => natWord depth + 1
      | .refresh => 0 := by
  cases step with
  | light =>
      apply Fin.ext
      simp [natWord, Poseidon2.ofNat, nextDepth, Fin.val_add, Nat.add_mod]
  | refresh => rfl

/-- The depth terms of `stepHolds_iff` for a light fold: a depth word that the guard accepts is
below the cap, and the next word is that word plus one. -/
theorem light_terms (interface : CanonicalU64.Interface) (env : Env) (offset : Nat)
    (spec : CanonicalU64.SpecHolds interface offset env) (high : HighBitsZero offset env)
    {depth : Nat} (depthBound : depth < goldilocksModulus)
    (word : (interface.source offset).eval env = natWord depth) :
    depth < productionLightFoldCap ∧
      natWord (nextDepth depth .light) = (interface.source offset).eval env + 1 := by
  have below := word_lt_cap interface env offset spec high
  rw [word] at below
  simp only [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt depthBound] at below
  rw [word]
  exact ⟨below, next_word depth .light⟩

end NightstreamFPrime.Lifecycle.OptionB.DepthGuard
