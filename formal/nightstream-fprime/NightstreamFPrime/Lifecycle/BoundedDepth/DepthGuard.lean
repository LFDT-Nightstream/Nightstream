import NightstreamFPrime.Gadgets.Range.CanonicalU64
import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Spec.Folding.Schedule

/-! The light circuit's depth guard and the depth update of `stepHolds_iff` (CW-2.4b). The cap
is `Schedule.productionLightFoldCap = 64 = 2 ^ 6`, so the guard decomposes the slot's depth word
with `CanonicalU64` (67 rows) and forces bits 6 to 63 to zero (58 rows): the word is then below
64. The next depth word is the depth word plus one after a light fold and zero after a refresh;
both are affine, so they cost no rows. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.DepthGuard

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
    have modulus : fieldModulus < 2 ^ 64 := by decide
    omega
  rw [zero] at window
  have quotient : ((interface.source offset).eval env).val / 2 ^ 6 < 2 ^ 58 := by
    rw [Nat.div_lt_iff_lt_mul (by decide)]
    omega
  rw [Nat.mod_eq_of_lt quotient] at window
  simp only [productionLightFoldCap]
  omega

/-- The converse of `weightedValue_zero`: a zero weighted value has zero bits. -/
theorem bits_zero_of_weightedValue_zero (env : Env) (offset start : Nat) :
    ∀ count, CanonicalU64.weightedValue env offset start count = 0 →
      ∀ index, index < count → CanonicalU64.bitValue env offset (start + index) = 0
  | 0, _, _, bound => absurd bound (Nat.not_lt_zero _)
  | count + 1, zero, index, bound => by
      simp only [CanonicalU64.weightedValue, List.range_succ, List.foldl_append,
        List.foldl_cons, List.foldl_nil] at zero
      obtain ⟨lower, top⟩ := Nat.add_eq_zero_iff.mp zero
      rcases Nat.lt_succ_iff_lt_or_eq.mp bound with below | rfl
      · exact bits_zero_of_weightedValue_zero env offset start count lower index below
      · rcases Nat.mul_eq_zero.mp top with power | bit
        · exact absurd power (Nat.pow_pos (by decide)).ne'
        · exact bit

/-- The converse of `word_lt_cap`: a word below the cap has zero bits 6 to 63 in a canonical
decomposition, so an honest depth word satisfies the high-bit rows. -/
theorem highBitsZero_of_lt (interface : CanonicalU64.Interface) (env : Env) (offset : Nat)
    (spec : CanonicalU64.SpecHolds interface offset env)
    (small : ((interface.source offset).eval env).val < productionLightFoldCap) :
    HighBitsZero offset env := by
  have window := CanonicalU64.windowValue_eq interface env offset 6 58 spec (by decide)
  have smallPower : ((interface.source offset).eval env).val < 2 ^ 6 := by
    simpa [productionLightFoldCap] using small
  rw [Nat.div_eq_of_lt smallPower, Nat.zero_mod] at window
  intro index low high
  have bit := bits_zero_of_weightedValue_zero env offset 6 58 window (index - 6) (by
    simp only [CanonicalU64.bitCount] at high
    omega)
  rw [Nat.add_sub_cancel' low] at bit
  apply Fin.ext
  simpa [CanonicalU64.bitValue] using bit

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
    {depth : Nat} (depthBound : depth < fieldModulus)
    (word : (interface.source offset).eval env = natWord depth) :
    depth < productionLightFoldCap ∧
      natWord (nextDepth depth .light) = (interface.source offset).eval env + 1 := by
  have below := word_lt_cap interface env offset spec high
  rw [word] at below
  simp only [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt depthBound] at below
  rw [word]
  exact ⟨below, next_word depth .light⟩

/-- The high-bit rows: one `assertZero` for each of bits 6 to 63 of the decomposition at
`rangeOffset`. They add 58 rows and no variables. -/
def highBitOps (rangeOffset : Nat) : List Op :=
  (List.range 58).map fun index => Op.assertZero (Expr.var (rangeOffset + (index + 6)))

private theorem highBit_member (rangeOffset index : Nat) (low : 6 ≤ index)
    (high : index < CanonicalU64.bitCount) :
    Op.assertZero (Expr.var (rangeOffset + index)) ∈ highBitOps rangeOffset := by
  simp only [highBitOps, List.mem_map, List.mem_range]
  refine ⟨index - 6, ?_, ?_⟩
  · simp only [CanonicalU64.bitCount] at high
    omega
  · rw [Nat.sub_add_cancel low]

theorem highBitOps_varsBelow (rangeOffset : Nat) :
    ∀ expression ∈ flatConstraints (highBitOps rangeOffset),
      expression.VarsBelow (rangeOffset + CanonicalU64.bitCount) := by
  intro expression member
  simp only [flatConstraints, highBitOps, List.flatMap_map, List.mem_flatMap, List.mem_range,
    Op.flatConstraints, List.mem_singleton] at member
  obtain ⟨index, bound, rfl⟩ := member
  simp only [Expr.VarsBelow, CanonicalU64.bitCount]
  omega

def highBitsCircuit (rangeOffset : Nat) : FormalCircuit where
  main := fun offset => ((), offset, highBitOps rangeOffset)
  spec := fun _ env => HighBitsZero rangeOffset env
  soundness := by
    intro env _ _ rows index low high
    exact rows _ (highBit_member rangeOffset index low high)
  completeness := by
    intro env _ _ zero
    refine ⟨env, fun _ _ => rfl, ?_⟩
    intro row member
    simp only [Circuit.ops, flatConstraints, highBitOps, List.flatMap_map, List.mem_flatMap,
      List.mem_range, Op.flatConstraints, List.mem_singleton] at member
    obtain ⟨index, bound, rfl⟩ := member
    exact zero (index + 6) (by omega) (by simp only [CanonicalU64.bitCount]; omega)

end NightstreamFPrime.Lifecycle.BoundedDepth.DepthGuard
