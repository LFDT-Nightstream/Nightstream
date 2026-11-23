import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerCount
import NightstreamFPrime.Spec.Compression.Akita.FitChance

/-! The sign bits of the decoder of the pinned Akita code (`Replay/Sampler.lean`,
`crates/akita-challenges/src/sampler/stream.rs`: the low `w` bits of a candidate are the signs of
its nonzero coefficients) and the search bound of level 1 of `key20.aks` for this decoder (Akita,
ePrint 2026/1983, revision of 2026-11-23, Lemma K.3, Lemma 10.7).

- `flipSigns G ε v` changes the sign bit of each coefficient of the candidate `v` whose position
  `ε` flips. It does not change the rank, so it is an involution of the `K`-bit candidates
  (`flipEquiv`), and it changes the challenge `c` to `signed ε c` (`slotDecode_flip`). So an
  unfiltered decoder of the code has `SignFair` (`signFair_slotDecode`).
- `level₁_search_code`: `Key20.level₁_search_total` holds for the decoder of the code of the
  family `(31, 10)` of level 1 (`level₁Group`, the group of level 1 of `Replay.layout20`) on each
  block: each of the `2^12` response nonces fails with probability at most `(2^−49)^4096`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.RandomOracle (mass)
open Sparse

/-- The number with the bits of a list, the first bit lowest. -/
def maskOf : List Bool → ℕ
  | [] => 0
  | b :: bs => b.toNat + 2 * maskOf bs

theorem testBit_maskOf : ∀ (bs : List Bool) (i : ℕ), (maskOf bs).testBit i = bs.getD i false
  | [], i => by simp [maskOf]
  | b :: bs, 0 => by cases b <;> simp [maskOf]
  | b :: bs, i + 1 => by
      rw [maskOf, Nat.testBit_add_one,
        show (b.toNat + 2 * maskOf bs) / 2 = maskOf bs by
          cases b with
          | false => simp
          | true => simp; omega,
        testBit_maskOf bs i]
      simp

theorem maskOf_lt : ∀ (bs : List Bool), maskOf bs < 2 ^ bs.length
  | [] => by simp [maskOf]
  | b :: bs => by
      have below := maskOf_lt bs
      cases b <;> simp [maskOf, Nat.pow_succ] <;> omega

/-- The positions of the challenge of a candidate, or none. -/
def positionsOf (G : FoldGroup) (value : ℕ) : List ℕ :=
  ((candidate G Nat.choose value).map fun c => c.map Prod.fst).getD []

/-- The sign bits of a candidate that the sign flips `ε` change. -/
def signMask (G : FoldGroup) (ε : ℕ → Bool) (value : ℕ) : ℕ :=
  maskOf ((positionsOf G value).map ε)

def flipSigns (G : FoldGroup) (ε : ℕ → Bool) (value : ℕ) : ℕ := value ^^^ signMask G ε value

/-- Two candidates with the same rank have the same positions. -/
theorem candidate_rank {G : FoldGroup} {value value' : ℕ}
    (same : value / 2 ^ (G.pm1 + G.pm2) = value' / 2 ^ (G.pm1 + G.pm2)) :
    candidate G Nat.choose value' = (candidate G Nat.choose value).map fun c =>
      pairs G.pm1 (value' % 2 ^ (G.pm1 + G.pm2)) (c.map Prod.fst) := by
  simp only [candidate, same]
  split_ifs
  · rfl
  · simp [Option.map_bind, Function.comp_def, map_fst_pairs]

theorem signMask_lt (G : FoldGroup) (ε : ℕ → Bool) (value : ℕ) :
    signMask G ε value < 2 ^ (G.pm1 + G.pm2) := by
  unfold signMask positionsOf
  cases found : candidate G Nat.choose value with
  | none => simp [maskOf]
  | some c =>
      obtain ⟨ones, twos, rfl, P, length⟩ := candidate_shape found
      simp only [Option.map_some, Option.getD_some, map_fst_pairs]
      have bound := maskOf_lt ((ones ++ twos).map ε)
      rwa [List.length_map, List.length_append, P.length, length] at bound

theorem div_flipSigns (G : FoldGroup) (ε : ℕ → Bool) (value : ℕ) :
    flipSigns G ε value / 2 ^ (G.pm1 + G.pm2) = value / 2 ^ (G.pm1 + G.pm2) := by
  refine Nat.eq_of_testBit_eq fun i => ?_
  rw [Nat.testBit_div_two_pow, Nat.testBit_div_two_pow, flipSigns, Nat.testBit_xor,
    Nat.testBit_lt_two_pow ((signMask_lt G ε value).trans_le
      (Nat.pow_le_pow_right (by decide) (Nat.le_add_left _ _))), Bool.xor_false]

theorem mod_flipSigns (G : FoldGroup) (ε : ℕ → Bool) (value : ℕ) :
    flipSigns G ε value % 2 ^ (G.pm1 + G.pm2) =
      value % 2 ^ (G.pm1 + G.pm2) ^^^ signMask G ε value := by
  refine Nat.eq_of_testBit_eq fun i => ?_
  rw [Nat.testBit_mod_two_pow, flipSigns, Nat.testBit_xor, Nat.testBit_xor,
    Nat.testBit_mod_two_pow]
  by_cases small : i < G.pm1 + G.pm2
  · simp [small]
  · simp [small, Nat.testBit_lt_two_pow ((signMask_lt G ε value).trans_le
      (Nat.pow_le_pow_right (by decide) (Nat.le_of_not_lt small)))]

theorem candidate_flipSigns (G : FoldGroup) (ε : ℕ → Bool) (value : ℕ) :
    candidate G Nat.choose (flipSigns G ε value) = (candidate G Nat.choose value).map fun c =>
      pairs G.pm1 (value % 2 ^ (G.pm1 + G.pm2) ^^^ signMask G ε value) (c.map Prod.fst) := by
  rw [candidate_rank (div_flipSigns G ε value).symm, mod_flipSigns]

theorem flipSigns_flipSigns (G : FoldGroup) (ε : ℕ → Bool) (value : ℕ) :
    flipSigns G ε (flipSigns G ε value) = value := by
  have same : signMask G ε (flipSigns G ε value) = signMask G ε value := by
    simp only [signMask, positionsOf, candidate_flipSigns, Option.map_map, Function.comp_def,
      map_fst_pairs]
  rw [flipSigns, same, flipSigns, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

theorem flipSigns_lt {G : FoldGroup} (ε : ℕ → Bool) {value : ℕ}
    (small : value < 2 ^ G.candidateBits) : flipSigns G ε value < 2 ^ G.candidateBits :=
  Nat.xor_lt_two_pow small ((signMask_lt G ε value).trans_le
    (Nat.pow_le_pow_right (by decide) (by rw [FoldGroup.candidateBits_eq]; omega)))

/-- The sign flips of one sign vector as a bijection of the `K`-bit candidates. -/
def flipEquiv (G : FoldGroup) (ε : ℕ → Bool) : Fin (2 ^ G.candidateBits) ≃ Fin (2 ^ G.candidateBits) where
  toFun v := ⟨flipSigns G ε v, flipSigns_lt ε v.isLt⟩
  invFun v := ⟨flipSigns G ε v, flipSigns_lt ε v.isLt⟩
  left_inv v := Fin.ext (flipSigns_flipSigns G ε v)
  right_inv v := Fin.ext (flipSigns_flipSigns G ε v)

theorem coefficient_flip {ones signs signs' i : ℕ} {e : Bool}
    (bit : signs'.testBit i = (signs.testBit i ^^ e)) :
    coefficient ones signs' i =
      if e then -coefficient ones signs i else coefficient ones signs i := by
  unfold coefficient
  rw [bit]
  generalize (if i < ones then (1 : ℤ) else 2) = m
  cases signs.testBit i <;> cases e <;> simp

/-- Flipping the sign bits of the flipped positions gives `signed ε` of the challenge. -/
theorem ring_flip {d ones signs : ℕ} [NeZero d] {P : List ℕ} (distinct : P.Nodup) (ε : ℕ → Bool) :
    ring d (pairs ones (signs ^^^ maskOf (P.map ε)) P) = signed ε (ring d (pairs ones signs P)) := by
  refine ext_coeff fun j small => ?_
  rw [coeff_ring _ small, coeff_signed _ _ small, coeff_ring _ small]
  by_cases present : j ∈ P
  · obtain ⟨i, inside, rfl⟩ := List.getElem_of_mem present
    rw [value_pairs distinct inside, value_pairs distinct inside]
    have bit : (signs ^^^ maskOf (P.map ε)).testBit i = (signs.testBit i ^^ ε P[i]) := by
      rw [Nat.testBit_xor, testBit_maskOf]
      simp [inside]
    rw [coefficient_flip bit]
    by_cases flipped : ε P[i] <;> simp [flipped]
  · rw [value_eq_zero (by rw [map_fst_pairs]; exact present),
      value_eq_zero (by rw [map_fst_pairs]; exact present)]
    simp

/-- The decoder of the code without a filter, on a flipped candidate. -/
theorem slotDecode_flip {G : FoldGroup} [NeZero G.degree] (unfiltered : G.filtered = false)
    (ε : ℕ → Bool) (slot : Fin (2 ^ G.candidateBits)) :
    slotDecode G (flipEquiv G ε slot) = (slotDecode G slot).map (signed ε) := by
  have keep (o : Option Sparse) : o.filter (fun _ => true) = o := by cases o <;> rfl
  simp only [slotDecode, unfiltered, Bool.not_false, Bool.true_or, keep]
  rw [show ((flipEquiv G ε slot : Fin (2 ^ G.candidateBits)) : ℕ) = flipSigns G ε slot from rfl,
    candidate_flipSigns]
  cases found : candidate G Nat.choose slot with
  | none => rfl
  | some c =>
      obtain ⟨ones, twos, rfl, P, -⟩ := candidate_shape found
      simp only [Option.map_some, signMask, positionsOf, found, map_fst_pairs]
      exact congrArg some (ring_flip P.distinct ε)

/-- An unfiltered decoder of the code has `SignFair`: the sign flips of a challenge act on its
candidates. -/
theorem signFair_slotDecode {G : FoldGroup} [NeZero G.degree] (unfiltered : G.filtered = false) :
    SignFair (slotDecode G) :=
  ⟨fun ε => flipEquiv G (finSigns ε), fun ε slot => slotDecode_flip unfiltered (finSigns ε) slot⟩

/-- The group of level 1 of `Replay.layout20`: 28 blocks, the subring of degree 64 in `A` of
degree 1024, the family `(31, 10)`, no filter. -/
abbrev level₁Group : FoldGroup :=
  ⟨Key20.level₁.shape.blocks, Key20.level₁.shape.innerDegree, Key20.level₁.opening.carrierDegree,
    31, 10, false⟩

/-- `Key20.level₁_search_total` for the decoder of the code on each block of level 1 of
`key20.aks`: each of the `2^12` response nonces fails with probability at most `(2^−49)^4096`. -/
theorem level₁_search_code {Point Coord : Type} [Fintype Point] [DecidableEq Point] [Fintype Coord]
    [DecidableEq Coord] (input : Fin Extract.responseTries → Point)
    (distinct : Function.Injective input) (block : Fin Key20.level₁.shape.blocks → Coord)
    (injective : Function.Injective block) (source : Fin Key20.level₁.shape.blocks → Key20.level₁.shape.Source)
    (bounded : ∀ i col, normInf (source i col) ≤ 4) :
    mass {table : Point → Option Coord → Fin (2 ^ level₁Group.candidateBits) | ∀ n,
      (∃ i, slotDecode level₁Group (Extract.view (table (input n)) (block i)) = none) ∨
        ¬ Encodable Key20.level₁.fold.responseBase Key20.level₁.fold.responseDigits
          (response Key20.level₁.opening
            (fun i => (slotDecode level₁Group (Extract.view (table (input n)) (block i))).getD 0)
            source)} ≤
      (1 / 2 ^ 49) ^ Extract.responseTries :=
  Key20.level₁_search_total input distinct block injective (fun _ => slotDecode level₁Group)
    (fun _ slot => slotDecode_isSome (G := level₁Group) rfl (by decide) slot)
    (fun _ => signFair_slotDecode (G := level₁Group) rfl)
    (fun _ slot c decoded => (normSq_of_mem (slotDecode_mem decoded).1).le) source bounded

end NightstreamFPrime.Spec.Compression.Akita.Replay
