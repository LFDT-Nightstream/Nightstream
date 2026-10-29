import CompressionLean.Ports.Shrink.Diagonal
import CompressionLean.Lib.Cube
import Mathlib.Data.Nat.Size

namespace CompressionLean.Shrink

open CompressionLean.Lib

variable {K : Type} [CommRing K]

def trackEq (point : List K) (v : ℕ) : K := if v < 2 ^ point.length then eqAt point v else 0

def highProduct (t q : ℕ) : List (Track K) → ℕ → K
  | [], _ => 1
  | (point, offset) :: rest, carry =>
      trackEq (point.drop t) (offset / 2 ^ t + q + carry % 2) * highProduct t q rest (carry / 2)

def suffix (tracks : List (Track K)) (count t carry below : ℕ) : K :=
  ∑ q ∈ Finset.range (count / 2 ^ t + below), highProduct t q tracks carry

def pairing (tracks : List (Track K)) (count t : ℕ) (states : ℕ → ℕ → K) : K :=
  ∑ carry ∈ Finset.range (2 ^ tracks.length), ∑ below ∈ Finset.range 2,
    states carry below * suffix tracks count t carry below

theorem trackEq_step (point : List K) (offset t q b c : ℕ) :
    trackEq (point.drop t) (offset / 2 ^ t + (2 * q + b) + c) =
      trackFactor point t ((offset / 2 ^ t % 2 + b + c) % 2) *
        trackEq (point.drop (t + 1)) (offset / 2 ^ (t + 1) + q + (offset / 2 ^ t % 2 + b + c) / 2) := by
  have high : offset / 2 ^ (t + 1) = offset / 2 ^ t / 2 := by rw [pow_succ, Nat.div_div_eq_div_mul]
  rw [high]
  generalize offset / 2 ^ t = u
  have value : u + (2 * q + b) + c = 2 * (u / 2 + q + (u % 2 + b + c) / 2) + (u % 2 + b + c) % 2 := by
    omega
  have small : (u % 2 + b + c) % 2 < 2 := Nat.mod_lt _ (by norm_num)
  rw [value]
  generalize u / 2 + q + (u % 2 + b + c) / 2 = w
  generalize (u % 2 + b + c) % 2 = bit at small ⊢
  have low : (2 * w + bit) % 2 = bit := by omega
  have half : (2 * w + bit) / 2 = w := by omega
  unfold trackEq trackFactor
  by_cases inside : t < point.length
  · have get : point.getD t 0 = point[t] := by simp [List.getD_eq_getElem?_getD, inside]
    rw [List.drop_eq_getElem_cons inside, ite_eq_left inside, get]
    simp only [List.length_cons, eqAt, low, half]
    by_cases fits : w < 2 ^ (point.drop (t + 1)).length
    · rw [ite_eq_left (by rw [pow_succ]; omega), ite_eq_left fits]
    · rw [ite_eq_right (by rw [pow_succ]; omega), ite_eq_right fits, mul_zero]
  · rw [List.drop_eq_nil_of_le (by omega), List.drop_eq_nil_of_le (by omega), ite_eq_right inside]
    simp only [List.length_nil, pow_zero, eqAt]
    rcases (show bit = 0 ∨ bit = 1 by omega) with rfl | rfl
    · by_cases zero : w < 1
      · rw [ite_eq_left (by omega), ite_eq_left zero]
        simp
      · rw [ite_eq_right (by omega), ite_eq_right zero]
        simp
    · rw [ite_eq_right (by omega)]
      simp

theorem highProduct_step (t q b : ℕ) (bit : b < 2) :
    ∀ (tracks : List (Track K)) (carry : ℕ),
      highProduct t (2 * q + b) tracks carry =
        (tracksStep t b tracks carry).1 * highProduct (t + 1) q tracks (tracksStep t b tracks carry).2
  | [], _ => by simp [highProduct, tracksStep]
  | (point, offset) :: rest, carry => by
      simp only [highProduct, tracksStep]
      rw [trackEq_step point offset t q b (carry % 2), highProduct_step t q b bit rest (carry / 2)]
      generalize (tracksStep t b rest (carry / 2)).2 = next
      generalize (tracksStep t b rest (carry / 2)).1 = factor
      have low : ((offset / 2 ^ t % 2 + b + carry % 2) / 2 + 2 * next) % 2 =
          (offset / 2 ^ t % 2 + b + carry % 2) / 2 := by omega
      have high : ((offset / 2 ^ t % 2 + b + carry % 2) / 2 + 2 * next) / 2 = next := by omega
      simp only [low, high]
      ring

theorem sum_range_parity {M : Type} [AddCommMonoid M] (f : ℕ → M) :
    ∀ N, ∑ q ∈ Finset.range N, f q =
      ∑ q ∈ Finset.range ((N + 1) / 2), f (2 * q) + ∑ q ∈ Finset.range (N / 2), f (2 * q + 1)
  | 0 => by simp
  | N + 1 => by
      rw [Finset.sum_range_succ, sum_range_parity f N]
      rcases Nat.even_or_odd' N with ⟨m, rfl | rfl⟩
      · rw [show (2 * m + 1 + 1) / 2 = m + 1 by omega, show (2 * m + 1) / 2 = m by omega,
          show 2 * m / 2 = m by omega, Finset.sum_range_succ (fun q => f (2 * q))]
        abel
      · rw [show (2 * m + 1 + 1 + 1) / 2 = m + 1 by omega, show (2 * m + 1 + 1) / 2 = m + 1 by omega,
          show (2 * m + 1) / 2 = m by omega, Finset.sum_range_succ (fun q => f (2 * q + 1))]
        abel

theorem suffix_step (tracks : List (Track K)) (count t carry below : ℕ) (small : below < 2) :
    suffix tracks count t carry below =
      ∑ b ∈ Finset.range 2, (tracksStep t b tracks carry).1 *
        suffix tracks count (t + 1) (tracksStep t b tracks carry).2 (nextBelow (count / 2 ^ t % 2) b below) := by
  have high : count / 2 ^ (t + 1) = count / 2 ^ t / 2 := by rw [pow_succ, Nat.div_div_eq_div_mul]
  have even : (count / 2 ^ t + below + 1) / 2 = count / 2 ^ (t + 1) + nextBelow (count / 2 ^ t % 2) 0 below := by
    rw [high]
    unfold nextBelow
    split_ifs <;> omega
  have odd : (count / 2 ^ t + below) / 2 = count / 2 ^ (t + 1) + nextBelow (count / 2 ^ t % 2) 1 below := by
    rw [high]
    unfold nextBelow
    split_ifs <;> omega
  unfold suffix
  rw [sum_range_parity, Finset.sum_range_succ, Finset.sum_range_one, even, odd, Finset.mul_sum, Finset.mul_sum]
  congr 1
  · apply Finset.sum_congr rfl
    intro q _
    have := highProduct_step t q 0 (by norm_num) tracks carry
    rwa [add_zero] at this
  · apply Finset.sum_congr rfl
    intro q _
    exact highProduct_step t q 1 (by norm_num) tracks carry

theorem tracksStep_carry_lt (t b : ℕ) (bit : b < 2) :
    ∀ (tracks : List (Track K)) (carry : ℕ), (tracksStep t b tracks carry).2 < 2 ^ tracks.length
  | [], _ => by simp [tracksStep]
  | (point, offset) :: rest, carry => by
      have := tracksStep_carry_lt t b bit rest (carry / 2)
      simp only [tracksStep, List.length_cons, pow_succ]
      omega

theorem highProduct_end (t : ℕ) :
    ∀ (tracks : List (Track K)) (carry : ℕ), (∀ track ∈ tracks, track.1.length ≤ t ∧ track.2 < 2 ^ t) →
      carry < 2 ^ tracks.length → highProduct t 0 tracks carry = if carry = 0 then 1 else 0
  | [], carry, _, lt => by
      rw [List.length_nil, pow_zero, Nat.lt_one_iff] at lt
      simp [highProduct, lt]
  | (point, offset) :: rest, carry, big, lt => by
      obtain ⟨short, small⟩ := big (point, offset) List.mem_cons_self
      have restLt : carry / 2 < 2 ^ rest.length := by
        rw [List.length_cons, pow_succ] at lt
        omega
      simp only [highProduct]
      rw [highProduct_end t rest (carry / 2) (fun track member => big track (List.mem_cons_of_mem _ member)) restLt,
        List.drop_eq_nil_of_le short, Nat.div_eq_of_lt small]
      simp only [trackEq, List.length_nil, pow_zero, eqAt, zero_add]
      by_cases zero : carry = 0
      · simp [zero]
      · by_cases low : carry % 2 < 1
        · rw [ite_eq_left low, ite_eq_right (by omega), ite_eq_right zero, mul_zero]
        · rw [ite_eq_right low, zero_mul, ite_eq_right zero]

theorem suffix_end (tracks : List (Track K)) (count t carry below : ℕ)
    (big : ∀ track ∈ tracks, track.1.length ≤ t ∧ track.2 < 2 ^ t) (countLt : count < 2 ^ t)
    (carryLt : carry < 2 ^ tracks.length) (small : below < 2) :
    suffix tracks count t carry below = if carry = 0 ∧ below = 1 then 1 else 0 := by
  unfold suffix
  rw [Nat.div_eq_of_lt countLt, zero_add]
  rcases (show below = 0 ∨ below = 1 by omega) with rfl | rfl
  · simp
  · rw [Finset.sum_range_one, highProduct_end t tracks carry big carryLt]
    simp

theorem scatter_pairing {ι : Type} (A : Finset ι) (X Y : Finset ℕ) (key1 key2 : ι → ℕ) (v : ι → K)
    (g : ℕ → ℕ → K) (in1 : ∀ i ∈ A, key1 i ∈ X) (in2 : ∀ i ∈ A, key2 i ∈ Y) :
    ∑ x ∈ X, ∑ y ∈ Y, (∑ i ∈ A, if key1 i = x ∧ key2 i = y then v i else 0) * g x y =
      ∑ i ∈ A, v i * g (key1 i) (key2 i) := by
  simp only [Finset.sum_mul]
  rw [Finset.sum_congr rfl fun x _ => Finset.sum_comm, Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i member
  rw [Finset.sum_eq_single_of_mem (key1 i) (in1 i member)
      (fun x _ ne => Finset.sum_eq_zero fun y _ => by rw [ite_eq_right (fun h => ne h.1.symm), zero_mul]),
    Finset.sum_eq_single_of_mem (key2 i) (in2 i member)
      (fun y _ ne => by rw [ite_eq_right (fun h => ne h.2.symm), zero_mul]),
    ite_eq_left ⟨rfl, rfl⟩]

omit [CommRing K] in
theorem sum_sum_point {M : Type} [AddCommMonoid M] (X Y : Finset ℕ) (a c : ℕ) (inX : a ∈ X) (inY : c ∈ Y)
    (f : ℕ → ℕ → M) : ∑ x ∈ X, ∑ y ∈ Y, (if x = a then if y = c then f x y else 0 else 0) = f a c := by
  rw [Finset.sum_eq_single_of_mem a inX (fun x _ ne => Finset.sum_eq_zero fun y _ => ite_eq_right ne),
    Finset.sum_eq_single_of_mem c inY (fun y _ ne => by rw [ite_eq_left rfl, ite_eq_right ne]), ite_eq_left rfl, ite_eq_left rfl]

theorem walkStep_eq (tracks : List (Track K)) (count t : ℕ) (states : ℕ → ℕ → K) (carry' below' : ℕ) :
    walkStep tracks count t states carry' below' =
      ∑ i ∈ (Finset.range (2 ^ tracks.length) ×ˢ Finset.range 2) ×ˢ Finset.range 2,
        if (tracksStep t i.2 tracks i.1.1).2 = carry' ∧ nextBelow (count / 2 ^ t % 2) i.2 i.1.2 = below' then
          states i.1.1 i.1.2 * (tracksStep t i.2 tracks i.1.1).1
        else 0 := by
  rw [Finset.sum_product, Finset.sum_product]
  rfl

theorem pairing_step (tracks : List (Track K)) (count t : ℕ) (states : ℕ → ℕ → K) :
    pairing tracks count (t + 1) (walkStep tracks count t states) = pairing tracks count t states := by
  unfold pairing
  simp only [walkStep_eq]
  rw [scatter_pairing _ _ _ _ _ _ _
    (fun i member => Finset.mem_range.mpr (tracksStep_carry_lt t i.2
      (Finset.mem_range.mp (Finset.mem_product.mp member).2) tracks i.1.1))
    (fun i member => by
      have bit := Finset.mem_range.mp (Finset.mem_product.mp member).2
      have below := Finset.mem_range.mp (Finset.mem_product.mp (Finset.mem_product.mp member).1).2
      simp only [Finset.mem_range]
      unfold nextBelow
      split_ifs <;> omega),
    Finset.sum_product, Finset.sum_product]
  apply Finset.sum_congr rfl
  intro carry _
  apply Finset.sum_congr rfl
  intro below member
  rw [suffix_step tracks count t carry below (Finset.mem_range.mp member), Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro b _
  ring

theorem pairing_walk (tracks : List (Track K)) (count : ℕ) :
    ∀ t, pairing tracks count t (walk tracks count t) = suffix tracks count 0 0 0
  | 0 => by
      unfold pairing
      simp only [walk, ite_mul, one_mul, zero_mul, ite_and]
      exact sum_sum_point _ _ 0 0 (Finset.mem_range.mpr (pow_pos (by norm_num) _)) (by simp) _
  | t + 1 => by rw [walk, pairing_step, pairing_walk tracks count t]

omit [CommRing K] in
theorem walkBits_covers (tracks : List (Track K)) (count : ℕ) :
    (∀ track ∈ tracks, track.1.length ≤ walkBits tracks count ∧ track.2 < 2 ^ walkBits tracks count) ∧
      count < 2 ^ walkBits tracks count := by
  have foldr (l : List ℕ) (init : ℕ) : init ≤ l.foldr max init ∧ ∀ x ∈ l, x ≤ l.foldr max init := by
    induction l with
    | nil => simp
    | cons a l ih =>
        simp only [List.foldr_cons, List.mem_cons, forall_eq_or_imp]
        exact ⟨ih.1.trans (le_max_right _ _), le_max_left _ _,
          fun x member => (ih.2 x member).trans (le_max_right _ _)⟩
  obtain ⟨countLe, trackLe⟩ := foldr (tracks.map fun track => max track.1.length (Nat.size track.2)) (Nat.size count)
  refine ⟨fun track member => ?_, Nat.size_le.mp countLe⟩
  have le := trackLe _ (List.mem_map.mpr ⟨track, member, rfl⟩)
  exact ⟨(le_max_left _ _).trans le, Nat.size_le.mp ((le_max_right _ _).trans le)⟩

theorem highProduct_zero (q : ℕ) :
    ∀ tracks : List (Track K), highProduct 0 q tracks 0 = (tracks.map fun track => trackEq track.1 (track.2 + q)).prod
  | [] => rfl
  | (point, offset) :: rest => by
      simp only [highProduct, List.drop_zero, pow_zero, Nat.div_one, Nat.zero_mod, add_zero, Nat.zero_div,
        highProduct_zero q rest, List.map_cons, List.prod_cons]

theorem diagonal_eq_sum (tracks : List (Track K)) (count : ℕ) :
    diagonal tracks count =
      ∑ b ∈ Finset.range count, (tracks.map fun track => trackEq track.1 (track.2 + b)).prod := by
  obtain ⟨big, countLt⟩ := walkBits_covers tracks count
  have last := pairing_walk tracks count (walkBits tracks count)
  unfold pairing at last
  rw [Finset.sum_congr rfl fun carry member => Finset.sum_congr rfl fun below below' => by
    rw [suffix_end tracks count _ carry below big countLt (Finset.mem_range.mp member) (Finset.mem_range.mp below')]]
    at last
  simp only [mul_ite, mul_one, mul_zero, ite_and] at last
  rw [sum_sum_point _ _ 0 1 (Finset.mem_range.mpr (pow_pos (by norm_num) _)) (by simp)] at last
  unfold diagonal
  rw [last]
  unfold suffix
  simp only [pow_zero, Nat.div_one, add_zero, highProduct_zero]

theorem diagonal_eq (tracks : List (Track K)) (count : ℕ)
    (fits : ∀ track ∈ tracks, ∀ b < count, track.2 + b < 2 ^ track.1.length) :
    diagonal tracks count =
      ∑ b ∈ Finset.range count, (tracks.map fun track => eqAt track.1 (track.2 + b)).prod := by
  rw [diagonal_eq_sum]
  apply Finset.sum_congr rfl
  intro b member
  congr 1
  apply List.map_congr_left
  intro track trackMember
  rw [trackEq, ite_eq_left (fits track trackMember b (Finset.mem_range.mp member))]

end CompressionLean.Shrink
