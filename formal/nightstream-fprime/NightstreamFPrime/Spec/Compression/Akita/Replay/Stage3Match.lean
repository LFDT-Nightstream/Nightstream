import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Table
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Sum
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3
import NightstreamFPrime.Spec.Compression.Algebra.Point

/-! The weight of Stage 3 of the replay (`stage3Weight`, `Replay/Stage3Weight.lean`) is the weight
of Stage 3 of the specification (`Batch.codeScan`, `Extract/SetupFactor.lean`) at the point `ρ`
of Stage 3, for the plan `SetupPlan.ofBatch B (groupOrder n) κ logBasis` of the code (Akita
revision `d50f700`, `akita-types/src/setup_contribution/`; ePrint 2026/1983, Section 9 with (133),
(134) and the view (35)).

- `sum_pow_eqAt`: `Σ_{m<2^c} b^m·ẽq(r, q·2^c + m) = powerScale(b, r[..c])·ẽq(r[c..], q)`, the
  factored form of the code for a lane (`laneIn_pow`, `runAt_lane`) and for an entry of a view
  (`setupAt_split`).
- `groupOf B g` is the group `g` of the plan. Its cells, rows and columns of `D` are those of the
  specification (`unitLen_eq`, `rows_eq`, `dCols_eq`), its slices start where the slices of the
  shape start (`sliceStart_eq`), and its widest slice is `Shape.sliceLength` (`maxSlice_eq`).
- `inner_match`, `outer_match` and `open_match` match the parts of `A`, `B` and `D` of one group;
  `open_groups` splits the part of `D` of the specification by groups.
- `stage3Weight_codeScan`: the weight of the replay at `(r_2, ρ)` is the multilinear extension at
  `ρ` of the weight of `codeScan` at `r_2`. `stage3Run_codeScan` is `stage3Run_runs` with this
  weight. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Sums

variable {E : Type} [CommRing E]

theorem eqAt_even (x : E) (rest : List E) (k : ℕ) :
    eqAt (x :: rest) (2 * k) = (1 - x) * eqAt rest k := by
  show (if 2 * k % 2 = 1 then x else 1 - x) * eqAt rest (2 * k / 2) = _
  rw [ite_eq_right (by omega), Nat.mul_div_cancel_left k two_pos]

theorem eqAt_odd (x : E) (rest : List E) (k : ℕ) :
    eqAt (x :: rest) (2 * k + 1) = x * eqAt rest k := by
  show (if (2 * k + 1) % 2 = 1 then x else 1 - x) * eqAt rest ((2 * k + 1) / 2) = _
  rw [ite_eq_left (by omega), show (2 * k + 1) / 2 = k by omega]

/-- `Σ_{m<2^c} b^m·ẽq(r, q·2^c + m) = powerScale(b, r[..c])·ẽq(r[c..], q)` for `c ≤ |r|`. -/
theorem sum_pow_eqAt : ∀ (c : ℕ) (b : E) (r : List E) (q : ℕ), c ≤ r.length →
    ∑ m ∈ Finset.range (2 ^ c), b ^ m * eqAt r (q * 2 ^ c + m) =
      powerScale b (r.take c) * eqAt (r.drop c) q
  | 0, b, r, q, _ => by simp [powerScale]
  | c + 1, _, [], _, short => by simp at short
  | c + 1, b, x :: rest, q, short => by
      have inner := sum_pow_eqAt c (b * b) rest q (by simpa using short)
      have split : ∀ m, b ^ (2 * m) * eqAt (x :: rest) (q * (2 * 2 ^ c) + 2 * m) +
          b ^ (2 * m + 1) * eqAt (x :: rest) (q * (2 * 2 ^ c) + (2 * m + 1)) =
            (1 - x + x * b) * ((b * b) ^ m * eqAt rest (q * 2 ^ c + m)) := fun m => by
        rw [show q * (2 * 2 ^ c) + 2 * m = 2 * (q * 2 ^ c + m) by ring,
          show q * (2 * 2 ^ c) + (2 * m + 1) = 2 * (q * 2 ^ c + m) + 1 by ring, eqAt_even,
          eqAt_odd]
        ring
      rw [pow_succ', Algebra.sum_range_double]
      simp only [split]
      rw [← Finset.mul_sum, inner]
      simp only [List.take_succ_cons, List.drop_succ_cons, powerScale]
      ring

/-- `powerScale(α, ρ[..a+b]) = powerScale(α, ρ[..a])·powerScale(α^{2^a}, ρ[a..a+b])`. -/
theorem powerScale_take_add : ∀ (a b : ℕ) (α : E) (ρ : List E),
    powerScale α (ρ.take (a + b)) =
      powerScale α (ρ.take a) * powerScale (α ^ 2 ^ a) ((ρ.drop a).take b)
  | 0, b, α, ρ => by simp [powerScale]
  | a + 1, b, α, [] => by simp [powerScale]
  | a + 1, b, α, x :: rest => by
      have power : (α * α) ^ 2 ^ a = α ^ 2 ^ (a + 1) := by
        rw [← pow_two, ← pow_mul, pow_succ' 2 a]
      rw [show a + 1 + b = a + b + 1 by omega]
      simp only [List.take_succ_cons, List.drop_succ_cons, powerScale]
      rw [powerScale_take_add a b (α * α) rest, power]
      ring

/-- The weight at `ρ` of an entry of a view of degree `2^k` in the factored form of the code: the
coordinates of `ρ` below `e`, from `e` to `k`, and above `k`. -/
theorem setupAt_split (ρ : List E) (α : E) {d k e : ℕ} (x : ℕ) (power : d = 2 ^ k) (le : e ≤ k)
    (short : k ≤ ρ.length) :
    setupAt ρ α d x = powerScale α (ρ.take e) *
      (powerScale (α ^ 2 ^ e) ((ρ.drop e).take (k - e)) * eqAt ((ρ.drop e).drop (k - e)) x) := by
  have whole : setupAt ρ α d x = powerScale α (ρ.take k) * eqAt (ρ.drop k) x := by
    subst power
    rw [← sum_pow_eqAt k α ρ x short]
    unfold setupAt
    exact Finset.sum_congr rfl fun l _ => mul_comm _ _
  have dropped : (ρ.drop e).drop (k - e) = ρ.drop k := by
    rw [List.drop_drop]
    congr 1
    omega
  rw [whole, dropped, ← mul_assoc, ← powerScale_take_add, Nat.add_sub_cancel' le]

end Sums

section Lanes

variable {E : Type} [Field E]

/-- `laneIn` for a ratio `2^c`: the lane sum, also in the factored branch. -/
theorem laneIn_pow (r : List E) (b : E) {c : ℕ} (ℓ : ℕ) (short : c ≤ r.length) :
    laneIn r b (2 ^ c) ℓ = ∑ m ∈ Finset.range (2 ^ c), b ^ m * eqAt r (ℓ + m) := by
  unfold laneIn
  split_ifs with multiple
  · obtain ⟨q, rfl⟩ := Nat.dvd_of_mod_eq_zero multiple
    rw [Nat.log_pow (by norm_num), Nat.mul_div_cancel_left q (by positivity), mul_comm (2 ^ c) q,
      sum_pow_eqAt c b r q short]
  · rfl

/-- The run of `d = 2^k` cells from the cell `s` is the lane of ratio `d/2^κ` from `s/2^κ`. -/
theorem runAt_lane (r : List E) (α : E) {κ k : ℕ} (d s : ℕ) (power : d = 2 ^ k) (le : κ ≤ k)
    (short : k ≤ r.length + κ) :
    runAt r α κ d s = laneIn r (α ^ 2 ^ κ) (d / 2 ^ κ) (s / 2 ^ κ) := by
  have ratio : d / 2 ^ κ = 2 ^ (k - κ) := by rw [power, Nat.pow_div le (by norm_num)]
  rw [ratio, laneIn_pow r _ _ (by omega)]
  unfold runAt
  rw [ratio]
  exact Finset.sum_congr rfl fun m _ => mul_comm _ _

end Lanes

section Numbers

theorem div_add_mul (a y : ℕ) {m d : ℕ} (divides : m ∣ d) :
    (a + y * d) / m = a / m + y * (d / m) := by
  obtain ⟨t, rfl⟩ := divides
  rcases Nat.eq_zero_or_pos m with zero | positive
  · subst zero
    simp
  · rw [show y * (m * t) = m * (y * t) by ring, Nat.add_mul_div_left _ _ positive,
      Nat.mul_div_cancel_left t positive]

theorem log_le_of {d k m : ℕ} (power : d = 2 ^ k) (le : d ≤ 2 ^ m) : k ≤ m :=
  (Nat.pow_le_pow_iff_right (by norm_num)).mp (power ▸ le)

theorem pow_log_of {d k : ℕ} (power : d = 2 ^ k) : 2 ^ Nat.log 2 d = d := by
  rw [power, Nat.log_pow (by norm_num)]

theorem log_ge_of {d k e : ℕ} (power : d = 2 ^ k) (le : 2 ^ e ≤ d) : e ≤ k :=
  (Nat.pow_le_pow_iff_right (by norm_num)).mp (power ▸ le)

theorem offset_rowsBefore {n : ℕ} (f : Fin n → ℕ) (k : Fin n) : offset f k = rowsBefore f k := by
  unfold offset rowsBefore
  rw [← Finset.sum_filter]
  symm
  refine Finset.sum_bij' (fun j _ => Fin.castLE k.2.le j)
    (fun g member => ⟨g, (Finset.mem_filter.mp member).2⟩) (fun j _ => ?_)
    (fun _ _ => Finset.mem_univ _) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
  exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, j.isLt⟩

theorem div_add_le (a b c : ℕ) (positive : 0 < c) : (a + b) / c ≤ a / c + (b + c - 1) / c := by
  have small := Nat.mod_lt a positive
  conv_lhs => rw [← Nat.div_add_mod a c]
  rw [Nat.add_assoc, Nat.mul_add_div positive]
  exact Nat.add_le_add_left (Nat.div_le_div_right (by omega)) _

/-- The last of `S` slices of `B` blocks has `⌈B/S⌉` blocks. -/
theorem last_slice (B S : ℕ) (positive : 0 < S) : B - (S - 1) * B / S = (B + S - 1) / S := by
  have upper := Nat.div_mul_le_self (B + S - 1) S
  have split := Nat.div_add_mod (B + S - 1) S
  have rest := Nat.mod_lt (B + S - 1) positive
  have spread : B ≤ B * S := Nat.le_mul_of_pos_right B positive
  have small : (B + S - 1) / S ≤ B := by
    have := (Nat.div_lt_iff_lt_mul positive).mpr
      (show B + S - 1 < (B + 1) * S by rw [add_mul, one_mul]; omega)
    omega
  have scaled : (B + S - 1) / S * S ≤ B * S := Nat.mul_le_mul_right S small
  rw [mul_comm S] at split
  have value : (S - 1) * B / S = B - (B + S - 1) / S := by
    apply Nat.div_eq_of_lt_le
    · rw [Nat.sub_mul B, Nat.sub_one_mul S B, mul_comm S B]
      omega
    · rw [Nat.add_mul, Nat.sub_mul B, Nat.sub_one_mul S B, mul_comm S B, one_mul]
      omega
  rw [value]
  generalize (B + S - 1) / S = L at *
  omega

theorem foldl_max_eq (f : ℕ → ℕ) (c : ℕ) : ∀ (l : List ℕ) (a : ℕ), a ≤ c → (∀ s ∈ l, f s ≤ c) →
    (a = c ∨ ∃ s ∈ l, f s = c) → l.foldl (fun m s => max m (f s)) a = c
  | [], a, _, _, reach => by
      rcases reach with same | ⟨s, member, _⟩
      · exact same
      · simp at member
  | s :: rest, a, below, each, reach => by
      have head : f s ≤ c := each s (by simp)
      have step : max a (f s) = c ∨ ∃ t ∈ rest, f t = c := by
        rcases reach with same | ⟨t, member, hit⟩
        · left
          rw [same]
          exact max_eq_left head
        · rcases List.mem_cons.mp member with equal | inside
          · left
            rw [← equal, hit]
            exact max_eq_right below
          · exact Or.inr ⟨t, inside, hit⟩
      exact foldl_max_eq f c rest (max a (f s)) (max_le below head)
        (fun t member => each t (by simp [member])) step

theorem foldl_min_pow (f : SetupGroup → ℕ) : ∀ (l : List SetupGroup) (a : ℕ),
    (∀ G ∈ l, ∃ k, f G = 2 ^ k) →
      ∃ e, e ≤ a ∧ l.foldl (fun m G => min m (f G)) (2 ^ a) = 2 ^ e ∧ ∀ G ∈ l, 2 ^ e ≤ f G
  | [], a, _ => ⟨a, le_rfl, rfl, fun _ member => by simp at member⟩
  | G :: rest, a, powers => by
      obtain ⟨k, hk⟩ := powers G (by simp)
      have least : min (2 ^ a) (f G) = 2 ^ min a k := by
        rw [hk]
        rcases le_total a k with h | h
        · rw [min_eq_left h, min_eq_left (Nat.pow_le_pow_right (by norm_num) h)]
        · rw [min_eq_right h, min_eq_right (Nat.pow_le_pow_right (by norm_num) h)]
      obtain ⟨e, le, value, each⟩ :=
        foldl_min_pow f rest (min a k) fun G' member => powers G' (by simp [member])
      refine ⟨e, le.trans (min_le_left _ _), ?_, fun G' member => ?_⟩
      · simp only [List.foldl_cons]
        rw [least, value]
      · rcases List.mem_cons.mp member with equal | inside
        · rw [equal, hk]
          exact Nat.pow_le_pow_right (by norm_num) (le.trans (min_le_right _ _))
        · exact each G' inside

end Numbers

section Plan

variable {n : ℕ} (B : Batch (Fin n))

/-- The group `g` of `SetupPlan.ofBatch`. -/
def groupOf (g : Fin n) : SetupGroup where
  innerDegree := (B.shape g).innerDegree
  outerDegree := (B.shape g).outerDegree
  innerRank := (B.shape g).innerRank
  positions := (B.shape g).blockLength
  witnessDigits := (B.shape g).sourceDigits
  foldDigits := (B.fold g).responseDigits
  commitDigits := (B.shape g).openingDigits
  openDigits := (B.shape g).openingDigits
  blocks := (B.shape g).blocks
  slices := (B.shape g).slices
  outerRank := (B.shape g).outerRank
  openWidth := B.planes g * B.carrierDegree g

theorem ofBatch_groups (order : Equiv.Perm (Fin n)) (κ logBasis : ℕ) :
    (SetupPlan.ofBatch B order κ logBasis).groups = List.ofFn fun k => groupOf B (order k) := by
  rw [List.ofFn_eq_map]
  rfl

theorem unitLen_eq (g : Fin n) : (groupOf B g).unitLen = B.mainCells g := by
  have openSplit : B.planes g * B.carrierDegree g / B.openDegree * B.openDegree =
      B.planes g * B.carrierDegree g := Nat.div_mul_cancel (B.openDivides g)
  have innerSplit : (B.shape g).innerDegree / (B.shape g).outerDegree * (B.shape g).outerDegree =
      (B.shape g).innerDegree := Nat.div_mul_cancel (B.shape g).divides
  show (B.shape g).blockLength * (B.shape g).sourceDigits * (B.fold g).responseDigits *
      (B.shape g).innerDegree + (B.shape g).blocks * (B.shape g).openingDigits *
        (B.planes g * B.carrierDegree g) + (B.shape g).blocks * (B.shape g).innerRank *
          (B.shape g).openingDigits * (B.shape g).innerDegree =
    (B.shape g).blockLength * (B.shape g).sourceDigits * (B.fold g).responseDigits *
      (B.shape g).innerDegree + (B.shape g).blocks *
        (B.planes g * B.carrierDegree g / B.openDegree) * (B.shape g).openingDigits *
          B.openDegree + (B.shape g).blocks * (B.shape g).innerRank *
            ((B.shape g).innerDegree / (B.shape g).outerDegree) * (B.shape g).openingDigits *
              (B.shape g).outerDegree
  rw [show (B.shape g).blocks * (B.planes g * B.carrierDegree g / B.openDegree) *
      (B.shape g).openingDigits * B.openDegree = (B.shape g).blocks * (B.shape g).openingDigits *
        (B.planes g * B.carrierDegree g / B.openDegree * B.openDegree) by ring,
    show (B.shape g).blocks * (B.shape g).innerRank *
      ((B.shape g).innerDegree / (B.shape g).outerDegree) * (B.shape g).openingDigits *
        (B.shape g).outerDegree = (B.shape g).blocks * (B.shape g).innerRank *
          (B.shape g).openingDigits *
            ((B.shape g).innerDegree / (B.shape g).outerDegree * (B.shape g).outerDegree) by ring,
    openSplit, innerSplit]

theorem rows_eq (g : Fin n) : (groupOf B g).rows = B.groupCount g := by
  show 1 + (B.shape g).innerRank + (B.shape g).slices * (B.shape g).outerRank =
    1 + ((B.shape g).innerRank + (B.shape g).slices * (B.shape g).outerRank)
  rw [Nat.add_assoc]

theorem dCols_eq (g : Fin n) : (groupOf B g).dCols B.openDegree =
    (B.shape g).blocks * (B.opening g).subcolumns * (B.shape g).openingDigits :=
  rfl

theorem zLen_eq (g : Fin n) : (groupOf B g).zLen = (B.shape g).responseCells (B.fold g) :=
  rfl

theorem innerStart_eq (g : Fin n) :
    (B.opening g).innerStart (B.fold g) = (groupOf B g).zLen + (groupOf B g).eLen := by
  have openSplit : B.planes g * B.carrierDegree g / B.openDegree * B.openDegree =
      B.planes g * B.carrierDegree g := Nat.div_mul_cancel (B.openDivides g)
  show (B.shape g).blockLength * (B.shape g).sourceDigits * (B.fold g).responseDigits *
      (B.shape g).innerDegree + (B.shape g).blocks *
        (B.planes g * B.carrierDegree g / B.openDegree) * (B.shape g).openingDigits *
          B.openDegree =
    (B.shape g).blockLength * (B.shape g).sourceDigits * (B.fold g).responseDigits *
      (B.shape g).innerDegree + (B.shape g).blocks * (B.shape g).openingDigits *
        (B.planes g * B.carrierDegree g)
  rw [show (B.shape g).blocks * (B.planes g * B.carrierDegree g / B.openDegree) *
      (B.shape g).openingDigits * B.openDegree = (B.shape g).blocks * (B.shape g).openingDigits *
        (B.planes g * B.carrierDegree g / B.openDegree * B.openDegree) by ring, openSplit]

theorem sliceStart_eq (S : Shape) (s : ℕ) :
    S.blocks / S.slices * s + S.blocks % S.slices * s / S.slices = S.sliceStart s := by
  have positive : 0 < S.slices := Nat.pos_of_ne_zero (NeZero.ne _)
  unfold Shape.sliceStart
  conv_rhs => rw [← Nat.div_add_mod S.blocks S.slices]
  rw [show s * (S.slices * (S.blocks / S.slices) + S.blocks % S.slices) =
      S.slices * (s * (S.blocks / S.slices)) + S.blocks % S.slices * s by ring,
    Nat.mul_add_div positive]
  ring

theorem groupSliceStart (g : Fin n) (s : ℕ) :
    (groupOf B g).sliceStart s = (B.shape g).sliceStart s :=
  sliceStart_eq (B.shape g) s

/-- The widest slice of the group is `Shape.sliceLength`, the width of the view of `B`. -/
theorem maxSlice_eq (g : Fin n) : (groupOf B g).maxSlice = (B.shape g).sliceLength := by
  have positive : 0 < (B.shape g).slices := Nat.pos_of_ne_zero (NeZero.ne _)
  have start : ∀ s, (groupOf B g).sliceStart s = s * (B.shape g).blocks / (B.shape g).slices :=
    groupSliceStart B g
  unfold SetupGroup.maxSlice
  refine foldl_max_eq _ _ _ 0 (Nat.zero_le _) (fun s _ => ?_)
    (Or.inr ⟨(B.shape g).slices - 1, List.mem_range.mpr (by show _ < (B.shape g).slices; omega),
      ?_⟩)
  · unfold SetupGroup.sliceLen Shape.sliceLength
    rw [start, start, add_mul, one_mul]
    have := div_add_le (s * (B.shape g).blocks) (B.shape g).blocks (B.shape g).slices positive
    generalize ((B.shape g).blocks + (B.shape g).slices - 1) / (B.shape g).slices = L at *
    generalize (s * (B.shape g).blocks + (B.shape g).blocks) / (B.shape g).slices = x at *
    generalize s * (B.shape g).blocks / (B.shape g).slices = y at *
    omega
  · unfold SetupGroup.sliceLen Shape.sliceLength
    rw [start, start, Nat.sub_add_cancel positive, Nat.mul_div_cancel_left _ positive]
    exact last_slice _ _ positive

theorem sliceStart_mono (S : Shape) (s : ℕ) : S.sliceStart s ≤ S.sliceStart (s + 1) :=
  Nat.div_le_div_right (Nat.mul_le_mul_right _ (Nat.le_succ s))

theorem min_pow (a b : ℕ) : min (2 ^ a) (2 ^ b) = 2 ^ min a b := by
  rcases le_total a b with h | h
  · rw [min_eq_left h, min_eq_left (Nat.pow_le_pow_right (by norm_num) h)]
  · rw [min_eq_right h, min_eq_right (Nat.pow_le_pow_right (by norm_num) h)]

/-- The base ring of the plan is `2^e` for an `e` at most the `log_2` of each degree. -/
theorem ofBatch_base (κ logBasis : ℕ)
    (innerPow : ∀ g, 2 ^ Nat.log 2 (B.shape g).innerDegree = (B.shape g).innerDegree)
    (outerPow : ∀ g, 2 ^ Nat.log 2 (B.shape g).outerDegree = (B.shape g).outerDegree)
    (openPow : 2 ^ Nat.log 2 B.openDegree = B.openDegree) :
    ∃ e, (SetupPlan.ofBatch B (groupOrder n) κ logBasis).base = 2 ^ e ∧
      e ≤ Nat.log 2 B.openDegree ∧
        ∀ g, e ≤ Nat.log 2 (B.shape g).innerDegree ∧ e ≤ Nat.log 2 (B.shape g).outerDegree := by
  have powers : ∀ G ∈ List.ofFn (fun k => groupOf B (groupOrder n k)),
      ∃ k, min G.innerDegree G.outerDegree = 2 ^ k := fun G member => by
    obtain ⟨i, rfl⟩ := List.mem_ofFn.mp member
    refine ⟨min (Nat.log 2 (B.shape (groupOrder n i)).innerDegree)
      (Nat.log 2 (B.shape (groupOrder n i)).outerDegree), ?_⟩
    show min (B.shape (groupOrder n i)).innerDegree (B.shape (groupOrder n i)).outerDegree = _
    rw [← min_pow, innerPow, outerPow]
  obtain ⟨e, le, value, each⟩ := foldl_min_pow (fun G => min G.innerDegree G.outerDegree) _
    (Nat.log 2 B.openDegree) powers
  refine ⟨e, ?_, le, fun g => ?_⟩
  · unfold SetupPlan.base
    rw [ofBatch_groups]
    show List.foldl _ B.openDegree _ = _
    rw [← openPow]
    exact value
  · have bound := each (groupOf B g)
      (List.mem_ofFn.mpr ⟨(groupOrder n).symm g, by simp only [Equiv.apply_symm_apply]⟩)
    exact ⟨log_ge_of (innerPow g).symm (bound.trans (min_le_left _ _)),
      log_ge_of (outerPow g).symm (bound.trans (min_le_right _ _))⟩

theorem ofBatch_openRow (κ logBasis : ℕ) :
    (SetupPlan.ofBatch B (groupOrder n) κ logBasis).openRow = ∑ g, B.groupCount g := by
  unfold SetupPlan.openRow
  rw [ofBatch_groups, List.map_ofFn, Fin.sum_ofFn, ← Equiv.sum_comp (groupOrder n) B.groupCount]
  exact Finset.sum_congr rfl fun k _ => rows_eq B _

theorem ofBatch_dColsAll (κ logBasis : ℕ) :
    (SetupPlan.ofBatch B (groupOrder n) κ logBasis).dColsAll =
      ∑ g, (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits) := by
  unfold SetupPlan.dColsAll
  rw [ofBatch_groups, List.map_ofFn, Fin.sum_ofFn, ← Equiv.sum_comp (groupOrder n)
    (fun g => (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits))]
  exact Finset.sum_congr rfl fun k _ => Nat.mul_assoc _ _ _

theorem mainStart_order (k : Fin n) :
    B.mainStart (groupOrder n k) = rowsBefore (fun j => B.mainCells (groupOrder n j)) k := by
  unfold Batch.mainStart codeOffset
  rw [Equiv.symm_apply_apply, offset_rowsBefore]

theorem responseCode_val (S : Shape) (fold : Descriptor)
    (col : (Fin S.blockLength × Fin S.sourceDigits) × Fin fold.responseDigits) :
    (S.responseCode fold col : ℕ) = S.innerSetupCode col.1 * fold.responseDigits + col.2 := by
  simp only [Shape.responseCode, Shape.innerSetupCode, Equiv.trans_apply, Equiv.prodCongr_apply,
    Equiv.refl_apply, Prod.map, finProdFinEquiv_apply_val]
  ring

/-- The position in the slice of the column `e` of `B`: its column code divided by the width
`g_B` of a block. -/
theorem outerSetupCode_div (S : Shape) (e : Fin S.sliceLength × S.OuterColumn) :
    (S.outerSetupCode e : ℕ) / (S.innerRank * (S.ratio * S.openingDigits)) = e.1 := by
  have inner :=
    (((Equiv.prodCongr (Equiv.refl _) finProdFinEquiv).trans finProdFinEquiv) e.2).isLt
  have value : (S.outerSetupCode e : ℕ) =
      (((Equiv.prodCongr (Equiv.refl _) finProdFinEquiv).trans finProdFinEquiv) e.2 : ℕ) +
        S.innerRank * (S.ratio * S.openingDigits) * e.1 := rfl
  rw [value, Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt inner, Nat.zero_add]

/-- The column code of the column `c` of `D` of the group `groupOrder n k`: the columns of the
groups before it, then its column in the group. -/
theorem openSetupCode_val (k : Fin n) (c : (B.opening (groupOrder n k)).OpenColumn) :
    (B.openSetupCode ⟨groupOrder n k, c⟩ : ℕ) =
      rowsBefore (fun i => (B.shape (groupOrder n i)).blocks *
        ((B.opening (groupOrder n i)).subcolumns * (B.shape (groupOrder n i)).openingDigits)) k +
        (B.opening (groupOrder n k)).partialCode c := by
  have inner : (B.openSetupCode ⟨groupOrder n k, c⟩ : ℕ) =
      ((place fun g => (B.shape g).blocks * ((B.opening g).subcolumns *
        (B.shape g).openingDigits)).symm ⟨groupOrder n k,
          ((Equiv.prodCongr (Equiv.refl _) finProdFinEquiv).trans finProdFinEquiv) c⟩ : ℕ) :=
    rfl
  rw [inner, place_symm_val]
  congr 1
  obtain ⟨i, y, u⟩ := c
  simp only [OpeningShape.partialCode, Equiv.trans_apply, Equiv.prodCongr_apply,
    Equiv.refl_apply, Equiv.prodAssoc_symm_apply, Prod.map, finProdFinEquiv_apply_val]
  ring

theorem modNat_one {m : ℕ} (j : Fin (1 * m)) : ((finProdFinEquiv.symm j).2 : ℕ) = j := by
  show (j : ℕ) % m = j
  exact Nat.mod_eq_of_lt (by have := j.isLt; omega)

end Plan

section Parts

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E]

/-- The part of `A` of one group. -/
theorem inner_match (g : Fin n) (row col : ℕ) (τ ρ r : List E) (α : E) {κ e lb : ℕ}
    (blocked : B.Blocked κ) (gadget : (B.fold g).responseBase = 2 ^ lb)
    (base : (SetupPlan.ofBatch B (groupOrder n) κ lb).base = 2 ^ e)
    (pow : 2 ^ Nat.log 2 (B.shape g).innerDegree = (B.shape g).innerDegree)
    (low : e ≤ Nat.log 2 (B.shape g).innerDegree)
    (fit : ((B.shape g).innerInstance false 0).size ≤ 2 ^ ρ.length)
    (run : B.mainStart g + B.mainCells g ≤ 2 ^ (r.length + κ)) :
    innerPart B τ ρ r α κ g row = powerScale α (ρ.take e) *
      aIn (SetupPlan.ofBatch B (groupOrder n) κ lb) τ (ρ.drop e) r α
        ⟨groupOf B g, B.mainStart g, row, col⟩ := by
  obtain ⟨k, power⟩ : ∃ k, (B.shape g).innerDegree = 2 ^ k := ⟨_, pow.symm⟩
  have lowk : e ≤ k := by
    rw [power, Nat.log_pow (by norm_num)] at low
    exact low
  have blockLe : κ ≤ k := (Nat.pow_dvd_pow_iff_le_right (by norm_num : 1 < 2)).mp
    (by rw [← power]; exact (blocked.1 g).2.2.2.1)
  have ratioBase : (B.shape g).innerDegree / 2 ^ e = 2 ^ (k - e) := by
    rw [power, Nat.pow_div lowk (by norm_num)]
  unfold innerPart aIn
  rw [base]
  dsimp only [groupOf, SetupPlan.ofBatch, SetupGroup.zCols]
  rw [ratioBase, Nat.log_pow (by norm_num)]
  simp only [Finset.sum_range]
  rw [Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [Fintype.sum_prod_type, ← Equiv.sum_comp (B.shape g).innerSetupCode]
  simp only [responseCode_val, Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun f _ => ?_
  have sizeLe : (B.shape g).innerDegree ≤ 2 ^ ρ.length := by
    refine le_trans ?_ fit
    show _ ≤ (B.shape g).innerRank * ((B.shape g).blockLength * (B.shape g).sourceDigits) *
      (B.shape g).innerDegree
    exact Nat.le_mul_of_pos_left _ (Nat.mul_pos (by have := a.isLt; omega)
      (by have := ((B.shape g).innerSetupCode x).isLt; omega))
  have runLe : (B.shape g).innerDegree ≤ 2 ^ (r.length + κ) := by
    refine le_trans ?_ run
    have cells : (B.shape g).responseCells (B.fold g) ≤ B.mainCells g :=
      Nat.le_add_right_of_le (Nat.le_add_right _ _)
    have one : (B.shape g).innerDegree ≤ (B.shape g).responseCells (B.fold g) := by
      unfold Shape.responseCells
      exact Nat.le_mul_of_pos_left _ (Nat.mul_pos
        (by have := ((B.shape g).innerSetupCode x).isLt; omega) (by have := f.isLt; omega))
    omega
  rw [setupAt_split ρ α _ power lowk (log_le_of power sizeLe),
    runAt_lane r α _ _ power blockLe (log_le_of power runLe),
    div_add_mul _ _ (blocked.1 g).2.2.2.1, gadget]
  ring

/-- The part of `B` of one group. -/
theorem outer_match (g : Fin n) (row col : ℕ) (τ ρ r : List E) (α : E) {κ e lb : ℕ}
    (blocked : B.Blocked κ) (base : (SetupPlan.ofBatch B (groupOrder n) κ lb).base = 2 ^ e)
    (pow : 2 ^ Nat.log 2 (B.shape g).outerDegree = (B.shape g).outerDegree)
    (low : e ≤ Nat.log 2 (B.shape g).outerDegree)
    (fit : (B.shape g).outerInstance.size ≤ 2 ^ ρ.length)
    (run : B.mainStart g + B.mainCells g ≤ 2 ^ (r.length + κ)) :
    outerPart B τ ρ r α κ g row = powerScale α (ρ.take e) *
      bIn (SetupPlan.ofBatch B (groupOrder n) κ lb) τ (ρ.drop e) r α
        ⟨groupOf B g, B.mainStart g, row, col⟩ := by
  obtain ⟨k, power⟩ : ∃ k, (B.shape g).outerDegree = 2 ^ k := ⟨_, pow.symm⟩
  have lowk : e ≤ k := by
    rw [power, Nat.log_pow (by norm_num)] at low
    exact low
  have blockLe : κ ≤ k := (Nat.pow_dvd_pow_iff_le_right (by norm_num : 1 < 2)).mp
    (by rw [← power]; exact (blocked.1 g).2.2.2.2)
  have ratioBase : (B.shape g).outerDegree / 2 ^ e = 2 ^ (k - e) := by
    rw [power, Nat.pow_div lowk (by norm_num)]
  have width : (groupOf B g).blockWidth =
      (B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits) := by
    show (B.shape g).innerRank * ((B.shape g).innerDegree / (B.shape g).outerDegree) *
      (B.shape g).openingDigits = _
    rw [Nat.mul_assoc]
    rfl
  have phys : (groupOf B g).physWidth = (B.shape g).sliceLength *
      ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) := by
    unfold SetupGroup.physWidth
    rw [maxSlice_eq, width]
  have start : B.mainStart g + (groupOf B g).zLen + (groupOf B g).eLen =
      B.mainStart g + (B.opening g).innerStart (B.fold g) := by
    rw [Nat.add_assoc, innerStart_eq]
  unfold outerPart bIn
  rw [base]
  dsimp only
  rw [width, phys, start]
  simp only [SetupGroup.sliceLen, groupSliceStart]
  dsimp only [groupOf, SetupPlan.ofBatch]
  rw [ratioBase, Nat.log_pow (by norm_num), ← Equiv.sum_comp finProdFinEquiv,
    Fintype.sum_prod_type_right]
  simp only [Equiv.symm_apply_apply, finProdFinEquiv_apply_val, Finset.sum_range]
  rw [Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← Equiv.sum_comp (B.shape g).outerSetupCode]
  simp only [Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun s _ => ?_
  have positive : 0 < (B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits) :=
    by have := (((Equiv.prodCongr (Equiv.refl _) finProdFinEquiv).trans finProdFinEquiv) c.2).isLt
       omega
  have iff : ((B.shape g).outerSetupCode c : ℕ) < ((B.shape g).sliceStart (s + 1) -
      (B.shape g).sliceStart s) * ((B.shape g).innerRank * ((B.shape g).ratio *
        (B.shape g).openingDigits)) ↔
      (B.shape g).sliceStart s + c.1 < (B.shape g).sliceStart (s + 1) := by
    rw [← Nat.div_lt_iff_lt_mul positive, outerSetupCode_div]
    omega
  rw [show row + 1 + (B.shape g).innerRank + ((j : ℕ) + (B.shape g).outerRank * s) =
    row + 1 + (B.shape g).innerRank + s * (B.shape g).outerRank + j by ring]
  by_cases inside : (B.shape g).sliceStart s + c.1 < (B.shape g).sliceStart (s + 1)
  · have mono := sliceStart_mono (B.shape g) s
    have top := (B.shape g).sliceStart_le (Nat.succ_le_of_lt s.isLt)
    have lt := iff.mpr inside
    have whole : ((B.shape g).sliceStart (s + 1) - (B.shape g).sliceStart s) *
        ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) +
        (B.shape g).sliceStart s *
          ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) =
        (B.shape g).sliceStart (s + 1) *
          ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) := by
      rw [← Nat.add_mul, Nat.sub_add_cancel mono]
    have below : (B.shape g).sliceStart (s + 1) *
        ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) ≤
        (B.shape g).blocks *
          ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) :=
      Nat.mul_le_mul_right _ top
    have cellLt : (B.shape g).sliceStart s *
        ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) +
        ((B.shape g).outerSetupCode c : ℕ) < (B.shape g).blocks *
          ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) := by
      omega
    simp only [inside, lt, cellLt, ↓reduceIte]
    have sizeLe : (B.shape g).outerDegree ≤ 2 ^ ρ.length := by
      refine le_trans ?_ fit
      show _ ≤ (B.shape g).outerRank * ((B.shape g).sliceLength * ((B.shape g).innerRank *
        ((B.shape g).ratio * (B.shape g).openingDigits))) * (B.shape g).outerDegree
      exact Nat.le_mul_of_pos_left _ (Nat.mul_pos (by have := j.isLt; omega)
        (by have := ((B.shape g).outerSetupCode c).isLt; omega))
    have runLe : (B.shape g).outerDegree ≤ 2 ^ (r.length + κ) := by
      refine le_trans ?_ run
      have cells : (B.shape g).innerCells ≤ B.mainCells g := Nat.le_add_left _ _
      have one : (B.shape g).outerDegree ≤ (B.shape g).innerCells := by
        show _ ≤ (B.shape g).blocks * (B.shape g).innerRank * (B.shape g).ratio *
          (B.shape g).openingDigits * (B.shape g).outerDegree
        refine Nat.le_mul_of_pos_left _ ?_
        rw [show (B.shape g).blocks * (B.shape g).innerRank * (B.shape g).ratio *
          (B.shape g).openingDigits = (B.shape g).blocks *
            ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) by ring]
        omega
      omega
    rw [setupAt_split ρ α _ power lowk (log_le_of power sizeLe),
      runAt_lane r α _ _ power blockLe (log_le_of power runLe), ← Nat.add_assoc (B.mainStart g),
      div_add_mul _ _ (blocked.1 g).2.2.2.2]
    ring
  · have outside : ¬((B.shape g).outerSetupCode c : ℕ) < ((B.shape g).sliceStart (s + 1) -
        (B.shape g).sliceStart s) * ((B.shape g).innerRank * ((B.shape g).ratio *
          (B.shape g).openingDigits)) := fun lt => inside (iff.mp lt)
    simp only [inside, outside, ↓reduceIte]
    ring

variable (B) in
/-- The part of `D` of the group `groupOrder n k`: the columns of `D` of the group. -/
noncomputable def openGroup (τ ρ r : List E) (α : E) (κ row : ℕ) (k : Fin n) : E :=
  ∑ j : Fin (1 * B.openRank), eqAt τ (row + j) *
    ∑ c : (B.opening (groupOrder n k)).OpenColumn,
      setupAt ρ α B.openDegree ((finProdFinEquiv.symm j).2 *
          (∑ g, (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits)) +
        (rowsBefore (fun i => (B.shape (groupOrder n i)).blocks *
          ((B.opening (groupOrder n i)).subcolumns * (B.shape (groupOrder n i)).openingDigits)) k +
          (B.opening (groupOrder n k)).partialCode c)) *
        runAt r α κ B.openDegree (B.mainStart (groupOrder n k) +
          ((B.shape (groupOrder n k)).responseCells (B.fold (groupOrder n k)) +
            (B.opening (groupOrder n k)).partialCode c * B.openDegree))

/-- The part of `D` of the specification is the sum of the parts of the groups. -/
theorem open_groups (τ ρ r : List E) (α : E) (κ row : ℕ) :
    openPart B τ ρ r α κ row = ∑ k, openGroup B τ ρ r α κ row k := by
  unfold openPart openGroup
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← Finset.mul_sum]
  congr 1
  rw [Fintype.sum_sigma, ← Equiv.sum_comp (groupOrder n)]
  refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun c _ => ?_
  rw [openSetupCode_val]

/-- The part of `D` of one group. -/
theorem open_match (k : Fin n) (row : ℕ) (τ ρ r : List E) (α : E) {κ e lb : ℕ}
    (blocked : B.Blocked κ) (base : (SetupPlan.ofBatch B (groupOrder n) κ lb).base = 2 ^ e)
    (pow : 2 ^ Nat.log 2 B.openDegree = B.openDegree) (low : e ≤ Nat.log 2 B.openDegree)
    (fit : B.openInstance.size ≤ 2 ^ ρ.length)
    (run : B.mainStart (groupOrder n k) + B.mainCells (groupOrder n k) ≤ 2 ^ (r.length + κ)) :
    openGroup B τ ρ r α κ (∑ g, B.groupCount g) k = powerScale α (ρ.take e) *
      dIn (SetupPlan.ofBatch B (groupOrder n) κ lb) τ (ρ.drop e) r α
        ⟨groupOf B (groupOrder n k), B.mainStart (groupOrder n k), row,
          rowsBefore (fun i => (B.shape (groupOrder n i)).blocks *
            ((B.opening (groupOrder n i)).subcolumns *
              (B.shape (groupOrder n i)).openingDigits)) k⟩ := by
  obtain ⟨d, power⟩ : ∃ d, B.openDegree = 2 ^ d := ⟨_, pow.symm⟩
  have lowk : e ≤ d := by
    rw [power, Nat.log_pow (by norm_num)] at low
    exact low
  have blockLe : κ ≤ d := (Nat.pow_dvd_pow_iff_le_right (by norm_num : 1 < 2)).mp
    (by rw [← power]; exact blocked.2)
  have ratioBase : B.openDegree / 2 ^ e = 2 ^ (d - e) := by
    rw [power, Nat.pow_div lowk (by norm_num)]
  have total : rowsBefore (fun i => (B.shape (groupOrder n i)).blocks *
      ((B.opening (groupOrder n i)).subcolumns * (B.shape (groupOrder n i)).openingDigits)) k +
      (B.shape (groupOrder n k)).blocks * ((B.opening (groupOrder n k)).subcolumns *
        (B.shape (groupOrder n k)).openingDigits) ≤
      ∑ g, (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits) :=
    (rowsBefore_add_le _ k).trans_eq (Equiv.sum_comp (groupOrder n)
      fun g => (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits))
  unfold openGroup dIn
  rw [base]
  dsimp only
  rw [ofBatch_openRow, ofBatch_dColsAll]
  dsimp only [SetupPlan.ofBatch]
  rw [dCols_eq, zLen_eq, ratioBase, Nat.log_pow (by norm_num)]
  simp only [Finset.sum_range]
  rw [Finset.mul_sum, Finset.mul_sum]
  refine Fintype.sum_equiv (finCongr (Nat.one_mul B.openRank)) _ _ fun j => ?_
  rw [modNat_one, ← Equiv.sum_comp (B.opening (groupOrder n k)).partialCode]
  simp only [finCongr_apply, Fin.val_cast, Finset.mul_sum, ← Nat.add_assoc]
  refine Finset.sum_congr rfl fun c _ => ?_
  have code := ((B.opening (groupOrder n k)).partialCode c).isLt
  have sizeLe : B.openDegree ≤ 2 ^ ρ.length := by
    refine le_trans ?_ fit
    show _ ≤ B.openRank * (∑ g, (B.shape g).blocks * ((B.opening g).subcolumns *
      (B.shape g).openingDigits)) * B.openDegree
    refine Nat.le_mul_of_pos_left _ (Nat.mul_pos (by have := j.isLt; omega) ?_)
    rw [← Nat.mul_assoc] at total
    omega
  have runLe : B.openDegree ≤ 2 ^ (r.length + κ) := by
    refine le_trans ?_ run
    have cells : (B.opening (groupOrder n k)).partialCells ≤ B.mainCells (groupOrder n k) :=
      Nat.le_add_right_of_le (Nat.le_add_left _ _)
    have one : B.openDegree ≤ (B.opening (groupOrder n k)).partialCells :=
      Nat.le_mul_of_pos_left _ (by omega)
    omega
  rw [setupAt_split ρ α _ power lowk (log_le_of power sizeLe),
    runAt_lane r α _ _ power blockLe (log_le_of power runLe), div_add_mul _ _ blocked.2]
  ring

end Parts

section Weight

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Fintype E] [Algebra Zq E]
  [Degree2 E]

/-- The weight of Stage 3 of the replay at `(r_2, ρ)` for the plan of the code is the multilinear
extension at `ρ` of the weight of Stage 3 of `Batch.codeScan` at `r_2`: the hypothesis `weighs` of
`stage3Run_runs` and `setupWeighs` of `batchOffloadRun_runs`. `gadget`: the response base of each
group is `2^logBasis`; `innerPow`, `outerPow`, `openPow`: the ring degrees are powers of two;
`innerFit`, `outerFit`, `openFit`: the views of `A`, `B` and `D` fit in the `2^ν` cells of the
setup prefix. -/
theorem stage3Weight_codeScan (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (st : B.Statement E) (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ)
    (base digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (ν : ℕ) (flat : Table) (κ : ℕ)
    (blocked : B.Blocked κ) (low : κ ≤ μ) (logBasis : ℕ)
    (gadget : ∀ g, (B.fold g).responseBase = 2 ^ logBasis)
    (innerPow : ∀ g, 2 ^ Nat.log 2 (B.shape g).innerDegree = (B.shape g).innerDegree)
    (outerPow : ∀ g, 2 ^ Nat.log 2 (B.shape g).outerDegree = (B.shape g).outerDegree)
    (openPow : 2 ^ Nat.log 2 B.openDegree = B.openDegree)
    (innerFit : ∀ g, ((B.shape g).innerInstance false 0).size ≤ 2 ^ ν)
    (outerFit : ∀ g, (B.shape g).outerInstance.size ≤ 2 ^ ν)
    (openFit : B.openInstance.size ≤ 2 ^ ν) (α : Ext) (τ r2 ρ : List Ext)
    (τLength : τ.length = batch) (r2Length : r2.length = μ) (ρLength : ρ.length = ν) :
    ((stage3Weight (SetupPlan.ofBatch B (groupOrder n) κ logBasis) α τ r2 ρ).into : E) =
      mle (ρ.map Ext.into) ((Batch.codeScan β st challenge μ fits base digits room batch rowsFit
        ν flat κ blocked low).weight α.into (τ.map Ext.into) (r2.map Ext.into)) := by
  obtain ⟨e, least, openLe, groupLe⟩ := ofBatch_base B κ logBasis innerPow outerPow openPow
  have setupLength : (ρ.map (Ext.into (E := E))).length = ν := by rw [List.length_map, ρLength]
  have run : ∀ g, B.mainStart g + B.mainCells g ≤
      2 ^ (((r2.map (Ext.into (E := E))).drop κ).length + κ) := fun g => by
    rw [List.length_drop, List.length_map, r2Length, Nat.sub_add_cancel low]
    exact (codeOffset_add_le B.mainCells g).trans (B.chainStart_le_cells.trans fits)
  rw [codeScan_mle β st challenge μ fits base digits room batch rowsFit ν flat κ blocked low
      α.into (τ.map Ext.into) (r2.map Ext.into) (ρ.map Ext.into)
      (by rw [List.length_map, τLength]) (by rw [List.length_map, r2Length]) setupLength
      innerFit outerFit openFit,
    stage3Weight_into, least, Nat.log_pow (by norm_num),
    show (SetupPlan.ofBatch B (groupOrder n) κ logBasis).block = 2 ^ κ from rfl,
    Nat.log_pow (by norm_num), ofBatch_groups, placedFrom_ofFn, open_groups,
    ← Finset.sum_add_distrib, Finset.mul_sum, List.map_take, List.map_drop, List.map_drop]
  refine Finset.sum_congr rfl fun k _ => ?_
  have zEq : 0 + rowsBefore (fun j => (groupOf B (groupOrder n j)).unitLen) k =
      B.mainStart (groupOrder n k) := by
    rw [Nat.zero_add, mainStart_order]
    simp only [unitLen_eq]
  have rowEq : 0 + rowsBefore (fun j => (groupOf B (groupOrder n j)).rows) k =
      rowsBefore (fun j => B.groupCount (groupOrder n j)) k := by
    rw [Nat.zero_add]
    simp only [rows_eq]
  have colEq : 0 + rowsBefore (fun j => (groupOf B (groupOrder n j)).dCols
      (SetupPlan.ofBatch B (groupOrder n) κ logBasis).openDegree) k =
      rowsBefore (fun i => (B.shape (groupOrder n i)).blocks *
        ((B.opening (groupOrder n i)).subcolumns * (B.shape (groupOrder n i)).openingDigits)) k := by
    rw [Nat.zero_add]
    congr 1
    funext j
    exact Nat.mul_assoc _ _ _
  rw [zEq, rowEq, colEq, groupIn, mul_add, mul_add,
    inner_match (groupOrder n k) _ _ _ _ _ _ blocked (gadget _) least (innerPow _)
      (groupLe _).1 (by rw [setupLength]; exact innerFit _) (run _),
    outer_match (groupOrder n k) _ _ _ _ _ _ blocked least (outerPow _) (groupLe _).2
      (by rw [setupLength]; exact outerFit _) (run _),
    open_match k _ _ _ _ _ blocked least openPow openLe (by rw [setupLength]; exact openFit)
      (run _)]

/-- `stage3Run_runs` for the weight of Stage 3 of the plan of the code: `setupCheck` for the
weight of `Batch.codeScan` at `r_2`, on the path of the messages and the challenges of the code,
ends in the result of `stage3Run` with the weight `stage3Weight` of the plan. -/
theorem stage3Run_codeScan [DecidableEq E] (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (st : B.Statement E) (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ)
    (base digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (ν : ℕ) (flat : Table) (κ : ℕ)
    (blocked : B.Blocked κ) (low : κ ≤ μ) (logBasis : ℕ)
    (gadget : ∀ g, (B.fold g).responseBase = 2 ^ logBasis)
    (innerPow : ∀ g, 2 ^ Nat.log 2 (B.shape g).innerDegree = (B.shape g).innerDegree)
    (outerPow : ∀ g, 2 ^ Nat.log 2 (B.shape g).outerDegree = (B.shape g).outerDegree)
    (openPow : 2 ^ Nat.log 2 B.openDegree = B.openDegree)
    (innerFit : ∀ g, ((B.shape g).innerInstance false 0).size ≤ 2 ^ ν)
    (outerFit : ∀ g, (B.shape g).outerInstance.size ≤ 2 ^ ν)
    (openFit : B.openInstance.size ≤ 2 ^ ν) (α : Ext) (τ r2 : List Ext)
    (τLength : τ.length = batch) (r2Length : r2.length = μ) (claim : Ext)
    (rounds : List (List Ext × Ext)) (value : Ext) (count : rounds.length = ν)
    (lengths : ∀ x ∈ rounds, x.1.length = 2) :
    (setupCheck ν flat ((Batch.codeScan β st challenge μ fits base digits room batch rowsFit ν
        flat κ blocked low).weight α.into (τ.map Ext.into) (r2.map Ext.into)) claim.into).Runs
      (stage3Path E claim rounds value)
      ((stage3Run claim rounds value
        (stage3Weight (SetupPlan.ofBatch B (groupOrder n) κ logBasis) α τ r2)).map
          fun out => (out.1.map Ext.into, out.2.into)) :=
  stage3Run_runs ν flat _ claim rounds value _ count lengths
    (stage3Weight_codeScan β st challenge μ fits base digits room batch rowsFit ν flat κ blocked
      low logBasis gadget innerPow outerPow openPow innerFit outerFit openFit α τ r2 _ τLength
      r2Length (by rw [plainRun_point_length, count]))

end Weight

end NightstreamFPrime.Spec.Compression.Akita.Replay
