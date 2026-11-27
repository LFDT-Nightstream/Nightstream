import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingGroup

/-! The quotient digits of a packing fold (ePrint 2026/1983, (111) and (348)): the cells of the
replay (`carrierQuotient`, `quotientLift` and `rowStarts` of `Replay/WeightPacking.lean`) against
`FlatRelation.quotientCell`, and their values at `r_2`.

- `flatJoin_entry_plane` and `flatJoin_entry_tail`: `quotientEntry` of the join
  `(Join.all β).append (Join.plain E m)` of `flatRelation`. Digit `a` of coefficient `j` of plane
  `t` is the entry `a·(P·d_car) + t·d_car + j`, and of the tail row `i` the entry
  `δ·P·d_car + δ·Σ_{i'<i} d_{i'} + a·d_i + j`.
- `Gaps.slot_main`, `Gaps.slot_middle` and `Gaps.slot_last` give the cell of an entry in the three
  quotient segments.
- `unitLift_value`: a quotient term of the replay is
  `c·Σ_{j<d} Σ_a α^j·b^a·ẽq(r_2, start_a + j)`. `rowStarts_sum` splits the rows of `rowStarts`
  into blocks of one degree. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Entries

variable {p m : ℕ}

/-- The slots of the join `(Join.all β).append (Join.plain E m)` of `flatRelation`: the planes in
slot 0 and the tail row `i` in slot `1 + i`. -/
def flatSlot (p m : ℕ) : Fin (p + m) → Fin (1 + m) :=
  Fin.addCases (fun _ => Fin.castAdd m 0) (Fin.natAdd 1)

theorem flatSlot_left (t : Fin p) : flatSlot p m (Fin.castAdd m t) = Fin.castAdd m 0 :=
  Fin.addCases_left _

theorem flatSlot_right (i : Fin m) : flatSlot p m (Fin.natAdd p i) = Fin.natAdd 1 i :=
  Fin.addCases_right _

theorem natAdd_ne_castAdd (i : Fin m) : Fin.natAdd 1 i ≠ Fin.castAdd m (0 : Fin 1) := fun same => by
  have values := congrArg Fin.val same
  simp only [Fin.val_natAdd, Fin.val_castAdd, Fin.val_zero] at values
  omega

theorem sum_range_ite_lt (c : ℕ) (t : ℕ) : ∀ n,
    ∑ i ∈ Finset.range n, (if i < t then c else 0) = min t n * c
  | 0 => by simp
  | n + 1 => by
      rw [Finset.sum_range_succ, sum_range_ite_lt c t n]
      by_cases small : n < t
      · rw [ite_eq_left small, min_eq_right small.le, min_eq_right small, Nat.succ_mul]
      · rw [ite_eq_right small, min_eq_left (by omega), min_eq_left (by omega), add_zero]

variable (degree : Fin (p + m) → ℕ) (dc digits : ℕ)

theorem flatSlot_width_left (planes : ∀ t : Fin p, degree (Fin.castAdd m t) = dc) :
    slotWidth (flatSlot p m) degree (Fin.castAdd m 0) = p * dc := by
  unfold slotWidth slotDegree
  rw [Fin.sum_univ_add, Finset.sum_eq_zero (s := Finset.univ) fun i _ =>
    ite_eq_right fun same => natAdd_ne_castAdd i ((flatSlot_right i).symm.trans same), add_zero]
  simp only [flatSlot_left, planes, ite_true, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    smul_eq_mul]

theorem flatSlot_width_right (i : Fin m) :
    slotWidth (flatSlot p m) degree (Fin.natAdd 1 i) = degree (Fin.natAdd p i) := by
  unfold slotWidth slotDegree
  rw [Fin.sum_univ_add, Finset.sum_eq_zero (s := Finset.univ) fun t _ =>
    ite_eq_right fun same => natAdd_ne_castAdd i ((flatSlot_left t).symm.trans same).symm, zero_add,
    Finset.sum_eq_single i (fun i' _ other => ite_eq_right fun same =>
      other ((Fin.natAdd_inj _).mp ((flatSlot_right i').symm.trans same)))
      (fun absent => absurd (Finset.mem_univ i) absent), flatSlot_right, ite_eq_left rfl]

theorem flatSlot_entry_plane (planes : ∀ t : Fin p, degree (Fin.castAdd m t) = dc) (t : Fin p)
    (j a : ℕ) :
    quotientEntry (flatSlot p m) degree digits (Fin.castAdd m t) j a =
      a * (p * dc) + (t * dc + j) := by
  unfold quotientEntry
  rw [flatSlot_left, flatSlot_width_left degree dc planes,
    show ((Fin.castAdd m (0 : Fin 1) : Fin (1 + m)) : ℕ) = 0 from rfl, offset_zero, zero_add]
  congr 2
  unfold offset slotDegree
  have after : ∑ i : Fin m, (if ((Fin.natAdd p i : Fin (p + m)) : ℕ) < (Fin.castAdd m t : Fin (p + m))
      then (if flatSlot p m (Fin.natAdd p i) = Fin.castAdd m 0 then degree (Fin.natAdd p i) else 0)
      else 0) = 0 :=
    Finset.sum_eq_zero fun i _ => ite_eq_right fun before => by
      simp only [Fin.val_natAdd, Fin.val_castAdd] at before
      omega
  rw [Fin.sum_univ_add, after, add_zero]
  simp only [flatSlot_left, planes, ite_true, Fin.val_castAdd]
  rw [Fin.sum_univ_eq_sum_range (fun i => if i < (t : ℕ) then dc else 0), sum_range_ite_lt,
    min_eq_left t.isLt.le]

theorem flatSlot_entry_tail (planes : ∀ t : Fin p, degree (Fin.castAdd m t) = dc) (i : Fin m)
    (j a : ℕ) :
    quotientEntry (flatSlot p m) degree digits (Fin.natAdd p i) j a =
      digits * (p * dc) + digits * offset (fun i' => degree (Fin.natAdd p i')) i +
        (a * degree (Fin.natAdd p i) + j) := by
  unfold quotientEntry
  rw [flatSlot_right, flatSlot_width_right degree]
  have zero : offset (slotDegree (flatSlot p m) degree (Fin.natAdd 1 i)) (Fin.natAdd p i) = 0 := by
    unfold offset slotDegree
    refine Finset.sum_eq_zero fun r _ => ?_
    refine Fin.addCases (fun t => ?_) (fun i' => ?_) r
    · have other : flatSlot p m (Fin.castAdd m t) ≠ Fin.natAdd 1 i := fun same =>
        natAdd_ne_castAdd i ((flatSlot_left t).symm.trans same).symm
      rw [ite_eq_right other, ite_self]
    · by_cases same : i' = i
      · subst same
        exact ite_eq_right (lt_irrefl _)
      · rw [ite_eq_right fun equal => same ((Fin.natAdd_inj _).mp
          ((flatSlot_right i').symm.trans equal)), ite_self]
  rw [zero, zero_add]
  congr 1
  unfold offset slotCells
  rw [Fin.sum_univ_add, Fin.sum_univ_one,
    show ((Fin.castAdd m (0 : Fin 1) : Fin (1 + m)) : ℕ) = 0 from rfl,
    ite_eq_left (by simp only [Fin.val_natAdd]; omega), flatSlot_width_left degree dc planes,
    Finset.mul_sum]
  congr 1
  refine Finset.sum_congr rfl fun i' _ => ?_
  simp only [Fin.val_natAdd, flatSlot_width_right degree, mul_ite, mul_zero]
  by_cases before : (i' : ℕ) < i
  · rw [ite_eq_left (by omega), ite_eq_left before]
  · rw [ite_eq_right (by omega), ite_eq_right before]

variable {E : Type} [CommRing E] (β : Fin p → E)

/-- Digit `a` of coefficient `j` of plane `t` of the carrier row is the entry
`a·(P·d_car) + t·d_car + j` of the quotient digits. -/
theorem flatJoin_entry_plane (planes : ∀ t : Fin p, degree (Fin.castAdd m t) = dc) (t : Fin p)
    (j a : ℕ) :
    quotientEntry ((Join.all β).append (Join.plain E m)).slot degree digits (Fin.castAdd m t) j a =
      a * (p * dc) + (t * dc + j) :=
  flatSlot_entry_plane degree dc digits planes t j a

/-- Digit `a` of coefficient `j` of the tail row `i` is the entry
`δ·P·d_car + δ·Σ_{i'<i} d_{i'} + a·d_i + j` of the quotient digits. -/
theorem flatJoin_entry_tail (planes : ∀ t : Fin p, degree (Fin.castAdd m t) = dc) (i : Fin m)
    (j a : ℕ) :
    quotientEntry ((Join.all β).append (Join.plain E m)).slot degree digits (Fin.natAdd p i) j a =
      digits * (p * dc) + digits * offset (fun i' => degree (Fin.natAdd p i')) i +
        (a * degree (Fin.natAdd p i) + j) :=
  flatSlot_entry_tail degree dc digits planes i j a

end Entries

section Slots

variable (G : Gaps) {y : ℕ}

theorem Gaps.slot_main (small : y < G.mainCells) : G.slot y = G.main + y :=
  ite_eq_left small

theorem Gaps.slot_middle (low : G.mainCells ≤ y) (small : y < G.mainCells + G.middleCells) :
    G.slot y = G.middle + y := by
  unfold Gaps.slot
  rw [ite_eq_right (by omega), ite_eq_left small]

theorem Gaps.slot_last (low : G.mainCells + G.middleCells ≤ y) : G.slot y = G.last + y := by
  unfold Gaps.slot
  rw [ite_eq_right (by omega), ite_eq_right (by omega)]

end Slots

section Scan

variable {E : Type} [CommRing E]

/-- The quotient digits of a ring row of degree `d` at `r_2`, with digit `a` of coefficient `j` at
the cell `cell a j`: `Σ_{j<d} Σ_a α^j·b^a·ẽq(r_2, cell a j)`. -/
noncomputable def quotientScan (α : E) (r : List E) (b digits d : ℕ) (cell : ℕ → ℕ → ℕ) : E :=
  ∑ j ∈ Finset.range d, ∑ a : Fin digits, α ^ j * (b : E) ^ (a : ℕ) * eqAt r (cell a j)

theorem quotientScan_congr (α : E) (r : List E) (b digits d : ℕ) {cell cell' : ℕ → ℕ → ℕ}
    (same : ∀ a < digits, ∀ j < d, cell a j = cell' a j) :
    quotientScan α r b digits d cell = quotientScan α r b digits d cell' :=
  Finset.sum_congr rfl fun j member => Finset.sum_congr rfl fun a _ => by
    rw [same a a.isLt j (Finset.mem_range.mp member)]

/-- `δ·(p + x·d) + (a·d + j) < δ·(p + n·d)` for `x < n`, `a < δ` and `j < d`. -/
theorem entry_lt {digits d prior a j x n : ℕ} (digit : a < digits) (coefficient : j < d)
    (row : x < n) : digits * (prior + x * d) + (a * d + j) < digits * (prior + n * d) := by
  have inner : a * d + j < digits * d := by
    have step : (a + 1) * d ≤ digits * d := Nat.mul_le_mul_right d digit
    rw [Nat.succ_mul] at step
    omega
  have outer : digits * (prior + x * d) + digits * d ≤ digits * (prior + n * d) := by
    rw [← Nat.mul_add]
    refine Nat.mul_le_mul_left digits ?_
    have step : (x + 1) * d ≤ n * d := Nat.mul_le_mul_right d row
    rw [Nat.succ_mul] at step
    omega
  omega

variable (α : E) (r : List E) (b digits d : ℕ) (G : Gaps) {base prior x n start : ℕ}

/-- The quotient digits of a row of a family in the segment `Q̂_main` from the cell `start`. -/
theorem quotientScan_mainRow (fits : base + digits * (prior + n * d) ≤ G.mainCells) (row : x < n)
    (starts : G.main + (base + digits * (prior + x * d)) = start) :
    quotientScan α r b digits d (fun a j => G.slot (base + digits * (prior + x * d) + (a * d + j))) =
      quotientScan α r b digits d fun a j => start + a * d + j :=
  quotientScan_congr α r b digits d fun a digit j coefficient => by
    have := entry_lt (prior := prior) digit coefficient row
    rw [Gaps.slot_main G (by omega)]
    omega

/-- The quotient digits of a row of a family in the segment `Q̂_1` from the cell `start`. -/
theorem quotientScan_middle (low : G.mainCells ≤ base + digits * prior)
    (fits : base + digits * (prior + n * d) ≤ G.mainCells + G.middleCells) (row : x < n)
    (starts : G.middle + (base + digits * (prior + x * d)) = start) :
    quotientScan α r b digits d (fun a j => G.slot (base + digits * (prior + x * d) + (a * d + j))) =
      quotientScan α r b digits d fun a j => start + a * d + j :=
  quotientScan_congr α r b digits d fun a digit j coefficient => by
    have := entry_lt (prior := prior) digit coefficient row
    have more : digits * prior ≤ digits * (prior + x * d) := Nat.mul_le_mul_left _ (by omega)
    rw [Gaps.slot_middle G (by omega) (by omega)]
    omega

/-- The quotient digits of a row of a family in the segment `Q̂_2` from the cell `start`. -/
theorem quotientScan_last (low : G.mainCells + G.middleCells ≤ base + digits * prior)
    (starts : G.last + (base + digits * (prior + x * d)) = start) :
    quotientScan α r b digits d (fun a j => G.slot (base + digits * (prior + x * d) + (a * d + j))) =
      quotientScan α r b digits d fun a j => start + a * d + j :=
  quotientScan_congr α r b digits d fun a _ j _ => by
    have more : digits * prior ≤ digits * (prior + x * d) := Nat.mul_le_mul_left _ (by omega)
    rw [Gaps.slot_last G (by omega)]
    omega

/-- The quotient digits of a row of a family in the segment `Q̂_main` from the cell `start`, for
the first family of the tail. -/
theorem quotientScan_mainRow₀ (fits : base + digits * (n * d) ≤ G.mainCells) (row : x < n)
    (starts : G.main + (base + digits * (x * d)) = start) :
    quotientScan α r b digits d (fun a j => G.slot (base + digits * (x * d) + (a * d + j))) =
      quotientScan α r b digits d fun a j => start + a * d + j := by
  have same := quotientScan_mainRow α r b digits d G (base := base) (prior := 0) (x := x) (n := n)
    (start := start) (by rw [zero_add]; exact fits) row (by rw [zero_add]; exact starts)
  simp only [zero_add] at same
  exact same

/-- The quotient digits of plane `t` of the carrier row in the segment `Q̂_main`. -/
theorem quotientScan_plane {planes t : ℕ} (fits : digits * (planes * d) ≤ G.mainCells)
    (plane : t < planes) (starts : G.main = start) :
    quotientScan α r b digits d (fun a j => G.slot (a * (planes * d) + (t * d + j))) =
      quotientScan α r b digits d fun a j => start + a * (planes * d) + t * d + j :=
  quotientScan_congr α r b digits d fun a digit j coefficient => by
    have inner : t * d + j < planes * d := by
      have step : (t + 1) * d ≤ planes * d := Nat.mul_le_mul_right d plane
      rw [Nat.succ_mul] at step
      omega
    have outer : (a + 1) * (planes * d) ≤ digits * (planes * d) := Nat.mul_le_mul_right _ digit
    rw [Nat.succ_mul] at outer
    rw [Gaps.slot_main G (by omega)]
    omega

/-- `quotientScan_mainRow` for the row `x` of a family of `n` rows from the cell `s`. -/
theorem quotientScan_row_main {n s : ℕ} (x : Fin n) (fits : base + digits * (prior + n * d) ≤ G.mainCells)
    (starts : G.main + (base + digits * prior) = s) :
    quotientScan α r b digits d (fun a j => G.slot (base + digits * (prior + x * d) + (a * d + j))) =
      quotientScan α r b digits d fun a j => s + x * (digits * d) + a * d + j := by
  rw [quotientScan_mainRow α r b digits d G fits x.isLt (start := s + x * (digits * d)) (by
    rw [← starts, Nat.mul_add, Nat.mul_left_comm]; omega)]

/-- `quotientScan_mainRow₀` for the row `x` of the first family of the tail from the cell `s`. -/
theorem quotientScan_row_main₀ {n s : ℕ} (x : Fin n) (fits : base + digits * (n * d) ≤ G.mainCells)
    (starts : G.main + base = s) :
    quotientScan α r b digits d (fun a j => G.slot (base + digits * (x * d) + (a * d + j))) =
      quotientScan α r b digits d fun a j => s + x * (digits * d) + a * d + j := by
  rw [quotientScan_mainRow₀ α r b digits d G fits x.isLt (start := s + x * (digits * d)) (by
    rw [← starts, Nat.mul_left_comm]; omega)]

/-- `quotientScan_middle` for the row `x` of a family of `n` rows from the cell `s`. -/
theorem quotientScan_row_middle {n s : ℕ} (x : Fin n) (low : G.mainCells ≤ base + digits * prior)
    (fits : base + digits * (prior + n * d) ≤ G.mainCells + G.middleCells)
    (starts : G.middle + (base + digits * prior) = s) :
    quotientScan α r b digits d (fun a j => G.slot (base + digits * (prior + x * d) + (a * d + j))) =
      quotientScan α r b digits d fun a j => s + x * (digits * d) + a * d + j := by
  rw [quotientScan_middle α r b digits d G low fits x.isLt (start := s + x * (digits * d)) (by
    rw [← starts, Nat.mul_add, Nat.mul_left_comm]; omega)]

/-- `quotientScan_last` for the row `x` of a family from the cell `s`. -/
theorem quotientScan_row_last {n s : ℕ} (x : Fin n) (low : G.mainCells + G.middleCells ≤ base + digits * prior)
    (starts : G.last + (base + digits * prior) = s) :
    quotientScan α r b digits d (fun a j => G.slot (base + digits * (prior + x * d) + (a * d + j))) =
      quotientScan α r b digits d fun a j => s + x * (digits * d) + a * d + j := by
  rw [quotientScan_last α r b digits d G low (start := s + x * (digits * d)) (by
    rw [← starts, Nat.mul_add, Nat.mul_left_comm]; omega)]

/-- `quotientScan_plane` for the plane `t` from the cell `s`. -/
theorem quotientScan_row_plane {planes s : ℕ} (t : Fin planes) (fits : digits * (planes * d) ≤ G.mainCells)
    (starts : G.main = s) :
    quotientScan α r b digits d (fun a j => G.slot (a * (planes * d) + (t * d + j))) =
      quotientScan α r b digits d fun a j => s + a * (planes * d) + t * d + j :=
  quotientScan_plane α r b digits d G fits t.isLt starts

end Scan

/-- A run of cells from `s` moves to the run from `s'` by a map that sends `s + p` to `s' + p`. -/
theorem moved_segCell {move : ℕ → ℕ} {s s' count : ℕ}
    (shift : ∀ p < count, move (s + p) = s' + p) :
    moved move (segCell s count) = segCell s' count := by
  funext p
  unfold moved segCell
  by_cases small : p < count
  · rw [ite_eq_left small, ite_eq_left small, Option.map_some, shift p small]
  · rw [ite_eq_right small, ite_eq_right small]
    rfl

section Values

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- A map that reads the cells from `s` on its first `d = 2^m` positions is a run of `2^m` cells
that ends at `bound` or before. -/
theorem Runs.of_start {m d s bound : ℕ} {f : ℕ → Option ℕ} (pow : 2 ^ m = d)
    (form : ∀ k < d, f k = some (s + k)) (inside : s + d ≤ bound) : Runs m 1 bound f :=
  ⟨run_of_start form (by rw [one_mul, pow]), fun y small b start => by
    obtain rfl : y = 0 := by omega
    rw [zero_mul, form 0 (pow ▸ Nat.two_pow_pos m), add_zero] at start
    cases start
    omega⟩

/-- The value of a quotient term of the replay: `c·Σ_{j<d} Σ_a α^j·b^a·ẽq(r_2, start_a + j)`
for the digit `a` from the cell `start_a`. -/
theorem unitLift_value (α : Ext) (r2 : List Ext) (bits top b digits d : ℕ) [NeZero d]
    (coef : Ext) (start : ℕ → ℕ) (short : bits ≤ r2.length) (pow : 2 ^ Nat.log 2 d = d)
    (le : Nat.log 2 d ≤ bits) (degree : d ≤ top)
    (inside : ∀ a < digits, start a + d ≤ 2 ^ r2.length) :
    (((⟨d, Nat.log 2 d, [(coef, unitRing d)], quotientReads b digits d start⟩ : Lift).value
      (Tables.of α r2 bits top)).into : E) =
      (coef.into : E) * quotientScan (α.into : E) (r2.map Ext.into) b digits d
        fun a j => start a + j := by
  have term := Lift.ofTerm_value_into (E := E) α r2 bits top
    ({ degree := d, reads := quotientReads b digits d start, muls := [(coef, unitRing d)] } : Term)
    short pow le degree (fun r member => by
      obtain ⟨a, small, rfl⟩ := List.mem_map.mp member
      exact Runs.of_start pow (fun k inside => ite_eq_left inside)
        (inside a (List.mem_range.mp small)))
  refine term.trans ?_
  unfold quotientScan
  simp only [quotientReads, List.map_map, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    add_zero, toRq_unitRing, aeval_rep_one, mul_one]
  rw [sum_range_list, Finset.sum_mul]
  conv_rhs => rw [Finset.mul_sum]; enter [2, j]; rw [Finset.mul_sum]
  rw [Finset.sum_comm (s := Finset.range d)]
  refine Finset.sum_congr rfl fun a _ => ?_
  simp only [Function.comp_apply, natFq, Fq.toZq_ofNat, Nat.cast_pow, map_pow, map_natCast]
  rw [Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl fun j member => ?_
  rw [ite_eq_left (Finset.mem_range.mp member), Option.elim_some]
  ring

theorem Tables.of_alpha (α : Ext) (r2 : List Ext) (bits top : ℕ) :
    (Tables.of α r2 bits top).alpha = α :=
  rfl

/-- The value of the quotient term of a ring row of degree `d` with the coefficient `c` and the
digits from `start`: `−c·(α^d + 1)·Σ_{j<d} Σ_a α^j·b^a·ẽq(r_2, start + a·d + j)`. -/
theorem quotientLift_value (α : Ext) (r2 : List Ext) (bits top b digits d : ℕ) [NeZero d]
    (c : Ext) (start : ℕ) (short : bits ≤ r2.length) (pow : 2 ^ Nat.log 2 d = d)
    (le : Nat.log 2 d ≤ bits) (degree : d ≤ top) (inside : start + digits * d ≤ 2 ^ r2.length) :
    (((quotientLift b digits d α c start).value (Tables.of α r2 bits top)).into : E) =
      -((c.into : E) * ((α.into : E) ^ d + 1) * quotientScan (α.into : E) (r2.map Ext.into) b
        digits d fun a j => start + a * d + j) := by
  rw [quotientLift, unitLift_value α r2 bits top b digits d _ _ short pow le degree fun a small => by
    have step : (a + 1) * d ≤ digits * d := Nat.mul_le_mul_right d small
    rw [Nat.succ_mul] at step
    omega]
  rw [Ext.into_neg, Ext.into_mul, Ext.into_add, Ext.into_pow, Ext.into_one]
  ring

/-- The value of the carrier quotient of a group in the order of the code: plane `t` with
`−c·β_t·(α^{d_car} + 1)` and digit `a` of coefficient `j` at `start + a·(P·d_car) + t·d_car + j`. -/
theorem carrierQuotient_value (F : FoldInputs) (code : F.order = .code) (P : PackGroup) (α : Ext)
    (r2 : List Ext) (bits top : ℕ) (c : Ext) (start : ℕ) (short : bits ≤ r2.length)
    (pow : 2 ^ Nat.log 2 P.opening.carrierDegree = P.opening.carrierDegree)
    (le : Nat.log 2 P.opening.carrierDegree ≤ bits) (degree : P.opening.carrierDegree ≤ top)
    (inside : start + F.digits * (P.opening.planes * P.opening.carrierDegree) ≤ 2 ^ r2.length) :
    ((carrierQuotient F P α c start).map fun l =>
      ((l.value (Tables.of α r2 bits top)).into : E)).sum =
      -∑ t : Fin P.opening.planes, (c.into : E) * (((extBasis.getD t Ext.zero).into : E) *
        (((α.into : E) ^ P.opening.carrierDegree + 1) * quotientScan (α.into : E)
          (r2.map Ext.into) F.base F.digits P.opening.carrierDegree fun a j =>
            start + a * (P.opening.planes * P.opening.carrierDegree) +
              t * P.opening.carrierDegree + j)) := by
  unfold carrierQuotient
  rw [List.map_map, sum_range_list, ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [Function.comp_apply, unitLift_value α r2 bits top _ _ _ _ _ short pow le degree
    fun a small => by
      rw [code]
      have step : (a + 1) * (P.opening.planes * P.opening.carrierDegree) ≤
          F.digits * (P.opening.planes * P.opening.carrierDegree) :=
        Nat.mul_le_mul_right _ small
      have plane : (t + 1) * P.opening.carrierDegree ≤ P.opening.planes * P.opening.carrierDegree :=
        Nat.mul_le_mul_right _ t.isLt
      rw [Nat.succ_mul] at step plane
      show start + a * (P.opening.planes * P.opening.carrierDegree) + t * P.opening.carrierDegree +
        0 + P.opening.carrierDegree ≤ _
      omega]
  rw [code]
  simp only [PackingOrder.cell, Nat.add_zero, Ext.into_neg, Ext.into_mul, Ext.into_add,
    Ext.into_pow, Ext.into_one]
  ring

/-- The quotient terms of `n` rows of degree `d` of the replay from the cell `s`, with the
coefficients `ẽq(τ_1, index x)`. -/
theorem quotientRows_value (α : Ext) (r2 : List Ext) (bits top b digits d : ℕ) [NeZero d]
    (τ : List Ext) (s n : ℕ) (index : Fin n → ℕ) (short : bits ≤ r2.length)
    (pow : 2 ^ Nat.log 2 d = d) (le : Nat.log 2 d ≤ bits) (degree : d ≤ top)
    (inside : s + n * (digits * d) ≤ 2 ^ r2.length) :
    ∑ x : Fin n, (((quotientLift b digits d α (eqIndex τ (index x)) (s + x * (digits * d))).value
      (Tables.of α r2 bits top)).into : E) =
      -∑ x : Fin n, eqAt (τ.map Ext.into) (index x) * (((α.into : E) ^ d + 1) *
        quotientScan (α.into : E) (r2.map Ext.into) b digits d
          fun a j => s + x * (digits * d) + a * d + j) := by
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [quotientLift_value α r2 bits top b digits d _ _ short pow le degree (by
    have step : (x + 1) * (digits * d) ≤ n * (digits * d) := Nat.mul_le_mul_right _ x.isLt
    rw [Nat.succ_mul] at step
    omega), eqIndex_into]
  ring

theorem zipIdx_range_map {β : Type} (f : ℕ → β) (n k : ℕ) :
    ((List.range n).map f).zipIdx k = (List.range n).map fun r => (f r, k + r) := by
  refine List.ext_getElem (by simp) fun i _ _ => ?_
  simp [List.getElem_zipIdx]

/-- The rows of `rowStarts` from the block `(d, n)`: the rows `r < n` of degree `d` from the cells
`s + r·δ·d`, then the rows of the next blocks. -/
theorem rowStarts_sum {M : Type} [AddCommMonoid M] (g : (ℕ × ℕ) × ℕ → M) (digits s d n k : ℕ)
    (rest : List (ℕ × ℕ)) :
    (((rowStarts digits s ((d, n) :: rest)).zipIdx k).map g).sum =
      ∑ r : Fin n, g ((d, s + r * (digits * d)), k + r) +
        (((rowStarts digits (s + n * (digits * d)) rest).zipIdx (k + n)).map g).sum := by
  rw [rowStarts, List.zipIdx_append, List.map_append, List.sum_append, List.length_map,
    List.length_range, zipIdx_range_map, List.map_map, sum_range_list]
  rfl

theorem rowStarts_nil {M : Type} [AddCommMonoid M] (g : (ℕ × ℕ) × ℕ → M) (digits s k : ℕ) :
    (((rowStarts digits s []).zipIdx k).map g).sum = 0 :=
  rfl

/-- `rowStarts_sum` for a last block of `n` rows, as a sum over `Fin (1·n)`. -/
theorem rowStarts_sum_one {M : Type} [AddCommMonoid M] (g : (ℕ × ℕ) × ℕ → M) (digits s d n k : ℕ) :
    (((rowStarts digits s [(d, n)]).zipIdx k).map g).sum =
      ∑ r : Fin (1 * n), g ((d, s + r * (digits * d)), k + r) := by
  rw [rowStarts_sum, rowStarts_nil, add_zero,
    Fin.sum_univ_eq_sum_range (fun r => g ((d, s + r * (digits * d)), k + r)),
    Fin.sum_univ_eq_sum_range (fun r => g ((d, s + r * (digits * d)), k + r)), one_mul]

theorem lv_append (T : Tables) (l₁ l₂ : List Lift) :
    ((liftsValue T (l₁ ++ l₂)).into : E) = (liftsValue T l₁).into + (liftsValue T l₂).into := by
  rw [liftsValue, sumList_into, liftsValue, sumList_into, liftsValue, sumList_into, List.map_append,
    List.sum_append]

/-- The quotient terms of the replay for the rows of one block `(d, n)` of `rowStarts` from `s`,
with the coefficients `ẽq(τ_1, c + r)`. -/
theorem rowQuotients_one (α : Ext) (r2 : List Ext) (bits top b digits d : ℕ) [NeZero d]
    (τ : List Ext) (c s n : ℕ) (short : bits ≤ r2.length) (pow : 2 ^ Nat.log 2 d = d)
    (le : Nat.log 2 d ≤ bits) (degree : d ≤ top)
    (inside : s + 1 * n * (digits * d) ≤ 2 ^ r2.length) :
    ((liftsValue (Tables.of α r2 bits top) (((rowStarts digits s [(d, n)]).zipIdx).map fun x =>
      quotientLift b digits x.1.1 α (eqIndex τ (c + x.2)) x.1.2)).into : E) =
      -∑ x : Fin (1 * n), eqAt (τ.map Ext.into) (c + (0 + x)) * (((α.into : E) ^ d + 1) *
        quotientScan (α.into : E) (r2.map Ext.into) b digits d
          fun a j => s + x * (digits * d) + a * d + j) := by
  rw [liftsValue, sumList_into, List.map_map, Function.comp_def, rowStarts_sum_one]
  exact quotientRows_value α r2 bits top b digits d τ s (1 * n) (fun x => c + (0 + x)) short pow
    le degree inside

/-- The quotient terms of the replay for the rows of two blocks `(d₁, n₁)`, `(d₂, n₂)` of
`rowStarts` from `s`, with the coefficients `ẽq(τ_1, c + r)`. -/
theorem rowQuotients_two (α : Ext) (r2 : List Ext) (bits top b digits d₁ n₁ d₂ n₂ : ℕ)
    [NeZero d₁] [NeZero d₂] (τ : List Ext) (c s : ℕ) (short : bits ≤ r2.length)
    (pow₁ : 2 ^ Nat.log 2 d₁ = d₁) (le₁ : Nat.log 2 d₁ ≤ bits) (degree₁ : d₁ ≤ top)
    (pow₂ : 2 ^ Nat.log 2 d₂ = d₂) (le₂ : Nat.log 2 d₂ ≤ bits) (degree₂ : d₂ ≤ top)
    (inside₁ : s + n₁ * (digits * d₁) ≤ 2 ^ r2.length)
    (inside₂ : s + n₁ * (digits * d₁) + n₂ * (digits * d₂) ≤ 2 ^ r2.length) :
    ((liftsValue (Tables.of α r2 bits top) (((rowStarts digits s [(d₁, n₁), (d₂, n₂)]).zipIdx).map
      fun x => quotientLift b digits x.1.1 α (eqIndex τ (c + x.2)) x.1.2)).into : E) =
      -∑ x : Fin n₁, eqAt (τ.map Ext.into) (c + (0 + x)) * (((α.into : E) ^ d₁ + 1) *
        quotientScan (α.into : E) (r2.map Ext.into) b digits d₁
          fun a j => s + x * (digits * d₁) + a * d₁ + j) +
      -∑ x : Fin n₂, eqAt (τ.map Ext.into) (c + (0 + n₁ + x)) * (((α.into : E) ^ d₂ + 1) *
        quotientScan (α.into : E) (r2.map Ext.into) b digits d₂
          fun a j => s + n₁ * (digits * d₁) + x * (digits * d₂) + a * d₂ + j) := by
  rw [liftsValue, sumList_into, List.map_map, Function.comp_def, rowStarts_sum, rowStarts_sum,
    rowStarts_nil, add_zero]
  rw [quotientRows_value α r2 bits top b digits d₁ τ s n₁ (fun x => c + (0 + x)) short pow₁ le₁
      degree₁ inside₁,
    quotientRows_value α r2 bits top b digits d₂ τ (s + n₁ * (digits * d₁)) n₂
      (fun x => c + (0 + n₁ + x)) short pow₂ le₂ degree₂ inside₂]

end Values

end NightstreamFPrime.Spec.Compression.Akita.Replay
