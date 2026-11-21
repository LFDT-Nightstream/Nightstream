import NightstreamFPrime.Spec.Compression.Akita.Extract.SetupWeight

/-! The split (132) of the public weight of an offloaded fold in the form of the pinned Akita code
(ePrint 2026/1983, revision of 2026-10-15, Section 9 with (132) and (133);
`eval_flat_at_point_with_deferred_setup` in `crates/akita-verifier/src/relation/evaluation.rs` and
`PreparedLiftedRelationPoint::new` in `crates/akita-verifier/src/relation/prepared_point.rs` of
the Akita code).

The code multiplies the claim of Stage 3 by the evaluation of the powers of `α` at the first `κ`
coordinates of `r_2`, the coefficient coordinates, and its Stage 3 has a weight that reads only
the other coordinates of `r_2`. This is the form of the paper with the weight `ω_setup` written as
a product. The product holds because each element of `ẑ`, `ê` and `t̂` that a view of the setup
multiplies starts at a multiple of `2^κ` in the witness, and each ring degree is a multiple of
`2^κ`: in a block of `2^κ` cells, the weight of the cell `k` is `α^k` times the weight of the
first cell.

- `powerScale α point` is `evaluate_power_sequence_mle` of the code
  (`crates/akita-algebra/src/ring/eval.rs`).
- `Geometric α κ t` states that the table `t` has this form in each block of `2^κ` cells, and
  `Geometric.split` is the product for its multilinear extension.
- `Form.Blocked` states that each occurrence of a sum (110) reads a whole run of cells that starts
  at a multiple of `2^κ`, or no cell. `Form.scan_geometric` gives `Geometric` for its weights.
- `Batch.Blocked B κ` has the conditions on the numbers of a fold, and
  `Batch.weight_geometric` states that the part of the public weight that depends on the setup
  vector is `Geometric`.
- `Batch.codeScan` is the split `SetupScan` with the factor `powerScale α (r_2.take κ)` and a
  weight that reads `r_2.drop κ`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib Polynomial

section Geometric

variable {E : Type} [CommRing E]

/-- The multilinear extension of the powers `α^0, α^1, …` at a point:
`Π_i ((1 − c_i) + c_i·α^{2^i})` (`evaluate_power_sequence_mle` of the Akita code). -/
def powerScale (α : E) : List E → E
  | [] => 1
  | c :: rest => (1 - c + c * α) * powerScale (α * α) rest

/-- In each block of `2^κ` cells, the cell `k` of the table is `α^k` times the first cell. -/
def Geometric (α : E) (κ : ℕ) (t : ℕ → E) : Prop :=
  ∀ y k, k < 2 ^ κ → t (2 ^ κ * y + k) = α ^ k * t (2 ^ κ * y)

namespace Geometric

variable {α : E} {κ : ℕ} {t u : ℕ → E}

theorem congr (same : ∀ x, t x = u x) (geo : Geometric α κ u) : Geometric α κ t := by
  rw [funext same]
  exact geo

theorem zero : Geometric α κ fun _ => (0 : E) := fun _ _ _ => (mul_zero _).symm

theorem sub (left : Geometric α κ t) (right : Geometric α κ u) :
    Geometric α κ fun x => t x - u x := fun y k small => by
  show t _ - u _ = α ^ k * (t _ - u _)
  rw [left y k small, right y k small, mul_sub]

theorem mul_left (geo : Geometric α κ t) (a : E) : Geometric α κ fun x => a * t x :=
  fun y k small => by
    show a * t _ = α ^ k * (a * t _)
    rw [geo y k small, mul_left_comm]

theorem sum {ι : Type} (s : Finset ι) {f : ι → ℕ → E} (each : ∀ i, Geometric α κ (f i)) :
    Geometric α κ fun x => ∑ i ∈ s, f i x := fun y k small => by
  show ∑ i ∈ s, f i _ = α ^ k * ∑ i ∈ s, f i _
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => each i y k small

theorem odd (geo : Geometric α (κ + 1) t) (i : ℕ) : t (2 * i + 1) = α * t (2 * i) := by
  obtain ⟨q, m, small, rfl⟩ : ∃ q m, m < 2 ^ κ ∧ i = 2 ^ κ * q + m :=
    ⟨i / 2 ^ κ, i % 2 ^ κ, Nat.mod_lt _ (by positivity), (Nat.div_add_mod i _).symm⟩
  have one := geo q (2 * m + 1) (by rw [pow_succ]; omega)
  have two := geo q (2 * m) (by rw [pow_succ]; omega)
  rw [show 2 * (2 ^ κ * q + m) + 1 = 2 ^ (κ + 1) * q + (2 * m + 1) by ring,
    show 2 * (2 ^ κ * q + m) = 2 ^ (κ + 1) * q + 2 * m by ring, one, two]
  ring

theorem even (geo : Geometric α (κ + 1) t) : Geometric (α * α) κ fun i => t (2 * i) :=
  fun y k small => by
    show t (2 * (2 ^ κ * y + k)) = (α * α) ^ k * t (2 * (2 ^ κ * y))
    rw [show 2 * (2 ^ κ * y + k) = 2 ^ (κ + 1) * y + 2 * k by ring,
      show 2 * (2 ^ κ * y) = 2 ^ (κ + 1) * y by ring, geo y (2 * k) (by rw [pow_succ]; omega),
      ← pow_two, ← pow_mul]

private theorem split_length : ∀ (lo hi : List E) {α : E} {t : ℕ → E},
    Geometric α lo.length t →
      mle (lo ++ hi) t = powerScale α lo * mle hi fun y => t (2 ^ lo.length * y)
  | [], hi, α, t, _ => by
      simp only [List.nil_append, powerScale, one_mul, List.length_nil, pow_zero]
  | c :: rest, hi, α, t, geo => by
      have low : bindLow c t = fun i => (1 - c + c * α) * t (2 * i) := funext fun i => by
        show (1 - c) * t (2 * i) + c * t (2 * i + 1) = _
        rw [geo.odd i]
        ring
      calc mle (c :: rest ++ hi) t = mle (rest ++ hi) (bindLow c t) := rfl
        _ = (1 - c + c * α) * mle (rest ++ hi) fun i => t (2 * i) := by rw [low, mle_mul_left]
        _ = (1 - c + c * α) * (powerScale (α * α) rest *
              mle hi fun y => t (2 * (2 ^ rest.length * y))) := by
            rw [split_length rest hi geo.even]
        _ = (1 - c + c * α) * powerScale (α * α) rest *
              mle hi fun y => t (2 ^ (rest.length + 1) * y) := by
            rw [← mul_assoc]
            exact congrArg _ (congrArg _ (funext fun y => by
              rw [pow_succ, mul_comm (2 ^ rest.length) 2, mul_assoc]))

/-- The multilinear extension of a table with the form `Geometric` is the product of
`powerScale` at the first `κ` coordinates and the extension of the first cells of the blocks at
the other coordinates. -/
theorem split (lo hi : List E) (length : lo.length = κ) (geo : Geometric α κ t) :
    mle (lo ++ hi) t = powerScale α lo * mle hi fun y => t (2 ^ κ * y) := by
  subst length
  exact split_length lo hi geo

end Geometric

/-- The weights `α^j` of one run of `d` cells from the cell `s`: the form `Geometric` when `d` and
`s` are multiples of `2^κ`. -/
theorem geometric_run (α : E) {κ d s : ℕ} (degree : 2 ^ κ ∣ d) (start : 2 ^ κ ∣ s) :
    Geometric α κ fun x => ∑ j ∈ Finset.range d, if s + j = x then α ^ j else 0 := by
  obtain ⟨b, rfl⟩ := degree
  obtain ⟨a, rfl⟩ := start
  intro y k small
  beta_reduce
  by_cases inside : a ≤ y ∧ y < a + b
  · have lower : 2 ^ κ * a ≤ 2 ^ κ * y := Nat.mul_le_mul_left _ inside.1
    have upper : 2 ^ κ * (y + 1) ≤ 2 ^ κ * (a + b) := Nat.mul_le_mul_left _ inside.2
    rw [Nat.mul_add, Nat.mul_add, Nat.mul_one] at upper
    rw [Finset.sum_eq_single (2 ^ κ * y + k - 2 ^ κ * a),
      Finset.sum_eq_single (2 ^ κ * y - 2 ^ κ * a)]
    · rw [ite_eq_left (by omega), ite_eq_left (by omega), ← pow_add]
      exact congrArg _ (by omega)
    · exact fun j _ other => ite_eq_right (by omega)
    · exact fun outside => absurd (Finset.mem_range.mpr (by omega)) outside
    · exact fun j _ other => ite_eq_right (by omega)
    · exact fun outside => absurd (Finset.mem_range.mpr (by omega)) outside
  · have outside : ∀ j < 2 ^ κ * b, ∀ k < 2 ^ κ, 2 ^ κ * a + j ≠ 2 ^ κ * y + k :=
      fun j member k small equal => inside ⟨by
        by_contra before
        have step : 2 ^ κ * (y + 1) ≤ 2 ^ κ * a := Nat.mul_le_mul_left _ (by omega)
        rw [Nat.mul_add, Nat.mul_one] at step
        omega, by
        by_contra after
        have step : 2 ^ κ * (a + b) ≤ 2 ^ κ * y := Nat.mul_le_mul_left _ (by omega)
        rw [Nat.mul_add] at step
        omega⟩
    rw [Finset.sum_eq_zero fun j member =>
        ite_eq_right (outside j (Finset.mem_range.mp member) k small),
      Finset.sum_eq_zero fun j member =>
        ite_eq_right fun equal => outside j (Finset.mem_range.mp member) 0 (by positivity)
          (by rw [Nat.add_zero]; exact equal), mul_zero]

end Geometric

section Rows

variable {N : ℕ} {E : Type} [CommRing E] [Algebra Zq E]

/-- The weight 1 of the addresses that `ψ` moves to the cell `x`. -/
def mark (ψ : Option ℕ → Option ℕ) (x : ℕ) (o : Option ℕ) : E := if ψ o = some x then 1 else 0

theorem RingRow.readWeight_eq_mark (row : RingRow) (α : E) (u : ℕ) :
    row.readWeight α u = row.scan α (mark (fun o => o) u) :=
  row.readWeight_eq_scan α u

/-- Each occurrence of the sum reads, after the move `ψ`, the `d` cells from a multiple of `2^κ`
in their order, or no cell. -/
def Form.Blocked {d : ℕ} (f : Form d) (κ : ℕ) (ψ : Option ℕ → Option ℕ) : Prop :=
  ∀ c, (∀ j < d, ψ (f.address c j) = none) ∨
    ∃ s, 2 ^ κ ∣ s ∧ ∀ j < d, ψ (f.address c j) = some (s + j)

theorem Form.scan_geometric {d : ℕ} [NeZero d] (f : Form d) (α : E) {κ : ℕ}
    {ψ : Option ℕ → Option ℕ} (degree : 2 ^ κ ∣ d) (blocked : f.Blocked κ ψ) :
    Geometric α κ fun x => f.scan α (mark ψ x) := by
  unfold Form.scan
  refine Geometric.sum _ fun c => ?_
  rcases blocked c with empty | ⟨s, start, reads⟩
  · refine Geometric.congr (fun x => Finset.sum_eq_zero fun j member => ?_) Geometric.zero
    show _ * _ * (if ψ (f.address c j) = some x then (1 : E) else 0) = 0
    rw [empty j (Finset.mem_range.mp member)]
    simp
  · refine Geometric.congr (u := fun x => aeval α (rep (f.multiplier c)) *
      ∑ j ∈ Finset.range d, if s + j = x then α ^ j else 0) (fun x => ?_)
      ((geometric_run α degree start).mul_left _)
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j member => ?_
    show _ * _ * (if ψ (f.address c j) = some x then (1 : E) else 0) = _
    rw [reads j (Finset.mem_range.mp member)]
    simp only [Option.some.injEq, mul_ite, mul_one, mul_zero]

/-- The part of the weights of a family of sums (110) that depends on `S` has the form
`Geometric`. -/
def GeoForm (α : E) (κ : ℕ) (ψ : Option ℕ → Option ℕ) {d : ℕ} [NeZero d]
    (F : (Fin N → Zq) → Form d) : Prop :=
  ∀ S, Geometric α κ fun x => (F S).scan α (mark ψ x) - (F 0).scan α (mark ψ x)

/-- The part of the weights of a family of ring rows that depends on `S` has the form
`Geometric`. -/
def GeoRow (α : E) (κ : ℕ) (ψ : Option ℕ → Option ℕ) (F : (Fin N → Zq) → RingRow) : Prop :=
  ∀ S, Geometric α κ fun x => (F S).scan α (mark ψ x) - (F 0).scan α (mark ψ x)

namespace GeoForm

variable {α : E} {κ : ℕ} {ψ : Option ℕ → Option ℕ} {d : ℕ} [NeZero d]
  {F : (Fin N → Zq) → Form d}

theorem of_blocked (degree : 2 ^ κ ∣ d) (blocked : ∀ S, (F S).Blocked κ ψ) : GeoForm α κ ψ F :=
  fun S => ((F S).scan_geometric α degree (blocked S)).sub
    ((F 0).scan_geometric α degree (blocked 0))

theorem sub (geo : GeoForm α κ ψ F) (g : Form d) : GeoForm α κ ψ fun S => (F S).sub g :=
  fun S => Geometric.congr (fun x => by
    show ((F S).sub g).scan α (mark ψ x) - ((F 0).sub g).scan α (mark ψ x) = _
    rw [Form.scan_sub, Form.scan_sub, sub_sub_sub_cancel_right]) (geo S)

end GeoForm

namespace GeoRow

variable {α : E} {κ : ℕ} {ψ : Option ℕ → Option ℕ} {F : (Fin N → Zq) → RingRow}

theorem of_eq (same : ∀ S, F S = F 0) : GeoRow α κ ψ F := fun S =>
  Geometric.congr (fun x => by rw [same S, sub_self]) Geometric.zero

theorem relocate {move : ℕ → ℕ} (geo : GeoRow α κ (fun o => ψ (o.map move)) F) :
    GeoRow α κ ψ fun S => (F S).relocate move :=
  geo

theorem row {d : ℕ} [NeZero d] {F : (Fin N → Zq) → Form d} (geo : GeoForm α κ ψ F)
    (target : (Fin N → Zq) → Rq d) : GeoRow α κ ψ fun S => (F S).row (target S) :=
  fun S => Geometric.congr (fun x => by
    show ((F S).row (target S)).scan α (mark ψ x) - ((F 0).row (target 0)).scan α (mark ψ x) = _
    rw [Form.scan_row, Form.scan_row]) (geo S)

theorem append {m k : ℕ} {a : (Fin N → Zq) → Fin m → RingRow}
    {b : (Fin N → Zq) → Fin k → RingRow} (left : ∀ i, GeoRow α κ ψ fun S => a S i)
    (right : ∀ j, GeoRow α κ ψ fun S => b S j) (r : Fin (m + k)) :
    GeoRow α κ ψ fun S => Fin.append (a S) (b S) r := by
  refine Fin.addCases (fun i => ?_) (fun j => ?_) r
  · simpa only [Fin.append_left] using left i
  · simpa only [Fin.append_right] using right j

theorem gather {n : ℕ} {count : Fin n → ℕ}
    {rows : (Fin N → Zq) → (g : Fin n) → Fin (count g) → RingRow}
    (each : ∀ g i, GeoRow α κ ψ fun S => rows S g i) (r : Fin (∑ g, count g)) :
    GeoRow α κ ψ fun S => Akita.gather (rows S) r :=
  each _ _

theorem imageRow {d n : ℕ} [NeZero d] : (chain : Option ChainShape) →
    {image : (Fin N → Zq) → Fin n → Form d} → (∀ j, GeoForm α κ ψ fun S => image S j) →
    (first : ℕ → Option ℕ) → (payload : Payload d n chain) →
    ∀ j, GeoRow α κ ψ fun S => Akita.imageRow chain (image S) first payload j
  | none, _, images, _, payload, j => row (images j) fun _ => payload j
  | some _, _, images, first, _, j => row ((images j).sub (Form.recompose d n first j)) fun _ => 0

/-- The part of the weight `liftWeight` that depends on `S` has the form `Geometric` when the
reads of the quotient do not depend on `S`. -/
theorem liftWeight (geo : GeoRow α κ (fun o => o) F) {q : (Fin N → Zq) → QuotientRead}
    (S : Fin N → Zq) (degree : (F S).degree = (F 0).degree)
    (same : ∀ u, (q S).weight (F S).degree α u = (q 0).weight (F 0).degree α u) :
    Geometric α κ fun x => (F S).liftWeight (q S) α x - (F 0).liftWeight (q 0) α x :=
  Geometric.congr (fun x => by
    unfold RingRow.liftWeight
    rw [same x, degree, RingRow.readWeight_eq_mark, RingRow.readWeight_eq_mark,
      sub_sub_sub_cancel_right]) (geo S)

end GeoRow

end Rows

variable {N : ℕ} {E : Type} [Field E] [Algebra Zq E]

omit [Algebra Zq E] in
theorem Geometric.liftEntry {α : E} {κ m count : ℕ} (slot : Fin m → Fin count)
    (basis : Fin m → E) {lift lift' : ℕ → Fin m → E} {scalar scalar' : ℕ → E}
    (lifts : ∀ r, Geometric α κ fun x => lift x r - lift' x r)
    (scalars : ∀ x, scalar x = scalar' x) (i : ℕ) :
    Geometric α κ fun x => Akita.liftEntry count slot basis (lift x) (scalar x) i -
      Akita.liftEntry count slot basis (lift' x) (scalar' x) i := by
  unfold Akita.liftEntry
  by_cases small : i < count
  · refine Geometric.congr (u := fun x => ∑ r, if slot r = ⟨i, small⟩ then
      basis r * (lift x r - lift' x r) else 0) (fun x => ?_)
      (Geometric.sum _ fun r => ?_)
    · rw [dite_eq_left small, dite_eq_left small, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun r _ => by split <;> ring
    · by_cases here : slot r = ⟨i, small⟩
      · exact Geometric.congr (fun x => ite_eq_left here) ((lifts r).mul_left _)
      · exact Geometric.congr (fun x => ite_eq_right here) Geometric.zero
  · exact Geometric.congr (fun x => by rw [dite_eq_right small, dite_eq_right small, scalars x, sub_self])
      Geometric.zero

omit [Algebra Zq E] in
theorem Geometric.batchWeight {α : E} {κ : ℕ} {rows rows' : FieldRows E}
    (entry : ∀ i, Geometric α κ fun x => rows.weightAt i x - rows'.weightAt i x) (τ : List E) :
    Geometric α κ fun x => rows.batchWeight τ x - rows'.batchWeight τ x := by
  refine Geometric.congr (u := fun x => ∑ i ∈ Finset.range (2 ^ τ.length),
    eqAt τ i * (rows.weightAt i x - rows'.weightAt i x)) (fun x => ?_)
    (Geometric.sum _ fun i => (entry i).mul_left _)
  simp only [FieldRows.batchWeight, mle_eq_cubeSum, cubeSum_eq_sum_range, mul_sub,
    Finset.sum_sub_distrib]

namespace Batch

variable {n : ℕ} (B : Batch (Fin n))

/-- The conditions of the split in the form of the code on the numbers of a fold, for the block
of `2^κ` cells (`relation_coefficient_block_len` of the Akita code): the main segment of each
group, its segments `ê` and `t̂`, and the ring degrees of `A`, `B` and `D` are multiples of
`2^κ`. -/
def Blocked (κ : ℕ) : Prop :=
  (∀ g, 2 ^ κ ∣ B.mainStart g ∧ 2 ^ κ ∣ (B.shape g).responseCells (B.fold g) ∧
    2 ^ κ ∣ (B.opening g).innerStart (B.fold g) ∧ 2 ^ κ ∣ (B.shape g).innerDegree ∧
    2 ^ κ ∣ (B.shape g).outerDegree) ∧ 2 ^ κ ∣ B.openDegree

instance (κ : ℕ) : Decidable (B.Blocked κ) := by
  unfold Blocked
  infer_instance

variable {B}

theorem place_main (digits : ℕ) (g : Fin n) {x : ℕ} (small : x < B.mainCells g) :
    (B.gaps digits).place (B.relocate g x) = B.mainStart g + x := by
  have before : B.relocate g x < (B.gaps digits).main := B.relocate_lt_chainStart g small
  rw [(B.gaps digits).place_of_lt before, B.relocate_main g small]

theorem responseForm_blocked {κ : ℕ} (blocked : B.Blocked κ) (digits : ℕ) (g : Fin n)
    (key : Key (B.shape g)) (a : Fin (B.shape g).innerRank) :
    (responseForm (B.fold g) key a).Blocked κ fun o =>
      (o.map (B.relocate g)).map (B.gaps digits).place := by
  intro col
  refine Or.inr ⟨B.mainStart g +
    (B.shape g).responseCode (B.fold g) col * (B.shape g).innerDegree,
    dvd_add (blocked.1 g).1 (Dvd.dvd.mul_left (blocked.1 g).2.2.2.1 _), fun j small => ?_⟩
  have cell : (B.shape g).responseCell (B.fold g) col j =
      some ((B.shape g).responseCode (B.fold g) col * (B.shape g).innerDegree + j) :=
    ite_eq_left small
  show (((B.shape g).responseCell (B.fold g) col j).map (B.relocate g)).map
    (B.gaps digits).place = _
  rw [cell]
  show some ((B.gaps digits).place (B.relocate g _)) = _
  rw [place_main digits g ((cell_lt _ small).trans_le
    (Nat.le_add_right_of_le (Nat.le_add_right _ _))), Nat.add_assoc]

theorem outerForm_blocked {κ : ℕ} (blocked : B.Blocked κ) (digits : ℕ) (g : Fin n)
    (key : Key (B.shape g)) (j : Fin ((B.shape g).slices * (B.shape g).outerRank)) :
    (outerForm (B.opening g) (B.fold g) key j).Blocked κ fun o =>
      (o.map (B.relocate g)).map (B.gaps digits).place := by
  intro entry
  by_cases inside : (B.shape g).sliceStart ((finProdFinEquiv.symm j).1 : ℕ) + (entry.1 : ℕ) <
      (B.shape g).sliceStart (((finProdFinEquiv.symm j).1 : ℕ) + 1)
  · refine Or.inr ⟨B.mainStart g + ((B.opening g).innerStart (B.fold g) +
      (B.shape g).innerCode (⟨_, inside.trans_le ((B.shape g).sliceStart_le
        (Nat.succ_le_of_lt (finProdFinEquiv.symm j).1.isLt))⟩, entry.2) *
        (B.shape g).outerDegree),
      dvd_add (blocked.1 g).1 (dvd_add (blocked.1 g).2.2.1
        (Dvd.dvd.mul_left (blocked.1 g).2.2.2.2 _)), fun k small => ?_⟩
    have cell : (B.opening g).sliceCell (B.fold g) (finProdFinEquiv.symm j).1 entry k =
        some ((B.opening g).innerStart (B.fold g) + ((B.shape g).innerCode
          (⟨_, inside.trans_le ((B.shape g).sliceStart_le
            (Nat.succ_le_of_lt (finProdFinEquiv.symm j).1.isLt))⟩, entry.2) *
          (B.shape g).outerDegree + k)) :=
      (dite_eq_left inside).trans (ite_eq_left small)
    show (((B.opening g).sliceCell (B.fold g) (finProdFinEquiv.symm j).1 entry k).map
      (B.relocate g)).map (B.gaps digits).place = _
    rw [cell]
    show some ((B.gaps digits).place (B.relocate g _)) = _
    rw [place_main digits g (Nat.add_lt_add_left (cell_lt _ small) _)]
    exact congrArg some (by omega)
  · refine Or.inl fun k _ => ?_
    have cell : (B.opening g).sliceCell (B.fold g) (finProdFinEquiv.symm j).1 entry k = none :=
      dite_eq_right inside
    show (((B.opening g).sliceCell (B.fold g) (finProdFinEquiv.symm j).1 entry k).map
      (B.relocate g)).map (B.gaps digits).place = _
    rw [cell]
    rfl

theorem openForm_blocked {κ : ℕ} (blocked : B.Blocked κ) (digits : ℕ) (key : B.OpenKey)
    (j : Fin (1 * B.openRank)) :
    (B.openForm key j).Blocked κ fun o => o.map (B.gaps digits).place := by
  intro column
  refine Or.inr ⟨B.mainStart column.1 + ((B.shape column.1).responseCells (B.fold column.1) +
    (B.opening column.1).partialCode column.2 * B.openDegree),
    dvd_add (blocked.1 column.1).1 (dvd_add (blocked.1 column.1).2.1
      (Dvd.dvd.mul_left blocked.2 _)), fun k small => ?_⟩
  have cell : (B.opening column.1).partialCell (B.fold column.1) column.2 k =
      some ((B.shape column.1).responseCells (B.fold column.1) +
        ((B.opening column.1).partialCode column.2 * B.openDegree + k)) :=
    ite_eq_left small
  show (((B.opening column.1).partialCell (B.fold column.1) column.2 k).map
    (B.relocate column.1)).map (B.gaps digits).place = _
  rw [cell]
  show some ((B.gaps digits).place (B.relocate column.1 _)) = _
  rw [place_main digits column.1 ((Nat.add_lt_add_left (cell_lt _ small) _).trans_le
    (Nat.le_add_right _ _))]
  exact congrArg some (by omega)

/-- The part that depends on `S` of the weights of each ring row of `Batch.flatRows` for the
statement `st.over S`, at the positions `place` of the witness, has the form `Geometric`. -/
theorem flatRows_geometric (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) {κ : ℕ} (blocked : B.Blocked κ) (digits : ℕ) (α : E) :
    ∀ r, GeoRow α κ (fun o => o.map (B.gaps digits).place)
      fun S : Fin N → Zq => B.flatRows β (st.over S) challenge r := by
  unfold flatRows groupRows innerRow
  refine GeoRow.append (GeoRow.gather fun g => ?_) ?_
  · refine GeoRow.append (fun t => GeoRow.of_eq fun _ => rfl) (GeoRow.append ?_ ?_)
    · intro a
      exact GeoRow.relocate (GeoRow.row (GeoForm.sub (GeoForm.of_blocked (blocked.1 g).2.2.2.1
        fun _ => responseForm_blocked blocked digits g _ a) _) fun _ => 0)
    · exact fun j => GeoRow.relocate (GeoRow.imageRow _ (fun j =>
        GeoForm.of_blocked (blocked.1 g).2.2.2.2
          fun _ => outerForm_blocked blocked digits g _ j) _ _ j)
  · refine GeoRow.append (GeoRow.imageRow _ (fun j => GeoForm.of_blocked blocked.2
        fun _ => openForm_blocked blocked digits _ j) _ _)
      (GeoRow.append (GeoRow.gather fun _ _ => GeoRow.of_eq fun _ => rfl)
        (GeoRow.append (fun _ => GeoRow.of_eq fun _ => rfl)
          (GeoRow.append (GeoRow.gather fun _ _ => GeoRow.of_eq fun _ => rfl)
            fun _ => GeoRow.of_eq fun _ => rfl)))

variable [Fintype E]

/-- The part that depends on `S` of the public weight `m_{τ_1}` of `Batch.liftedReduction` for
the statement `st.over S` has the form `Geometric`. -/
theorem weight_geometric (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ)
    (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    {κ : ℕ} (blocked : B.Blocked κ) (α : E) (τ : List E) (S : Fin N → Zq) :
    Geometric α κ fun x =>
      ((liftedReduction β (st.over S) challenge μ fits base digits room batch rowsFit).rows
        α).batchWeight τ x -
      ((liftedReduction β (st.over (0 : Fin N → Zq)) challenge μ fits base digits room batch
        rowsFit).rows α).batchWeight τ x := by
  have rows := flatRows_geometric (N := N) β st challenge blocked digits α
  refine Geometric.batchWeight (fun i => ?_) τ
  refine Geometric.congr (fun x => congrArg₂ (· - ·)
    (LiftRelation.liftRows_weightAt _ α i x) (LiftRelation.liftRows_weightAt _ α i x)) ?_
  refine Geometric.liftEntry (m := B.planeRows)
    ((Join.gather fun g => (Join.all (β g)).append (Join.plain E (B.groupTail g))).append
      (Join.plain E B.restCount)).slot
    ((Join.gather fun g => (Join.all (β g)).append (Join.plain E (B.groupTail g))).append
      (Join.plain E B.restCount)).basis (fun r => ?_) (fun x => ?_) i
  · exact GeoRow.liftWeight
      (F := fun S => (B.flatRows β (st.over S) challenge r).relocate (B.gaps digits).place)
      (q := fun S => ⟨digits, fun a => (base : Zq) ^ (a : ℕ),
        (B.flatRelation β (st.over S) challenge μ fits).quotientCell (B.gaps digits) digits r⟩)
      (GeoRow.relocate (rows r)) S ((flatRows_affine (N := N) β st challenge r).degree S)
      fun u => quotientWeight_over β st challenge μ fits base digits α S r u
  · simp only [FlatRelation.withQuotients, Batch.flatRelation, Batch.scalarWeight,
      Batch.Statement.over]

/-- The split `SetupScan` in the form of the pinned Akita code (Akita (132) and (133);
`eval_flat_at_point_with_deferred_setup` of the code) of the public weight of
`Batch.liftedReduction` for the statement `st.over S` with `S_p = flat(p)` for the `2^ν` cells of
the setup prefix, when the fold has the conditions `Batch.Blocked` for the block of `2^κ` cells.
`m_local` is the weight at `S = 0`. The factor is `powerScale` of `α` at the first `κ`
coordinates of `r_2` (`common_alpha_evaluation` of the code). The weight of Stage 3 at `p` reads
the other coordinates of `r_2`: it is the extension at them of the first cells of the blocks of
the weight at the unit vector `e_p` minus the weight at `S = 0`. -/
noncomputable def codeScan (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ)
    (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (ν : ℕ) (flat : Table) (κ : ℕ) (blocked : B.Blocked κ) (low : κ ≤ μ) :
    SetupScan (liftedReduction β (st.over fun p : Fin (2 ^ ν) => flat p) challenge μ fits base
      digits room batch rowsFit) ν flat where
  part α τ r2 := mle r2 (((liftedReduction β (st.over (0 : Fin (2 ^ ν) → Zq)) challenge μ
    fits base digits room batch rowsFit).rows α).batchWeight τ)
  scale α _ r2 := powerScale α (r2.take κ)
  weight α τ r2 p := if small : p < 2 ^ ν then
      mle (r2.drop κ) fun y =>
        ((liftedReduction β (st.over (Pi.single ⟨p, small⟩ 1 : Fin (2 ^ ν) → Zq))
          challenge μ fits base digits room batch rowsFit).rows α).batchWeight τ (2 ^ κ * y) -
        ((liftedReduction β (st.over (0 : Fin (2 ^ ν) → Zq)) challenge μ fits base digits
          room batch rowsFit).rows α).batchWeight τ (2 ^ κ * y)
    else 0
  split α τ r2 length := by
    refine (weight_affine β st challenge μ fits base digits room batch rowsFit α (List.ofFn τ)
      r2 (fun p => flat p)).trans ?_
    beta_reduce
    rw [← cubeSum_mul_left, cubeSum_eq_sum_range, Finset.sum_range]
    refine congrArg _ (Finset.sum_congr rfl fun p _ => ?_)
    rw [dite_eq_left p.isLt, mul_left_comm]
    refine congrArg₂ (· * ·) rfl ((mle_sub r2 _ _).symm.trans ?_)
    have joined := (weight_geometric β st challenge μ fits base digits room batch rowsFit
      blocked α (List.ofFn τ) (Pi.single p 1 : Fin (2 ^ ν) → Zq)).split (r2.take κ) (r2.drop κ)
      (by rw [List.length_take, length]; exact Nat.min_eq_left low)
    rw [List.take_append_drop] at joined
    exact joined

end Batch

end NightstreamFPrime.Spec.Compression.Akita
