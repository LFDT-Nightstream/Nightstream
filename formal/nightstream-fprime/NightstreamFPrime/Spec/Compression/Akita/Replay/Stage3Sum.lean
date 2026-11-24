import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Weight
import NightstreamFPrime.Spec.Compression.Akita.Replay.Rows

/-! The weight of Stage 3 of the replay (`stage3Weight` of `Replay/Stage3Weight.lean`) in `E`.

- `laneIn` is `Side.lane` in `E` (`lane_into`), and `groupIn` is `groupWeight` in `E`
  (`groupWeight_into`): the parts `aIn`, `bIn` and `dIn` of `A`, `B` and `D` of one group as sums
  over ranges, with the factor `powerScale(α^base, ρ_mid)` of the setup side and `ẽq` of the other
  coordinates.
- `stage3Weight_into`: `stage3Weight` is `powerScale(α, ρ_low)` times the sum of `groupIn` over
  the placed groups of the plan.
- `placedFrom` is the list `SetupPlan.placed` (`placed_eq`), and `placedFrom_ofFn` gives its sum
  for the groups `f 0, …, f (n − 1)`: each group has the cells, rows and columns of `D` of the
  groups before it. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

theorem pick_of (l : List ℕ) (b : Ext) (r : List Ext) (ratio : ℕ) :
    ((l.map (Side.of b r)).find? (·.ratio = ratio)).getD (Side.of b r ratio) =
      Side.of b r ratio := by
  rw [List.find?_map]
  cases h : l.find? ((fun s : Side => decide (s.ratio = ratio)) ∘ Side.of b r) with
  | none => rfl
  | some x =>
      have found := List.find?_some h
      simp only [Function.comp, Side.of] at found
      have same : x = ratio := of_decide_eq_true found
      subst same
      rfl

/-- `stage3Weight` with the sides of the code: the side of a role is `Side.of` of its ratio. -/
theorem stage3Weight_eq (P : SetupPlan) (α : Ext) (τ r2 ρ : List Ext) :
    stage3Weight P α τ r2 ρ = powerScaleExt α (ρ.take (Nat.log 2 P.base)) *
      (P.placed.map fun p => groupWeight P (eqIndex τ)
        (Side.of (powExt α P.block) (r2.drop (Nat.log 2 P.block)))
        (Side.of (powExt α P.base) (ρ.drop (Nat.log 2 P.base))) (powExt α P.block)
        (r2.drop (Nat.log 2 P.block))
        (fun f => powExt (Ext.ofBase (Fq.ofNat (2 ^ P.logBasis))) f) p).foldl (· + ·)
          Ext.zero := by
  unfold stage3Weight
  simp only [pick_of]

/-- The list `SetupPlan.placed` from the first cell `z`, the first row `r` and the first column
`c` of `D`. -/
def placedFrom (P : SetupPlan) : List SetupGroup → ℕ → ℕ → ℕ → List Placed
  | [], _, _, _ => []
  | G :: rest, z, r, c =>
      ⟨G, z, r, c⟩ :: placedFrom P rest (z + G.unitLen) (r + G.rows) (c + G.dCols P.openDegree)

theorem placed_foldl (P : SetupPlan) : ∀ (l : List SetupGroup) (acc : Array Placed) (z r c : ℕ),
    (l.foldl (fun (acc : Array Placed × ℕ × ℕ × ℕ) G =>
      (acc.1.push ⟨G, acc.2.1, acc.2.2.1, acc.2.2.2⟩, acc.2.1 + G.unitLen, acc.2.2.1 + G.rows,
        acc.2.2.2 + G.dCols P.openDegree)) (acc, z, r, c)).1.toList =
      acc.toList ++ placedFrom P l z r c
  | [], acc, _, _, _ => by simp [placedFrom]
  | G :: rest, acc, z, r, c => by
      rw [List.foldl_cons, placed_foldl P rest, Array.toList_push, placedFrom, List.append_assoc,
        List.singleton_append]

theorem placed_eq (P : SetupPlan) : P.placed = placedFrom P P.groups 0 0 0 := by
  unfold SetupPlan.placed
  rw [placed_foldl]
  rfl

theorem rowsBefore_zero {n : ℕ} (f : Fin (n + 1) → ℕ) : rowsBefore f 0 = 0 :=
  Finset.sum_eq_zero fun j _ => absurd j.isLt (Nat.not_lt_zero _)

theorem rowsBefore_succ {n : ℕ} (f : Fin (n + 1) → ℕ) (k : Fin n) :
    rowsBefore f k.succ = f 0 + rowsBefore (fun j => f j.succ) k := by
  show ∑ j : Fin (k + 1), f (Fin.castLE k.succ.2.le j) = _
  rw [Fin.sum_univ_succ]
  exact congrArg₂ (· + ·) (congrArg f (Fin.ext rfl))
    (Finset.sum_congr rfl fun j _ => congrArg f (Fin.ext rfl))

/-- The sum over the placed groups `f 0, …, f (n − 1)`: each group starts at the cells, rows and
columns of `D` of the groups before it. -/
theorem placedFrom_ofFn (P : SetupPlan) {M : Type} [AddCommMonoid M] (F : Placed → M) :
    ∀ {n : ℕ} (f : Fin n → SetupGroup) (z r c : ℕ),
      ((placedFrom P (List.ofFn f) z r c).map F).sum =
        ∑ k : Fin n, F ⟨f k, z + rowsBefore (fun j => (f j).unitLen) k,
          r + rowsBefore (fun j => (f j).rows) k,
          c + rowsBefore (fun j => (f j).dCols P.openDegree) k⟩
  | 0, f, z, r, c => by simp [placedFrom]
  | n + 1, f, z, r, c => by
      rw [List.ofFn_succ, placedFrom, List.map_cons, List.sum_cons,
        placedFrom_ofFn P F (fun i => f i.succ), Fin.sum_univ_succ]
      simp only [rowsBefore_zero, rowsBefore_succ, Nat.add_zero, Nat.add_assoc]

section In

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

theorem Ext.neg_into (z : Ext) : ((-z).into : E) = -z.into := by
  have value := Ext.into_sub (E := E) Ext.zero z
  rw [Ext.into_zero, zero_sub] at value
  rw [← value]
  simp only [Ext.into, Ext.toE_sub, Ext.toE_neg, Ext.toE_zero, zero_sub]

theorem sumBelow_into (n : ℕ) (f : ℕ → Ext) :
    ((sumBelow n f).into : E) = ∑ i ∈ Finset.range n, (f i).into := by
  induction n with
  | zero => simp [sumBelow, Ext.into_zero]
  | succ n ih =>
      simp only [sumBelow, Nat.fold_succ] at ih ⊢
      rw [Ext.into_add, ih, Finset.sum_range_succ]

theorem range_getD {α : Type} (n : ℕ) (f : ℕ → α) (d : α) (i : ℕ) :
    ((Array.range n).map f).getD i d = if i < n then f i else d := by
  unfold Array.getD
  split_ifs with inside small small
  · simp
  · simp at inside
    omega
  · simp at inside
    omega
  · rfl

theorem sum_range_if {M : Type} [AddCommMonoid M] (n : ℕ) (f g : ℕ → M) :
    ∑ c ∈ Finset.range n, (if c < n then f c else g c) = ∑ c ∈ Finset.range n, f c :=
  Finset.sum_congr rfl fun _ member => ite_eq_left (Finset.mem_range.mp member)

theorem Side.of_ratio (b : Ext) (r : List Ext) (ratio : ℕ) : (Side.of b r ratio).ratio = ratio :=
  rfl

theorem Side.of_proj (b : Ext) (r : List Ext) (ratio : ℕ) :
    (Side.of b r ratio).proj = powerScaleExt b (r.take (Nat.log 2 ratio)) :=
  rfl

theorem Side.of_high (b : Ext) (r : List Ext) (ratio : ℕ) :
    (Side.of b r ratio).high = EqSplit.of (r.drop (Nat.log 2 ratio)) :=
  rfl

/-- `Side.lane` in `E`: the factored form for a cell `ℓ` at a multiple of the ratio, and the sum
`Σ_{m<ratio} b^m·ẽq(r, ℓ + m)` otherwise. -/
noncomputable def laneIn (r : List E) (b : E) (ratio ℓ : ℕ) : E :=
  if ℓ % ratio = 0 then
    powerScale b (r.take (Nat.log 2 ratio)) * eqAt (r.drop (Nat.log 2 ratio)) (ℓ / ratio)
  else ∑ m ∈ Finset.range ratio, b ^ m * eqAt r (ℓ + m)

theorem lane_into (b : Ext) (r : List Ext) (ratio ℓ : ℕ) :
    (((Side.of b r ratio).lane b r ℓ).into : E) = laneIn (r.map Ext.into) b.into ratio ℓ := by
  show ((if ℓ % ratio = 0 then powerScaleExt b (r.take (Nat.log 2 ratio)) *
      (EqSplit.of (r.drop (Nat.log 2 ratio))).eval (ℓ / ratio)
    else sumBelow ratio fun m => powExt b m * eqIndex r (ℓ + m)).into : E) = _
  unfold laneIn
  split_ifs
  · rw [Ext.into_mul, powerScaleExt_into, EqSplit.eval_into, List.map_take, List.map_drop]
  · rw [sumBelow_into]
    exact Finset.sum_congr rfl fun m _ => by
      rw [Ext.into_mul, map_powExt degree2_extHom, eqIndex_into]

/-- The part of `A` of `groupWeight` in `E`, for the batching point `τ`, the coordinates `ρ` of
the setup point after the first `log_2 base` and the coordinates `r` of the relation point after
the first `log_2 block`. -/
noncomputable def aIn (P : SetupPlan) (τ ρ r : List E) (α : E) (p : Placed) : E :=
  powerScale (α ^ P.base) (ρ.take (Nat.log 2 (p.G.innerDegree / P.base))) *
    ∑ i ∈ Finset.range p.G.innerRank, eqAt τ (p.row + 1 + i) *
      ∑ c ∈ Finset.range p.G.zCols,
        (∑ f ∈ Finset.range p.G.foldDigits, -(((2 ^ P.logBasis : ℕ) : E) ^ f *
          laneIn r (α ^ P.block) (p.G.innerDegree / P.block)
            (p.z / P.block + (c * p.G.foldDigits + f) * (p.G.innerDegree / P.block)))) *
        eqAt (ρ.drop (Nat.log 2 (p.G.innerDegree / P.base))) (i * p.G.zCols + c)

/-- The part of `B` of `groupWeight` in `E`. -/
noncomputable def bIn (P : SetupPlan) (τ ρ r : List E) (α : E) (p : Placed) : E :=
  powerScale (α ^ P.base) (ρ.take (Nat.log 2 (p.G.outerDegree / P.base))) *
    ∑ j ∈ Finset.range p.G.outerRank, ∑ x ∈ Finset.range p.G.physWidth,
      (∑ s ∈ Finset.range p.G.slices, if x < p.G.sliceLen s * p.G.blockWidth then
        eqAt τ (p.row + 1 + p.G.innerRank + s * p.G.outerRank + j) *
          (if p.G.sliceStart s * p.G.blockWidth + x < p.G.blocks * p.G.blockWidth then
            laneIn r (α ^ P.block) (p.G.outerDegree / P.block)
              ((p.z + p.G.zLen + p.G.eLen) / P.block +
                (p.G.sliceStart s * p.G.blockWidth + x) * (p.G.outerDegree / P.block))
          else 0)
        else 0) *
      eqAt (ρ.drop (Nat.log 2 (p.G.outerDegree / P.base))) (j * p.G.physWidth + x)

/-- The part of `D` of `groupWeight` in `E`. -/
noncomputable def dIn (P : SetupPlan) (τ ρ r : List E) (α : E) (p : Placed) : E :=
  powerScale (α ^ P.base) (ρ.take (Nat.log 2 (P.openDegree / P.base))) *
    ∑ j ∈ Finset.range P.openRank, eqAt τ (P.openRow + j) *
      ∑ L ∈ Finset.range (p.G.dCols P.openDegree),
        laneIn r (α ^ P.block) (P.openDegree / P.block)
          ((p.z + p.G.zLen) / P.block + L * (P.openDegree / P.block)) *
        eqAt (ρ.drop (Nat.log 2 (P.openDegree / P.base))) (j * P.dColsAll + p.col + L)

/-- `groupWeight` in `E`: the parts of `A`, `B` and `D`. -/
noncomputable def groupIn (P : SetupPlan) (τ ρ r : List E) (α : E) (p : Placed) : E :=
  aIn P τ ρ r α p + bIn P τ ρ r α p + dIn P τ ρ r α p

theorem ofNat_into (m : ℕ) : ((Ext.ofBase (Fq.ofNat m)).into : E) = (m : E) := by
  rw [Ext.into_ofBase, Fq.toZq_ofNat, map_natCast]

theorem groupWeight_into (P : SetupPlan) (τ ρ r : List Ext) (α : Ext) (p : Placed) :
    ((groupWeight P (eqIndex τ) (Side.of (powExt α P.block) r) (Side.of (powExt α P.base) ρ)
        (powExt α P.block) r (fun f => powExt (Ext.ofBase (Fq.ofNat (2 ^ P.logBasis))) f)
        p).into : E) =
      groupIn P (τ.map Ext.into) (ρ.map Ext.into) (r.map Ext.into) α.into p := by
  unfold groupWeight groupIn aIn bIn dIn
  simp only [Ext.into_add, Ext.into_mul]
  refine congrArg₂ (· + ·) (congrArg₂ (· + ·) ?_ ?_) ?_
  · simp only [Side.of_ratio, Side.of_proj, Side.of_high, powerScaleExt_into,
      map_powExt degree2_extHom, sumBelow_into, Ext.into_mul, eqIndex_into, range_getD,
      apply_ite Ext.into, Ext.into_zero, ite_mul, zero_mul, sum_range_if, Ext.neg_into, lane_into,
      ofNat_into, EqSplit.eval_into, List.map_take, List.map_drop, Nat.cast_pow]
  · simp only [Side.of_ratio, Side.of_proj, Side.of_high, powerScaleExt_into,
      map_powExt degree2_extHom, sumBelow_into, Ext.into_mul, EqSplit.eval_into, List.map_take,
      List.map_drop]
    refine congrArg _ (Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun x _ => ?_)
    refine congrArg (· * _) (Finset.sum_congr rfl fun s member => ?_)
    have small := Finset.mem_range.mp member
    simp only [range_getD, ite_eq_left small, apply_ite Ext.into, Ext.into_zero, Ext.into_mul,
      eqIndex_into, lane_into, map_powExt degree2_extHom, Nat.add_assoc]
  · simp only [Side.of_ratio, Side.of_proj, Side.of_high, powerScaleExt_into,
      map_powExt degree2_extHom, sumBelow_into, Ext.into_mul, eqIndex_into, range_getD,
      apply_ite Ext.into, Ext.into_zero, ite_mul, zero_mul, sum_range_if, lane_into,
      EqSplit.eval_into, List.map_take, List.map_drop]

/-- The weight of Stage 3 of the replay in `E`: `powerScale(α, ρ_low)` times the sum of
`groupIn` over the placed groups of the plan. -/
theorem stage3Weight_into (P : SetupPlan) (α : Ext) (τ r2 ρ : List Ext) :
    ((stage3Weight P α τ r2 ρ).into : E) =
      powerScale α.into ((ρ.take (Nat.log 2 P.base)).map Ext.into) *
        ((placedFrom P P.groups 0 0 0).map (groupIn P (τ.map Ext.into)
          ((ρ.drop (Nat.log 2 P.base)).map Ext.into)
          ((r2.drop (Nat.log 2 P.block)).map Ext.into) α.into)).sum := by
  rw [stage3Weight_eq, Ext.into_mul, powerScaleExt_into, ← placed_eq]
  congr 1
  suffices sum : ∀ (l : List Placed) (a : Ext), ((l.map fun p => groupWeight P (eqIndex τ)
      (Side.of (powExt α P.block) (r2.drop (Nat.log 2 P.block)))
      (Side.of (powExt α P.base) (ρ.drop (Nat.log 2 P.base))) (powExt α P.block)
      (r2.drop (Nat.log 2 P.block))
      (fun f => powExt (Ext.ofBase (Fq.ofNat (2 ^ P.logBasis))) f) p).foldl (· + ·) a).into =
        (a.into : E) + (l.map (groupIn P (τ.map Ext.into)
          ((ρ.drop (Nat.log 2 P.base)).map Ext.into)
          ((r2.drop (Nat.log 2 P.block)).map Ext.into) α.into)).sum by
    rw [sum, Ext.into_zero, zero_add]
  intro l
  induction l with
  | nil => simp
  | cons p rest ih =>
      intro a
      rw [List.map_cons, List.foldl_cons, ih, Ext.into_add, groupWeight_into, List.map_cons,
        List.sum_cons, add_assoc]

end In

end NightstreamFPrime.Spec.Compression.Akita.Replay
