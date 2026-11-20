import NightstreamFPrime.Spec.Compression.Akita.Extract.Bound
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tier

/-! The stage errors, the retry loops and the tree size of one level of the opening suffix of
Akita over the later stages (ePrint 2026/1983, revision of 2026-10-15, Proposition 10.44 (iv):
the tree size and the sum (206) of the stage errors).

- `Tier.over_chargeFrom` states that the stage errors from the first stage of the level sum to
  `r·p + Σ_j ε_j` plus the sum of the later stages. `r` is the number of blocks, `p` is the largest
  probability of one fold challenge, and `ε_j` is the bound of draw `j` of the checks.
- `Tier.over_loopsFrom` and `Tier.over_treeFrom` add `r + 1` retry loops and multiply the tree
  size by `2·(r + 1)`.
- `finish_chargeFrom`, `finish_loopsFrom` and `finish_treeFrom` are the values after the last
  stage. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Checkpoint

section Finish

variable {Point Coord Slot Proof : Type} (protocol : Protocol Point (Coord → Slot) Proof)

theorem finish_chargeFrom (level : ℕ) : (finish protocol).chargeFrom level = 0 := by
  refine Finset.sum_eq_zero fun later _ => ?_
  unfold Structure.charge
  split <;> rfl

theorem finish_loopsFrom (level : ℕ) : (finish protocol).loopsFrom level = 0 :=
  Finset.sum_eq_zero fun _ _ => rfl

theorem finish_treeFrom (level : ℕ) : (finish protocol).treeFrom level = 1 :=
  Finset.prod_eq_one fun _ _ => rfl

end Finish

namespace Tier

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}
  {E Wire Coord Slot Result Source : Type} [CommRing E] [Module Zq E]
  (T : Tier O fold E Wire Coord Slot Result Source) {total : ℕ}
  [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT WitT : Type} {accept : Prefix Wire (Coord → Slot) total → Prop}
  {tail : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT WitT}
  {source : Inner → Source} {clash : Inner → Inner → BreakT}

theorem over_kind_start :
    (T.over accept tail source clash).kind T.start =
      .indexed ((List.finRange S.blocks).map T.block) := by
  have ends := T.finish_eq
  rw [over_kind, ite_eq_left (by omega)]
  unfold Tier.kind
  rw [ite_eq_left le_rfl]

theorem over_kind_check {index : ℕ} (inside : index < T.costs.length) :
    (T.over accept tail source clash).kind (T.start + (index + 1)) =
      if index = 0 then .checkpoint (T.costs.length - 1) else .plain := by
  have ends := T.finish_eq
  rw [over_kind, ite_eq_left (by omega)]
  unfold Tier.kind
  rw [ite_eq_right (by omega)]
  by_cases first : index = 0
  · rw [ite_eq_left first, ite_eq_left (by omega)]
  · rw [ite_eq_right first, ite_eq_right (by omega)]

theorem over_kind_tail {level : ℕ} (outside : T.finish ≤ level) :
    (T.over accept tail source clash).kind level = tail.kind level := by
  rw [over_kind, ite_eq_right (by omega)]

theorem over_charge_start (fits : T.finish ≤ total) :
    (T.over accept tail source clash).charge T.start = S.blocks * T.chance := by
  have ends := T.finish_eq
  have weights : (T.over accept tail source clash).weight T.start ∘ T.block =
      fun _ => T.chance :=
    funext fun i => by rw [Function.comp, over_weight, ite_eq_left (by omega)]
  unfold Structure.charge
  rw [ite_eq_left (show T.start < (T.over accept tail source clash).protocol.stages by
    show T.start < total
    omega), T.over_kind_start]
  show (((List.finRange S.blocks).map T.block).map
    ((T.over accept tail source clash).weight T.start)).sum = _
  rw [List.map_map, weights]
  simp

theorem over_charge_check (fits : T.finish ≤ total) {index : ℕ}
    (inside : index < T.costs.length) :
    (T.over accept tail source clash).charge (T.start + (index + 1)) =
      T.costs.getD index 0 := by
  have ends := T.finish_eq
  have error : (T.over accept tail source clash).error (T.start + (index + 1)) =
      T.costs.getD index 0 := by
    rw [over_error, ite_eq_left (by omega),
      show T.start + (index + 1) - T.start - 1 = index by omega]
  unfold Structure.charge
  rw [ite_eq_left (show T.start + (index + 1) <
      (T.over accept tail source clash).protocol.stages by
    show T.start + (index + 1) < total
    omega), T.over_kind_check inside]
  by_cases first : index = 0
  · rw [ite_eq_left first]
    exact error
  · rw [ite_eq_right first]
    exact error

theorem over_charge_tail (stages : tail.protocol.stages = total) {level : ℕ}
    (outside : T.finish ≤ level) :
    (T.over accept tail source clash).charge level = tail.charge level := by
  have weight : (T.over accept tail source clash).weight level = tail.weight level :=
    funext fun coordinate => by rw [over_weight, ite_eq_right (by omega)]
  have error : (T.over accept tail source clash).error level = tail.error level := by
    rw [over_error, ite_eq_right (by omega)]
  unfold Structure.charge
  rw [T.over_kind_tail outside, weight, error, stages]
  rfl

theorem over_loops_start :
    (T.over accept tail source clash).loops T.start = S.blocks := by
  unfold Structure.loops
  rw [T.over_kind_start]
  show ((List.finRange S.blocks).map T.block).length = _
  simp

theorem over_loops_check {index : ℕ} (inside : index < T.costs.length) :
    (T.over accept tail source clash).loops (T.start + (index + 1)) =
      if index = 0 then 1 else 0 := by
  unfold Structure.loops
  rw [T.over_kind_check inside]
  by_cases first : index = 0
  · rw [ite_eq_left first, ite_eq_left first]
  · rw [ite_eq_right first, ite_eq_right first]

theorem over_loops_tail {level : ℕ} (outside : T.finish ≤ level) :
    (T.over accept tail source clash).loops level = tail.loops level := by
  unfold Structure.loops
  rw [T.over_kind_tail outside]

theorem over_arity_start :
    (T.over accept tail source clash).arity T.start = 1 + S.blocks := by
  unfold Structure.arity
  rw [T.over_kind_start]
  show 1 + ((List.finRange S.blocks).map T.block).length = _
  simp

theorem over_arity_check {index : ℕ} (inside : index < T.costs.length) :
    (T.over accept tail source clash).arity (T.start + (index + 1)) =
      if index = 0 then 2 else 1 := by
  unfold Structure.arity
  rw [T.over_kind_check inside]
  by_cases first : index = 0
  · rw [ite_eq_left first, ite_eq_left first]
  · rw [ite_eq_right first, ite_eq_right first]

theorem over_arity_tail {level : ℕ} (outside : T.finish ≤ level) :
    (T.over accept tail source clash).arity level = tail.arity level := by
  unfold Structure.arity
  rw [T.over_kind_tail outside]

/-- The stage errors from the first stage of the level: the weights of the fold challenges of
the blocks, the bounds of the draws of the checks, and the stage errors of the later stages
(the sum of Akita (206) for the stages from one level). -/
theorem over_chargeFrom (fits : T.finish ≤ total)
    (stages : tail.protocol.stages = total) :
    (T.over accept tail source clash).chargeFrom T.start =
      S.blocks * T.chance + T.costs.sum + tail.chargeFrom T.finish := by
  have ends := T.finish_eq
  unfold Structure.chargeFrom
  rw [stages]
  show ∑ later ∈ Finset.Ico T.start total, (T.over accept tail source clash).charge later = _
  rw [← Finset.sum_Ico_consecutive _ (show T.start ≤ T.finish by omega) fits]
  refine congrArg₂ (· + ·) ?_
    (Finset.sum_congr rfl fun level member =>
      T.over_charge_tail stages (Finset.mem_Ico.mp member).1)
  rw [Finset.sum_Ico_eq_sum_range, show T.finish - T.start = T.costs.length + 1 by omega,
    Finset.sum_range_succ', Nat.add_zero, T.over_charge_start fits, add_comm, ← sum_range_getD]
  exact congrArg _ (Finset.sum_congr rfl fun index member =>
    T.over_charge_check fits (Finset.mem_range.mp member))

/-- The extractor has one retry loop for each block and one for the checkpoint of the level,
and the loops of the later stages. -/
theorem over_loopsFrom (draws : T.costs ≠ []) (fits : T.finish ≤ total)
    (stages : tail.protocol.stages = total) :
    (T.over accept tail source clash).loopsFrom T.start =
      S.blocks + 1 + tail.loopsFrom T.finish := by
  have ends := T.finish_eq
  have long : 0 < T.costs.length := List.length_pos_iff.mpr draws
  unfold Structure.loopsFrom
  rw [stages]
  show ∑ later ∈ Finset.Ico T.start total, (T.over accept tail source clash).loops later = _
  rw [← Finset.sum_Ico_consecutive _ (show T.start ≤ T.finish by omega) fits]
  refine congrArg₂ (· + ·) ?_
    (Finset.sum_congr rfl fun level member => T.over_loops_tail (Finset.mem_Ico.mp member).1)
  rw [Finset.sum_Ico_eq_sum_range, show T.finish - T.start = T.costs.length + 1 by omega,
    Finset.sum_range_succ', Nat.add_zero, T.over_loops_start,
    Finset.sum_congr rfl fun index member => T.over_loops_check (Finset.mem_range.mp member),
    Finset.sum_ite_eq', ite_eq_left (Finset.mem_range.mpr long), add_comm]

/-- The tree size from the first stage of the level: `1 + r` children at the fold challenge of
`r` blocks, two at the checkpoint, and the tree size of the later stages. -/
theorem over_treeFrom (draws : T.costs ≠ []) (fits : T.finish ≤ total)
    (stages : tail.protocol.stages = total) :
    (T.over accept tail source clash).treeFrom T.start =
      2 * (S.blocks + 1) * tail.treeFrom T.finish := by
  have ends := T.finish_eq
  have long : 0 < T.costs.length := List.length_pos_iff.mpr draws
  unfold Structure.treeFrom
  rw [stages]
  show ∏ later ∈ Finset.Ico T.start total, (T.over accept tail source clash).arity later = _
  rw [← Finset.prod_Ico_consecutive _ (show T.start ≤ T.finish by omega) fits]
  refine congrArg₂ (· * ·) ?_
    (Finset.prod_congr rfl fun level member => T.over_arity_tail (Finset.mem_Ico.mp member).1)
  rw [Finset.prod_Ico_eq_prod_range, show T.finish - T.start = T.costs.length + 1 by omega,
    Finset.prod_range_succ', Nat.add_zero, T.over_arity_start,
    Finset.prod_congr rfl fun index member => T.over_arity_check (Finset.mem_range.mp member),
    Finset.prod_ite_eq', ite_eq_left (Finset.mem_range.mpr long)]
  ring

end Tier

end NightstreamFPrime.Spec.Compression.Akita.Extract
