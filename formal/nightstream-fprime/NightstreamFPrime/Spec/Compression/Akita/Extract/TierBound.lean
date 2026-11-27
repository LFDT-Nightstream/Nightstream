import NightstreamFPrime.Spec.Checkpoint.Extraction
import NightstreamFPrime.Spec.Checkpoint.Runs
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tier

/-! The stage errors, the retry loops and the tree size of one level of the opening suffix of
Akita over the later stages (ePrint 2026/1983, revision of 2026-11-23, Proposition 10.44 (iv):
the tree size and the sum (206) of the stage errors).

- `Tier.fork_chargeFrom`, `Tier.fork_loopsFrom` and `Tier.fork_treeFrom` are the values of the
  indexed stage of the fold challenges: the sum of the weights `p_b` of the fold coordinates, one
  retry loop for each coordinate, and the tree factor `1 + r` for `r` coordinates.
- `Tier.over_chargeFrom` states that the stage errors from the first stage of the level sum to
  `Tier.levelError = 2^{−g}·Σ_b p_b + Σ_j 2^{−g_j}·ε_j` plus the sum of the later stages. `p_b` is
  the largest probability of one fold challenge of the coordinate `b`, `ε_j` is the bound of draw
  `j` of the checks, and `g` and `g_j` are their work bits (Remark 10.18).
- `Tier.over_loopsFrom` and `Tier.over_treeFrom` add `r + 1` retry loops and multiply the tree
  size by `2·(r + 1)`.
- `finish_chargeFrom`, `finish_loopsFrom` and `finish_treeFrom` are the values after the last
  stage. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Checkpoint

section Finish

variable {Addr Point Coord Slot Proof : Type} (protocol : Protocol Addr Point (Coord → Slot) Proof)

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

variable {Block : Type} {Ring : Block → Type}
  {Stmt Brk Wit Weak Wire Coord Slot Result Source : Type} [∀ b, Zero (Ring b)]
  (T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source) {total : ℕ}
  [Fintype Block] [Fintype Slot] [Nonempty Slot]

section Fork

variable {Inner BreakM CandM KeyM WitM : Type} {accept : Prefix Wire (Option Coord → Slot) total →
  Prop}
  {mid : Structure (Address Wire total) (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire) Inner
    BreakM CandM
    KeyM WitM}
  {source : Inner → Source}

theorem fork_charge_start (injective : Function.Injective T.block) (fits : T.start < total) :
    (T.fork accept mid source).charge T.start = 1 / 2 ^ T.foldBits * ∑ b, T.chance b := by
  unfold Structure.charge
  rw [ite_eq_left (show T.start < (T.fork accept mid source).protocol.stages from fits),
    T.fork_kind_start]
  show (T.fork accept mid source).chance T.start *
    ((Finset.univ.toList.map T.block).map ((T.fork accept mid source).weight T.start)).sum = _
  rw [show (T.fork accept mid source).chance T.start = 1 / 2 ^ T.foldBits from ite_eq_left le_rfl,
    List.map_map, show (T.fork accept mid source).weight T.start ∘ T.block = T.chance from
    funext fun b => T.fork_weight_block injective b, Finset.sum_map_toList]

theorem fork_charge_above (stages : mid.protocol.stages = total) {level : ℕ}
    (above : T.start < level) :
    (T.fork accept mid source).charge level = mid.charge level := by
  have weight : (T.fork accept mid source).weight level = mid.weight level :=
    funext fun coordinate => T.fork_weight_above above coordinate
  have chance : (T.fork accept mid source).chance level = mid.chance level :=
    ite_eq_right (Nat.not_le.mpr above)
  unfold Structure.charge
  rw [T.fork_kind_above above, weight, chance, stages]
  rfl

/-- The stage errors from the stage of the fold challenges: the weights of the fold coordinates
times `2^{−g}` for the work bits `g` of the fold (Akita Remark 10.18 for the indexed stage), and
the stage errors of the later stages. -/
theorem fork_chargeFrom (injective : Function.Injective T.block) (fits : T.start < total)
    (stages : mid.protocol.stages = total) :
    (T.fork accept mid source).chargeFrom T.start =
      1 / 2 ^ T.foldBits * ∑ b, T.chance b + mid.chargeFrom (T.start + 1) := by
  unfold Structure.chargeFrom
  rw [stages]
  show ∑ later ∈ Finset.Ico T.start total, (T.fork accept mid source).charge later = _
  rw [Finset.sum_eq_sum_Ico_succ_bot fits, T.fork_charge_start injective fits]
  exact congrArg _ (Finset.sum_congr rfl fun level member =>
    T.fork_charge_above stages (Finset.mem_Ico.mp member).1)

/-- The extractor has one retry loop for each fold coordinate, and the loops of the later
stages. -/
theorem fork_loopsFrom (fits : T.start < total) (stages : mid.protocol.stages = total) :
    (T.fork accept mid source).loopsFrom T.start =
      Fintype.card Block + mid.loopsFrom (T.start + 1) := by
  have first : (T.fork accept mid source).loops T.start = Fintype.card Block := by
    unfold Structure.loops
    rw [T.fork_kind_start]
    show (Finset.univ.toList.map T.block).length = _
    rw [List.length_map, Finset.length_toList, Finset.card_univ]
  unfold Structure.loopsFrom
  rw [stages]
  show ∑ later ∈ Finset.Ico T.start total, (T.fork accept mid source).loops later = _
  rw [Finset.sum_eq_sum_Ico_succ_bot fits, first]
  exact congrArg _ (Finset.sum_congr rfl fun level member => by
    unfold Structure.loops
    rw [T.fork_kind_above (Finset.mem_Ico.mp member).1])

/-- The tree size from the stage of the fold challenges: `1 + r` children for `r` fold
coordinates, and the tree size of the later stages. -/
theorem fork_treeFrom (fits : T.start < total) (stages : mid.protocol.stages = total) :
    (T.fork accept mid source).treeFrom T.start =
      (1 + Fintype.card Block) * mid.treeFrom (T.start + 1) := by
  have first : (T.fork accept mid source).arity T.start = 1 + Fintype.card Block := by
    unfold Structure.arity
    rw [T.fork_kind_start]
    show 1 + (Finset.univ.toList.map T.block).length = _
    rw [List.length_map, Finset.length_toList, Finset.card_univ]
  unfold Structure.treeFrom
  rw [stages]
  show ∏ later ∈ Finset.Ico T.start total, (T.fork accept mid source).arity later = _
  rw [Finset.prod_eq_prod_Ico_succ_bot fits, first]
  exact congrArg _ (Finset.prod_congr rfl fun level member => by
    unfold Structure.arity
    rw [T.fork_kind_above (Finset.mem_Ico.mp member).1])

end Fork

variable [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT WitT : Type} {accept : Prefix Wire (Option Coord → Slot) total → Prop}
  {tail : Structure (Address Wire total) (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire) Inner
    BreakT CandT
    KeyT WitT}
  {source : Inner → Source} {clash : Inner → Inner → BreakT}

omit [Fintype Coord] [DecidableEq Coord] in
/-- The stage errors of one level: the weights of its fold coordinates and the bounds of the
draws of its checks, each times `2^{−g}` for its work bits `g` (Akita Remark 10.18). -/
noncomputable def levelError : ℝ :=
  1 / 2 ^ T.foldBits * ∑ b, T.chance b + (credit T.costs T.bits).sum

/-- The stage errors from the first stage of the level: the weights of the fold challenges of
the fold coordinates and the bounds of the draws of the checks, each times `2^{−g}` for its work
bits `g` (Akita Remark 10.18), and the stage errors of the later stages (the sum of Akita (206)
for the stages from one level). -/
theorem over_chargeFrom (injective : Function.Injective T.block) (fits : T.finish ≤ total)
    (stages : tail.protocol.stages = total) (aligned : T.bits.length = T.costs.length) :
    (T.over accept tail source clash).chargeFrom T.start =
      T.levelError + tail.chargeFrom T.finish := by
  have ends := T.finish_eq
  have checks := T.segment.over_chargeFrom (accept := accept) (source := source) (clash := clash)
    (relation := fun _ _ => True) fits stages aligned
  show (T.fork accept (T.segment.over accept tail source clash fun _ _ => True)
    source).chargeFrom T.start = _
  rw [T.fork_chargeFrom injective (by omega) rfl, levelError, add_assoc]
  exact congrArg _ checks

/-- The extractor has one retry loop for each fold coordinate and one for the checkpoint of the
level, and the loops of the later stages. -/
theorem over_loopsFrom (draws : T.costs ≠ []) (fits : T.finish ≤ total)
    (stages : tail.protocol.stages = total) :
    (T.over accept tail source clash).loopsFrom T.start =
      Fintype.card Block + 1 + tail.loopsFrom T.finish := by
  have ends := T.finish_eq
  have checks := T.segment.over_loopsFrom (accept := accept) (source := source) (clash := clash)
    (relation := fun _ _ => True) draws fits stages
  show (T.fork accept (T.segment.over accept tail source clash fun _ _ => True)
    source).loopsFrom T.start = _
  rw [T.fork_loopsFrom (by omega) rfl, add_assoc]
  exact congrArg _ checks

/-- The tree size from the first stage of the level: `1 + r` children at the fold challenge of
`r` fold coordinates, two at the checkpoint, and the tree size of the later stages. -/
theorem over_treeFrom (draws : T.costs ≠ []) (fits : T.finish ≤ total)
    (stages : tail.protocol.stages = total) :
    (T.over accept tail source clash).treeFrom T.start =
      2 * (Fintype.card Block + 1) * tail.treeFrom T.finish := by
  have ends := T.finish_eq
  have checks := T.segment.over_treeFrom (accept := accept) (source := source) (clash := clash)
    (relation := fun _ _ => True) draws fits stages
  show (T.fork accept (T.segment.over accept tail source clash fun _ _ => True)
    source).treeFrom T.start = _
  rw [T.fork_treeFrom (by omega) rfl, show (T.segment.over accept tail source clash
    fun _ _ => True).treeFrom (T.start + 1) = 2 * tail.treeFrom T.finish from checks]
  ring

end Tier

end NightstreamFPrime.Spec.Compression.Akita.Extract
