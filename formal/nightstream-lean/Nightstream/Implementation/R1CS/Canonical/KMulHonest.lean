import Nightstream.Implementation.R1CS.Canonical.KFrames

/-!
Contract: honest completeness for one `K` multiplication.

Owns: the witness that extends an assignment to a multiplication's three frame
columns, and the proof that it satisfies the emitted rows.

Does not own: soundness (`KMul`), the allocator (`KFrames`), or completeness of
a whole Horner evaluation.

## The two hypotheses, and why neither is decoration

**Freshness.** The witness writes the three frame columns. If an operand
combination mentioned one of them, writing would change that operand's value
and the row it is supposed to satisfy might then fail. So the operands must not
mention the frame — which is exactly what a disjoint allocator delivers, and
why `KFrames` had to come first.

**Distinctness.** The witness is an `if`-chain on the three columns. If two
coincided the chain would silently drop one product and write the other twice.
`KFrames.frameColumn_slot_disjoint` discharges this for the canonical
allocator.

Neither is assumed away: both are premises a real allocator constructs, which
is the test §3 applies to any new premise.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KMulHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul

/-- Assignments agreeing on every column a combination mentions give it the
same value. -/
theorem lcEval_congr (z z' : Nat → Nat) (comb : LinComb)
    (agree : ∀ column, Mentions comb column → z column = z' column) :
    lcEval z comb = lcEval z' comb := by
  rw [lcEval_eq_rawSum, lcEval_eq_rawSum]
  congr 1
  induction comb with
  | nil => rfl
  | cons term rest hypothesis =>
      have tailAgree : ∀ column, Mentions rest column → z column = z' column := by
        intro column mentioned
        refine agree column ?_
        simp only [Mentions, List.map_cons, List.mem_cons]
        exact Or.inr mentioned
      rw [rawSum_cons, rawSum_cons, agree term.1 (by simp [Mentions]),
        hypothesis tailAgree]

/-- A frame whose three columns are distinct. -/
structure Distinct (frame : Frame) : Prop where
  lowNeHigh : frame.lowLow ≠ frame.highHigh
  lowNeCross : frame.lowLow ≠ frame.cross
  highNeCross : frame.highHigh ≠ frame.cross

/-- The canonical allocator produces distinct frames, so `Distinct` is a
premise a real consumer constructs rather than an obligation moved. -/
theorem canonical_distinct (base step : Nat) :
    Distinct (KFrames.frameAt base step) where
  lowNeHigh := KFrames.frameColumn_slot_disjoint base step 0 1 (by decide)
  lowNeCross := KFrames.frameColumn_slot_disjoint base step 0 2 (by decide)
  highNeCross := KFrames.frameColumn_slot_disjoint base step 1 2 (by decide)

/-- An operand combination that mentions no frame column. -/
def Fresh (comb : LinComb) (frame : Frame) : Prop :=
  ¬ Mentions comb frame.lowLow ∧ ¬ Mentions comb frame.highHigh
    ∧ ¬ Mentions comb frame.cross

/-- **The witness.**  Writes each product to its frame column, leaving every
other column alone. -/
def witness (z : Nat → Nat) (left right : Carried) (frame : Frame) :
    Nat → Nat :=
  fun column =>
    if column = frame.lowLow then
      lcEval z left.low * lcEval z right.low % goldilocksP
    else if column = frame.highHigh then
      lcEval z left.high * lcEval z right.high % goldilocksP
    else if column = frame.cross then
      lcEval z (sumComb left) * lcEval z (sumComb right) % goldilocksP
    else z column

theorem witness_off_frame
    (z : Nat → Nat) (left right : Carried) (frame : Frame) (column : Nat)
    (notLow : column ≠ frame.lowLow) (notHigh : column ≠ frame.highHigh)
    (notCross : column ≠ frame.cross) :
    witness z left right frame column = z column := by
  unfold witness
  rw [if_neg notLow, if_neg notHigh, if_neg notCross]

/-- A canonical multiplication witness leaves the completed prefix before
its own frame unchanged. -/
theorem witness_off_before
    (z : Nat → Nat) (left right : Carried) (base step column : Nat)
    (below : column < base + 3 * step) :
    witness z left right (KFrames.frameAt base step) column = z column := by
  apply witness_off_frame
  all_goals
    simp only [KFrames.frameAt, KFrames.frameColumn,
      KFrames.columnsPerFrame]
    omega

/-- A multiplication witness preserves canonical representatives.  Every
newly written product is reduced modulo the Goldilocks modulus; every other
coordinate comes from the prior canonical assignment. -/
theorem witness_residues
    (z : Nat → Nat) (left right : Carried) (frame : Frame)
    (residues : ∀ column, z column < goldilocksP) :
    ∀ column, witness z left right frame column < goldilocksP := by
  intro column
  unfold witness
  split
  · exact Nat.mod_lt _ (by decide)
  · split
    · exact Nat.mod_lt _ (by decide)
    · split
      · exact Nat.mod_lt _ (by decide)
      · exact residues column

/-- Any combination confined to the completed prefix is fresh for the next
canonical multiplication frame. -/
theorem fresh_of_before
    (comb : LinComb) (base step : Nat)
    (below :
      ∀ column, Mentions comb column → column < base + 3 * step) :
    Fresh comb (KFrames.frameAt base step) := by
  refine ⟨?_, ?_, ?_⟩ <;> intro mentioned
  all_goals
    exact absurd (below _ mentioned) (by
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame]
      omega)

/-- A fresh combination keeps its value under the witness. -/
theorem witness_preserves
    (z : Nat → Nat) (left right : Carried) (frame : Frame) (comb : LinComb)
    (fresh : Fresh comb frame) :
    lcEval (witness z left right frame) comb = lcEval z comb := by
  refine (lcEval_congr z _ comb (fun column mentioned => ?_)).symm
  refine (witness_off_frame z left right frame column ?_ ?_ ?_).symm
  · exact fun equal => fresh.1 (equal ▸ mentioned)
  · exact fun equal => fresh.2.1 (equal ▸ mentioned)
  · exact fun equal => fresh.2.2 (equal ▸ mentioned)

theorem witness_lowLow
    (z : Nat → Nat) (left right : Carried) (frame : Frame) :
    witness z left right frame frame.lowLow
      = lcEval z left.low * lcEval z right.low % goldilocksP := by
  unfold witness; rw [if_pos rfl]

theorem witness_highHigh
    (z : Nat → Nat) (left right : Carried) (frame : Frame)
    (distinct : Distinct frame) :
    witness z left right frame frame.highHigh
      = lcEval z left.high * lcEval z right.high % goldilocksP := by
  unfold witness
  rw [if_neg (Ne.symm distinct.lowNeHigh), if_pos rfl]

theorem witness_cross
    (z : Nat → Nat) (left right : Carried) (frame : Frame)
    (distinct : Distinct frame) :
    witness z left right frame frame.cross
      = lcEval z (sumComb left) * lcEval z (sumComb right) % goldilocksP := by
  unfold witness
  rw [if_neg (Ne.symm distinct.lowNeCross), if_neg (Ne.symm distinct.highNeCross),
    if_pos rfl]

/-! ## Completeness -/

theorem productRow_holds
    (z : Nat → Nat) (left right : LinComb) (target : Nat)
    (value : z target % goldilocksP
      = lcEval z left * lcEval z right % goldilocksP) :
    RowHolds z (productRow left right target) := by
  show lcEval z left * lcEval z right % goldilocksP = lcEval z [(target, 1)]
  rw [KMul.lcEval_singleton_col, value]

/-- **An honest execution satisfies the multiplication.**  The witness writes
each schoolbook-or-Karatsuba product to its column, and the operands keep their
values because they are fresh for the frame. -/
theorem witness_satisfies
    (z : Nat → Nat) (left right : Carried) (frame : Frame)
    (distinct : Distinct frame)
    (leftLowFresh : Fresh left.low frame) (leftHighFresh : Fresh left.high frame)
    (rightLowFresh : Fresh right.low frame)
    (rightHighFresh : Fresh right.high frame) :
    Satisfies (rows left right frame) (witness z left right frame) := by
  have sumLeft : Fresh (sumComb left) frame :=
    ⟨fun m => (List.mem_append.1 (by simpa [Mentions, sumComb] using m)).elim
        leftLowFresh.1 leftHighFresh.1,
     fun m => (List.mem_append.1 (by simpa [Mentions, sumComb] using m)).elim
        leftLowFresh.2.1 leftHighFresh.2.1,
     fun m => (List.mem_append.1 (by simpa [Mentions, sumComb] using m)).elim
        leftLowFresh.2.2 leftHighFresh.2.2⟩
  have sumRight : Fresh (sumComb right) frame :=
    ⟨fun m => (List.mem_append.1 (by simpa [Mentions, sumComb] using m)).elim
        rightLowFresh.1 rightHighFresh.1,
     fun m => (List.mem_append.1 (by simpa [Mentions, sumComb] using m)).elim
        rightLowFresh.2.1 rightHighFresh.2.1,
     fun m => (List.mem_append.1 (by simpa [Mentions, sumComb] using m)).elim
        rightLowFresh.2.2 rightHighFresh.2.2⟩
  intro row member
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · refine productRow_holds _ _ _ _ ?_
    rw [witness_lowLow, witness_preserves _ _ _ _ _ leftLowFresh,
      witness_preserves _ _ _ _ _ rightLowFresh, Nat.mod_mod]
  · refine productRow_holds _ _ _ _ ?_
    rw [witness_highHigh _ _ _ _ distinct,
      witness_preserves _ _ _ _ _ leftHighFresh,
      witness_preserves _ _ _ _ _ rightHighFresh, Nat.mod_mod]
  · refine productRow_holds _ _ _ _ ?_
    rw [witness_cross _ _ _ _ distinct, witness_preserves _ _ _ _ _ sumLeft,
      witness_preserves _ _ _ _ _ sumRight, Nat.mod_mod]

end Nightstream.Implementation.R1CS.Canonical.KMulHonest
