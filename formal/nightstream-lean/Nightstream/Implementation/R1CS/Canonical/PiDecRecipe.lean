import Nightstream.Implementation.R1CS.Canonical.KRecomposition
import Nightstream.Implementation.R1CS.Canonical.KLowNormBatch
import Nightstream.Implementation.R1CS.Canonical.KZeroCheck
import Nightstream.Implementation.R1CS.Canonical.KConsistency

/-!
Contract: Π_DEC as one emitted row program.

Owns: the assembly of the decomposition's atoms, its folded row count, its
allocation, soundness to each named check, a single honest witness, and the
folded `Typed.Cost`.

Does not own: any of the atoms.  `KRecomposition` owns the radix-`b` relation,
`KLowNormBatch` the digit range, `KZeroCheck` the zeroing, `KConsistency` the
pairwise agreement.  This module owns only how they compose.

## What the assembly has to get right

Every atom but one is **allocation-free**: recompositions, zero checks, padding
and consistency read carried values and write nothing.  `KLowNormBatch` is the
sole allocator, one column per digit.

That asymmetry is the whole content of the composition.  The honest witness for
the assembled program is the low-norm batch's witness — there is nothing else to
extend — and every other part must therefore still hold *under that extension*.
It does, but only if no other part reads an allocated column, which is `Fresh`
below and is a hypothesis rather than an assumption.

## Not the whole of `verify_dec_public`

This is the decomposition algebra.  Two obligations from the same verifier are
outside it and stay named rather than silently absorbed:

- `combine_b_pows(children.c, b) = parent.c` — the Ajtai commitment fold, which
  is not arithmetic this layer owns;
- the parent's old-point `y_zcol`, which `verify_dec_public` does **not**
  validate.  Rust's own comment says so, and says the delayed-projection
  authority bridge must close it.

`PIDEC-RECOMPOSITION` records both.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.PiDecRecipe

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KLowNormBatch

/-! ## Freshness against the sole allocator -/

/-- A combination reads no column the low-norm batch allocates. -/
def FreshComb (digits : List Digit) (comb : LinComb) : Prop :=
  ∀ digit ∈ digits, ¬ Mentions comb digit.squareColumn

/-- Both coordinates of a carried value are fresh. -/
def FreshCarried (digits : List Digit) (value : Carried) : Prop :=
  FreshComb digits value.low ∧ FreshComb digits value.high

theorem lcEval_fresh
    (z : Nat → Nat) (digits : List Digit) (comb : LinComb)
    (fresh : FreshComb digits comb) :
    lcEval (batchWitness z digits) comb = lcEval z comb := by
  refine KMulHonest.lcEval_congr _ z comb (fun column mentioned => ?_)
  exact batchWitness_off_columns z digits column
    (fun digit member equal => fresh digit member (equal ▸ mentioned))

theorem carriedValue_fresh
    (z : Nat → Nat) (digits : List Digit) (value : Carried)
    (fresh : FreshCarried digits value) :
    carriedValue (batchWitness z digits) value = carriedValue z value := by
  unfold carriedValue
  rw [lcEval_fresh z digits value.low fresh.1,
    lcEval_fresh z digits value.high fresh.2]

/-! ## The claim's shape

Every field is a list the decoder supplies from a claim.  How long each one is
is a property of that claim, not a protocol constant, which is why every count
below is a fold. -/

/-- One Π_DEC transition, as the checks it owes. -/
structure Decomposition where
  /-- The decomposition base `b`. -/
  base : Nat
  /-- Every public `X` entry, against its `k_rho` child digits. -/
  xEntries : List (List Carried × Carried)
  /-- Every `y_ring` lane. -/
  yRingLanes : List (List Carried × Carried)
  /-- Every `y_zcol` lane, when the verifier enforces their recomposition. -/
  yZcolLanes : List (List Carried × Carried)
  /-- Every `aux_openings` entry. -/
  auxOpenings : List (List Carried × Carried)
  /-- Every child digit, range-checked.  The sole allocator. -/
  xDigits : List Digit
  /-- Every inactive `X` entry, forced to zero. -/
  inactiveX : List LinComb
  /-- Every `y_ring` lane past `D`, forced to zero. -/
  yRingPadding : List Carried
  /-- `s_col` agreement and `ct = y_ring[j][0]`. -/
  consistency : List (Carried × Carried)

namespace Decomposition

/-- The four radix-`b` recompositions, in the verifier's own order.  They are
one relation on four carriers, so they are one row program. -/
def recompositions (claim : Decomposition) : List (List Carried × Carried) :=
  claim.xEntries ++ claim.yRingLanes ++ claim.yZcolLanes ++ claim.auxOpenings

end Decomposition

/-! ## The emitted program -/

/-- **Π_DEC's emitted row program.** -/
def rows (claim : Decomposition) : List Row :=
  KRecomposition.recompositionsRows claim.base claim.recompositions
    ++ batchRows claim.xDigits
    ++ claim.inactiveX.flatMap KZeroCheck.zeroRows
    ++ KZeroCheck.paddingRows claim.yRingPadding
    ++ KConsistency.consistencyRows claim.consistency

/-! ## The derived row count

Stated as the sum of per-atom receipts, each of which is itself a fold.  A
closed formula alone would be a subtotal presented as a total. -/

private theorem sum_ones {α : Type} (items : List α) :
    (items.map (fun _ => 1)).sum = items.length := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis]
      omega

theorem inactiveRows_length (entries : List LinComb) :
    (entries.flatMap KZeroCheck.zeroRows).length = entries.length := by
  rw [List.length_flatMap,
    List.map_congr_left (fun entry _ => KZeroCheck.zeroRows_length entry),
    sum_ones]

/-- **The derived row count**, from the five atoms' receipts. -/
theorem rows_length (claim : Decomposition) :
    (rows claim).length
      = 2 * claim.recompositions.length
        + 2 * claim.xDigits.length
        + claim.inactiveX.length
        + 2 * claim.yRingPadding.length
        + 2 * claim.consistency.length := by
  unfold rows
  simp only [List.length_append,
    KRecomposition.recompositionsRows_length_eq, batchRows_length_eq,
    inactiveRows_length, KZeroCheck.paddingRows_length_eq,
    KConsistency.consistencyRows_length_eq]

/-! ## Allocation

Only the low-norm batch allocates, so the program's allocation is exactly its
allocation and the non-collision proof is exactly its non-collision proof. -/

def columns (claim : Decomposition) : List Nat :=
  batchColumns claim.xDigits

theorem columns_length (claim : Decomposition) :
    (columns claim).length = claim.xDigits.length :=
  batchColumns_length claim.xDigits

theorem columns_nodup (claim : Decomposition)
    (wellFormed : WellFormed claim.xDigits) : (columns claim).Nodup :=
  wellFormed.distinct

/-! ## Soundness

Satisfaction of the assembled program implies each atom's named relation. -/

theorem satisfies_parts
    (claim : Decomposition) (z : Nat → Nat)
    (satisfied : Satisfies (rows claim) z) :
    Satisfies (KRecomposition.recompositionsRows claim.base
        claim.recompositions) z
      ∧ Satisfies (batchRows claim.xDigits) z
      ∧ Satisfies (claim.inactiveX.flatMap KZeroCheck.zeroRows) z
      ∧ Satisfies (KZeroCheck.paddingRows claim.yRingPadding) z
      ∧ Satisfies (KConsistency.consistencyRows claim.consistency) z := by
  unfold rows at satisfied
  refine ⟨fun row member => satisfied row ?_, fun row member => satisfied row ?_,
    fun row member => satisfied row ?_, fun row member => satisfied row ?_,
    fun row member => satisfied row ?_⟩
  · exact List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_left _ (List.mem_append_left _ member)))
  · exact List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_left _ (List.mem_append_right _ member)))
  · exact List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_right _ member))
  · exact List.mem_append_left _ (List.mem_append_right _ member)
  · exact List.mem_append_right _ member

/-- **Every carrier recomposes.** -/
theorem rows_sound_recomposition
    (claim : Decomposition) (z : Nat → Nat) (constantWire : z 0 = 1)
    (satisfied : Satisfies (rows claim) z)
    (check : List Carried × Carried) (member : check ∈ claim.recompositions) :
    KRecomposition.hornerPair claim.base (check.1.map (carriedValue z))
      = carriedValue z check.2 :=
  KRecomposition.recompositionsRows_sound z claim.base claim.recompositions
    constantWire (satisfies_parts claim z satisfied).1 check member

/-- **Every child digit is in the centered window.** -/
theorem rows_sound_lowNorm
    (claim : Decomposition) (z : Nat → Nat)
    (satisfied : Satisfies (rows claim) z)
    (digit : Digit) (member : digit ∈ claim.xDigits) :
    lcEval z digit.value * lcEval z digit.value % goldilocksP
        * lcEval z digit.value % goldilocksP
      = lcEval z digit.value :=
  batchRows_sound z claim.xDigits (satisfies_parts claim z satisfied).2.1
    digit member

/-- **Every inactive `X` entry is zero.** -/
theorem rows_sound_inactive
    (claim : Decomposition) (z : Nat → Nat) (constantWire : z 0 = 1)
    (satisfied : Satisfies (rows claim) z)
    (entry : LinComb) (member : entry ∈ claim.inactiveX) :
    lcEval z entry = 0 :=
  KZeroCheck.zeroRows_sound z entry constantWire
    (fun row rowMember =>
      (satisfies_parts claim z satisfied).2.2.1 row
        (List.mem_flatMap.2 ⟨entry, member, rowMember⟩))

/-- **Every padded `y_ring` lane is zero.** -/
theorem rows_sound_padding
    (claim : Decomposition) (z : Nat → Nat) (constantWire : z 0 = 1)
    (satisfied : Satisfies (rows claim) z)
    (lane : Carried) (member : lane ∈ claim.yRingPadding) :
    carriedValue z lane = ⟨0, 0⟩ :=
  KZeroCheck.paddingRows_sound z claim.yRingPadding constantWire
    (satisfies_parts claim z satisfied).2.2.2.1 lane member

/-- **Every consistency pair agrees.** -/
theorem rows_sound_consistency
    (claim : Decomposition) (z : Nat → Nat) (constantWire : z 0 = 1)
    (satisfied : Satisfies (rows claim) z)
    (pair : Carried × Carried) (member : pair ∈ claim.consistency) :
    carriedValue z pair.1 = carriedValue z pair.2 :=
  KConsistency.consistencyRows_sound z claim.consistency constantWire
    (satisfies_parts claim z satisfied).2.2.2.2 pair member

/-! ## Honest completeness

One witness: the low-norm batch's, because nothing else allocates.  Every other
part must survive that extension, which is what `Fresh` records. -/

/-- Everything the allocation-free atoms read is blind to the allocation. -/
structure Fresh (claim : Decomposition) : Prop where
  recompositions : ∀ check ∈ claim.recompositions,
    (∀ child ∈ check.1, FreshCarried claim.xDigits child)
      ∧ FreshCarried claim.xDigits check.2
  inactive : ∀ entry ∈ claim.inactiveX, FreshComb claim.xDigits entry
  padding : ∀ lane ∈ claim.yRingPadding, FreshCarried claim.xDigits lane
  consistency : ∀ pair ∈ claim.consistency,
    FreshCarried claim.xDigits pair.1 ∧ FreshCarried claim.xDigits pair.2
  constantWire : FreshComb claim.xDigits [(0, 1)]

/-- The honest values the assembled program must exhibit.  Each field is
exactly the corresponding soundness conclusion, so the recipe is complete for
precisely the transitions it accepts. -/
structure Honest (claim : Decomposition) (z : Nat → Nat) : Prop where
  recomposes : ∀ check ∈ claim.recompositions,
    KRecomposition.hornerPair claim.base (check.1.map (carriedValue z))
      = carriedValue z check.2
  inWindow : ∀ digit ∈ claim.xDigits,
    lcEval z digit.value * lcEval z digit.value % goldilocksP
        * lcEval z digit.value % goldilocksP
      = lcEval z digit.value
  inactiveZero : ∀ entry ∈ claim.inactiveX, lcEval z entry = 0
  paddingZero : ∀ lane ∈ claim.yRingPadding, carriedValue z lane = ⟨0, 0⟩
  agrees : ∀ pair ∈ claim.consistency,
    carriedValue z pair.1 = carriedValue z pair.2

/-- **The assembled honest witness.**  The batch's, and nothing else. -/
def honestWitness (claim : Decomposition) (z : Nat → Nat) : Nat → Nat :=
  batchWitness z claim.xDigits

/-- **An honest Π_DEC transition satisfies the assembled program.** -/
theorem rows_honest
    (claim : Decomposition) (z : Nat → Nat) (constantWire : z 0 = 1)
    (wellFormed : WellFormed claim.xDigits)
    (fresh : Fresh claim) (honest : Honest claim z) :
    Satisfies (rows claim) (honestWitness claim z) := by
  have freshCarried : ∀ value : Carried, FreshCarried claim.xDigits value →
      carriedValue (honestWitness claim z) value = carriedValue z value :=
    fun value isFresh => carriedValue_fresh z claim.xDigits value isFresh
  have freshComb : ∀ comb : LinComb, FreshComb claim.xDigits comb →
      lcEval (honestWitness claim z) comb = lcEval z comb :=
    fun comb isFresh => lcEval_fresh z claim.xDigits comb isFresh
  have wire : honestWitness claim z 0 = 1 := by
    unfold honestWitness
    rw [batchWitness_off_columns z claim.xDigits 0
      (fun digit member equal => fresh.constantWire digit member
        (by simp [Mentions, equal])), constantWire]
  intro row member
  unfold rows at member
  rcases List.mem_append.1 member with inFirst | inConsistency
  · rcases List.mem_append.1 inFirst with inSecond | inPadding
    · rcases List.mem_append.1 inSecond with inThird | inInactive
      · rcases List.mem_append.1 inThird with inRecomposition | inLowNorm
        · refine KRecomposition.recompositionsRows_honest _ claim.base
            claim.recompositions wire (fun check checkMember => ?_) row
            inRecomposition
          have carriers := fresh.recompositions check checkMember
          have children : check.1.map (carriedValue (honestWitness claim z))
              = check.1.map (carriedValue z) :=
            List.map_congr_left (fun child childMember =>
              carriedValue_fresh z claim.xDigits child
                (carriers.1 child childMember))
          rw [children, freshCarried check.2 carriers.2]
          exact honest.recomposes check checkMember
        · exact batchRows_honest z claim.xDigits wellFormed honest.inWindow row
            inLowNorm
      · rcases List.mem_flatMap.1 inInactive with ⟨entry, entryMember, rowMember⟩
        refine KZeroCheck.zeroRows_honest _ entry wire ?_ row rowMember
        rw [freshComb entry (fresh.inactive entry entryMember)]
        exact honest.inactiveZero entry entryMember
    · refine KZeroCheck.paddingRows_honest _ claim.yRingPadding wire
        (fun lane laneMember => ?_) row inPadding
      rw [freshCarried lane (fresh.padding lane laneMember)]
      exact honest.paddingZero lane laneMember
  · refine KConsistency.consistencyRows_honest _ claim.consistency wire
      (fun pair pairMember => ?_) row inConsistency
    have carriers := fresh.consistency pair pairMember
    rw [freshCarried pair.1 carriers.1, freshCarried pair.2 carriers.2]
    exact honest.agrees pair pairMember

/-! ## The allocation is used, not only bounded

Conservation says no emitted row reaches outside the claim.  It says nothing
about the other direction, and the other direction is what makes
`Typed.Cost.auxiliaryColumns` a count of the **emitted program's** columns rather
than of a declaration.

Without it a claim could carry more digits than its rows constrain and the cost
would overcount, with conservation raising no objection — conservation bounds
mentions from above, and an unused declared column is not a mention. -/

/-- **Every column Π_DEC declares is one its rows use.** -/
theorem rows_use_columns
    (claim : Decomposition) (column : Nat) (member : column ∈ columns claim) :
    ∃ row ∈ rows claim, Mentions row.c column := by
  rcases batchRows_use_columns claim.xDigits column member with
    ⟨row, rowMember, mentions⟩
  refine ⟨row, ?_, mentions⟩
  unfold rows
  exact List.mem_append_left _ (List.mem_append_left _
    (List.mem_append_left _ (List.mem_append_right _ rowMember)))

/-! ## Conservation

Section 2 item 5, and the one item this recipe never had.  Every other recipe in
the assembly carried it; Π_DEC's absence was invisible because each of its five
atoms has its own, and having five parts each conserving is not the same as the
whole conserving — the composition is what says no emitted row reaches a column
the claim never supplied.

`Touches` names the columns the claim legitimately reaches.  It is a disjunction
over the five atoms because the atoms carry different data, not because the
statement is loose: each arm is that atom's own conservation conclusion. -/

/-- Every column Π_DEC may touch, traced to the claim that supplied it. -/
def Touches (claim : Decomposition) (column : Nat) : Prop :=
  column = 0
    ∨ (∃ check ∈ claim.recompositions,
        Mentions check.2.low column ∨ Mentions check.2.high column
          ∨ ∃ child ∈ check.1,
              Mentions child.low column ∨ Mentions child.high column)
    ∨ (∃ digit ∈ claim.xDigits,
        Mentions digit.value column ∨ column = digit.squareColumn)
    ∨ (∃ entry ∈ claim.inactiveX, Mentions entry column)
    ∨ (∃ lane ∈ claim.yRingPadding,
        Mentions lane.low column ∨ Mentions lane.high column)
    ∨ (∃ pair ∈ claim.consistency,
        Mentions pair.1.low column ∨ Mentions pair.1.high column
          ∨ Mentions pair.2.low column ∨ Mentions pair.2.high column)

/-- **No emitted row reaches a column the claim did not supply.**

Without this, Π_DEC could emit a row mentioning an arbitrary column — reading
another recipe's allocation — and nothing in the record would say otherwise.
That matters most under `CanonicalProgram`'s placement, where each part sits in
its own window and a stray column is a cross-window read. -/
theorem rows_conservation
    (claim : Decomposition) (row : Row) (member : row ∈ rows claim)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    Touches claim column := by
  unfold rows at member
  rcases List.mem_append.1 member with inFour | inConsistency
  · rcases List.mem_append.1 inFour with inThree | inPadding
    · rcases List.mem_append.1 inThree with inTwo | inInactive
      · rcases List.mem_append.1 inTwo with inRecompositions | inDigits
        · rcases KRecomposition.recompositionsRows_conservation claim.base
              claim.recompositions row inRecompositions column mentioned with
            ⟨check, checkMember, traced⟩
          rcases traced with wire | rest
          · exact Or.inl wire
          · exact Or.inr (Or.inl ⟨check, checkMember, rest⟩)
        · exact Or.inr (Or.inr (Or.inl
            (batchRows_conservation claim.xDigits row inDigits column
              mentioned)))
      · rcases List.mem_flatMap.1 inInactive with ⟨entry, entryMember, rowMember⟩
        rcases KZeroCheck.zeroRows_conservation entry row rowMember column
            mentioned with inEntry | wire
        · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨entry, entryMember, inEntry⟩)))
        · exact Or.inl wire
    · rcases KZeroCheck.paddingRows_conservation claim.yRingPadding row
          inPadding column mentioned with ⟨lane, laneMember, traced⟩
      rcases traced with low | high | wire
      · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl
          ⟨lane, laneMember, Or.inl low⟩))))
      · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl
          ⟨lane, laneMember, Or.inr high⟩))))
      · exact Or.inl wire
  · rcases KConsistency.consistencyRows_conservation claim.consistency row
        inConsistency column mentioned with ⟨pair, pairMember, traced⟩
    rcases traced with wire | rest
    · exact Or.inl wire
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨pair, pairMember, rest⟩))))

/-- **The allocation is exact.**

Both directions in one statement: the declared columns are precisely reached, and
nothing outside the claim is.  Section 2 item 4 asks for *exact* column
ownership, and one inclusion is not exactness. -/
theorem columns_exact (claim : Decomposition) :
    (∀ column ∈ columns claim, ∃ row ∈ rows claim, Mentions row.c column)
      ∧ (∀ row ∈ rows claim, ∀ column,
          (Mentions row.a column ∨ Mentions row.b column
            ∨ Mentions row.c column) → Touches claim column) :=
  ⟨fun column member => rows_use_columns claim column member,
    fun row member column mentioned =>
      rows_conservation claim row member column mentioned⟩

/-! ## Cost -/

/-- **Π_DEC's folded cost.**  Only the digit range check allocates. -/
def cost (claim : Decomposition) : Lowering.Typed.Cost where
  recurringRows :=
    2 * claim.recompositions.length + 2 * claim.xDigits.length
      + claim.inactiveX.length + 2 * claim.yRingPadding.length
      + 2 * claim.consistency.length
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := claim.xDigits.length

theorem cost_rows (claim : Decomposition) :
    (rows claim).length = (cost claim).recurringRows :=
  rows_length claim

theorem cost_columns (claim : Decomposition) :
    (columns claim).length = (cost claim).auxiliaryColumns :=
  columns_length claim

/-! ## The fourth carrier is dead in the shipping profile

`PIDEC-RECOMPOSITION` recorded four recomposed carriers, and that is exactly
right about `verify_dec_public` read on its own.  It is wrong about
`pi_dec::verify`, which is the actual entry point.

`pi_dec::verify` calls `validate_supported_sidecars` before it calls the engine,
and that validator **rejects any claim with a non-empty `aux_openings`**.  So by
the time `verify_dec_public`'s `aux_openings` recomposition loop is reached, the
list it iterates is empty.  The loop runs zero times, always.

The recipe is unaffected — an empty carrier list contributes no rows through the
fold — but the *count* is not four carriers' worth, and a reader deriving a
figure from "four carriers" would overcount.  Prompt section 4.3: a number can
be exactly right about something narrower than the sentence containing it. -/

/-- **With sidecars rejected, only three carriers recompose.** -/
theorem recompositions_length_without_sidecars
    (claim : Decomposition) (noSidecars : claim.auxOpenings = []) :
    claim.recompositions.length
      = claim.xEntries.length + claim.yRingLanes.length
        + claim.yZcolLanes.length := by
  unfold Decomposition.recompositions
  rw [noSidecars]
  simp only [List.length_append, List.append_nil]

/-- **And the row count loses that term.**  Stated so the dead carrier is
visible in the cost rather than hidden inside a fold. -/
theorem rows_length_without_sidecars
    (claim : Decomposition) (noSidecars : claim.auxOpenings = []) :
    (rows claim).length
      = 2 * (claim.xEntries.length + claim.yRingLanes.length
              + claim.yZcolLanes.length)
        + 2 * claim.xDigits.length
        + claim.inactiveX.length
        + 2 * claim.yRingPadding.length
        + 2 * claim.consistency.length := by
  rw [rows_length, recompositions_length_without_sidecars claim noSidecars]

/-! ## Two checks can share a row value, and that is not an ownership failure

`RECIPE-ROW-OWNERSHIP-RECOMPOSITION` exhibits two distinct recomposition checks
sharing a row.  This assembly **contains** `recompositionsRows`, so the same two
checks placed in `xEntries` put the same row value in `rows`.

That was carried from cycle 368 to cycle 392 as an obstruction to section 2
item 3.  **It is an obstruction to the wrong contract.**  Two checks that
constrain the same relation *should* emit the same row; a program that
deduplicated them would be a different program.  `PiDecOwnership` states the
contract this tree already settled on for Poseidon2 — position, not row value —
and `PiDecOwnership.ownership_is_positional` discharges item 3 for Π_DEC under
it.

`rows_owner_not_unique` below stays true and stays guarded.  It is a statement
about row *values*, which is not the ABI. -/

/-- A decomposition carrying only the two colliding recomposition checks. -/
def witnessDecomposition : Decomposition where
  base := 2
  xEntries := [KRecomposition.witnessCheckA, KRecomposition.witnessCheckB]
  yRingLanes := []
  yZcolLanes := []
  auxOpenings := []
  xDigits := []
  inactiveX := []
  yRingPadding := []
  consistency := []

/-- **A row of the assembled program belongs to two distinct checks.**

The parents' low combinations differ, so the checks are distinct; the row is
emitted by both and lies in the assembled program. -/
theorem rows_owner_not_unique :
    ∃ row : Row,
      row ∈ rows witnessDecomposition
        ∧ row ∈ KRecomposition.recompositionRows witnessDecomposition.base
            KRecomposition.witnessCheckA.1 KRecomposition.witnessCheckA.2
        ∧ row ∈ KRecomposition.recompositionRows witnessDecomposition.base
            KRecomposition.witnessCheckB.1 KRecomposition.witnessCheckB.2
        ∧ KRecomposition.witnessCheckA.2.low
            ≠ KRecomposition.witnessCheckB.2.low := by
  refine ⟨⟨[(1, 1)], [(0, 1)], [(3, 1)]⟩, ?_, ?_, ?_, ?_⟩ <;> decide

end Nightstream.Implementation.R1CS.Canonical.PiDecRecipe
