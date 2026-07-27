import Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
import Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-!
Contract: the support recurrence for the never-materialize Poseidon2 encoding.

Owns: the bound on how many distinct columns a carried combination can
reference, and the explicit witness list that realizes it.

Does not own: row counts (`Poseidon2Program`), matrix values
(`Poseidon2Matrices`), or semantic conformance (`POSEIDON2-ROUND-INDUCTION`).

## Why this is the obligation that makes 352 rows implementable

The encoding carries state symbolically, so a linear layer emits no row.  The
standing objection is that combinations must then blow up.  They do
*syntactically* — `applyMatrix` concatenates and never aggregates, so list
length grows geometrically.  They do not *mathematically*:

  * a full round S-boxes all eight lanes, and every S-box output is a fresh
    column, so support resets to exactly the eight fresh outputs;
  * a partial round S-boxes lane 0 only, so lanes 1..7 carry their support
    forward and exactly one fresh column joins it.

Hence support is at most `8 + 22 = 30` columns entering the terminal full
rounds, or `31` counting the constant wire that carries round constants.
`LinCombNormal.normalize` is what turns that bound into an implementable
representation, and `lcEval_normalize` is what makes using it sound.

The bound here is on *syntactic* support — the columns actually listed.  It is
therefore an upper bound on true support: `mentions_map_scale` keeps a column
listed even when its coefficient scales to zero.  Upper is the safe direction
for a width bound, so no cancellation argument is needed.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Support

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-! ## Support of a matrix application

A matrix application unions the supports of all its sources.  The result does
not depend on the target lane, which is why every lane of a post-layer state
carries the same support. -/

theorem mentions_applyMatrix
    (matrix : Fin width → Fin width → Nat) (state : State)
    (target : Fin width) (column : Nat) :
    Mentions (applyMatrix matrix state target) column ↔
      ∃ source : Fin width, Mentions (state source) column := by
  unfold applyMatrix
  rw [mentions_normalize]
  simp only [Mentions, List.map_flatMap, List.mem_flatMap, scale, List.map_map]
  constructor
  · rintro ⟨source, _, member⟩
    exact ⟨source, by simpa [Mentions, List.mem_map] using member⟩
  · rintro ⟨source, member⟩
    exact ⟨source, List.mem_finRange source, by
      simpa [Mentions, List.mem_map] using member⟩

/-- A round constant occupies the constant wire and nothing else. -/
theorem mentions_addConstant
    (constant : Nat) (comb : Poseidon2Core.LinComb) (column : Nat) :
    Mentions (addConstant constant comb) column ↔
      column = 0 ∨ Mentions comb column := by
  simp [Mentions, addConstant]

/-! ## Witness lists

Support is expressed as membership in an explicit list, so the bound is a
`List.length` rather than a cardinality argument — this project carries no set
library. -/

/-- The eight fresh outputs of the full round based at `base`. -/
def fullRoundOutputs (layout : Layout) (base : Nat) : List Nat :=
  (List.finRange width).map (fun lane => sboxOutput layout (base + lane.val))

theorem fullRoundOutputs_length (layout : Layout) (base : Nat) :
    (fullRoundOutputs layout base).length = width := by
  simp [fullRoundOutputs]

/-- Columns reachable in the state entering partial round `round`: the eight
outputs of the last initial full round, plus one fresh column per partial round
already taken. -/
def partialSupportList (layout : Layout) : Nat → List Nat
  | 0 => fullRoundOutputs layout (initialSboxIndex (halfFullRounds - 1) 0)
  | round + 1 =>
      sboxOutput layout (partialSboxIndex round) :: partialSupportList layout round

/-- **The recurrence, as a length.**  Eight to start, one per partial round. -/
theorem partialSupportList_length (layout : Layout) (round : Nat) :
    (partialSupportList layout round).length = width + round := by
  induction round with
  | zero => simp [partialSupportList, fullRoundOutputs_length]
  | succ previous hypothesis =>
      simp [partialSupportList, hypothesis]
      omega

/-! ## The bound -/

/-- The state entering the partial block is exactly the last full round's eight
fresh outputs. -/
theorem partialState_zero_mentions
    (layout : Layout) (lane : Fin width) (column : Nat)
    (mentioned : Mentions (partialState layout 0 lane) column) :
    column ∈ partialSupportList layout 0 := by
  simp only [partialState, halfFullRounds, initialState] at mentioned
  rcases (mentions_applyMatrix _ _ _ _).1 mentioned with ⟨source, member⟩
  simp only [Mentions, List.map_cons, List.map_nil, List.mem_singleton] at member
  simp only [partialSupportList, fullRoundOutputs, halfFullRounds,
    List.mem_map]
  exact ⟨source, List.mem_finRange source, by
    simpa [initialSboxIndex] using member.symm⟩

/-- **Support recurrence.**  Every column a partial-round state can reference
lies in the witness list, whose length is `8 + round`. -/
theorem partialState_mentions_subset
    (layout : Layout) (round : Nat) (lane : Fin width) (column : Nat)
    (mentioned : Mentions (partialState layout round lane) column) :
    column ∈ partialSupportList layout round := by
  induction round generalizing lane with
  | zero => exact partialState_zero_mentions layout lane column mentioned
  | succ previous hypothesis =>
      simp only [partialState] at mentioned
      rcases (mentions_applyMatrix _ _ _ _).1 mentioned with ⟨source, member⟩
      by_cases isLaneZero : source.val = 0
      · rw [if_pos isLaneZero] at member
        simp only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] at member
        simp only [partialSupportList, List.mem_cons]
        exact Or.inl member
      · rw [if_neg isLaneZero] at member
        simp only [partialSupportList, List.mem_cons]
        exact Or.inr (hypothesis source member)

/-- **Thirty columns entering the terminal rounds.**  Derived from the
recurrence, not asserted. -/
theorem partialSupport_bound (layout : Layout) :
    (partialSupportList layout partialRounds).length = 30 := by
  rw [partialSupportList_length]; decide

/-! ### The bound is not vacuous

`partialState_mentions_subset` would hold trivially of a state that mentioned
nothing.  These two facts establish that it does not: each partial round's
fresh output really is referenced by the next state, and every lane of the
entering state really does reference the last full round's outputs.  So support
genuinely grows by one per partial round and the list above is an upper bound
on something nonempty. -/

theorem partialState_mentions_fresh
    (layout : Layout) (round : Nat) (lane : Fin width) :
    Mentions (partialState layout (round + 1) lane)
      (sboxOutput layout (partialSboxIndex round)) := by
  simp only [partialState]
  refine (mentions_applyMatrix _ _ _ _).2 ⟨⟨0, by decide⟩, ?_⟩
  simp [Mentions]

theorem partialState_zero_mentions_output
    (layout : Layout) (lane source : Fin width) :
    Mentions (partialState layout 0 lane)
      (sboxOutput layout (initialSboxIndex (halfFullRounds - 1) source.val)) := by
  simp only [partialState, halfFullRounds, initialState]
  refine (mentions_applyMatrix _ _ _ _).2 ⟨source, ?_⟩
  simp [Mentions, initialSboxIndex]

/-! ### Terminal rounds

The maximum is attained entering the terminal block and then collapses.
`terminalState layout 0` is definitionally `partialState layout partialRounds`,
so the bound transfers with no new argument; every later terminal state is a
full round and resets to its eight fresh outputs. -/

theorem terminalState_zero_mentions_subset
    (layout : Layout) (lane : Fin width) (column : Nat)
    (mentioned : Mentions (terminalState layout 0 lane) column) :
    column ∈ partialSupportList layout partialRounds :=
  partialState_mentions_subset layout partialRounds lane column mentioned

/-- **A full round resets support to eight.**  Every lane of a post-full-round
state references only that round's eight fresh S-box outputs, which is why
support never accumulates outside the partial block. -/
theorem terminalState_succ_mentions
    (layout : Layout) (round : Nat) (lane : Fin width) (column : Nat)
    (mentioned : Mentions (terminalState layout (round + 1) lane) column) :
    ∃ source : Fin width,
      column = sboxOutput layout (terminalSboxIndex round source.val) := by
  simp only [terminalState] at mentioned
  rcases (mentions_applyMatrix _ _ _ _).1 mentioned with ⟨source, member⟩
  simp only [Mentions, List.map_cons, List.map_nil,
    List.mem_singleton] at member
  exact ⟨source, member⟩

/-! ## With the constant wire

The schedule hands each S-box its lane combination plus a round constant on
column `0`, so a partial-round S-box input references at most `31` columns. -/

/-- The combination the partial-round `round` S-box consumes. -/
def partialSboxInput (layout : Layout) (constants : Constants) (round : Nat) :
    Poseidon2Core.LinComb :=
  addConstant (constants.internal round) (partialState layout round ⟨0, by decide⟩)

/-- The schedule's index decomposition really does select this combination for
partial-family indices.  Without this, the `/ width` and `% width` case
analysis in `scheduleOf` would be an unchecked guess at which round an index
denotes. -/
theorem scheduleOf_partial
    (layout : Layout) (constants : Constants)
    (index : Fin sboxCount) (round : Nat)
    (isIndex : index.val = 32 + round)
    (inRange : round < partialRounds) :
    scheduleOf layout constants index = partialSboxInput layout constants round := by
  have roundLt : round < 22 := by simpa [partialRounds] using inRange
  have notInitial : ¬ (index.val < halfFullRounds * width) := by
    simp only [halfFullRounds, width, isIndex]; omega
  have isPartial : index.val < halfFullRounds * width + partialRounds := by
    simp only [halfFullRounds, width, partialRounds, isIndex]; omega
  have reduce : index.val - halfFullRounds * width = round := by
    simp only [halfFullRounds, width, isIndex]; omega
  unfold scheduleOf partialSboxInput
  rw [if_neg notInitial, if_pos isPartial, reduce]

/-- **Thirty-one columns, constant wire included.** -/
theorem partialSboxInput_mentions_bound
    (layout : Layout) (constants : Constants) (round : Nat) (column : Nat)
    (mentioned : Mentions (partialSboxInput layout constants round) column) :
    column = 0 ∨ column ∈ partialSupportList layout round := by
  rcases (mentions_addConstant _ _ _).1 mentioned with isConstantWire | inState
  · exact Or.inl isConstantWire
  · exact Or.inr (partialState_mentions_subset layout round _ column inState)


/-! ## Exact support for the partial block

`partialState_mentions_subset` bounds the support; these pin it.  Together with
`LinCombNormal.normalize_length_eq_witness` they turn the bound into the exact
normalized length, which is what the coefficient count needs. -/

/-- Every column in the witness list really is referenced.  The two ingredients
were already proved: the fresh output joins at each round, and the entering
state references all eight of the last full round's outputs. -/
theorem partialState_mentions_superset
    (layout : Layout) (round : Nat) (lane : Fin width) (column : Nat)
    (member : column ∈ partialSupportList layout round) :
    Mentions (partialState layout round lane) column := by
  induction round generalizing lane with
  | zero =>
      simp only [partialSupportList, fullRoundOutputs, List.mem_map] at member
      rcases member with ⟨source, _, rfl⟩
      have bridge : initialSboxIndex (halfFullRounds - 1) 0 + source.val
          = initialSboxIndex (halfFullRounds - 1) source.val := by
        simp only [initialSboxIndex]; omega
      rw [bridge]
      exact partialState_zero_mentions_output layout lane source
  | succ previous hypothesis =>
      simp only [partialSupportList, List.mem_cons] at member
      rcases member with rfl | inPrevious
      · exact partialState_mentions_fresh layout previous lane
      · simp only [partialState]
        refine (mentions_applyMatrix _ _ _ _).2 ⟨⟨1, by decide⟩, ?_⟩
        rw [if_neg (by decide)]
        exact hypothesis ⟨1, by decide⟩ inPrevious

/-- Every witness column is an S-box output with a bounded index.  This is what
makes the fresh column of the next round genuinely new. -/
theorem partialSupportList_index
    (layout : Layout) (round : Nat) (column : Nat)
    (member : column ∈ partialSupportList layout round) :
    ∃ index, index < halfFullRounds * width + round
      ∧ column = sboxOutput layout index := by
  induction round with
  | zero =>
      simp only [partialSupportList, fullRoundOutputs, List.mem_map] at member
      rcases member with ⟨source, _, rfl⟩
      have bound := source.isLt
      simp only [width] at bound
      exact ⟨initialSboxIndex (halfFullRounds - 1) 0 + source.val, by
        simp only [initialSboxIndex, halfFullRounds, width]; omega, rfl⟩
  | succ previous hypothesis =>
      simp only [partialSupportList, List.mem_cons] at member
      rcases member with rfl | inPrevious
      · exact ⟨partialSboxIndex previous, by
          simp only [partialSboxIndex]; omega, rfl⟩
      · rcases hypothesis inPrevious with ⟨index, bound, image⟩
        exact ⟨index, by omega, image⟩

theorem sboxOutput_injective (layout : Layout) (a b : Nat)
    (equal : sboxOutput layout a = sboxOutput layout b) : a = b := by
  simp only [sboxOutput, columnsPerSbox] at equal
  omega

theorem partialSupportList_nodup (layout : Layout) (round : Nat) :
    (partialSupportList layout round).Nodup := by
  induction round with
  | zero =>
      simp only [partialSupportList, fullRoundOutputs]
      refine nodup_map _ _ (fun a b image => ?_) (by decide)
      exact Fin.ext (by
        have := sboxOutput_injective layout _ _ image
        simp only [initialSboxIndex] at this; omega)
  | succ previous hypothesis =>
      simp only [partialSupportList, List.nodup_cons]
      refine ⟨?_, hypothesis⟩
      intro member
      rcases partialSupportList_index layout previous _ member with
        ⟨index, bound, image⟩
      have := sboxOutput_injective layout _ _ image
      simp only [partialSboxIndex] at this
      omega

/-- **The exact normalized length of a partial-round state.**  Eight columns
entering the block, one more per round taken. -/
theorem partialState_normalize_length
    (layout : Layout) (round : Nat) (lane : Fin width) :
    (normalize (partialState layout round lane)).length = width + round := by
  rw [normalize_length_eq_witness _ (partialSupportList layout round)
    (partialSupportList_nodup layout round)
    (fun column => ⟨partialState_mentions_subset layout round lane column,
      partialState_mentions_superset layout round lane column⟩)]
  exact partialSupportList_length layout round


/-! ## Support with a general entry state

The sponge enters a permutation carrying `state + chunk` rather than a single
column per lane.  `mentions_applyMatrix` is already general in the state, so the
support story transfers with no new argument: round 0 references exactly the
entry's columns, and every later round is entry-independent by
`initialStateFrom_succ_entry_irrelevant`, so the existing recurrence applies
unchanged from round one onward. -/

theorem initialStateFrom_zero_mentions
    (layout : Layout) (entry : State) (lane : Fin width) (column : Nat) :
    Mentions (initialStateFrom layout entry 0 lane) column
      ↔ ∃ source : Fin width, Mentions (entry source) column :=
  mentions_applyMatrix _ _ _ _

/-- Past round 0 the support no longer depends on the entry at all — a full
round has replaced every lane with a fresh output.  This is what lets a sponge
inherit the whole recurrence instead of restating it per permutation call. -/
theorem initialStateFrom_succ_mentions
    (layout : Layout) (entry : State) (round : Nat) (lane : Fin width)
    (column : Nat)
    (mentioned : Mentions (initialStateFrom layout entry (round + 1) lane) column) :
    ∃ source : Fin width,
      column = sboxOutput layout (initialSboxIndex round source.val) := by
  simp only [initialStateFrom] at mentioned
  rcases (mentions_applyMatrix _ _ _ _).1 mentioned with ⟨source, member⟩
  simp only [Mentions, List.map_cons, List.map_nil,
    List.mem_singleton] at member
  exact ⟨source, member⟩


/-- **The normalized length of a linear layer over any entry state.**
`normalize_length_applyMatrix_singletons` assumed one column per lane, which a
sponge entry (`state + chunk`, two terms per lane) violates.  This form takes a
duplicate-free witness list for the entry's combined support instead, so it
covers both: singleton lanes give the eight lane columns, and a sponge entry
gives the eight carried columns plus the absorbed ones. -/
theorem normalize_length_applyMatrix_witness
    (matrix : Fin width → Fin width → Nat) (entry : State) (witness : List Nat)
    (witnessNodup : witness.Nodup)
    (agree : ∀ column,
      (∃ source : Fin width, Mentions (entry source) column) ↔ column ∈ witness)
    (target : Fin width) :
    (normalize (applyMatrix matrix entry target)).length = witness.length := by
  refine normalize_length_eq_witness _ witness witnessNodup (fun column => ?_)
  rw [mentions_applyMatrix]
  exact agree column

/-- The same statement at the sponge's entry point, so round 0 of a carried
permutation is covered directly. -/
theorem initialStateFrom_zero_normalize_length
    (layout : Layout) (entry : State) (witness : List Nat)
    (witnessNodup : witness.Nodup)
    (agree : ∀ column,
      (∃ source : Fin width, Mentions (entry source) column) ↔ column ∈ witness)
    (lane : Fin width) :
    (normalize (initialStateFrom layout entry 0 lane)).length = witness.length :=
  normalize_length_applyMatrix_witness _ entry witness witnessNodup agree lane

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Support
