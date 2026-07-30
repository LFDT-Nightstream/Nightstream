import Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
import Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized

/-!
Contract: the reference sponge over the width-8 Poseidon2 permutation.

Owns: rate, capacity, digest length, absorption, padding, and the digest — on
*values*.  This is the Phase 3 specification, and like `Poseidon2Reference` it
is deliberately free of `Row`, `Layout` and `Satisfies` so it can be read
against the Rust without understanding any encoding.

Does not own: any row program.  The Phase 3 recipes `hashPrior` and `hashNext`
must be proved to compute this, exactly as `canonicalProgram` was proved to
compute `referencePermutation`.

## Provenance

Mirrors `build_bit_backed_poseidon2_hash_values` in
`crates/neo-fold-clean/src/engine/ccs_native/poseidon2.rs`:

    state = 0
    for each rate-sized chunk:  state[lane] += chunk[lane];  state = permute state
    state[0] += 1                                            -- padding
    state = permute state
    digest = state[0 .. DIGEST_LEN)

Parameters from `neo-params::poseidon2_goldilocks`: `RATE = 4`,
`DIGEST_LEN = 4`, and `WIDTH = 8` from the permutation, so capacity is `4`.
These are production choices, not paper-derived — neither SuperNeo nor
HyperNova selects a sponge.

## Chunking is a separate concern

`absorb` takes the input already split into chunks.  Splitting a flat field
list into rate-sized pieces is a distinct obligation — it fixes what happens to
a final short chunk and therefore what the padding must be — and folding it in
here would hide that choice inside the sponge.  It is named
`POSEIDON2-SPONGE-CHUNKING` and is not discharged.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.Lowering
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized

/-! ## Sponge parameters -/

/-- `neo-params::poseidon2_goldilocks::RATE`. -/
def rate : Nat := 4

/-- `neo-params::poseidon2_goldilocks::DIGEST_LEN`. -/
def digestLength : Nat := 4

/-- The capacity is whatever the rate leaves. -/
def capacity : Nat := width - rate

theorem capacity_eq : capacity = 4 := by decide

/-- Rate and capacity partition the state, so no lane is both absorbed into and
reserved. -/
theorem rate_add_capacity : rate + capacity = width := by decide

/-- The digest never exceeds the rate, so it reads only absorbed lanes. -/
theorem digest_within_rate : digestLength ≤ rate := by decide

/-! ## The sponge on values -/

/-- The all-zero starting state. -/
def initialSpongeState : Values := fun _ => 0

/-- Add one chunk into the low lanes.  A short chunk touches only the lanes it
covers, which is what makes the final padding step necessary. -/
def absorbChunk (chunk : List Nat) (state : Values) : Values :=
  fun lane =>
    match chunk[lane.val]? with
    | some value => (state lane + value) % goldilocksP
    | none => state lane

/-- A chunk that cannot exceed the rate by construction.  Absorption is written
against this rather than `List Nat`, so no absorbed chunk can reach a capacity
lane — see `oversized_chunk_touches_capacity` for why the unbounded carrier was
unsound. -/
structure RateChunk where
  values : List Nat
  bounded : values.length ≤ rate

/-- Absorb every chunk, permuting after each. -/
def absorb (constants : Constants) : List RateChunk → Values → Values
  | [], state => state
  | chunk :: rest, state =>
      absorb constants rest
        (referencePermutation constants (absorbChunk chunk.values state))

/-- Domain separation: a single `1` into lane 0 before the final permutation. -/
def pad (state : Values) : Values :=
  fun lane => if lane.val = 0 then (state lane + 1) % goldilocksP else state lane

/-- The sponge's final state. -/
def spongeFinal (constants : Constants) (chunks : List RateChunk) : Values :=
  referencePermutation constants (pad (absorb constants chunks initialSpongeState))

/-- The digest: the first `digestLength` lanes of the final state. -/
def digest (constants : Constants) (chunks : List RateChunk)
    (index : Fin digestLength) : Nat :=
  spongeFinal constants chunks
    ⟨index.val, by have := index.isLt; simp only [digestLength, width] at *; omega⟩

/-! ## Basic facts

Enough to pin the shape; the recipes must still be proved to compute this. -/

theorem absorb_nil (constants : Constants) (state : Values) :
    absorb constants [] state = state := rfl

theorem absorb_cons
    (constants : Constants) (chunk : RateChunk) (rest : List RateChunk)
    (state : Values) :
    absorb constants (chunk :: rest) state
      = absorb constants rest
          (referencePermutation constants (absorbChunk chunk.values state)) := rfl

/-- Absorbing touches no lane the chunk does not cover, so capacity lanes are
never written directly — only through the permutation. -/
theorem absorbChunk_beyond_chunk
    (chunk : List Nat) (state : Values) (lane : Fin width)
    (beyond : chunk.length ≤ lane.val) :
    absorbChunk chunk state lane = state lane := by
  unfold absorbChunk
  rw [List.getElem?_eq_none beyond]

/-- Padding touches lane 0 only. -/
theorem pad_beyond_zero (state : Values) (lane : Fin width)
    (notZero : lane.val ≠ 0) : pad state lane = state lane := by
  unfold pad; rw [if_neg notZero]


/-! ## How many permutations the spec demands

A fact about the *specification*, not about any encoding: the sponge invokes
the permutation once per chunk plus once for the padding block.  Any conforming
row program must therefore contain that many permutation instances, so this
fixes the multiplier on the permutation's 352 rows / 344 auxiliaries before a
single sponge row is written.

This is a lower bound on structure, not a cost claim.  No sponge encoding
exists yet, and absorption and padding will add rows of their own. -/

def permutationCalls (chunks : List RateChunk) : Nat := chunks.length + 1

theorem permutationCalls_eq (chunks : List RateChunk) :
    permutationCalls chunks = chunks.length + 1 := rfl

/-- Absorbing `n` chunks applies the permutation `n` times; `spongeFinal` adds
the padding block's one more. -/
theorem absorb_permutation_count
    (constants : Constants) (chunks : List RateChunk) (state : Values) :
    absorb constants chunks state
      = chunks.foldl
          (fun current chunk =>
            referencePermutation constants (absorbChunk chunk.values current))
          state := by
  induction chunks generalizing state with
  | nil => rfl
  | cons chunk rest hypothesis =>
      rw [absorb_cons, hypothesis, List.foldl_cons]

/-- The empty input still costs one permutation: the padding block is
unconditional, which is what makes the domain separator reach every digest. -/
theorem spongeFinal_empty (constants : Constants) :
    spongeFinal constants []
      = referencePermutation constants (pad initialSpongeState) := rfl

theorem permutationCalls_empty : permutationCalls [] = 1 := by decide


/-! ## Trailing zeros inside a chunk are invisible

The padding adds `1` to lane 0 after all chunk permutations.  That is a
finalization marker, not a length encoding: a chunk and the same chunk with a
trailing zero absorb identically, so they reach the same digest.

`absorbChunk_trailing_zero` is that collision, kernel-checked.  It is a fact
about the *specification* — and therefore about the Rust routine this mirrors —
not about any encoding.

What it does **not** say: that this is a defect.  It is a defect only if some
caller hashes variable-length inputs where a trailing zero is meaningful.  If
every preimage is fixed-width, or carries its own length tag, nothing is wrong.
Establishing which is `POSEIDON2-SPONGE-PREIMAGE-WIDTH`, and it must be settled
before `hashPrior` or `hashNext` can claim any binding property. -/

theorem absorbChunk_trailing_zero
    (value : Nat) (state : Values)
    (residues : ∀ lane : Fin width, state lane < goldilocksP)
    (lane : Fin width) :
    absorbChunk [value] state lane = absorbChunk [value, 0] state lane := by
  unfold absorbChunk
  match lane with
  | ⟨0, _⟩ => rfl
  | ⟨1, _⟩ =>
      simp only [List.getElem?_cons_succ, List.getElem?_cons_zero,
        List.getElem?_nil, Nat.add_zero]
      exact (Nat.mod_eq_of_lt (residues ⟨1, by decide⟩)).symm
  | ⟨lane + 2, bound⟩ => rfl

/-- The two chunk lists are genuinely different, so the collision is between
distinct inputs rather than a restatement. -/
theorem trailing_zero_inputs_differ : ([1] : List Nat) ≠ [1, 0] := by decide


/-! ## Capacity lanes must be unreachable by absorption

A sponge's security argument rests on the adversary never writing capacity
directly.  `absorbChunk` takes a `List Nat`, which does not enforce that: a
chunk longer than the rate writes into lanes 4..7.  Production never builds one
— it uses `input.chunks(RATE)` — but the type permits what the construction
must exclude, which is the same weakness class as the original
`SboxFrame.input : Nat`.

`oversized_chunk_touches_capacity` shows the gap is real rather than
theoretical, and `absorbChunk_capacity_untouched` is the guarantee once the
length is bounded.  A rate-bounded carrier is the fix; until absorption is
rewritten against `RateChunk`, the bound must be carried as a hypothesis
wherever the sponge argument is used. -/

/-- **The weakness is real.**  A five-element chunk writes capacity lane 4. -/
theorem oversized_chunk_touches_capacity :
    absorbChunk [0, 0, 0, 0, 7] (fun _ => 0) ⟨4, by decide⟩
      ≠ (fun _ : Fin width => (0 : Nat)) ⟨4, by decide⟩ := by
  decide

/-- **The guarantee, once the chunk respects the rate.**  No capacity lane is
written by absorption. -/
theorem absorbChunk_capacity_untouched
    (chunk : List Nat) (bounded : chunk.length ≤ rate)
    (state : Values) (lane : Fin width) (isCapacity : rate ≤ lane.val) :
    absorbChunk chunk state lane = state lane :=
  absorbChunk_beyond_chunk chunk state lane (Nat.le_trans bounded isCapacity)

/-- Absorption through the bounded carrier never writes capacity. -/
theorem RateChunk_capacity_untouched
    (chunk : RateChunk) (state : Values) (lane : Fin width)
    (isCapacity : rate ≤ lane.val) :
    absorbChunk chunk.values state lane = state lane :=
  absorbChunk_capacity_untouched chunk.values chunk.bounded state lane isCapacity

/-! ## Absorption does not mask a difference

`absorbChunk` **adds** the chunk into the rate lane modulo the prime.  Addition
is injective, so a difference between two chunks survives into the state — which
is the step that has to hold before any claim that a separator "reaches" the
digest means anything.

Without it, a recipe could absorb two different preimages into the same state
and the separation would be lost before a single permutation ran.  That failure
would be structural rather than cryptographic, and it is the one this rules
out. -/

private theorem add_mod_cancel_left
    (state first second : Nat)
    (firstCanonical : first < goldilocksP)
    (secondCanonical : second < goldilocksP)
    (equal : (state + first) % goldilocksP = (state + second) % goldilocksP) :
    first = second := by
  simp only [goldilocksP] at firstCanonical secondCanonical equal ⊢
  omega

/-- **Absorption is injective in the chunk, lane by lane.**

Two canonical values absorbed at the same lane of the same state give different
lanes exactly when they differ.  Stated at a lane rather than on whole chunks
because that is where a separator lands: one slot, not a whole block. -/
theorem absorbChunk_injective_at_lane
    (first second : List Nat) (state : Values) (lane : Fin width)
    (firstValue secondValue : Nat)
    (firstAt : first[lane.val]? = some firstValue)
    (secondAt : second[lane.val]? = some secondValue)
    (firstCanonical : firstValue < goldilocksP)
    (secondCanonical : secondValue < goldilocksP)
    (differ : firstValue ≠ secondValue) :
    absorbChunk first state lane ≠ absorbChunk second state lane := by
  unfold absorbChunk
  rw [firstAt, secondAt]
  intro equal
  exact differ (add_mod_cancel_left (state lane) firstValue secondValue
    firstCanonical secondCanonical equal)


/-! ## The fixed 23-field recipe

`hashPrior` and `hashNext` are not generic sponge calls: they consume a fixed
23-field preimage (`fpr-production-program-instantiation.md`, and the same
arity as `CIR-POSEIDON2-SPONGE23-RECIPE`).  Fixing the arity settles two things
the generic sponge left open.

`23 = 5·4 + 3`, so six absorb chunks — five full and one of three — plus the
unconditional padding block.  Seven permutation calls. -/

/-- The recipe's fixed preimage arity. -/
def sponge23Fields : Nat := 23

/-- Six absorb chunks: five at the rate, one short. -/
def sponge23Chunks : Nat := 6

theorem sponge23_chunk_arithmetic :
    sponge23Fields = (sponge23Chunks - 1) * rate + 3 := by decide

/-- **Seven permutation calls**, derived from the arity rather than counted off
the production recipe. -/
theorem sponge23_permutationCalls
    (chunks : List RateChunk) (count : chunks.length = sponge23Chunks) :
    permutationCalls chunks = 7 := by
  rw [permutationCalls_eq, count]
  decide

/-- The short final chunk still respects the rate, so the capacity guarantee
covers it. -/
theorem sponge23_final_chunk_bounded : 3 ≤ rate := by decide

/-- **The trailing-zero collision is unreachable at fixed arity.**  It needs two
preimages of different lengths; this recipe admits exactly one length.  So
`POSEIDON2-SPONGE-PREIMAGE-WIDTH` does not arise for `hashPrior` or `hashNext`,
and `POSEIDON2-SPONGE-CHUNKING` is determined rather than chosen. -/
theorem sponge23_single_arity
    (first second : List Nat)
    (firstFixed : first.length = sponge23Fields)
    (secondFixed : second.length = sponge23Fields) :
    first.length = second.length := by
  rw [firstFixed, secondFixed]


/-! ## Per-call layout and the absorption entry

A bound-shape sponge chains permutation calls: call `k`'s output ports are read
by call `k + 1`'s entry, together with that step's absorbed chunk columns.
Absorption emits no row — it adds a term to the carried combination — which is
what `Poseidon2Schedule.initialStateFrom` was generalized for.

`entryOf` is the `State` a call enters on.  Lane `l` carries the previous
call's output port, plus a chunk column when that lane is absorbed into.  Call
`0` has no predecessor, so it carries the chunk alone. -/

structure SpongeLayout where
  /-- The permutation layout for call `k`. -/
  call : Nat → Layout
  /-- The column holding chunk `k`'s value for lane `l`. -/
  chunkColumn : Nat → Fin width → Nat

/-- The entry state for call `k`, given how many lanes chunk `k` covers. -/
def entryOf (spongeLayout : SpongeLayout) (chunkLength : Nat → Nat) (call : Nat) :
    State :=
  fun lane =>
    let absorbed : Poseidon2Core.LinComb :=
      if lane.val < chunkLength call
      then [(spongeLayout.chunkColumn call lane, 1)] else []
    match call with
    | 0 => absorbed
    | previous + 1 =>
        ((spongeLayout.call previous).outputPort lane, 1) :: absorbed

/-- **The entry references only carried ports and chunk columns.**  This is the
classification a sponge-level conservation argument needs, and it is what
bounds round 0's support: at most two columns per lane, hence at most `16`
across the eight lanes before the constant wire. -/
theorem entryOf_mentions
    (spongeLayout : SpongeLayout) (chunkLength : Nat → Nat) (call : Nat)
    (lane : Fin width) (column : Nat)
    (mentioned : Mentions (entryOf spongeLayout chunkLength call lane) column) :
    (∃ previous, call = previous + 1
        ∧ column = (spongeLayout.call previous).outputPort lane)
      ∨ column = spongeLayout.chunkColumn call lane := by
  unfold entryOf at mentioned
  cases call with
  | zero =>
      simp only [Mentions] at mentioned
      split at mentioned
      · simp only [List.map_cons, List.map_nil, List.mem_singleton] at mentioned
        exact Or.inr mentioned
      · simp at mentioned
  | succ previous =>
      simp only [Mentions, List.map_cons, List.mem_cons] at mentioned
      rcases mentioned with isPort | inAbsorbed
      · exact Or.inl ⟨previous, rfl, isPort⟩
      · split at inAbsorbed
        · simp only [List.map_cons, List.map_nil,
            List.mem_singleton] at inAbsorbed
          exact Or.inr inAbsorbed
        · simp at inAbsorbed

/-- A lane the chunk does not cover carries only the previous call's output, so
capacity lanes are never absorbed into.  With `RateChunk` this holds for every
lane at or above the rate. -/
theorem entryOf_beyond_chunk
    (spongeLayout : SpongeLayout) (chunkLength : Nat → Nat)
    (previous : Nat) (lane : Fin width)
    (beyond : chunkLength (previous + 1) ≤ lane.val) :
    entryOf spongeLayout chunkLength (previous + 1) lane
      = [((spongeLayout.call previous).outputPort lane, 1)] := by
  unfold entryOf
  rw [if_neg (by omega)]



/-! ## The entry evaluates to the absorb step

This is the hinge of the seven-call chain.  Call `k`'s output ports carry the
reference image of its entry (`canonicalProgramFrom_computes_reference`); this
says call `k + 1`'s entry then evaluates to exactly that image with the chunk
added — which is `absorbChunk`.  So the chain telescopes to `absorb`. -/

theorem entryOf_succ_eval_absorbed
    (spongeLayout : SpongeLayout) (chunkLength : Nat → Nat) (z : Nat → Nat)
    (previous : Nat) (lane : Fin width)
    (covered : lane.val < chunkLength (previous + 1)) :
    lcEval z (entryOf spongeLayout chunkLength (previous + 1) lane)
      = (z ((spongeLayout.call previous).outputPort lane)
          + z (spongeLayout.chunkColumn (previous + 1) lane)) % goldilocksP := by
  unfold entryOf
  rw [if_pos covered]
  simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul]

theorem entryOf_succ_eval_carried
    (spongeLayout : SpongeLayout) (chunkLength : Nat → Nat) (z : Nat → Nat)
    (previous : Nat) (lane : Fin width)
    (residue : z ((spongeLayout.call previous).outputPort lane) < goldilocksP)
    (beyond : ¬ (lane.val < chunkLength (previous + 1))) :
    lcEval z (entryOf spongeLayout chunkLength (previous + 1) lane)
      = z ((spongeLayout.call previous).outputPort lane) := by
  unfold entryOf
  rw [if_neg beyond]
  simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul]
  exact Nat.mod_eq_of_lt residue

/-- **The entry is the absorb step.**  With call `k`'s outputs as the carried
state and the chunk columns holding the absorbed values, call `k + 1`'s entry
evaluates lanewise to `absorbChunk` applied to that state — the sponge's own
absorption, reproduced by the encoding at no row cost. -/
theorem entryOf_eval_is_absorbChunk
    (spongeLayout : SpongeLayout) (chunkLength : Nat → Nat) (z : Nat → Nat)
    (previous : Nat) (chunk : List Nat) (carried : Values)
    (residues : ∀ column, z column < goldilocksP)
    (chunkLengthAgrees : chunkLength (previous + 1) = chunk.length)
    (carriedAgrees : ∀ lane : Fin width,
      z ((spongeLayout.call previous).outputPort lane) = carried lane)
    (chunkAgrees : ∀ lane : Fin width, ∀ value,
      chunk[lane.val]? = some value
        → z (spongeLayout.chunkColumn (previous + 1) lane) = value)
    (lane : Fin width) :
    lcEval z (entryOf spongeLayout chunkLength (previous + 1) lane)
      = absorbChunk chunk carried lane := by
  unfold absorbChunk
  cases covering : chunk[lane.val]? with
  | none =>
      have beyond : ¬ (lane.val < chunkLength (previous + 1)) := by
        rw [chunkLengthAgrees]
        exact fun below => by simp [List.getElem?_eq_getElem below] at covering
      rw [entryOf_succ_eval_carried _ _ _ _ _ (residues _) beyond,
        carriedAgrees lane]
  | some value =>
      have covered : lane.val < chunkLength (previous + 1) := by
        rw [chunkLengthAgrees]
        rcases Nat.lt_or_ge lane.val chunk.length with below | beyond
        · exact below
        · rw [List.getElem?_eq_none beyond] at covering
          exact absurd covering (by simp)
      rw [entryOf_succ_eval_absorbed _ _ _ _ _ covered,
        carriedAgrees lane, chunkAgrees lane value covering]


/-! ## Permutation-block layout well-formedness

Per-call well-formedness is not enough: distinct permutation calls must also
allocate disjoint columns, or their frame-column costs would not add.
`WellFormed` bundles exactly that permutation-block obligation, and
`canonicalSpongeLayout` constructs it.

This predicate deliberately does **not** classify `chunkColumn`.  A complete
sponge-call receipt must additionally prove that every live chunk coordinate is
either an authoritative input coordinate or the shared constant-one coordinate,
and that it does not alias any permutation-owned temporary.  Keeping that
obligation out of this generic carrier is necessary because the fixed padding
chunk intentionally reads the shared constant wire rather than allocating a
new chunk column. -/

structure SpongeLayout.WellFormed (spongeLayout : SpongeLayout) (stride : Nat) :
    Prop where
  /-- Every call's own layout is coherent. -/
  perCall : ∀ call, Poseidon2Layout.WellFormed (spongeLayout.call call)
  /-- The stride leaves room for a full column space per call. -/
  strideClears : Poseidon2Layout.canonicalColumnTotal ≤ stride
  /-- Calls are placed at successive strides. -/
  callAtShift : ∀ call,
    spongeLayout.call call = Poseidon2Layout.shiftedLayout (call * stride)

/-- **Distinct calls allocate disjoint auxiliary columns.**  The consequence the
cost fold needs. -/
theorem SpongeLayout.WellFormed.auxDisjoint
    {spongeLayout : SpongeLayout} {stride : Nat}
    (wellFormed : SpongeLayout.WellFormed spongeLayout stride)
    (first second : Nat) (distinct : first ≠ second)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox)
    (other : Fin sboxCount) (otherSlot : Fin columnsPerSbox) :
    sboxColumn (spongeLayout.call first) index slot
      ≠ sboxColumn (spongeLayout.call second) other otherSlot := by
  rw [wellFormed.callAtShift first, wellFormed.callAtShift second]
  exact Poseidon2Layout.shiftedLayout_aux_disjoint first second stride
    wellFormed.strideClears distinct index slot other otherSlot

/-- Stride for the canonical sponge: a full column space plus eight chunk
columns. -/
def spongeStride : Nat := 369

theorem spongeStride_clears :
    Poseidon2Layout.canonicalColumnTotal ≤ spongeStride := by decide

/-- A concrete sponge layout: call `k` at stride `k`, with its chunk columns in
the gap the stride leaves above that call's column space. -/
def canonicalSpongeLayout : SpongeLayout where
  call := fun index => Poseidon2Layout.shiftedLayout (index * spongeStride)
  chunkColumn := fun index lane =>
    index * spongeStride + Poseidon2Layout.canonicalColumnTotal + lane.val

theorem canonicalSpongeLayout_wellFormed :
    SpongeLayout.WellFormed canonicalSpongeLayout spongeStride where
  perCall := fun call => Poseidon2Layout.shiftedLayout_wellFormed _
  strideClears := spongeStride_clears
  callAtShift := fun _ => rfl


/-! ## Sponge-level column classification

Composing `Poseidon2Conservation.scheduleOfFrom_columns` with `entryOf_mentions`:
every column a call's scheduled input can reference is the constant wire, a
carried output port from the previous call, one of this call's chunk columns, or
one of this call's own S-box outputs.  Nothing else. -/

theorem spongeScheduleOf_columns
    (spongeLayout : SpongeLayout) (chunkLength : Nat → Nat)
    (constants : Constants) (call : Nat) (index : Fin sboxCount) (column : Nat)
    (mentioned : Mentions
      (scheduleOfFrom (spongeLayout.call call)
        (entryOf spongeLayout chunkLength call) constants index) column) :
    column = 0
      ∨ (∃ previous, ∃ lane : Fin width, call = previous + 1
          ∧ column = (spongeLayout.call previous).outputPort lane)
      ∨ (∃ lane : Fin width, column = spongeLayout.chunkColumn call lane)
      ∨ (∃ other, other < sboxCount
          ∧ column = sboxOutput (spongeLayout.call call) other) := by
  rcases Poseidon2Conservation.scheduleOfFrom_columns (spongeLayout.call call)
    (entryOf spongeLayout chunkLength call) constants index column mentioned with
    wire | fromEntry | output
  · exact Or.inl wire
  · rcases fromEntry with ⟨source, inEntry⟩
    rcases entryOf_mentions spongeLayout chunkLength call source column inEntry with
      carried | chunk
    · rcases carried with ⟨previous, isSucc, isPort⟩
      exact Or.inr (Or.inl ⟨previous, source, isSucc, isPort⟩)
    · exact Or.inr (Or.inr (Or.inl ⟨source, chunk⟩))
  · exact Or.inr (Or.inr (Or.inr output))

/-- **The sponge stays inside its declared column space.**  Under a well-formed
sponge layout, every referenced column belongs to the call that references it or
to its immediate predecessor — never to an unrelated call. -/
theorem spongeScheduleOf_no_foreign_aux
    (spongeLayout : SpongeLayout) (stride : Nat)
    (wellFormed : SpongeLayout.WellFormed spongeLayout stride)
    (chunkLength : Nat → Nat) (constants : Constants)
    (call : Nat) (index : Fin sboxCount) (column : Nat)
    (mentioned : Mentions
      (scheduleOfFrom (spongeLayout.call call)
        (entryOf spongeLayout chunkLength call) constants index) column)
    (foreign : Nat) (distinct : foreign ≠ call)
    (foreignIndex : Fin sboxCount) (foreignSlot : Fin columnsPerSbox) :
    column = sboxColumn (spongeLayout.call foreign) foreignIndex foreignSlot
      → column = 0
        ∨ (∃ previous, ∃ lane : Fin width, call = previous + 1
            ∧ column = (spongeLayout.call previous).outputPort lane)
        ∨ (∃ lane : Fin width, column = spongeLayout.chunkColumn call lane) := by
  intro isForeign
  rcases spongeScheduleOf_columns spongeLayout chunkLength constants call index
    column mentioned with wire | carried | chunk | own
  · exact Or.inl wire
  · exact Or.inr (Or.inl carried)
  · exact Or.inr (Or.inr chunk)
  · rcases own with ⟨other, otherLt, isOwn⟩
    exact absurd (isOwn ▸ isForeign)
      (by
        intro equal
        exact wellFormed.auxDisjoint foreign call (fun same => distinct same)
          foreignIndex foreignSlot ⟨other, otherLt⟩ ⟨3, by decide⟩ equal.symm)


/-! ## Assembled sponge conservation

Cycle 227 proved the classification for a scheduled *input*; the emitted rows
also carry frame columns and the terminal binding's ports.  This is the
assembled statement over `spongeProgram`, mirroring
`Poseidon2ProgramConservation.normalizedCanonicalProgram_conservation`. -/

/-- Every column any emitted sponge row can reference, attributed to the call
that emitted it. -/
def SpongeColumn (spongeLayout : SpongeLayout) (call : Nat) (column : Nat) :
    Prop :=
  column = 0
    ∨ (∃ previous, ∃ lane : Fin width, call = previous + 1
        ∧ column = (spongeLayout.call previous).outputPort lane)
    ∨ (∃ lane : Fin width, column = spongeLayout.chunkColumn call lane)
    ∨ (∃ lane : Fin width, column = (spongeLayout.call call).outputPort lane)
    ∨ (∃ index : Fin sboxCount, ∃ slot : Fin columnsPerSbox,
        column = sboxColumn (spongeLayout.call call) index slot)

theorem spongeCall_conservation
    (spongeLayout : SpongeLayout) (chunkLength : Nat → Nat)
    (constants : Constants) (call : Nat) (row : Row)
    (member : row ∈ canonicalProgramFrom (spongeLayout.call call)
      (entryOf spongeLayout chunkLength call) constants)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    SpongeColumn spongeLayout call column := by
  unfold canonicalProgramFrom permutationProgram sboxProgram bindingProgram
    terminalBindingRows at member
  rcases List.mem_append.1 member with inSbox | inBinding
  · rcases List.mem_flatMap.1 inSbox with ⟨index, _, rowMember⟩
    have scheduled := spongeScheduleOf_columns spongeLayout chunkLength constants
      call index column
    have frame : ∀ slot : Fin columnsPerSbox,
        Mentions [(sboxColumn (spongeLayout.call call) index slot, 1)] column
          → SpongeColumn spongeLayout call column := by
      intro slot inSingleton
      simp only [Mentions, List.map_cons, List.map_nil,
        List.mem_singleton] at inSingleton
      exact Or.inr (Or.inr (Or.inr (Or.inr ⟨index, slot, inSingleton⟩)))
    have viaSchedule : Mentions
        (scheduleOfFrom (spongeLayout.call call)
          (entryOf spongeLayout chunkLength call) constants index) column
        → SpongeColumn spongeLayout call column := by
      intro inInput
      rcases scheduled inInput with wire | carried | chunk | own
      · exact Or.inl wire
      · exact Or.inr (Or.inl carried)
      · exact Or.inr (Or.inr (Or.inl chunk))
      · rcases own with ⟨other, otherLt, isOwn⟩
        exact Or.inr (Or.inr (Or.inr (Or.inr
          ⟨⟨other, by simpa [sboxCount, externalRounds, width, partialRounds]
              using otherLt⟩, ⟨3, by decide⟩, isOwn⟩)))
    simp only [sboxRows, List.mem_cons, List.not_mem_nil, or_false] at rowMember
    rcases rowMember with rfl | rfl | rfl | rfl <;>
      simp only [rowSquare, rowFourth, rowSixth, rowSeventh, frameAt] at mentioned
    · rcases mentioned with a | b | c
      · exact viaSchedule a
      · exact viaSchedule b
      · exact frame ⟨0, by decide⟩ c
    · rcases mentioned with a | b | c
      · exact frame ⟨0, by decide⟩ a
      · exact frame ⟨0, by decide⟩ b
      · exact frame ⟨1, by decide⟩ c
    · rcases mentioned with a | b | c
      · exact frame ⟨0, by decide⟩ a
      · exact frame ⟨1, by decide⟩ b
      · exact frame ⟨2, by decide⟩ c
    · rcases mentioned with a | b | c
      · exact viaSchedule a
      · exact frame ⟨2, by decide⟩ b
      · exact frame ⟨3, by decide⟩ c
  · rcases List.mem_map.1 inBinding with ⟨lane, _, rfl⟩
    simp only [bindRow] at mentioned
    rcases mentioned with final | wire | port
    · rcases Poseidon2Conservation.terminalState_columns (spongeLayout.call call)
        halfFullRounds (Nat.le_refl _) lane column final with ⟨index, bound, image⟩
      exact Or.inr (Or.inr (Or.inr (Or.inr
        ⟨⟨index, by simpa [sboxCount, externalRounds, width, partialRounds]
            using bound⟩, ⟨3, by decide⟩, image⟩)))
    · simp only [Mentions, List.map_cons, List.map_nil,
        List.mem_singleton] at wire
      exact Or.inl wire
    · simp only [Mentions, List.map_cons, List.map_nil,
        List.mem_singleton] at port
      exact Or.inr (Or.inr (Or.inr (Or.inl ⟨lane, port⟩)))


/-! ## The sponge row program

Seven calls, each the permutation program entered on that call's carried state.
Absorption and padding contribute no rows of their own — they are terms in the
entry combination — so the whole cost is permutation cost.

This is a row program for the **selected comparison encoding** (bound shape,
`POSEIDON2-ENCODING-CLASS-AND-ORDER`), not a canonical minimum. -/

def spongeProgram (spongeLayout : SpongeLayout) (constants : Constants)
    (chunkLength : Nat → Nat) (calls : Nat) : List Row :=
  (List.range calls).flatMap
    (fun call =>
      normalizedCanonicalProgramFrom (spongeLayout.call call)
        (entryOf spongeLayout chunkLength call) constants)

/-- **The row count, folded from the per-call receipts.** -/
theorem spongeProgram_length
    (spongeLayout : SpongeLayout) (constants : Constants)
    (chunkLength : Nat → Nat) (calls : Nat) :
    (spongeProgram spongeLayout constants chunkLength calls).length
      = calls * 352 := by
  unfold spongeProgram
  rw [length_flatMap_uniform _ _ 352
    (by intro call; exact normalizedCanonicalProgramFrom_length _ _ _),
    List.length_range]

/-- **The fixed-23 sponge core has 2,464 rows.**  Seven calls at 352, derived
from the emitted program rather than declared.  This is not yet either typed
hash call: it contains no fixed preimage serialization, padding-coordinate
authority, activation wrapper, or output receipt. -/
theorem sponge23Program_length
    (spongeLayout : SpongeLayout) (constants : Constants)
    (chunkLength : Nat → Nat) :
    (spongeProgram spongeLayout constants chunkLength 7).length = 2464 := by
  rw [spongeProgram_length]


/-- **Satisfying the sponge satisfies every call.**  This is what lets the
seven-call induction apply `canonicalProgramFrom_computes_reference` at each
step: the sponge program is the concatenation of the per-call programs, so its
satisfaction restricts to each one. -/
theorem spongeProgram_satisfies_call
    (spongeLayout : SpongeLayout) (constants : Constants)
    (chunkLength : Nat → Nat) (calls : Nat) (z : Nat → Nat)
    (satisfied : Satisfies (spongeProgram spongeLayout constants chunkLength calls) z)
    (call : Nat) (inRange : call < calls) :
    Satisfies
      (normalizedCanonicalProgramFrom (spongeLayout.call call)
        (entryOf spongeLayout chunkLength call) constants) z := by
  intro row member
  refine satisfied row ?_
  unfold spongeProgram
  exact List.mem_flatMap.2 ⟨call, List.mem_range.2 inRange, member⟩

/-- Each call's outputs therefore carry the reference image of that call's
entry.  Composing this across calls — with `entryOf_eval_is_absorbChunk`
supplying the entry values at each step — is the remaining induction. -/
theorem spongeProgram_call_computes_reference
    (spongeLayout : SpongeLayout) (constants : Constants)
    (chunkLength : Nat → Nat) (calls : Nat) (z : Nat → Nat)
    (entryValues : Values)
    (residues : ∀ column, z column < goldilocksP)
    (constantWire : z 0 = 1)
    (satisfied : Satisfies (spongeProgram spongeLayout constants chunkLength calls) z)
    (call : Nat) (inRange : call < calls)
    (entryAgrees : ∀ source : Fin width,
      lcEval z (entryOf spongeLayout chunkLength call source) = entryValues source)
    (lane : Fin width) :
    z ((spongeLayout.call call).outputPort lane)
      = referencePermutation constants entryValues lane :=
  Poseidon2Normalized.normalizedCanonicalProgramFrom_computes_reference
    (spongeLayout.call call) constants z
    (entryOf spongeLayout chunkLength call) entryValues residues constantWire
    entryAgrees
    (spongeProgram_satisfies_call spongeLayout constants chunkLength calls z
      satisfied call inRange)
    lane


/-! ## The seven-call chain

`chainValues` is the value each call enters on, defined to mirror `absorb`
exactly, so the induction's invariant is definitional rather than threaded by
hand.  `spongeChain` then says every call's outputs carry the reference image of
its accumulated entry — which at the last call is the sponge's own state. -/

def chainValues (constants : Constants) (chunkAt : Nat → List Nat) :
    Nat → Values
  | 0 => absorbChunk (chunkAt 0) (fun _ => 0)
  | previous + 1 =>
      absorbChunk (chunkAt (previous + 1))
        (referencePermutation constants (chainValues constants chunkAt previous))

theorem entryOf_zero_eval_is_absorbChunk
    (spongeLayout : SpongeLayout) (chunkLength : Nat → Nat) (z : Nat → Nat)
    (chunkAt : Nat → List Nat)
    (chunkLengthAgrees : chunkLength 0 = (chunkAt 0).length)
    (chunkAgrees : ∀ lane : Fin width, ∀ value,
      (chunkAt 0)[lane.val]? = some value
        → z (spongeLayout.chunkColumn 0 lane) = value)
    (lane : Fin width) :
    lcEval z (entryOf spongeLayout chunkLength 0 lane)
      = absorbChunk (chunkAt 0) (fun _ => 0) lane := by
  unfold entryOf absorbChunk
  cases covering : (chunkAt 0)[lane.val]? with
  | none =>
      have beyond : ¬ (lane.val < chunkLength 0) := by
        rw [chunkLengthAgrees]
        rcases Nat.lt_or_ge lane.val (chunkAt 0).length with below | _
        · simp [List.getElem?_eq_getElem below] at covering
        · omega
      rw [if_neg beyond]
      simp [lcEval, List.foldl]
  | some value =>
      have covered : lane.val < chunkLength 0 := by
        rw [chunkLengthAgrees]
        rcases Nat.lt_or_ge lane.val (chunkAt 0).length with below | beyond
        · exact below
        · rw [List.getElem?_eq_none beyond] at covering
          exact absurd covering (by simp)
      rw [if_pos covered]
      simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul,
        chunkAgrees lane value covering]

/-- **The chain.**  Every call's output ports carry the reference image of its
accumulated entry values. -/
theorem spongeChain
    (spongeLayout : SpongeLayout) (constants : Constants)
    (chunkLength : Nat → Nat) (calls : Nat) (z : Nat → Nat)
    (chunkAt : Nat → List Nat)
    (residues : ∀ column, z column < goldilocksP)
    (constantWire : z 0 = 1)
    (satisfied : Satisfies (spongeProgram spongeLayout constants chunkLength calls) z)
    (chunkLengthAgrees : ∀ call, chunkLength call = (chunkAt call).length)
    (chunkAgrees : ∀ call, ∀ lane : Fin width, ∀ value,
      (chunkAt call)[lane.val]? = some value
        → z (spongeLayout.chunkColumn call lane) = value)
    (call : Nat) (inRange : call < calls) (lane : Fin width) :
    z ((spongeLayout.call call).outputPort lane)
      = referencePermutation constants
          (chainValues constants chunkAt call) lane := by
  induction call generalizing lane with
  | zero =>
      refine spongeProgram_call_computes_reference spongeLayout constants
        chunkLength calls z _ residues constantWire satisfied 0 inRange
        (fun source => ?_) lane
      exact entryOf_zero_eval_is_absorbChunk spongeLayout chunkLength z chunkAt
        (chunkLengthAgrees 0) (chunkAgrees 0) source
  | succ previous hypothesis =>
      have previousRange : previous < calls := by omega
      refine spongeProgram_call_computes_reference spongeLayout constants
        chunkLength calls z _ residues constantWire satisfied (previous + 1)
        inRange (fun source => ?_) lane
      rw [entryOf_eval_is_absorbChunk spongeLayout chunkLength z previous
        (chunkAt (previous + 1))
        (referencePermutation constants (chainValues constants chunkAt previous))
        residues (chunkLengthAgrees (previous + 1))
        (fun other => hypothesis previousRange other)
        (chunkAgrees (previous + 1)) source]
      rfl



/-! ## Padding is absorption

`pad` adds `1` to lane 0 and leaves the rest.  `absorbChunk [1]` does exactly
that: lane 0 gets `state 0 + 1`, every other lane falls through the `none`
branch.  So the padding block is not a special case — it is the sponge
absorbing the one-element chunk `[1]`, and the encoding needs no separate
construction for it.

That collapses the seven calls into a single uniform recursion: six data chunks
and one `[1]`, all handled by `entryOf` and `absorbAt` unchanged. -/

theorem pad_eq_absorbChunk_one (state : Values) (lane : Fin width) :
    pad state lane = absorbChunk [1] state lane := by
  unfold pad absorbChunk
  match lane with
  | ⟨0, _⟩ => rfl
  | ⟨lane + 1, bound⟩ => rfl

theorem pad_eq_absorbChunk_one_funext (state : Values) :
    pad state = absorbChunk [1] state :=
  funext (fun lane => pad_eq_absorbChunk_one state lane)

/-- The one-element padding chunk respects the rate, so it is a legitimate
`RateChunk` and the capacity guarantee covers it. -/
def paddingChunk : RateChunk where
  values := [1]
  bounded := by decide

theorem paddingChunk_absorbs (state : Values) :
    absorbChunk paddingChunk.values state = pad state :=
  (pad_eq_absorbChunk_one_funext state).symm


/-! ## The sponge is absorption alone

With padding recognised as absorption of `[1]`, the whole sponge is one
`absorb` over the data chunks followed by the padding chunk.  No separate
finalization step exists in the specification either — it was only ever written
that way. -/

theorem absorb_append
    (constants : Constants) (left right : List RateChunk) (state : Values) :
    absorb constants (left ++ right) state
      = absorb constants right (absorb constants left state) := by
  induction left generalizing state with
  | nil => rfl
  | cons chunk rest hypothesis =>
      simp only [List.cons_append, absorb_cons, hypothesis]

/-- **The sponge's final state is absorption over the chunks plus `[1]`.** -/
theorem spongeFinal_eq_absorb_padding
    (constants : Constants) (chunks : List RateChunk) :
    spongeFinal constants chunks
      = absorb constants (chunks ++ [paddingChunk]) initialSpongeState := by
  rw [absorb_append]
  unfold spongeFinal
  rw [absorb_cons, absorb_nil, paddingChunk_absorbs]

/-- The digest is therefore the first `digestLength` lanes of a pure
absorption. -/
theorem digest_eq_absorb_padding
    (constants : Constants) (chunks : List RateChunk)
    (index : Fin digestLength) :
    digest constants chunks index
      = absorb constants (chunks ++ [paddingChunk]) initialSpongeState
          ⟨index.val, by
            have := index.isLt; simp only [digestLength, width] at *; omega⟩ := by
  unfold digest
  rw [spongeFinal_eq_absorb_padding]

/-! ## The chain is the sponge's absorption

`chainValues` tracks the state a call *enters* on; `absorbAt` tracks the state
after that call's permutation.  They differ by exactly one permutation, which is
what `chainValues_permuted` says, and `absorbAt` is `absorb`'s recursion in the
same indexing `chainValues` uses. -/

def absorbAt (constants : Constants) (chunkAt : Nat → List Nat) : Nat → Values
  | 0 => referencePermutation constants (absorbChunk (chunkAt 0) (fun _ => 0))
  | previous + 1 =>
      referencePermutation constants
        (absorbChunk (chunkAt (previous + 1))
          (absorbAt constants chunkAt previous))

theorem chainValues_permuted
    (constants : Constants) (chunkAt : Nat → List Nat) (call : Nat) :
    referencePermutation constants (chainValues constants chunkAt call)
      = absorbAt constants chunkAt call := by
  induction call with
  | zero => rfl
  | succ previous hypothesis =>
      simp only [chainValues, absorbAt, hypothesis]

/-- **Each call's outputs are the sponge state after that call.**  Composing
`spongeChain` with the bridge: satisfying the row program puts the sponge's own
absorbed-and-permuted state on every call's output ports. -/
theorem spongeChain_is_absorption
    (spongeLayout : SpongeLayout) (constants : Constants)
    (chunkLength : Nat → Nat) (calls : Nat) (z : Nat → Nat)
    (chunkAt : Nat → List Nat)
    (residues : ∀ column, z column < goldilocksP)
    (constantWire : z 0 = 1)
    (satisfied : Satisfies (spongeProgram spongeLayout constants chunkLength calls) z)
    (chunkLengthAgrees : ∀ call, chunkLength call = (chunkAt call).length)
    (chunkAgrees : ∀ call, ∀ lane : Fin width, ∀ value,
      (chunkAt call)[lane.val]? = some value
        → z (spongeLayout.chunkColumn call lane) = value)
    (call : Nat) (inRange : call < calls) (lane : Fin width) :
    z ((spongeLayout.call call).outputPort lane)
      = absorbAt constants chunkAt call lane := by
  rw [spongeChain spongeLayout constants chunkLength calls z chunkAt residues
    constantWire satisfied chunkLengthAgrees chunkAgrees call inRange lane,
    chainValues_permuted]


/-! ## The indexing bridge

`absorbAt` indexes chunks by `Nat`; `absorb` consumes a `List RateChunk`.  They
are the same recursion, and `chunkList` is the conversion. -/

def chunkList (chunkAt : Nat → List Nat)
    (bounded : ∀ index, (chunkAt index).length ≤ rate) (count : Nat) :
    List RateChunk :=
  (List.range count).map (fun index => ⟨chunkAt index, bounded index⟩)

theorem chunkList_succ
    (chunkAt : Nat → List Nat)
    (bounded : ∀ index, (chunkAt index).length ≤ rate) (count : Nat) :
    chunkList chunkAt bounded (count + 1)
      = chunkList chunkAt bounded count ++ [⟨chunkAt count, bounded count⟩] := by
  unfold chunkList
  rw [List.range_succ, List.map_append]
  rfl

/-- **The two indexings agree.**  `absorbAt` at `count` is `absorb` over the
first `count + 1` chunks. -/
theorem absorbAt_eq_absorb
    (constants : Constants) (chunkAt : Nat → List Nat)
    (bounded : ∀ index, (chunkAt index).length ≤ rate) (count : Nat) :
    absorbAt constants chunkAt count
      = absorb constants (chunkList chunkAt bounded (count + 1))
          initialSpongeState := by
  induction count with
  | zero =>
      unfold absorbAt chunkList
      simp only [List.range_succ, List.range_zero, List.nil_append,
        List.map_cons, List.map_nil]
      rw [absorb_cons, absorb_nil]
      rfl
  | succ previous hypothesis =>
      rw [chunkList_succ, absorb_append, absorb_cons, absorb_nil, ← hypothesis]
      rfl

/-- **Sponge-prefix soundness.**  Satisfying the row program forces a selected
call's output ports to the specification's absorbed state after the same
prefix.  A fixed-23 wrapper must still select call six, bind the final chunk to
`[1]`, and project lanes zero through three as the digest. -/
theorem spongeProgram_computes_digest
    (spongeLayout : SpongeLayout) (constants : Constants)
    (chunkLength : Nat → Nat) (calls : Nat) (z : Nat → Nat)
    (chunkAt : Nat → List Nat)
    (bounded : ∀ index, (chunkAt index).length ≤ rate)
    (residues : ∀ column, z column < goldilocksP)
    (constantWire : z 0 = 1)
    (satisfied : Satisfies (spongeProgram spongeLayout constants chunkLength calls) z)
    (chunkLengthAgrees : ∀ call, chunkLength call = (chunkAt call).length)
    (chunkAgrees : ∀ call, ∀ lane : Fin width, ∀ value,
      (chunkAt call)[lane.val]? = some value
        → z (spongeLayout.chunkColumn call lane) = value)
    (final : Nat) (inRange : final < calls) (lane : Fin width) :
    z ((spongeLayout.call final).outputPort lane)
      = absorb constants (chunkList chunkAt bounded (final + 1))
          initialSpongeState lane := by
  rw [spongeChain_is_absorption spongeLayout constants chunkLength calls z
    chunkAt residues constantWire satisfied chunkLengthAgrees chunkAgrees final
    inRange lane, absorbAt_eq_absorb constants chunkAt bounded final]


/-! ## The sponge-core cost

Every call allocates both its 344 S-box frame columns and its eight bound output
ports.  The output ports of all nonfinal calls are carried into the next call;
the final eight are also internal until a typed hash wrapper exports four digest
lanes.  Counting only S-box frames would therefore undercount this core by eight
columns per call.

Chunk coordinates are not allocated here: the fixed-23 wrapper owns their exact
mapping to its 23 visible inputs and the shared padding constant. -/

/-- Columns allocated internally by one permutation call inside the sponge. -/
def spongeCallTemporaryColumns (layout : Layout) : List Nat :=
  auxiliaryColumns layout ++
    List.ofFn (fun lane : Fin width => layout.outputPort lane)

theorem spongeCallTemporaryColumns_length (layout : Layout) :
    (spongeCallTemporaryColumns layout).length = 352 := by
  unfold spongeCallTemporaryColumns
  rw [List.length_append, auxiliaryColumns_length_eq]
  simp [width]

/-- Complete internal allocation list of the generic sponge core. -/
def spongeTemporaryColumns (spongeLayout : SpongeLayout) (calls : Nat) :
    List Nat :=
  (List.range calls).flatMap
    (fun call => spongeCallTemporaryColumns (spongeLayout.call call))

theorem spongeTemporaryColumns_length
    (spongeLayout : SpongeLayout) (calls : Nat) :
    (spongeTemporaryColumns spongeLayout calls).length = calls * 352 := by
  unfold spongeTemporaryColumns
  rw [length_flatMap_uniform _ _ 352
    (by intro call; exact spongeCallTemporaryColumns_length _),
    List.length_range]

/-- Cost of the unwrapped sponge core.  All eight per-call output ports remain
internal at this layer. -/
def spongeCost (spongeLayout : SpongeLayout) (calls : Nat) : Typed.Cost :=
  ⟨calls * 352, 0, 0, (spongeTemporaryColumns spongeLayout calls).length⟩

theorem spongeCost_rows_eq_program
    (spongeLayout : SpongeLayout) (constants : Constants)
    (chunkLength : Nat → Nat) (calls : Nat) :
    (spongeCost spongeLayout calls).recurringRows
      = (spongeProgram spongeLayout constants chunkLength calls).length := by
  rw [spongeProgram_length]
  rfl

/-- **The fixed-23 unwrapped core cost.**  Seven calls at 352 rows and 352
internal columns: 2,464 rows over 2,464 auxiliary columns.  A complete activated
hash recipe has a separate footprint because it must export four digest lanes
while remaining satisfiable when inactive. -/
theorem sponge23Cost :
    spongeCost canonicalSpongeLayout 7 = ⟨2464, 0, 0, 2464⟩ := by
  unfold spongeCost
  rw [spongeTemporaryColumns_length]

/-- Per-call internal columns include the eight carried output ports in
addition to the 344 S-box frames. -/
theorem spongeCost_auxiliary_per_call
    (spongeLayout : SpongeLayout) (calls : Nat) :
    (spongeCost spongeLayout calls).auxiliaryColumns = calls * 352 := by
  unfold spongeCost
  exact spongeTemporaryColumns_length spongeLayout calls

/-- **Assembled sponge conservation.**  Every column of every emitted row
belongs to the call that emitted it, or is that call's carried predecessor
port. -/
theorem spongeProgram_conservation
    (spongeLayout : SpongeLayout) (chunkLength : Nat → Nat)
    (constants : Constants) (calls : Nat) (row : Row)
    (member : row ∈ spongeProgram spongeLayout constants chunkLength calls)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    ∃ call, call < calls ∧ SpongeColumn spongeLayout call column := by
  unfold spongeProgram at member
  rcases List.mem_flatMap.1 member with ⟨call, inRange, rowMember⟩
  unfold Poseidon2Normalized.normalizedCanonicalProgramFrom
    Poseidon2Normalized.normalizeProgram at rowMember
  rcases List.mem_map.1 rowMember with ⟨source, sourceMember, rfl⟩
  refine ⟨call, List.mem_range.1 inRange,
    spongeCall_conservation spongeLayout chunkLength constants call source
      sourceMember column ?_⟩
  rcases mentioned with inA | inB | inC
  · exact Or.inl ((Poseidon2Normalized.mentions_normalizeRow source column).1 inA)
  · exact Or.inr (Or.inl
      ((Poseidon2Normalized.mentions_normalizeRow source column).2.1 inB))
  · exact Or.inr (Or.inr
      ((Poseidon2Normalized.mentions_normalizeRow source column).2.2 inC))


/-! ## Sponge ownership

Positional across calls: the emitted program is the image of a call-indexed
receipt list, that list repeats nothing, and the lengths agree.  So position
`i` is emitted by receipt `i` and by no other, across all seven calls. -/

def spongeOwners (calls : Nat) : List (Nat × Poseidon2Ownership.RowOwner) :=
  (List.range calls).flatMap
    (fun call => Poseidon2Ownership.allOwners.map (fun owner => (call, owner)))

theorem spongeOwners_length (calls : Nat) :
    (spongeOwners calls).length = calls * 352 := by
  unfold spongeOwners
  rw [length_flatMap_uniform _ _ 352
    (by intro call; rw [List.length_map, Poseidon2Ownership.allOwners_length]),
    List.length_range]

set_option maxRecDepth 4000 in
/-- **The emitted sponge program is the receipt list's image.** -/
theorem spongeProgram_eq_map_owners
    (spongeLayout : SpongeLayout) (constants : Constants)
    (chunkLength : Nat → Nat) (calls : Nat) :
    spongeProgram spongeLayout constants chunkLength calls
      = (spongeOwners calls).map (fun entry =>
          Poseidon2Normalized.normalizeRow
            (Poseidon2Ownership.ownedRowFrom (spongeLayout.call entry.1)
              (entryOf spongeLayout chunkLength entry.1) constants entry.2)) := by
  unfold spongeProgram spongeOwners
    Poseidon2Normalized.normalizedCanonicalProgramFrom
    Poseidon2Normalized.normalizeProgram
  rw [List.map_flatMap]
  congr 1

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
