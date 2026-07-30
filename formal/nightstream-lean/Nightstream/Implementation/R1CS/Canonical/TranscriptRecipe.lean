import Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex
import Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
import Nightstream.Implementation.R1CS.Canonical.Poseidon2HonestFrom
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: the emitted row program for the Fiat–Shamir duplex.

Owns: the overwrite entry combination, the per-round program, its derived row
count, per-round soundness, the chain that carries round `r`'s output ports into
round `r + 1`'s entry, honest completeness, conservation, and the cost.

Does not own: the permutation (that is `Poseidon2Schedule`), the value-level
duplex semantics (`Poseidon2Duplex`), or what a caller absorbs.

## Overwrite absorption emits no row, and the entry is where the mode lives

A duplex round is one permutation applied to an entry state.  In this
construction the entry lane is either

- the **absorbed column alone**, when the round writes that lane, or
- the previous round's **output port alone**, when it does not.

Neither is a row.  That is the same "absorption is free" fact the sponge already
had, but the combination is different: the sponge's add mode carries
`[(chunk, 1), (previousPort, 1)]` where this carries one term or the other.

`TranscriptModeBoundary` proves those denote different values from any carried
state.  Here the difference is visible in the emitted syntax, which is where an
encoder can actually get it wrong.

## The static cursor

`Poseidon2Duplex`'s cursor is a runtime value.  In a row program it is not: the
absorb schedule is fixed at encoding time, so which lane each round overwrites
is a parameter of the program rather than something the rows compute.  That is
why no row implements the `if absorbed >= RATE then permute` guard — the encoder
has already placed the permutations where the guard would have fired.

`absorbedAt` is that schedule.  A caller that supplies one inconsistent with the
duplex's cursor arithmetic gets a program that is internally sound and models
something else, so the schedule is the caller's obligation and is named here
rather than assumed away.

## Scope: all ten section-2 items

Constructive row program, derived count, row ownership, column ownership,
conservation, soundness, honest completeness, `Typed.Cost`, fail-closed axiom
guard, spec and ledger.

Honest completeness and conservation were open when this module was first
written and are now proved.  Both came from `Poseidon2HonestFrom` and
`Poseidon2Conservation`'s carried-entry lemmas, which were already layout-generic
— the overwrite entry needed no new machinery, only to be handed to them.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
open Nightstream.Implementation.R1CS.Canonical.Poseidon2HonestFrom

private theorem sum_const {α : Type} (items : List α) (value : Nat) :
    (items.map (fun _ => value)).sum = items.length * value := by
  induction items with
  | nil => simp
  | cons item rest inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis, Nat.succ_mul]
      omega

/-- Which lane each round overwrites, and from which column.  Fixed at encoding
time: the cursor is static in a row program. -/
abbrev Schedule := Nat → Fin width → Option Nat

/-- **The overwrite entry.**  One term, never two: the absorbed column when the
round writes that lane, otherwise the previous round's output port.  Round zero
carries the empty combination on unwritten lanes, which is the all-zero initial
state. -/
def entryAt (layouts : Nat → Layout) (absorbedAt : Schedule) (round : Nat) :
    State :=
  fun lane =>
    match absorbedAt round lane with
    | some column => [(column, 1)]
    | none =>
        match round with
        | 0 => []
        | previous + 1 => [((layouts previous).outputPort lane, 1)]

/-- **The emitted transcript program.**  One permutation per round. -/
def transcriptRows (layouts : Nat → Layout) (absorbedAt : Schedule)
    (constants : Constants) (rounds : Nat) : List Row :=
  (List.range rounds).flatMap
    (fun round =>
      normalizedCanonicalProgramFrom (layouts round)
        (entryAt layouts absorbedAt round) constants)

/-! ## The derived row count

Each round is one permutation program, and `canonicalProgramFrom_length` already
records that a carried entry costs no extra row.  So the absorption really is
free here too, and the count is rounds times the permutation. -/

theorem transcriptRows_length (layouts : Nat → Layout) (absorbedAt : Schedule)
    (constants : Constants) (rounds : Nat) :
    (transcriptRows layouts absorbedAt constants rounds).length
      = (List.range rounds).length * 352 := by
  unfold transcriptRows
  rw [List.length_flatMap]
  have each : (List.range rounds).map
        (fun round => (normalizedCanonicalProgramFrom (layouts round)
          (entryAt layouts absorbedAt round) constants).length)
      = (List.range rounds).map (fun _ => 352) :=
    List.map_congr_left (fun round _ =>
      normalizedCanonicalProgramFrom_length (layouts round)
        (entryAt layouts absorbedAt round) constants)
  rw [each, sum_const, List.length_range]

/-- **352 rows per round**, once the range is evaluated. -/
theorem transcriptRows_length_eq (layouts : Nat → Layout)
    (absorbedAt : Schedule) (constants : Constants) (rounds : Nat) :
    (transcriptRows layouts absorbedAt constants rounds).length = rounds * 352 := by
  rw [transcriptRows_length, List.length_range]

/-! ## Soundness

Satisfaction restricts to each round, and each round's 352 rows put the
reference image of its entry on its output ports. -/

theorem satisfies_round
    (layouts : Nat → Layout) (absorbedAt : Schedule) (constants : Constants)
    (rounds : Nat) (z : Nat → Nat)
    (satisfied : Satisfies (transcriptRows layouts absorbedAt constants rounds) z)
    (round : Nat) (inRange : round < rounds) :
    Satisfies (normalizedCanonicalProgramFrom (layouts round)
      (entryAt layouts absorbedAt round) constants) z := by
  intro row member
  exact satisfied row
    (List.mem_flatMap.2 ⟨round, List.mem_range.2 inRange, member⟩)

/-- **Each round computes the permutation of its entry.** -/
theorem round_computes_reference
    (layouts : Nat → Layout) (absorbedAt : Schedule) (constants : Constants)
    (rounds : Nat) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP) (constantWire : z 0 = 1)
    (satisfied : Satisfies (transcriptRows layouts absorbedAt constants rounds) z)
    (round : Nat) (inRange : round < rounds)
    (entryValues : Values)
    (entryAgrees : ∀ source : Fin width,
      lcEval z (entryAt layouts absorbedAt round source) = entryValues source)
    (lane : Fin width) :
    z ((layouts round).outputPort lane)
      = referencePermutation constants entryValues lane :=
  normalizedCanonicalProgramFrom_computes_reference (layouts round) constants z
    (entryAt layouts absorbedAt round) entryValues residues constantWire
    entryAgrees
    (satisfies_round layouts absorbedAt constants rounds z satisfied round inRange)
    lane

/-! ## The chain

What makes a sequence of permutations a *duplex*: an unwritten lane at round
`r + 1` carries exactly round `r`'s output. -/

/-- **An unwritten lane reads the previous round's output port**, and nothing
else — the overwrite mode, in the emitted syntax. -/
theorem entry_carried
    (layouts : Nat → Layout) (absorbedAt : Schedule)
    (previous : Nat) (lane : Fin width)
    (notWritten : absorbedAt (previous + 1) lane = none) :
    entryAt layouts absorbedAt (previous + 1) lane
      = [((layouts previous).outputPort lane, 1)] := by
  unfold entryAt
  rw [notWritten]

/-- **A written lane reads the absorbed column alone**, with no carried term.
This is precisely where add-mode absorption would have differed: it would carry
the previous port as a second term. -/
theorem entry_overwritten
    (layouts : Nat → Layout) (absorbedAt : Schedule)
    (round : Nat) (lane : Fin width) (column : Nat)
    (written : absorbedAt round lane = some column) :
    entryAt layouts absorbedAt round lane = [(column, 1)] := by
  unfold entryAt
  rw [written]

/-- **Round zero's unwritten lanes are the all-zero initial state.** -/
theorem entry_initial
    (layouts : Nat → Layout) (absorbedAt : Schedule) (lane : Fin width)
    (notWritten : absorbedAt 0 lane = none) :
    entryAt layouts absorbedAt 0 lane = [] := by
  unfold entryAt
  rw [notWritten]

/-- **The carried value is the previous round's output value.**  Combined with
`round_computes_reference` this is the duplex chain: what round `r + 1` absorbs
on an unwritten lane is what round `r` computed there. -/
theorem chain_value
    (layouts : Nat → Layout) (absorbedAt : Schedule) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP)
    (previous : Nat) (lane : Fin width)
    (notWritten : absorbedAt (previous + 1) lane = none) :
    lcEval z (entryAt layouts absorbedAt (previous + 1) lane)
      = z ((layouts previous).outputPort lane) := by
  rw [entry_carried layouts absorbedAt previous lane notWritten]
  simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul]
  exact Nat.mod_eq_of_lt (residues _)

/-- **An absorbed value enters unchanged.** -/
theorem absorbed_value
    (layouts : Nat → Layout) (absorbedAt : Schedule) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP)
    (round : Nat) (lane : Fin width) (column : Nat)
    (written : absorbedAt round lane = some column) :
    lcEval z (entryAt layouts absorbedAt round lane) = z column := by
  rw [entry_overwritten layouts absorbedAt round lane column written]
  simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul]
  exact Nat.mod_eq_of_lt (residues _)


/-! ## Honest completeness

One round's honest witness is `Poseidon2HonestFrom`'s: an assignment agreeing
with the reference on that round's entry, S-box columns and output ports.  The
transcript's is the same, round by round.

`RoundHonest` bundles the three agreements rather than threading them, and each
field is exactly the hypothesis `honest_satisfies_normalizedFrom` consumes — so
nothing is moved to a premise a real consumer would not construct.  The consumer
is a prover running the duplex: `chain_value` and `round_computes_reference`
already say its round `r + 1` entry values are round `r`'s outputs. -/

/-- What an honest assignment must satisfy at one round. -/
structure RoundHonest (layouts : Nat → Layout) (absorbedAt : Schedule)
    (constants : Constants) (z : Nat → Nat) (round : Nat)
    (values : Values) : Prop where
  entryAgrees : ∀ lane : Fin width,
    lcEval z (entryAt layouts absorbedAt round lane) = values lane
  sboxAgrees : ∀ (index : Fin sboxCount) (slot : Fin columnsPerSbox),
    z (sboxColumn (layouts round) index slot)
      = chainSlot (sboxInputValue constants values index.val) slot.val
  outputAgrees : ∀ lane : Fin width,
    z ((layouts round).outputPort lane)
      = referencePermutation constants values lane

/-- **An honest duplex run satisfies the transcript program.** -/
theorem transcriptRows_honest
    (layouts : Nat → Layout) (absorbedAt : Schedule) (constants : Constants)
    (rounds : Nat) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP) (constantWire : z 0 = 1)
    (values : Nat → Values)
    (honest : ∀ round, round < rounds →
      RoundHonest layouts absorbedAt constants z round (values round)) :
    Satisfies (transcriptRows layouts absorbedAt constants rounds) z := by
  intro row member
  unfold transcriptRows at member
  rcases List.mem_flatMap.1 member with ⟨round, roundMember, rowMember⟩
  have inRange : round < rounds := List.mem_range.1 roundMember
  have step := honest round inRange
  exact honest_satisfies_normalizedFrom (layouts round)
    (entryAt layouts absorbedAt round) constants (values round) z residues
    constantWire step.entryAgrees step.sboxAgrees step.outputAgrees
    row rowMember

/-! ## Conservation

Every column of every emitted row belongs to one round: the constant wire, one
of that round's entry combinations, one of its S-box columns, or one of its
output ports.  The placement introduces nothing, and normalization introduces no
column either — `mentions_normalizeRow` gives exactly that direction. -/

/-- **Every column belongs to some round.** -/
theorem transcriptRows_conservation
    (layouts : Nat → Layout) (absorbedAt : Schedule) (constants : Constants)
    (rounds : Nat) (row : Row)
    (member : row ∈ transcriptRows layouts absorbedAt constants rounds)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    ∃ round, round < rounds ∧
      (column = 0
        ∨ (∃ source : Fin width,
            Mentions (entryAt layouts absorbedAt round source) column)
        ∨ (∃ index, index < sboxCount
            ∧ column = sboxOutput (layouts round) index)
        ∨ (∃ slot : Fin columnsPerSbox, ∃ index : Fin sboxCount,
            column = sboxColumn (layouts round) index slot)
        ∨ (∃ lane : Fin width, column = (layouts round).outputPort lane)) := by
  unfold transcriptRows at member
  rcases List.mem_flatMap.1 member with ⟨round, roundMember, rowMember⟩
  refine ⟨round, List.mem_range.1 roundMember, ?_⟩
  rcases List.mem_map.1 rowMember with ⟨raw, rawMember, rfl⟩
  have rawMentioned : Mentions raw.a column ∨ Mentions raw.b column
      ∨ Mentions raw.c column := by
    rcases mentioned with inA | inB | inC
    · exact Or.inl ((mentions_normalizeRow raw column).1 inA)
    · exact Or.inr (Or.inl ((mentions_normalizeRow raw column).2.1 inB))
    · exact Or.inr (Or.inr ((mentions_normalizeRow raw column).2.2 inC))
  unfold canonicalProgramFrom permutationProgram at rawMember
  rcases List.mem_append.1 rawMember with inSbox | inBinding
  · rcases List.mem_flatMap.1 inSbox with ⟨index, _, sboxMember⟩
    simp only [sboxRows, List.mem_cons, List.not_mem_nil, or_false]
      at sboxMember
    have scheduled : ∀ target,
        Mentions (scheduleOfFrom (layouts round)
          (entryAt layouts absorbedAt round) constants index) target →
        column = target →
        column = 0
          ∨ (∃ source : Fin width,
              Mentions (entryAt layouts absorbedAt round source) column)
          ∨ (∃ other, other < sboxCount
              ∧ column = sboxOutput (layouts round) other)
          ∨ (∃ slot : Fin columnsPerSbox, ∃ other : Fin sboxCount,
              column = sboxColumn (layouts round) other slot)
          ∨ (∃ lane : Fin width, column = (layouts round).outputPort lane) := by
      intro target inSchedule same
      subst same
      rcases Poseidon2Conservation.scheduleOfFrom_columns (layouts round)
        (entryAt layouts absorbedAt round) constants index column inSchedule with
        wire | fromEntry | output
      · exact Or.inl wire
      · exact Or.inr (Or.inl fromEntry)
      · exact Or.inr (Or.inr (Or.inl output))
    have frameSlot : ∀ slot : Fin columnsPerSbox,
        column = sboxColumn (layouts round) index slot →
        column = 0
          ∨ (∃ source : Fin width,
              Mentions (entryAt layouts absorbedAt round source) column)
          ∨ (∃ other, other < sboxCount
              ∧ column = sboxOutput (layouts round) other)
          ∨ (∃ s : Fin columnsPerSbox, ∃ other : Fin sboxCount,
              column = sboxColumn (layouts round) other s)
          ∨ (∃ lane : Fin width, column = (layouts round).outputPort lane) :=
      fun slot same => Or.inr (Or.inr (Or.inr (Or.inl ⟨slot, index, same⟩)))
    rcases sboxMember with rfl | rfl | rfl | rfl <;>
      simp only [rowSquare, rowFourth, rowSixth, rowSeventh, frameAt]
        at rawMentioned
    · rcases rawMentioned with a | b | c
      · exact scheduled column a rfl
      · exact scheduled column b rfl
      · exact frameSlot ⟨0, by decide⟩ (by
          simpa only [Mentions, List.map_cons, List.map_nil,
            List.mem_singleton] using c)
    · rcases rawMentioned with a | b | c
      · exact frameSlot ⟨0, by decide⟩ (by
          simpa only [Mentions, List.map_cons, List.map_nil,
            List.mem_singleton] using a)
      · exact frameSlot ⟨0, by decide⟩ (by
          simpa only [Mentions, List.map_cons, List.map_nil,
            List.mem_singleton] using b)
      · exact frameSlot ⟨1, by decide⟩ (by
          simpa only [Mentions, List.map_cons, List.map_nil,
            List.mem_singleton] using c)
    · rcases rawMentioned with a | b | c
      · exact frameSlot ⟨0, by decide⟩ (by
          simpa only [Mentions, List.map_cons, List.map_nil,
            List.mem_singleton] using a)
      · exact frameSlot ⟨1, by decide⟩ (by
          simpa only [Mentions, List.map_cons, List.map_nil,
            List.mem_singleton] using b)
      · exact frameSlot ⟨2, by decide⟩ (by
          simpa only [Mentions, List.map_cons, List.map_nil,
            List.mem_singleton] using c)
    · rcases rawMentioned with a | b | c
      · exact scheduled column a rfl
      · exact frameSlot ⟨2, by decide⟩ (by
          simpa only [Mentions, List.map_cons, List.map_nil,
            List.mem_singleton] using b)
      · exact frameSlot ⟨3, by decide⟩ (by
          simpa only [Mentions, List.map_cons, List.map_nil,
            List.mem_singleton] using c)
  · rcases List.mem_map.1 inBinding with ⟨lane, _, rfl⟩
    simp only [bindRow] at rawMentioned
    rcases rawMentioned with final | wire | port
    · rcases Poseidon2Conservation.terminalState_columns (layouts round)
        halfFullRounds (Nat.le_refl _) lane column final with ⟨index, bound, rfl⟩
      exact Or.inr (Or.inr (Or.inl ⟨index, bound, rfl⟩))
    · exact Or.inl (by
        simpa only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] using wire)
    · exact Or.inr (Or.inr (Or.inr (Or.inr ⟨lane, by
        simpa only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] using port⟩)))

/-! ## Cost -/


/-- **The transcript's cost**, folded over rounds.  One permutation each: 352
rows and 344 auxiliaries, with absorption contributing neither. -/
def transcriptCost (rounds : Nat) : Lowering.Typed.Cost where
  recurringRows := rounds * 352
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := rounds * 344

theorem transcriptCost_rows
    (layouts : Nat → Layout) (absorbedAt : Schedule) (constants : Constants)
    (rounds : Nat) :
    (transcriptRows layouts absorbedAt constants rounds).length
      = (transcriptCost rounds).recurringRows :=
  transcriptRows_length_eq layouts absorbedAt constants rounds

/-! ## The canonical round layout

`transcriptRows` takes `layouts` as a parameter, and the same distinction
`PIDEC-LOWNORM-CANONICAL-ALLOCATION` drew applies here: *which* layout each
round gets is the encoder's choice, and the properties of that choice are
theorems.

`canonicalLayouts` places round `r` at stride `r`, reusing the stride
`Poseidon2Sponge` already proved clears a full column space.  Distinct rounds
then allocate disjoint columns — not by a new argument, but by
`SpongeLayout.WellFormed.auxDisjoint`, which the canonical sponge layout already
satisfies and which is definitionally the same placement.

What a caller still supplies is the **absorb schedule**: which lane each round
overwrites, and from which column.  That is about the data being absorbed and no
layout choice determines it. -/

/-- **The canonical per-round layout.**  Round `r` at stride `r`. -/
def canonicalLayouts : Nat → Layout :=
  fun round =>
    Poseidon2Layout.shiftedLayout (round * Poseidon2Sponge.spongeStride)

/-- It is the canonical sponge's own call layout, so nothing new is introduced
by using it for a transcript. -/
theorem canonicalLayouts_eq_spongeCall (round : Nat) :
    canonicalLayouts round = Poseidon2Sponge.canonicalSpongeLayout.call round :=
  rfl

theorem canonicalLayouts_wellFormed (round : Nat) :
    Poseidon2Layout.WellFormed (canonicalLayouts round) :=
  Poseidon2Layout.shiftedLayout_wellFormed _

/-- **Distinct rounds allocate disjoint columns.**

The transcript's analogue of `KLowNormBatch.canonicalDigits_nodup`: the encoder
picks the layout and proves the collision-freedom that choice buys, rather than
asking a deployment to establish it. -/
theorem canonicalLayouts_disjoint
    (first second : Nat) (distinct : first ≠ second)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox)
    (other : Fin sboxCount) (otherSlot : Fin columnsPerSbox) :
    sboxColumn (canonicalLayouts first) index slot
      ≠ sboxColumn (canonicalLayouts second) other otherSlot :=
  Poseidon2Sponge.canonicalSpongeLayout_wellFormed.auxDisjoint first second
    distinct index slot other otherSlot

/-- **The canonical transcript program.**  Rows and count unchanged; what is
fixed is the layout. -/
def canonicalTranscriptRows (absorbedAt : Schedule) (constants : Constants)
    (rounds : Nat) : List Row :=
  transcriptRows canonicalLayouts absorbedAt constants rounds

theorem canonicalTranscriptRows_length
    (absorbedAt : Schedule) (constants : Constants) (rounds : Nat) :
    (canonicalTranscriptRows absorbedAt constants rounds).length = rounds * 352 :=
  transcriptRows_length_eq canonicalLayouts absorbedAt constants rounds

/-! ## Round separation on every column family

`canonicalLayouts_disjoint` covers S-box columns, which is what the sponge's
cost fold needed.  Row ownership needs more: a row can mention an input port or
an output port, so separating rounds requires separating **every** family.

`shiftedLayout` is explicit — `inputPort lane = base + 1 + lane`,
`outputPort lane = base + 9 + lane`, `sboxColumn index slot = base + 17 +
4·index + slot` — so every column it names lies in `(base, base + 361]`.  Two
rounds at stride 369 therefore share nothing.

This is the ingredient `RECIPE-ROW-OWNERSHIP` needs for this recipe, and it is
stated over the whole space rather than one family so that adding a row shape
later does not reopen it. -/

/-- **Every column a round names lies in that round's window.** -/
theorem canonicalLayouts_column_window (round : Nat) :
    (∀ lane : Fin width,
        round * Poseidon2Sponge.spongeStride < (canonicalLayouts round).inputPort lane
          ∧ (canonicalLayouts round).inputPort lane
              ≤ round * Poseidon2Sponge.spongeStride + canonicalColumnTotal)
      ∧ (∀ lane : Fin width,
        round * Poseidon2Sponge.spongeStride < (canonicalLayouts round).outputPort lane
          ∧ (canonicalLayouts round).outputPort lane
              ≤ round * Poseidon2Sponge.spongeStride + canonicalColumnTotal)
      ∧ (∀ (index : Fin sboxCount) (slot : Fin columnsPerSbox),
        round * Poseidon2Sponge.spongeStride
            < sboxColumn (canonicalLayouts round) index slot
          ∧ sboxColumn (canonicalLayouts round) index slot
              ≤ round * Poseidon2Sponge.spongeStride + canonicalColumnTotal) := by
  refine ⟨fun lane => ?_, fun lane => ?_, fun index slot => ?_⟩
  · have := lane.isLt
    simp only [width] at this
    simp only [canonicalLayouts, Poseidon2Layout.shiftedLayout,
      canonicalColumnTotal, width, sboxCount, externalRounds, partialRounds,
      columnsPerSbox]
    omega
  · have := lane.isLt
    simp only [width] at this
    simp only [canonicalLayouts, Poseidon2Layout.shiftedLayout,
      canonicalColumnTotal, width, sboxCount, externalRounds, partialRounds,
      columnsPerSbox]
    omega
  · have indexLt := index.isLt
    have slotLt := slot.isLt
    simp only [sboxCount, externalRounds, width, partialRounds,
      columnsPerSbox] at indexLt slotLt
    simp only [canonicalLayouts, Poseidon2Layout.shiftedLayout, sboxColumn,
      canonicalColumnTotal, width, sboxCount, externalRounds, partialRounds,
      columnsPerSbox]
    omega

/-- **Distinct rounds' windows are disjoint.**

The stride clears a full column space, so a column cannot lie in two rounds'
windows.  Combined with `canonicalLayouts_column_window` this separates rounds
on every family at once. -/
theorem canonicalLayouts_windows_disjoint
    (first second column : Nat) (distinct : first ≠ second)
    (inFirst : first * Poseidon2Sponge.spongeStride < column
      ∧ column ≤ first * Poseidon2Sponge.spongeStride + canonicalColumnTotal)
    (inSecond : second * Poseidon2Sponge.spongeStride < column
      ∧ column ≤ second * Poseidon2Sponge.spongeStride + canonicalColumnTotal) :
    False := by
  have clears : canonicalColumnTotal ≤ Poseidon2Sponge.spongeStride := by
    simp only [canonicalColumnTotal, Poseidon2Sponge.spongeStride, width,
      sboxCount, externalRounds, partialRounds, columnsPerSbox]
    omega
  rcases Nat.lt_or_ge first second with below | above
  · have gap : first * Poseidon2Sponge.spongeStride + Poseidon2Sponge.spongeStride
        ≤ second * Poseidon2Sponge.spongeStride := by
      calc first * Poseidon2Sponge.spongeStride + Poseidon2Sponge.spongeStride
          = (first + 1) * Poseidon2Sponge.spongeStride :=
            (Nat.succ_mul first _).symm
        _ ≤ second * Poseidon2Sponge.spongeStride :=
            Nat.mul_le_mul_right _ below
    omega
  · have secondBelow : second < first := by omega
    have gap : second * Poseidon2Sponge.spongeStride + Poseidon2Sponge.spongeStride
        ≤ first * Poseidon2Sponge.spongeStride := by
      calc second * Poseidon2Sponge.spongeStride + Poseidon2Sponge.spongeStride
          = (second + 1) * Poseidon2Sponge.spongeStride :=
            (Nat.succ_mul second _).symm
        _ ≤ first * Poseidon2Sponge.spongeStride :=
            Nat.mul_le_mul_right _ secondBelow
    omega

/-- **No column is named by two rounds.**

The full-family separation, which is what row ownership for this recipe
consumes: a row mentioning any port or auxiliary of round `r` cannot also be a
row of round `r'`. -/
theorem canonicalLayouts_no_shared_column
    (first second : Nat) (distinct : first ≠ second)
    (firstLane secondLane : Fin width)
    (sameOutput : (canonicalLayouts first).outputPort firstLane
      = (canonicalLayouts second).outputPort secondLane) :
    False := by
  have firstWindow := (canonicalLayouts_column_window first).2.1 firstLane
  have secondWindow := (canonicalLayouts_column_window second).2.1 secondLane
  rw [sameOutput] at firstWindow
  exact canonicalLayouts_windows_disjoint first second _ distinct firstWindow
    secondWindow

/-! ## Row ownership

Cycle 359 recorded that a row of round `r` mentions columns of window `r - 1`
too — the entry reads the previous round's output ports — and concluded the
argument needed a two-window analysis.

That was over-complicated.  **Only the target column needs tracking.** Every
emitted row's `c` field names a column in its *own* window:

| row | target |
|---|---|
| `rowSquare` | `sboxColumn (layouts r) index ⟨0,_⟩` |
| `rowFourth` | `sboxColumn (layouts r) index ⟨1,_⟩` |
| `rowSixth` | `sboxColumn (layouts r) index ⟨2,_⟩` |
| `rowSeventh` | `sboxColumn (layouts r) index ⟨3,_⟩` |
| `bindRow` | `(layouts r).outputPort lane` |

The entry appears only in `a` and `b`.  So a shared row would put one column in
two windows, and `canonicalLayouts_windows_disjoint` closes it — no two-window
case analysis, no crossed case.

This is the rule `RECIPE-ROW-OWNERSHIP-STATUS` predicts: a round **allocates**
its whole window, and every row of the round exposes that window in its target.
`KRecomposition` fails item 3 because its two rows expose unrelated halves;
here the many rows all expose the same round. -/

/-- **Every raw row of a round targets that round's window.** -/
theorem rawRow_target_in_window
    (absorbedAt : Schedule) (constants : Constants) (round : Nat)
    (row : Row)
    (member : row ∈ canonicalProgramFrom (canonicalLayouts round)
      (entryAt canonicalLayouts absorbedAt round) constants)
    (column : Nat) (mentioned : Mentions row.c column) :
    round * Poseidon2Sponge.spongeStride < column
      ∧ column ≤ round * Poseidon2Sponge.spongeStride + canonicalColumnTotal := by
  unfold canonicalProgramFrom permutationProgram at member
  rcases List.mem_append.1 member with inSbox | inBinding
  · rcases List.mem_flatMap.1 inSbox with ⟨index, _, sboxMember⟩
    simp only [sboxRows, List.mem_cons, List.not_mem_nil, or_false]
      at sboxMember
    have windows := (canonicalLayouts_column_window round).2.2
    rcases sboxMember with rfl | rfl | rfl | rfl <;>
      simp only [rowSquare, rowFourth, rowSixth, rowSeventh, frameAt,
        Mentions, List.map_cons, List.map_nil, List.mem_singleton] at mentioned
    · exact mentioned ▸ windows index ⟨0, by decide⟩
    · exact mentioned ▸ windows index ⟨1, by decide⟩
    · exact mentioned ▸ windows index ⟨2, by decide⟩
    · exact mentioned ▸ windows index ⟨3, by decide⟩
  · rcases List.mem_map.1 inBinding with ⟨lane, _, rfl⟩
    simp only [bindRow, Mentions, List.map_cons, List.map_nil,
      List.mem_singleton] at mentioned
    exact mentioned ▸ (canonicalLayouts_column_window round).2.1 lane

/-- **Every emitted row of a round targets that round's window.**

Normalization introduces no column, so the raw statement carries through. -/
theorem row_target_in_window
    (absorbedAt : Schedule) (constants : Constants) (round : Nat)
    (row : Row)
    (member : row ∈ normalizedCanonicalProgramFrom (canonicalLayouts round)
      (entryAt canonicalLayouts absorbedAt round) constants)
    (column : Nat) (mentioned : Mentions row.c column) :
    round * Poseidon2Sponge.spongeStride < column
      ∧ column ≤ round * Poseidon2Sponge.spongeStride + canonicalColumnTotal := by
  rcases List.mem_map.1 member with ⟨raw, rawMember, rfl⟩
  exact rawRow_target_in_window absorbedAt constants round raw rawMember column
    ((mentions_normalizeRow raw column).2.2 mentioned)

/-- **A row emitted by two rounds forces the rounds equal.**

Section 2 item 3 for this recipe, given that the row targets something: the
target lies in both rounds' windows, and the windows are disjoint. -/
theorem transcriptRows_owner_unique
    (absorbedAt : Schedule) (constants : Constants)
    (first second : Nat) (row : Row) (column : Nat)
    (targets : Mentions row.c column)
    (inFirst : row ∈ normalizedCanonicalProgramFrom (canonicalLayouts first)
      (entryAt canonicalLayouts absorbedAt first) constants)
    (inSecond : row ∈ normalizedCanonicalProgramFrom (canonicalLayouts second)
      (entryAt canonicalLayouts absorbedAt second) constants) :
    first = second := by
  by_cases same : first = second
  · exact same
  · exact (canonicalLayouts_windows_disjoint first second column same
      (row_target_in_window absorbedAt constants first row inFirst column targets)
      (row_target_in_window absorbedAt constants second row inSecond column
        targets)).elim

/-! ## The declared allocation

The cost carries `rounds · 344` auxiliary columns.  Until now no list of those
columns existed, which is the same count-without-columns defect
`CANONICAL-PROGRAM-SELECTION-ALLOCATION` closed for selections — and the
assembly briefly papered over it with a placeholder of the right length.

`transcriptColumns` is the list: each round's 344 S-box columns, at that round's
stride. -/

/-- **The declared allocation.**  Round `r` owns `r · stride + 17 … + 360`. -/
def transcriptColumns (rounds : Nat) : List Nat :=
  (List.range rounds).flatMap
    (fun round =>
      (List.range 344).map
        (fun offset => round * Poseidon2Sponge.spongeStride + 17 + offset))

theorem transcriptColumns_length (rounds : Nat) :
    (transcriptColumns rounds).length = rounds * 344 := by
  unfold transcriptColumns
  rw [List.length_flatMap]
  have each : (List.range rounds).map
      (fun round => ((List.range 344).map
        (fun offset => round * Poseidon2Sponge.spongeStride + 17 + offset)).length)
      = (List.range rounds).map (fun _ => 344) :=
    List.map_congr_left (fun round _ => by
      rw [List.length_map, List.length_range])
  rw [each, sum_const, List.length_range]

theorem transcriptColumns_length_eq (rounds : Nat) :
    (transcriptColumns rounds).length
      = (transcriptCost rounds).auxiliaryColumns :=
  transcriptColumns_length rounds

/-! ### The declared allocation is the canonical layout's, and only its

`transcriptColumns` takes **only `rounds`**.  It never mentions `layouts`, while
`transcriptRows` does.  So the declared allocation cannot describe the emitted
program in general — it describes it at `canonicalLayouts` and at nothing else,
and that had never been stated.

`sboxCount = 8 · 8 + 22 = 86` and `columnsPerSbox = 4`, so a round owns
`86 · 4 = 344` S-box columns starting at `auxBase = base + 17`.  That is exactly
the declared range, which is why the count matched while the connection was
missing. -/

/-- **The declared allocation is exactly the canonical layout's S-box
columns.** -/
theorem transcriptColumns_eq_canonical_sbox (rounds : Nat) (column : Nat) :
    column ∈ transcriptColumns rounds
      ↔ ∃ round, round < rounds ∧
          ∃ (index : Fin sboxCount) (slot : Fin columnsPerSbox),
            column = sboxColumn (canonicalLayouts round) index slot := by
  unfold transcriptColumns
  constructor
  · intro member
    rcases List.mem_flatMap.1 member with ⟨round, roundMember, inRound⟩
    rcases List.mem_map.1 inRound with ⟨offset, offsetMember, rfl⟩
    have offsetLt : offset < 344 := List.mem_range.1 offsetMember
    refine ⟨round, List.mem_range.1 roundMember,
      ⟨offset / 4, by simp only [sboxCount, externalRounds, width,
        partialRounds]; omega⟩,
      ⟨offset % 4, by simp only [columnsPerSbox]; omega⟩, ?_⟩
    show _ = (Poseidon2Layout.shiftedLayout
      (round * Poseidon2Sponge.spongeStride)).auxBase + _ * _ + _
    simp only [Poseidon2Layout.shiftedLayout, columnsPerSbox]
    omega
  · rintro ⟨round, roundLt, index, slot, rfl⟩
    have indexLt : index.val < 86 := by
      have := index.isLt
      simp only [sboxCount, externalRounds, width, partialRounds] at this
      omega
    have slotLt : slot.val < 4 := by
      have := slot.isLt
      simp only [columnsPerSbox] at this
      omega
    refine List.mem_flatMap.2 ⟨round, List.mem_range.2 roundLt,
      List.mem_map.2 ⟨4 * index.val + slot.val,
        List.mem_range.2 (by omega), ?_⟩⟩
    show _ = (Poseidon2Layout.shiftedLayout
      (round * Poseidon2Sponge.spongeStride)).auxBase + _ * _ + _
    simp only [Poseidon2Layout.shiftedLayout, columnsPerSbox]
    omega

/-- **The declaration says nothing at a non-canonical layout.**

A layout shifted far away names S-box columns the declaration does not contain,
so `transcriptColumns` is not an allocation list for an arbitrary `layouts` — it
is one for `canonicalLayouts`.  Stated with a witness rather than as a caveat. -/
theorem transcriptColumns_not_layout_generic :
    sboxColumn (Poseidon2Layout.shiftedLayout 1000000)
        ⟨0, by decide⟩ ⟨0, by decide⟩
      ∉ transcriptColumns 1 := by
  intro member
  unfold transcriptColumns at member
  rcases List.mem_flatMap.1 member with ⟨round, roundMember, inRound⟩
  rcases List.mem_map.1 inRound with ⟨offset, offsetMember, equal⟩
  have roundZero : round = 0 := by
    have := List.mem_range.1 roundMember
    omega
  have offsetLt : offset < 344 := List.mem_range.1 offsetMember
  rw [roundZero] at equal
  simp only [Poseidon2Layout.shiftedLayout, sboxColumn] at equal
  omega

/-- **Every declared column is written by an emitted row.**

`CANONICAL-PROGRAM-TRANSCRIPT-ALLOCATION-UNUSED`, closed.  The chain is: a
declared column is a canonical S-box column
(`transcriptColumns_eq_canonical_sbox`); each S-box writes its four columns in
the `c` of its four rows (`Poseidon2Program.sboxProgram_writes_sboxColumn`); and
a coefficient-one write survives normalization
(`Poseidon2Normalized.mentions_normalizeRow_singleton`).

That last step is not decoration.  `mentions_normalizeRow` runs one way only —
normalization drops a column whose coefficient vanishes modulo the prime, so
support equality is false and an arbitrary write does **not** lift.  Every S-box
write has coefficient `1`, so this one does. -/
theorem transcriptColumns_written
    (absorbedAt : Schedule) (constants : Constants) (rounds : Nat)
    (column : Nat) (member : column ∈ transcriptColumns rounds) :
    ∃ row ∈ canonicalTranscriptRows absorbedAt constants rounds,
      Mentions row.c column := by
  rcases (transcriptColumns_eq_canonical_sbox rounds column).1 member with
    ⟨round, roundLt, index, slot, rfl⟩
  have declared : sboxColumn (canonicalLayouts round) index slot
      ∈ Poseidon2Program.auxiliaryColumns (canonicalLayouts round) :=
    List.mem_flatMap.2 ⟨index, List.mem_finRange _,
      List.mem_map.2 ⟨slot, List.mem_finRange _, rfl⟩⟩
  rcases Poseidon2Program.permutationProgram_writes_auxiliaryColumns
      (canonicalLayouts round)
      (Poseidon2Schedule.scheduleOfFrom (canonicalLayouts round)
        (entryAt canonicalLayouts absorbedAt round) constants)
      (Poseidon2Schedule.finalState (canonicalLayouts round)) _ declared with
    ⟨raw, rawMember, write⟩
  refine ⟨Poseidon2Normalized.normalizeRow raw, ?_,
    Poseidon2Normalized.mentions_normalizeRow_singleton raw _ write⟩
  unfold canonicalTranscriptRows transcriptRows
  exact List.mem_flatMap.2 ⟨round, List.mem_range.2 roundLt,
    List.mem_map.2 ⟨raw, rawMember, rfl⟩⟩

/-- **Every declared column is a real column**, not the constant wire. -/
theorem transcriptColumns_nonzero (rounds : Nat) :
    ∀ column ∈ transcriptColumns rounds, column ≠ 0 := by
  intro column member
  unfold transcriptColumns at member
  rcases List.mem_flatMap.1 member with ⟨round, _, inRound⟩
  rcases List.mem_map.1 inRound with ⟨offset, _, rfl⟩
  omega

/-- **Every declared column lies in its round's window.** -/
theorem transcriptColumns_in_window (rounds : Nat) :
    ∀ column ∈ transcriptColumns rounds,
      ∃ round, round < rounds
        ∧ round * Poseidon2Sponge.spongeStride < column
        ∧ column ≤ round * Poseidon2Sponge.spongeStride + canonicalColumnTotal := by
  intro column member
  unfold transcriptColumns at member
  rcases List.mem_flatMap.1 member with ⟨round, roundMember, inRound⟩
  rcases List.mem_map.1 inRound with ⟨offset, offsetMember, rfl⟩
  have offsetLt : offset < 344 := List.mem_range.1 offsetMember
  refine ⟨round, List.mem_range.1 roundMember, ?_, ?_⟩
  · omega
  · simp only [canonicalColumnTotal, width, sboxCount, externalRounds,
      partialRounds, columnsPerSbox]
    omega

end Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe
