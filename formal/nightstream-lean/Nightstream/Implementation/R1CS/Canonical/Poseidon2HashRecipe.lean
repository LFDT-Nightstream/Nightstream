import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Honest
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: `hashPrior` and `hashNext` as emitted row programs.

Owns: relocation of the fixed-23 sponge into a base-addressed column window,
the two named instances, their derived row counts, their disjoint column
ownership, conservation of the relocation, soundness to the sponge digest,
honest completeness, and cost.

Does not own: absorption, padding, rate, capacity or the digest value — those
are `Poseidon2Sponge` and `Poseidon2Sponge23`, and this module consumes them.

## Why relocation is the content

`Poseidon2Sponge23` is written at **fixed absolute columns**: seven permutation
blocks of stride 361 followed by 23 preimage columns at 2527. That is fine for
one hash. The step program contains **two** — the prior link digest and the
public next digest — and two instances of a fixed-column program collide on
every column they use.

So the Phase-3 obligation that is not already discharged by the sponge is
exactly: place two instances so that "distinct owners never collide" is true,
and carry soundness and completeness across the placement. `relocate` shifts
every column except the constant wire, which is shared by construction.

## Where domain separation lives, and why not here

`Vocabulary`'s docstring says `hashPrior` and `hashNext` "have the same
semantics but different ownership".  That is true *of the typed call
vocabulary*, where both invoke `parameters.machine.hash`.  An earlier version of
this header concluded from it that no domain tag exists.  **That was wrong.**

The separator is `Poseidon23ApplicationProfile.normalizedIteration`:

```text
if next then coordinate + 1 else coordinate
```

with `Poseidon23HashPriorRecipe` passing `false` and `Poseidon23HashNextRecipe`
passing `true`.  Prior and next digests over otherwise-identical data differ.

It is **not owned here, and should not be**.  Separation is a property of the
*preimage*, and this module owns the *placement of a sponge*.  The preimage is
built one layer up, by the encoding profile, which is also where the refinement
equations binding it to the frozen hash live.  Re-exporting the separator into
this module would invert the layering — `Encoding` depends on `Canonical`, not
the reverse — and would put a preimage fact in a module that cannot see a
preimage.

`POSEIDON2-HASH-NO-DOMAIN-TAG` carries the retraction and the evidence.

## What separation this recipe does own

The sponge's padding rule: `Poseidon2Sponge.pad` absorbs a single `1`, which is
what makes preimages of different lengths distinct.  At the fixed arity this
recipe is placed at, `committed_single_arity` shows there is only one length, so
that rule separates nothing *here* — a fact about this instantiation, not about
the rule.

## What padding does own, which is soundness rather than separation

Separating nothing is not the same as doing nothing.  The seventh call absorbs
`1` from column `0`, so `committed_padding_value` pins it under any assignment
that pins the constant wire: **the prover does not choose the padding.**  A free
absorbed lane would let two preimages differing only there produce the same
digest, which is a soundness failure and not a separation one.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23

/-! ## The sponge's column window -/

/-- Every column one sponge instance can touch: seven permutation blocks of
stride `callStride`, then the 23 preimage columns. -/
def spongeColumnTotal : Nat := inputBase + sponge23Fields

theorem spongeColumnTotal_eq : spongeColumnTotal = 2550 := by decide

/-- **Relocation.**  The constant wire is shared by every instance and stays at
column `0`; every other column shifts by `base`. -/
def relocate (base : Nat) : Nat → Nat :=
  fun column => if column = 0 then 0 else base + column

theorem relocate_zero (base : Nat) : relocate base 0 = 0 := rfl

theorem relocate_pos (base column : Nat) (nonZero : column ≠ 0) :
    relocate base column = base + column := if_neg nonZero

/-- Relocation is injective on the sponge's window, which is what makes two
placed instances separable. -/
theorem relocate_injective_on_window
    (base column other : Nat)
    (relocated : relocate base column = relocate base other) :
    column = other := by
  unfold relocate at relocated
  by_cases columnZero : column = 0 <;> by_cases otherZero : other = 0 <;>
    simp only [columnZero, otherZero, reduceIte] at relocated ⊢
  · omega
  · omega
  · omega

/-! ## The emitted program -/

/-- **The emitted hash program at a base.**  The fixed-23 sponge, relocated. -/
def hashProgram (base : Nat) (constants : Constants) : List Row :=
  (program constants).map (renameRow (relocate base))

/-- **The derived row count.**  Relocation emits no row and drops none. -/
theorem hashProgram_length (base : Nat) (constants : Constants) :
    (hashProgram base constants).length = (program constants).length := by
  unfold hashProgram
  exact List.length_map _

theorem hashProgram_length_eq (base : Nat) (constants : Constants) :
    (hashProgram base constants).length = 2464 := by
  rw [hashProgram_length, program_length]

/-! ## The two named instances -/

/-- The prior link digest is an auxiliary value, and is placed first. -/
def hashPriorBase : Nat := 0

/-- The public next digest follows a full sponge window later, so the two
windows cannot overlap. -/
def hashNextBase : Nat := spongeColumnTotal

/-- **`hashPrior`.**  Auxiliary link digest. -/
def hashPrior (constants : Constants) : List Row :=
  hashProgram hashPriorBase constants

/-- **`hashNext`.**  The sole public step output. -/
def hashNext (constants : Constants) : List Row :=
  hashProgram hashNextBase constants

theorem hashPrior_length (constants : Constants) :
    (hashPrior constants).length = 2464 :=
  hashProgram_length_eq _ constants

theorem hashNext_length (constants : Constants) :
    (hashNext constants).length = 2464 :=
  hashProgram_length_eq _ constants

/-! ## Column ownership

Each instance owns `[base + 1, base + spongeColumnTotal)`.  Column `0` is a
shared read, not an allocation, and belongs to neither. -/

private theorem nodup_ofFn_of_injective
    {α : Type} :
    ∀ {n : Nat} (function : Fin n → α),
      Function.Injective function → (List.ofFn function).Nodup
  | 0, _, _ => by simp
  | _ + 1, function, injective => by
      rw [List.ofFn_succ, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index (injective equal)
      · exact nodup_ofFn_of_injective
          (fun index => function index.succ)
          (fun first second equal => Fin.succ_inj.mp (injective equal))

def ownedColumns (base : Nat) : List Nat :=
  List.ofFn (fun offset : Fin (spongeColumnTotal - 1) => base + offset.val + 1)

theorem ownedColumns_length (base : Nat) :
    (ownedColumns base).length = spongeColumnTotal - 1 := by
  simp [ownedColumns]

theorem ownedColumns_nodup (base : Nat) : (ownedColumns base).Nodup := by
  apply nodup_ofFn_of_injective
  intro first second equal
  apply Fin.ext
  change base + first.val + 1 = base + second.val + 1 at equal
  omega

theorem mem_ownedColumns (base column : Nat) :
    column ∈ ownedColumns base
      ↔ base + 1 ≤ column ∧ column < base + spongeColumnTotal := by
  unfold ownedColumns
  rw [List.mem_ofFn]
  constructor
  · rintro ⟨offset, rfl⟩
    have total : spongeColumnTotal = 2550 := spongeColumnTotal_eq
    have offsetLt := offset.isLt
    omega
  · rintro ⟨lower, upper⟩
    have total : spongeColumnTotal = 2550 := spongeColumnTotal_eq
    exact ⟨⟨column - base - 1, by omega⟩, by simp only []; omega⟩

/-- **The two instances never collide.**  Neither window contains a column of
the other, so "distinct owners never collide" is a theorem rather than a naming
convention. -/
theorem hashPrior_hashNext_disjoint (column : Nat)
    (inPrior : column ∈ ownedColumns hashPriorBase) :
    column ∉ ownedColumns hashNextBase := by
  rw [mem_ownedColumns] at inPrior
  rw [mem_ownedColumns]
  simp only [hashPriorBase, hashNextBase, spongeColumnTotal_eq] at inPrior ⊢
  omega

/-! ## Conservation of the relocation

This module owns what relocation does to a row's support; the sponge owns what
its own rows touch (`Poseidon2Sponge23Ownership.program_conservation`).  The two
compose. -/

theorem mentions_renameTerms (base : Nat) (terms : LinCombNormal.LinComb)
    (column : Nat)
    (mentioned : Mentions (renameTerms (relocate base) terms) column) :
    ∃ source, Mentions terms source ∧ column = relocate base source := by
  simp only [Mentions, renameTerms, List.map_map, List.mem_map,
    Function.comp] at mentioned ⊢
  rcases mentioned with ⟨term, member, rfl⟩
  exact ⟨term.1, ⟨term, member, rfl⟩, rfl⟩

/-- **Every column of a relocated row is the relocation of a source column.**
Nothing is introduced by the placement. -/
theorem hashProgram_conservation
    (base : Nat) (constants : Constants) (row : Row)
    (member : row ∈ hashProgram base constants) (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    ∃ source, Poseidon2Sponge23Ownership.Allocated source
      ∧ column = relocate base source := by
  unfold hashProgram at member
  rcases List.mem_map.1 member with ⟨raw, rawMember, rfl⟩
  simp only [renameRow] at mentioned
  rcases mentioned with inA | inB | inC
  · rcases mentions_renameTerms base raw.a column inA with ⟨source, m, rfl⟩
    exact ⟨source, Poseidon2Sponge23Ownership.program_conservation constants raw
      rawMember source (Or.inl m), rfl⟩
  · rcases mentions_renameTerms base raw.b column inB with ⟨source, m, rfl⟩
    exact ⟨source, Poseidon2Sponge23Ownership.program_conservation constants raw
      rawMember source (Or.inr (Or.inl m)), rfl⟩
  · rcases mentions_renameTerms base raw.c column inC with ⟨source, m, rfl⟩
    exact ⟨source, Poseidon2Sponge23Ownership.program_conservation constants raw
      rawMember source (Or.inr (Or.inr m)), rfl⟩

/-! ## Soundness

Satisfaction of a placed instance is satisfaction of the sponge under the
pulled-back assignment, so the sponge's digest theorem applies unchanged. -/

theorem hashProgram_pull
    (base : Nat) (constants : Constants) (z : Nat → Nat)
    (satisfied : Satisfies (hashProgram base constants) z) :
    Satisfies (program constants) (pullAssignment z (relocate base)) := by
  intro row member
  refine (rowHolds_pull_iff z (relocate base) row).2 ?_
  exact satisfied _ (List.mem_map.2 ⟨row, member, rfl⟩)

/-- **A placed hash program computes the sponge digest**, at the relocated
output columns. -/
theorem hashProgram_computes_digest
    (base : Nat) (constants : Constants) (z : Nat → Nat) (input : Preimage)
    (residues : ∀ column, z column < goldilocksP) (constantWire : z 0 = 1)
    (inputsAgree : InputsAgree (pullAssignment z (relocate base)) input)
    (satisfied : Satisfies (hashProgram base constants) z)
    (lane : Fin digestLength) :
    z (relocate base ((layout.call 6).outputPort
        ⟨lane.val, by
          have laneLt := lane.isLt
          simp only [digestLength, width] at laneLt ⊢
          omega⟩))
      = digest constants (dataChunks input) lane :=
  program_computes_digest constants (pullAssignment z (relocate base)) input
    (fun column => residues _)
    (by unfold pullAssignment; rw [relocate_zero]; exact constantWire)
    inputsAgree (hashProgram_pull base constants z satisfied) lane

/-! ## Honest completeness

The honest witness for a placed instance is the sponge's own witness, read
through the placement. -/

/-- **The placed honest assignment.**  `column - base` lands on `0` for the
constant wire at any base, which is exactly the shared-read convention. -/
def honestAssignment (base : Nat) (constants : Constants) (input : Preimage) :
    Nat → Nat :=
  fun column => Poseidon2Sponge23Honest.assignment constants input (column - base)

theorem pull_honestAssignment (base : Nat) (constants : Constants)
    (input : Preimage) :
    pullAssignment (honestAssignment base constants input) (relocate base)
      = Poseidon2Sponge23Honest.assignment constants input := by
  funext column
  unfold pullAssignment honestAssignment relocate
  by_cases columnZero : column = 0
  · simp only [columnZero, reduceIte, Nat.zero_sub]
  · simp only [columnZero, reduceIte]
    congr 1
    omega

/-- **An honest sponge execution satisfies the placed program.** -/
theorem hashProgram_honest
    (base : Nat) (constants : Constants) (input : Preimage)
    (inputResidues : ∀ index, input index < goldilocksP) :
    Satisfies (hashProgram base constants)
      (honestAssignment base constants input) := by
  intro row member
  unfold hashProgram at member
  rcases List.mem_map.1 member with ⟨raw, rawMember, rfl⟩
  refine (rowHolds_pull_iff _ (relocate base) raw).1 ?_
  rw [pull_honestAssignment]
  exact Poseidon2Sponge23Honest.honest_satisfies constants input inputResidues
    raw rawMember

/-! ## Cost -/

/-- **One placed hash's cost.**  2,464 rows and 2,549 allocated columns: the
sponge's 2,464 temporaries plus its 23 preimage columns, minus the shared
constant wire, which is a read rather than an allocation. -/
def hashCost : Lowering.Typed.Cost where
  recurringRows := 2464
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 2549

theorem hashCost_rows (base : Nat) (constants : Constants) :
    (hashProgram base constants).length = hashCost.recurringRows :=
  hashProgram_length_eq base constants

theorem hashCost_columns (base : Nat) :
    (ownedColumns base).length = hashCost.auxiliaryColumns := by
  rw [ownedColumns_length]
  decide

/-- **Both calls together.**  Two placements, no shared allocation. -/
def hashPairCost : Lowering.Typed.Cost where
  recurringRows := 2 * hashCost.recurringRows
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 2 * hashCost.auxiliaryColumns

theorem hashPairCost_rows (constants : Constants) :
    (hashPrior constants).length + (hashNext constants).length
      = hashPairCost.recurringRows := by
  rw [hashPrior_length, hashNext_length]
  rfl

/-! ## The recipe's parameter commitments

Absorption, padding, rate and capacity are `Poseidon2Sponge`'s to prove.  They
are *this recipe's to commit to*, and the difference matters: a reader auditing
`hashPrior` should be able to check what it promises without first reconstructing
which sponge it was built on.  Each statement below is discharged from the
sponge; none is re-proved. -/

theorem committed_rate : rate = 4 := rfl

theorem committed_capacity : capacity = 4 := Poseidon2Sponge.capacity_eq

/-- Rate and capacity partition the state, so no lane is both absorbed into and
reserved. -/
theorem committed_partition : rate + capacity = width :=
  Poseidon2Sponge.rate_add_capacity

/-- The digest reads only absorbed lanes. -/
theorem committed_digest_within_rate : digestLength ≤ rate :=
  Poseidon2Sponge.digest_within_rate

/-- The preimage arity this recipe is fixed at. -/
theorem committed_arity : sponge23Fields = 23 := rfl

/-- The absorption schedule: five full chunks and one of three. -/
theorem committed_chunking :
    sponge23Fields = (sponge23Chunks - 1) * rate + 3 :=
  Poseidon2Sponge.sponge23_chunk_arithmetic

/-- Seven permutation calls, derived from the arity. -/
theorem committed_permutationCalls
    (chunks : List RateChunk) (count : chunks.length = sponge23Chunks) :
    permutationCalls chunks = 7 :=
  Poseidon2Sponge.sponge23_permutationCalls chunks count

/-- **Absorption never writes capacity.**  The commitment that makes the
capacity a capacity rather than eight more absorbed lanes. -/
theorem committed_capacity_untouched
    (chunk : RateChunk) (state : Values) (lane : Fin width)
    (isCapacity : rate ≤ lane.val) :
    absorbChunk chunk.values state lane = state lane :=
  Poseidon2Sponge.RateChunk_capacity_untouched chunk state lane isCapacity

/-! ### Padding

The section above named absorption, padding, rate and capacity as this recipe's
to commit to, and committed to four things that were not padding.  Padding is
the one of the four with a **soundness** consequence rather than only a shape
one: an absorbed lane a prover can choose is a lane that lets one preimage be
extended into another.

The seventh call is the padding call.  Six data calls carry `4+4+4+4+4+3 = 23`
fields; the seventh absorbs the single value `1`, read from column `0`. -/

/-- **The padding chunk is the singleton one**, for every preimage. -/
theorem committed_padding_chunk (input : Preimage) :
    chunkAt input 6 = [1] :=
  chunkAt_padding input

/-- **Padding reads the constant wire**, not an allocated column. -/
theorem committed_padding_on_constant_wire (lane : Fin width) :
    layout.chunkColumn 6 lane = 0 :=
  chunkColumn_padding lane

/-- **The prover does not choose the padding.**

The consequence the other two exist for.  Padding is read from column `0`, so
any assignment that pins the constant wire pins the absorbed value — there is no
witness column a prover could vary to absorb something other than `1`.

Without this, a recipe could absorb 23 fields and a free lane, and two preimages
differing only in that lane would be indistinguishable to anything downstream of
the digest. -/
theorem committed_padding_value (z : Nat → Nat) (wire : z 0 = 1)
    (lane : Fin width) :
    z (layout.chunkColumn 6 lane) = 1 := by
  rw [chunkColumn_padding]
  exact wire

/-- **Padding carries no preimage field**, so it separates nothing by content
and is not part of the absorbed message. -/
theorem committed_padding_input_independent (first second : Preimage) :
    chunkAt first 6 = chunkAt second 6 := by
  rw [chunkAt_padding, chunkAt_padding]

/-- **Seven calls: six of data and one of padding.** -/
theorem committed_padding_call : calls = dataCalls + 1 := rfl

/-- **The absorbed total is the arity plus the one padding lane.**  Derived from
the emitted chunk lengths rather than declared. -/
theorem committed_absorbed_total :
    ((List.range calls).map chunkLength).sum = sponge23Fields + 1 := by
  decide

/-! ### Domain separation

The fifth of Phase 3's five items is `committed_separation_survives`, at the end
of this file — it needs theorems stated below and so cannot sit here with the
other four. -/

/-! ## Separation, stated exactly

Two questions a reader will ask, answered as theorems rather than prose. -/

/-- **Length separation does not arise.**  The recipe admits exactly one
preimage length, so the padding rule's length-distinguishing job is vacuous
here — there are no two lengths to distinguish.

What this does **not** say — corrected in cycle 384 — is that nothing separates
the two calls.  `POSEIDON2-HASH-NO-DOMAIN-TAG` was retracted in cycle 345: the
profile's `normalizedIteration` adds one to the iteration coordinate.  Length
separation being vacuous at fixed arity is a fact about *this recipe's* padding
rule, not about the protocol's separation. -/
theorem committed_single_arity
    (first second : List Nat)
    (firstFixed : first.length = sponge23Fields)
    (secondFixed : second.length = sponge23Fields) :
    first.length = second.length :=
  Poseidon2Sponge.sponge23_single_arity first second firstFixed secondFixed

/-- **Placement carries no separation.**

Two placed instances compute the same digest from the same preimage.  This is
the formal content of "`hashPrior` and `hashNext` have the same semantics": the
base is a column offset and nothing more, so nothing about *where* a hash sits
enters *what* it computes.

A reader should take the consequence seriously rather than as a technicality:
if two call sites ever hash equal 23-field preimages, they obtain equal
digests. -/
theorem digest_independent_of_placement
    (constants : Constants) (input : Preimage) (lane : Fin digestLength)
    (baseA baseB : Nat) (za zb : Nat → Nat)
    (residuesA : ∀ column, za column < goldilocksP) (wireA : za 0 = 1)
    (agreeA : InputsAgree (pullAssignment za (relocate baseA)) input)
    (satisfiedA : Satisfies (hashProgram baseA constants) za)
    (residuesB : ∀ column, zb column < goldilocksP) (wireB : zb 0 = 1)
    (agreeB : InputsAgree (pullAssignment zb (relocate baseB)) input)
    (satisfiedB : Satisfies (hashProgram baseB constants) zb) :
    za (relocate baseA ((layout.call 6).outputPort
        ⟨lane.val, by
          have laneLt := lane.isLt
          simp only [digestLength, width] at laneLt ⊢
          omega⟩))
      = zb (relocate baseB ((layout.call 6).outputPort
        ⟨lane.val, by
          have laneLt := lane.isLt
          simp only [digestLength, width] at laneLt ⊢
          omega⟩)) := by
  rw [hashProgram_computes_digest baseA constants za input residuesA wireA
      agreeA satisfiedA lane,
    hashProgram_computes_digest baseB constants zb input residuesB wireB
      agreeB satisfiedB lane]

/-! ## What this layer does contribute to separation

The header records that the *separator* between the two calls lives in the
encoding profile.  That is not the same as this layer contributing nothing.

Separation by preimage content only means anything if the preimage is actually
absorbed.  A recipe that silently dropped a field would make two distinct
preimages hash identically no matter what the profile did upstream — and this
recipe sees the preimage, as a parameter, so this is its property to state.

`chunkValue input call lane = input ⟨call * rate + lane⟩` places field
`call * 4 + lane` at chunk `call`, lane `lane`.  Calls `0 … 4` carry four lanes
each and call `5` carries three: `5 · 4 + 3 = 23`, every index exactly once.
`chunkAt_determines` turns that into the statement that matters — the chunking
loses nothing. -/

/-- Each preimage index is the position its own chunk lane reads. -/
theorem chunkValue_at_index (input : Preimage) (index : Fin sponge23Fields) :
    chunkValue input (index.val / rate) (index.val % rate) = input index := by
  have indexLt : index.val < sponge23Fields := index.isLt
  simp only [sponge23Fields] at indexLt
  have callLt : index.val / rate < dataCalls := by
    show index.val / 4 < 6
    omega
  have inRange : index.val / rate * rate + index.val % rate < sponge23Fields := by
    show index.val / 4 * 4 + index.val % 4 < 23
    omega
  unfold chunkValue
  rw [dif_pos ⟨callLt, inRange⟩]
  congr 1
  apply Fin.ext
  simp only [rate]
  omega

/-- **The chunking loses nothing.**

Two preimages with the same chunks are the same preimage, so the chunk
decomposition loses nothing.

This is a fact about **chunking**, not about absorption — the two are separate
steps and this one does not reach the state.  `separator_survives_absorption`
carries it the rest of the way. -/
theorem chunkAt_determines (first second : Preimage)
    (equal : ∀ call, chunkAt first call = chunkAt second call) :
    first = second := by
  funext index
  have indexLt : index.val < sponge23Fields := index.isLt
  simp only [sponge23Fields] at indexLt
  have laneLt : index.val % rate < chunkLength (index.val / rate) := by
    simp only [rate]
    have cases : index.val / 4 = 0 ∨ index.val / 4 = 1 ∨ index.val / 4 = 2
        ∨ index.val / 4 = 3 ∨ index.val / 4 = 4 ∨ index.val / 4 = 5 := by omega
    rcases cases with h | h | h | h | h | h <;> rw [h] <;>
      simp only [chunkLength] <;> omega
  have entries := congrArg (fun list => list[index.val % rate]?) (equal (index.val / rate))
  simp only [chunkAt, List.getElem?_ofFn, laneLt, dif_pos, reduceDIte] at entries
  rw [← chunkValue_at_index first index, ← chunkValue_at_index second index]
  exact Option.some_inj.1 entries

/-! ## Where the separator lands

`chunkAt_determines` says the chunking is lossless.  This says something
sharper about the specific field the profile's separator touches.

`Poseidon23ApplicationProfile.normalizedIteration` differs between the two calls
in the **first source coordinate**, and the profile's projection places source
coordinates into preimage slots.  Whatever slot the iteration lands in, this
recipe carries a difference there into a difference in that slot's chunk lane —
field `i` is read by chunk `i / 4` at lane `i % 4`, and by nothing else.

So the composition is complete and each half is where it belongs: the profile
**applies** the separator, and the recipe **delivers** it to a lane of a chunk
the sponge absorbs.  Neither statement is the other, and neither layer can make
the other's. -/

/-- **A difference at any preimage index survives into that index's chunk.**

The contrapositive of `chunkAt_determines` localised: it is not merely that some
chunk differs, but that the chunk carrying that index does. -/
theorem chunk_differs_at_index
    (first second : Preimage) (index : Fin sponge23Fields)
    (differ : first index ≠ second index) :
    chunkAt first (index.val / rate) ≠ chunkAt second (index.val / rate) := by
  intro sameChunk
  refine differ ?_
  rw [← chunkValue_at_index first index, ← chunkValue_at_index second index]
  have indexLt : index.val < sponge23Fields := index.isLt
  simp only [sponge23Fields] at indexLt
  have laneLt : index.val % rate < chunkLength (index.val / rate) := by
    simp only [rate]
    have cases : index.val / 4 = 0 ∨ index.val / 4 = 1 ∨ index.val / 4 = 2
        ∨ index.val / 4 = 3 ∨ index.val / 4 = 4 ∨ index.val / 4 = 5 := by omega
    rcases cases with h | h | h | h | h | h <;> rw [h] <;>
      simp only [chunkLength] <;> omega
  have entries := congrArg (fun list => list[index.val % rate]?) sameChunk
  simp only [chunkAt, List.getElem?_ofFn, laneLt, dif_pos, reduceDIte] at entries
  exact Option.some_inj.1 entries

/-- **The separator's own field.**  The profile distinguishes the two calls in
the first source coordinate; whichever preimage slot carries it, a difference
there reaches chunk `slot / 4` at lane `slot % 4`.

Stated for slot zero, the case the production projection uses. -/
theorem separator_reaches_chunk_zero
    (first second : Preimage)
    (differ : first ⟨0, by decide⟩ ≠ second ⟨0, by decide⟩) :
    chunkAt first 0 ≠ chunkAt second 0 := by
  have shifted := chunk_differs_at_index first second ⟨0, by decide⟩ differ
  simpa only [rate, Nat.zero_div] using shifted

/-! ## Applying a separator at this layer

Everything above says this recipe *preserves* a separation applied elsewhere.
It can also *apply* one, and the distinction between what that establishes and
what it does not is the point of this section.

`separatedPreimage` is the action of a first-slot separator on a preimage:
add one to slot zero, in the field, leaving the rest alone.  The reduction
modulo the prime is not decoration — `Poseidon23ApplicationProfile`'s
`normalizedIteration` adds one to a **field** coordinate, and a `Nat` increment
would be a different function wherever the coordinate is `p - 1`.  That is the
`TRANSCRIPT-MODE-BOUNDARY` defect in miniature, so it is avoided by
construction.

**This is a reconstruction, not the authoritative separator.**  The profile's
version is authoritative; whether the two agree is a conformance obligation
between layers and is *not* established here — this module cannot see an
iteration.  What is established is that a separator of this shape distinguishes
every preimage, and reaches chunk zero, so a profile that applies one loses
nothing at this layer. -/

/-- The action of a first-slot separator on a preimage: `+1` in the field, at
slot zero only. -/
def separatedPreimage (next : Bool) (input : Preimage) : Preimage :=
  fun index =>
    if index.val = 0 ∧ next then (input index + 1) % goldilocksP else input index

theorem separatedPreimage_false (input : Preimage) :
    separatedPreimage false input = input := by
  funext index
  unfold separatedPreimage
  simp

/-- **The separator moves slot zero**, for every residue.

The wrap case is where a `Nat` increment would have failed: at `p - 1` the field
sum is `0`, which still differs from `p - 1`. -/
theorem separatedPreimage_differs
    (input : Preimage) (canonical : input ⟨0, by decide⟩ < goldilocksP) :
    separatedPreimage true input ⟨0, by decide⟩ ≠ input ⟨0, by decide⟩ := by
  unfold separatedPreimage
  simp only [and_true, reduceIte]
  by_cases atTop : input ⟨0, by decide⟩ = goldilocksP - 1
  · rw [atTop]
    have : (goldilocksP - 1 + 1) % goldilocksP = 0 := by decide
    rw [this]
    decide
  · have below : input ⟨0, by decide⟩ + 1 < goldilocksP := by
      simp only [goldilocksP] at canonical atTop ⊢
      omega
    rw [Nat.mod_eq_of_lt below]
    omega

/-- **A separated preimage reaches a different chunk zero.**

Applying the separator and delivering it, composed: the two calls absorb
different chunks. -/
theorem separatedPreimage_reaches_chunk_zero
    (input : Preimage) (canonical : input ⟨0, by decide⟩ < goldilocksP) :
    chunkAt (separatedPreimage true input) 0 ≠ chunkAt input 0 :=
  separator_reaches_chunk_zero (separatedPreimage true input) input
    (separatedPreimage_differs input canonical)

/-! ## Chunking is not absorption

`chunkAt_determines` shows the chunk decomposition loses nothing, and its
docstring said that separation therefore "survives absorption".  Those are two
steps, not one: chunking arranges the preimage into blocks, absorption folds a
block into the state.  A recipe whose chunking was injective could still absorb
two distinct chunks into one state, and the separator would be lost with no
permutation involved — a structural failure, not a cryptographic one.

`Poseidon2Sponge.absorbChunk_injective_at_lane` is the missing step, and this
composes the two at the slot the separator moves. -/

private theorem chunk_zero_lane_zero (input : Preimage) :
    (chunkAt input 0)[0]? = some (input ⟨0, by decide⟩) := by
  have value : chunkValue input 0 0 = input ⟨0, by decide⟩ := by
    simpa using chunkValue_at_index input ⟨0, by decide⟩
  simp only [chunkAt, chunkLength, List.getElem?_ofFn]
  exact congrArg some value

/-- **The separator survives absorption**, and not only chunking.

The state after absorbing chunk zero differs between a preimage and its
slot-zero increment.  This is the last step that can be settled by arithmetic:
whether the *permutation* then preserves the difference is Poseidon2's business
and not this recipe's. -/
theorem separator_survives_absorption
    (input : Preimage) (state : Values)
    (canonical : input ⟨0, by decide⟩ < goldilocksP) :
    absorbChunk (chunkAt (separatedPreimage true input) 0) state ⟨0, by decide⟩
      ≠ absorbChunk (chunkAt input 0) state ⟨0, by decide⟩ :=
  Poseidon2Sponge.absorbChunk_injective_at_lane _ _ state ⟨0, by decide⟩ _ _
    (chunk_zero_lane_zero (separatedPreimage true input))
    (chunk_zero_lane_zero input)
    (by
      unfold separatedPreimage
      simp only [and_true, reduceIte]
      exact Nat.mod_lt _ (by decide))
    canonical
    (separatedPreimage_differs input canonical)

/-! ## The fifth commitment: separation, as preservation

Phase 3 names five things this recipe should own — absorption, padding, rate,
capacity and domain separation.  The first four are committed to above, each
discharged from `Poseidon2Sponge` rather than re-proved.  This is the fifth, in
the same form.

**Separation at this layer is preservation, not choice.**  The recipe cannot see
an iteration coordinate, so it cannot pick the separator.  What it can commit to
is that a separator applied to the preimage survives every step this module owns:

| step | what it rules out |
|---|---|
| chunking loses none | two distinct preimages chunked to the same blocks |
| the separator moves chunk zero | a separator that lands where nothing reads |
| absorption loses none | two distinct blocks folded into the same state |

A fourth element belongs to this commitment and is **not** restated in the
conjunction below: `digest_independent_of_placement`, which rules out a separator
invented by *where* a hash sits.  It is proved and guarded above; repeating its
thirteen-argument signature here would obscure rather than sharpen.  This
theorem's name says what it carries — the three preservation steps — and does not
claim to be the whole commitment.

What is **not** committed anywhere in this module is that a separator is applied
at all, or that its argument is the iteration.  That is the profile's, and
`Poseidon23SeparatorConformance.SeparatingPlan` is where it lands.  The boundary
is therefore exact: this module owns that separation is not *lost*, the profile
owns that it is *applied*. -/

/-- **The separator survives every step this module owns.** -/
theorem committed_separation_survives
    (input : Preimage) (state : Values)
    (canonical : input ⟨0, by decide⟩ < goldilocksP) :
    (∀ first second : Preimage,
        (∀ call, chunkAt first call = chunkAt second call) → first = second)
      ∧ chunkAt (separatedPreimage true input) 0 ≠ chunkAt input 0
      ∧ absorbChunk (chunkAt (separatedPreimage true input) 0) state
            ⟨0, by decide⟩
          ≠ absorbChunk (chunkAt input 0) state ⟨0, by decide⟩ :=
  ⟨chunkAt_determines,
    separatedPreimage_reaches_chunk_zero input canonical,
    separator_survives_absorption input state canonical⟩

end Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe
