import Nightstream.Implementation.R1CS.Canonical.Poseidon2Core

/-!
Contract: the assembled canonical Poseidon2 permutation program.

Owns: the round-state allocator with unique column ownership; the round-ordered
row program built from it; terminal binding; and the row and column counts
derived from the assembled receipts rather than written by hand.

This module is what turns `Poseidon2Core`'s subtotals into an assembled
program.  It resolves the auxiliary-column count that `Poseidon2Core`
deliberately left open.

Does not own: absorption/padding (Phase 3), the wrapper, or any artifact.

Scope: the **width-8 F'/`neo_ccs` permutation only**.  The width-16 Poseidon2
in `neo-reductions` (CCS digest machinery) is out of scope.

Authority: Poseidon2 and its parameters are project-level production choices,
not paper-derived.  Neither SuperNeo nor HyperNova selects a hash; both take
one as an abstract parameter.  Ownership splits: `neo-params` owns width,
capacity, rate, digest length and seed; p3 and the Rust circuit own `x^7`, the
8/22 round selection, and the two linear layers.

Obligations, renamed 2026-07-25 so each names what it actually buys:

  * `POSEIDON2-STRUCTURAL-PROFILE` — width/rate/capacity/digest, `x^7`, 8/22
    rounds, matrix formulas.  Cheap; does **not** license "the selected
    instance" while constants remain arbitrary.
  * `POSEIDON2-ROUND-CONSTANT-CONFORMANCE` — constants match the seed under the
    p3/ChaCha8 convention.  Required only for bit-for-bit claims.
  * `POSEIDON2-RUST-CONFORMANCE` — shipping Rust evaluates the same
    instantiated permutation.
  * `POSEIDON2-SUPPORT-BOUND` — normalized combination support stays at most
    31 terms; see `applyMatrix`.

Conditionality: the counts below are the exact structural cost of the selected
folding normal form **given a conforming schedule**.  `Schedule` is abstract
here precisely so allocation, ownership and counts are proved independently of
the matrix arithmetic; instantiating it and proving
`permutationProgram_exec_iff_spec` is `POSEIDON2-ROUND-INDUCTION`.

| Obligation | Lean owner |
|---|---|
| unique column ownership | `sboxColumn_injective` |
| assembled row program | `permutationProgram` |
| row count from receipts | `permutationProgram_length` |
| auxiliary count from receipts | `auxiliaryColumns_length` |
| every row has an owner (coverage only) | `everyPermutationRow_has_owner` |
| every column has one owner | `everyPermutationColumn_has_exact_owner` |
| composed S-box soundness | `permutationProgram_sbox_chains` |
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Program

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Lowering
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core

/-! ## Allocator

Each S-box owns four consecutive auxiliary columns.  `sboxCount` is derived in
`Poseidon2Core` from the published round structure. -/

/-- Columns per S-box frame: `square`, `fourth`, `sixth`, `output`.  All four are
auxiliaries: an S-box output feeds the following symbolic linear layer and is
never a declared permutation port. -/
def columnsPerSbox : Nat := 4

/-- Layout supplied by the caller: where the auxiliary block starts, and the
declared input and output ports. -/
structure Layout where
  auxBase : Nat
  inputPort : Fin width → Nat
  outputPort : Fin width → Nat

/-- The `slot`-th column owned by S-box `index`. -/
def sboxColumn (layout : Layout) (index : Fin sboxCount) (slot : Fin columnsPerSbox) :
    Nat :=
  layout.auxBase + columnsPerSbox * index.val + slot.val

/-- **Unique ownership.**  Distinct `(S-box, slot)` pairs own distinct
columns. -/
theorem sboxColumn_injective (layout : Layout)
    {i j : Fin sboxCount} {s t : Fin columnsPerSbox}
    (equal : sboxColumn layout i s = sboxColumn layout j t) :
    i = j ∧ s = t := by
  have sBound := s.isLt
  have tBound := t.isLt
  simp only [sboxColumn, columnsPerSbox] at equal sBound tBound
  constructor
  · apply Fin.ext; omega
  · apply Fin.ext; omega

/-- The frame for one S-box, over an input combination supplied by the
schedule. -/
def frameAt (layout : Layout) (index : Fin sboxCount) (input : LinComb) :
    SboxFrame where
  input := input
  square := sboxColumn layout index ⟨0, by decide⟩
  fourth := sboxColumn layout index ⟨1, by decide⟩
  sixth := sboxColumn layout index ⟨2, by decide⟩
  output := sboxColumn layout index ⟨3, by decide⟩

/-! ## Uniform flatMap length

Every inner block has the same size, so a flatMap length is a product.  Used
for both the row program and the auxiliary allocation. -/

theorem length_flatMap_uniform {A B : Type} (list : List A) (f : A -> List B)
    (size : Nat) (uniform : forall x, (f x).length = size) :
    (list.flatMap f).length = list.length * size := by
  induction list with
  | nil => simp
  | cons head tail hypothesis =>
      simp [List.flatMap_cons, uniform head, hypothesis, Nat.succ_mul,
        Nat.add_comm]

/-! ## Assembled auxiliary columns

Derived from the allocator, not asserted. -/

def auxiliaryColumns (layout : Layout) : List Nat :=
  (List.finRange sboxCount).flatMap
    (fun index =>
      (List.finRange columnsPerSbox).map (fun slot => sboxColumn layout index slot))

/-- **The auxiliary-column count, derived from the assembled allocator.**
`Poseidon2Core` deliberately left this unresolved; it is `86 * 4`. -/
theorem auxiliaryColumns_length (layout : Layout) :
    (auxiliaryColumns layout).length = sboxCount * columnsPerSbox := by
  unfold auxiliaryColumns
  rw [length_flatMap_uniform _ _ columnsPerSbox (by intro x; simp)]
  simp

theorem auxiliaryColumns_length_eq (layout : Layout) :
    (auxiliaryColumns layout).length = 344 := by
  rw [auxiliaryColumns_length]; decide

/-! ## Round schedule

Four leading external rounds over all lanes, twenty-two internal rounds over
lane zero, four trailing external rounds over all lanes.  The S-box index
families are `[0,32)`, `[32,54)`, `[54,86)`. -/

/-- The input combination each S-box consumes is supplied by the schedule.  It
is abstracted here so the assembled program's *shape* — allocation, ownership,
row count — is proved independently of the matrix arithmetic, which is the
separate `POSEIDON2-ROUND-INDUCTION` obligation. -/
abbrev Schedule := Fin sboxCount → LinComb

/-- All S-box rows, in round order. -/
def sboxProgram (layout : Layout) (schedule : Schedule) : List Row :=
  (List.finRange sboxCount).flatMap
    (fun index => sboxRows (frameAt layout index (schedule index)))

theorem sboxProgram_length (layout : Layout) (schedule : Schedule) :
    (sboxProgram layout schedule).length = sboxCount * 4 := by
  unfold sboxProgram
  rw [length_flatMap_uniform _ _ 4 (by intro x; simp [sboxRows])]
  simp

/-! ## The allocation is written, not only declared

`auxiliaryColumns` lists what an S-box program allocates.  Ownership theorems
bound which columns a row may mention; none said an allocated column is reached
at all, so the list could have been longer than the program and every count
would still have agreed.

Each S-box writes its four columns in the `c` of its four rows — square, fourth,
sixth, output — so the converse holds by construction and needs only to be
stated. -/

/-- **Every column an S-box allocates is written by one of its own rows.** -/
theorem sboxProgram_writes_sboxColumn
    (layout : Layout) (schedule : Schedule)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox) :
    ∃ row ∈ sboxProgram layout schedule,
      row.c = [(sboxColumn layout index slot, 1)] := by
  have slotLt : slot.val < 4 := by
    have := slot.isLt
    simp only [columnsPerSbox] at this
    omega
  have inner : ∃ row ∈ sboxRows (frameAt layout index (schedule index)),
      row.c = [(sboxColumn layout index slot, 1)] := by
    have choice : slot.val = 0 ∨ slot.val = 1 ∨ slot.val = 2 ∨ slot.val = 3 := by
      omega
    rcases choice with slotIs | slotIs | slotIs | slotIs
    · exact ⟨rowSquare (frameAt layout index (schedule index)),
        by simp [sboxRows],
        by simp [rowSquare, frameAt, sboxColumn, slotIs]⟩
    · exact ⟨rowFourth (frameAt layout index (schedule index)),
        by simp [sboxRows],
        by simp [rowFourth, frameAt, sboxColumn, slotIs]⟩
    · exact ⟨rowSixth (frameAt layout index (schedule index)),
        by simp [sboxRows],
        by simp [rowSixth, frameAt, sboxColumn, slotIs]⟩
    · exact ⟨rowSeventh (frameAt layout index (schedule index)),
        by simp [sboxRows],
        by simp [rowSeventh, frameAt, sboxColumn, slotIs]⟩
  rcases inner with ⟨row, rowMember, mentions⟩
  exact ⟨row, List.mem_flatMap.2 ⟨index, List.mem_finRange _, rowMember⟩,
    mentions⟩

/-- **Every declared auxiliary column is written.**  The list form. -/
theorem sboxProgram_writes_auxiliaryColumns
    (layout : Layout) (schedule : Schedule)
    (column : Nat) (member : column ∈ auxiliaryColumns layout) :
    ∃ row ∈ sboxProgram layout schedule, row.c = [(column, 1)] := by
  unfold auxiliaryColumns at member
  rcases List.mem_flatMap.1 member with ⟨index, _, inIndex⟩
  rcases List.mem_map.1 inIndex with ⟨slot, _, rfl⟩
  exact sboxProgram_writes_sboxColumn layout schedule index slot

/-- Terminal binding of the final state to declared output ports. -/
def bindingProgram (layout : Layout) (final : State) : List Row :=
  terminalBindingRows final layout.outputPort

theorem bindingProgram_length (layout : Layout) (final : State) :
    (bindingProgram layout final).length = width := by
  simpa [bindingProgram] using terminalBindingRows_length final layout.outputPort

/-! ## The assembled program -/

def permutationProgram (layout : Layout) (schedule : Schedule) (final : State) :
    List Row :=
  sboxProgram layout schedule ++ bindingProgram layout final

/-- **Every declared auxiliary column is written by the assembled program.** -/
theorem permutationProgram_writes_auxiliaryColumns
    (layout : Layout) (schedule : Schedule) (final : State)
    (column : Nat) (member : column ∈ auxiliaryColumns layout) :
    ∃ row ∈ permutationProgram layout schedule final,
      row.c = [(column, 1)] := by
  rcases sboxProgram_writes_auxiliaryColumns layout schedule column member with
    ⟨row, rowMember, write⟩
  exact ⟨row, List.mem_append_left _ rowMember, write⟩

/-- **Row count from the assembled receipts.** -/
theorem permutationProgram_length
    (layout : Layout) (schedule : Schedule) (final : State) :
    (permutationProgram layout schedule final).length = sboxCount * 4 + width := by
  simp [permutationProgram, sboxProgram_length, bindingProgram_length]

theorem permutationProgram_length_eq
    (layout : Layout) (schedule : Schedule) (final : State) :
    (permutationProgram layout schedule final).length = 352 := by
  rw [permutationProgram_length]; decide

/-! ## Exact ownership -/

/-- **Every row has at least one owner**: an S-box index, or the terminal
binding for one lane.

This is COVERAGE, not uniqueness — the conclusion is a disjunction of
existentials with no clause forbidding two receipts from emitting the same row
value.  The column theorem below is the stronger one; it carries a `∀ otherIndex
otherSlot` clause and does establish uniqueness.  Row uniqueness needs distinct
receipts to emit distinct row VALUES, which follows from column distinctness but
is not proved here: `POSEIDON2-ROW-OWNERSHIP-UNIQUENESS`. -/
theorem everyPermutationRow_has_owner
    (layout : Layout) (schedule : Schedule) (final : State)
    (row : Row) (member : row ∈ permutationProgram layout schedule final) :
    (∃ index : Fin sboxCount,
        row ∈ sboxRows (frameAt layout index (schedule index))) ∨
      (∃ lane : Fin width,
        row = bindRow (final lane) (layout.outputPort lane)) := by
  rcases List.mem_append.1 member with inSbox | inBinding
  · rcases List.mem_flatMap.1 inSbox with ⟨index, _, rowMember⟩
    exact Or.inl ⟨index, rowMember⟩
  · rcases List.mem_map.1 inBinding with ⟨lane, _, rfl⟩
    exact Or.inr ⟨lane, rfl⟩

/-- **Every auxiliary column has exactly one owner.** -/
theorem everyPermutationColumn_has_exact_owner
    (layout : Layout) (column : Nat)
    (member : column ∈ auxiliaryColumns layout) :
    ∃ index : Fin sboxCount, ∃ slot : Fin columnsPerSbox,
      column = sboxColumn layout index slot ∧
        ∀ otherIndex otherSlot,
          column = sboxColumn layout otherIndex otherSlot →
            otherIndex = index ∧ otherSlot = slot := by
  rcases List.mem_flatMap.1 member with ⟨index, _, slotMember⟩
  rcases List.mem_map.1 slotMember with ⟨slot, _, rfl⟩
  refine ⟨index, slot, rfl, ?_⟩
  intro otherIndex otherSlot equal
  exact sboxColumn_injective layout equal.symm

/-! ## Composed S-box soundness

Satisfaction of the assembled program gives every S-box's addition chain.  The
remaining semantic obligation — that the scheduled input combinations are the
matrix images — is `POSEIDON2-ROUND-INDUCTION` and is not discharged here. -/

theorem permutationProgram_sbox_chains
    (layout : Layout) (schedule : Schedule) (final : State)
    (z : Nat → Nat) (residues : ∀ column, z column < goldilocksP)
    (satisfied : Satisfies (permutationProgram layout schedule final) z)
    (index : Fin sboxCount) :
    let frame := frameAt layout index (schedule index)
    z frame.square = lcEval z frame.input * lcEval z frame.input % goldilocksP ∧
      z frame.fourth = z frame.square * z frame.square % goldilocksP ∧
      z frame.sixth = z frame.square * z frame.fourth % goldilocksP ∧
      z frame.output = lcEval z frame.input * z frame.sixth % goldilocksP := by
  refine sboxRows_chain _ z residues ?_
  intro row rowMember
  refine satisfied row (List.mem_append.2 (Or.inl ?_))
  exact List.mem_flatMap.2 ⟨index, List.mem_finRange index, rowMember⟩

/-! ## Assembled cost

Both components now come from the receipts. -/

def permutationCost (layout : Layout) (schedule : Schedule) (final : State) :
    Typed.Cost :=
  ⟨(permutationProgram layout schedule final).length, 0, 0,
    (auxiliaryColumns layout).length⟩

/-- **The assembled permutation cost**, every component derived from the
receipt fold rather than declared. -/
theorem permutationProgram_cost_eq_receiptFold
    (layout : Layout) (schedule : Schedule) (final : State) :
    permutationCost layout schedule final = ⟨352, 0, 0, 344⟩ := by
  unfold permutationCost
  rw [permutationProgram_length_eq, auxiliaryColumns_length_eq]

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
