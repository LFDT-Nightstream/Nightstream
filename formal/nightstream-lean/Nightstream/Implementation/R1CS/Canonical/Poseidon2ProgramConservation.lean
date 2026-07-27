import Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership

/-!
Contract: conservation for every operand of every emitted row.

Owns: the assembled whole-program statement that `Poseidon2Conservation` left
as ingredients.

## What was missing

`Poseidon2Conservation` bounds the columns of *scheduled inputs* and *final
states*. Emitted rows also carry frame columns — `square`, `fourth`, `sixth`,
`output` — and output ports, and nothing quantified over every operand of every
row. Cycle 182 claimed the §2 checklist was complete on the strength of the
ingredients; cycle 197 withdrew that. This module closes it properly.

The case analysis runs over `RowOwner` rather than over row values, reusing the
positional decomposition from `Poseidon2Ownership`: the program *is* the owner
list's image, so bounding every owner's row bounds every row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2ProgramConservation

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

theorem singleton_mentions_lt
    (column target : Nat) (bound : column < canonicalColumnTotal)
    (mentioned : Mentions [(column, 1)] target) :
    target < canonicalColumnTotal := by
  simp only [Mentions, List.map_cons, List.map_nil,
    List.mem_singleton] at mentioned
  rw [mentioned]; exact bound

theorem constantWire_lt : (0 : Nat) < canonicalColumnTotal := by
  simp only [canonicalColumnTotal]; decide

/-- **Every operand of every receipt's row stays inside the column space.** -/
theorem ownedRow_operand_lt
    (constants : Constants) (owner : RowOwner) (column : Nat)
    (mentioned :
      Mentions (ownedRow canonicalLayout constants owner).a column ∨
        Mentions (ownedRow canonicalLayout constants owner).b column ∨
        Mentions (ownedRow canonicalLayout constants owner).c column) :
    column < canonicalColumnTotal := by
  cases owner with
  | sbox index step =>
      have scheduled := scheduleOf_conservation constants index column
      have frameBound : ∀ slot : Fin columnsPerSbox,
          sboxColumn canonicalLayout index slot < canonicalColumnTotal :=
        fun slot => canonicalLayout_sboxColumn_lt index slot
      match step with
      | ⟨0, _⟩ =>
          simp only [ownedRow, sboxRowAt, rowSquare, frameAt] at mentioned
          rcases mentioned with input | input | target
          · exact scheduled input
          · exact scheduled input
          · exact singleton_mentions_lt _ _ (frameBound ⟨0, by decide⟩) target
      | ⟨1, _⟩ =>
          simp only [ownedRow, sboxRowAt, rowFourth, frameAt] at mentioned
          rcases mentioned with target | target | target
          · exact singleton_mentions_lt _ _ (frameBound ⟨0, by decide⟩) target
          · exact singleton_mentions_lt _ _ (frameBound ⟨0, by decide⟩) target
          · exact singleton_mentions_lt _ _ (frameBound ⟨1, by decide⟩) target
      | ⟨2, _⟩ =>
          simp only [ownedRow, sboxRowAt, rowSixth, frameAt] at mentioned
          rcases mentioned with target | target | target
          · exact singleton_mentions_lt _ _ (frameBound ⟨0, by decide⟩) target
          · exact singleton_mentions_lt _ _ (frameBound ⟨1, by decide⟩) target
          · exact singleton_mentions_lt _ _ (frameBound ⟨2, by decide⟩) target
      | ⟨3, _⟩ =>
          simp only [ownedRow, sboxRowAt, rowSeventh, frameAt] at mentioned
          rcases mentioned with input | target | target
          · exact scheduled input
          · exact singleton_mentions_lt _ _ (frameBound ⟨2, by decide⟩) target
          · exact singleton_mentions_lt _ _ (frameBound ⟨3, by decide⟩) target
  | binding lane =>
      simp only [ownedRow, bindRow] at mentioned
      rcases mentioned with final | wire | port
      · exact finalState_conservation lane column final
      · exact singleton_mentions_lt _ _ constantWire_lt wire
      · exact singleton_mentions_lt _ _
          (canonicalLayout_outputPort_lt lane) port

/-- **Whole-program conservation.**  No row of the emitted program touches a
column outside the declared space of constant wire, eight inputs, eight outputs
and 344 auxiliaries.

This is the statement cycle 182 claimed on the strength of ingredients and cycle
197 withdrew.  It runs through `RowOwner`, so it inherits the positional
decomposition rather than reasoning about row values, and through
`mentions_normalizeRow`, which gives exactly the direction needed: field
normalization introduces no column, so bounding the raw rows bounds the emitted
ones. -/
theorem normalizedCanonicalProgram_conservation
    (constants : Constants) (row : Row)
    (member : row ∈ normalizedCanonicalProgram canonicalLayout constants)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column < canonicalColumnTotal := by
  rw [normalizedCanonicalProgram_eq_map_owners] at member
  rcases List.mem_map.1 member with ⟨owner, _, rfl⟩
  refine ownedRow_operand_lt constants owner column ?_
  rcases mentioned with inA | inB | inC
  · exact Or.inl ((mentions_normalizeRow _ column).1 inA)
  · exact Or.inr (Or.inl ((mentions_normalizeRow _ column).2.1 inB))
  · exact Or.inr (Or.inr ((mentions_normalizeRow _ column).2.2 inC))

end Nightstream.Implementation.R1CS.Canonical.Poseidon2ProgramConservation
