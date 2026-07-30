import Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
import Nightstream.Implementation.R1CS.Canonical.Poseidon2ProgramConservation
import Nightstream.Implementation.R1CS.Canonical.KMulHonest

/-!
Contract: the canonical Poseidon2 program's satisfying assignments are exactly
its honest executions.

Owns: the soundness-side scheduled-input bridge, witness uniqueness on every
allocated column, assignment congruence for row programs, and the biconditional
those three assemble into.

## Why a biconditional, and which one

`POSEIDON2-ROUND-INDUCTION` was carried for many cycles as
`permutationProgram_exec_iff_spec`. The obvious reading of that name —

```text
Satisfies program z ↔ outputs z = reference
```

— is **false of this program and should be**. An assignment can carry correct
output ports while carrying wrong intermediate S-box columns: it hits the right
spec value and fails the row program. A biconditional in that shape would say
the encoding accepts assignments it must reject.

The true statement in that shape is stronger, not weaker: satisfaction pins
**every** column of the declared space, not just the output ports. So

```text
Satisfies program z ↔ z agrees with the honest execution everywhere
```

which is witness uniqueness in one direction and honest completeness in the
other. `Poseidon2Honest.honest_directions_agree` already recorded that it was
*not* uniqueness; this module supplies what it disclaimed.

## What makes uniqueness available

`Poseidon2Schedule.canonicalProgram_sbox_chains` forces all four chain columns
per S-box — square, fourth, sixth and output — not merely the output. Every
allocated column of the program is one of those four or an output port, so
pinning them pins the whole witness. Had the chain lemma forced only the S-box
output, uniqueness would be false and the honest route would be the only one.

## The bridge that had to be built

Soundness needs `lcEval z (scheduleOf i) = sboxInputValue constants (inputValues z) i`
for an arbitrary satisfying `z`. `Poseidon2Honest.honest_scheduleOf` proves that
for the honest assignment only, through `honest_initialState` and its siblings.
`scheduleOf_eval` below is its mirror through the round inductions
`initialState_eval`, `partialState_eval` and `terminalState_eval`, which is the
one piece neither existing direction had.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Uniqueness

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Support
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program

/-! ## Assignment congruence

Satisfaction depends on an assignment only through the columns the rows
mention. This is what lets the honest execution's satisfaction transport to any
assignment that agrees with it. -/

theorem rowHolds_congr (z z' : Nat → Nat) (row : Row)
    (agree : ∀ column,
      Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column →
        z column = z' column) :
    RowHolds z row ↔ RowHolds z' row := by
  unfold RowHolds
  rw [KMulHonest.lcEval_congr z z' row.a (fun c m => agree c (Or.inl m)),
    KMulHonest.lcEval_congr z z' row.b (fun c m => agree c (Or.inr (Or.inl m))),
    KMulHonest.lcEval_congr z z' row.c (fun c m => agree c (Or.inr (Or.inr m)))]

theorem satisfies_congr (rows : List Row) (z z' : Nat → Nat)
    (agree : ∀ row ∈ rows, ∀ column,
      Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column →
        z column = z' column) :
    Satisfies rows z ↔ Satisfies rows z' := by
  constructor
  · intro satisfied row member
    exact (rowHolds_congr z z' row (agree row member)).1 (satisfied row member)
  · intro satisfied row member
    exact (rowHolds_congr z z' row (agree row member)).2 (satisfied row member)

/-! ## Whole-program conservation for the raw program

`Poseidon2ProgramConservation` states this for the normalized program; the raw
one follows from the same receipt decomposition. -/

theorem canonicalProgram_conservation
    (constants : Constants) (row : Row)
    (member : row ∈ canonicalProgram canonicalLayout constants)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column < canonicalColumnTotal := by
  rw [Poseidon2Ownership.canonicalProgram_eq_map_owners] at member
  rcases List.mem_map.1 member with ⟨owner, _, rfl⟩
  exact Poseidon2ProgramConservation.ownedRow_operand_lt constants owner column
    mentioned

/-! ## The soundness-side scheduled-input bridge

Mirrors `Poseidon2Honest.honest_scheduleOf`, with the round inductions in place
of the honest state lemmas. -/

theorem scheduleOf_eval
    (constants : Constants) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP) (constantWire : z 0 = 1)
    (chain : SboxChain canonicalLayout constants z) (index : Fin sboxCount) :
    lcEval z (scheduleOf canonicalLayout constants index)
      = sboxInputValue constants (inputValues canonicalLayout z) index.val := by
  have indexLt : index.val < sboxCount := index.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  by_cases isInitial : index.val < 32
  · have laneLt : index.val % 8 < width := by simp only [width]; omega
    have roundLt : index.val / 8 < halfFullRounds := by
      simp only [halfFullRounds]; omega
    have isIdx : index.val
        = initialSboxIndex (index.val / 8)
            (⟨index.val % 8, laneLt⟩ : Fin width).val := by
      simp only [initialSboxIndex, width]; omega
    rw [scheduleOf_initial canonicalLayout constants index (index.val / 8)
        ⟨index.val % 8, laneLt⟩ isIdx roundLt,
      initialSboxInput, lcEval_addConstant _ _ _ constantWire,
      initialState_eval canonicalLayout constants z residues constantWire chain
        _ (Nat.le_of_lt roundLt),
      ← sboxInputValue_initial constants (inputValues canonicalLayout z)
        (index.val / 8) ⟨index.val % 8, laneLt⟩ roundLt, ← isIdx]
  · by_cases isPartial : index.val < 54
    · have roundLt : index.val - 32 < partialRounds := by
        simp only [partialRounds]; omega
      have isIdx : index.val = 32 + (index.val - 32) := by omega
      rw [Poseidon2Support.scheduleOf_partial canonicalLayout constants index
          (index.val - 32) isIdx roundLt,
        Poseidon2Support.partialSboxInput,
        lcEval_addConstant _ _ _ constantWire,
        partialState_eval canonicalLayout constants z residues constantWire
          chain _ (Nat.le_of_lt roundLt),
        ← sboxInputValue_partial constants (inputValues canonicalLayout z)
          (index.val - 32) roundLt]
      congr 2
      simp only [partialSboxIndex, halfFullRounds, width]; omega
    · have laneLt : (index.val - 54) % 8 < width := by simp only [width]; omega
      have roundLt : (index.val - 54) / 8 < halfFullRounds := by
        simp only [halfFullRounds]; omega
      have isIdx : index.val = terminalSboxIndex ((index.val - 54) / 8)
          (⟨(index.val - 54) % 8, laneLt⟩ : Fin width).val := by
        simp only [terminalSboxIndex, halfFullRounds, width, partialRounds]
        omega
      rw [scheduleOf_terminal canonicalLayout constants index
          ((index.val - 54) / 8) ⟨(index.val - 54) % 8, laneLt⟩ isIdx roundLt,
        terminalSboxInput, lcEval_addConstant _ _ _ constantWire,
        terminalState_eval canonicalLayout constants z residues constantWire
          chain _ (Nat.le_of_lt roundLt),
        ← sboxInputValue_terminal constants (inputValues canonicalLayout z)
          ((index.val - 54) / 8) ⟨(index.val - 54) % 8, laneLt⟩ roundLt, ← isIdx]


/-! ## Witness uniqueness

Every allocated column is forced.  `canonicalProgram_sbox_chains` supplies all
four chain values per S-box, and `scheduleOf_eval` turns the combination each
consumes into the reference-derived value the honest witness is built from. -/

/-- **Every S-box chain column is forced to its honest value.**

Not just the S-box output: the square, fourth and sixth intermediates are pinned
too.  That is what makes uniqueness true rather than merely the outputs
matching. -/
theorem sboxColumn_forced
    (constants : Constants) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP) (constantWire : z 0 = 1)
    (satisfied : Satisfies (canonicalProgram canonicalLayout constants) z)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox) :
    z (sboxColumn canonicalLayout index slot)
      = chainSlot
          (sboxInputValue constants (inputValues canonicalLayout z) index.val)
          slot.val := by
  obtain ⟨square, fourth, sixth, output⟩ :=
    canonicalProgram_sbox_chains canonicalLayout constants z residues satisfied
      index
  simp only [frameAt] at square fourth sixth output
  have bridge := scheduleOf_eval constants z residues constantWire
    (satisfies_sboxChain canonicalLayout constants z residues satisfied) index
  have slotLt : slot.val < columnsPerSbox := slot.isLt
  simp only [columnsPerSbox] at slotLt
  match slot with
  | ⟨0, _⟩ =>
      rw [show (⟨0, by decide⟩ : Fin columnsPerSbox) = ⟨0, by decide⟩ from rfl,
        square, bridge]
      rfl
  | ⟨1, _⟩ =>
      rw [fourth, square, bridge]
      rfl
  | ⟨2, _⟩ =>
      rw [sixth, fourth, square, bridge]
      rfl
  | ⟨3, _⟩ =>
      rw [output, sixth, fourth, square, bridge]
      rfl

/-! ## The biconditional

`POSEIDON2-ROUND-INDUCTION`, in the only shape that is both true and worth
having. -/

/-- **The canonical program's satisfying assignments are exactly its honest
executions.**

Forward is witness uniqueness: satisfaction pins every column of the declared
space — constant wire, eight inputs, eight outputs and 344 S-box auxiliaries —
to the value the honest execution puts there.

Backward is honest completeness transported: an assignment that agrees with the
honest execution everywhere the program can see satisfies it.

The input ports are shared reads rather than allocations, so their arm is
definitional: `inputValues` is the assignment restricted to them, and the honest
execution is built from exactly that. Uniqueness is therefore a statement about
the 352 allocated columns, given the inputs. -/
theorem canonicalProgram_exec_iff_spec
    (constants : Constants) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP) (constantWire : z 0 = 1) :
    Satisfies (canonicalProgram canonicalLayout constants) z
      ↔ ∀ column, column < canonicalColumnTotal →
          z column
            = honestAssignment constants (inputValues canonicalLayout z)
                column := by
  constructor
  · intro satisfied column columnLt
    simp only [canonicalColumnTotal, width, sboxCount, externalRounds,
      partialRounds, columnsPerSbox] at columnLt
    by_cases isWire : column = 0
    · subst isWire
      rw [constantWire]
      rfl
    · by_cases isInput : column < 9
      · have laneLt : column - 1 < width := by simp only [width]; omega
        have portEq :
            canonicalLayout.inputPort ⟨column - 1, laneLt⟩ = column := by
          show 1 + (column - 1) = column
          omega
        rw [← portEq, honest_inputPort]
        rfl
      · by_cases isOutput : column < 17
        · have laneLt : column - 9 < width := by simp only [width]; omega
          have portEq :
              canonicalLayout.outputPort ⟨column - 9, laneLt⟩ = column := by
            show 9 + (column - 9) = column
            omega
          rw [← portEq, honest_outputPort]
          exact canonicalProgram_computes_reference canonicalLayout constants z
            residues constantWire satisfied ⟨column - 9, laneLt⟩
        · have indexLt : (column - 17) / 4 < sboxCount := by
            simp only [sboxCount, externalRounds, width, partialRounds]; omega
          have slotLt : (column - 17) % 4 < columnsPerSbox := by
            simp only [columnsPerSbox]; omega
          have colEq :
              sboxColumn canonicalLayout ⟨(column - 17) / 4, indexLt⟩
                  ⟨(column - 17) % 4, slotLt⟩ = column := by
            show 17 + 4 * ((column - 17) / 4) + (column - 17) % 4 = column
            omega
          rw [← colEq, honest_sboxColumn]
          exact sboxColumn_forced constants z residues constantWire satisfied
            ⟨(column - 17) / 4, indexLt⟩ ⟨(column - 17) % 4, slotLt⟩
  · intro agrees
    refine (satisfies_congr (canonicalProgram canonicalLayout constants) z
      (honestAssignment constants (inputValues canonicalLayout z))
      (fun row member column mentioned =>
        agrees column
          (canonicalProgram_conservation constants row member column
            mentioned))).2 ?_
    exact honest_satisfies constants (inputValues canonicalLayout z)
      (fun lane => residues _)

/-! ## The name the phase plan uses

`Poseidon2Program`'s docstring forward-references this obligation as
`permutationProgram_exec_iff_spec`, and the phase plan names it the same way.
`canonicalProgram` *is* `permutationProgram` at the canonical schedule and
final state, so the theorem is restated at that name and that spelling.

The restatement is definitional in the *program*, not in the *claim*: what makes
it true is uniqueness, proved above. Only the way the program is spelled changes.
-/

/-- **`POSEIDON2-ROUND-INDUCTION`, at the name the phase plan uses.**

Identical content to `canonicalProgram_exec_iff_spec`, spelled at
`permutationProgram` because that is the name the plan refers to. -/
theorem permutationProgram_exec_iff_spec
    (constants : Constants) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP) (constantWire : z 0 = 1) :
    Satisfies (permutationProgram canonicalLayout
        (scheduleOf canonicalLayout constants)
        (finalState canonicalLayout)) z
      ↔ ∀ column, column < canonicalColumnTotal →
          z column
            = honestAssignment constants (inputValues canonicalLayout z)
                column :=
  canonicalProgram_exec_iff_spec constants z residues constantWire

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Uniqueness
