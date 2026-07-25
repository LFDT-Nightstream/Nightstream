import Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Generated.Program

/-!
Contract: artifact-checked refinement of the Rust-emitted terminal-link source
schedule into the selected arbitrary-batch row receipts.

Owns exact program equality, scalar and batch costs, and expansion into the
selected local receipt order. Does not own compiled Rust semantics, production
column placement, or whole-decider refinement.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch
open Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program

abbrev generatedPlain : Program :=
  Generated.Program.plain

theorem generated_plain_eq_canonical :
    generatedPlain = plain := by
  decide

theorem generated_plain_cost :
    cost generatedPlain = 270 := by
  rw [generated_plain_eq_canonical]
  exact plain_cost

theorem generated_plain_expansion :
    expand? generatedPlain = some selectedOwnerOrder := by
  rw [generated_plain_eq_canonical]
  exact plain_expansion

/-- The Rust-emitted typed program is accepted by the checked compiler and
produces the exact selected row list for every batch size. -/
theorem generated_plain_compile (batchSize : Nat) :
    compile generatedPlain batchSize = some (rows batchSize) := by
  rw [generated_plain_eq_canonical]
  exact compile_plain batchSize

/-- Acceptance of the Rust-emitted schedule is exactly satisfaction of the
selected receipt-owned rows. -/
theorem generated_plain_accepts_iff_selectedRows
    (batchSize : Nat)
    (assignment : Nat -> Nat) :
    Accepts generatedPlain batchSize assignment <->
      Satisfies (rows batchSize) assignment := by
  rw [generated_plain_eq_canonical]
  exact accepts_plain_iff batchSize assignment

def batchCost (program : Program) (batchSize : Nat) : Nat :=
  batchSize * cost program

/-- Program-derived recurring-row cost agrees with the selected batch
compiler for every batch size. -/
theorem generated_batchCost_eq_rowCount (batchSize : Nat) :
    batchCost generatedPlain batchSize = rowCount batchSize := by
  simp [batchCost, generated_plain_cost, rowCount]

/-- The generated source program expands to the selected owner order, whose
receipt map covers every physical row position exactly once. -/
theorem generated_program_exact_row_ownership (batchSize : Nat) :
    compile generatedPlain batchSize = some (rows batchSize) ∧
      Function.Injective
        (physicalIndex :
          Receipt batchSize → Fin (rows batchSize).length) ∧
      Function.Surjective
        (physicalIndex :
          Receipt batchSize → Fin (rows batchSize).length) := by
  exact
    ⟨generated_plain_compile batchSize,
      physicalIndex_injective batchSize,
      physicalIndex_surjective batchSize⟩

end Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement
