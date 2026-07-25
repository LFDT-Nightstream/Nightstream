import Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Generated.RowsTwo
import Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement

/-!
Contract: bounded physical-row refinement for the production terminal-link
emitter.

Owns exact equality between the Rust-captured two-claim row list and the
selected arbitrary-batch receipt compiler, including row/column counts and
unique receipt ownership. Does not generalize the capture to all Rust batch
sizes, formalize compiled Rust semantics, bind producer digest computation,
or place this block inside the full-history artifact.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement

open Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch
open Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program
open Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement

set_option maxRecDepth 32768

theorem generated_batchSize_eq :
    Generated.RowsTwo.batchSize = 2 := by
  rfl

theorem generated_rowCount_eq :
    Generated.RowsTwo.rowCount = rowCount 2 := by
  decide

theorem generated_columnCount_eq :
    Generated.RowsTwo.columnCount = columnCount 2 := by
  decide

/-- The literal sparse rows captured from the private production emitter are
exactly the rows compiled from the selected two-claim receipts. -/
theorem generated_rows_eq_selected :
    Generated.RowsTwo.rows = rows 2 := by
  decide

/-- The physical row capture is literally the checked compiler output for the
Rust-emitted typed source program at the captured batch size. -/
theorem generated_rows_eq_compiler_output :
    compile generatedPlain Generated.RowsTwo.batchSize =
      some Generated.RowsTwo.rows := by
  rw [generated_batchSize_eq, generated_rows_eq_selected]
  exact generated_plain_compile 2

/-- Every captured physical row has exactly one selected typed receipt owner,
and every selected receipt owns one captured row. -/
theorem generated_rows_exact_receipt_ownership :
    ∃ owner :
        Receipt 2 → Fin Generated.RowsTwo.rows.length,
      Function.Injective owner ∧ Function.Surjective owner := by
  rw [generated_rows_eq_selected]
  exact
    ⟨physicalIndex,
      physicalIndex_injective 2,
      physicalIndex_surjective 2⟩

end Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement
