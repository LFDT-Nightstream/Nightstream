import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofExport

/-!
Contract: exact terminal R1CS lowering of the 42-times-6 native CCS
application proof.

Assurance tier: model-level.

Owns: the ordered 63-row R1CS image, exact reconstruction of the native
selected rows, satisfaction equivalence under the verifier-fixed one column,
and transport of soundness and honest completeness.

Does not own: a general CCS-to-R1CS compiler, Rust matrix construction,
Spartan, WHIR, recursive F-prime, or a security reduction.

Emits constraints: none. It removes a constant-one selector from the existing
rows and does not add a row or column.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofR1csLowering

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofProgram

private abbrev Field := Nightstream.SuperNeo.Concrete.F

/-- The terminal backend receives the source equations in their original
order. No activation row or residual column is introduced. -/
def rows : List OwnedRow :=
  program.rows.map SelectedRow.source

@[simp] theorem rows_length : rows.length = 63 := by
  simp [rows]

/-- Reattaching the verifier-fixed one selector reconstructs the exact
Lean-owned native CCS row stream. -/
theorem selected_rows_exact :
    select oneColumn rows = program.rows := by
  decide

/-- The manifest serializes these same source equations in the same order. -/
theorem manifest_rows_exact :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.ApplicationProofManifest.normalizedRows
        program =
      rows.map ManifestRow.ofOwnedRow :=
  rfl

/-- For the verifier-fixed active selector, native CCS satisfaction and
terminal R1CS satisfaction are exactly equivalent. -/
theorem satisfies_iff
    (assignment : ColumnId → Field)
    (oneExact : assignment oneColumn = 1) :
    program.Satisfies assignment ↔ Goldilocks.Satisfies rows assignment := by
  constructor
  · intro satisfied
    apply active_sound oneColumn rows assignment oneExact
    rw [selected_rows_exact]
    exact satisfied.2
  · intro satisfied
    refine ⟨oneExact, ?_⟩
    rw [← selected_rows_exact]
    exact complete oneColumn rows assignment satisfied

/-- Soundness survives the exact terminal lowering. -/
theorem soundness
    (assignment : ColumnId → Field)
    (oneExact : assignment oneColumn = 1)
    (satisfied : Goldilocks.Satisfies rows assignment) :
    (∀ index, assignment (moduleByteColumn index) = moduleByteValue index) ∧
      assignment outputColumn = 252 ∧
      module.run = some (assignment outputColumn).val := by
  exact ModuleProofProgram.soundness assignment
    ((satisfies_iff assignment oneExact).2 satisfied)

/-- The native honest assignment also satisfies the exact terminal rows. -/
theorem honest_satisfies :
    Goldilocks.Satisfies rows honestAssignment := by
  exact (satisfies_iff honestAssignment honestAssignment_one).1
    ModuleProofProgram.honest_satisfies

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofR1csLowering
