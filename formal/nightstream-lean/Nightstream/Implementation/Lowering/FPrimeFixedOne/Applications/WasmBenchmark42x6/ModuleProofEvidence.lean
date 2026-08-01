import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofR1csLowering

/-!
Contract: one assembled M4 evidence value for the 42-times-6 application
module proof.

Assurance tier: model-level.

Owns: exact rows, columns, occurrence ownership, receipt-derived cost,
native-CCS-to-R1CS identity, soundness, and honest completeness for this
module proof.

Does not own: the 19-million-row recursive F-prime relation, Rust execution,
Spartan, WHIR, Fiat--Shamir security, or polynomial-commitment binding.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofEvidence

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofProgram
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofR1csLowering

private abbrev Field := Nightstream.SuperNeo.Concrete.F

/-- Complete model-level evidence for the small application module relation.
The application proof is separate from the recursive F-prime terminal proof. -/
structure Evidence where
  rowsExact : program.rows.length = 63
  r1csRowsExact : rows.length = 63
  columnsUnique : program.columnIds.Nodup
  rowOccurrencesUnique : program.rowIds.Nodup
  costExact : program.cost = ⟨63, 0, 63, 1⟩
  selectedRowsExact :
    NativeCcsSelector.select oneColumn rows = program.rows
  manifestRowsExact :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.ApplicationProofManifest.normalizedRows
        program =
      rows.map CanonicalManifest.ManifestRow.ofOwnedRow
  sound :
    ∀ (assignment : ColumnId → Field),
      assignment oneColumn = 1 →
      Goldilocks.Satisfies rows assignment →
      (∀ index,
        assignment (moduleByteColumn index) = moduleByteValue index) ∧
        assignment outputColumn = 252 ∧
        module.run = some (assignment outputColumn).val
  honest : Goldilocks.Satisfies rows honestAssignment

def m4 : Evidence where
  rowsExact := program_rows_length
  r1csRowsExact := rows_length
  columnsUnique := program_columnIds_nodup
  rowOccurrencesUnique := program_rowIds_nodup
  costExact := program_cost_exact
  selectedRowsExact := selected_rows_exact
  manifestRowsExact := manifest_rows_exact
  sound := ModuleProofR1csLowering.soundness
  honest := ModuleProofR1csLowering.honest_satisfies

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofEvidence
