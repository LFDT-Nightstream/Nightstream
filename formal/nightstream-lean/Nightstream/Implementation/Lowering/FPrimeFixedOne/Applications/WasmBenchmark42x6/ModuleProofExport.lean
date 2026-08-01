import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.ApplicationProofManifest
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleExport
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofProgram

/-!
Contract: exact native-CCS proof manifest for the 42-times-6 module.

Assurance tier: model-level.

Owns: the complete ordered role binding for all sixty-four physical columns,
exact cost and density facts, and deterministic JSON derived from the
Lean-owned module and native-CCS program.

Does not own: file I/O, Rust parsing, Spartan, WHIR, recursive F-prime, or a
security reduction.

Emits constraints: none. It serializes the exact 63-row program.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofExport

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.ApplicationProofManifest
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofProgram

def bindings : List ColumnBinding :=
  { column := oneColumn, role := .one } ::
    (List.ofFn fun index : Fin moduleByteCount =>
      { column := moduleByteColumn index, role := .moduleByte index.val }) ++
    [ { column := productColumn, role := .privateWitness 0 }
    , { column := outputColumn, role := .output 0 }
    ]

theorem bindings_cover_program :
    bindings.map ColumnBinding.column = program.columnIds := by
  decide

theorem bindings_nodup :
    (bindings.map ColumnBinding.column).Nodup := by
  rw [bindings_cover_program]
  exact program_columnIds_nodup

theorem exact_r1cs_nonzero_coefficients :
    r1csNonzeroCoefficients program = 173 := by
  decide

theorem exact_native_ccs_nonzero_coefficients :
    nativeCcsNonzeroCoefficients program = 236 := by
  decide

theorem exact_maximum_r1cs_row_density :
    maximumR1csRowDensity program = 3 := by
  decide

theorem exact_maximum_native_ccs_row_density :
    maximumNativeCcsRowDensity program = 4 := by
  decide

def render : String :=
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.ApplicationProofManifest.render
    ModuleExport.identifier certifiedModule program valid bindings 0 1

theorem render_exact :
    render =
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.ApplicationProofManifest.render
        ModuleExport.identifier certifiedModule program valid bindings 0 1 :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofExport
