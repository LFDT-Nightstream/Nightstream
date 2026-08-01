import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.ModuleManifest
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Module

/-!
Contract: compact module artifact for the 42-times-6 deployment.

Assurance tier: model-level.

Owns: the stable deployment identifier and exact JSON derived from the
Lean-owned certified module.

Does not own: the native CCS manifest, terminal R1CS, Rust, file I/O, or a
security reduction.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleExport

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm

def identifier : String := "wasm-benchmark-42x6"

def render : String :=
  ModuleManifest.render identifier certifiedModule

theorem render_exact :
    render = ModuleManifest.render identifier certifiedModule :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleExport
