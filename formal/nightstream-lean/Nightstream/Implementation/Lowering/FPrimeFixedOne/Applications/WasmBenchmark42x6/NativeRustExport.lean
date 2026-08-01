import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointFamily
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifestJson

/-!
Exact native-CCS JSON export for the selected 42-times-6 WASM benchmark.

Owns: the direct path from the proof-carrying benchmark deployment to the
schema-versioned native CCS bytes consumed by Rust, and the exact selected
Step cost at the matrix-count-four recursive fixed point.

Does not own: file I/O, deployment seed selection, witness generation, or
authority for operator-selected bytes.

Emits constraints: no. It serializes the exact Lean manifest.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeRustExport

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointFamily
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointSource
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifestJson
open Nightstream.Implementation.R1CS.Canonical

/-- Deterministic schema-v3 JSON for the exact compiler-produced benchmark
fixed point under one setup-owned verifier key. No caller-selected relation
matrices and no legacy activated Step rows are present. -/
noncomputable def render
    (template : Template) : String :=
  ConcreteNifsNativeCcsManifestJson.render
    template.ajtai.identity
    (nativeManifest (finalSetup template))

theorem render_exact
    (template : Template) :
    render template =
      ConcreteNifsNativeCcsManifestJson.render
        template.ajtai.identity
        (nativeManifest (finalSetup template)) :=
  rfl

/-- The exported Step cost is the exact receipt-derived native fixed-point
cost. -/
theorem stepCost_exact
    (template : Template) :
    (nativeManifest (finalSetup template)).stepCost =
      ⟨5_299_490, 143_428, 6, 5_211_101⟩ := by
  change
    (NativeFixedPointCost.nativeProgram (finalSetup template)).cost =
      ⟨5_299_490, 143_428, 6, 5_211_101⟩
  exact NativeFixedPointCost.nativeCost_exact
    (finalSetup template) (finalSetup_polynomial template)

theorem matrixCount_exact
    (template : Template) :
    (nativeManifest (finalSetup template)).stepProgram.matrixCount = 4 :=
  rfl

theorem polynomialDegree_exact
    (template : Template) :
    (nativeManifest (finalSetup template)).stepProgram.polynomialDegree = 3 :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeRustExport
