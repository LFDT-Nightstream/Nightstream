import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Cost
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Boundary
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Manifest

/-!
Contract: the no-op optimizer boundary for the selected 42-times-6 WASM
benchmark.

Assurance tier: model-level.

Owns: the exact canonical manifest, its replacement theorem, and its exact
current cost at the benchmark relation polynomial.

Does not own: output or transcript column selection, a cost reduction,
native CCS, a concrete setup value, Rust, or a security reduction.

Emits constraints: the current normalized canonical Step rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Optimization.Identity

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Cost
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.Optimization
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private abbrev Assignment :=
  Nightstream.Implementation.Lowering.Goldilocks.Optimization.R1CS.Assignment

/-- The default physical boundary protects every committed and public
coordinate. Output and transcript roles are added by later selected passes,
where their exact columns are available. -/
noncomputable def boundary
    (setup : RelationSetup dimensions commitmentRows) :
    Boundary.Columns :=
  Boundary.ofEncoding (encoding setup) [] []

noncomputable def observe
    (setup : RelationSetup dimensions commitmentRows) :
    Assignment -> Boundary.Values :=
  Boundary.values (boundary setup)

noncomputable def sourceSystem
    (setup : RelationSetup dimensions commitmentRows) :=
  R1CS.ofEncoding (encoding setup) (observe setup)

noncomputable def manifest
    (setup : RelationSetup dimensions commitmentRows) :=
  CanonicalManifest.Program.ofEncoding (encoding setup)

noncomputable def targetSystem
    (setup : RelationSetup dimensions commitmentRows) :=
  Manifest.decodedSystem (manifest setup) (observe setup)

/-- Degree three is the selected pipeline limit. The identity stage itself
remains degree two. -/
noncomputable def replacement
    (setup : RelationSetup dimensions commitmentRows) :
    Replacement (sourceSystem setup) (targetSystem setup) 3 :=
  Manifest.ofEncodingReplacement (encoding setup) (observe setup) 3
    (by decide)

theorem manifest_cost_exact
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial) :
    (manifest setup).cost =
      ⟨19_859_562, 244_058, 6, 19_725_249⟩ := by
  change
    (CanonicalManifest.Program.ofEncoding (encoding setup)).cost =
      ⟨19_859_562, 244_058, 6, 19_725_249⟩
  rw [Manifest.cost_exact, encodingCost_exact setup polynomialExact]

theorem manifest_rows_exact
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial) :
    (manifest setup).rows.length = 19_859_562 := by
  change
    (CanonicalManifest.Program.ofEncoding (encoding setup)).rows.length =
      19_859_562
  calc
    _ = (encoding setup).rows.length :=
      Manifest.rows_exact (encoding setup)
    _ = (EncodingRows.program (encoding setup)).length :=
      (EncodingRows.program_length (encoding setup)).symm
    _ = 19_859_562 :=
      encodingRows_exact setup polynomialExact

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Optimization.Identity
