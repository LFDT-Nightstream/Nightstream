import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.StepRecipeCore

/-!
Contract: certified Step call for the 42-times-6 WASM integration fixture.

Assurance tier: model-level.

Owns: the public `CallRecipe` that packages the exact eleven rows, four
temporary columns, support, positional ownership, active soundness, active
honest completeness, and inactive satisfiability proved by `StepRecipeCore`.

Does not own: a general WASM compiler, a production application selection,
NIFS, a recursive fixed point, Rust, or artifacts.

Emits constraints: exactly eleven rows and four auxiliary columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.Goldilocks

private theorem footprint_exact_public
    (parameters : Parameters)
    (profile : StepProfile parameters) :
    (signature parameters).callFootprint Call.step = stepFootprint := by
  simpa [signature, callFootprint] using profile.stepFootprintExact

/-- Certified physical recipe for the benchmark application step.

This recipe certifies the Lean-owned benchmark model. It is not a compiler
for arbitrary WASM programs. -/
noncomputable def stepRecipe
    (parameters : Parameters)
    (profile : StepProfile parameters) :
    CallRecipe (signature parameters) profile.family Call.step := by
  refine
    { rows := ?_
      rowCount := ?_
      rowsOwned := ?_
      rowIdsNodup := ?_
      rowsSupported := ?_
      activeSoundness := ?_
      activeHonestCompleteness := ?_
      inactiveSatisfiable := ?_ }
  · intro context references frame
    cases references with
    | cons stateReference tail =>
        cases tail with
        | cons witnessReference tail =>
            cases tail
            exact rows parameters profile frame
  · intro context references frame
    cases references with
    | cons stateReference tail =>
        cases tail with
        | cons witnessReference tail =>
            cases tail
            rw [footprint_exact_public parameters profile]
            exact rows_length parameters profile frame
  · intro context references frame row member
    cases references with
    | cons stateReference tail =>
        cases tail with
        | cons witnessReference tail =>
            cases tail
            exact rows_owner parameters profile frame row member
  · intro context references frame
    cases references with
    | cons stateReference tail =>
        cases tail with
        | cons witnessReference tail =>
            cases tail
            exact rowIds_nodup parameters profile frame
  · intro context references frame row member column columnMember
    cases references with
    | cons stateReference tail =>
        cases tail with
        | cons witnessReference tail =>
            cases tail
            exact rows_supported parameters profile frame
              row member column columnMember
  · intro context references frame assignment inputs
      constantOne activeOne decoded holds
    cases references with
    | cons stateReference tail =>
        cases tail with
        | cons witnessReference tail =>
            cases tail
            exact active_soundness parameters profile frame assignment
              inputs constantOne activeOne decoded holds
  · intro context references frame assignment inputs outputs
      constantOne activeOne inputsEncoded outputsEncoded evaluated
    cases references with
    | cons stateReference tail =>
        cases tail with
        | cons witnessReference tail =>
            cases tail
            exact active_honest_completeness parameters profile frame
              assignment inputs outputs constantOne activeOne
              inputsEncoded outputsEncoded evaluated
  · intro context references frame assignment constantOne activeZero
    cases references with
    | cons stateReference tail =>
        cases tail with
        | cons witnessReference tail =>
            cases tail
            exact inactive_satisfiable parameters profile frame assignment
              constantOne activeZero

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
