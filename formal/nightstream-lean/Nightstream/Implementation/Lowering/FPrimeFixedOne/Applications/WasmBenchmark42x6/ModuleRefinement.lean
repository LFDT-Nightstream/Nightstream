import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Module
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Semantics

/-!
Contract: connect the application transition to the Lean-owned 42-times-6
WASM module.

Assurance tier: model-level.

Owns: exact agreement of the two application batches with the module's
instruction execution and terminal result.

Does not own: Rust, Wasmtime, arbitrary WASM compilation, F-prime, Spartan,
WHIR, or cryptographic soundness.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm

/-- The first application transition is exactly the first instruction batch:
the loaded word is 42 and the second operand is 6. -/
theorem afterPreparation_refines_module_batch :
    executeInstructions module.initialMemory preparationInstructions [] =
      some
        [ (read afterPreparation .rightOperand).val
        , (read afterPreparation .leftOperand).val
        ] := by
  decide

/-- The second application transition is exactly the multiplication batch. -/
theorem final_refines_module_execution :
    module.run = some (read final .output).val := by
  decide

/-- Headline application binding. The value accepted by the application
transition is computed by the exact bytes emitted from the same module. -/
theorem application_result_is_module_result :
    module.run =
      some (read (step (step initial noWitness) noWitness) .output).val := by
  decide

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
