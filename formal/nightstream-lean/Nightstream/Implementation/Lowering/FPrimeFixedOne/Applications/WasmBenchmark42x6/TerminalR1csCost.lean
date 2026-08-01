import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointFamily
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Program

/-!
Contract: measure the direct terminal R1CS for the selected 42-times-6
benchmark fixed point.

Assurance tier: model-level.

Owns: exact per-running-claim, fresh-claim, and aggregate terminal costs
derived from the Lean terminal program.

Does not own: a practical terminal proof backend, Spartan setup, WHIR,
Rust, or a security reduction.

Emits constraints: none. It measures the existing Lean-owned terminal
relation.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointCost
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointFamily
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointSource
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- Cost of one running claim in the direct terminal R1CS. -/
noncomputable def runningClaimCost (template : Template) : Cost :=
  TerminalR1cs.Program.runningCost
    (compiledShape template) commitmentRows

/-- Cost of the fresh claim in the direct terminal R1CS. -/
noncomputable def freshClaimCost (template : Template) : Cost :=
  TerminalR1cs.Program.freshCost
    (program template) (compiledShape template) commitmentRows

/-- Cost of all fourteen running claims and the one fresh claim. -/
noncomputable def terminalCost (template : Template) : Cost :=
  TerminalR1cs.Program.cost
    (program template) (compiledShape template) commitmentRows

theorem runningClaimCost_exact (template : Template) :
    runningClaimCost template =
      ⟨10_710_846, 5_354_586, 1_674, 5_354_586⟩ := by
  rw [runningClaimCost, compiledShape_eq]
  rfl

theorem freshClaimCost_exact (template : Template) :
    freshClaimCost template =
      ⟨21_309_394, 5_354_586, 1_242, 10_654_076⟩ := by
  rw [freshClaimCost, compiledShape_eq]
  unfold TerminalR1cs.Program.freshCost
  rw [rowsExact]
  rfl

/-- The direct terminal R1CS is exact but too large to be the deployment
backend. This theorem keeps the obstruction in the Lean authority chain. -/
theorem terminalCost_exact (template : Template) :
    terminalCost template =
      ⟨171_261_238, 80_318_790, 24_679, 85_618_280⟩ := by
  rw [terminalCost, compiledShape_eq]
  unfold TerminalR1cs.Program.cost
  rw [rowsExact]
  rfl

/-- The selected benchmark cannot enter the bounded direct Spartan backend.
The inequality is derived from the Lean-owned terminal program, not from a
Rust measurement. -/
theorem terminalRows_exceedDirectBackendCap (template : Template) :
    1_000_000 < (terminalCost template).recurringRows := by
  rw [terminalCost_exact]
  decide

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost
