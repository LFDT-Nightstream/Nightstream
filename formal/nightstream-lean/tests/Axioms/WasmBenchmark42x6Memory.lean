import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Memory
import tests.Axioms.Support

/-! Fail-closed dependency guard for the exact benchmark memory trace. -/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Memory.balanced' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Memory.balanced
