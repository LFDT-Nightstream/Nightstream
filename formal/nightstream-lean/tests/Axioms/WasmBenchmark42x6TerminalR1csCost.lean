import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost
import tests.Axioms.Support

/-! Fail-closed dependency guard for the benchmark terminal R1CS cost. -/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost.runningClaimCost_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost.runningClaimCost_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost.freshClaimCost_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost.freshClaimCost_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost.terminalCost_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost.terminalCost_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost.terminalRows_exceedDirectBackendCap' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost.terminalRows_exceedDirectBackendCap
