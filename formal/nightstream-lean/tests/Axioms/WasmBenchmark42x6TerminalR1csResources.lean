import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csResources
import tests.Axioms.Support

/-! Fail-closed dependency guard for the benchmark terminal resource audit. -/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csResources.ajtaiCoefficientSlots_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csResources.ajtaiCoefficientSlots_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csResources.maximumAjtaiRowTermSlots_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csResources.maximumAjtaiRowTermSlots_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csResources.privateColumns_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csResources.privateColumns_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csResources.ajtaiCoefficientSlots_exceed_seventyEightBillion' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csResources.ajtaiCoefficientSlots_exceed_seventyEightBillion
