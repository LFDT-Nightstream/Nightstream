import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeRustExport
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointStability
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsCompiler
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsPhi81
import tests.Axioms.Support

/-!
Fail-closed dependency guards for the native four-matrix CCS selector,
selected NIFS replacement, finite compiler, Phi81 bridge, and benchmark
fixed-point export.
-/

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector.active_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector.active_sound

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector.complete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStep.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStep.sound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStep.committedAllocations_preserved' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStep.committedAllocations_preserved

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStep.publicAllocations_preserved' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStep.publicAllocations_preserved

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStepCompleteness.complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStepCompleteness.complete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsCompiler.valid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsCompiler.valid

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NativeCcsCompiler.indexedAssignment_accepts_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.Goldilocks.NativeCcsCompiler.indexedAssignment_accepts_iff

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NativeCcsPhi81.accepts_assignment_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.Goldilocks.NativeCcsPhi81.accepts_assignment_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointCost.nativeCost_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointCost.nativeCost_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeRustExport.stepCost_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeRustExport.stepCost_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeRustExport.matrixCount_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeRustExport.matrixCount_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeRustExport.polynomialDegree_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeRustExport.polynomialDegree_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointStability.finalRows_eq_source' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointStability.finalRows_eq_source

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointStability.finalColumnIds_eq_source' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointStability.finalColumnIds_eq_source

/-- info: 'Nightstream.Implementation.R1CS.SeededAjtai.Setup.execution_eq_some_outputs' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.R1CS.SeededAjtai.Setup.execution_eq_some_outputs

/-- info: 'Nightstream.Implementation.R1CS.SeededAjtai.Setup.verifierKey_val' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.R1CS.SeededAjtai.Setup.verifierKey_val
