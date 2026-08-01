import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofEvidence
import tests.Axioms.Support

set_option autoImplicit false

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6

/-! Fail-closed axiom boundary for the exact 42-times-6 application proof. -/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofProgram.soundness' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ModuleProofProgram.soundness

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofProgram.honest_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ModuleProofProgram.honest_satisfies

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofProgram.finite_accepts_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ModuleProofProgram.finite_accepts_honest

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofExport.bindings_cover_program' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ModuleProofExport.bindings_cover_program

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofExport.exact_native_ccs_nonzero_coefficients' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ModuleProofExport.exact_native_ccs_nonzero_coefficients

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofExport.render_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ModuleProofExport.render_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofR1csLowering.selected_rows_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ModuleProofR1csLowering.selected_rows_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofR1csLowering.satisfies_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ModuleProofR1csLowering.satisfies_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofR1csLowering.soundness' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ModuleProofR1csLowering.soundness

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofR1csLowering.honest_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ModuleProofR1csLowering.honest_satisfies

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofEvidence.m4' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ModuleProofEvidence.m4
