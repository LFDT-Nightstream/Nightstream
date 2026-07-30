import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4FootprintAudit
import tests.Axioms.Support

/-! Fail-closed dependency guards for the benchmark M4 footprint census. -/

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4FootprintAudit

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4FootprintAudit.transcriptPermutationCount_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms transcriptPermutationCount_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4FootprintAudit.selectedNifsCost_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms selectedNifsCost_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4FootprintAudit.transcript_dominates_piRlcAction' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms transcript_dominates_piRlcAction
