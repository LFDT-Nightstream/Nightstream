import tests.FPrimeFullHistoryXOutSpongeReceipts
import tests.Axioms.Support

/-!
Fail-closed guards for the source-computed and artifact-checked plain-state
XOut Poseidon2 sponge receipts.
-/

namespace NightstreamTests.Axioms.FPrimeFullHistoryXOutSpongeReceipts

open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2Sponge.EmissionReceipt.traceRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms EmissionReceipt.traceRows_length

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2Sponge.EmissionReceipt.rowIndices_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms EmissionReceipt.rowIndices_nodup

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2Sponge.EmissionReceipt.allocatedColumns_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms EmissionReceipt.allocatedColumns_nodup

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2Sponge.EmissionReceipt.row_column_conservation' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms EmissionReceipt.row_column_conservation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.sourceProgram_eq_generated' does not depend on any axioms -/
#guard_msgs in
#audit_axioms sourceProgram_eq_generated

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.inputFields_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms inputFields_eq

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.physicalCost_eq' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms physicalCost_eq

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.baseSchedule_exact' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms baseSchedule_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.priorSchedule_exact' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms priorSchedule_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.recursiveOutputSchedule_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms recursiveOutputSchedule_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.baseReceipt' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms baseReceipt

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.priorReceipt' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms priorReceipt

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.recursiveOutputReceipt' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms recursiveOutputReceipt

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.baseRows_exact_cost' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms baseRows_exact_cost

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.priorRows_exact_cost' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms priorRows_exact_cost

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.recursiveOutputRows_exact_cost' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms recursiveOutputRows_exact_cost

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.base_conservation' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms base_conservation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.prior_conservation' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms prior_conservation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.recursiveOutput_conservation' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms recursiveOutput_conservation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts.pureExecutions_equal' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms pureExecutions_equal

end NightstreamTests.Axioms.FPrimeFullHistoryXOutSpongeReceipts
