import tests.FPrimeTerminalLinkBatch
import tests.Axioms.Support

/-!
Fail-closed guards for arbitrary-batch terminal-link ownership and refinement.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.rows_one_eq_artifact' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.rows_one_eq_artifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.cost_conservation' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.cost_conservation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.physicalIndex_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.physicalIndex_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.physicalIndex_surjective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.physicalIndex_surjective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.publicColumn_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.publicColumn_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.publicColumn_surjective_interval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.publicColumn_surjective_interval

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.satisfies_iff_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.satisfies_iff_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.satisfies_iff_logicalPaperLinks' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch.satisfies_iff_logicalPaperLinks
