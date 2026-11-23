import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingSpec
import NightstreamFPrime.Spec.Compression.Akita.Replay.FusedJoin

/-! Axiom audits for the weight at `r_2` of the packing levels of the replay of Akita openings
(`Replay/WeightPacking.lean`, `Replay/WeightPackingSpec.lean`), its end check in the replay
(`Replay/Transcript.lean`, `Replay/FusedJoin.lean`) and the setup prefix in tasks
(`Replay/TerminalSetup.lean`). -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.dotFrom_toE
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.dotFq_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.powersOf_getD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.lowLevel_size
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqAt_take_succ
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.lowLevel_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.lowSums_getD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tables.of_high
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tables.run_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.PackShape.setupCount
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.FusedEnd.passing
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fold.setupCount
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.claimPointIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.packGroupsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.packInputsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.packingValuesIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.setupClaimIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.packingIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.fusedEndIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_fused
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_packing_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Setup.expandTasks
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Setup.expandTasks_eq
