import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Closed

/-! Axiom audits for the recording prover of Akita Lemma 10.47 on a closed run
(`Replay/Sound20Closed.lean`), with the guards `closed_guard` and
`simulateDeps_closed_entry_guard`. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.Named
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.named_empty
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.ready_none
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.recorded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.recorded_inl
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.recorded_inr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.split_append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.closed_before
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.settleAll_cached
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.settleAll_last
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.settle_named
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.settleAll_named
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.simulateDeps_asking
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.simulateDeps_closed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.oracleDeps_closed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.simulateDeps_logged_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.simulateDeps_closed_logged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.simulateDeps_reads_asked
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.simulateDeps_asked
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.simulateDeps_entry
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.simulateDeps_closed_entry
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.oracleDeps_asked
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.simulateDeps_closed_asked
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.compDeps_closed_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.compDeps_closed_run_empty
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.closed_guard
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.simulateDeps_closed_entry_guard
