import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Count

/-! Axiom audits for the counts of the recording prover of Akita Lemma 10.47 on a closed run
(`Replay/Sound20Count.lean`), with the guards `coherent_none`, `compDeps_closed_classes_guard`
and `closed_bad_guard`. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.toComp_asking
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.toComp_closed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.collided_closed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.mass_distinct_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.guessed_distinct_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.distinct_closed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.closed_bad_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.closed_bad_guard
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.Coherent
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.coherent_none
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.settleAll_fresh
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.settleAll_class
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.simulateDeps_closed_classes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.compDeps_queries_fixed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.compDeps_closed_classes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.oracleDeps_bad_le_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.oracleDeps_closed_bad_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.compDeps_closed_classes_guard
