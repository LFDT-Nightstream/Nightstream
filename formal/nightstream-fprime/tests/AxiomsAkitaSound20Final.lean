import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20

/-! Axiom audits for `Replay/Sound20Reads.lean` and `Replay/Sound20.lean`: the soundness bound
of the claims for `key20.aks` on the byte verifier under the ideal sponge and the ideal
digest. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.mem_trace_closed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.mem_answered_inr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.mem_digests
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.expect_fintype
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.indicator_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.expect_union_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.expect_cover_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.decoded_of_accepted
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.scripted_of_accepted
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.game20_queried
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.wires20_of_fresh
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.entry20_of_fresh
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.accepted20_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.seen20_entry
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.seen20_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.closedGame20_law
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.reads20_of_good
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.scripted20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.bad20_closed_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.closedProver20_spread
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_bound20_guarded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.guarded20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_sound20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_sound20_real
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sealed_setupFor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sound20_guard
