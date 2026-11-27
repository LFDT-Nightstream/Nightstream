import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Digest

/-! Axiom audits for `Replay/Sound20Digest.lean`: the stage data of the replay of `key20.aks` do
not read the hash `Public.digest`. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.take_free
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ops_scripted20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ops_free20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stageIndex20_le_last
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.free20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.withDigest_layout
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.decode_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stageIndex20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.bits20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.run20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ran20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stateAt20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eventsAt20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tDigestAt_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eDigestAt_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceAt20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.noteOf_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.recvs20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pageAt20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stagePage20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.addressAt20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.recordAddress20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.resolve20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.resolveWith20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.back20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.resolveAt_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.backAt_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tower20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tower20_sealed_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Readings20.of_events
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.readings20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.terminalState20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.digests20_withDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.last20_withDigest
