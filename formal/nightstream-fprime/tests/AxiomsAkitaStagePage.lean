import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinStagePage
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageDigest

/-! Axiom audits for theorem (R), `Agrees20` and the bound of `key20.aks` with the wire messages
`stagePage20` (`Replay/JoinStagePage.lean`), and for the link of the digests of the last blocks
before a stage to the digests of the terminal (`Replay/StageDigest.lean`). -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Prefix.views_of_replies
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tier.Agrees.congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.cells20_stage
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stagePage20_cells
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stagePage20_nonce
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Readings20.mapWire
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presentsAt20_of_cells
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.agreesAt20_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presents20_of_agrees_staged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.agrees20_of_reads_staged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.exists_agrees20_staged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.exists_readings20_staged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_bound20_of_staged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_tDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_eDigest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stateBefore_bytes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held_bytes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.bytesIn_terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tDigestAt_terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eDigestAt_terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.digestAt_of_last
