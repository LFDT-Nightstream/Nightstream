import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Wire

/-! Axiom audits for `Replay/Sound20Wire.lean`: the transcript of a prover and the wire messages
of the recording prover of `key20.aks`. -/

#audit_axioms NightstreamFPrime.Spec.RandomOracle.OracleComp.run_map
#audit_axioms NightstreamFPrime.Spec.RandomOracle.OracleComp.queries_map
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Digest.Run.answered_of_trace
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.wireAddress_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.path_parts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.final_length
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.final_wire
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.final_reply
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tower20_protocol
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.final_length20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.final_wire20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.final_reply20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.logTable_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.messagesOf_decoded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.recordOf_listed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.closedProver20_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_quietLast
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stageIndex20_last
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quietLast20_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stageIndex20_succ_le_last
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stateBefore_suffixLast20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.lastEvents20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.squeeze_last20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.agreeOn_last20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stagePage20_tables
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sealed_withOracles
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.reideal_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stagePage20_found
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.wires20_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.wires20_guard
