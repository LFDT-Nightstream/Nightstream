import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkKey20
import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkPack
import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkTerminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkRoot

/-! Axiom audits for the links between the levels of `key20.aks` on the replay side: the frames
of the states before two operations, the tensor reduction of the receiver on the claim of the
level before, and the statement of each level of `key20.aks` as the claim of the link of the model
from the result of the level before (`Replay/LinkTrace.lean`, `Replay/LinkKey20.lean`,
`Replay/LinkPack.lean`, `Replay/LinkTerminal.lean`, `Replay/LinkRoot.lean`). -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorKeys
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pathRun_source
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.dataIn_witness
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.dataIn_payload
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.firstIn_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stateBefore_frees
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorRun_some
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.into_dot_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.segments_append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.segments_length
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceStatementFor_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_tensorLink
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_tensored
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.dataIn_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_traceLink
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.key20_segments
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.layout20_receiver
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_traceLink
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.cast_ne_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.layout20_trace
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_level1_tensored
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_trace_tensored
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_traceSq_tensored
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.claimPointIn_succ
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.point_segments
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.point_lengths
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_recorded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_absorbed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_fusedPoint
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_packLink
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_packLink
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_terminalLink
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.layout20_terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.step_absorb_point
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.step_absorb_values
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchRun_source
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.into_dot_three
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_rootLink
