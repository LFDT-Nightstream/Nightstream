import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Positions

/-! Axiom audits for `Replay/Sound20Positions.lean`: the digest positions of a sponge key of
`key20.aks` and the record facts that follow from "no bad event". -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.mem_recorded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.recorded_of_unique
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate.recorded_of_fresh
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.bytes32_array
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.length_bytes32
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.length_stageDigests20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.positions20_inr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.length_positions20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.positions20_stage
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.positions20_local
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.positions20_alike
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage20_of_accepted
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.digestAt20_hash
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.positions20_hit
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.recordOf_single
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.record20_of_fresh
