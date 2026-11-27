import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Chain

/-! Axiom audits for `Replay/Sound20Chain.lean`: the addresses and the digest positions of the
keys of one chain of `key20.aks`. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.exists_widest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.trim_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.depsOf_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.depsOf_step
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.depsOf_nested
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.chain_nested
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage20_base
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage20_dep
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage20_nested
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage20_of_resolved
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.recordAddress20_trim
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.resolveAt_dep
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.resolveAt_chain
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stageDigests20_subset
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.positions20_dep
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.positions20_chain
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fresh20
