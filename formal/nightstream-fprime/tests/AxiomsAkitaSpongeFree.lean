import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.SpongeFreeBases

/-! Axiom audits for the address of a sponge key of `key20.aks` with two answer functions of an
ideal sponge (`Replay/SpongeFree.lean`, `Replay/SpongeFreeBases.lean`): the run, the events, the
pages and `resolve20` read the answers only at the squeezes of the events, and each such squeeze
is a squeeze of an earlier stage. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.AgreeOn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.AgreeOn.refl
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.AgreeOn.symm
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.AgreeOn.suffix
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.agreeOn_answerOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.reideal_self
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.withSponge_spongeOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.core_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.asks_none
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.runFrom_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.channelStart_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stateBefore_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eventsAt20_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.digestAt_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pageAt20_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stateBefore_suffix20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stagePage20_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.addressAt20_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage20_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage20_of_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.recordAddress20_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.addressAt20_of_events
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eventsAt20_of_address
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.resolve20_reideal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.resolveWith20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.resolveWith20_agree
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.resolveWith20_back
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.resolveWith20_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Absorbs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.absorbs_path
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.absorbs_absorb
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_absorbs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.runFrom_absorbs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quietFit20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_quiet
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_start
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quiet20_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.start20_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.absorbs_events_trans
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.squeeze_absorbs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.squeeze_squeezes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.channelStart_absorbs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eventsAt20_zero_parts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eventsAt20_succ_parts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.squeeze_stage20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.agreeOn_stages
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.agreeOn_tables
