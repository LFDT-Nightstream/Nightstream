import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3
import NightstreamFPrime.Spec.Compression.Akita.Replay.Prefix
import NightstreamFPrime.Spec.Compression.Akita.Replay.FusedJoin

/-! Axiom audits for Stage 3 of the replay of Akita openings: the weight of Stage 3 of the code
(`Replay/Stage3Weight.lean`) and the run of Stage 3 against `setupCheck`
(`Replay/Stage3.lean`), and the setup in pages of the code against the setup expansion of the
specification (`Replay/Prefix.lean`). -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.powerScaleExt_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqTable_size
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqTable_getD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqIndex_mod
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqIndex_append_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.EqSplit.eval_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage3Run_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.paged_apply
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.forIn_kept
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.kept_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.page_toList
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.kept_words
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.finEquiv_mk
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.page_expand
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.card_page
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_paged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.short_fraction
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.paged_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage3In
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_stage3_runs
