import tests.AxiomAudit
import tests.AkitaVectors
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageReadings

/-! Axiom audits for the readings of the code at each stage of the opening of `key20.aks` as
functions of the answers of an ideal sponge (`Replay/StageReadings.lean`), and the run check of
`Bits20` on the plans of the two positive vectors of `key20.aks`. The run check is a test and
not a theorem: Lean reads the plan of a vector from its file. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Bits20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.bitsOk20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.bits20_of_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.bits20_length
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.bits20_ne_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.work20_answer
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fieldDraw20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldDraw20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_fields
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_folds
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawCounts20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fieldDraw20_count
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldDraw20_lt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldDraw20_bytes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawEvents20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.poolBytes_ne_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.op_of_skeleton
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fieldDraw20_op
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldDraw20_op
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawEvents20_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_challenge_values
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_squeeze_values
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.challenge20_answer
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.squeeze20_answer
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offset20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_offsets
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_fresh
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldDraw20_fresh
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offset20_size
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.challenge20_recorded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.find_drawn_none
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.squeeze20_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.row20_first
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.weights20_answer
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.folds20_before
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldLevels20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawn20_answer

open NightstreamFPrime.Spec.Compression.Akita.Replay in
#eval show IO Unit from do
  for name in ["key20-opening.json", "key20-opening-retry.json"] do
    let (v, _) ← AkitaVectors.load name
    let ops ← IO.ofExcept (script (← AkitaVectors.layoutOf v) v.plan)
    unless bitsOk20 ops do
      throw <| IO.userError s!"{name}: a work of the script does not have the bits of bits20"
    IO.println s!"run check {name}: bitsOk20 holds on the script of the plan of the file"
