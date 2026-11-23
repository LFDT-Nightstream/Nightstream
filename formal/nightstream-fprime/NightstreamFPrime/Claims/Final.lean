import NightstreamFPrime.Export.BoundedDepth.FinalHistory
import NightstreamFPrime.Export.BoundedDepth.FinalRows
import NightstreamFPrime.Export.BoundedDepth.FinalVerifier
import NightstreamFPrime.Export.BoundedDepth.NebulaFinal

/-! # Claims: the final verifier

The results for the full path of a compressed proof (`SPEC.md` Algorithm 10 and section 9,
`SECURITY_MODEL.md` adversary 6).

- `history_sound_with_akita` is the theorem of `nightstream::Verifier::verify_final`, and
  `memory_sound_with_akita` is the theorem of the memory `verify_final`.
- Assumptions: `[HN-A1]` for the history, and `[AKITA]` for each claim that ends in
  `_with_akita` (`SECURITY_MODEL.md`, trust item 7). The key commitment needs binding (`binding`).
- Open: `akita` (`Spec.Compression.Akita.Use.Knowledge`) includes extraction at the fixed setup
  (`akita.fixedSetup`) and schedule admissibility (`akita.admissible`). The schedule file checks
  the witness lengths of Akita C3. The other part is in `Schedule.Rest`.
- Open: the premise `composition` of each claim that ends in `_with_akita`
  (`Spec.Compression.Akita.Use.Composition`). No result of Akita gives it, because the prover
  commits `V` after the lookup challenges (`SECURITY_MODEL.md`, trust item 7). Lean proves the rest
  of the step from the game with ideal openings to Akita openings
  (`Spec.Compression.Akita.Use.wins_le`).
- Mode: each claim other than `light_gate_complete` holds for each mode of the verifier's circuit
  (`Export.BoundedDepth.FinalVerifier.Mode`): the argument has both slot claims, or only the claim
  of slot 0 (the light mode of `SPEC.md` Algorithm 9).
- The game runs the gate of `SPEC.md` Algorithm 10, step 5
  (`Export.BoundedDepth.FinalVerifier.Setup.Gate`), with the layout of the Lean model of Rust
  `Layout::new`. In the light mode, the honest proof passes it (`light_gate_complete`).
- Link: the premise `rows` says that the dense rows of the setup are Lean's model of Rust
  `FactoredRows` (`Export.BoundedDepth.FinalRows.packageRows`). Then
  `Export.BoundedDepth.FinalVerifier.Premises` holds
  (`Export.BoundedDepth.FinalRows.premises_of_rows`). A Rust test compares a checksum of these
  rows with Lean's (`Export.BoundedDepth.RowsParity.fastSum_eq`).
- Knowledge: `history_knowledge` bounds the event that the witnesses of the reverse extractor are
  not a valid history of the accepted statement, with the bound of `history_sound`, and
  `history_knowledge_with_akita` is its Akita form.
- Bound: `history_sound`, `history_knowledge`, `memory_sound` and their Akita forms contain
  `Export.BoundedDepth.IvcSecurity.walkBound` for the prover's IVC adversary, the sum over the
  reverse extractor's stages of `Claims.Ivc`.
- Collision term: in these claims it is the success chance of the Poseidon2 collision finder
  of each reverse stage (`Export.BoundedDepth.IvcSecurity.finderChance`), as in `Claims.Ivc`. Lean
  does not bound it.
- Open: each open premise of `Claims.Ivc` and `Claims.Nebula`. The error is not one security
  level. -/

namespace NightstreamFPrime.Claims.Final

open Spec Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth
open scoped Spec.Compression.Production

/-- Steps 1 to 5 of the final verifier in the light mode: under the key of `2^19` blocks, which
Algorithm 11 gives for the light relation, and a security minimum of at most 118 bits, a recursive
proof that the verifier of Algorithm 6 accepts, with the light circuit last and slot 1 at the
default slot of depth 0, passes the checks of steps 1 to 4 and the gate of step 5 with each fresh
commitment. `Claims.Ivc.light_run_complete` gives such a proof for each valid run of 1 to
`cap + 1` steps. -/
theorem light_gate_complete
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (light : FinalVerifier.Setup FinalVerifier.Mode.light logicalWidth publicFits)
    {packageWidths : Fin 2 → ℕ}
    (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2)
    (key : Compression.Rust.keyBlocksFor (light.blocks 0) = some light.keyBlocks)
    (shipped : light.keyBlocks = 2 ^ 19)
    (minimum_le : light.minimum ≤ 118)
    (statement : HyperNova.Construction2.Paper.TerminalStatement AppState)
    (payload : HyperNova.Construction2.Paper.TerminalProof (SlotData logicalWidth publicFits)
        (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
        (StepFresh logicalWidth publicFits)
        (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (verifies : RunCompleteness.Verifies light.ajtai light.circuitKeys step packageWidths control
        statement (HyperNova.Construction2.Paper.OuterTerminalProof.recursive payload))
    (pcOne : payload.pc = 1)
    (slotOne : payload.running 1 = (defaultSlot, 0))
    (commitment : PaperAlgebra.Commitment) :
    FinalVerifier.Checks FinalVerifier.Mode.light statement
        ⟨payload.pc, payload.running, commitment⟩ ∧
      light.Gate ⟨payload.pc, payload.running, commitment⟩ :=
  @Export.BoundedDepth.FinalVerifier.accepts_light logicalWidth publicFits packageWidths step
    control light key shipped minimum_le statement payload verifies pcOne slotOne commitment

/-- In the state-restoration experiment with `q` queries, the final verifier accepts a statement
for which no proof with witnesses satisfies `Verifies` with probability at most
`q · (1/2^119 + 38/fieldModulus²)`. -/
theorem verifier_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (s : Compression.Game.Strategy (Option (Option (FinalVerifier.Outcome logicalWidth publicFits))))
    (q : ℕ) :
    Compression.Game.prob s (setup.game.restore q [])
          (FinalVerifier.Wins setup complete.packageWidth step control) ≤
        (↑q : ℝ) * FinalVerifier.finalError :=
  Export.BoundedDepth.FinalRows.soundness setup step control exports complete rows s q

/-- With Akita openings, the final verifier accepts a statement for which no proof with
witnesses satisfies `Verifies` with probability at most `q · (1/2^119 + 38/fieldModulus²)` plus
`Spec.Compression.Akita.Use.error`. This adds the binding error of the key to
`Schedule.knowledgeError`, the bound assumed by `akita.fixedSetup` for the prover `real` at the
fixed parameters. The bound uses Akita (210) and the digest term of Akita (212).
`event` is this event for `real`, and `composition` relates it to the event of the game prover `s`. -/
theorem verifier_sound_with_akita
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (opening : Compression.Akita.Use)
    (real : Compression.Akita.Prover opening.scheme opening.claims)
    (event : real.Tape → Prop)
    (s : Compression.Game.Strategy (Option (Option (FinalVerifier.Outcome logicalWidth publicFits))))
    (q : ℕ)
    (akita : opening.Knowledge real)
    (binding : opening.Binding)
    (accepts : ∀ (tape : real.Tape), event tape → opening.Opens real tape)
    (composition : opening.Composition real event
        (Compression.Game.prob s (setup.game.restore q [])
          (FinalVerifier.Wins setup complete.packageWidth step control))) :
    ENNReal.toReal (real.tape.toOuterMeasure {tape | event tape}) ≤
      (↑q : ℝ) * FinalVerifier.finalError + opening.error real :=
  Export.BoundedDepth.FinalRows.real_soundness setup step control exports complete rows opening
    real event s q akita binding accepts composition

/-- In the state-restoration experiment with `q` queries, the probability of a false history is at
most `q · (1/2^119 + 38/fieldModulus²)` plus `IvcSecurity.walkBound` of the prover's IVC adversary:
for each of the first `depth` stages of the reverse extractor, the success chance of the stage's
Poseidon2 collision finder (`IvcSecurity.finderChance`) and the stage's error. -/
theorem history_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
          (IvcWalk.Answer logicalWidth publicFits) →
        Prop}
    {Efficient : (adversary :
          HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
        HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary →
          Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
        (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
        (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (s : Compression.Game.Strategy (Option (Option (FinalVerifier.Outcome logicalWidth publicFits))))
    (q : ℕ)
    (admitted : StageAdmitted
        (HyperNova.ReverseStages.Stage.start
          (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q))))
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (depth : ℕ)
    (depthBound : ∀ (o : FinalVerifier.Outcome logicalWidth publicFits),
        some (some o) ∈ (Compression.Game.law s (setup.game.restore q [])).support →
          o.1.iteration ≤ depth) :
    Compression.Game.prob s (setup.game.restore q [])
          (FinalHistory.FalseHistory exports.program) ≤
        (↑q : ℝ) * FinalVerifier.finalError +
          IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
            (FinalHistory.adversary setup s q) admitted depth :=
  @Export.BoundedDepth.FinalHistory.soundness logicalWidth publicFits mode setup relation step cubeFits
    exports Admitted StageAdmitted Efficient error assumption closed s q admitted complete
    rows keysShaped depth depthBound

/-- The knowledge form of `history_sound` (HyperNova Definition 11 (ii)): in the state-restoration
experiment with `q` queries, the probability that the final verifier accepts a statement and the
witnesses that the reverse extractor returns from the first `depth` stages are not a valid history
of that statement is at most `q · (1/2^119 + 38/fieldModulus²)` plus `IvcSecurity.walkBound`. -/
theorem history_knowledge
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
          (IvcWalk.Answer logicalWidth publicFits) →
        Prop}
    {Efficient : (adversary :
          HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
        HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary →
          Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
        (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
        (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (s : Compression.Game.Strategy (Option (Option (FinalVerifier.Outcome logicalWidth publicFits))))
    (q : ℕ)
    (admitted : StageAdmitted
        (HyperNova.ReverseStages.Stage.start
          (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q))))
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (depth : ℕ)
    (depthBound : ∀ (o : FinalVerifier.Outcome logicalWidth publicFits),
        some (some o) ∈ (Compression.Game.law s (setup.game.restore q [])).support →
          o.1.iteration ≤ depth) :
    ENNReal.toReal
        ((HyperNova.ReverseStages.reverseStages assumption closed
            (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q)) admitted depth).1.tape.toOuterMeasure
          {tape | FinalHistory.AdviceFailure setup exports assumption closed s q admitted depth tape}) ≤
        (↑q : ℝ) * FinalVerifier.finalError +
          IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
            (FinalHistory.adversary setup s q) admitted depth :=
  @Export.BoundedDepth.FinalHistory.knowledge logicalWidth publicFits mode setup relation step cubeFits
    exports Admitted StageAdmitted Efficient error assumption closed s q admitted complete
    rows keysShaped depth depthBound

/-- With Akita openings, the probability of a false history is at most
`q · (1/2^119 + 38/fieldModulus²)` plus `Spec.Compression.Akita.Use.error` and
`IvcSecurity.walkBound`, as in `history_sound` and `verifier_sound_with_akita`. `event` is a false
history of the prover `real` with Akita openings, and `composition` relates it to the false
history of the game prover `s`. -/
theorem history_sound_with_akita
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
          (IvcWalk.Answer logicalWidth publicFits) →
        Prop}
    {Efficient : (adversary :
          HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
        HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary →
          Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
        (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
        (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (s : Compression.Game.Strategy (Option (Option (FinalVerifier.Outcome logicalWidth publicFits))))
    (q : ℕ)
    (admitted : StageAdmitted
        (HyperNova.ReverseStages.Stage.start
          (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q))))
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (depth : ℕ)
    (depthBound : ∀ (o : FinalVerifier.Outcome logicalWidth publicFits),
        some (some o) ∈ (Compression.Game.law s (setup.game.restore q [])).support →
          o.1.iteration ≤ depth)
    (opening : Compression.Akita.Use)
    (real : Compression.Akita.Prover opening.scheme opening.claims)
    (event : real.Tape → Prop)
    (akita : opening.Knowledge real)
    (binding : opening.Binding)
    (accepts : ∀ (tape : real.Tape), event tape → opening.Opens real tape)
    (composition : opening.Composition real event
        (Compression.Game.prob s (setup.game.restore q [])
          (FinalHistory.FalseHistory exports.program))) :
    ENNReal.toReal (real.tape.toOuterMeasure {tape | event tape}) ≤
      (↑q : ℝ) * FinalVerifier.finalError + opening.error real +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
          (FinalHistory.adversary setup s q) admitted depth :=
  @Export.BoundedDepth.FinalHistory.real_soundness logicalWidth publicFits mode setup relation step
    cubeFits exports Admitted StageAdmitted Efficient error assumption closed s q admitted
    complete rows keysShaped depth depthBound opening real event akita binding accepts composition

/-- The Akita form of `history_knowledge`: with Akita openings, the probability of `event` is
at most `q · (1/2^119 + 38/fieldModulus²)` plus `Spec.Compression.Akita.Use.error` and
`IvcSecurity.walkBound`. `event` is the counterpart of `FinalHistory.AdviceFailure` for the prover
`real` with Akita openings: the final verifier accepts, and the witnesses that an extractor returns
are not a valid history of the accepted statement. `composition` relates it to
`FinalHistory.AdviceFailure` of the game prover `s`. -/
theorem history_knowledge_with_akita
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
          (IvcWalk.Answer logicalWidth publicFits) →
        Prop}
    {Efficient : (adversary :
          HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
        HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary →
          Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
        (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
        (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (s : Compression.Game.Strategy (Option (Option (FinalVerifier.Outcome logicalWidth publicFits))))
    (q : ℕ)
    (admitted : StageAdmitted
        (HyperNova.ReverseStages.Stage.start
          (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q))))
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (depth : ℕ)
    (depthBound : ∀ (o : FinalVerifier.Outcome logicalWidth publicFits),
        some (some o) ∈ (Compression.Game.law s (setup.game.restore q [])).support →
          o.1.iteration ≤ depth)
    (opening : Compression.Akita.Use)
    (real : Compression.Akita.Prover opening.scheme opening.claims)
    (event : real.Tape → Prop)
    (akita : opening.Knowledge real)
    (binding : opening.Binding)
    (accepts : ∀ (tape : real.Tape), event tape → opening.Opens real tape)
    (composition : opening.Composition real event
        (ENNReal.toReal
          ((HyperNova.ReverseStages.reverseStages assumption closed
              (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q)) admitted
              depth).1.tape.toOuterMeasure
            {tape | FinalHistory.AdviceFailure setup exports assumption closed s q admitted depth
              tape}))) :
    ENNReal.toReal (real.tape.toOuterMeasure {tape | event tape}) ≤
      (↑q : ℝ) * FinalVerifier.finalError + opening.error real +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
          (FinalHistory.adversary setup s q) admitted depth :=
  @Export.BoundedDepth.FinalHistory.real_knowledge logicalWidth publicFits mode setup relation step
    cubeFits exports Admitted StageAdmitted Efficient error assumption closed s q admitted
    complete rows keysShaped depth depthBound opening real event akita binding accepts composition

/-- The bound for Rust `nebula::Verifier::verify_final`. In the state-restoration experiment with
`q` queries and the walk's `S_max · N` stages (`p.sMax * p.n`), the probability that the final
verifier accepts, the checks of SPEC.md Algorithm 8 hold, and the decoded history does not attest
the memory statement is at most `q · (1/2^119 + 38/fieldModulus²)` plus the terms of
`Claims.Nebula.memory_sound` for the prover's IVC adversary: `IvcSecurity.walkBound`, the
state-link and chain collision masses, and the memory-round error. The checks of Algorithm 8
bound the iteration by `S_max · N` (`Lifecycle.Nebula.EndToEnd.VerifierChecks.iteration_le`), so
no premise bounds it. -/
theorem memory_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    {p : Nebula.Plan}
    (app : Nebula.PortApp p.toGeometry)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
          (IvcWalk.Answer logicalWidth publicFits) →
        Prop}
    {Efficient : (adversary :
          HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
        HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary →
          Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
        (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
        (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (s : Compression.Game.Strategy (Option (Option (FinalVerifier.Outcome logicalWidth publicFits))))
    (q : ℕ)
    (claim : Option (Option (FinalVerifier.Outcome logicalWidth publicFits)) →
        PMF (Nebula.EndToEnd.MemoryStatement app.width))
    (admitted : StageAdmitted
        (HyperNova.ReverseStages.Stage.start
          (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim))))
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (valid : p.Valid)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (programEq : exports.program = Nebula.MemoryApp.program p.toGeometry app)
    {memoryError : NebulaSecurity.ClaimedStage (logicalWidth := logicalWidth) (publicFits := publicFits) app →
        ℝ}
    (round : NebulaSecurity.MemoryRound app setup.ajtai setup.circuitKeys exports StageAdmitted
        memoryError) :
    ENNReal.toReal
          (((↑(HyperNova.ReverseStages.reverseStages (Succeeds :=
                        IvcWalk.WalkSucceeds setup.ajtai) (Extracted :=
                        IvcWalk.WalkExtracted setup.ajtai) (Admitted := Admitted)
                        (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                        closed
                        (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim))
                        admitted (p.sMax * p.n)) :
                    HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
              Set
                  (↑(HyperNova.ReverseStages.reverseStages assumption closed
                          (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim))
                          admitted (p.sMax * p.n)) :
                      HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                        (IvcWalk.Answer logicalWidth publicFits)).Tape →
                ENNReal)
            {tape :
              (↑(HyperNova.ReverseStages.reverseStages (Succeeds :=
                      IvcWalk.WalkSucceeds setup.ajtai) (Extracted :=
                      IvcWalk.WalkExtracted setup.ajtai) (Admitted := Admitted) (StageAdmitted :=
                      StageAdmitted) (Efficient := Efficient) assumption closed
                      (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim))
                      admitted (p.sMax * p.n)) :
                  HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)).Tape |
              NebulaFinal.FalseMemory setup app exports
                (NebulaFinal.tapeOf setup app exports (Admitted := Admitted) (StageAdmitted :=
                    StageAdmitted) (Efficient := Efficient) assumption closed s q claim admitted
                    (p.sMax * p.n) tape).1
                (NebulaFinal.sampleOf setup app exports (Admitted := Admitted) (StageAdmitted :=
                  StageAdmitted) (Efficient := Efficient) assumption closed s q claim admitted
                  (p.sMax * p.n) tape)}) ≤
        (↑q : ℝ) * FinalVerifier.finalError +
          (IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
                  (NebulaFinal.adversary setup app s q claim) admitted (p.sMax * p.n) +
                ENNReal.toReal
                  (((NebulaSecurity.ClaimedStage.law app
                          (NebulaFinal.claimed setup app exports (Admitted := Admitted)
                            (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                            closed s q claim admitted (p.sMax * p.n))).toOuterMeasure :
                      Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                    {sample : NebulaSecurity.Sample app.width logicalWidth publicFits |
                      Nebula.EndToEnd.Experiment.StateLinkCollision app
                        (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys exports)
                        sample}) +
              ENNReal.toReal
                (((NebulaSecurity.ClaimedStage.law app
                        (NebulaFinal.claimed setup app exports (Admitted := Admitted)
                          (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                          closed s q claim admitted (p.sMax * p.n))).toOuterMeasure :
                    Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                  {sample : NebulaSecurity.Sample app.width logicalWidth publicFits |
                    Nebula.EndToEnd.Experiment.ChainCollision app
                      (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys exports)
                      sample}) +
            memoryError
              (NebulaFinal.claimed setup app exports (Admitted := Admitted) (StageAdmitted :=
                StageAdmitted) (Efficient := Efficient) assumption closed s q claim admitted
                (p.sMax * p.n))) :=
  @Export.BoundedDepth.NebulaFinal.soundness logicalWidth publicFits mode setup p app relation step
    cubeFits exports Admitted StageAdmitted Efficient error assumption closed s q claim admitted
    complete rows valid keysShaped programEq memoryError round

/-- With Akita openings, the probability of a false memory statement is at most
`q · (1/2^119 + 38/fieldModulus²)` plus `Spec.Compression.Akita.Use.error` and the terms of
`Claims.Nebula.memory_sound`, over the walk's `S_max · N` stages, as in `memory_sound`. `event` is
a false memory statement of the prover `real` with Akita openings, and `composition` relates it to
`NebulaFinal.FalseMemory` of the game prover `s`. -/
theorem memory_sound_with_akita
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    {p : Nebula.Plan}
    (app : Nebula.PortApp p.toGeometry)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
          (IvcWalk.Answer logicalWidth publicFits) →
        Prop}
    {Efficient : (adversary :
          HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
        HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary →
          Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
        (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
        (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (s : Compression.Game.Strategy (Option (Option (FinalVerifier.Outcome logicalWidth publicFits))))
    (q : ℕ)
    (claim : Option (Option (FinalVerifier.Outcome logicalWidth publicFits)) →
        PMF (Nebula.EndToEnd.MemoryStatement app.width))
    (admitted : StageAdmitted
        (HyperNova.ReverseStages.Stage.start
          (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim))))
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (valid : p.Valid)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (programEq : exports.program = Nebula.MemoryApp.program p.toGeometry app)
    {memoryError : NebulaSecurity.ClaimedStage (logicalWidth := logicalWidth) (publicFits := publicFits) app →
        ℝ}
    (round : NebulaSecurity.MemoryRound app setup.ajtai setup.circuitKeys exports StageAdmitted
        memoryError)
    (opening : Compression.Akita.Use)
    (real : Compression.Akita.Prover opening.scheme opening.claims)
    (event : real.Tape → Prop)
    (akita : opening.Knowledge real)
    (binding : opening.Binding)
    (accepts : ∀ (tape : real.Tape), event tape → opening.Opens real tape)
    (composition : opening.Composition real event
        (ENNReal.toReal
          ((HyperNova.ReverseStages.reverseStages assumption closed
              (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim)) admitted
              (p.sMax * p.n)).1.tape.toOuterMeasure
            {tape | NebulaFinal.FalseMemory setup app exports
              (NebulaFinal.tapeOf setup app exports assumption closed s q claim admitted
                (p.sMax * p.n) tape).1
              (NebulaFinal.sampleOf setup app exports assumption closed s q claim admitted
                (p.sMax * p.n) tape)}))) :
    ENNReal.toReal (real.tape.toOuterMeasure {tape | event tape}) ≤
        (↑q : ℝ) * FinalVerifier.finalError + opening.error real +
          (IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
                  (NebulaFinal.adversary setup app s q claim) admitted (p.sMax * p.n) +
                ENNReal.toReal
                  (((NebulaSecurity.ClaimedStage.law app
                          (NebulaFinal.claimed setup app exports (Admitted := Admitted)
                            (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                            closed s q claim admitted (p.sMax * p.n))).toOuterMeasure :
                      Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                    {sample : NebulaSecurity.Sample app.width logicalWidth publicFits |
                      Nebula.EndToEnd.Experiment.StateLinkCollision app
                        (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys exports)
                        sample}) +
              ENNReal.toReal
                (((NebulaSecurity.ClaimedStage.law app
                        (NebulaFinal.claimed setup app exports (Admitted := Admitted)
                          (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                          closed s q claim admitted (p.sMax * p.n))).toOuterMeasure :
                    Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                  {sample : NebulaSecurity.Sample app.width logicalWidth publicFits |
                    Nebula.EndToEnd.Experiment.ChainCollision app
                      (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys exports)
                      sample}) +
            memoryError
              (NebulaFinal.claimed setup app exports (Admitted := Admitted) (StageAdmitted :=
                StageAdmitted) (Efficient := Efficient) assumption closed s q claim admitted
                (p.sMax * p.n))) :=
  @Export.BoundedDepth.NebulaFinal.real_soundness logicalWidth publicFits mode setup p app relation
    step cubeFits exports Admitted StageAdmitted Efficient error assumption closed s q claim
    admitted complete rows valid keysShaped programEq memoryError round opening real event akita
    binding accepts composition

end NightstreamFPrime.Claims.Final
