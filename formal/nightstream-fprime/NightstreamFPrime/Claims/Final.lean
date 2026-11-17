import NightstreamFPrime.Export.BoundedDepth.FinalHistory
import NightstreamFPrime.Export.BoundedDepth.FinalRows
import NightstreamFPrime.Export.BoundedDepth.FinalVerifier
import NightstreamFPrime.Export.BoundedDepth.NebulaFinal

/-! # Claims: the final verifier

The results for the full path of a compressed proof (`SPEC.md` Algorithm 10 and section 9,
`SECURITY_MODEL.md` adversary 6).

- `history_sound_with_akita` is the theorem of `nightstream::Circuit::verify_final`, and
  `memory_sound_with_akita` is the theorem of the memory `verify_final`.
- Assumptions: `[HN-A1]` for the history, and `[AKITA]` for each claim that ends in
  `_with_akita`.
- The game takes the layout of the Lean model of Rust `Layout::new`
  (`Export.BoundedDepth.FinalVerifier.Setup.Laid`).
- Link: the premise `rows` says that the dense rows of the setup are Lean's model of Rust
  `FactoredRows` (`Export.BoundedDepth.FinalRows.packageRows`). Then
  `Export.BoundedDepth.FinalVerifier.Premises` holds
  (`Export.BoundedDepth.FinalRows.premises_of_rows`). A Rust test compares a checksum of these
  rows with Lean's (`Export.BoundedDepth.RowsParity.fastSum_eq`).
- Collision term: in `history_sound`, `memory_sound` and their Akita forms it is the success
  chance of the Poseidon2 collision finder of each reverse stage
  (`Export.BoundedDepth.IvcSecurity.finderChance`), as in `Claims.Ivc`. Lean does not bound it.
- Open: each open premise of `Claims.Ivc` and `Claims.Nebula`. The error is not one security
  level. -/

namespace NightstreamFPrime.Claims.Final

open Spec Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth
open scoped Spec.Compression.Production

/-- In the state-restoration experiment with `q` queries, the final verifier accepts a statement
for which no proof with witnesses satisfies `Verifies` with probability at most
`q · (1/2^119 + 38/q²)`. -/
theorem verifier_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (setup : FinalVerifier.Setup logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
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
          (FinalVerifier.Wins setup complete.packageWidth step cap control) ≤
        (↑q : ℝ) * FinalVerifier.finalError :=
  Export.BoundedDepth.FinalRows.soundness setup step cap control exports complete rows s q

/-- With Akita openings, the final verifier accepts a statement for which no proof with
witnesses satisfies `Verifies` with probability at most `q · (1/2^119 + 38/q²) + εAkita`. -/
theorem verifier_sound_with_akita
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (setup : FinalVerifier.Setup logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    {Real : Type}
    (winsOf : Real → ℝ)
    (q : ℕ)
    (εAkita : ℝ)
    (akita : FinalVerifier.AkitaKnowledge winsOf setup complete.packageWidth step cap control q εAkita)
    (real : Real) :
    winsOf real ≤ (↑q : ℝ) * FinalVerifier.finalError + εAkita :=
  Export.BoundedDepth.FinalRows.real_soundness setup step cap control exports complete rows winsOf q
    εAkita akita real

/-- In the state-restoration experiment with `q` queries, the probability of a false history is at
most `q · (1/2^119 + 38/q²)` plus, for each of the first `depth` stages of the reverse extractor
of the prover's IVC adversary, the success chance of the stage's Poseidon2 collision finder
(`IvcSecurity.finderChance`) and the stage's error. -/
theorem history_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (setup : FinalVerifier.Setup logicalWidth publicFits)
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
          ∑ j : Fin depth,
            (IvcSecurity.finderChance setup.ajtai setup.circuitKeys exports
                (HyperNova.ReverseStages.reverseStages assumption closed
                  (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q))
                  admitted (↑j : ℕ)).1 +
              error
                (HyperNova.ReverseStages.Stage.nifs
                  (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
                  (IvcWalk.back setup.ajtai setup.circuitKeys exports)
                  (↑(HyperNova.ReverseStages.reverseStages (Succeeds :=
                        IvcWalk.WalkSucceeds setup.ajtai) (Extracted :=
                        IvcWalk.WalkExtracted setup.ajtai) (Admitted := Admitted)
                        (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                        closed (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q))
                        admitted (↑j : ℕ)) :
                    HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)))) :=
  @Export.BoundedDepth.FinalHistory.soundness logicalWidth publicFits setup relation step cubeFits
    exports Admitted StageAdmitted Efficient error assumption closed s q admitted complete
    rows keysShaped depth depthBound

/-- With Akita openings, the probability `history` of a false history is at most
`q · (1/2^119 + 38/q²) + εAkita` plus the sum of `history_sound`. `akita` is Akita's knowledge
soundness (`FinalVerifier.AkitaKnowledge`) for the event `FalseHistory`: `s` is the prover of the
game with ideal openings that it gives for the prover with Akita openings. -/
theorem history_sound_with_akita
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (setup : FinalVerifier.Setup logicalWidth publicFits)
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
    (history : ℝ)
    (εAkita : ℝ)
    (akita : history ≤
        Compression.Game.prob s (setup.game.restore q [])
            (FinalHistory.FalseHistory exports.program) +
          εAkita) :
    history ≤
        (↑q : ℝ) * FinalVerifier.finalError + εAkita +
          ∑ j : Fin depth,
            (IvcSecurity.finderChance setup.ajtai setup.circuitKeys exports
                (HyperNova.ReverseStages.reverseStages assumption closed
                  (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q))
                  admitted (↑j : ℕ)).1 +
              error
                (HyperNova.ReverseStages.Stage.nifs
                  (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
                  (IvcWalk.back setup.ajtai setup.circuitKeys exports)
                  (↑(HyperNova.ReverseStages.reverseStages (Succeeds :=
                        IvcWalk.WalkSucceeds setup.ajtai) (Extracted :=
                        IvcWalk.WalkExtracted setup.ajtai) (Admitted := Admitted)
                        (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                        closed (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q))
                        admitted (↑j : ℕ)) :
                    HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)))) :=
  @Export.BoundedDepth.FinalHistory.real_soundness logicalWidth publicFits setup relation step
    cubeFits exports Admitted StageAdmitted Efficient error assumption closed s q admitted
    complete rows keysShaped depth depthBound history εAkita akita

/-- The bound for Rust `nebula::Verifier::verify_final`. In the state-restoration experiment with
`q` queries and the walk's `depth` stages, the probability that the final verifier accepts, the
checks of SPEC.md Algorithm 8 hold, and the decoded history does not attest the memory statement
is at most `q · (1/2^119 + 38/q²)` plus the terms of `Claims.Nebula.memory_sound` for the prover's IVC
adversary. -/
theorem memory_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (setup : FinalVerifier.Setup logicalWidth publicFits)
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
    (depth : ℕ)
    (depthBound : ∀ (o : FinalVerifier.Outcome logicalWidth publicFits),
        some (some o) ∈ (Compression.Game.law s (setup.game.restore q [])).support →
          o.1.iteration ≤ depth)
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
                        admitted depth) :
                    HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
              Set
                  (↑(HyperNova.ReverseStages.reverseStages assumption closed
                          (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim))
                          admitted depth) :
                      HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                        (IvcWalk.Answer logicalWidth publicFits)).Tape →
                ENNReal)
            {tape :
              (↑(HyperNova.ReverseStages.reverseStages (Succeeds :=
                      IvcWalk.WalkSucceeds setup.ajtai) (Extracted :=
                      IvcWalk.WalkExtracted setup.ajtai) (Admitted := Admitted) (StageAdmitted :=
                      StageAdmitted) (Efficient := Efficient) assumption closed
                      (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim))
                      admitted depth) :
                  HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)).Tape |
              NebulaFinal.FalseMemory setup app exports
                (NebulaFinal.tapeOf setup app exports (Admitted := Admitted) (StageAdmitted :=
                    StageAdmitted) (Efficient := Efficient) assumption closed s q claim admitted
                    depth tape).1
                (NebulaFinal.sampleOf setup app exports (Admitted := Admitted) (StageAdmitted :=
                  StageAdmitted) (Efficient := Efficient) assumption closed s q claim admitted
                  depth tape)}) ≤
        (↑q : ℝ) * FinalVerifier.finalError +
          (∑ j : Fin depth,
                  (IvcSecurity.finderChance setup.ajtai setup.circuitKeys exports
                      (HyperNova.ReverseStages.reverseStages assumption closed
                        (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim))
                        admitted (↑j : ℕ)).1 +
                    error
                      (HyperNova.ReverseStages.Stage.nifs
                        (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
                        (IvcWalk.back setup.ajtai setup.circuitKeys exports)
                        (↑(HyperNova.ReverseStages.reverseStages (Succeeds :=
                              IvcWalk.WalkSucceeds setup.ajtai) (Extracted :=
                              IvcWalk.WalkExtracted setup.ajtai) (Admitted := Admitted)
                              (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                              closed
                              (IvcSecurity.walkAdversary
                                (NebulaFinal.adversary setup app s q claim))
                              admitted (↑j : ℕ)) :
                          HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                            (IvcWalk.Answer logicalWidth publicFits)))) +
                ENNReal.toReal
                  (((NebulaSecurity.ClaimedStage.law app
                          (NebulaFinal.claimed setup app exports (Admitted := Admitted)
                            (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                            closed s q claim admitted depth)).toOuterMeasure :
                      Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                    {sample : NebulaSecurity.Sample app.width logicalWidth publicFits |
                      Nebula.EndToEnd.Experiment.StateLinkCollision app
                        (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys exports)
                        sample}) +
              ENNReal.toReal
                (((NebulaSecurity.ClaimedStage.law app
                        (NebulaFinal.claimed setup app exports (Admitted := Admitted)
                          (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                          closed s q claim admitted depth)).toOuterMeasure :
                    Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                  {sample : NebulaSecurity.Sample app.width logicalWidth publicFits |
                    Nebula.EndToEnd.Experiment.ChainCollision app
                      (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys exports)
                      sample}) +
            memoryError
              (NebulaFinal.claimed setup app exports (Admitted := Admitted) (StageAdmitted :=
                StageAdmitted) (Efficient := Efficient) assumption closed s q claim admitted
                depth)) :=
  @Export.BoundedDepth.NebulaFinal.soundness logicalWidth publicFits setup p app relation step
    cubeFits exports Admitted StageAdmitted Efficient error assumption closed s q claim admitted
    complete rows valid keysShaped programEq depth depthBound memoryError round

/-- With Akita openings, the probability `memory` of a false memory statement is at most
`q · (1/2^119 + 38/q²) + εAkita` plus the terms of `Claims.Nebula.memory_sound`. `akita` is Akita's
knowledge soundness (`FinalVerifier.AkitaKnowledge`) for the event `FalseMemory`: `s` is the
prover of the game with ideal openings that it gives for the prover with Akita openings. -/
theorem memory_sound_with_akita
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (setup : FinalVerifier.Setup logicalWidth publicFits)
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
    (depth : ℕ)
    (depthBound : ∀ (o : FinalVerifier.Outcome logicalWidth publicFits),
        some (some o) ∈ (Compression.Game.law s (setup.game.restore q [])).support →
          o.1.iteration ≤ depth)
    {memoryError : NebulaSecurity.ClaimedStage (logicalWidth := logicalWidth) (publicFits := publicFits) app →
        ℝ}
    (round : NebulaSecurity.MemoryRound app setup.ajtai setup.circuitKeys exports StageAdmitted
        memoryError)
    (memory : ℝ)
    (εAkita : ℝ)
    (akita : memory ≤
        ENNReal.toReal
            (((↑(HyperNova.ReverseStages.reverseStages (Succeeds :=
                          IvcWalk.WalkSucceeds setup.ajtai) (Extracted :=
                          IvcWalk.WalkExtracted setup.ajtai) (Admitted := Admitted)
                          (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                          closed
                          (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim))
                          admitted depth) :
                      HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                        (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
                Set
                    (↑(HyperNova.ReverseStages.reverseStages assumption closed
                            (IvcSecurity.walkAdversary
                              (NebulaFinal.adversary setup app s q claim))
                            admitted depth) :
                        HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                          (IvcWalk.Answer logicalWidth publicFits)).Tape →
                  ENNReal)
              {tape :
                (↑(HyperNova.ReverseStages.reverseStages (Succeeds :=
                        IvcWalk.WalkSucceeds setup.ajtai) (Extracted :=
                        IvcWalk.WalkExtracted setup.ajtai) (Admitted := Admitted)
                        (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                        closed
                        (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim))
                        admitted depth) :
                    HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)).Tape |
                NebulaFinal.FalseMemory setup app exports
                  (NebulaFinal.tapeOf setup app exports (Admitted := Admitted) (StageAdmitted :=
                      StageAdmitted) (Efficient := Efficient) assumption closed s q claim admitted
                      depth tape).1
                  (NebulaFinal.sampleOf setup app exports (Admitted := Admitted) (StageAdmitted :=
                    StageAdmitted) (Efficient := Efficient) assumption closed s q claim admitted
                    depth tape)}) +
          εAkita) :
    memory ≤
        (↑q : ℝ) * FinalVerifier.finalError + εAkita +
          (∑ j : Fin depth,
                  (IvcSecurity.finderChance setup.ajtai setup.circuitKeys exports
                      (HyperNova.ReverseStages.reverseStages assumption closed
                        (IvcSecurity.walkAdversary (NebulaFinal.adversary setup app s q claim))
                        admitted (↑j : ℕ)).1 +
                    error
                      (HyperNova.ReverseStages.Stage.nifs
                        (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
                        (IvcWalk.back setup.ajtai setup.circuitKeys exports)
                        (↑(HyperNova.ReverseStages.reverseStages (Succeeds :=
                              IvcWalk.WalkSucceeds setup.ajtai) (Extracted :=
                              IvcWalk.WalkExtracted setup.ajtai) (Admitted := Admitted)
                              (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                              closed
                              (IvcSecurity.walkAdversary
                                (NebulaFinal.adversary setup app s q claim))
                              admitted (↑j : ℕ)) :
                          HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                            (IvcWalk.Answer logicalWidth publicFits)))) +
                ENNReal.toReal
                  (((NebulaSecurity.ClaimedStage.law app
                          (NebulaFinal.claimed setup app exports (Admitted := Admitted)
                            (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                            closed s q claim admitted depth)).toOuterMeasure :
                      Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                    {sample : NebulaSecurity.Sample app.width logicalWidth publicFits |
                      Nebula.EndToEnd.Experiment.StateLinkCollision app
                        (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys exports)
                        sample}) +
              ENNReal.toReal
                (((NebulaSecurity.ClaimedStage.law app
                        (NebulaFinal.claimed setup app exports (Admitted := Admitted)
                          (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                          closed s q claim admitted depth)).toOuterMeasure :
                    Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                  {sample : NebulaSecurity.Sample app.width logicalWidth publicFits |
                    Nebula.EndToEnd.Experiment.ChainCollision app
                      (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys exports)
                      sample}) +
            memoryError
              (NebulaFinal.claimed setup app exports (Admitted := Admitted) (StageAdmitted :=
                StageAdmitted) (Efficient := Efficient) assumption closed s q claim admitted
                depth)) :=
  @Export.BoundedDepth.NebulaFinal.real_soundness logicalWidth publicFits setup p app relation
    step cubeFits exports Admitted StageAdmitted Efficient error assumption closed s q claim
    admitted complete rows valid keysShaped programEq depth depthBound memoryError round
    memory εAkita akita

end NightstreamFPrime.Claims.Final
