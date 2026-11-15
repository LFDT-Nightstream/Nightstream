import NightstreamFPrime.Export.BoundedDepth.IvcSecurity
import NightstreamFPrime.Export.BoundedDepth.RunCompleteness
import NightstreamFPrime.Export.BoundedDepth.TerminalBridge
import NightstreamFPrime.Export.BoundedDepth.VerifierSecurity
import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth
import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge

/-! # Claims: bounded-depth IVC

The terminal relation, the reverse extractor and the completeness of a run
(`SPEC.md` Algorithm 6, `SECURITY_MODEL.md` adversaries 2 and 4).

- Paper: HyperNova Construction 2, Lemma 17 and Definition 11.
- Assumptions: `false_acceptance_bound`, `extraction_bound` and `verifier_sound` take HyperNova
  errata Assumption 1 for the two folds (`[HN-A1]`), with its closure premise and an admitted
  adversary. Their error term is the `error` of that assumption.
- Open: Lean does not bound the collision term of these three claims. No instance of
  `Export.BoundedDepth.TerminalBridge.Exports` exists for the emitted packages. The bound on the
  iteration of an accepted statement is an argument (`depthBound`). -/

namespace NightstreamFPrime.Claims.Ivc

open Spec Spec.HyperNova Spec.HyperNova.Construction2 Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth

/-- If the terminal relation accepts a recursive proof and the key of each circuit carries the
relation that Lean exports, the circuit that ran last holds its step relation on the step that
the fresh witness decodes to, and that step ends in the terminal's statement, or a Poseidon2
collision occurred. -/
theorem terminal_bridge
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation ajtai circuitKeys step cubeFits)
    (keysShaped : ∀ (key : Fin 2), List.length (circuitKeys key).digest = 4)
    (control : AppState → AppWitness → Fin 2)
    (cap : ℕ)
    (statement : Paper.TerminalStatement AppState)
    (payload : Paper.TerminalProof (SlotData logicalWidth publicFits)
        (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
        (StepFresh logicalWidth publicFits)
        (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (accepted : Terminal.HoldsFor cap ajtai circuitKeys step control statement
        (Paper.OuterTerminalProof.recursive payload)) :
    BoundedDepth.StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step
            (TerminalBridge.chosen exports.program (TerminalBridge.circuitOf payload.pc))
            (TerminalBridge.circuitOf payload.pc) (exports.last payload).input
            (exports.last payload).output ∧
          TerminalBridge.EndsIn statement payload.running payload.pc
            (TerminalBridge.circuitOf payload.pc) (exports.last payload).input
            (exports.last payload).output ∨
        TerminalBridge.Collision exports statement payload :=
  @Export.BoundedDepth.TerminalBridge.bridge logicalWidth publicFits relation ajtai circuitKeys
    step cubeFits exports keysShaped control cap statement payload accepted

/-- Every accumulator of an accepted history stays below `B` when honest, and extraction
recovers each below `B + cap · T · (b − 1)`. -/
theorem accepted_bounds.{uKey, uClaim, uFresh, uProof}
    {Key : Type uKey}
    {Claim : Type uClaim}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (p : GlobalParams)
    {cap : ℕ}
    (capFits : 1 + cap ≤ p.maxFresh)
    (folds : BoundedDepth.Folds Key Claim Fresh Proof)
    (history : List (BoundedDepth.Fold Key Claim Fresh Proof))
    (depth : ℕ)
    (start : depth ≤ cap)
    (accepted : BoundedDepth.AcceptedFrom cap folds depth history)
    {final : ℕ}
    (finalAtMostB : final ≤ p.bigB) :
    (∀ reached ∈ Folding.Schedule.depths depth (BoundedDepth.steps history),
          p.accumulatorBound reached < p.bigB) ∧
        ∀ bound ∈ Folding.Schedule.recoveredBounds p final (BoundedDepth.steps history),
          bound ≤ p.bigB + cap * (p.expansionT * (p.b - 1)) :=
  @Spec.HyperNova.Construction2.BoundedDepth.accepted_bounds Key Claim Fresh Proof p cap capFits
    folds history depth start accepted final finalAtMostB

/-- HyperNova Lemma 17's reverse extraction for the bounded-depth scheme on one slot of an accepted execution:
from a witness of the slot's final claim below `B`, every claim the slot held has a witness at a
bound of at most `B + cap · T · (b − 1)` and every fresh instance it folded has a witness, or a
fold of the slot failed. -/
theorem execution_extracts.{uKey, uClaim, uFresh, uProof, uClaimWitness, uFreshWitness}
    {Key : Type uKey}
    {Claim : Type uClaim}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {ClaimWitness : Type uClaimWitness}
    {FreshWitness : Type uFreshWitness}
    {p : GlobalParams}
    {cap : ℕ}
    {folds : BoundedDepth.Folds Key Claim Fresh Proof}
    {Holds : ℕ → Claim → ClaimWitness → Prop}
    {FreshHolds : Fresh → FreshWitness → Prop}
    {Failure : Folding.Schedule.Step → Key → Claim → Fresh → Proof → ClaimWitness → Prop}
    {Digest : Type}
    {State : Type}
    {StepWitness : Type}
    {Encoded : Type}
    (knowledge : BoundedDepthKnowledge.StepKnowledge p folds Holds FreshHolds Failure)
    (verifierKeys : Fin 2 → Key)
    (defaultClaim : Claim)
    (machine : Paper.Machine.{uKey, 0, 0, 0, uClaim, uFresh, 0} Key Digest State StepWitness
        (BoundedDepth.Running Claim) Fresh Encoded 2)
    (trace : List (BoundedDepthKnowledge.MachineStep Key Claim Fresh Proof Digest State StepWitness))
    (start : Fin 2 → BoundedDepth.Running Claim)
    (final : Fin 2 → BoundedDepth.Running Claim)
    (execution : BoundedDepthKnowledge.Execution
        (Construction2.BoundedDepth.setup cap folds verifierKeys defaultClaim) machine start trace
        final)
    (slot : Fin 2)
    (bound : ℕ)
    (finalWitness : ClaimWitness)
    (holds : Holds bound (final slot).1 finalWitness)
    (finalAtMostB : bound ≤ p.bigB) :
    have history := BoundedDepthKnowledge.slotHistory verifierKeys slot trace;
      (BoundedDepthKnowledge.ValidHistory p Holds FreshHolds (start slot).1 history bound ∧
          ∀ recovered ∈ Folding.Schedule.recoveredBounds p bound (BoundedDepth.steps history),
            recovered ≤ p.bigB + cap * (p.expansionT * (p.b - 1))) ∨
        BoundedDepthKnowledge.Failed p Holds Failure bound history :=
  @Spec.HyperNova.Construction2.BoundedDepthKnowledge.execution_extracts Key Claim Fresh Proof
    ClaimWitness FreshWitness p cap folds Holds FreshHolds Failure Digest State StepWitness
    Encoded knowledge verifierKeys defaultClaim machine trace start final execution slot bound
    finalWitness holds finalAtMostB

/-- HyperNova Lemma 17 for the bounded-depth scheme. A false acceptance is an extraction failure, because no
list of witnesses reaches its statement. -/
theorem false_acceptance_bound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation ajtai circuitKeys step cubeFits)
    {Admitted : ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
          (IvcWalk.Answer logicalWidth publicFits) →
        Prop}
    {Efficient : (adversary : ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
        ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary → Prop}
    {error : ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : ReverseStages.Assumption1 (IvcWalk.WalkSucceeds ajtai) (IvcWalk.WalkExtracted ajtai)
        Admitted Efficient error)
    (closed : ReverseStages.Closed (IvcWalk.attemptOf ajtai circuitKeys exports)
        (IvcWalk.back ajtai circuitKeys exports) Admitted StageAdmitted Efficient)
    (adversary : ReverseStages.IvcAdversary (IvcSecurity.Output logicalWidth publicFits))
    (admitted : StageAdmitted (ReverseStages.Stage.start (IvcSecurity.walkAdversary adversary)))
    (keysShaped : ∀ (key : Fin 2), List.length (circuitKeys key).digest = 4)
    (control : AppState → AppWitness → Fin 2)
    (cap : ℕ)
    (depth : ℕ)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    ENNReal.toReal
          ((adversary.tape.toOuterMeasure : Set adversary.Tape → ENNReal)
            {tape : adversary.Tape |
              IvcSecurity.FalseAcceptance ajtai circuitKeys exports control cap
                (adversary.output tape)}) ≤
        ∑ j : Fin depth,
          (ENNReal.toReal
              (((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                            (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                            (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                            closed (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                        ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                          (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
                  Set
                      (↑(ReverseStages.reverseStages assumption closed
                              (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                          ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                            (IvcWalk.Answer logicalWidth publicFits)).Tape →
                    ENNReal)
                {tape :
                  (↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                          (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                          (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                          closed (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                      ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                        (IvcWalk.Answer logicalWidth publicFits)).Tape |
                  IvcWalk.WalkCollision ajtai circuitKeys exports
                    (ReverseStages.visitAfter (IvcWalk.back ajtai circuitKeys exports)
                      ((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                                (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                                (StageAdmitted := StageAdmitted) (Efficient := Efficient)
                                assumption closed (IvcSecurity.walkAdversary adversary) admitted
                                (↑j : ℕ)) :
                            ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                              (IvcWalk.Answer logicalWidth publicFits)).input
                        tape)
                      ((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                                (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                                (StageAdmitted := StageAdmitted) (Efficient := Efficient)
                                assumption closed (IvcSecurity.walkAdversary adversary) admitted
                                (↑j : ℕ)) :
                            ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                              (IvcWalk.Answer logicalWidth publicFits)).results
                        tape))}) +
            error
              (ReverseStages.Stage.nifs (IvcWalk.attemptOf ajtai circuitKeys exports)
                (IvcWalk.back ajtai circuitKeys exports)
                (↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                      (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                      (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption closed
                      (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                  ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)))) :=
  @Export.BoundedDepth.IvcSecurity.probability_bound logicalWidth publicFits relation ajtai
    circuitKeys step cubeFits exports Admitted StageAdmitted Efficient error assumption closed
    adversary admitted keysShaped control cap depth depthBound

/-- HyperNova Lemma 17 for the bounded-depth scheme, on the extractor's output: the probability that the terminal
accepts the adversary's output and the witnesses that the reverse extractor returns do not lead
from `z0` to `zi`. The law is the one of the last stage: the adversary's tape and the coins of
each stage's extractor. -/
theorem extraction_bound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation ajtai circuitKeys step cubeFits)
    {Admitted : ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
          (IvcWalk.Answer logicalWidth publicFits) →
        Prop}
    {Efficient : (adversary : ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
        ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary → Prop}
    {error : ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : ReverseStages.Assumption1 (IvcWalk.WalkSucceeds ajtai) (IvcWalk.WalkExtracted ajtai)
        Admitted Efficient error)
    (closed : ReverseStages.Closed (IvcWalk.attemptOf ajtai circuitKeys exports)
        (IvcWalk.back ajtai circuitKeys exports) Admitted StageAdmitted Efficient)
    (adversary : ReverseStages.IvcAdversary (IvcSecurity.Output logicalWidth publicFits))
    (admitted : StageAdmitted (ReverseStages.Stage.start (IvcSecurity.walkAdversary adversary)))
    (keysShaped : ∀ (key : Fin 2), List.length (circuitKeys key).digest = 4)
    (control : AppState → AppWitness → Fin 2)
    (cap : ℕ)
    (depth : ℕ)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    ENNReal.toReal
          (((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai) (Extracted :=
                        IvcWalk.WalkExtracted ajtai) (Admitted := Admitted) (StageAdmitted :=
                        StageAdmitted) (Efficient := Efficient) assumption closed
                        (IvcSecurity.walkAdversary adversary) admitted depth) :
                    ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
              Set
                  (↑(ReverseStages.reverseStages assumption closed
                          (IvcSecurity.walkAdversary adversary) admitted depth) :
                      ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                        (IvcWalk.Answer logicalWidth publicFits)).Tape →
                ENNReal)
            {tape :
              (↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                      (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                      (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption closed
                      (IvcSecurity.walkAdversary adversary) admitted depth) :
                  ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)).Tape |
              IvcSecurity.ExtractionFailure ajtai circuitKeys exports control cap
                ((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                          (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                          (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                          closed (IvcSecurity.walkAdversary adversary) admitted depth) :
                      ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                        (IvcWalk.Answer logicalWidth publicFits)).input
                  tape)
                ((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                          (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                          (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                          closed (IvcSecurity.walkAdversary adversary) admitted depth) :
                      ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                        (IvcWalk.Answer logicalWidth publicFits)).results
                  tape)}) ≤
        ∑ j : Fin depth,
          (ENNReal.toReal
              (((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                            (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                            (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                            closed (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                        ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                          (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
                  Set
                      (↑(ReverseStages.reverseStages assumption closed
                              (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                          ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                            (IvcWalk.Answer logicalWidth publicFits)).Tape →
                    ENNReal)
                {tape :
                  (↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                          (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                          (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                          closed (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                      ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                        (IvcWalk.Answer logicalWidth publicFits)).Tape |
                  IvcWalk.WalkCollision ajtai circuitKeys exports
                    (ReverseStages.visitAfter (IvcWalk.back ajtai circuitKeys exports)
                      ((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                                (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                                (StageAdmitted := StageAdmitted) (Efficient := Efficient)
                                assumption closed (IvcSecurity.walkAdversary adversary) admitted
                                (↑j : ℕ)) :
                            ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                              (IvcWalk.Answer logicalWidth publicFits)).input
                        tape)
                      ((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                                (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                                (StageAdmitted := StageAdmitted) (Efficient := Efficient)
                                assumption closed (IvcSecurity.walkAdversary adversary) admitted
                                (↑j : ℕ)) :
                            ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                              (IvcWalk.Answer logicalWidth publicFits)).results
                        tape))}) +
            error
              (ReverseStages.Stage.nifs (IvcWalk.attemptOf ajtai circuitKeys exports)
                (IvcWalk.back ajtai circuitKeys exports)
                (↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                      (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                      (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption closed
                      (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                  ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)))) :=
  @Export.BoundedDepth.IvcSecurity.extraction_bound logicalWidth publicFits relation ajtai
    circuitKeys step cubeFits exports Admitted StageAdmitted Efficient error assumption closed
    adversary admitted keysShaped control cap depth depthBound

/-- The probability that the verifier of SPEC.md Algorithm 6 accepts the adversary's output and no
`iteration` valid steps lead from `z0` to `zi` is at most the sum of `probability_bound`. -/
theorem verifier_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation ajtai circuitKeys step cubeFits)
    {Admitted : ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
          (IvcWalk.Answer logicalWidth publicFits) →
        Prop}
    {Efficient : (adversary : ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
        ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary → Prop}
    {error : ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : ReverseStages.Assumption1 (IvcWalk.WalkSucceeds ajtai) (IvcWalk.WalkExtracted ajtai)
        Admitted Efficient error)
    (closed : ReverseStages.Closed (IvcWalk.attemptOf ajtai circuitKeys exports)
        (IvcWalk.back ajtai circuitKeys exports) Admitted StageAdmitted Efficient)
    (adversary : ReverseStages.IvcAdversary (IvcSecurity.Output logicalWidth publicFits))
    (admitted : StageAdmitted (ReverseStages.Stage.start (IvcSecurity.walkAdversary adversary)))
    (keysShaped : ∀ (key : Fin 2), List.length (circuitKeys key).digest = 4)
    (packageWidths : Fin 2 → ℕ)
    (control : AppState → AppWitness → Fin 2)
    (cap : ℕ)
    (depth : ℕ)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    ENNReal.toReal
          ((adversary.tape.toOuterMeasure : Set adversary.Tape → ENNReal)
            {tape : adversary.Tape |
              VerifierSecurity.FalseVerification ajtai circuitKeys exports packageWidths control
                cap (adversary.output tape)}) ≤
        ∑ j : Fin depth,
          (ENNReal.toReal
              (((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                            (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                            (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                            closed (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                        ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                          (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
                  Set
                      (↑(ReverseStages.reverseStages assumption closed
                              (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                          ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                            (IvcWalk.Answer logicalWidth publicFits)).Tape →
                    ENNReal)
                {tape :
                  (↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                          (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                          (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption
                          closed (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                      ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                        (IvcWalk.Answer logicalWidth publicFits)).Tape |
                  IvcWalk.WalkCollision ajtai circuitKeys exports
                    (ReverseStages.visitAfter (IvcWalk.back ajtai circuitKeys exports)
                      ((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                                (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                                (StageAdmitted := StageAdmitted) (Efficient := Efficient)
                                assumption closed (IvcSecurity.walkAdversary adversary) admitted
                                (↑j : ℕ)) :
                            ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                              (IvcWalk.Answer logicalWidth publicFits)).input
                        tape)
                      ((↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                                (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                                (StageAdmitted := StageAdmitted) (Efficient := Efficient)
                                assumption closed (IvcSecurity.walkAdversary adversary) admitted
                                (↑j : ℕ)) :
                            ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                              (IvcWalk.Answer logicalWidth publicFits)).results
                        tape))}) +
            error
              (ReverseStages.Stage.nifs (IvcWalk.attemptOf ajtai circuitKeys exports)
                (IvcWalk.back ajtai circuitKeys exports)
                (↑(ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                      (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                      (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption closed
                      (IvcSecurity.walkAdversary adversary) admitted (↑j : ℕ)) :
                  ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)))) :=
  @Export.BoundedDepth.VerifierSecurity.soundness logicalWidth publicFits relation ajtai
    circuitKeys step cubeFits exports Admitted StageAdmitted Efficient error assumption closed
    adversary admitted keysShaped packageWidths control cap depth depthBound

/-- Completeness of bounded-depth IVC: for every valid run of the application from `z0` whose
witnesses have the application's witness length, the terminal accepts the honest prover's proof
of the run's last state. -/
theorem run_complete
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation ajtai circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (keysShaped : ∀ (key : Fin 2), List.length (circuitKeys key).digest = 4)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    (z0 : AppState)
    (z0Length : List.length z0 = Stage1.Application.stateWordCount)
    (witnesses : List AppWitness)
    (validRun : IvcWalk.ValidFrom exports.program z0 witnesses)
    (witnessLengths : ∀ witness ∈ witnesses, List.length witness = exports.program.witnessWordCount)
    (runBelow : witnesses.length < fieldModulus) :
    RunCompleteness.Accepts ajtai circuitKeys step complete.packageWidth cap control
        { iteration := witnesses.length, z0 := z0, zi := List.foldl step z0 witnesses } :=
  @Export.BoundedDepth.RunCompleteness.run_complete logicalWidth publicFits relation ajtai
    circuitKeys step cubeFits exports complete keysShaped cap control z0 z0Length witnesses validRun
    witnessLengths runBelow

/-- HyperNova Definition 11 (i) for the bounded-depth scheme: one valid step extends each proof that the verifier
accepts to a recursive proof that the verifier accepts at the next iteration. -/
theorem step_extends
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation ajtai circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (keysShaped : ∀ (key : Fin 2), List.length (circuitKeys key).digest = 4)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    (statement : Paper.TerminalStatement AppState)
    (proof : Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits))
    (verifies : RunCompleteness.Verifies ajtai circuitKeys step complete.packageWidth cap control statement
        proof)
    (witness : AppWitness)
    (valid : exports.program.valid statement.zi witness)
    (iterationNext : statement.iteration + 1 < fieldModulus)
    (witnessLength : List.length witness = exports.program.witnessWordCount) :
    ∃ (next :
        Paper.TerminalProof (SlotData logicalWidth publicFits)
          (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
          (StepFresh logicalWidth publicFits)
          (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2),
        RunCompleteness.Verifies ajtai circuitKeys step complete.packageWidth cap control
          { iteration := statement.iteration + 1, z0 := statement.z0,
            zi := step statement.zi witness }
          (Paper.OuterTerminalProof.recursive next) :=
  @Export.BoundedDepth.RunCompleteness.verifies_step logicalWidth publicFits relation ajtai
    circuitKeys step cubeFits exports complete keysShaped cap control statement proof verifies
    witness valid iterationNext witnessLength

end NightstreamFPrime.Claims.Ivc
