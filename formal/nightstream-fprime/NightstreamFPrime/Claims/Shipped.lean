import NightstreamFPrime.Export.BoundedDepth.Shipped

/-! # Claims: the public verifiers at the shipped packages

One claim for each verifier method of `nightstream`, at the shipped configuration
(`Export.BoundedDepth.Shipped`): the key width, the step relation argument, the two packages that
the emitter writes for the reference application, and the keys of a package identity.

- Premises: Lean proves each premise of class `lean` of `Claims.Ivc.verifier_sound` and
  `Claims.Final.history_sound_with_akita` at this configuration. The claims keep the capacity of
  the packages (`facts`, which the emitter checks), the relations of the two keys and the rows of
  the final verifier (links), the bound on the iteration (`depthBound`), and the assumptions.
- Assumptions: HyperNova errata Assumption 1 for the two folds (`[HN-A1]`), with its closure
  premise and an admitted adversary, and `[AKITA]` for key binding in `verify_final_sound`.
  The fixed-setup extraction bound, schedule admissibility and composition are open premises
  of `Claims.Final.history_sound_with_akita`.
- Scope: the reference application only. A Rust-built application has no Lean package. -/

namespace NightstreamFPrime.Claims.Shipped

open Spec Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth
open scoped Spec.Compression.Production

/-- `Verifier::verify` (SPEC.md Algorithm 6) at the shipped packages of the reference
application: the probability that the verifier accepts a statement that no valid run of the
program reaches is at most `IvcSecurity.walkBound`. -/
theorem verify_sound
    {ajtai : Shipped.Ajtai}
    (facts : Shipped.Facts Shipped.program ajtai)
    (identity : List F)
    (relations : Fin 2 → Shipped.Relation)
    (lightKey : relations 0 = Shipped.lightRelation facts)
    (refreshKey : relations 1 = Shipped.refreshRelation facts)
    {Admitted : HyperNova.ReverseStages.NifsAdversary
        (IvcWalk.WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage
          (IvcWalk.Visit StepRelation.keyWidth StepRelation.keyPublicFits)
          (IvcWalk.Answer StepRelation.keyWidth StepRelation.keyPublicFits) →
        Prop}
    {Efficient : (adversary : HyperNova.ReverseStages.NifsAdversary
          (IvcWalk.WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits)) →
        HyperNova.ReverseStages.NifsExtractor
          (IvcWalk.Answer StepRelation.keyWidth StepRelation.keyPublicFits) adversary → Prop}
    {error : HyperNova.ReverseStages.NifsAdversary
        (IvcWalk.WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds ajtai)
        (IvcWalk.WalkExtracted ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed
        (IvcWalk.attemptOf ajtai (Shipped.keys identity relations)
          (Shipped.exports facts identity relations lightKey refreshKey))
        (IvcWalk.back ajtai (Shipped.keys identity relations)
          (Shipped.exports facts identity relations lightKey refreshKey))
        Admitted StageAdmitted Efficient)
    (adversary : HyperNova.ReverseStages.IvcAdversary
        (IvcSecurity.Output StepRelation.keyWidth StepRelation.keyPublicFits))
    (admitted : StageAdmitted
        (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary adversary)))
    (packageWidths : Fin 2 → ℕ)
    (control : AppState → AppWitness → Fin 2)
    (depth : ℕ)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    ENNReal.toReal
          ((adversary.tape.toOuterMeasure : Set adversary.Tape → ENNReal)
            {tape : adversary.Tape |
              VerifierSecurity.FalseVerification ajtai (Shipped.keys identity relations)
                (Shipped.exports facts identity relations lightKey refreshKey) packageWidths
                control (adversary.output tape)}) ≤
        IvcSecurity.walkBound ajtai (Shipped.keys identity relations)
          (Shipped.exports facts identity relations lightKey refreshKey) assumption closed
          adversary admitted depth :=
  @Export.BoundedDepth.Shipped.verify_sound ajtai facts identity relations lightKey refreshKey
    Admitted StageAdmitted Efficient error assumption closed adversary admitted packageWidths
    control depth depthBound

/-- `Verifier::verify_final` (SPEC.md Algorithm 10) at the shipped packages of the reference
application, with Akita openings: the probability of a false history is at most
`q · (1/2^119 + 38/fieldModulus²)` plus `Spec.Compression.Akita.Use.error` and
`IvcSecurity.walkBound`, under the premises of `Claims.Final.history_sound_with_akita`. -/
theorem verify_final_sound
    {ajtai : Shipped.Ajtai}
    {mode : FinalVerifier.Mode}
    (facts : Shipped.Facts Shipped.program ajtai)
    (identity : List F)
    (relations : Fin 2 → Shipped.Relation)
    (lightKey : relations 0 = Shipped.lightRelation facts)
    (refreshKey : relations 1 = Shipped.refreshRelation facts)
    (setupRows : ℕ → Compression.Verifier.Factored F)
    (keyBlocks : ℕ)
    (zeros : ℕ)
    (minimum : ℕ)
    {Admitted : HyperNova.ReverseStages.NifsAdversary
        (IvcWalk.WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage
          (IvcWalk.Visit StepRelation.keyWidth StepRelation.keyPublicFits)
          (IvcWalk.Answer StepRelation.keyWidth StepRelation.keyPublicFits) →
        Prop}
    {Efficient : (adversary : HyperNova.ReverseStages.NifsAdversary
          (IvcWalk.WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits)) →
        HyperNova.ReverseStages.NifsExtractor
          (IvcWalk.Answer StepRelation.keyWidth StepRelation.keyPublicFits) adversary → Prop}
    {error : HyperNova.ReverseStages.NifsAdversary
        (IvcWalk.WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds ajtai)
        (IvcWalk.WalkExtracted ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed
        (IvcWalk.attemptOf ajtai (Shipped.keys identity relations)
          (Shipped.exports facts identity relations lightKey refreshKey))
        (IvcWalk.back ajtai (Shipped.keys identity relations)
          (Shipped.exports facts identity relations lightKey refreshKey))
        Admitted StageAdmitted Efficient)
    (s : Compression.Game.Strategy
        (Option (Option (FinalVerifier.Outcome StepRelation.keyWidth StepRelation.keyPublicFits))))
    (q : ℕ)
    (admitted : StageAdmitted
        (HyperNova.ReverseStages.Stage.start
          (IvcSecurity.walkAdversary
            (FinalHistory.adversary
              (Shipped.setup mode ajtai identity relations setupRows keyBlocks zeros minimum) s q))))
    (rows : ∀ circuit : Fin 2,
      (setupRows circuit).matrices =
        (FinalRows.packageRows (Shipped.exports facts identity relations lightKey refreshKey)
          (Shipped.complete facts identity relations lightKey refreshKey) circuit).matrices)
    (depth : ℕ)
    (depthBound : ∀ (o : FinalVerifier.Outcome StepRelation.keyWidth StepRelation.keyPublicFits),
        some (some o) ∈ (Compression.Game.law s
          ((Shipped.setup mode ajtai identity relations setupRows keyBlocks zeros
            minimum).game.restore q [])).support →
          o.1.iteration ≤ depth)
    (opening : Compression.Akita.Use)
    (real : Compression.Akita.Prover opening.scheme opening.claims)
    (event : real.Tape → Prop)
    (akita : opening.Knowledge real)
    (binding : opening.Binding)
    (accepts : ∀ (tape : real.Tape), event tape → opening.Opens real tape)
    (composition : opening.Composition real event
        (Compression.Game.prob s
          ((Shipped.setup mode ajtai identity relations setupRows keyBlocks zeros
            minimum).game.restore q [])
          (FinalHistory.FalseHistory Shipped.program))) :
    ENNReal.toReal (real.tape.toOuterMeasure {tape | event tape}) ≤
      (↑q : ℝ) * FinalVerifier.finalError + opening.error real +
        IvcSecurity.walkBound ajtai (Shipped.keys identity relations)
          (Shipped.exports facts identity relations lightKey refreshKey) assumption closed
          (FinalHistory.adversary (Shipped.setup mode ajtai identity relations setupRows keyBlocks
            zeros minimum) s q)
          admitted depth :=
  @Export.BoundedDepth.Shipped.verify_final_sound ajtai mode facts identity relations lightKey
    refreshKey setupRows keyBlocks zeros minimum Admitted StageAdmitted Efficient error assumption
    closed s q admitted rows depth depthBound opening real event akita binding accepts composition

end NightstreamFPrime.Claims.Shipped
