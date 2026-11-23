import NightstreamFPrime.Export.BoundedDepth.Shipped

/-! # Claims: the public verifiers at the shipped packages

The claim of `Verifier::verify` of `nightstream`, at the shipped configuration
(`Export.BoundedDepth.Shipped`): the key width, the step relation argument, the two packages that
the emitter writes for the reference application, and the keys of a package identity.

- Premises: Lean proves each premise of class `lean` of `Claims.Ivc.verifier_sound` at this
  configuration. The claim keeps the capacity of the packages (`facts`, which the emitter checks),
  the relations of the two keys, the bound on the iteration (`depthBound`), and the assumptions.
- Assumptions: HyperNova errata Assumption 1 for the two folds (`[HN-A1]`), with its closure
  premise and an admitted adversary.
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
    (cap : ℕ)
    (depth : ℕ)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    ENNReal.toReal
          ((adversary.tape.toOuterMeasure : Set adversary.Tape → ENNReal)
            {tape : adversary.Tape |
              VerifierSecurity.FalseVerification ajtai (Shipped.keys identity relations)
                (Shipped.exports facts identity relations lightKey refreshKey) packageWidths
                control cap (adversary.output tape)}) ≤
        IvcSecurity.walkBound ajtai (Shipped.keys identity relations)
          (Shipped.exports facts identity relations lightKey refreshKey) assumption closed
          adversary admitted depth :=
  @Export.BoundedDepth.Shipped.verify_sound ajtai facts identity relations lightKey refreshKey
    Admitted StageAdmitted Efficient error assumption closed adversary admitted packageWidths
    control cap depth depthBound

end NightstreamFPrime.Claims.Shipped
