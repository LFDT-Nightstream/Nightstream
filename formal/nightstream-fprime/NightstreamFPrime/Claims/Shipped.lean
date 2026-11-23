import NightstreamFPrime.Export.BoundedDepth.Shipped

/-! # Claims: the public verifiers at the shipped packages

One claim for each verifier method of `nightstream`, at the shipped configuration
(`Export.BoundedDepth.Shipped`): the key width, the step relation argument, the two packages that
the emitter writes for the reference application, and the keys of a package identity.

- Premises: Lean proves each premise of class `lean` of `Claims.Ivc.verifier_sound` and
  `Claims.AkitaKey30.key30_history_sound` at this configuration. The claims keep the capacity of
  the packages (`facts`, which the emitter checks), the relations of the two keys and the rows of
  the final verifier (links), the bound on the iteration (`depthBound`), and the assumptions.
  `verify_final_sound` keeps the hypotheses `shared`, `drawn`, `coords`, `bounded` and `positive`
  of `Claims.AkitaKey30.key30_history_sound` on the Akita openings.
- Assumptions: HyperNova errata Assumption 1 for the two folds (`[HN-A1]`), with its closure
  premise and an admitted adversary. `verify_final_sound` is in the oracle model of
  `Claims.AkitaKey30` (`[KECCAK]`), and its term `FinalKey30.shapesBound` has the probability of a
  solution of Module-SIS (`[MSIS]`).
- Scope: the reference application only. A Rust-built application has no Lean package. -/

namespace NightstreamFPrime.Claims.Shipped

open Spec Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth
open scoped Spec.Compression.Production Spec.FieldExtensionRing
attribute [local instance low] Classical.propDecidable
attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK

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

/-- `Verifier::verify_final` (SPEC.md Algorithm 10) at the shipped packages of the reference
application, with the Akita openings of `crates/ns-compress/artifacts/key30.aks` (524288 key
blocks): `Claims.AkitaKey30.key30_history_sound` at this configuration. For each prover with at
most `queries` queries to the oracle whose accepted statements have at most `depth` steps, the
probability of a false history is at most `FinalKey30.shapesBound` plus `IvcSecurity.walkBound`.
The hypotheses on the Akita openings (`shared`, `drawn`, `coords`) are those of
`Claims.AkitaKey30.key30_history_sound`. The bound is a Nightstream bound (`Claims.AkitaKey30`,
Bounds) for the Lean verifier of the Akita opening, not for the pinned Akita verifier that Rust
calls, so the entry point is open in `SPEC.md` section 9. -/
theorem verify_final_sound
    {ajtai : Shipped.Ajtai}
    {mode : FinalVerifier.Mode}
    (facts : Shipped.Facts Shipped.program ajtai)
    (identity : List F)
    (relations : Fin 2 → Shipped.Relation)
    (lightKey : relations 0 = Shipped.lightRelation facts)
    (refreshKey : relations 1 = Shipped.refreshRelation facts)
    (setupRows : ℕ → Compression.Verifier.Factored F)
    (zeros : ℕ)
    (minimum : ℕ)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    (read : Compression.Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First StepRelation.keyWidth StepRelation.keyPublicFits)
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {ι : Type}
    [fintypeι : Fintype ι]
    (kind : Wire → Option ι)
    (family : ι → FinalKey30.Opening StepRelation.keyWidth StepRelation.keyPublicFits Wire Coord total)
    (shared : ∀ (index : ι), (family index).tower.Shared)
    (prover : RandomOracle.OracleComp (Compression.Akita.Extract.Address Wire total) (Option Coord → K)
      (ℕ → Wire))
    [finiteWire : Finite Wire]
    (budget : ℕ)
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
    (rows : ∀ circuit : Fin 2,
      (setupRows circuit).matrices =
        (FinalRows.packageRows (Shipped.exports facts identity relations lightKey refreshKey)
          (Shipped.complete facts identity relations lightKey refreshKey) circuit).matrices)
    (coords : 40 ≤ Fintype.card Coord)
    (drawn : ∀ (index : ι),
      FinalKey30.Opening.Drawn (Shipped.setup mode ajtai identity relations setupRows zeros minimum)
        (Shipped.complete facts identity relations lightKey refreshKey).packageWidth
        Shipped.program.step cap control read head (family index) (shared index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (positive : 0 < budget)
    (admitted : StageAdmitted
        (HyperNova.ReverseStages.Stage.start
          (IvcSecurity.walkAdversary
            (FinalKey30.adversary
              (Shipped.setup mode ajtai identity relations setupRows zeros minimum)
              (Shipped.complete facts identity relations lightKey refreshKey).packageWidth
              Shipped.program.step cap control read head kind family shared prover budget))))
    (depth : ℕ)
    (depthBound : ∀ (table : Compression.Akita.Extract.Address Wire total → Option Coord → K),
        FinalKey30.finalAccepts
            (Shipped.setup mode ajtai identity relations setupRows zeros minimum)
            (Shipped.complete facts identity relations lightKey refreshKey).packageWidth
            Shipped.program.step cap control read head kind family shared
            (FinalKey20.played prover table) →
          (head (FinalKey20.played prover table).first).statement.iteration ≤ depth) :
    ENNReal.toReal
        (((FinalKey30.adversary
              (Shipped.setup mode ajtai identity relations setupRows zeros minimum)
              (Shipped.complete facts identity relations lightKey refreshKey).packageWidth
              Shipped.program.step cap control read head kind family shared prover
              budget).tape.toOuterMeasure :
            Set (FinalKey30.adversary
              (Shipped.setup mode ajtai identity relations setupRows zeros minimum)
              (Shipped.complete facts identity relations lightKey refreshKey).packageWidth
              Shipped.program.step cap control read head kind family shared prover budget).Tape →
              ENNReal)
          {sample |
            FinalKey30.FalseHistory
              (Shipped.setup mode ajtai identity relations setupRows zeros minimum)
              (Shipped.complete facts identity relations lightKey refreshKey).packageWidth
              Shipped.program.step cap control read head kind family shared prover Shipped.program
              sample.1}) ≤
      FinalKey30.shapesBound (Shipped.setup mode ajtai identity relations setupRows zeros minimum)
          (Shipped.complete facts identity relations lightKey refreshKey).packageWidth
          Shipped.program.step cap control read head family shared prover queries budget +
        IvcSecurity.walkBound ajtai (Shipped.keys identity relations)
          (Shipped.exports facts identity relations lightKey refreshKey) assumption closed
          (FinalKey30.adversary
            (Shipped.setup mode ajtai identity relations setupRows zeros minimum)
            (Shipped.complete facts identity relations lightKey refreshKey).packageWidth
            Shipped.program.step cap control read head kind family shared prover budget)
          admitted depth :=
  @Export.BoundedDepth.Shipped.verify_final_sound ajtai mode facts identity relations lightKey
    refreshKey setupRows zeros minimum cap control Wire Coord total read head fintypeCoord
    decidableEqCoord nonemptyCoord ι fintypeι kind family shared prover finiteWire budget Admitted
    StageAdmitted Efficient error assumption closed rows coords drawn queries bounded positive
    admitted depth depthBound

end NightstreamFPrime.Claims.Shipped
