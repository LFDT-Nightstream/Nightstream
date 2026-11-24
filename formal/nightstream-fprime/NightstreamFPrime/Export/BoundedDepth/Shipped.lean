import NightstreamFPrime.Export.BoundedDepth.EmittedPackages
import NightstreamFPrime.Export.BoundedDepth.FinalHistory
import NightstreamFPrime.Export.BoundedDepth.FinalKey30All
import NightstreamFPrime.Export.BoundedDepth.VerifierSecurity
import NightstreamFPrime.Lifecycle.BoundedDepth.CircuitKeyDigest
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1

/-! The shipped configuration: the key width `StepRelation.keyWidth`, the step relation argument
`StepRelation.keyRelation`, the two packages that the emitter writes for an application
(`EmittedPackages.Emitted`), and the keys of a package identity
(`Lifecycle.BoundedDepth.CircuitKeyDigest.keys`). The reference application is the Poseidon2
hash chain (`EmitMain.program`).

At this configuration Lean proves each premise of class `lean` of the claims: the program's
fields, the widths, the slot layouts and the key digest lengths. A theorem here keeps the capacity
of the two packages (`EmittedPackages.Facts`), the relations of the two keys, and the premises of
the other classes. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.Shipped

open NightstreamFPrime.Spec.Compression.Akita.Replay (CodeSlot)

open scoped ENNReal
open scoped NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.ReverseStages
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.BoundedDepth
open NightstreamFPrime.Export.BoundedDepth.StepCompleteness
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.BoundedDepth.TerminalBridge
open NightstreamFPrime.Export.BoundedDepth.IvcWalk
open NightstreamFPrime.Export.BoundedDepth.IvcSecurity
open NightstreamFPrime.Export.BoundedDepth.EmittedPackages
open NightstreamFPrime.Export.BoundedDepth.FinalVerifier

/-- The reference application of the shipped packages. -/
abbrev program : Stage1.Application.Program := Stage1.Poseidon2HashChainV1.program

/-- A logical relation at the key width. -/
abbrev Relation := ProductionKey.LogicalRelation StepRelation.keyWidth StepRelation.keyPublicFits

/-- An Ajtai key at the key width. -/
abbrev Ajtai := PaperAlgebra.AjtaiKey (logicalWidth := StepRelation.keyWidth)
  (publicFits := StepRelation.keyPublicFits)

/-- The capacity of the two packages of `application` (SPEC.md section 4.4), which the emitter
checks. -/
abbrev Facts (application : Stage1.Application.Program) (ajtai : Ajtai) :=
  Emitted application StepRelation.keyRelation ajtai

/-- The first fresh column of the light package is the end of the light circuit's locals. Stated
for every program, so that no proof at a shipped program unfolds its operations. -/
theorem lightAfterLocals (application : Stage1.Application.Program) :
    LightAssembler.sharedEnd StepRelation.base application.witnessWordCount +
      localLength (lightOps application StepRelation.keyRelation) ≤
        lightFresh application StepRelation.keyRelation :=
  Nat.le_refl _

/-- The first fresh column of the refresh package is the end of the refresh circuit's locals. -/
theorem refreshAfterLocals (application : Stage1.Application.Program) :
    RefreshAssembler.sharedEnd StepRelation.base application.witnessWordCount +
      localLength (refreshOps application StepRelation.keyRelation) ≤
        refreshFresh application StepRelation.keyRelation :=
  Nat.le_refl _

variable {application : Stage1.Application.Program} {ajtai : Ajtai}

/-- The logical relation of the light package. -/
def lightRelation (facts : Facts application ajtai) : Relation :=
  (CircuitRelation.plan _ _ (lightFresh application StepRelation.keyRelation) sourceWidth
    (facts.lightFits (lightAfterLocals application)) facts.lightLayout).logicalRelation
      StepRelation.keyCubeFits

/-- The logical relation of the refresh package. -/
def refreshRelation (facts : Facts application ajtai) : Relation :=
  (CircuitRelation.plan _ _ (refreshFresh application StepRelation.keyRelation) sourceWidth
    (facts.refreshFits (refreshAfterLocals application)) facts.refreshLayout).logicalRelation
      StepRelation.keyCubeFits

/-- The keys of a package identity with the relations `relations`. -/
abbrev keys (identity : List F) (relations : Fin 2 → Relation) :
    Fin 2 → CircuitKey StepRelation.keyWidth StepRelation.keyPublicFits :=
  CircuitKeyDigest.keys identity relations

/-- The bundle of the two packages, when the keys carry their relations. -/
def exports (facts : Facts application ajtai) (identity : List F) (relations : Fin 2 → Relation)
    (lightKey : relations 0 = lightRelation facts)
    (refreshKey : relations 1 = refreshRelation facts) :
    Exports StepRelation.keyRelation ajtai (keys identity relations) application.step
      StepRelation.keyCubeFits :=
  facts.exports (circuitKeys := keys identity relations) (cubeFits := StepRelation.keyCubeFits)
    (lightAfterLocals application) (refreshAfterLocals application) lightKey refreshKey

/-- The completeness bundle of the two packages. -/
def complete (facts : Facts application ajtai) (identity : List F) (relations : Fin 2 → Relation)
    (lightKey : relations 0 = lightRelation facts)
    (refreshKey : relations 1 = refreshRelation facts) :
    StepExtend.Complete (exports facts identity relations lightKey refreshKey) :=
  Emitted.complete (circuitKeys := keys identity relations) (cubeFits := StepRelation.keyCubeFits)
    facts lightKey refreshKey

/-- The setup of the final verifier with the Ajtai key, the keys of a package identity, the rows
`rows`, the 524288 key blocks of `crates/ns-compress/artifacts/key30.aks`, and the security
minimum. -/
def setup (mode : Mode) (ajtai : Ajtai) (identity : List F) (relations : Fin 2 → Relation)
    (rows : ℕ → Compression.Verifier.Factored F) (zeros minimum : ℕ) :
    Setup mode StepRelation.keyWidth StepRelation.keyPublicFits :=
  ⟨ajtai, keys identity relations, rows, 524288, zeros, minimum⟩

/-- `Verifier::verify` (SPEC.md Algorithm 6) at the shipped packages of the reference
application: the probability that the verifier accepts a statement that no valid run of the
program reaches is at most `walkBound` (`VerifierSecurity.soundness`). -/
theorem verify_sound (facts : Facts program ajtai) (identity : List F)
    (relations : Fin 2 → Relation) (lightKey : relations 0 = lightRelation facts)
    (refreshKey : relations 1 = refreshRelation facts)
    {Admitted : NifsAdversary (WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits) → Prop}
    {StageAdmitted : Stage (Visit StepRelation.keyWidth StepRelation.keyPublicFits)
      (Answer StepRelation.keyWidth StepRelation.keyPublicFits) → Prop}
    {Efficient : (adversary : NifsAdversary (WalkAttempt StepRelation.keyWidth
        StepRelation.keyPublicFits)) →
      NifsExtractor (Answer StepRelation.keyWidth StepRelation.keyPublicFits) adversary → Prop}
    {error : NifsAdversary (WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits) → ℝ}
    (assumption : Assumption1 (WalkSucceeds ajtai) (WalkExtracted ajtai) Admitted Efficient error)
    (closed : Closed (attemptOf ajtai (keys identity relations)
        (exports facts identity relations lightKey refreshKey))
      (back ajtai (keys identity relations) (exports facts identity relations lightKey refreshKey))
      Admitted StageAdmitted Efficient)
    (adversary : IvcAdversary (Output StepRelation.keyWidth StepRelation.keyPublicFits))
    (admitted : StageAdmitted (Stage.start (walkAdversary adversary)))
    (packageWidths : Fin 2 → Nat) (control : AppState → AppWitness → Fin 2) (cap depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    (adversary.tape.toOuterMeasure
        {tape | VerifierSecurity.FalseVerification ajtai (keys identity relations)
          (exports facts identity relations lightKey refreshKey) packageWidths control cap
          (adversary.output tape)}).toReal ≤
      walkBound ajtai (keys identity relations) (exports facts identity relations lightKey refreshKey)
        assumption closed adversary admitted depth :=
  VerifierSecurity.soundness ajtai (keys identity relations)
    (exports facts identity relations lightKey refreshKey) assumption closed adversary admitted
    (CircuitKeyDigest.keys_shaped identity relations) packageWidths control cap depth depthBound

section Final

open NightstreamFPrime.Spec.FieldExtensionRing

attribute [local instance low] Classical.propDecidable

attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK
  NightstreamFPrime.Export.BoundedDepth.FinalKey20.instDegree2K

/-- `Verifier::verify_final` (SPEC.md Algorithm 10) at the shipped packages of the reference
application, with the Akita openings of `crates/ns-compress/artifacts/key30.aks`: for each prover
with at most `queries` queries to the oracle, the probability of a false history is at most
`FinalKey30.shapesBound` plus `walkBound` (`FinalKey30.history_sound_drawn`). -/
theorem verify_final_sound {mode : Mode} (facts : Facts program ajtai) (identity : List F)
    (relations : Fin 2 → Relation) (lightKey : relations 0 = lightRelation facts)
    (refreshKey : relations 1 = refreshRelation facts)
    (setupRows : ℕ → Compression.Verifier.Factored F) (zeros minimum cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    {Wire Coord : Type} {total : ℕ} (read : Compression.Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First StepRelation.keyWidth StepRelation.keyPublicFits)
    [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] {ι : Type} [Fintype ι]
    (kind : Wire → Option ι)
    (family : ι → FinalKey30.Opening StepRelation.keyWidth StepRelation.keyPublicFits Wire Coord
      total)
    (shared : ∀ index, (family index).tower.Shared)
    (prover : RandomOracle.OracleComp (Compression.Akita.Extract.Address Wire total)
      (Option Coord → CodeSlot K) (ℕ → Wire))
    [Finite Wire] (budget : ℕ)
    {Admitted : NifsAdversary (WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits) → Prop}
    {StageAdmitted : Stage (Visit StepRelation.keyWidth StepRelation.keyPublicFits)
      (Answer StepRelation.keyWidth StepRelation.keyPublicFits) → Prop}
    {Efficient : (adversary : NifsAdversary (WalkAttempt StepRelation.keyWidth
        StepRelation.keyPublicFits)) →
      NifsExtractor (Answer StepRelation.keyWidth StepRelation.keyPublicFits) adversary → Prop}
    {error : NifsAdversary (WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits) → ℝ}
    (assumption : Assumption1 (WalkSucceeds ajtai) (WalkExtracted ajtai) Admitted Efficient error)
    (closed : Closed (attemptOf ajtai (keys identity relations)
        (exports facts identity relations lightKey refreshKey))
      (back ajtai (keys identity relations) (exports facts identity relations lightKey refreshKey))
      Admitted StageAdmitted Efficient)
    (rows : ∀ circuit : Fin 2, (setupRows circuit).matrices =
      (FinalRows.packageRows (exports facts identity relations lightKey refreshKey)
        (complete facts identity relations lightKey refreshKey) circuit).matrices)
    (coords : 40 ≤ Fintype.card Coord)
    (drawn : ∀ index, (family index).Drawn
      (setup mode ajtai identity relations setupRows zeros minimum)
      (complete facts identity relations lightKey refreshKey).packageWidth program.step cap control
      read head (shared index))
    {queries : ℕ} (bounded : prover.QueryBound queries) (positive : 0 < budget)
    (admitted : StageAdmitted (Stage.start (walkAdversary (FinalKey30.adversary
      (setup mode ajtai identity relations setupRows zeros minimum)
      (complete facts identity relations lightKey refreshKey).packageWidth program.step cap control
      read head kind family shared prover budget))))
    (depth : ℕ)
    (depthBound : ∀ table, FinalKey30.finalAccepts
        (setup mode ajtai identity relations setupRows zeros minimum)
        (complete facts identity relations lightKey refreshKey).packageWidth program.step cap
        control read head kind family shared (FinalKey20.played prover table) →
      (head (FinalKey20.played prover table).first).statement.iteration ≤ depth) :
    ((FinalKey30.adversary (setup mode ajtai identity relations setupRows zeros minimum)
        (complete facts identity relations lightKey refreshKey).packageWidth program.step cap
        control read head kind family shared prover budget).tape.toOuterMeasure {sample |
          FinalKey30.FalseHistory (setup mode ajtai identity relations setupRows zeros minimum)
            (complete facts identity relations lightKey refreshKey).packageWidth program.step cap
            control read head kind family shared prover program sample.1}).toReal ≤
      FinalKey30.shapesBound (setup mode ajtai identity relations setupRows zeros minimum)
          (complete facts identity relations lightKey refreshKey).packageWidth program.step cap
          control read head family shared prover queries budget +
        walkBound ajtai (keys identity relations)
          (exports facts identity relations lightKey refreshKey) assumption closed
          (FinalKey30.adversary (setup mode ajtai identity relations setupRows zeros minimum)
            (complete facts identity relations lightKey refreshKey).packageWidth program.step cap
            control read head kind family shared prover budget)
          admitted depth :=
  FinalKey30.history_sound_drawn (setup := setup mode ajtai identity relations setupRows zeros
      minimum)
    (exports := exports facts identity relations lightKey refreshKey) (assumption := assumption)
    (closed := closed) (complete := complete facts identity relations lightKey refreshKey)
    (rows := rows) (keysShaped := CircuitKeyDigest.keys_shaped identity relations) (blocks := rfl)
    (coords := coords) (drawn := drawn) (bounded := bounded) (positive := positive)
    (admitted := admitted) (depth := depth) (depthBound := depthBound)

end Final

end NightstreamFPrime.Export.BoundedDepth.Shipped
