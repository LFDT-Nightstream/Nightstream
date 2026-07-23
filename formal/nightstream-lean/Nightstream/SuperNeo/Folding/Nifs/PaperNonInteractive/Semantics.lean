import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Verifier

/-!
Independent transition and named-event boundary for paper SuperNeo NIFS.

Owns: source and child opening witnesses, the exact semantic realization of
one transcript-reachable fold, five closed failure classes, deterministic
soundness of the executable verifier modulo those classes, and completeness
for every such realization.

Does not own: the probability bound for Fiat--Shamir coordinate-fork
extraction, concrete transcript or commitment security, Rust, R1CS, artifacts,
minimality, or costs.

Emits constraints: no.

The only extraction boundary is specific: failure to obtain the complete
corrected-ambient `K+k` opening required by the paper `Pi_RLC` coordinate
fork.  It is not `not Transition`, `outputUnbound`, or a generic refinement
failure.

| Semantic phase | Mathematical obligation or event | Lean owner |
|---|---|---|
| source extraction | every fresh/running source satisfies the paper relation | `SourceValid` |
| ambient extraction | recover the corrected `K+k` opening at the verifier-derived point | `AmbientTargetOpenings` |
| accepted transition | state the fixed-width SumCheck chain, five `Pi_DEC` equations, source/child membership, and parent assignment directly | `Realization`, `Transition` |
| `Pi_CCS` security | isolate the exact mixing root or SumCheck bad challenge | `PiCcsMixingRoot`, `PiCcsSumCheckCollision` |
| `Pi_RLC` security | isolate failure of complete coordinate-fork extraction | `PiRlcCoordinateForkExtractionFailure` |
| `Pi_DEC` security | isolate child extraction or parent-opening binding failure | `PiDecChildExtractionFailure`, `BadEvent.parentBindingCollision` |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar uState

/-- Re-index an extracted `Pi_CCS` target witness into the exact `K+k`
product consumed by `Pi_RLC`. -/
def sourceAssignments
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (witness : OutputWitness shape columns) :
    Fin key.arity.total -> Assignment F columns :=
  fun source => witness.assignments (Fin.cast key.total_eq_sourceCount source)

/-- Exact source membership extracted by paper `Pi_CCS`. -/
def SourceValid
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns) : Prop :=
  SourceHolds key.extensionOps key.lift key.openingMaps key.params
    (key.statement running fresh) witness

/-- Complete corrected-ambient opening obtained by the `Pi_RLC` coordinate
fork. -/
def AmbientTargetOpenings
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (witness : OutputWitness shape columns) : Prop :=
  AmbientOutputHolds key.extensionOps key.lift key.openingMaps key.params
    (key.statement running fresh) (key.piCcsProbe running fresh proof) witness

/-- Every exact ambient opening plus extracted source truth upgrades the
actual `Pi_CCS` output back to the fresh `CE(b)` relation expected by
`Pi_RLC`. -/
theorem piCcsOutputs_hold
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (witness : OutputWitness shape columns)
    (source : SourceValid key running fresh witness)
    (ambient : AmbientTargetOpenings key running fresh proof witness) :
    forall index,
      CE.Holds key.semantics key.params
        (key.piCcsOutputs running fresh proof index)
        (sourceAssignments key witness index) := by
  intro index
  let sourceIndex : Fin shape.sourceCount :=
    Fin.cast key.total_eq_sourceCount index
  have opening := source.1 sourceIndex
  have ambientAt := ambient sourceIndex
  refine ⟨?_, ambientAt.2.1, ambientAt.2.2⟩
  simpa [Key.piCcsOutputs, Key.piCcsProbe, Key.statement,
    sourceAssignments, sourceIndex, Key.semantics, NormStage.bound] using opening

/-- A complete semantic realization of one transcript-reachable paper fold.
The finite SumCheck chain and the five `Pi_DEC` equations are stated directly;
this structure does not contain either executable checker result or either
protocol's bundled acceptance predicate.  Source and child membership are
independent relation facts. -/
structure Realization
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (result : Running Extension Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (sourceWitness : OutputWitness shape columns)
    (childAssignments : Fin key.params.k -> Assignment F columns) : Prop where
  piCcsRoundChain :
    SumCheck.Finite.FixedPhase.Chain key.extensionOps.toOps
      (((key.statement running fresh).verifierInput key.lift).initial
        key.extensionOps
        (key.piCcsExecution running fresh proof).coins.gamma)
      (key.piCcsFixedCertificate running fresh proof).rounds
      (key.piCcsExecution running fresh proof).coins.roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage key.extensionOps
        ((key.statement running fresh).verifierInput key.lift)
        (key.piCcsExecution running fresh proof).coins.alpha
        (key.piCcsExecution running fresh proof).coins.gamma
        (key.piCcsExecution running fresh proof).coins.roundPoint
        (key.piCcsCertificate running fresh proof).output)
  piDecParentCombined :
    (key.piDecAttempt running fresh proof).parent.stage = .combined
  piDecParentEvaluationSize :
    (key.piDecAttempt running fresh proof).parent.evaluations.size =
      key.piDecEvaluationArity.count
        (key.piDecAttempt running fresh proof).parent.constraintSystem
  piDecMessageEvaluationSize : forall child,
    ((key.piDecAttempt running fresh proof).messages child).evaluations.size =
      key.piDecEvaluationArity.count
        (key.piDecAttempt running fresh proof).parent.constraintSystem
  piDecCommitmentEquation :
    (key.piDecAttempt running fresh proof).parent.commitment =
      key.piDecAlgebra.recomposeCommitment fun child =>
        ((key.piDecAttempt running fresh proof).messages child).commitment
  piDecEvaluationEquation :
    (key.piDecAttempt running fresh proof).parent.evaluations =
      key.piDecAlgebra.recomposeEvaluations fun child =>
        ((key.piDecAttempt running fresh proof).messages child).evaluations
  sourceValid : SourceValid key running fresh sourceWitness
  piCcsInputsValid : forall index,
    CE.Holds key.semantics key.params
      (key.piCcsOutputs running fresh proof index)
      (sourceAssignments key sourceWitness index)
  childValid : forall child,
    CE.Holds key.semantics key.params
      (PiDEC.PaperVerifier.children key.piDecPublicInputSplit
        (key.piDecAttempt running fresh proof) child)
      (childAssignments child)
  parentAssignment :
    PiRLC.combinedWitness key.piRlcAlgebra
        (key.piRlcChallenges running fresh proof)
        (sourceAssignments key sourceWitness) =
      key.piDecAlgebra.recomposeAssignment childAssignments
  resultComputed : result = key.output running fresh proof

/-- Independently stated public NIFS transition.  It exposes relation
membership and exact algebraic ownership while hiding all witnesses. -/
def Transition
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (result : Running Extension Commitment PublicInput shape) : Prop :=
  exists proof : Proof Extension Commitment shape degreeBound,
  exists sourceWitness : OutputWitness shape columns,
  exists childAssignments : Fin key.params.k -> Assignment F columns,
    Realization key running fresh result proof sourceWitness childAssignments

/-- The exact extraction failure left by the paper `Pi_RLC` coordinate fork:
no complete corrected-ambient opening exists for the verifier's actual
`K+k` output batch. -/
def PiRlcCoordinateForkExtractionFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) : Prop :=
  ¬ (exists witness : OutputWitness shape columns,
    AmbientTargetOpenings key running fresh proof witness)

/-- Exact child-opening extraction failure at the `Pi_DEC` output. -/
def PiDecChildExtractionFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) : Prop :=
  ¬ (exists childAssignments : Fin key.params.k -> Assignment F columns,
    forall child,
      CE.Holds key.semantics key.params
        (PiDEC.PaperVerifier.children key.piDecPublicInputSplit
          (key.piDecAttempt running fresh proof) child)
        (childAssignments child))

/-- The precise alpha/gamma mixing-polynomial root for the source witness and
the verifier-derived `Pi_CCS` coins. -/
def PiCcsMixingRoot
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (witness : OutputWitness shape columns) : Prop :=
  SignedCoefficientObject.MixingRoot key.extensionOps
    (((key.statement running fresh).sourceProtocolData key.lift witness).toJointData
      key.extensionOps)
    (key.piCcsProbe running fresh proof).coins.alpha
    (key.piCcsProbe running fresh proof).coins.gamma

/-- The precise fixed-width SumCheck bad-challenge event for the source
witness, verifier-derived round challenges, and submitted polynomials.  Both
the claimed and independently expected round functions carry explicit
degree-`degreeBound` polynomial witnesses. -/
def PiCcsSumCheckCollision
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (witness : OutputWitness shape columns) : Prop :=
  exists round,
    SumCheck.Finite.FixedPhase.BadChallenge key.extensionOps.toOps
      (ProtocolPolynomial.polynomial key.extensionOps
        ((key.statement running fresh).sourceProtocolData key.lift witness)
        (key.piCcsExecution running fresh proof).coins.alpha
        (key.piCcsExecution running fresh proof).coins.gamma)
      degreeBound key.challengeSetSize
      (((key.statement running fresh).verifierInput key.lift).initial
        key.extensionOps
        (key.piCcsExecution running fresh proof).coins.gamma)
      (key.piCcsExecution running fresh proof).coins.roundPoint.coordinates
      (key.piCcsFixedCertificate running fresh proof)
      round

/-- Closed finite family of genuine algebraic/security failures. -/
inductive BadEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (_result : Running Extension Commitment PublicInput shape) : Prop where
  | piRlcCoordinateForkExtraction
      (failure : PiRlcCoordinateForkExtractionFailure key running fresh proof)
  | piDecChildExtraction
      (failure : PiDecChildExtractionFailure key running fresh proof)
  | piCcsMixingRoot
      (sourceWitness : OutputWitness shape columns)
      (root : PiCcsMixingRoot key running fresh proof sourceWitness)
  | piCcsSumCheckCollision
      (sourceWitness : OutputWitness shape columns)
      (collision : PiCcsSumCheckCollision key running fresh proof sourceWitness)
  | parentBindingCollision
      (collision : Nonempty (PiDEC.ParentOpeningBindingCollision
        key.semantics key.params
        (key.parent running fresh proof).commitment))

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
