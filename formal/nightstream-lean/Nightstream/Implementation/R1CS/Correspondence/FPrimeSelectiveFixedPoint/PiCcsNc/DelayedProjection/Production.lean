import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionBoundary
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker

/-!
Bounded delayed-`yZcol` composition over the generated running-`X`
public-prefix decoder.

Assurance tier: artifact-checked for the bounded public-prefix decoding,
model-level for the surrounding accepted protocol fixture.

Owns: specialization of the already-proved combined-NC projection and
adjacent-step production boundary to `RawRunningDecoder.ArtifactRefinement.decodedData`.
Consequently every delayed child scalar in this 270-coordinate fixture is read
from an exact generated `running[child].x` final assignment column, never from
a `CeClaim.y_zcol` sidecar or digest.

Does not own: the full packed `CcsWitness.Z`/`CeWitness.Z`, its private suffix,
sparse combined-NC verifier rows, source/input decoder rows,
state-binding rows, terminal opening rows, concrete Poseidon2 tag encodings,
Ajtai coordinate alignment or binding, Rust agreement for the delayed
terminal, costs, or row-removal permission. Those remaining physical
contracts are intentionally visible in the theorem premises inherited from
the independent production boundary.

Emits constraints: none; correspondence theorem only.

| Stable stage path | Obligation | Authority |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed_projection.public_prefix_fixture` | Compose generated running-`X` columns with bounded delayed projection semantics | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual

universe uState uEncoding uDigest

variable
  {shape : SemanticShape}
  {State : Type uState}
  {Encoding : Type uEncoding}
  {Digest : Type uDigest}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Execute the model checker over the exact generated 270-coordinate
public-prefix fixture. This is not a full production-witness checker. -/
def check
    (profile : Profile shape)
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (assignment : PhysicalAssignment)
    (certificate : FixedActive.Certificate context.materialize) : Bool :=
  CombinedNc.ProductionChecker.check context
    (ArtifactRefinement.decodedData profile context.materialize template
      assignment)
    certificate

/-- Concrete execution is exact to production acceptance instantiated with
the generated raw-child decoder. Thus later soundness theorems derive
`ProductionNifs.Accepted` from a Boolean check instead of assuming it. -/
theorem check_eq_true_iff_accepted
    (profile : Profile shape)
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (assignment : PhysicalAssignment)
    (certificate : FixedActive.Certificate context.materialize) :
    check profile context template assignment certificate = true <->
      CombinedNc.ProductionNifs.Accepted context.materialize
        (ArtifactRefinement.decodedData profile context.materialize template
          assignment)
        certificate := by
  exact CombinedNc.ProductionChecker.check_eq_true_iff_accepted context
    (ArtifactRefinement.decodedData profile context.materialize template
      assignment)
    certificate

/-- Every live semantic running coordinate used by combined NC is the exact
generated final assignment column selected by the semantic/product alignment. -/
theorem decodedRunning_live_eq_finalColumn
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (assignment : PhysicalAssignment)
    (running : Fin shape.runningCount)
    (lane : PackedLane)
    (block : LiveBlock) :
    (ArtifactRefinement.decodedData profile context template assignment
        ).runningAssignments running
        (profile.semanticColumn
          (logicalColumnAt { lane := lane, block := block })) =
      assignment
        (Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.allocationAt
          (ArtifactRefinement.childOfSemanticRunning context running)
          (logicalColumnAt { lane := lane, block := block })).finalColumn := by
  rfl

/-- Adjacent accepted production steps over the generated physical decoder
close the previous independent semantic fold, leave only the specifically
named `yRing` branch, or expose one structured bad event.

There is no `SourceProjectionMatches`, `UpstreamProducerColumnsBound`,
`BindingsHoldFor .yZcolOutput`, raw-child authority, or generic
`outputUnbound` premise. Raw children are definitionally the generated
physical assignment table. -/
theorem acceptedPair_implies_previousSemanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (profile : Profile shape)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousAssignment : PhysicalAssignment)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousInput : SemanticInput previousContext
      (ArtifactRefinement.decodedData profile previousContext
        previousTemplate previousAssignment))
    (previousChildren : ChildOpenings previousContext
      (ArtifactRefinement.decodedData profile previousContext
        previousTemplate previousAssignment)
      previousCertificate)
    (previousAccepted : CombinedNc.ProductionNifs.Accepted previousContext
      (ArtifactRefinement.decodedData profile previousContext
        previousTemplate previousAssignment)
      previousCertificate)
    (nextContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextAssignment : PhysicalAssignment)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextInput : SemanticInput nextContext
      (ArtifactRefinement.decodedData profile nextContext nextTemplate
        nextAssignment))
    (nextAccepted : CombinedNc.ProductionNifs.Accepted nextContext
      (ArtifactRefinement.decodedData profile nextContext nextTemplate
        nextAssignment)
      nextCertificate)
    (nextParent : Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (nextParentBound : nextContext.runningParent = some nextParent)
    (sameKey : nextContext.key = previousContext.key)
    (previousBinds : StateBinds scheme stateDigest
      (derive previousContext previousCertificate).piRlcOutput
      (outputChildren previousContext previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext
        previousCertificate)))
    (nextBinds : StateBinds scheme stateDigest nextParent
      nextContext.input.running nextContext.pending) :
    SemanticFold.Holds previousContext
        (ArtifactRefinement.decodedData profile previousContext
          previousTemplate previousAssignment)
        (derive previousContext previousCertificate).piRlcOutput
        (outputChildren previousContext previousCertificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext
        (ArtifactRefinement.decodedData profile previousContext
          previousTemplate previousAssignment)
        previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme previousContext
        (ArtifactRefinement.decodedData profile previousContext
          previousTemplate previousAssignment)
        previousCertificate nextContext
        (ArtifactRefinement.decodedData profile nextContext nextTemplate
          nextAssignment)
        nextCertificate := by
  exact
    CombinedNc.ProductionBoundary.acceptedPair_implies_previousSemanticFold_or_badEvent
      noZeroDivisors scheme stateDigest previousContext
      (ArtifactRefinement.decodedData profile previousContext
        previousTemplate previousAssignment)
      previousCertificate previousInput previousChildren previousAccepted
      nextContext
      (ArtifactRefinement.decodedData profile nextContext nextTemplate
        nextAssignment)
      nextCertificate nextInput nextAccepted nextParent nextParentBound
      sameKey previousBinds nextBinds

/-- Two adjacent successful executions of the concrete Boolean checker close
the previous semantic fold or expose only the named `yRing`, algebraic, and
binding events. `ProductionNifs.Accepted` is not a caller premise: it is
derived from each Boolean result by `check_eq_true_iff_accepted`. -/
theorem checkedPair_implies_previousSemanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (profile : Profile shape)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousAssignment : PhysicalAssignment)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousInput : SemanticInput previousContext.materialize
      (ArtifactRefinement.decodedData profile previousContext.materialize
        previousTemplate previousAssignment))
    (previousChildren : ChildOpenings previousContext.materialize
      (ArtifactRefinement.decodedData profile previousContext.materialize
        previousTemplate previousAssignment)
      previousCertificate)
    (previousChecked :
      check profile previousContext previousTemplate previousAssignment
        previousCertificate = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextAssignment : PhysicalAssignment)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextInput : SemanticInput nextContext.materialize
      (ArtifactRefinement.decodedData profile nextContext.materialize
        nextTemplate nextAssignment))
    (nextChecked :
      check profile nextContext nextTemplate nextAssignment nextCertificate =
        true)
    (nextParent : Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (nextParentBound : nextContext.materialize.runningParent = some nextParent)
    (sameKey : nextContext.materialize.key = previousContext.materialize.key)
    (previousBinds : StateBinds scheme stateDigest
      (derive previousContext.materialize previousCertificate).piRlcOutput
      (outputChildren previousContext.materialize previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext.materialize
        previousCertificate)))
    (nextBinds : StateBinds scheme stateDigest nextParent
      nextContext.materialize.input.running nextContext.materialize.pending) :
    SemanticFold.Holds previousContext.materialize
        (ArtifactRefinement.decodedData profile previousContext.materialize
          previousTemplate previousAssignment)
        (derive previousContext.materialize previousCertificate).piRlcOutput
        (outputChildren previousContext.materialize previousCertificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext.materialize
        (ArtifactRefinement.decodedData profile previousContext.materialize
          previousTemplate previousAssignment)
        previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme
        previousContext.materialize
        (ArtifactRefinement.decodedData profile previousContext.materialize
          previousTemplate previousAssignment)
        previousCertificate nextContext.materialize
        (ArtifactRefinement.decodedData profile nextContext.materialize
          nextTemplate nextAssignment)
        nextCertificate := by
  exact acceptedPair_implies_previousSemanticFold_or_badEvent
    noZeroDivisors profile scheme stateDigest previousContext.materialize
    previousTemplate previousAssignment previousCertificate previousInput
    previousChildren
    ((check_eq_true_iff_accepted profile previousContext previousTemplate
      previousAssignment previousCertificate).mp previousChecked)
    nextContext.materialize nextTemplate nextAssignment nextCertificate
    nextInput
    ((check_eq_true_iff_accepted profile nextContext nextTemplate
      nextAssignment nextCertificate).mp nextChecked)
    nextParent nextParentBound sameKey previousBinds nextBinds

/-- Variant of `checkedPair_implies_previousSemanticFold_or_badEvent` whose
accumulator continuity facts and canonical next-parent presence are derived
from executable checks. Only verifier-key continuity remains an explicit
cross-step setup invariant. -/
theorem checkedPair_of_stateChecks_implies_previousSemanticFold_or_badEvent
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (profile : Profile shape)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousAssignment : PhysicalAssignment)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousInput : SemanticInput previousContext.materialize
      (ArtifactRefinement.decodedData profile previousContext.materialize
        previousTemplate previousAssignment))
    (previousChildren : ChildOpenings previousContext.materialize
      (ArtifactRefinement.decodedData profile previousContext.materialize
        previousTemplate previousAssignment)
      previousCertificate)
    (previousChecked :
      check profile previousContext previousTemplate previousAssignment
        previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextAssignment : PhysicalAssignment)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextInput : SemanticInput nextContext.materialize
      (ArtifactRefinement.decodedData profile nextContext.materialize
        nextTemplate nextAssignment))
    (nextChecked :
      check profile nextContext nextTemplate nextAssignment nextCertificate =
        true)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true)
    (sameKey : nextContext.materialize.key = previousContext.materialize.key) :
    SemanticFold.Holds previousContext.materialize
        (ArtifactRefinement.decodedData profile previousContext.materialize
          previousTemplate previousAssignment)
        (derive previousContext.materialize previousCertificate).piRlcOutput
        (outputChildren previousContext.materialize previousCertificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext.materialize
        (ArtifactRefinement.decodedData profile previousContext.materialize
          previousTemplate previousAssignment)
        previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme
        previousContext.materialize
        (ArtifactRefinement.decodedData profile previousContext.materialize
          previousTemplate previousAssignment)
        previousCertificate nextContext.materialize
        (ArtifactRefinement.decodedData profile nextContext.materialize
          nextTemplate nextAssignment)
        nextCertificate := by
  let nextParent :=
    nextContext.input.parent.materialize nextContext.input.system
  have previousBinds : StateBinds scheme stateDigest
      (derive previousContext.materialize previousCertificate).piRlcOutput
      (outputChildren previousContext.materialize previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext.materialize
        previousCertificate)) :=
    (CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff scheme
      stateDigest
      (derive previousContext.materialize previousCertificate).piRlcOutput
      (outputChildren previousContext.materialize previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext.materialize
        previousCertificate))).mp previousStateChecked
  have nextBinds : StateBinds scheme stateDigest nextParent
      nextContext.materialize.input.running nextContext.materialize.pending :=
    (CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff scheme
      stateDigest nextParent nextContext.materialize.input.running
      nextContext.materialize.pending).mp nextStateChecked
  exact checkedPair_implies_previousSemanticFold_or_badEvent
    noZeroDivisors profile scheme stateDigest previousContext previousTemplate
    previousAssignment previousCertificate previousInput previousChildren
    previousChecked nextContext nextTemplate nextAssignment nextCertificate
    nextInput nextChecked nextParent rfl sameKey previousBinds nextBinds

/-- A successful final production check plus the executable raw-child terminal
check closes the last delayed output. Both semantic acceptance predicates are
derived from Boolean checks. -/
theorem checkedTerminal_implies_semanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (profile : Profile shape)
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (assignment : PhysicalAssignment)
    (certificate : FixedActive.Certificate context.materialize)
    (input : SemanticInput context.materialize
      (ArtifactRefinement.decodedData profile context.materialize template
        assignment))
    (children : ChildOpenings context.materialize
      (ArtifactRefinement.decodedData profile context.materialize template
        assignment)
      certificate)
    (checked : check profile context template assignment certificate = true)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (terminal : CombinedNc.ProductionTerminal.check context.materialize
      certificate rawChildren = true) :
    SemanticFold.Holds context.materialize
        (ArtifactRefinement.decodedData profile context.materialize template
          assignment)
        (derive context.materialize certificate).piRlcOutput
        (outputChildren context.materialize certificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context.materialize
        (ArtifactRefinement.decodedData profile context.materialize template
          assignment)
        certificate ∨
      CombinedNc.ProductionBoundary.TerminalBadEvent context.materialize
        (ArtifactRefinement.decodedData profile context.materialize template
          assignment)
        certificate := by
  exact CombinedNc.ProductionBoundary.acceptedTerminal_implies_previousSemanticFold_or_badEvent
    noZeroDivisors context.materialize
    (ArtifactRefinement.decodedData profile context.materialize template
      assignment)
    certificate input children
    ((check_eq_true_iff_accepted profile context template assignment
      certificate).mp checked)
    rawChildren terminal

/-- Exact external binding failures still outside the generated raw-running
decoder and canonical combined-NC checker. Each constructor names one physical
contract; none is a generic output mismatch or a packed-`yZcol` premise. -/
inductive ExternalBindingFailure
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (profile : Profile shape)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousAssignment : PhysicalAssignment)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextAssignment : PhysicalAssignment)
    (nextCertificate : FixedActive.Certificate nextContext.materialize) : Prop where
  | previousInput
      (failure : ¬ SemanticInput previousContext.materialize
        (ArtifactRefinement.decodedData profile previousContext.materialize
          previousTemplate previousAssignment)) :
      ExternalBindingFailure scheme stateDigest profile previousContext
        previousTemplate previousAssignment previousCertificate nextContext
        nextTemplate nextAssignment nextCertificate
  | previousChildren
      (failure : ¬ ChildOpenings previousContext.materialize
        (ArtifactRefinement.decodedData profile previousContext.materialize
          previousTemplate previousAssignment)
        previousCertificate) :
      ExternalBindingFailure scheme stateDigest profile previousContext
        previousTemplate previousAssignment previousCertificate nextContext
        nextTemplate nextAssignment nextCertificate
  | nextInput
      (failure : ¬ SemanticInput nextContext.materialize
        (ArtifactRefinement.decodedData profile nextContext.materialize
          nextTemplate nextAssignment)) :
      ExternalBindingFailure scheme stateDigest profile previousContext
        previousTemplate previousAssignment previousCertificate nextContext
        nextTemplate nextAssignment nextCertificate
  | previousState
      (failure : ¬ StateBinds scheme stateDigest
        (derive previousContext.materialize previousCertificate).piRlcOutput
        (outputChildren previousContext.materialize previousCertificate)
        (some (DelayedProduction.outgoingPending previousContext.materialize
          previousCertificate))) :
      ExternalBindingFailure scheme stateDigest profile previousContext
        previousTemplate previousAssignment previousCertificate nextContext
        nextTemplate nextAssignment nextCertificate
  | nextState
      (failure : ¬ StateBinds scheme stateDigest
        (nextContext.input.parent.materialize nextContext.input.system)
        nextContext.materialize.input.running nextContext.materialize.pending) :
      ExternalBindingFailure scheme stateDigest profile previousContext
        previousTemplate previousAssignment previousCertificate nextContext
        nextTemplate nextAssignment nextCertificate
  | verifierKeyContinuity
      (failure : nextContext.materialize.key ≠
        previousContext.materialize.key) :
      ExternalBindingFailure scheme stateDigest profile previousContext
        previousTemplate previousAssignment previousCertificate nextContext
        nextTemplate nextAssignment nextCertificate

/-- Two adjacent concrete Boolean acceptances yield the previous independent
semantic fold, the specifically named `yRing` failure, a delayed algebraic or
cryptographic bad event, or one exact external binding failure above.

All former semantic premises are decided inside the proof partition. In
particular there is no `SourceProjectionMatches`, raw-child authority,
`BindingsHoldFor .yZcolOutput`, or generic `outputUnbound` branch. -/
theorem checkedPair_implies_previousSemanticFold_or_namedFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (profile : Profile shape)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousAssignment : PhysicalAssignment)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousChecked :
      check profile previousContext previousTemplate previousAssignment
        previousCertificate = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextAssignment : PhysicalAssignment)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextChecked :
      check profile nextContext nextTemplate nextAssignment nextCertificate =
        true) :
    SemanticFold.Holds previousContext.materialize
        (ArtifactRefinement.decodedData profile previousContext.materialize
          previousTemplate previousAssignment)
        (derive previousContext.materialize previousCertificate).piRlcOutput
        (outputChildren previousContext.materialize previousCertificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext.materialize
        (ArtifactRefinement.decodedData profile previousContext.materialize
          previousTemplate previousAssignment)
        previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme
        previousContext.materialize
        (ArtifactRefinement.decodedData profile previousContext.materialize
          previousTemplate previousAssignment)
        previousCertificate nextContext.materialize
        (ArtifactRefinement.decodedData profile nextContext.materialize
          nextTemplate nextAssignment)
        nextCertificate ∨
      ExternalBindingFailure scheme stateDigest profile previousContext
        previousTemplate previousAssignment previousCertificate nextContext
        nextTemplate nextAssignment nextCertificate := by
  classical
  let previousData := ArtifactRefinement.decodedData profile
    previousContext.materialize previousTemplate previousAssignment
  let nextData := ArtifactRefinement.decodedData profile nextContext.materialize
    nextTemplate nextAssignment
  let nextParent :=
    nextContext.input.parent.materialize nextContext.input.system
  by_cases previousInput : SemanticInput previousContext.materialize previousData
  · by_cases previousChildren : ChildOpenings previousContext.materialize
        previousData previousCertificate
    · by_cases nextInput : SemanticInput nextContext.materialize nextData
      · by_cases previousBinds : StateBinds scheme stateDigest
            (derive previousContext.materialize previousCertificate).piRlcOutput
            (outputChildren previousContext.materialize previousCertificate)
            (some (DelayedProduction.outgoingPending
              previousContext.materialize previousCertificate))
        · by_cases nextBinds : StateBinds scheme stateDigest nextParent
              nextContext.materialize.input.running
              nextContext.materialize.pending
          · by_cases sameKey : nextContext.materialize.key =
                previousContext.materialize.key
            · rcases checkedPair_implies_previousSemanticFold_or_badEvent
                  noZeroDivisors profile scheme stateDigest previousContext
                  previousTemplate previousAssignment previousCertificate
                  (by simpa [previousData] using previousInput)
                  (by simpa [previousData] using previousChildren)
                  previousChecked nextContext nextTemplate nextAssignment
                  nextCertificate (by simpa [nextData] using nextInput)
                  nextChecked nextParent rfl sameKey previousBinds nextBinds with
                semantic | yRing | bad
              · exact Or.inl semantic
              · exact Or.inr (Or.inl yRing)
              · exact Or.inr (Or.inr (Or.inl bad))
            · exact Or.inr (Or.inr (Or.inr
                (.verifierKeyContinuity sameKey)))
          · exact Or.inr (Or.inr (Or.inr (.nextState (by
              simpa [nextParent] using nextBinds))))
        · exact Or.inr (Or.inr (Or.inr (.previousState previousBinds)))
      · exact Or.inr (Or.inr (Or.inr (.nextInput (by
          simpa [nextData] using nextInput))))
    · exact Or.inr (Or.inr (Or.inr (.previousChildren (by
        simpa [previousData] using previousChildren))))
  · exact Or.inr (Or.inr (Or.inr (.previousInput (by
      simpa [previousData] using previousInput))))

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production
