import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionContext
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionState
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal

/-!
Base, recursive, and terminal boundaries for production delayed `yZcol`.

Assurance tier: model-level, pending generated-row and Rust refinement.

Owns: the explicit no-pending base specialization; the adjacent accepted-step
soundness theorem with state-derived continuity; the terminal exact-opening
closure; and final partitions in which `yRing` remains separate while no
generic output-unbound event can conceal a packed-`yZcol` failure.

Does not own: concrete assignment decoding, state-digest or terminal rows,
Ajtai/Poseidon2 internals, Rust conformance, probability bounds, costs, or row
removal.

Emits constraints: none.

Authority boundary: the recursive theorem derives child/pending continuity
from complete payload recomputation. The terminal theorem derives final child
authority from genuine openings. Both still expose the exact concrete input,
opening, state-row, and terminal-row contracts which later artifact theorems
must discharge.

| Boundary | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| base | no old pending value; NC is exactly the ordinary raw-source polynomial | computed | `baseNcAccepted_iff` |
| recursive | next accepted raw NC closes the previous packed output | derived/security partition | `acceptedPair_implies_previousSemanticFold_or_badEvent` |
| terminal | final raw child openings close the last packed output exactly | derived/security partition | `acceptedTerminal_implies_previousSemanticFold_or_badEvent` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionBoundary

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending

private abbrev ops := ConcreteCarrier.extensionOps

universe uState uEncoding uDigest

variable
  {shape : SemanticShape}
  {State : Type uState}
  {Encoding : Type uEncoding}
  {Digest : Type uDigest}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- At the base boundary there is no previous packed output. Consequently the
raw NC verifier is definitionally the ordinary source polynomial with zero
initial claim. The current output is still emitted as pending for its
successor or the terminal boundary. -/
theorem baseNcAccepted_iff
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (base : context.pending = none) :
    CombinedNc.ProductionPiCcs.NcAccepted context data certificate ↔
      FixedPhase.Accepted ops.toOps
        (InitialSum.sumcheckPolynomial context.covers data context.ncCoins)
        InitialSum.claimedInitial
        (CombinedNc.ProductionPiCcs.ncPoint context certificate).coordinates
        certificate.piCcs.nc.toSumCheck := by
  simp [CombinedNc.ProductionPiCcs.NcAccepted,
    CombinedNc.ProductionPiCcs.rawPolynomial,
    CombinedNc.ProductionPiCcs.rawInitial, base]

/-- Named recursive-boundary failures. `delayedProjection` expands to the
selector, residual-weight, producer-root, SumCheck, mixing, and child-opening
events; it is not a generic output mismatch. -/
inductive RecursiveBadEvent
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext) : Prop where
  | currentPiCcs
      (bad : CombinedNc.ProductionPiCcs.BadEvent previousContext previousData
        previousCertificate) :
      RecursiveBadEvent scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | delayedProjection
      (bad : CombinedNc.ProductionSequence.PreviousClosureBadEvent
        previousContext previousData previousCertificate nextContext nextData
        nextCertificate nextContext.challengeSetSize) :
      RecursiveBadEvent scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | accumulatorBinding
      (bad : Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure
        scheme) :
      RecursiveBadEvent scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate

/-- Failures of the y_zcol-only backward edge.  Unlike `RecursiveBadEvent`,
this partition does not contain a previous child-opening obligation: the
successor's raw assignment table is the only child-value source.  The private
parent mismatch remains separate until concrete opening/state refinement
reduces it to a commitment/binding failure. -/
inductive RawParentRecursiveBadEvent
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext) : Prop where
  | privateParentState
      (failure : CombinedNc.ProductionSequence.RawParentStateMismatch
        previousContext previousData previousCertificate nextContext nextData) :
      RawParentRecursiveBadEvent scheme previousContext previousData
        previousCertificate nextContext nextData nextCertificate
  | delayedProjection
      (bad : CombinedNc.ProductionSequence.RawParentStateBadEvent
        previousContext previousData previousCertificate nextContext nextData
        nextCertificate nextContext.challengeSetSize) :
      RawParentRecursiveBadEvent scheme previousContext previousData
        previousCertificate nextContext nextData nextCertificate
  | accumulatorBinding
      (bad : Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure
        scheme) :
      RawParentRecursiveBadEvent scheme previousContext previousData
        previousCertificate nextContext nextData nextCertificate

/-- Exact recursive failures once the previous canonical parent opening and
the successor decoded-input binding are available.  The former private-state
mismatch has become a standard parent-opening collision inside
`ParentOpeningClosureBadEvent`. -/
inductive ParentOpeningRecursiveBadEvent
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext) : Prop where
  | delayedProjection
      (bad : CombinedNc.ProductionSequence.ParentOpeningClosureBadEvent
        previousContext previousData previousCertificate nextContext nextData
        nextCertificate nextContext.challengeSetSize) :
      ParentOpeningRecursiveBadEvent scheme previousContext previousData
        previousCertificate nextContext nextData nextCertificate
  | accumulatorBinding
      (bad : Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure
        scheme) :
      ParentOpeningRecursiveBadEvent scheme previousContext previousData
        previousCertificate nextContext nextData nextCertificate

/-- Backward claims edge for packed `yZcol` alone.  The successor packed
equation first turns its public message acceptance into the raw combined-NC
predicate over `nextData`.  Exact state recomputation then binds the carried
pending value, and the successor raw table closes the predecessor output or
exposes one precisely owned failure.

The predecessor message contributes only its accepted tail for state
recomputation; its own packed/raw acceptance is not assumed. -/
theorem messageAcceptedPair_of_nextPacked_implies_previousPacked_or_rawParentBadEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousAccepted : CombinedNc.ProductionNifs.MessageAccepted
      previousContext previousCertificate)
    (nextContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextAccepted : CombinedNc.ProductionNifs.MessageAccepted nextContext
      nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock nextContext.covers nextData
      (CombinedNc.ProductionPiCcs.ncPoint nextContext nextCertificate).block
      nextCertificate.piCcs.output)
    (nextParent : Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (nextParentBound : nextContext.runningParent = some nextParent)
    (previousBinds : StateBinds scheme stateDigest
      (derive previousContext previousCertificate).piRlcOutput
      (outputChildren previousContext previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext
        previousCertificate)))
    (nextBinds : StateBinds scheme stateDigest nextParent
      nextContext.input.running nextContext.pending) :
    Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      RawParentRecursiveBadEvent scheme previousContext previousData
        previousCertificate nextContext nextData nextCertificate := by
  have nextRaw : CombinedNc.ProductionNifs.Accepted nextContext nextData
      nextCertificate :=
    CombinedNc.ProductionNifs.accepted_of_messageAccepted_and_packed
      nextContext nextData nextCertificate nextAccepted nextPacked
  rcases
      CombinedNc.ProductionState.acceptedNext_of_stateBinding_implies_previousPackedYZcolBound_or_rawParentStateMismatch_or_badEvent
        noZeroDivisors scheme stateDigest previousContext previousData
        previousCertificate previousAccepted.tail nextContext nextData
        nextCertificate nextRaw nextParent nextParentBound previousBinds
        nextBinds with packed | stateMismatch | delayedBad | bindingFailure
  · exact Or.inl packed
  · exact Or.inr (.privateParentState stateMismatch)
  · exact Or.inr (.delayedProjection delayedBad)
  · exact Or.inr (.accumulatorBinding bindingFailure)

/-- Backward claims edge with concrete parent-opening authority.  Successor
packed authority extracts its claims check to the raw combined-NC predicate;
state recomputation supplies exact child/pending continuity; and the decoded
successor table supplies raw child commitment alignment while combined-NC
supplies the norms.

The predecessor contributes only its accepted tail and its independently
checked canonical parent commitment and norm, so the proof is not circular in
the packed equation being derived. -/
theorem messageAcceptedPair_of_nextPacked_of_parentOpening_implies_previousPacked_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousParentBound : DelayedRawChildren.CanonicalParentBinding
      previousContext previousData previousCertificate)
    (previousAccepted : CombinedNc.ProductionNifs.MessageAccepted
      previousContext previousCertificate)
    (nextContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextCommitments : DelayedRawChildren.RawRunningCommitmentsBound
      nextContext nextData)
    (nextAccepted : CombinedNc.ProductionNifs.MessageAccepted nextContext
      nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock nextContext.covers nextData
      (CombinedNc.ProductionPiCcs.ncPoint nextContext nextCertificate).block
      nextCertificate.piCcs.output)
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
    Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      ParentOpeningRecursiveBadEvent scheme previousContext previousData
        previousCertificate nextContext nextData nextCertificate := by
  have nextRaw : CombinedNc.ProductionNifs.Accepted nextContext nextData
      nextCertificate :=
    CombinedNc.ProductionNifs.accepted_of_messageAccepted_and_packed
      nextContext nextData nextCertificate nextAccepted nextPacked
  rcases
      CombinedNc.ProductionState.acceptedNext_of_stateBinding_of_parentOpening_implies_previousPackedYZcolBound_or_badEvent
        noZeroDivisors scheme stateDigest previousContext previousData
        previousCertificate previousParentBound previousAccepted.tail
        nextContext nextData nextCertificate nextCommitments nextRaw nextParent
        nextParentBound sameKey previousBinds nextBinds with
    packed | delayedBad | bindingFailure
  · exact Or.inl packed
  · exact Or.inr (.delayedProjection delayedBad)
  · exact Or.inr (.accumulatorBinding bindingFailure)

/-- Two adjacent physically accepted production steps yield the previous
independent semantic fold, the specifically named previous `yRing` failure,
or a structured bad event whose delayed branch contains every packed-`yZcol`
failure explicitly. -/
theorem acceptedPair_implies_previousSemanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousInput : SemanticInput previousContext previousData)
    (previousChildren : ChildOpenings previousContext previousData
      previousCertificate)
    (previousAccepted : CombinedNc.ProductionNifs.Accepted previousContext
      previousData previousCertificate)
    (nextContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextInput : SemanticInput nextContext nextData)
    (nextAccepted : CombinedNc.ProductionNifs.Accepted nextContext nextData
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
    SemanticFold.Holds previousContext previousData
        (derive previousContext previousCertificate).piRlcOutput
        (outputChildren previousContext previousCertificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext previousData
        previousCertificate ∨
      RecursiveBadEvent scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate := by
  rcases
      CombinedNc.ProductionNifs.accepted_implies_delayedRefinement_or_yRingUnbound_or_badEvent
        noZeroDivisors previousContext previousData previousCertificate
        previousInput previousChildren previousAccepted with
    refinement | yRingUnbound | piCcsBad
  · rcases
        CombinedNc.ProductionState.acceptedNext_of_stateBinding_implies_previousSemanticFold_or_badEvent
          noZeroDivisors scheme stateDigest previousContext previousData
          previousCertificate previousAccepted refinement nextContext nextData
          nextCertificate nextInput nextAccepted nextParent nextParentBound
          sameKey previousBinds nextBinds with semantic | delayedBad |
            bindingFailure
    · exact Or.inl semantic
    · exact Or.inr (Or.inr (.delayedProjection delayedBad))
    · exact Or.inr (Or.inr (.accumulatorBinding bindingFailure))
  · exact Or.inr (Or.inl yRingUnbound)
  · exact Or.inr (Or.inr (.currentPiCcs piCcsBad))

/-- Strong backward claims-level recursive step. Besides the previous
semantic fold, the successful branch retains the packed equation derived from
the accepted successor. That equation is the induction value consumed by the
preceding edge of a finite delayed trace. -/
theorem messageAcceptedPair_of_nextPacked_implies_previousPackedAndSemanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousInput : SemanticInput previousContext previousData)
    (previousChildren : ChildOpenings previousContext previousData
      previousCertificate)
    (previousAccepted : CombinedNc.ProductionNifs.MessageAccepted
      previousContext previousCertificate)
    (nextContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextInput : SemanticInput nextContext nextData)
    (nextAccepted : CombinedNc.ProductionNifs.MessageAccepted nextContext
      nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock nextContext.covers nextData
      (CombinedNc.ProductionPiCcs.ncPoint nextContext nextCertificate).block
      nextCertificate.piCcs.output)
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
    (Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
          (DelayedProduction.outgoingPending previousContext
            previousCertificate).oldBlock
          previousCertificate.piCcs.output ∧
        SemanticFold.Holds previousContext previousData
          (derive previousContext previousCertificate).piRlcOutput
          (outputChildren previousContext previousCertificate)) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext previousData
        previousCertificate ∨
      RecursiveBadEvent scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate := by
  have nextRaw : CombinedNc.ProductionNifs.Accepted nextContext nextData
      nextCertificate :=
    CombinedNc.ProductionNifs.accepted_of_messageAccepted_and_packed
      nextContext nextData nextCertificate nextAccepted nextPacked
  rcases
      CombinedNc.ProductionState.acceptedNext_of_stateBinding_implies_previousPackedYZcolBound_or_badEvent
        noZeroDivisors scheme stateDigest previousContext previousData
        previousCertificate previousChildren previousAccepted.tail nextContext
        nextData nextCertificate nextInput nextRaw nextParent nextParentBound
        sameKey previousBinds nextBinds with
    previousPacked | delayedBad | bindingFailure
  · have previousRaw : CombinedNc.ProductionNifs.Accepted previousContext
        previousData previousCertificate :=
      CombinedNc.ProductionNifs.accepted_of_messageAccepted_and_packed
        previousContext previousData previousCertificate previousAccepted (by
          simpa [CombinedNc.ProductionPiCcs.ncPoint] using previousPacked)
    rcases
        CombinedNc.ProductionNifs.accepted_implies_delayedRefinement_or_yRingUnbound_or_badEvent
          noZeroDivisors previousContext previousData previousCertificate
          previousInput previousChildren previousRaw with
      refinement | yRingUnbound | piCcsBad
    · exact Or.inl ⟨previousPacked,
        refinement.toSemanticFold (by simpa using previousPacked)⟩
    · exact Or.inr (Or.inl yRingUnbound)
    · exact Or.inr (Or.inr (.currentPiCcs piCcsBad))
  · exact Or.inr (Or.inr (.delayedProjection delayedBad))
  · exact Or.inr (Or.inr (.accumulatorBinding bindingFailure))

/-- Backward claims-level recursive step. A packed opening already derived
for the successor first refines the successor public-message execution. Its
raw combined-NC acceptance and recomputed state then derive the previous
packed opening, which in turn refines the previous public-message execution.
No output-binding negation is returned at either extraction seam. -/
theorem messageAcceptedPair_of_nextPacked_implies_previousSemanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousInput : SemanticInput previousContext previousData)
    (previousChildren : ChildOpenings previousContext previousData
      previousCertificate)
    (previousAccepted : CombinedNc.ProductionNifs.MessageAccepted
      previousContext previousCertificate)
    (nextContext : FixedActive.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextInput : SemanticInput nextContext nextData)
    (nextAccepted : CombinedNc.ProductionNifs.MessageAccepted nextContext
      nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock nextContext.covers nextData
      (CombinedNc.ProductionPiCcs.ncPoint nextContext nextCertificate).block
      nextCertificate.piCcs.output)
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
    SemanticFold.Holds previousContext previousData
        (derive previousContext previousCertificate).piRlcOutput
        (outputChildren previousContext previousCertificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext previousData
        previousCertificate ∨
      RecursiveBadEvent scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate := by
  rcases
      messageAcceptedPair_of_nextPacked_implies_previousPackedAndSemanticFold_or_badEvent
        noZeroDivisors scheme stateDigest previousContext previousData
        previousCertificate previousInput previousChildren previousAccepted
        nextContext nextData nextCertificate nextInput nextAccepted nextPacked
        nextParent nextParentBound sameKey previousBinds nextBinds with
    success | yRing | bad
  · exact Or.inl success.2
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr bad)

/-- Packed-y-zcol-only terminal failures under the direct-parent route. The
canonical parent commitment/norm is checked independently; terminal raw
children are checked by `ProductionTerminal`; and any remaining ambiguity is
an explicit mixing or parent-opening binding event. -/
inductive ParentOpeningTerminalBadEvent
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop where
  | canonicalParentBinding
      (failure : ¬ DelayedRawChildren.CanonicalParentBinding context data
        certificate) :
      ParentOpeningTerminalBadEvent context data certificate
  | mixing
      (bad : PiRlcSidecar.MixingCollision context.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (DelayedProduction.outgoingPending context certificate).oldBlock
        (PackedYZcol.sourceClaims context certificate)) :
      ParentOpeningTerminalBadEvent context data certificate
  | parentBinding
      (bad : Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics context.key) productionGlobalParams
        (derive context certificate).piRlcOutput.commitment)) :
      ParentOpeningTerminalBadEvent context data certificate

/-- Terminal-boundary failures remain precise: the current raw `Pi_CCS`
event, `Pi_RLC` source mixing, or an indexed final-child commitment collision. -/
inductive TerminalBadEvent
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop where
  | currentPiCcs
      (bad : CombinedNc.ProductionPiCcs.BadEvent context data certificate) :
      TerminalBadEvent context data certificate
  | mixing
      (bad : PiRlcSidecar.MixingCollision context.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (DelayedProduction.outgoingPending context certificate).oldBlock
        (PackedYZcol.sourceClaims context certificate)) :
      TerminalBadEvent context data certificate
  | childBinding
      (bad : ∃ child, Nonempty
        (Opening.BindingCollision (semantics context.key)
          productionGlobalParams.b
          (outputChildren context certificate child).commitment)) :
      TerminalBadEvent context data certificate

/-- The executable terminal decider closes the last delayed output. No
next-step context is invented, and no generic output-unbound branch remains. -/
theorem acceptedTerminal_implies_previousSemanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (input : SemanticInput context data)
    (children : ChildOpenings context data certificate)
    (accepted : CombinedNc.ProductionNifs.Accepted context data certificate)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (terminal : CombinedNc.ProductionTerminal.check context certificate
      rawChildren = true) :
    SemanticFold.Holds context data
        (derive context certificate).piRlcOutput
        (outputChildren context certificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context data certificate ∨
      TerminalBadEvent context data certificate := by
  rcases
      CombinedNc.ProductionNifs.accepted_implies_delayedRefinement_or_yRingUnbound_or_badEvent
        noZeroDivisors context data certificate input children accepted with
    refinement | yRingUnbound | piCcsBad
  · rcases
        CombinedNc.ProductionTerminal.accepted_implies_previousSemanticFold_or_badEvent
          context data certificate refinement rawChildren
          (CombinedNc.ProductionTerminal.accepted_of_check context certificate
            rawChildren terminal) with
      semantic | mixing | binding
    · exact Or.inl semantic
    · exact Or.inr (Or.inr (.mixing mixing))
    · exact Or.inr (Or.inr (.childBinding binding))
  · exact Or.inr (Or.inl yRingUnbound)
  · exact Or.inr (Or.inr (.currentPiCcs piCcsBad))

/-- Claims-level terminal closure is anchored before weak-output extraction.
The complete raw child openings first derive the packed current output (or a
named mixing/binding event); only that positive equation is then used to
refine the public-message NIFS acceptance.  Consequently no
`OutputBindingFailure` branch survives this boundary. -/
theorem messageAcceptedTerminal_implies_previousSemanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (input : SemanticInput context data)
    (children : ChildOpenings context data certificate)
    (accepted : CombinedNc.ProductionNifs.MessageAccepted context certificate)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (terminal : CombinedNc.ProductionTerminal.check context certificate
      rawChildren = true) :
    SemanticFold.Holds context data
        (derive context certificate).piRlcOutput
        (outputChildren context certificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context data certificate ∨
      TerminalBadEvent context data certificate := by
  have terminalAccepted :=
    CombinedNc.ProductionTerminal.accepted_of_check context certificate
      rawChildren terminal
  rcases
      CombinedNc.ProductionTerminal.accepted_implies_packedYZcolBound_or_badEvent
        context data certificate children rawChildren terminalAccepted with
    packed | mixing | binding
  · have rawAccepted : CombinedNc.ProductionNifs.Accepted context data
        certificate :=
      CombinedNc.ProductionNifs.accepted_of_messageAccepted_and_packed context
        data certificate accepted (by
          simpa [CombinedNc.ProductionPiCcs.ncPoint] using packed)
    rcases
        CombinedNc.ProductionNifs.accepted_implies_delayedRefinement_or_yRingUnbound_or_badEvent
          noZeroDivisors context data certificate input children rawAccepted with
      refinement | yRingUnbound | piCcsBad
    · exact Or.inl (refinement.toSemanticFold packed)
    · exact Or.inr (Or.inl yRingUnbound)
    · exact Or.inr (Or.inr (.currentPiCcs piCcsBad))
  · exact Or.inr (Or.inr (.mixing mixing))
  · exact Or.inr (Or.inr (.childBinding binding))

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionBoundary
