import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionSequence
import Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending

/-!
Accumulator-state derivation of delayed production continuity.

Assurance tier: model-level security partition.

Owns: recomputation of the same child-family-plus-pending accumulator payload
on two adjacent production views; derivation of exact public child continuity
and exact pending continuity; and composition with the successor raw-NC
theorem.

Does not own: a concrete Poseidon2 encoding, final R1CS state-digest rows,
source-data decoding, commitment hardness, terminal closure, Rust conformance,
costs, or row removal.

Emits constraints: none.

Authority boundary: equal state coordinates gain authority only because both
sides satisfy `DelayedPending.StateBinds` by recomputation from the complete
typed payload. A serialization/hash failure remains explicit. The incoming
parent itself is separately checked by strict `Pi_DEC`; it is not used as a
digest proxy for the children.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.production.state.previous` | recompute children plus outgoing pending from the previous accepted fold | checked boundary | `previousBinds` |
| `nifs.production.state.next` | recompute the same state coordinate from next running children plus carried pending | checked boundary | `nextBinds` |
| `nifs.production.state.continuity` | recover exact child and pending equality or binding failure | derived/security boundary | `stateBinding_implies_continuity_or_failure` |
| `nifs.production.state.semantic` | feed derived continuity into the successor projection theorem | derived | `acceptedNext_of_stateBinding_implies_previousSemanticFold_or_badEvent` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionState

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending

universe uPreviousState uNextState uEncoding uDigest

variable
  {shape : SemanticShape}
  {PreviousState : Type uPreviousState}
  {NextState : Type uNextState}
  {Encoding : Type uEncoding}
  {Digest : Type uDigest}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Exact child and pending continuity needs only the previous accepted tail
and the next accepted running relation.  In particular, it does not require
the previous packed output to have been extracted already. -/
theorem tailAndRunningStateBinding_implies_continuity_or_failure
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousTail : TailAccepted previousContext previousCertificate)
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextRunning : RunningAuthority.Accepted nextContext)
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
    (nextContext.input.running =
        outputChildren previousContext previousCertificate ∧
      nextContext.pending = some
        (DelayedProduction.outgoingPending previousContext
          previousCertificate)) ∨
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  have previousPiDec := previousTail.piDec
  rcases
    (RunningAuthority.Accepted.iff_nonemptyBound_of_active
        (context := nextContext) rfl).1 nextRunning with
    ⟨nextBound⟩
  have parentEq : nextBound.parent = nextParent :=
    Option.some.inj (nextBound.parentBound.symm.trans nextParentBound)
  subst nextParent
  have nextPiDec : PiDEC.Accepted (decAlgebra nextContext.key) {
      parent := nextBound.parent
      children := nextContext.input.running
    } := by
    simpa [RunningAuthority.attempt, RunningAuthority.children,
      RunningAuthority.activeIndex, nextBound.active] using nextBound.piDec
  rcases children_pending_eq_or_failure_of_stateBinding scheme
      (canonicalFamily_of_accepted previousPiDec)
      (canonicalFamily_of_accepted nextPiDec)
      previousBinds nextBinds rfl with exactPayload | failure
  · exact Or.inl ⟨exactPayload.1.symm, exactPayload.2.symm⟩
  · exact Or.inr failure

/-- Backwards-compatible accepted-object spelling of the state-continuity
partition. -/
theorem stateBinding_implies_continuity_or_failure
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousAccepted : CombinedNc.ProductionNifs.Accepted previousContext
      previousData previousCertificate)
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextAccepted : CombinedNc.ProductionNifs.Accepted nextContext
      nextData nextCertificate)
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
    (nextContext.input.running =
        outputChildren previousContext previousCertificate ∧
      nextContext.pending = some
        (DelayedProduction.outgoingPending previousContext
          previousCertificate)) ∨
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  exact tailAndRunningStateBinding_implies_continuity_or_failure scheme
    stateDigest previousContext previousCertificate previousAccepted.tail
    nextContext nextAccepted.running nextParent nextParentBound previousBinds
    nextBinds

/-- State recomputation plus an accepted raw successor derives the previous
packed output before the previous claims-level execution is extracted. -/
theorem acceptedNext_of_stateBinding_implies_previousPackedYZcolBound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousOpenings :
      ChildOpenings previousContext previousData previousCertificate)
    (previousTail : TailAccepted previousContext previousCertificate)
    (nextContext : FixedActive.Context shape NextState
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
    Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      CombinedNc.ProductionSequence.PreviousClosureBadEvent previousContext
        previousData previousCertificate nextContext nextData nextCertificate
        nextContext.challengeSetSize ∨
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  rcases tailAndRunningStateBinding_implies_continuity_or_failure scheme
      stateDigest previousContext previousCertificate previousTail nextContext
      nextAccepted.running nextParent nextParentBound previousBinds nextBinds with
    continuity | failure
  · rcases
        CombinedNc.ProductionSequence.acceptedNext_implies_previousPackedYZcolBound_or_badEvent
          noZeroDivisors previousContext previousData previousCertificate
          previousOpenings nextContext nextData nextCertificate nextInput
          sameKey continuity.1 continuity.2 nextAccepted with packed | bad
    · exact Or.inl packed
    · exact Or.inr (Or.inl bad)
  · exact Or.inr (Or.inr failure)

/-- Claims-compatible state composition for the raw-parent refinement seam.
The recomputed accumulator payload derives exact pending continuity (and the
complete public child-family equality, although this theorem does not need to
treat that public equality as private opening authority).  Successor raw
acceptance then closes the predecessor packed output, exposes the exact
raw-parent assignment mismatch, or returns a named algebraic/state event.

No predecessor raw acceptance or `ChildOpenings` premise occurs, so this
theorem is suitable for backward induction after the successor packed equation
has been established. -/
theorem acceptedNext_of_stateBinding_implies_previousPackedYZcolBound_or_rawParentStateMismatch_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousTail : TailAccepted previousContext previousCertificate)
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextAccepted : CombinedNc.ProductionNifs.Accepted nextContext nextData
      nextCertificate)
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
      CombinedNc.ProductionSequence.RawParentStateMismatch previousContext
        previousData previousCertificate nextContext nextData ∨
      CombinedNc.ProductionSequence.RawParentStateBadEvent previousContext
        previousData previousCertificate nextContext nextData nextCertificate
        nextContext.challengeSetSize ∨
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  rcases tailAndRunningStateBinding_implies_continuity_or_failure scheme
      stateDigest previousContext previousCertificate previousTail nextContext
      nextAccepted.running nextParent nextParentBound previousBinds nextBinds with
    continuity | failure
  · rcases
        CombinedNc.ProductionSequence.acceptedNext_implies_previousPackedYZcolBound_or_rawParentStateMismatch_or_badEvent
          noZeroDivisors previousContext previousData previousCertificate
          nextContext nextData nextCertificate continuity.2 nextAccepted with
      packed | stateMismatch | bad
    · exact Or.inl packed
    · exact Or.inr (Or.inl stateMismatch)
    · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr failure))

/-- State-recomputed direct-parent closure.  The accumulator payload derives
both public child continuity and pending continuity. The separately checked
canonical parent commitment/norm plus the successor raw-table commitment
alignment reduce the remaining private ambiguity to the standard
parent-opening collision.

This theorem is the non-circular production edge: it uses only the
predecessor accepted tail, never predecessor raw Π_CCS acceptance or the
packed equation being proved. -/
theorem acceptedNext_of_stateBinding_of_parentOpening_implies_previousPackedYZcolBound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousParentBound : DelayedRawChildren.CanonicalParentBinding
      previousContext previousData previousCertificate)
    (previousTail : TailAccepted previousContext previousCertificate)
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextCommitments : DelayedRawChildren.RawRunningCommitmentsBound
      nextContext nextData)
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
    Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      CombinedNc.ProductionSequence.ParentOpeningClosureBadEvent
        previousContext previousData previousCertificate nextContext nextData
        nextCertificate nextContext.challengeSetSize ∨
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  rcases tailAndRunningStateBinding_implies_continuity_or_failure scheme
      stateDigest previousContext previousCertificate previousTail nextContext
      nextAccepted.running nextParent nextParentBound previousBinds nextBinds with
    continuity | failure
  · rcases
        CombinedNc.ProductionSequence.acceptedNext_of_parentOpening_implies_previousPackedYZcolBound_or_badEvent
          noZeroDivisors previousContext previousData previousCertificate
          previousParentBound previousTail.piDec nextContext nextData
          nextCertificate nextCommitments sameKey continuity.1 continuity.2
          nextAccepted with packed | bad
    · exact Or.inl packed
    · exact Or.inr (Or.inl bad)
  · exact Or.inr (Or.inr failure)

/-- State recomputation discharges both continuity premises of the adjacent
raw-NC theorem. The only new branch is the explicit accumulator payload
binding failure. -/
theorem acceptedNext_of_stateBinding_implies_previousSemanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousAccepted : CombinedNc.ProductionNifs.Accepted previousContext
      previousData previousCertificate)
    (previousRefinement :
      CombinedNc.ProductionNifs.DelayedRefinement previousContext previousData
        previousCertificate)
    (nextContext : FixedActive.Context shape NextState
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
      CombinedNc.ProductionSequence.PreviousClosureBadEvent previousContext
        previousData previousCertificate nextContext nextData nextCertificate
        nextContext.challengeSetSize ∨
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  rcases
      acceptedNext_of_stateBinding_implies_previousPackedYZcolBound_or_badEvent
        noZeroDivisors scheme stateDigest previousContext previousData
        previousCertificate previousRefinement.children previousAccepted.tail
        nextContext nextData nextCertificate nextInput nextAccepted nextParent
        nextParentBound sameKey previousBinds nextBinds with
    packed | bad | failure
  · exact Or.inl (previousRefinement.toSemanticFold (by simpa using packed))
  · exact Or.inr (Or.inl bad)
  · exact Or.inr (Or.inr failure)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionState
