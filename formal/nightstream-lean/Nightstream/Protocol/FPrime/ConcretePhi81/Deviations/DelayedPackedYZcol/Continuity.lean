import Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedProduction
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.SourceInput

/-!
Accumulator continuity for the one-fold delayed packed-`yZcol` deviation.

Owns: recovery of the exact ordered child family and pending payload from two
recomputed state bindings plus strict `Pi_DEC` acceptance on both sides.

Does not own: a digest encoding, hash binding, combined-NC acceptance, output
projection truth, Rust/R1CS refinement, costs, or rows.

Emits constraints: no.

Authority boundary: equality of carried digests is never sufficient by
itself. Both sides must recompute `StateBinds` from complete typed payloads
and pass strict `Pi_DEC`; failure remains an explicit `BindingFailure`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.delayed.continuity.families` | strict `Pi_DEC` fixes the exact previous output and successor running child families | checked/security boundary | `of_piDec_and_stateBindings` |
| `fprime.delayed.continuity.pending` | equal recomputed state bindings recover the exact delayed pending payload or a named binding failure | derived/security partition | `of_piDec_and_stateBindings` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Continuity

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
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

/-- Exact child and pending continuity from strict public `Pi_DEC` on the
previous output and the successor opening-derived running relation. -/
theorem of_piDec_and_stateBindings
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousPiDec : PiDEC.Accepted (decAlgebra previousContext.key)
      ((derive previousContext previousCertificate).piDecAttempt
        previousCertificate))
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
        outputChildren previousContext previousCertificate /\
      nextContext.pending = some
        (DelayedProduction.outgoingPending previousContext
          previousCertificate)) \/
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
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

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Continuity
