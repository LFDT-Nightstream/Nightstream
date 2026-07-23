import Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Edge
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.PaperStep
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Terminal

/-!
Finite traces for the one-fold delayed packed-`yZcol` deviation.

Assurance tier: model-level registered production deviation.

Owns: one common verifier-key index for every step; the literal pending-none
base boundary; recursive closure of each predecessor from its successor;
terminal closure of the final output from fourteen ordered raw child
assignments; and the exact trace-level failure partition.

Does not own: construction of accepted step receipts, concrete transcript
hashing, commitment hardness, Rust/R1CS refinement, generated rows, costs, or
row-removal authority.

Emits constraints: no.

Authority boundary: recursive edges use the successor's authoritative raw NC
assignments and two accumulator handles recomputed from complete typed
payloads. The terminal edge uses the ordered raw child assignments themselves.
No child `CeClaim.y_zcol`, digest authority, generic output-unbound premise, or
implementation-refinement failure appears here.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.delayed.trace.context` | keep one verifier key and production profile across every checked step | checked | `SharedContext`, `CheckedStep`, `CheckedStep.key_eq` |
| `fprime.delayed.trace.base` | require the literal no-pending base boundary | checked | `BaseStep`, `BaseStep.pending_eq_none` |
| `fprime.delayed.trace.edge` | close each predecessor from its accepted successor and recomputed state bindings | derived/security boundary | `Edge`, `Tail`, `tail_implies_allClosed_or_failure` |
| `fprime.delayed.trace.terminal` | close the final output from fourteen authoritative raw child openings | derived/security boundary | `TerminalClosure`, `singleton_implies_closed_or_terminalFailure` |
| `fprime.delayed.trace.closed` | expose base and all-output closure or the exact trace failure partition | derived/security partition | `ClosedTrace`, `closedTrace_implies_baseAndAllClosed_or_failure` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Trace

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
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending

universe uState uEncoding uDigest

variable {shape : SemanticShape}
variable {State : Type uState}
variable {Encoding : Type uEncoding}
variable {Digest : Type uDigest}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- A trace-wide context indexed by its verifier key. Key agreement is thus a
data invariant rather than a caller-supplied proposition. -/
structure SharedContext
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) where
  value : FixedActive.CanonicalOpening.Context shape State publicRingColumns
    publicFits verifierRows
  key_eq : value.key = key

/-- One opening-derived step with all protocol-level paper obligations
accepted. -/
structure CheckedStep
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) where
  carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
    publicRingColumns publicFits
  sharedContext : SharedContext (State := State) key
  certificate : FixedActive.Certificate
    (carrier.install sharedContext.value).full
  accepted : PaperStep.PaperStepAccepted carrier sharedContext.value
    certificate

namespace CheckedStep

def context
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : CheckedStep (State := State) key) :=
  step.sharedContext.value

def full
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : CheckedStep (State := State) key) :=
  (step.carrier.install step.context).full

theorem key_eq
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : CheckedStep (State := State) key) :
    step.context.key = key :=
  step.sharedContext.key_eq

end CheckedStep

/-- The explicit base boundary. The first full input literally carries no
pending delayed projection. -/
structure BaseStep
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) where
  step : CheckedStep (State := State) key
  noPending : step.full.pending = none

namespace BaseStep

def toCheckedStep
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (base : BaseStep (State := State) key) : CheckedStep (State := State) key :=
  base.step

theorem pending_eq_none
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (base : BaseStep (State := State) key) :
    base.toCheckedStep.full.pending = none :=
  base.noPending

end BaseStep

/-- The exact delayed projection and independent paper transition closed for
one trace step. -/
structure OutputClosed
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : CheckedStep (State := State) key) : Prop where
  packed :
    _root_.Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
      step.full.covers step.carrier.data
      (derive step.full step.certificate).piCcs.ncPoint.block
      step.certificate.piCcs.output
  paper : FixedActive.PaperProfile.Transition
    (FixedActive.paperProfileOf step.full) step.full.input
    (outputChildren step.full step.certificate)

/-- The final output is closed by an opening over exactly the fourteen
ordered production children. -/
structure TerminalClosure
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : CheckedStep (State := State) key) where
  rawChildren : Fin productionGlobalParams.k ->
    Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits)
  accepted :
    DelayedPackedYZcol.Terminal.ProjectionOpeningAccepted step.full
      step.certificate rawChildren

/-- One recursive-edge receipt. Both sides bind the same state coordinate by
recomputing it from their complete typed child family and delayed value. -/
structure Edge
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (previous next : CheckedStep (State := State) key) where
  stateDigest : Digest
  previousBinds : StateBinds scheme stateDigest
    (derive previous.full previous.certificate).piRlcOutput
    (outputChildren previous.full previous.certificate)
    (some (DelayedProduction.outgoingPending previous.full
      previous.certificate))
  nextBinds : StateBinds scheme stateDigest
    (next.carrier.opening.parent next.context.key next.carrier.system)
    next.full.input.running next.full.pending

/-- A nonempty trace tail. The final output has an actual terminal opening;
each `prepend` carries the receipt that closes its predecessor. -/
inductive Tail
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    {key : VerifierKey shape publicRingColumns publicFits verifierRows} :
    CheckedStep (State := State) key -> Type _ where
  | terminal (last : CheckedStep (State := State) key)
      (closure : TerminalClosure last) : Tail scheme last
  | prepend (previous next : CheckedStep (State := State) key)
      (edge : Edge scheme previous next)
      (tail : Tail scheme next) : Tail scheme previous

/-- Exact final-step failures. -/
inductive TerminalFailure
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : CheckedStep (State := State) key) : Prop where
  | yRingUnbound
      (failure : PaperStep.ProductionPiCcs.YRingUnbound step.full
        step.carrier.data step.certificate) :
      TerminalFailure step
  | piCcs
      (failure : PaperStep.ProductionPiCcs.BadEvent step.full
        step.carrier.data step.certificate) :
      TerminalFailure step
  | piRlcMixing
      (failure : PiRlcSidecar.MixingCollision step.full.covers
        step.certificate.piRlcChallenges
        (InputAuthority.productAssignments step.carrier.data
          step.full.alignment)
        (DelayedProduction.outgoingPending step.full
          step.certificate).oldBlock
        (PackedYZcol.sourceClaims step.full step.certificate)) :
      TerminalFailure step
  | parentOpeningBinding
      (failure : Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics step.full.key) productionGlobalParams
        (derive step.full step.certificate).piRlcOutput.commitment)) :
      TerminalFailure step

/-- Recursive failures are exactly the named constructors of the
protocol-owned edge theorem. This wrapper adds no fallback event. -/
structure EdgeFailure
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (previous next : CheckedStep (State := State) key) : Prop where
  event :
    DelayedPackedYZcol.Edge.Failure scheme previous.full previous.carrier.data
      previous.certificate next.full next.carrier.data next.certificate

/-- Every output in one trace tail is closed. -/
inductive AllOutputsClosed
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {key : VerifierKey shape publicRingColumns publicFits verifierRows} :
    {head : CheckedStep (State := State) key} -> Tail scheme head -> Prop where
  | terminal {last} {closure : TerminalClosure last}
      (closed : OutputClosed last) :
      AllOutputsClosed (Tail.terminal last closure)
  | prepend {previous next} {edge : Edge scheme previous next}
      {tail : Tail scheme next}
      (closed : OutputClosed previous)
      (rest : AllOutputsClosed tail) :
      AllOutputsClosed (Tail.prepend previous next edge tail)

namespace AllOutputsClosed

theorem head
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    {first : CheckedStep (State := State) key}
    {tail : Tail scheme first}
    (closed : AllOutputsClosed tail) : OutputClosed first := by
  cases closed with
  | terminal closed => exact closed
  | prepend closed _ => exact closed

end AllOutputsClosed

/-- One exact event at a definite terminal or recursive edge. `later` records
only its location; it introduces no new failure kind. -/
inductive Failure
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {key : VerifierKey shape publicRingColumns publicFits verifierRows} :
    {head : CheckedStep (State := State) key} -> Tail scheme head -> Prop where
  | terminal {last} {closure : TerminalClosure last}
      (failure : TerminalFailure last) :
      Failure (Tail.terminal last closure)
  | edge {previous next} {edge : Edge scheme previous next}
      {tail : Tail scheme next}
      (failure : EdgeFailure scheme previous next) :
      Failure (Tail.prepend previous next edge tail)
  | later {previous next} {edge : Edge scheme previous next}
      {tail : Tail scheme next}
      (failure : Failure tail) :
      Failure (Tail.prepend previous next edge tail)

/-- A complete trace has an explicit base boundary and a nonempty tail ending
in an actual terminal opening. -/
structure ClosedTrace
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) where
  base : BaseStep (State := State) key
  tail : Tail scheme base.toCheckedStep

/-- A singleton output is closed directly by its terminal raw-child
opening, or yields one exact terminal failure. -/
theorem singleton_implies_closed_or_terminalFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : CheckedStep (State := State) key)
    (closure : TerminalClosure step) :
    OutputClosed step \/ TerminalFailure step := by
  rcases
      DelayedPackedYZcol.Terminal.projectionOpeningAccepted_implies_packedYZcolBound_or_badEvent
        step.full step.carrier.data step.certificate
        step.accepted.canonicalParent step.accepted.piDecAccepted
        closure.rawChildren closure.accepted with
    packed | mixing | binding
  · rcases
        PaperStep.accepted_and_packed_implies_transition_or_yRingUnbound_or_badEvent
          noZeroDivisors step.carrier step.context step.certificate
          step.accepted packed with
      paper | yRing | piCcs
    · exact Or.inl ⟨packed, paper⟩
    · exact Or.inr (.yRingUnbound yRing)
    · exact Or.inr (.piCcs piCcs)
  · exact Or.inr (.piRlcMixing mixing)
  · exact Or.inr (.parentOpeningBinding binding)

/-- The explicit singleton boundary additionally exposes that the base input
has no delayed predecessor. -/
theorem baseSingleton_implies_noPendingAndClosed_or_terminalFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (base : BaseStep (State := State) key)
    (closure : TerminalClosure base.toCheckedStep) :
    (base.toCheckedStep.full.pending = none ∧
      OutputClosed base.toCheckedStep) \/
      TerminalFailure base.toCheckedStep := by
  rcases singleton_implies_closed_or_terminalFailure noZeroDivisors
      base.toCheckedStep closure with closed | failure
  · exact Or.inl ⟨base.pending_eq_none, closed⟩
  · exact Or.inr failure

/-- Backward induction over the one-fold delay. The final output is closed by
the terminal opening, then each successor closes its predecessor. -/
theorem tail_implies_allClosed_or_failure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    {head : CheckedStep (State := State) key}
    (tail : Tail scheme head) :
    AllOutputsClosed tail \/ Failure tail := by
  induction tail with
  | terminal last closure =>
      rcases singleton_implies_closed_or_terminalFailure noZeroDivisors last
          closure with closed | failure
      · exact Or.inl (.terminal closed)
      · exact Or.inr (.terminal failure)
  | prepend previous next edge tail ih =>
      rcases ih with rest | later
      · have nextClosed : OutputClosed next := rest.head
        have sameKey : next.context.key = previous.context.key :=
          next.key_eq.trans previous.key_eq.symm
        rcases
            DelayedPackedYZcol.Edge.acceptedPair_of_nextPacked_implies_previousClosed_or_failure
              noZeroDivisors scheme edge.stateDigest previous.carrier
              previous.context previous.certificate previous.accepted
              next.carrier next.context next.certificate next.accepted
              nextClosed.packed sameKey edge.previousBinds edge.nextBinds with
          closed | failure
        · exact Or.inl (.prepend {
            packed := by
              simpa using closed.1
            paper := closed.2
          } rest)
        · exact Or.inr (.edge ⟨failure⟩)
      · exact Or.inr (.later later)

/-- Complete base/recursive/terminal composition. Successful traces expose
the pending-none base boundary and close every output. The only alternative
is one exact terminal or recursive-edge event enumerated above. -/
theorem closedTrace_implies_baseAndAllClosed_or_failure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (trace : ClosedTrace (State := State) scheme key) :
    (trace.base.toCheckedStep.full.pending = none ∧
      AllOutputsClosed trace.tail) \/
      Failure trace.tail := by
  rcases tail_implies_allClosed_or_failure noZeroDivisors scheme trace.tail with
    closed | failure
  · exact Or.inl ⟨trace.base.pending_eq_none, closed⟩
  · exact Or.inr failure

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Trace
