import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperSequence
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperTerminal

/-!
Finite production traces for the opening-derived paper checker.

Protocol: SuperNeo Sections 7.3--7.5.
Phase: base, recursive one-fold delay, and terminal closure.
Assurance tier: model-level composition.

Owns: one common verifier-key index for every step, the executable
pending-none base boundary, recursive closure of each predecessor by its
successor, terminal closure of the final output, and the exact trace-level
failure partition.

Does not own: Rust differential execution, physical rows, transcript or
commitment primitive internals, costs, or row-removal authority.

Authority boundary: every step is checked over one opening-derived carrier.
The recursive edge reads the successor's raw NC table and two recomputed
state-binding checks. The terminal edge reads the ordered raw child
assignments. No child `CeClaim.y_zcol`, generic output-unbound proposition,
or implementation-refinement failure appears here.
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperTrace

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
variable [DecidableEq Digest]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- A production context indexed by the trace-wide verifier key. This makes
key agreement an invariant of the trace data, rather than a theorem premise
or a possible runtime failure. -/
structure SharedContext
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) where
  value : FixedActive.CanonicalOpening.Context shape State publicRingColumns
    publicFits verifierRows
  key_eq : value.key = key

/-- One claims-level step accepted by the compact executable paper checker. -/
structure CheckedStep
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) where
  carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
    publicRingColumns publicFits
  sharedContext : SharedContext (State := State) key
  certificate : FixedActive.Certificate
    (carrier.install sharedContext.value).full
  checked : ProductionPaperChecker.check carrier sharedContext.value
    certificate = true

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

theorem accepted
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : CheckedStep (State := State) key) :
    ProductionPaperNifs.PaperStepAccepted step.carrier step.context
      step.certificate :=
  (ProductionPaperChecker.check_eq_true_iff_accepted step.carrier step.context
    step.certificate).1 step.checked

end CheckedStep

/-- The first step is accepted by the executable base checker, including its
literal pending-none test. -/
structure BaseStep
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) where
  carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
    publicRingColumns publicFits
  sharedContext : SharedContext (State := State) key
  certificate : FixedActive.Certificate
    (carrier.install sharedContext.value).full
  checked : ProductionPaperChecker.baseCheck carrier sharedContext.value
    certificate = true

namespace BaseStep

def context
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (base : BaseStep (State := State) key) :=
  base.sharedContext.value

def toCheckedStep
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (base : BaseStep (State := State) key) : CheckedStep (State := State) key := {
  carrier := base.carrier
  sharedContext := base.sharedContext
  certificate := base.certificate
  checked := (ProductionPaperChecker.check_eq_true_iff_accepted base.carrier
    base.context base.certificate).2
      ((ProductionPaperChecker.baseCheck_eq_true_iff_accepted base.carrier
        base.context base.certificate).1 base.checked).step
}

theorem pending_eq_none
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (base : BaseStep (State := State) key) :
    (base.carrier.install base.context).full.pending = none :=
  ((ProductionPaperChecker.baseCheck_eq_true_iff_accepted base.carrier
    base.context base.certificate).1 base.checked).noPending

end BaseStep

/-- The exact packed projection and independent paper transition established
for one trace step. -/
structure OutputClosed
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : CheckedStep (State := State) key) : Prop where
  packed : Terminal.PackedYZcolBoundAtBlock step.full.covers step.carrier.data
    (ProductionPiCcs.ncPoint step.full step.certificate).block
    step.certificate.piCcs.output
  paper : FixedActive.PaperProfile.Transition
    (FixedActive.paperProfileOf step.full) step.full.input
    (outputChildren step.full step.certificate)

/-- A final-step closure is the actual terminal checker over its ordered raw
child assignments. -/
structure TerminalClosure
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : CheckedStep (State := State) key) where
  rawChildren : Fin productionGlobalParams.k ->
    Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits)
  checked : ProductionTerminal.check step.full step.certificate rawChildren =
    true

/-- One recursive edge. Both sides recompute the same edge digest from their
complete typed payloads; the digest is not authority by itself. -/
structure Edge
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (previous next : CheckedStep (State := State) key) where
  stateDigest : Digest
  previousChecked :
    ProductionChecker.stateBindingCheck scheme stateDigest
      (derive previous.full previous.certificate).piRlcOutput
      (outputChildren previous.full previous.certificate)
      (some (DelayedProduction.outgoingPending previous.full
        previous.certificate)) = true
  nextChecked :
    ProductionChecker.stateBindingCheck scheme stateDigest
      (next.carrier.opening.parent next.context.key next.carrier.system)
      next.full.input.running next.full.pending = true

/-- A nonempty backward-closed trace. `terminal` closes the final output;
`prepend` closes the previous output with the successor at the head of the
already-closed tail. -/
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
      (failure : ProductionPiCcs.YRingUnbound step.full step.carrier.data
        step.certificate) : TerminalFailure step
  | piCcs
      (failure : ProductionPiCcs.BadEvent step.full step.carrier.data
        step.certificate) : TerminalFailure step
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

/-- Exact recursive-edge failures. The successor `Pi_CCS` event is separate
from the predecessor event because it is the source of raw NC truth. -/
inductive EdgeFailure
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (previous next : CheckedStep (State := State) key) : Prop where
  | previousYRingUnbound
      (failure : ProductionPiCcs.YRingUnbound previous.full
        previous.carrier.data previous.certificate) :
      EdgeFailure scheme previous next
  | previousPiCcs
      (failure : ProductionPiCcs.BadEvent previous.full previous.carrier.data
        previous.certificate) : EdgeFailure scheme previous next
  | nextPiCcs
      (failure : ProductionPiCcs.BadEvent next.full next.carrier.data
        next.certificate) : EdgeFailure scheme previous next
  | parentOpeningClosure
      (failure : ProductionSequence.ParentOpeningClosureBadEvent previous.full
        previous.carrier.data previous.certificate next.full next.carrier.data
        next.certificate next.full.challengeSetSize) :
      EdgeFailure scheme previous next
  | accumulatorBinding
      (failure : Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure
        scheme) : EdgeFailure scheme previous next

/-- Every step in a trace has both its packed equation and paper transition. -/
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

/-- One exact event at a definite terminal or recursive edge. `later` only
records its position in the finite trace; it introduces no new failure kind. -/
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

/-- A complete production trace starts with the executable base check and is
nonempty because its tail ends in an actual terminal closure. -/
structure ClosedTrace
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) where
  base : BaseStep (State := State) key
  tail : Tail scheme base.toCheckedStep

/-- A singleton base/final step is closed directly by the terminal checker. -/
theorem singleton_implies_closed_or_terminalFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : CheckedStep (State := State) key)
    (closure : TerminalClosure step) :
    OutputClosed step \/ TerminalFailure step := by
  rcases
      ProductionPaperTerminal.checkedTerminal_implies_packedAndPaper_or_namedFailure
        noZeroDivisors step.carrier step.context step.certificate step.accepted
        closure.rawChildren closure.checked with
    closed | yRing | piCcs | mixing | binding
  · exact Or.inl { packed := closed.1, paper := closed.2 }
  · exact Or.inr (.yRingUnbound yRing)
  · exact Or.inr (.piCcs piCcs)
  · exact Or.inr (.piRlcMixing mixing)
  · exact Or.inr (.parentOpeningBinding binding)

/-- The explicit singleton boundary: the one step is a checked base with no
pending predecessor, and its own output is closed by the terminal verifier. -/
theorem baseSingleton_implies_noPendingAndClosed_or_terminalFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (base : BaseStep (State := State) key)
    (closure : TerminalClosure base.toCheckedStep) :
    (((base.carrier.install base.context).full.pending = none) ∧
      OutputClosed base.toCheckedStep) \/
      TerminalFailure base.toCheckedStep := by
  rcases singleton_implies_closed_or_terminalFailure noZeroDivisors
      base.toCheckedStep closure with closed | failure
  · exact Or.inl ⟨base.pending_eq_none, closed⟩
  · exact Or.inr failure

/-- Backward induction over the one-fold delay. The final output is closed by
the terminal checker; every recursive predecessor is then closed by the next
step's already-derived packed equation. -/
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
            ProductionPaperSequence.checkedPair_of_nextPacked_implies_previousPackedAndPaper_or_namedFailure
              noZeroDivisors scheme edge.stateDigest previous.carrier
              previous.context previous.certificate previous.accepted next.carrier
              next.context next.certificate next.accepted nextClosed.packed
              sameKey edge.previousChecked edge.nextChecked with
          closed | yRing | previousPiCcs | nextPiCcs | parentClosure |
            accumulatorBinding
        · exact Or.inl (.prepend {
            packed := closed.1
            paper := closed.2
          } rest)
        · exact Or.inr (.edge (.previousYRingUnbound yRing))
        · exact Or.inr (.edge (.previousPiCcs previousPiCcs))
        · exact Or.inr (.edge (.nextPiCcs nextPiCcs))
        · exact Or.inr (.edge (.parentOpeningClosure parentClosure))
        · exact Or.inr (.edge (.accumulatorBinding accumulatorBinding))
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
    ((trace.base.carrier.install trace.base.context).full.pending = none ∧
      AllOutputsClosed trace.tail) \/
      Failure trace.tail := by
  rcases tail_implies_allClosed_or_failure noZeroDivisors scheme trace.tail with
    closed | failure
  · exact Or.inl ⟨trace.base.pending_eq_none, closed⟩
  · exact Or.inr failure

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperTrace
