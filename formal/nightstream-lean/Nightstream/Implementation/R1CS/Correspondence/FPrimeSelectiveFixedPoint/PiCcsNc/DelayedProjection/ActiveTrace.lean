import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary

/-!
Finite production trace closure for delayed packed-`y_zcol` authority.

Assurance tier: model-level for the final semantic composition, with the
step premise reduced to the executable claims checker. Exact generated
combined-NC, state, transcript, and terminal-opening row refinement remains a
separate artifact boundary.

Owns: a nonempty digest-linked claims trace; the explicit no-pending base
boundary; one terminal opening on the final step; backward propagation of the
terminal-derived packed equation; and Construction-2 soundness for every step.

Does not own: generated rows, Rust dataflow, commitment-key alignment,
Poseidon2 or Ajtai internals, `y_ring`, costs, or row-removal permission.

Emits constraints: none; executable/refinement contract only.

| Stable stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `f_prime.pi_ccs_nc.delayed.trace.base` | the first claims step has no incoming pending value and uses ordinary NC | checked/derived | `terminalChecked_implies_baseAndAllPaper_or_namedFailure` |
| `f_prime.pi_ccs_nc.delayed.trace.edge` | an accepted successor propagates one positive packed equation backward | derived/security partition | `terminalChecked_implies_headPackedAndAllPaper_or_failure` |
| `f_prime.pi_ccs_nc.delayed.trace.terminal` | complete final child openings create the backward induction anchor | checked/security partition | `TerminalChecked` |
| `f_prime.pi_ccs_nc.delayed.trace.paper` | every successful trace step reaches Construction-2 | derived | `AllPaper` |

The successful theorem contains no generic output-unbound or packed-`y_zcol`
opening-failure branch. Every such failure is reduced at an accepted edge or
at the terminal anchor to the specifically typed failures below.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
open PackedWitness

universe uOuterKey uAppState uWitness uDigest uTranscriptState uEncoding

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {Encoding : Type uEncoding}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- One actual claims-level production step with both accumulator endpoints
fixed in its type. Adjacent trace constructors share the middle digest by
construction. -/
structure Step
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (incoming outgoing : Digest) where
  input : ProductionContext.Input OuterKey AppState Witness shape
    publicRingColumns publicFits verifierRows
  template : Data shape
  witnesses : Fin shape.runningCount -> Matrix shape
  certificate : FixedActive.Certificate (ProductionContext.full setup input)
  checked : ActiveBoundary.claimsCheck scheme incoming outgoing machine setup
    input certificate = true

namespace Step

/-- Every trace step derives the structured claims contract from the exact
executable production Boolean. The trace can no longer receive
`ClaimsAccepted` as a caller-supplied semantic premise. -/
theorem accepted
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (step : Step scheme machine setup incoming outgoing) :
    ActiveBoundary.ClaimsAccepted scheme incoming outgoing machine setup
      step.input step.template step.witnesses step.certificate :=
  (ActiveBoundary.claimsCheck_eq_true_iff scheme incoming outgoing machine
    setup step.input step.template step.witnesses step.certificate).mp
    step.checked

/-- The delayed packed equation carried backward from the terminal anchor. -/
def Packed
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (step : Step scheme machine setup incoming outgoing) : Prop :=
  Terminal.PackedYZcolBoundAtBlock
    (ProductionContext.full setup step.input).covers
    (decodedData step.template step.witnesses)
    (ProductionPiCcs.ncPoint (ProductionContext.full setup step.input)
      step.certificate).block step.certificate.piCcs.output

/-- Independent HyperNova Construction-2 result for one trace step. -/
def Paper
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (functionIndex : Fin 1)
    (step : Step scheme machine setup incoming outgoing) : Prop :=
  Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
    (SelectedNifsSemantics.family
      (ActiveSemantics.Construction2.selectedNifsSetup setup))
    machine functionIndex (step.input.fixedOne.toActive setup).toPaper
    (ActiveBoundary.outputOf machine setup step.input
      step.certificate).toPaper

/-- Ordinary combined-NC base claim after proving the first step has no
incoming delayed value. -/
def BaseNc
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (step : Step scheme machine setup incoming outgoing) : Prop :=
  FixedPhase.Accepted ConcreteCarrier.extensionOps.toOps
    (InitialSum.sumcheckPolynomial
      (ProductionContext.full setup step.input).covers
      (decodedData step.template step.witnesses)
      (ProductionContext.full setup step.input).ncCoins)
    InitialSum.claimedInitial
    (ProductionPiCcs.ncPoint (ProductionContext.full setup step.input)
      step.certificate).coordinates
    step.certificate.piCcs.nc.toSumCheck

/-- Exact failures owned by the terminal step. -/
def TerminalFailure
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (step : Step scheme machine setup incoming outgoing) : Prop :=
  ProductionPiCcs.YRingUnbound (ProductionContext.full setup step.input)
      (decodedData step.template step.witnesses) step.certificate \/
    ProductionBoundary.TerminalBadEvent
      (ProductionContext.full setup step.input)
      (decodedData step.template step.witnesses) step.certificate \/
    RefinementBoundary.TerminalRefinementFailure
      (ProductionContext.canonical setup step.input) step.template
      step.witnesses step.certificate

/-- Exact failures owned by one predecessor/successor edge. -/
def RecursiveFailure
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming shared outgoing : Digest}
    (previous : Step scheme machine setup incoming shared)
    (next : Step scheme machine setup shared outgoing) : Prop :=
  ProductionPiCcs.YRingUnbound
      (ProductionContext.full setup previous.input)
      (decodedData previous.template previous.witnesses)
      previous.certificate \/
    ProductionBoundary.RecursiveBadEvent scheme
      (ProductionContext.full setup previous.input)
      (decodedData previous.template previous.witnesses)
      previous.certificate (ProductionContext.full setup next.input)
      (decodedData next.template next.witnesses) next.certificate \/
    RefinementBoundary.RecursiveRefinementFailure
      (ProductionContext.canonical setup previous.input) previous.template
      previous.witnesses previous.certificate
      (ProductionContext.canonical setup next.input) next.template
      next.witnesses

/-- Packed-y-zcol-only terminal failure.  It contains no y-ring or semantic
child-opening obligation. -/
def ParentOpeningTerminalFailure
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (step : Step scheme machine setup incoming outgoing) : Prop :=
  ProductionBoundary.ParentOpeningTerminalBadEvent
    (ProductionContext.full setup step.input)
    (decodedData step.template step.witnesses) step.certificate

/-- Packed-y-zcol-only adjacent failure.  Algebraic/commitment/state events
and physical parent/input/key binding failures retain separate owners. -/
def ParentOpeningRecursiveFailure
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming shared outgoing : Digest}
    (previous : Step scheme machine setup incoming shared)
    (next : Step scheme machine setup shared outgoing) : Prop :=
  ProductionBoundary.ParentOpeningRecursiveBadEvent scheme
      (ProductionContext.full setup previous.input)
      (decodedData previous.template previous.witnesses)
      previous.certificate (ProductionContext.full setup next.input)
      (decodedData next.template next.witnesses) next.certificate ∨
    ActiveBoundary.ParentOpeningActiveBindingFailure setup previous.input
      next.input previous.template next.template previous.witnesses
      next.witnesses previous.certificate

end Step

/-- A nonempty claims trace. Digest equality at every adjacent edge is encoded
by the shared `middle` index, rather than supplied as an equality premise. -/
inductive Trace
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1) : Digest -> Digest -> Type _ where
  | single {incoming outgoing : Digest}
      (step : Step scheme machine setup incoming outgoing) :
      Trace scheme machine setup incoming outgoing
  | cons {incoming middle outgoing : Digest}
      (head : Step scheme machine setup incoming middle)
      (tail : Trace scheme machine setup middle outgoing) :
      Trace scheme machine setup incoming outgoing

namespace Trace

/-- First step, retaining its outgoing digest dependently. -/
def headStep
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) :
    Σ nextDigest, Step scheme machine setup incoming nextDigest :=
  match trace with
  | .single step => ⟨_, step⟩
  | .cons head _ => ⟨_, head⟩

/-- The first step is the base step and has no incoming delayed value. -/
def BaseBoundary
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) : Prop :=
  trace.headStep.2.input.pending = none

/-- One complete terminal opening is attached only to the final step. -/
def TerminalChecked
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) : Prop :=
  match trace with
  | .single step => ∃ terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape,
      PackedWitnessProduction.terminalCheck
        (ProductionContext.canonical setup step.input) step.certificate
        terminalWitnesses = true
  | .cons _ tail => tail.TerminalChecked

/-- Executable no-pending test on the actual first input. -/
def baseCheck
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) : Bool :=
  match trace.headStep.2.input.pending with
  | none => true
  | some _ => false

theorem baseCheck_eq_true_iff
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) :
    trace.baseCheck = true <-> trace.BaseBoundary := by
  constructor
  · intro checked
    cases pendingEq : trace.headStep.2.input.pending with
    | none => exact pendingEq
    | some pending => simp [baseCheck, pendingEq] at checked
  · intro base
    change trace.headStep.2.input.pending = none at base
    unfold baseCheck
    rw [base]

/-- Execute the actual terminal checker on the last trace step. -/
def terminalCheck
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape) : Bool :=
  match trace with
  | .single step =>
      PackedWitnessProduction.terminalCheck
        (ProductionContext.canonical setup step.input) step.certificate
        terminalWitnesses
  | .cons _ tail => tail.terminalCheck terminalWitnesses

theorem terminalCheck_eq_true_implies
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (checked : trace.terminalCheck terminalWitnesses = true) :
    trace.TerminalChecked := by
  induction trace with
  | single step => exact ⟨terminalWitnesses, checked⟩
  | cons head tail ih => exact ih checked

/-- Actual production-trace acceptance data. Every step already stores its
`claimsCheck = true` result; this final package adds only the executable base
and terminal results on concrete trace inputs. It contains no semantic
acceptance predicate, packed equation, source-binding proposition, or caller
provided `ClaimsAccepted`. -/
structure RuntimeAccepted
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) : Type _ where
  base : trace.baseCheck = true
  terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape
  terminal : trace.terminalCheck terminalWitnesses = true

/-- Every step in the trace satisfies independent Construction-2 semantics. -/
def AllPaper
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (functionIndex : Fin 1)
    (trace : Trace scheme machine setup incoming outgoing) : Prop :=
  match trace with
  | .single step => step.Paper functionIndex
  | .cons head tail => head.Paper functionIndex ∧ tail.AllPaper functionIndex

/-- Every step carries the source-bound packed y-zcol equation.  This trace
property is intentionally independent of Construction-2 and y-ring. -/
def AllPacked
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) : Prop :=
  match trace with
  | .single step => step.Packed
  | .cons head tail => head.Packed ∧ tail.AllPacked

/-- The packed equation for the first step, used as the backward induction
value and then consumed by the explicit base theorem. -/
def HeadPacked
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) : Prop :=
  trace.headStep.2.Packed

/-- Ordinary combined-NC statement for the first, no-pending step. -/
def BaseNc
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) : Prop :=
  trace.headStep.2.BaseNc

/-- Failure tree with exact terminal and edge ownership. No constructor can
hide a packed-output mismatch behind a generic output-unbound proposition. -/
def Failure
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) : Prop :=
  match trace with
  | .single step => step.TerminalFailure
  | .cons head tail =>
      head.RecursiveFailure tail.headStep.2 ∨ tail.Failure

/-- Y-zcol-only failure tree.  It has no y-ring, Construction-2, generic
output-unbound, or semantic child-opening branch. -/
def ParentOpeningFailure
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) : Prop :=
  match trace with
  | .single step => step.ParentOpeningTerminalFailure
  | .cons head tail =>
      head.ParentOpeningRecursiveFailure tail.headStep.2 ∨
        tail.ParentOpeningFailure

/-- An all-packed trace exposes the packed equation at its first step. -/
private theorem allPacked_headPacked
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (allPacked : trace.AllPacked) : trace.HeadPacked := by
  cases trace with
  | single step => exact allPacked
  | cons head tail => exact allPacked.1

/-- Internal packed-only induction.  The final raw child matrices seed the
last packed equation; every accepted successor then closes exactly one
predecessor. -/
private theorem terminalChecked_implies_allPacked_or_parentOpeningFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (terminal : trace.TerminalChecked) :
    trace.AllPacked ∨ trace.ParentOpeningFailure := by
  induction trace with
  | single step =>
      rcases terminal with ⟨terminalWitnesses, terminalCheck⟩
      rcases
          ActiveBoundary.claimsAcceptedTerminal_implies_packed_or_parentOpeningBadEvent
            scheme _ _ machine setup step.input step.template step.witnesses
            step.certificate step.accepted terminalWitnesses terminalCheck with
        packed | bad
      · exact Or.inl packed
      · exact Or.inr bad
  | cons head tail inductionHypothesis =>
      rcases inductionHypothesis terminal with tailPacked | tailFailure
      · let next := tail.headStep
        have nextPacked : next.2.Packed :=
          allPacked_headPacked tail tailPacked
        rcases
            ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousPacked_or_parentOpeningBadEvent
              noZeroDivisors scheme _ _ next.1 machine setup head.input
              head.template head.witnesses head.certificate head.accepted
              next.2.input next.2.template next.2.witnesses
              next.2.certificate next.2.accepted nextPacked with
          packed | bad | binding
        · exact Or.inl ⟨packed, tailPacked⟩
        · exact Or.inr (Or.inl (Or.inl bad))
        · exact Or.inr (Or.inl (Or.inr binding))
      · exact Or.inr (Or.inr tailFailure)

/-- Complete finite-trace packed-y-zcol theorem.  The first step is an
ordinary no-pending NC claim, the final step is anchored by complete raw child
matrices, and every intermediate output is bound exactly one fold later.

The result contains no generic output-unbound branch and no `ChildOpenings`
premise.  `ParentOpeningFailure` names only algebraic roots, SumCheck or
commitment/binding failures, accumulator binding, input/key alignment, and
terminal parent-opening failure. -/
theorem terminalChecked_implies_baseAndAllPacked_or_parentOpeningFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (base : trace.BaseBoundary)
    (terminal : trace.TerminalChecked) :
    (trace.BaseNc ∧ trace.AllPacked) ∨ trace.ParentOpeningFailure := by
  rcases terminalChecked_implies_allPacked_or_parentOpeningFailure
      noZeroDivisors scheme machine setup trace terminal with
    allPacked | failure
  · let first := trace.headStep
    have headPacked : first.2.Packed :=
      allPacked_headPacked trace allPacked
    have baseNc : first.2.BaseNc :=
      ActiveBoundary.claimsAcceptedBase_of_packed_implies_ordinaryNc
        scheme _ _ machine setup first.2.input first.2.template
        first.2.witnesses first.2.certificate first.2.accepted headPacked base
    exact Or.inl ⟨baseNc, allPacked⟩
  · exact Or.inr failure

/-- Internal strong induction: terminal authority yields the last packed
equation, and every accepted successor returns the preceding packed equation
together with that predecessor's Construction-2 result. -/
private theorem terminalChecked_implies_headPackedAndAllPaper_or_failure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (terminal : trace.TerminalChecked) :
    (trace.HeadPacked ∧ trace.AllPaper functionIndex) ∨ trace.Failure := by
  induction trace with
  | single step =>
      rcases terminal with ⟨terminalWitnesses, terminalCheck⟩
      rcases
          ActiveBoundary.claimsAcceptedTerminal_implies_packedAndConstruction2_or_namedFailure
            noZeroDivisors scheme _ _ machine setup functionIndex step.input
            step.template step.witnesses step.certificate step.accepted
            terminalWitnesses terminalCheck with
        success | yRing | bad | refinement
      · exact Or.inl success
      · exact Or.inr (Or.inl yRing)
      · exact Or.inr (Or.inr (Or.inl bad))
      · exact Or.inr (Or.inr (Or.inr refinement))
  | cons head tail inductionHypothesis =>
      rcases inductionHypothesis terminal with tailSuccess | tailFailure
      · let next := tail.headStep
        rcases
            ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousPackedAndConstruction2_or_namedFailure
              noZeroDivisors scheme _ _ next.1 machine setup functionIndex
              head.input head.template head.witnesses head.certificate
              head.accepted next.2.input next.2.template next.2.witnesses
              next.2.certificate next.2.accepted tailSuccess.1 with
          success | yRing | bad | refinement
        · exact Or.inl ⟨success.1, success.2, tailSuccess.2⟩
        · exact Or.inr (Or.inl (Or.inl yRing))
        · exact Or.inr (Or.inl (Or.inr (Or.inl bad)))
        · exact Or.inr (Or.inl (Or.inr (Or.inr refinement)))
      · exact Or.inr (Or.inr tailFailure)

/-- Complete finite-trace delayed-`y_zcol` theorem. The first step is checked
as an ordinary no-pending base, the final step is closed by complete terminal
witnesses, every intermediate output is closed exactly one successor later,
and every successful step satisfies independent Construction-2 semantics. -/
theorem terminalChecked_implies_baseAndAllPaper_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (base : trace.BaseBoundary)
    (terminal : trace.TerminalChecked) :
    (trace.BaseNc ∧ trace.AllPaper functionIndex) ∨ trace.Failure := by
  rcases terminalChecked_implies_headPackedAndAllPaper_or_failure
      noZeroDivisors scheme machine setup functionIndex trace terminal with
    success | failure
  · let first := trace.headStep
    have baseNc : first.2.BaseNc :=
      ActiveBoundary.claimsAcceptedBase_of_packed_implies_ordinaryNc
        scheme _ _ machine setup first.2.input first.2.template
        first.2.witnesses first.2.certificate first.2.accepted success.1 base
    exact Or.inl ⟨baseNc, success.2⟩
  · exact Or.inr failure

/-- Strong final model-level active production composition. The packed-y-zcol
authority trace and the independent Construction-2 trace share the same
claims objects, digest-linked edges, no-pending base, and raw-matrix terminal
check. If the independent paper track fails, the result retains the already
derived base and all-packed facts. Consequently a y-ring or other paper
failure cannot hide loss of y-zcol authority. -/
theorem terminalChecked_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (base : trace.BaseBoundary)
    (terminal : trace.TerminalChecked) :
    (trace.BaseNc ∧ trace.AllPacked ∧ trace.AllPaper functionIndex) ∨
      trace.ParentOpeningFailure ∨
      (trace.BaseNc ∧ trace.AllPacked ∧ trace.Failure) := by
  rcases terminalChecked_implies_baseAndAllPacked_or_parentOpeningFailure
      noZeroDivisors scheme machine setup trace base terminal with
    packed | parentFailure
  · rcases terminalChecked_implies_baseAndAllPaper_or_namedFailure
        noZeroDivisors scheme machine setup functionIndex trace base terminal with
      paper | failure
    · exact Or.inl ⟨packed.1, packed.2, paper.2⟩
    · exact Or.inr (Or.inr ⟨packed.1, packed.2, failure⟩)
  · exact Or.inr (Or.inl parentFailure)

/-- Headline production theorem from executable acceptance only. Step-level
acceptance is `claimsCheck = true`; base and terminal are the concrete Boolean
checks packaged by `RuntimeAccepted`. The conclusion retains every packed
`y_zcol` equation even when the independent paper track reports
`yRingUnbound` or another paper failure. -/
theorem runtimeAccepted_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (accepted : trace.RuntimeAccepted) :
    (trace.BaseNc ∧ trace.AllPacked ∧ trace.AllPaper functionIndex) ∨
      trace.ParentOpeningFailure ∨
      (trace.BaseNc ∧ trace.AllPacked ∧ trace.Failure) := by
  exact
    terminalChecked_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure
      noZeroDivisors scheme machine setup functionIndex trace
      ((baseCheck_eq_true_iff trace).mp accepted.base)
      (terminalCheck_eq_true_implies trace accepted.terminalWitnesses
        accepted.terminal)

/-- Compatibility projection of the strong active production composition.
It forgets the independently proved packed trace on a paper-track failure;
new protocol composition should use the stronger theorem above.

Neither failure tree contains a generic `outputUnbound`; no constructor can
hide a y-zcol failure.  Y-ring remains solely in `Failure`, while
`ParentOpeningFailure` owns delayed projection, parent/input/key, commitment,
accumulator, and terminal-opening authority. -/
theorem terminalChecked_implies_baseAllPackedAndAllPaper_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (base : trace.BaseBoundary)
    (terminal : trace.TerminalChecked) :
    (trace.BaseNc ∧ trace.AllPacked ∧ trace.AllPaper functionIndex) ∨
      trace.ParentOpeningFailure ∨ trace.Failure := by
  rcases
      terminalChecked_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure
        noZeroDivisors scheme machine setup functionIndex trace base terminal with
    success | parentFailure | paperFailure
  · exact Or.inl success
  · exact Or.inr (Or.inl parentFailure)
  · exact Or.inr (Or.inr paperFailure.2.2)

end Trace

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace
