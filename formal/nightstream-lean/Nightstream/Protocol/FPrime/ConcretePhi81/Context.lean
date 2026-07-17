import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Evaluator

/-!
Outer-to-NIFS context construction for the fixed active ConcretePhi81 fold.

Protocol: HyperNova Construction 2 over the concrete SuperNeo NIFS.
Phase: selected F-prime accumulator and fresh claim → complete NIFS context.
Constraint family: authority-preserving context construction only; this file
emits no rows.

Owns: the exact active invocation carrier; construction of the one-fresh plus
fourteen-running public source product; a static verifier template; and the
only constructor for a ConcretePhi81 context consumed by the outer F-prime
verifier.

Does not own: program-counter selection, prior/output hashes, application
semantics, raw NIFS checking, hidden source assignments, Poseidon2
instantiation, Rust, R1CS, rows, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: callers do not supply a completed
`ConcretePhi81.Context`. `Template.build` installs the exact fresh statement,
the exact fourteen children, and `some` of the exact checked parent from the
selected running accumulator. The public Split-NC input and transcript prefix
remain explicit verifier-owned invocation fields; later outer refinement must
show they are derived from the F-prime message schedule.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.zero_arity.branch.recursive.nifs.input.fresh` | exactly one fresh CCS statement enters the fold | direct dataflow | `Invocation.sourceProduct` |
| `fprime.zero_arity.branch.recursive.nifs.input.running` | all fourteen selected children enter in canonical order | direct dataflow | `Invocation.sourceProduct` |
| `fprime.zero_arity.branch.recursive.nifs.running_parent` | the selected derived parent is present as the checked-parent carrier | computed | `Template.build` |
| `fprime.zero_arity.branch.recursive.nifs.pi_ccs.public` | retain the exact verifier-owned Split-NC public input | direct dataflow | `Template.build` |
| `fprime.zero_arity.branch.recursive.nifs.transcript.prefix` | retain the exact outer-derived prior transcript state | direct dataflow | `Template.build` |
| `fprime.zero_arity.branch.recursive.nifs.setup` | key, alignment, domain, schedule, sampler, and profile are template-owned | verifier setup | `Template` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Context

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Public values from one selected active F-prime slot.

The semantic source assignments are deliberately absent. They belong to the
independent `ConcretePhi81.SemanticInput` bridge, not to physical context
construction. -/
structure Invocation
    (shape : SemanticShape)
    (State : Type uState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  fresh :
    Phi81Relation.CCSStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)
  running :
    FixedActive.FoldResult shape publicRingColumns publicFits verifierRows
  piCcsInput : PiCCS.SplitNc.Verifier.PublicInput shape
  priorState : State

namespace Invocation

/-- Canonical physical source product: one fresh statement followed by the
complete selected child accumulator. -/
def sourceProduct
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (invocation :
      Invocation shape State publicRingColumns publicFits verifierRows) :
    SourceProduct shape publicRingColumns publicFits
      (CommitmentValue verifierRows) productionGlobalParams
      FixedActive.arity where
  fresh := fun _ => invocation.fresh
  running := fun child => invocation.running.children child

@[simp] theorem sourceProduct_fresh
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (invocation :
      Invocation shape State publicRingColumns publicFits verifierRows)
    (source : Fin FixedActive.arity.freshCount) :
    (invocation.sourceProduct.fresh source) = invocation.fresh := rfl

@[simp] theorem sourceProduct_running
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (invocation :
      Invocation shape State publicRingColumns publicFits verifierRows)
    (child :
      Fin (FixedActive.arity.mode.count productionGlobalParams)) :
    invocation.sourceProduct.running child =
      invocation.running.children child := rfl

end Invocation

/-- Static verifier setup shared by every active invocation at one concrete
relation shape. Dynamic claim and transcript-prefix values are intentionally
absent. -/
structure Template
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (State : Type uState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  covers : domain.Covers shape
  key : ConcretePhi81.VerifierKey
    shape publicRingColumns publicFits verifierRows
  alignment :
    SourceAlignment shape productionGlobalParams FixedActive.arity
  piCcsSchedule :
    PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.Schedule
      (ConcretePhi81.VerifierKey
        shape publicRingColumns publicFits verifierRows)
      (ConcretePhi81.StatementInput
        shape publicRingColumns publicFits verifierRows FixedActive.arity)
      shape domain State
  piRlcMachine :
    Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.Machine State
  profile : PiCCS.SplitNc.Verifier.Polynomial.Fe.SupportedProfile shape domain
  challengeSetSize : Nat

namespace Template

/-- Construct the sole ConcretePhi81 context for one selected active
invocation. Every dynamic authority field is installed from `invocation`. -/
def build
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (template :
      Template shape domain State publicRingColumns publicFits verifierRows)
    (invocation :
      Invocation shape State publicRingColumns publicFits verifierRows) :
    FixedActive.Context
      shape domain State publicRingColumns publicFits verifierRows where
  covers := template.covers
  key := template.key
  alignment := template.alignment
  input := invocation.sourceProduct
  runningParent := some invocation.running.parent
  piCcsInput := invocation.piCcsInput
  priorState := invocation.priorState
  piCcsSchedule := template.piCcsSchedule
  piRlcMachine := template.piRlcMachine
  profile := template.profile
  challengeSetSize := template.challengeSetSize

@[simp] theorem build_input
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (template :
      Template shape domain State publicRingColumns publicFits verifierRows)
    (invocation :
      Invocation shape State publicRingColumns publicFits verifierRows) :
    (template.build invocation).input = invocation.sourceProduct := rfl

@[simp] theorem build_runningParent
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (template :
      Template shape domain State publicRingColumns publicFits verifierRows)
    (invocation :
      Invocation shape State publicRingColumns publicFits verifierRows) :
    (template.build invocation).runningParent =
      some invocation.running.parent := rfl

@[simp] theorem build_piCcsInput
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (template :
      Template shape domain State publicRingColumns publicFits verifierRows)
    (invocation :
      Invocation shape State publicRingColumns publicFits verifierRows) :
    (template.build invocation).piCcsInput = invocation.piCcsInput := rfl

@[simp] theorem build_priorState
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (template :
      Template shape domain State publicRingColumns publicFits verifierRows)
    (invocation :
      Invocation shape State publicRingColumns publicFits verifierRows) :
    (template.build invocation).priorState = invocation.priorState := rfl

end Template

end Nightstream.Protocol.FPrime.ConcretePhi81.Context
