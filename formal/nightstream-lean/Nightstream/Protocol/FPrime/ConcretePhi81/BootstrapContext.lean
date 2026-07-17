import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap

/-!
Outer-to-NIFS context construction for the zero-running ConcretePhi81
bootstrap fold.

Protocol: the production first recursive F-prime fold.
Phase: one fresh claim and no incoming accumulator -> complete bootstrap NIFS
context.
Constraint family: authority-preserving context construction only; this file
emits no rows.

Owns: the exact bootstrap invocation carrier; construction of the one-fresh,
zero-running source product; a static verifier template at bootstrap arity;
and the only constructor for the bootstrap context consumed by later outer
semantics.

Does not own: equivalence to HyperNova's default-instance vector, the outer
base/bootstrap/active lifecycle, program dispatch, prior/output hashes,
application semantics, raw NIFS checking, hidden source assignments,
Poseidon2 instantiation, Rust/R1CS refinement, rows, costs, necessity, or row
removal.

Emits constraints: no.

Authority boundary: callers supply one fresh public statement, the exact
Split-NC public input, and the prior transcript state. They cannot supply a
running product or parent. `Template.build` constructs the empty running
product and installs `Option.none` as the complete incoming-parent carrier.
This module does not claim that zero-running bootstrap is equivalent to
folding HyperNova's `k` default instances.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.zero_arity.branch.bootstrap.nifs.input.fresh` | exactly one fresh CCS statement enters the fold | direct dataflow | `Invocation.sourceProduct` |
| `fprime.zero_arity.branch.bootstrap.nifs.input.running` | the running source index is empty | typed/computed | `Invocation.noRunningSource` |
| `fprime.zero_arity.branch.bootstrap.nifs.running_parent` | the complete incoming-parent carrier is exactly absent | computed | `Template.build`, `Template.build_runningAuthority` |
| `fprime.zero_arity.branch.bootstrap.nifs.pi_ccs.public` | retain the exact verifier-owned Split-NC public input | direct dataflow | `Template.build` |
| `fprime.zero_arity.branch.bootstrap.nifs.transcript.prefix` | retain the exact outer-derived prior transcript state | direct dataflow | `Template.build` |
| `fprime.zero_arity.branch.bootstrap.nifs.setup` | key, alignment, domain, schedule, sampler, and profile are template-owned | verifier setup | `Template` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext

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

/-- Public values for the production zero-running bootstrap invocation.

There is intentionally no running-accumulator field and no optional parent
field. Hidden source assignments remain outside physical context
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
  piCcsInput : PiCCS.SplitNc.Verifier.PublicInput shape
  priorState : State

namespace Invocation

/-- Canonical physical bootstrap source product: one fresh statement and an
empty running family. -/
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
      FixedBootstrap.arity where
  fresh := fun _ => invocation.fresh
  running := fun source => Fin.elim0 source

@[simp] theorem sourceProduct_fresh
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (invocation :
      Invocation shape State publicRingColumns publicFits verifierRows)
    (source : Fin FixedBootstrap.arity.freshCount) :
    invocation.sourceProduct.fresh source = invocation.fresh := rfl

/-- The bootstrap running-source index has no inhabitants. -/
theorem noRunningSource
    (source :
      Fin (FixedBootstrap.arity.mode.count productionGlobalParams)) :
    False :=
  Fin.elim0 source

end Invocation

/-- Static verifier setup shared by every bootstrap invocation at one
concrete relation shape. Dynamic claim and transcript-prefix values are
intentionally absent. -/
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
    SourceAlignment shape productionGlobalParams FixedBootstrap.arity
  piCcsSchedule :
    PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.Schedule
      (ConcretePhi81.VerifierKey
        shape publicRingColumns publicFits verifierRows)
      (ConcretePhi81.StatementInput
        shape publicRingColumns publicFits verifierRows FixedBootstrap.arity)
      shape domain State
  piRlcMachine :
    Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.Machine State
  profile : PiCCS.SplitNc.Verifier.Polynomial.Fe.SupportedProfile shape domain
  challengeSetSize : Nat

namespace Template

/-- Construct the sole ConcretePhi81 context for one bootstrap invocation.
The empty running product and absent parent are construction facts. -/
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
    FixedBootstrap.Context
      shape domain State publicRingColumns publicFits verifierRows where
  covers := template.covers
  key := template.key
  alignment := template.alignment
  input := invocation.sourceProduct
  runningParent := none
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
    (template.build invocation).runningParent = none := rfl

/-- The constructed bootstrap context satisfies the exact incoming-authority
contract without a caller-supplied proof or digest convention. -/
theorem build_runningAuthority
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
    RunningAuthority.Accepted (template.build invocation) :=
  FixedBootstrap.runningAuthority_of_parentAbsent
    (template.build invocation) rfl

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

end Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext
