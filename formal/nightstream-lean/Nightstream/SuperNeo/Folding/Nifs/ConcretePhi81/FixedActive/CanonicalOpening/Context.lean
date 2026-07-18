import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.SourceValidated
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Context

/-!
Opening-derived fixed-active NIFS context.

Assurance tier: model-level carrier refinement.

Owns: one point-plus-complete-opening payload; deterministic materialization
of the incoming combined parent and all fourteen fresh running children; and
exact projection into the existing canonical fixed-active NIFS context.

Does not own: child-source validity, transcript replay, prior-link hashing,
privacy of an opening-derived handle, Rust/R1CS decoding, physical rows,
costs, or row removal.

Emits constraints: no.

Authority boundary: the caller no longer supplies a parent commitment,
public input, evaluations, or any child statement. The verifier-owned key and
system compute all of them from one complete typed opening and one typed
point. This module only constructs values; semantic authority is established
separately from the NIFS source-validation facts.

| Stage path | Mathematical object | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_active.opening.input` | complete parent opening and common point | private typed payload | `OpeningPayload` |
| `nifs.fixed_active.opening.parent` | combined parent statement | computed | `OpeningPayload.parent`, `parentPayload_materialize` |
| `nifs.fixed_active.opening.children` | fourteen deterministic signed-binary children | computed | `OpeningPayload.children`, `runningPayload_materialize` |
| `nifs.fixed_active.opening.context` | exact existing fixed-active context | derived refinement | `Context.materialize` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Smallest typed value carrier currently needed to reconstruct one incoming
canonical PiDEC family. -/
structure OpeningPayload
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) where
  point : Point (RelationShape shape publicRingColumns publicFits)
  assignment : Assignment (RelationShape shape publicRingColumns publicFits)

namespace OpeningPayload

def carrier
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (payload : OpeningPayload shape publicRingColumns publicFits) :=
  CanonicalParentVerifier.SourceValidated.computedCarrier
    key payload.point payload.assignment

def parent
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (system : Structure (RelationShape shape publicRingColumns publicFits))
    (payload : OpeningPayload shape publicRingColumns publicFits) :
    CEStatement (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) :=
  CanonicalParentVerifier.SourceValidated.computedParent
    key system payload.point payload.assignment

def children
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (system : Structure (RelationShape shape publicRingColumns publicFits))
    (payload : OpeningPayload shape publicRingColumns publicFits) :
    Fin productionGlobalParams.k ->
      CEStatement (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows) :=
  CanonicalParentVerifier.SourceValidated.computedChildren
    key system payload.point payload.assignment

def parentPayload
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (system : Structure (RelationShape shape publicRingColumns publicFits))
    (payload : OpeningPayload shape publicRingColumns publicFits) :
    Canonical.ParentPayload shape publicRingColumns publicFits verifierRows :=
  let statement := payload.parent key system
  {
    commitment := statement.commitment
    publicInput := statement.publicInput
    point := statement.point
    evaluations := statement.evaluations
  }

def runningPayload
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (system : Structure (RelationShape shape publicRingColumns publicFits))
    (payload : OpeningPayload shape publicRingColumns publicFits)
    (child : Fin productionGlobalParams.k) :
    Canonical.RunningPayload shape publicRingColumns publicFits verifierRows :=
  let statement := payload.children key system child
  {
    commitment := statement.commitment
    publicInput := statement.publicInput
    point := statement.point
    evaluations := statement.evaluations
  }

@[simp] theorem parentPayload_materialize
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (system : Structure (RelationShape shape publicRingColumns publicFits))
    (payload : OpeningPayload shape publicRingColumns publicFits) :
    (payload.parentPayload key system).materialize system =
      payload.parent key system := by
  rfl

@[simp] theorem runningPayload_materialize
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (system : Structure (RelationShape shape publicRingColumns publicFits))
    (payload : OpeningPayload shape publicRingColumns publicFits)
    (child : Fin productionGlobalParams.k) :
    (payload.runningPayload key system child).materialize system =
      payload.children key system child := by
  rfl

end OpeningPayload

/-- Dynamic statement input after the complete incoming PiDEC family is
reduced to one opening-derived payload. -/
structure Input
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  system : Structure (RelationShape shape publicRingColumns publicFits)
  fresh : Canonical.FreshPayload
    shape publicRingColumns publicFits verifierRows
  opening : OpeningPayload shape publicRingColumns publicFits

namespace Input

def materialize
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (input : Input shape publicRingColumns publicFits verifierRows) :
    Canonical.Input shape publicRingColumns publicFits verifierRows where
  system := input.system
  fresh := input.fresh
  running := input.opening.runningPayload key input.system
  parent := input.opening.parentPayload key input.system

@[simp] theorem materialize_parent
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (input : Input shape publicRingColumns publicFits verifierRows) :
    ((input.materialize key).parent.materialize input.system) =
      input.opening.parent key input.system := by
  rfl

@[simp] theorem materialize_running
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (input : Input shape publicRingColumns publicFits verifierRows)
    (child : Fin productionGlobalParams.k) :
    ((input.materialize key).running child).materialize input.system =
      input.opening.children key input.system child := by
  rfl

end Input

/-- Full verifier context with the opening-derived dynamic statement input. -/
structure Context
    (shape : SemanticShape)
    (State : Type uState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  covers : PiCcsDomains.production.nc.Covers shape
  key : VerifierKey shape publicRingColumns publicFits verifierRows
  alignment : SourceAlignment shape productionGlobalParams arity
  input : Input shape publicRingColumns publicFits verifierRows
  piCcsInput : PiCCS.SplitNc.Verifier.PublicInput shape
  priorState : State
  piCcsSchedule :
    PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.Schedule
      (VerifierKey shape publicRingColumns publicFits verifierRows)
      (StatementInput shape publicRingColumns publicFits verifierRows arity)
      shape PiCcsDomains.production State
  piRlcMachine :
    Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.Machine State
  profile : PiCCS.SplitNc.Verifier.Polynomial.Fe.SupportedProfile shape
    PiCcsDomains.production.fe
  challengeSetSize : Nat

namespace Context

/-- Exact refinement into the already-audited canonical fixed-active context. -/
def materialize
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : Context shape State publicRingColumns publicFits verifierRows) :
    Canonical.Context shape State publicRingColumns publicFits verifierRows where
  covers := context.covers
  key := context.key
  alignment := context.alignment
  input := context.input.materialize context.key
  piCcsInput := context.piCcsInput
  priorState := context.priorState
  piCcsSchedule := context.piCcsSchedule
  piRlcMachine := context.piRlcMachine
  profile := context.profile
  challengeSetSize := context.challengeSetSize

/-- Exact existing independent NIFS context reached by both carrier layers. -/
def full
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : Context shape State publicRingColumns publicFits verifierRows) :
    FixedActive.Context shape State publicRingColumns publicFits verifierRows :=
  context.materialize.materialize

@[simp] theorem full_parent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : Context shape State publicRingColumns publicFits verifierRows) :
    context.full.runningParent =
      some (context.input.opening.parent context.key context.input.system) := by
  rfl

@[simp] theorem full_running
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : Context shape State publicRingColumns publicFits verifierRows)
    (child : Fin productionGlobalParams.k) :
    context.full.input.running child =
      context.input.opening.children context.key context.input.system child := by
  rfl

end Context

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening
