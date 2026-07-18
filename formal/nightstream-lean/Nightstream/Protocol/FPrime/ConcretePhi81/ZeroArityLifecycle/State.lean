import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result

/-!
Typed state carriers for the concrete zero-running F-prime lifecycle.

Protocol: candidate F-prime scheduling over the concrete Phi81 NIFS.
Phase: initial, primed, and running state.
Constraint family: semantic state shape only; this file emits no rows.

Owns: the one-claim carrier, complete NIFS result payload, and three lifecycle
phases that make an empty active accumulator unrepresentable.

Does not own: evidence that a raw state is valid, NIFS context construction,
transitions, transcript provenance, HyperNova refinement, Rust/R1CS
refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: `State` is data, not proof of provenance. A raw `running`
value becomes usable only together with the evidence defined by the transition
module.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.zero_arity.state.initial` | no prior latest claim or accumulator payload exists | typed | `State.initial` |
| `fprime.zero_arity.state.primed` | exactly one prior claim exists and no accumulator field exists | typed | `State.primed` |
| `fprime.zero_arity.state.running` | one complete parent-and-children result plus one delayed claim exists | typed data | `State.running` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-- One foldable F-prime claim in the fixed concrete relation. -/
abbrev Fresh
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  Phi81Relation.CCSStatement
    (RelationShape shape publicRingColumns publicFits)
    (CommitmentValue verifierRows)

/-- Complete semantic NIFS result payload retained between active folds. -/
abbrev Accumulator
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  Result.FoldResult shape publicRingColumns publicFits verifierRows

/-- Lifecycle phases with invalid empty-active combinations unrepresentable. -/
inductive State
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  | initial
  | primed
      (currentLatest :
        Fresh shape publicRingColumns publicFits verifierRows)
  | running
      (accumulator :
        Accumulator shape publicRingColumns publicFits verifierRows)
      (currentLatest :
        Fresh shape publicRingColumns publicFits verifierRows)

end Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle
