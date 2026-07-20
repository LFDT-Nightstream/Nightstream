import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionContext
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold

/-!
Semantic erasure of the production-only delayed pending carrier.

Assurance tier: model-level.

Owns: exact transport of the independent `SemanticFold.Holds` relation
between the production context carrying an optional delayed packed value and
the ordinary fixed-one context. The proof reconstructs every semantic witness
and realization from the context projections actually used by the relation.

Does not own: erasure of the physical pending check, certificate transport,
active evaluator acceptance, state continuity, digest or commitment authority,
Rust/R1CS refinement, generated rows, costs, or row-removal permission.

Emits constraints: none; correspondence theorem only.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.pending_erasure` | Production pending data does not alter the independent fold semantics | derived from exact context projections |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PendingErasure

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc

universe uOuterKey uAppState uWitness uTranscriptState

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

private theorem runningAuthority_iff_withoutPending
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows) :
    RunningAuthority.Accepted (ProductionContext.full setup input) ↔
      RunningAuthority.Accepted
        (FixedOneCanonical.nifsContext setup input.fixedOne).materialize := by
  constructor
  · intro accepted
    cases accepted with
    | bootstrap mode parentAbsent =>
        exact .bootstrap mode (by
          simpa only [ProductionContext.full_runningParent] using parentAbsent)
    | active bound =>
        exact .active {
          active := bound.active
          parent := bound.parent
          parentBound := by
            simpa only [ProductionContext.full_runningParent] using
              bound.parentBound
          piDec := by
            simpa only [RunningAuthority.attempt, RunningAuthority.children,
              ProductionContext.full_key, ProductionContext.full_input] using
              bound.piDec
        }
  · intro accepted
    cases accepted with
    | bootstrap mode parentAbsent =>
        exact .bootstrap mode (by
          simpa only [ProductionContext.full_runningParent] using parentAbsent)
    | active bound =>
        exact .active {
          active := bound.active
          parent := bound.parent
          parentBound := by
            simpa only [ProductionContext.full_runningParent] using
              bound.parentBound
          piDec := by
            simpa only [RunningAuthority.attempt, RunningAuthority.children,
              ProductionContext.full_key, ProductionContext.full_input] using
              bound.piDec
        }

/-- The production-only optional delayed value is absent from the independent
fold semantics. This is an exact semantic transport, not permission to erase
the physical pending value or its successor/terminal checks. -/
theorem semanticFoldHolds_iff_withoutPending
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (parent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
    (children : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    SemanticFold.Holds (ProductionContext.full setup input) data parent
        children ↔
      SemanticFold.Holds
        (FixedOneCanonical.nifsContext setup input.fixedOne).materialize
        data parent children := by
  constructor
  · rintro ⟨witness, realization⟩
    let erasedWitness :
        SemanticFold.Witness
          (FixedOneCanonical.nifsContext setup input.fixedOne).materialize := {
      point := witness.point
      challenges := witness.challenges
    }
    refine ⟨erasedWitness, {
      paper := realization.paper
      input := ?_
      running := ?_
      challengesValid := ?_
      parent_eq := ?_
      children_eq := ?_
    }⟩
    · exact {
        publicInput := by
          simpa only [SemanticFold.PublicInputBound,
            ProductionContext.full_piCcsInput] using
              realization.input.publicInput
        sources := by
          simpa only [SemanticFold.InputBound, ProductionContext.full_key,
            ProductionContext.full_alignment, ProductionContext.full_input]
            using realization.input.sources
      }
    · exact
        (runningAuthority_iff_withoutPending setup input).mp
          realization.running
    · intro source
      simpa only [erasedWitness, ProductionContext.full_key] using
        realization.challengesValid source
    · simpa only [SemanticFold.parentOf, SemanticFold.outputs,
        SemanticFold.systemOf, SemanticFold.assignments, erasedWitness,
        ProductionContext.full_key, ProductionContext.full_input,
        ProductionContext.full_alignment] using realization.parent_eq
    · simpa only [SemanticFold.childrenOf, SemanticFold.parentOf,
        SemanticFold.combinedAssignment, SemanticFold.outputs,
        SemanticFold.systemOf, SemanticFold.assignments, erasedWitness,
        ProductionContext.full_key, ProductionContext.full_input,
        ProductionContext.full_alignment] using realization.children_eq
  · rintro ⟨witness, realization⟩
    let restoredWitness :
        SemanticFold.Witness (ProductionContext.full setup input) := {
      point := witness.point
      challenges := witness.challenges
    }
    refine ⟨restoredWitness, {
      paper := realization.paper
      input := ?_
      running := ?_
      challengesValid := ?_
      parent_eq := ?_
      children_eq := ?_
    }⟩
    · exact {
        publicInput := by
          simpa only [SemanticFold.PublicInputBound,
            ProductionContext.full_piCcsInput] using
              realization.input.publicInput
        sources := by
          simpa only [SemanticFold.InputBound, ProductionContext.full_key,
            ProductionContext.full_alignment, ProductionContext.full_input]
            using realization.input.sources
      }
    · exact
        (runningAuthority_iff_withoutPending setup input).mpr
          realization.running
    · intro source
      simpa only [restoredWitness, ProductionContext.full_key] using
        realization.challengesValid source
    · simpa only [SemanticFold.parentOf, SemanticFold.outputs,
        SemanticFold.systemOf, SemanticFold.assignments, restoredWitness,
        ProductionContext.full_key, ProductionContext.full_input,
        ProductionContext.full_alignment] using realization.parent_eq
    · simpa only [SemanticFold.childrenOf, SemanticFold.parentOf,
        SemanticFold.combinedAssignment, SemanticFold.outputs,
        SemanticFold.systemOf, SemanticFold.assignments, restoredWitness,
        ProductionContext.full_key, ProductionContext.full_input,
        ProductionContext.full_alignment] using realization.children_eq

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PendingErasure
