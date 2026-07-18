import Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2

/-!
Complete ConcretePhi81 F-prime semantics: base or recursive.

Assurance tier: model-level.

Owns: the branch-complete setup, exact two-constructor relation, branch
disjointness, and projection to the abstract HyperNova Construction-2 outer
relation.

Does not own: physical transcript/security refinement, default parent-cache
authority, executable checking, Poseidon2, Rust, R1CS, costs, necessity, or
row removal.

Emits constraints: no.

Authority boundary: base setup and active NIFS setup are separate fields.
This prevents base default caches from becoming transcript authority and
prevents active verifier callbacks from defining the base relation. Full
projection requires only the explicit active selected-NIFS refinement.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.setup.base_default` | rich defaults erase to the configured paper defaults | setup boundary | `Setup.default` |
| `fprime.setup.active_nifs` | construct selected ConcretePhi81 contexts | setup boundary | `Setup.active` |
| `fprime.branch.base` | accept the three-check base relation | checked/computed | `Holds.base` |
| `fprime.branch.recursive` | accept the six-family active relation | checked/computed | `Holds.recursive` |
| `fprime.branch.disjoint` | `i = 0` and `i > 0` cannot both hold | derived | `base_recursive_disjoint` |
| `fprime.refinement.outer` | both branches project to abstract Construction 2 | model-level theorem | `sound` |
| `fprime.refinement.outer.concrete_nifs` | install the independent selected public ConcretePhi81 edge | model-level theorem | `sound_selectedNifs` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Semantics

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Protocol.FPrime.ConcretePhi81.Outer

universe uOuterKey uAppState uWitness uDigest uTranscriptState

/-- Branch-complete setup with no shared mutable authority surface. -/
structure Setup
    (OuterKey : Type uOuterKey)
    (Digest : Type uDigest)
    (AppState : Type uAppState)
    (Witness : Type uWitness)
    (TranscriptState : Type uTranscriptState)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows slotCount : Nat) where
  machine : Machine OuterKey Digest AppState Witness shape
    publicRingColumns publicFits verifierRows slotCount
  active : ActiveSemantics.Setup OuterKey AppState Witness TranscriptState
    shape publicRingColumns publicFits verifierRows slotCount
  default : BaseSemantics.DefaultCarrier machine

/-- The full augmented relation is exactly the paper base case or the
independent ConcretePhi81 recursive case. -/
inductive Holds
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup : Setup OuterKey Digest AppState Witness TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount)
    (output : Output Digest AppState shape publicRingColumns publicFits
      verifierRows slotCount) : Prop where
  | base (accepted : BaseSemantics.Holds setup.machine setup.default
      functionIndex input output)
  | recursive (accepted : ActiveSemantics.Holds setup.active setup.machine
      functionIndex input output)

/-- The iteration split makes the two constructors mutually exclusive for
the same input and output. -/
theorem base_recursive_disjoint
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup : Setup OuterKey Digest AppState Witness TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount}
    {output : Output Digest AppState shape publicRingColumns publicFits
      verifierRows slotCount}
    (base : BaseSemantics.Holds setup.machine setup.default functionIndex input
      output)
    (recursive : ActiveSemantics.Holds setup.active setup.machine functionIndex
      input output) : False := by
  rcases base with ⟨baseObligations, _⟩
  rcases recursive with ⟨_selected, _next, recursiveObligations, _⟩
  have iterationZero : input.iteration = 0 :=
    baseObligations.iterationZero
  have iterationPositive : 0 < input.iteration :=
    recursiveObligations.iterationPositive
  omega

/-- Complete outer soundness. The only non-structural premise is the explicit
independent selected-NIFS refinement required by the recursive branch. -/
theorem sound
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup : Setup OuterKey Digest AppState Witness TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount}
    {family : Paper.Construction2.Family OuterKey
      (RelationStructure shape publicRingColumns publicFits)
      (RelationPublicInput shape publicRingColumns publicFits)
      (RelationPoint shape publicRingColumns publicFits)
      Phi81Relation.Evaluation (CommitmentValue verifierRows)
      productionGlobalParams slotCount}
    {functionIndex : Fin slotCount}
    {input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount}
    {output : Output Digest AppState shape publicRingColumns publicFits
      verifierRows slotCount}
    (refinement : ActiveSemantics.Construction2.Refinement setup.active family)
    (accepted : Holds setup functionIndex input output) :
    Paper.Construction2.Holds family setup.machine functionIndex input.toPaper
      output.toPaper := by
  cases accepted with
  | base baseAccepted =>
      exact Paper.Construction2.Holds.base
        (BaseSemantics.sound baseAccepted)
  | recursive recursiveAccepted =>
      exact Paper.Construction2.Holds.recursive
        (ActiveSemantics.Construction2.sound refinement recursiveAccepted)

/-- Full model-level soundness with the independent selected public
ConcretePhi81 NIFS edge installed. This theorem does not claim transcript
security or executable/R1CS refinement. -/
theorem sound_selectedNifs
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup : Setup OuterKey Digest AppState Witness TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount}
    {output : Output Digest AppState shape publicRingColumns publicFits
      verifierRows slotCount}
    (accepted : Holds setup functionIndex input output) :
    Paper.Construction2.Holds
      (SelectedNifsSemantics.family
        (ActiveSemantics.Construction2.selectedNifsSetup setup.active))
      setup.machine functionIndex input.toPaper output.toPaper :=
  sound (ActiveSemantics.Construction2.selectedNifsRefinement setup.active)
    accepted

end Nightstream.Protocol.FPrime.ConcretePhi81.Semantics
