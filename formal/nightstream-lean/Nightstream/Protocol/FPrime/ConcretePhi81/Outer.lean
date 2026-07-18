import Nightstream.Protocol.FPrime.ConcretePhi81.Context
import Nightstream.Protocol.FPrime.Paper

/-!
Branch-neutral outer carrier for HyperNova Construction 2 over ConcretePhi81.

Owns: the concrete relation aliases, complete parent-and-children running
slots, nondeterministic outer input, canonical rich output, and exact
projections to the paper carrier.

Does not own: base or recursive acceptance, selected NIFS semantics,
certificate checking, transcript replay, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: `Running.toPaper` erases only the cached parent. The fresh
statement, every child, the verifier key, iteration, states, prior counter,
and witness are preserved exactly. Branch relations must separately justify
whether a parent is absent, authoritative, or computed.

| Stage path | Mathematical object | Authority class | Lean owner |
|---|---|---|---|
| `fprime.outer.carrier.slot` | one cached parent plus all `k` children | typed carrier | `Slot` |
| `fprime.outer.carrier.running` | one complete accumulator per HyperNova slot | typed carrier | `Running` |
| `fprime.outer.carrier.input` | exact Construction-2 nondeterministic input | typed carrier | `Input` |
| `fprime.outer.carrier.output` | next state, running product, counter, and digest | typed carrier | `Output` |
| `fprime.outer.projection.running` | erase only cached parents | direct dataflow | `Running.toPaper` |
| `fprime.outer.projection.input` | preserve every paper input field exactly | direct dataflow | `Input.toPaper` |
| `fprime.outer.projection.output` | preserve every paper output field exactly | direct dataflow | `Output.toPaper` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Outer

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uOuterKey uAppState uWitness uDigest

/-- Concrete relation structure shared by the outer and selected NIFS views. -/
abbrev RelationStructure
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Phi81Relation.Structure
    (RelationShape shape publicRingColumns publicFits)

/-- Concrete relation public input shared by CCS and CE statements. -/
abbrev RelationPublicInput
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Phi81Relation.PublicInput
    (RelationShape shape publicRingColumns publicFits)

/-- Concrete relation evaluation point carried by each CE statement. -/
abbrev RelationPoint
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Phi81Relation.Point
    (RelationShape shape publicRingColumns publicFits)

/-- One complete selected-slot accumulator: derived parent plus all children. -/
abbrev Slot
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  FixedActive.FoldResult
    shape publicRingColumns publicFits verifierRows

/-- The outer product of independently selectable fixed-active accumulators. -/
abbrev Running
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows slotCount : Nat) :=
  Fin slotCount ->
    Slot shape publicRingColumns publicFits verifierRows

namespace Running

/-- Project away derived parent caches to the exact paper running product. -/
def toPaper
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (running :
      Running shape publicRingColumns publicFits verifierRows slotCount) :
    Paper.RunningProduct
      (RelationStructure shape publicRingColumns publicFits)
      (RelationPublicInput shape publicRingColumns publicFits)
      (RelationPoint shape publicRingColumns publicFits)
      Phi81Relation.Evaluation (CommitmentValue verifierRows)
      productionGlobalParams slotCount :=
  fun slot child => (running slot).children child

@[simp] theorem toPaper_apply
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (running :
      Running shape publicRingColumns publicFits verifierRows slotCount)
    (slot : Fin slotCount)
    (child : Fin productionGlobalParams.k) :
    running.toPaper slot child = (running slot).children child := rfl

end Running

/-- Verifier/advice input to one ConcretePhi81 augmented-function invocation. -/
structure Input
    (OuterKey : Type uOuterKey)
    (AppState : Type uAppState)
    (Witness : Type uWitness)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows slotCount : Nat) where
  verifierKey : OuterKey
  iteration : Nat
  z0 : AppState
  zi : AppState
  running :
    Running shape publicRingColumns publicFits verifierRows slotCount
  fresh :
    Phi81Relation.CCSStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)
  priorPc : Nat
  witness : Witness

namespace Input

/-- Exact projection to the independent paper carrier. -/
def toPaper
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    Paper.Input OuterKey AppState Witness
      (RelationStructure shape publicRingColumns publicFits)
      (RelationPublicInput shape publicRingColumns publicFits)
      (RelationPoint shape publicRingColumns publicFits)
      Phi81Relation.Evaluation (CommitmentValue verifierRows)
      productionGlobalParams slotCount where
  verifierKey := input.verifierKey
  iteration := input.iteration
  z0 := input.z0
  zi := input.zi
  running := input.running.toPaper
  fresh := input.fresh
  priorPc := input.priorPc
  witness := input.witness

@[simp] theorem toPaper_running
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    input.toPaper.running = input.running.toPaper := rfl

@[simp] theorem toPaper_fresh
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    input.toPaper.fresh = input.fresh := rfl

end Input

/-- Rich output shared by the base and recursive ConcretePhi81 branches. -/
structure Output
    (Digest : Type uDigest)
    (AppState : Type uAppState)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows slotCount : Nat) where
  zNext : AppState
  runningNext :
    Running shape publicRingColumns publicFits verifierRows slotCount
  pcNext : Paper.ProgramCounter slotCount
  x : Digest

namespace Output

/-- Project away parent caches to the exact paper output carrier. -/
def toPaper
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount) :
    Paper.Output Digest AppState
      (RelationStructure shape publicRingColumns publicFits)
      (RelationPublicInput shape publicRingColumns publicFits)
      (RelationPoint shape publicRingColumns publicFits)
      Phi81Relation.Evaluation (CommitmentValue verifierRows)
      productionGlobalParams slotCount where
  zNext := output.zNext
  runningNext := output.runningNext.toPaper
  pcNext := output.pcNext
  x := output.x

end Output

/-- The paper machine specialized to the concrete Phi81 statement carrier. -/
abbrev Machine
    (OuterKey : Type uOuterKey)
    (Digest : Type uDigest)
    (AppState : Type uAppState)
    (Witness : Type uWitness)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows slotCount : Nat) :=
  Paper.Machine OuterKey Digest AppState Witness
    (RelationStructure shape publicRingColumns publicFits)
    (RelationPublicInput shape publicRingColumns publicFits)
    (RelationPoint shape publicRingColumns publicFits)
    Phi81Relation.Evaluation (CommitmentValue verifierRows)
    productionGlobalParams slotCount

end Nightstream.Protocol.FPrime.ConcretePhi81.Outer
