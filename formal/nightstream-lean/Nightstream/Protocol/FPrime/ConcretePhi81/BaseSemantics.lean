import Nightstream.Protocol.FPrime.ConcretePhi81.Outer
import Nightstream.Protocol.FPrime.Paper.Output

/-!
ConcretePhi81 base semantics for HyperNova Construction 2.

Assurance tier: model-level.

Owns: the rich default-running lift, the three checked base obligations, the
canonical rich output, and projection to `Paper.BaseHolds`.

Does not own: NIFS, bootstrap folding, validity of default child openings,
authority of cached parents, hash injectivity, Poseidon2, Rust, R1CS, costs,
necessity, or row removal.

Emits constraints: no.

Authority boundary: the paper default vector contains only `k` public children
per slot. `DefaultCarrier` supplies the richer ConcretePhi81 parent caches and
proves only that erasing those caches yields the configured paper vector.
Parent-cache authority is a setup/lifecycle obligation, not a base-step check.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.base.iteration` | base execution has `i = 0` | checked | `Obligations.iterationZero` |
| `fprime.base.initial_state` | `z_0 = z_i` | checked | `Obligations.initialState` |
| `fprime.base.dispatch` | control selects this fixed `F_j` | checked | `Obligations.dispatch` |
| `fprime.base.output.default_children` | rich defaults erase to the paper default vector | setup-owned dataflow | `DefaultCarrier.projection` |
| `fprime.base.output.application` | compute `pcNext` and `zNext` | computed | `outputOf` |
| `fprime.base.output.hash` | hash the exact typed next preimage | computed | `outputOf` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Protocol.FPrime.ConcretePhi81.Outer

universe uOuterKey uAppState uWitness uDigest

/-- Setup-owned rich lift of the paper default child vector. -/
structure DefaultCarrier
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount) where
  running :
    Running shape publicRingColumns publicFits verifierRows slotCount
  projection : running.toPaper = machine.defaultRunning

/-- Canonical rich base output. Cached parents are retained locally while the
paper hash sees only the projected child vector. -/
def outputOf
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount)
    (default : DefaultCarrier machine)
    (input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount) :
    Output Digest AppState shape publicRingColumns publicFits verifierRows
      slotCount :=
  let paperOutput :=
    Paper.derivedOutput machine input.toPaper default.running.toPaper
  {
    zNext := paperOutput.zNext
    runningNext := default.running
    pcNext := paperOutput.pcNext
    x := paperOutput.x
  }

@[simp] theorem outputOf_toPaper
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount)
    (default : DefaultCarrier machine)
    (input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount) :
    (outputOf machine default input).toPaper =
      Paper.derivedOutput machine input.toPaper default.running.toPaper := rfl

/-- The only three base equations not computed by `outputOf`. -/
structure Obligations
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount) : Prop where
  iterationZero : input.iteration = 0
  initialState : input.z0 = input.zi
  dispatch : machine.control input.zi input.witness =
    Paper.ProgramCounter.ofIndex functionIndex

/-- Independent base relation with every output field computed. -/
def Holds
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount)
    (default : DefaultCarrier machine)
    (functionIndex : Fin slotCount)
    (input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount)
    (output : Output Digest AppState shape publicRingColumns publicFits
      verifierRows slotCount) : Prop :=
  Obligations machine functionIndex input /\
    output = outputOf machine default input

/-- The three base obligations construct the canonical accepted rich output. -/
theorem complete
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount)
    (default : DefaultCarrier machine)
    (functionIndex : Fin slotCount)
    (input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount)
    (obligations : Obligations machine functionIndex input) :
    Holds machine default functionIndex input (outputOf machine default input) :=
  ⟨obligations, rfl⟩

/-- Every rich base execution projects to the exact Construction-2 base
branch. This theorem says nothing about parent-cache authority. -/
theorem sound
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount}
    {default : DefaultCarrier machine}
    {functionIndex : Fin slotCount}
    {input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount}
    {output : Output Digest AppState shape publicRingColumns publicFits
      verifierRows slotCount}
    (accepted : Holds machine default functionIndex input output) :
    Paper.BaseHolds machine functionIndex input.toPaper output.toPaper := by
  rcases accepted with ⟨obligations, rfl⟩
  refine {
    iterationZero := obligations.iterationZero
    initialState := obligations.initialState
    application := ?_
    defaultRunning := ?_
    outputHash := ?_
  }
  · have derived := Paper.derivedOutput_application
      machine input.toPaper default.running.toPaper
    have controlEq :
        machine.control input.toPaper.zi input.toPaper.witness =
          Paper.ProgramCounter.ofIndex functionIndex :=
      obligations.dispatch
    have indexEq :
        (machine.control input.toPaper.zi input.toPaper.witness).index =
          functionIndex := by
      rw [controlEq]
      exact Paper.ProgramCounter.index_ofIndex functionIndex
    rw [indexEq] at derived
    simpa using derived
  · exact default.projection
  · exact Paper.derivedOutput_outputHolds
      machine input.toPaper default.running.toPaper

end Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics
