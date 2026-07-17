import Nightstream.Protocol.FPrime.ConcretePhi81.Context
import Nightstream.Protocol.FPrime.Paper.Output

/-!
Independent fixed-active outer F-prime semantics over the concrete Phi81 NIFS.

Owns: the full per-slot parent-and-children carrier, deterministic construction
of the selected NIFS context, the irreducible active-branch obligations, and
the canonical output computed from one semantic NIFS result.

Does not own: executable certificate checking, the concrete-to-paper NIFS
bridge, base steps, compact `XOut` hashing, Rust, R1CS, costs, necessity, or row
removal.

Emits constraints: no.

Authority boundary: the certificate layer may later choose a typed slot, but
the semantic slot is bound to the raw prior counter. Split-NC public input and
transcript state are verifier computations in `Setup`; callers never supply a
completed NIFS context. The outgoing parent, inactive copies, application
output, next counter, and output digest are computed rather than checked.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.active.iteration` | recursive execution has `i > 0` | checked | `Obligations.iterationPositive` |
| `fprime.active.prior_slot` | raw `pc_i` selects one typed slot | checked | `Obligations.priorSlot` |
| `fprime.active.prior_link` | fresh public input binds the exact prior preimage | checked | `Obligations.priorPublicInput` |
| `fprime.active.structure` | fresh structure is the verifier-owned selected structure | checked | `Obligations.expectedStructure` |
| `fprime.active.structure.running` | all selected running/output structures follow from fresh structure plus NIFS semantics | derived | `Obligations.selectedStructures_eq_expected` |
| `fprime.active.nifs` | selected source transition yields one complete fold result | checked | `Obligations.selectedNifs` |
| `fprime.active.dispatch` | verifier control selects this fixed `F_j` | checked | `Obligations.dispatch` |
| `fprime.active.output` | update one slot and compute `pcNext`, `zNext`, and `x` | computed | `outputOf` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uOuterKey uAppState uWitness uDigest uTranscriptState

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

/-- Verifier/advice input to one fixed-active augmented-function invocation. -/
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

/-- Rich output computed by the fixed-active verifier. -/
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

/-- The paper machine specialized to the concrete Phi81 statement carrier.

Its base-only `defaultRunning` field is deliberately unused by this active
component. -/
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

/-- Verifier-owned construction of every selected concrete NIFS invocation. -/
structure Setup
    (OuterKey : Type uOuterKey)
    (AppState : Type uAppState)
    (Witness : Type uWitness)
    (TranscriptState : Type uTranscriptState)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows slotCount : Nat) where
  template :
    OuterKey -> Fin slotCount ->
      Nightstream.Protocol.FPrime.ConcretePhi81.Context.Template
        shape TranscriptState publicRingColumns publicFits verifierRows
  expectedStructure :
    OuterKey -> Fin slotCount ->
      RelationStructure shape publicRingColumns publicFits
  piCcsInput :
    Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount ->
      Fin slotCount ->
        PiCCS.SplitNc.Verifier.PublicInput shape
  priorTranscriptState :
    Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount ->
      Fin slotCount -> TranscriptState

/-- Construct the exact public invocation at one typed slot. -/
def invocationAt
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (slot : Fin slotCount) :
    Nightstream.Protocol.FPrime.ConcretePhi81.Context.Invocation
      shape TranscriptState publicRingColumns publicFits verifierRows where
  fresh := input.fresh
  running := input.running slot
  piCcsInput := setup.piCcsInput input slot
  priorState := setup.priorTranscriptState input slot

/-- Build the sole concrete NIFS context consumed at one selected slot. -/
def contextAt
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (slot : Fin slotCount) :
    FixedActive.Context
      shape TranscriptState publicRingColumns publicFits verifierRows :=
  (setup.template input.verifierKey slot).build
    (invocationAt setup input slot)

@[simp] theorem contextAt_runningParent
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (slot : Fin slotCount) :
    (contextAt setup input slot).runningParent =
      some (input.running slot).parent := rfl

/-- Replace exactly one complete parent-and-children slot. -/
def updatedRunning
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) :
    Running shape publicRingColumns publicFits verifierRows slotCount :=
  fun slot =>
    if slot = selected then selectedNext else input.running slot

@[simp] theorem updatedRunning_selected
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) :
    updatedRunning input selected selectedNext selected = selectedNext := by
  simp [updatedRunning]

theorem updatedRunning_other
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows)
    (slot : Fin slotCount)
    (notSelected : slot ≠ selected) :
    updatedRunning input selected selectedNext slot = input.running slot := by
  simp [updatedRunning, notSelected]

/-- Canonical active output. Every advice-like output field is computed. -/
def outputOf
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) :
    Output Digest AppState shape publicRingColumns publicFits verifierRows
      slotCount :=
  let runningNext := updatedRunning input selected selectedNext
  let paperOutput :=
    Paper.derivedOutput machine input.toPaper runningNext.toPaper
  {
    zNext := paperOutput.zNext
    runningNext := runningNext
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
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) :
    (outputOf machine input selected selectedNext).toPaper =
      Paper.derivedOutput machine input.toPaper
        (updatedRunning input selected selectedNext).toPaper := rfl

@[simp] theorem outputOf_runningNext_selected
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) :
    (outputOf machine input selected selectedNext).runningNext selected =
      selectedNext := by
  simp [outputOf]

theorem outputOf_runningNext_other
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows)
    (slot : Fin slotCount)
    (notSelected : slot ≠ selected) :
    (outputOf machine input selected selectedNext).runningNext slot =
      input.running slot := by
  exact updatedRunning_other input selected selectedNext slot notSelected

/-- The six irreducible active-branch semantic obligations.

The selected result is semantic NIFS output, not a caller-supplied digest.
Output construction and inactive-slot copying are absent because `outputOf`
computes them. -/
structure Obligations
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) : Prop where
  iterationPositive : 0 < input.iteration
  priorSlot : input.priorPc = selected.val + 1
  priorPublicInput :
    input.fresh.publicInput =
      machine.encodeInstance
        (machine.hash (Paper.priorHashPreimage input.toPaper))
  expectedStructure :
    input.fresh.constraintSystem =
      setup.expectedStructure input.verifierKey selected
  selectedNifs :
    FixedActive.ResultTransition
      (contextAt setup input selected) selectedNext
  dispatch :
    machine.control input.zi input.witness =
      Paper.ProgramCounter.ofIndex functionIndex

namespace Obligations

/-- The raw counter-to-slot equality already implies the paper range check. -/
theorem priorPcValid
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount}
    {selected : Fin slotCount}
    {selectedNext :
      Slot shape publicRingColumns publicFits verifierRows}
    (obligations :
      Obligations setup machine functionIndex input selected selectedNext) :
    Paper.InRange slotCount input.priorPc := by
  rw [obligations.priorSlot]
  exact (Paper.ProgramCounter.ofIndex selected).valid

/-- No separate range-selected index witness is semantically required. -/
theorem selectedIndex_eq
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount}
    {selected : Fin slotCount}
    {selectedNext :
      Slot shape publicRingColumns publicFits verifierRows}
    (obligations :
      Obligations setup machine functionIndex input selected selectedNext) :
    Paper.selectedIndex obligations.priorPcValid = selected := by
  apply Fin.ext
  simp [Paper.selectedIndex, Paper.ProgramCounter.index,
    obligations.priorSlot]

/-- The one fresh-structure check plus the semantic NIFS transition determines
every selected incoming and outgoing child structure. No per-child outer
structure checks remain. -/
theorem selectedStructures_eq_expected
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount}
    {selected : Fin slotCount}
    {selectedNext :
      Slot shape publicRingColumns publicFits verifierRows}
    (obligations :
      Obligations setup machine functionIndex input selected selectedNext) :
    (∀ running,
        ((input.running selected).children running).constraintSystem =
          setup.expectedStructure input.verifierKey selected) ∧
      (∀ child,
        (selectedNext.children child).constraintSystem =
          setup.expectedStructure input.verifierKey selected) := by
  constructor
  · intro running
    have sameFresh :=
      obligations.selectedNifs.runningStructure_eq_fresh running
    calc
      ((input.running selected).children running).constraintSystem =
          input.fresh.constraintSystem := by
        simpa [contextAt, invocationAt,
          Nightstream.Protocol.FPrime.ConcretePhi81.Context.Template.build,
          Nightstream.Protocol.FPrime.ConcretePhi81.Context.Invocation.sourceProduct]
          using sameFresh
      _ = setup.expectedStructure input.verifierKey selected :=
        obligations.expectedStructure
  · intro child
    have sameFresh :=
      obligations.selectedNifs.childStructure_eq_fresh child
    calc
      (selectedNext.children child).constraintSystem =
          input.fresh.constraintSystem := by
        simpa [contextAt, invocationAt,
          Nightstream.Protocol.FPrime.ConcretePhi81.Context.Template.build,
          Nightstream.Protocol.FPrime.ConcretePhi81.Context.Invocation.sourceProduct]
          using sameFresh
      _ = setup.expectedStructure input.verifierKey selected :=
        obligations.expectedStructure

end Obligations

/-- Independent fixed-active outer relation with a canonical rich output. -/
def Holds
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount) : Prop :=
  exists selected : Fin slotCount,
    exists selectedNext :
        Slot shape publicRingColumns publicFits verifierRows,
      Obligations setup machine functionIndex input selected selectedNext /\
        output = outputOf machine input selected selectedNext

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
