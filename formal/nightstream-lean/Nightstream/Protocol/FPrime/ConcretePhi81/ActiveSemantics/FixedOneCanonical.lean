import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

/-!
Canonical fixed-one-slot carrier for the independent active F-prime semantics.

Owns: the sole typed slot; a fresh-claim body with no caller-owned relation
structure; an active input with neither a raw prior counter nor a fresh
structure; reconstruction of both values from verifier-owned setup; and exact
projection to the existing independent active relation.

Does not own: selection of the one-slot profile by production, decoding of a
Rust or R1CS input into this carrier, executable checking, honest NIFS
construction, necessity of the retained obligations, costs, or row removal.

Emits constraints: no.

Authority boundary: this is a model-level profile theorem. It permits a future
implementation to omit the prior-counter and fresh-structure checks only if
that implementation constructs those values exactly as `Input.toActive` does.
If either value remains prover supplied, the corresponding raw-carrier check
remains a genuine security boundary.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.fixed_one.carrier.slot` | the only selected slot is `0` and its paper counter is `1` | computed | `selected`, `fin_eq_selected` |
| `fprime.fixed_one.carrier.fresh` | reconstruct the fresh CCS structure from `(vk, slot)` setup | computed | `Fresh.toStatement`, `Input.toActive` |
| `fprime.fixed_one.carrier.prior_pc` | reconstruct the sole valid one-based prior counter | computed | `Input.toActive` |
| `fprime.fixed_one.obligations` | retain iteration, prior-link, and semantic NIFS only | checked | `Obligations` |
| `fprime.fixed_one.projection.obligations` | the three-field carrier is exactly the six-field active target | model-level equivalence | `obligations_iff_active` |
| `fprime.fixed_one.projection.relation` | canonical acceptance is exactly active acceptance after reconstruction | model-level equivalence | `holds_iff_active` |
| `fprime.fixed_one.projection.raw` | an authoritative raw input round-trips through the smaller carrier | conditional model-level equivalence | `Input.toActive_erase_of_authority`, `holds_projection_iff` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical

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

/-- The sole selected slot in the fixed-one profile. -/
def selected : Fin 1 := ⟨0, by decide⟩

@[simp] theorem selected_val : selected.val = 0 := rfl

/-- Every typed fixed-one slot is the sole canonical slot. -/
theorem fin_eq_selected (slot : Fin 1) : slot = selected := by
  exact Subsingleton.elim slot selected

/-- The public fresh-claim fields that remain after the relation structure is
made verifier owned. -/
structure Fresh
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  commitment : CommitmentValue verifierRows
  publicInput : RelationPublicInput shape publicRingColumns publicFits
  stage : NormStage

namespace Fresh

/-- Install the verifier-owned relation structure into the fresh claim. -/
def toStatement
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (fresh : Fresh shape publicRingColumns publicFits verifierRows)
    (system : RelationStructure shape publicRingColumns publicFits) :
    Phi81Relation.CCSStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) where
  constraintSystem := system
  commitment := fresh.commitment
  publicInput := fresh.publicInput
  stage := fresh.stage

/-- Erase only the structure field from a raw fresh CCS statement. -/
def erase
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (statement :
      Phi81Relation.CCSStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    Fresh shape publicRingColumns publicFits verifierRows where
  commitment := statement.commitment
  publicInput := statement.publicInput
  stage := statement.stage

@[simp] theorem erase_toStatement
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (fresh : Fresh shape publicRingColumns publicFits verifierRows)
    (system : RelationStructure shape publicRingColumns publicFits) :
    erase (fresh.toStatement system) = fresh := by
  cases fresh
  rfl

/-- Reinstalling the observed structure reconstructs the raw statement. -/
@[simp] theorem toStatement_erase
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (statement :
      Phi81Relation.CCSStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    (erase statement).toStatement statement.constraintSystem = statement := by
  cases statement
  rfl

end Fresh

/-- Fixed-one active input with no raw prior counter and no caller-owned fresh
relation structure. -/
structure Input
    (OuterKey : Type uOuterKey)
    (AppState : Type uAppState)
    (Witness : Type uWitness)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  verifierKey : OuterKey
  iteration : Nat
  z0 : AppState
  zi : AppState
  running :
    Running shape publicRingColumns publicFits verifierRows 1
  fresh : Fresh shape publicRingColumns publicFits verifierRows
  witness : Witness

namespace Input

/-- Reconstruct the general active input from the canonical fixed-one
carrier. Both omitted authority fields are verifier computations. -/
def toActive
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    ActiveSemantics.Input OuterKey AppState Witness shape publicRingColumns
      publicFits verifierRows 1 where
  verifierKey := input.verifierKey
  iteration := input.iteration
  z0 := input.z0
  zi := input.zi
  running := input.running
  fresh := input.fresh.toStatement
    (setup.expectedStructure input.verifierKey selected)
  priorPc := selected.val + 1
  witness := input.witness

/-- Erase the two fields computed by `toActive` from a raw active input. -/
def erase
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (input :
      ActiveSemantics.Input OuterKey AppState Witness shape publicRingColumns
        publicFits verifierRows 1) :
    Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows where
  verifierKey := input.verifierKey
  iteration := input.iteration
  z0 := input.z0
  zi := input.zi
  running := input.running
  fresh := Fresh.erase input.fresh
  witness := input.witness

@[simp] theorem erase_toActive
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    erase (input.toActive setup) = input := by
  cases input
  rfl

/-- The omitted prior-counter equation holds for every typed fixed-one slot. -/
theorem priorSlot_derived
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (slot : Fin 1) :
    (input.toActive setup).priorPc = slot.val + 1 := by
  have slotEq : slot = selected := fin_eq_selected slot
  subst slot
  rfl

/-- The omitted fresh-structure equation holds for every typed fixed-one
slot. -/
theorem expectedStructure_derived
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (slot : Fin 1) :
    (input.toActive setup).fresh.constraintSystem =
      setup.expectedStructure input.verifierKey slot := by
  have slotEq : slot = selected := fin_eq_selected slot
  subst slot
  rfl

/-- A raw one-slot input is reconstructed exactly when its two erased fields
already equal their verifier-owned canonical values. -/
theorem toActive_erase_of_authority
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      ActiveSemantics.Input OuterKey AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (slot : Fin 1)
    (priorSlot : input.priorPc = slot.val + 1)
    (expectedStructure : input.fresh.constraintSystem =
      setup.expectedStructure input.verifierKey slot) :
    (erase input).toActive setup = input := by
  have slotEq : slot = selected := fin_eq_selected slot
  subst slot
  cases input with
  | mk verifierKey iteration z0 zi running fresh priorPc witness =>
      cases fresh with
      | mk constraintSystem commitment publicInput stage =>
          have priorPcEq : priorPc = selected.val + 1 := priorSlot
          have structureEq : constraintSystem =
              setup.expectedStructure verifierKey selected := expectedStructure
          subst priorPc
          subst constraintSystem
          rfl

end Input

section

variable {OuterKey : Type uOuterKey}
variable {Digest : Type uDigest}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {domain : FlatNcDomain}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Dispatch is computed by the fixed-one program-counter codomain. -/
theorem dispatch_derived
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    machine.control input.zi input.witness =
      Paper.ProgramCounter.ofIndex functionIndex := by
  calc
    machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex
          (machine.control input.zi input.witness).index :=
      (Paper.ProgramCounter.ofIndex_index
        (machine.control input.zi input.witness)).symm
    _ = Paper.ProgramCounter.ofIndex functionIndex :=
      congrArg Paper.ProgramCounter.ofIndex
        (Subsingleton.elim
          (machine.control input.zi input.witness).index functionIndex)

/-- The three obligations not already encoded by the canonical carrier. This
is an exact reduction, not yet an inclusion-minimality claim. -/
structure Obligations
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) : Prop where
  iterationPositive : 0 < input.iteration
  priorPublicInput :
    (input.toActive setup).fresh.publicInput =
      machine.encodeInstance
        (machine.hash (Paper.priorHashPreimage (input.toActive setup).toPaper))
  selectedNifs :
    FixedActive.ResultTransition
      (contextAt setup (input.toActive setup) selected) selectedNext

namespace Obligations

/-- Expand the three canonical obligations into the full active target. -/
def toActive
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    (functionIndex : Fin 1)
    {input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows}
    {selectedNext :
      Slot shape publicRingColumns publicFits verifierRows}
    (obligations : Obligations setup machine input selectedNext) :
    ActiveSemantics.Obligations setup machine functionIndex
      (input.toActive setup) selected selectedNext where
  iterationPositive := obligations.iterationPositive
  priorSlot := Input.priorSlot_derived setup input selected
  priorPublicInput := obligations.priorPublicInput
  expectedStructure := Input.expectedStructure_derived setup input selected
  selectedNifs := obligations.selectedNifs
  dispatch := dispatch_derived machine functionIndex input

/-- Project the full active target back to the three canonical obligations. -/
def ofActive
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {functionIndex : Fin 1}
    {input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows}
    {selectedNext :
      Slot shape publicRingColumns publicFits verifierRows}
    (obligations :
      ActiveSemantics.Obligations setup machine functionIndex
        (input.toActive setup) selected selectedNext) :
    Obligations setup machine input selectedNext where
  iterationPositive := obligations.iterationPositive
  priorPublicInput := obligations.priorPublicInput
  selectedNifs := obligations.selectedNifs

end Obligations

/-- Exact field-level equivalence between the canonical target and the full
active target at the sole selected slot. -/
theorem obligations_iff_active
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) :
    Obligations setup machine input selectedNext <->
      ActiveSemantics.Obligations setup machine functionIndex
        (input.toActive setup) selected selectedNext := by
  constructor
  · exact fun obligations => obligations.toActive functionIndex
  · exact Obligations.ofActive

/-- Independent canonical relation over the smaller fixed-one carrier. -/
def Holds
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        1) : Prop :=
  exists selectedNext :
      Slot shape publicRingColumns publicFits verifierRows,
    Obligations setup machine input selectedNext /\
      output = outputOf machine (input.toActive setup) selected selectedNext

/-- Canonical acceptance is sound and complete for the independent active
relation after reconstructing the two verifier-owned fields. -/
theorem holds_iff_active
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        1) :
    Holds setup machine input output <->
      ActiveSemantics.Holds setup machine functionIndex
        (input.toActive setup) output := by
  constructor
  · rintro ⟨selectedNext, obligations, outputEq⟩
    exact ⟨selected, selectedNext,
      (obligations_iff_active setup machine functionIndex input selectedNext).1
        obligations,
      outputEq⟩
  · rintro ⟨slot, selectedNext, obligations, outputEq⟩
    have slotEq : slot = selected := fin_eq_selected slot
    subst slot
    exact ⟨selectedNext,
      (obligations_iff_active setup machine functionIndex input selectedNext).2
        obligations,
      outputEq⟩

/-- On raw inputs whose erased values already satisfy verifier authority, the
canonical relation is exactly the existing independent active relation. This
theorem does not establish that any production decoder meets those premises. -/
theorem holds_projection_iff
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      ActiveSemantics.Input OuterKey AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        1)
    (slot : Fin 1)
    (priorSlot : input.priorPc = slot.val + 1)
    (expectedStructure : input.fresh.constraintSystem =
      setup.expectedStructure input.verifierKey slot) :
    Holds setup machine (Input.erase input) output <->
      ActiveSemantics.Holds setup machine functionIndex input output := by
  rw [holds_iff_active]
  rw [Input.toActive_erase_of_authority setup input slot priorSlot
    expectedStructure]

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical
