import Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
import Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs

/-!
Contract: select the paper SuperNeo NIFS as the exact semantic evaluator of
the fixed-one lowering vocabulary.

Owns: a constructor for `Vocabulary.Parameters` whose `nifsVerify` call is
definitionally the one-message `PaperNonInteractive.verify`, together with
the exact call-evaluation equation used by the physical recipe.

Does not own: application codecs, physical rows, a `CallRecipe`, Fiat--Shamir
instantiation, event transport, Rust, artifacts, or an application `step`.

Assurance tier: model-level.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsParameters

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Protocol.FPrime.CanonicalVerifier
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.Nifs

abbrev SelectedKey
    (Extension : Type)
    (Commitment : Type)
    (PublicInput : Type)
    (Scalar : Type)
    (TranscriptState : Type)
    (shape : Shape)
    (columns blockCount degreeBound : Nat) :=
  PaperNonInteractive.Key Extension Commitment PublicInput Scalar
    TranscriptState shape columns blockCount degreeBound

abbrev SelectedRunning
    (Extension : Type)
    (Commitment : Type)
    (PublicInput : Type)
    (shape : Shape) :=
  PaperNonInteractive.Running Extension Commitment PublicInput shape

abbrev SelectedFresh
    (Commitment : Type)
    (PublicInput : Type)
    (shape : Shape) :=
  PaperNonInteractive.Fresh Commitment PublicInput shape

abbrev SelectedProof
    (Extension : Type)
    (Commitment : Type)
    (shape : Shape)
    (degreeBound : Nat) :=
  PaperNonInteractive.Proof Extension Commitment shape degreeBound

/-- Construct the fixed-one vocabulary around the exact paper NIFS.

The application machine and terminal relations remain explicit setup inputs,
as HyperNova requires. The NIFS verifier is not an input: it is selected by
this constructor. -/
def selected
    {Extension Commitment PublicInput Scalar TranscriptState : Type}
    {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
    [DecidableEq Extension]
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (oneFresh : shape.freshCount = 1)
    (keys : Fin 1 ->
      SelectedKey Extension Commitment PublicInput Scalar TranscriptState
        shape columns blockCount degreeBound)
    (defaultRunning :
      SelectedRunning Extension Commitment PublicInput shape)
    (machine :
      Machine
        (SelectedKey Extension Commitment PublicInput Scalar TranscriptState
          shape columns blockCount degreeBound)
        Digest AppState Witness
        (SelectedRunning Extension Commitment PublicInput shape)
        (SelectedFresh Commitment PublicInput shape)
        Encoded 1)
    (terminalRelations :
      TerminalRelations
        (SelectedKey Extension Commitment PublicInput Scalar TranscriptState
          shape columns blockCount degreeBound)
        (SelectedRunning Extension Commitment PublicInput shape)
        RunningWitness
        (SelectedFresh Commitment PublicInput shape)
        FreshWitness 1)
    (terminalChecks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        terminalRelations)
    (widths : Widths)
    (footprints : Footprints) :
    Parameters where
  Field := F
  fieldZero := 0
  fieldAdd := (· + ·)
  fieldMul := (· * ·)
  Key :=
    SelectedKey Extension Commitment PublicInput Scalar TranscriptState
      shape columns blockCount degreeBound
  Digest := Digest
  State := AppState
  Witness := Witness
  Running := SelectedRunning Extension Commitment PublicInput shape
  Fresh := SelectedFresh Commitment PublicInput shape
  NifsProof := SelectedProof Extension Commitment shape degreeBound
  Encoded := Encoded
  RunningWitness := RunningWitness
  FreshWitness := FreshWitness
  stateDecidableEq := inferInstance
  encodedDecidableEq := inferInstance
  setup :=
    PaperNonInteractiveNifs.construction2Setup
      oneFresh keys defaultRunning
  machine := machine
  terminalRelations := terminalRelations
  terminalChecks := terminalChecks
  widths := widths
  footprints := footprints

@[simp] theorem selected_setup_nifs
    {Extension Commitment PublicInput Scalar TranscriptState : Type}
    {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
    [DecidableEq Extension]
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (oneFresh : shape.freshCount = 1)
    (keys : Fin 1 ->
      SelectedKey Extension Commitment PublicInput Scalar TranscriptState
        shape columns blockCount degreeBound)
    (defaultRunning :
      SelectedRunning Extension Commitment PublicInput shape)
    (machine :
      Machine
        (SelectedKey Extension Commitment PublicInput Scalar TranscriptState
          shape columns blockCount degreeBound)
        Digest AppState Witness
        (SelectedRunning Extension Commitment PublicInput shape)
        (SelectedFresh Commitment PublicInput shape)
        Encoded 1)
    (terminalRelations :
      TerminalRelations
        (SelectedKey Extension Commitment PublicInput Scalar TranscriptState
          shape columns blockCount degreeBound)
        (SelectedRunning Extension Commitment PublicInput shape)
        RunningWitness
        (SelectedFresh Commitment PublicInput shape)
        FreshWitness 1)
    (terminalChecks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        terminalRelations)
    (widths : Widths)
    (footprints : Footprints) :
    (selected oneFresh keys defaultRunning machine terminalRelations
        terminalChecks widths footprints).setup.nifs =
      PaperNonInteractiveNifs.nifsVerifier :=
  rfl

/-- The selected lowering call is exactly the paper verifier graph. -/
@[simp] theorem callEval_nifsVerify
    {Extension Commitment PublicInput Scalar TranscriptState : Type}
    {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
    [DecidableEq Extension]
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (oneFresh : shape.freshCount = 1)
    (keys : Fin 1 ->
      SelectedKey Extension Commitment PublicInput Scalar TranscriptState
        shape columns blockCount degreeBound)
    (defaultRunning :
      SelectedRunning Extension Commitment PublicInput shape)
    (machine :
      Machine
        (SelectedKey Extension Commitment PublicInput Scalar TranscriptState
          shape columns blockCount degreeBound)
        Digest AppState Witness
        (SelectedRunning Extension Commitment PublicInput shape)
        (SelectedFresh Commitment PublicInput shape)
        Encoded 1)
    (terminalRelations :
      TerminalRelations
        (SelectedKey Extension Commitment PublicInput Scalar TranscriptState
          shape columns blockCount degreeBound)
        (SelectedRunning Extension Commitment PublicInput shape)
        RunningWitness
        (SelectedFresh Commitment PublicInput shape)
        FreshWitness 1)
    (terminalChecks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        terminalRelations)
    (widths : Widths)
    (footprints : Footprints)
    (running : SelectedRunning Extension Commitment PublicInput shape)
    (fresh : SelectedFresh Commitment PublicInput shape)
    (proof : SelectedProof Extension Commitment shape degreeBound) :
    callEval
        (selected oneFresh keys defaultRunning machine terminalRelations
          terminalChecks widths footprints)
        Call.nifsVerify
        (.cons running (.cons fresh (.cons proof .nil))) =
      match PaperNonInteractive.verify
          (keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof with
      | none => none
      | some folded => some (.cons folded .nil) :=
  by
    simp [callEval, selected, PaperNonInteractiveNifs.construction2Setup,
      PaperNonInteractiveNifs.nifsVerifier]
    cases verifierResult :
        PaperNonInteractive.verify
          (keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof <;>
      simp

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsParameters
