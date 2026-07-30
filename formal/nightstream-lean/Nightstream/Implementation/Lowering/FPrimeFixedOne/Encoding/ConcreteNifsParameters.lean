import Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
import Nightstream.Protocol.FPrime.ConcretePhi81.Context
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Checker

/-!
Lean-owned fixed-active ConcretePhi81 NIFS adapter for the fixed-one lowering
vocabulary.

Owns: the exact `Key × Running × Fresh × Proof` carrier consumed by the
fixed-one `nifsVerify` call; canonical reconstruction of the verifier context;
the deterministic concrete evaluator; and its exact acceptance equation.

Does not own: an application step, terminal witnesses, physical R1CS rows,
paper-event probability bounds, Rust, generated artifacts, or costs.

Authority boundary: relation structure and every static verifier component
belong to `SelectedKey`. The running input retains the complete parent and all
fourteen 270-coordinate child payloads. The proof carries only the public
Split-NC input, transcript prefix, and raw phase messages. No accepted result
or semantic conclusion is a proof field.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.HyperNova.NonInteractiveMultiFold
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uTranscriptState

/-- Static verifier authority for one selected fixed-active relation. -/
structure SelectedKey
    (shape : SemanticShape)
    (TranscriptState : Type uTranscriptState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  template :
    Nightstream.Protocol.FPrime.ConcretePhi81.Context.Template
      shape TranscriptState publicRingColumns publicFits verifierRows
  system :
    Phi81Relation.Structure
      (RelationShape shape publicRingColumns publicFits)

/-- Complete canonical running accumulator. No relation structure or norm
stage is caller-owned, but the parent and all fourteen public-input payloads
remain present in full. -/
structure SelectedRunning
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  parent :
    FixedActive.Canonical.ParentPayload
      shape publicRingColumns publicFits verifierRows
  children : Fin productionGlobalParams.k ->
    FixedActive.Canonical.RunningPayload
      shape publicRingColumns publicFits verifierRows

/-- The selected fresh claim after relation structure and fresh stage are
made verifier-owned. -/
abbrev SelectedFresh
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  FixedActive.Canonical.FreshPayload
    shape publicRingColumns publicFits verifierRows

/-- Raw prover message. The certificate is indexed by the exact public
Split-NC input carried in the same message; it carries no output or accepted
proposition. -/
structure SelectedProof
    (shape : SemanticShape)
    (TranscriptState : Type uTranscriptState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  piCcsInput : PiCCS.SplitNc.Verifier.PublicInput shape
  priorState : TranscriptState
  certificate :
    ConcretePhi81.Certificate
      (arity := FixedActive.arity)
      publicRingColumns publicFits verifierRows piCcsInput

/-- Reconstruct the sole canonical context consumed by the physical
fixed-active checker. -/
def context
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    FixedActive.Canonical.Context shape TranscriptState publicRingColumns
      publicFits verifierRows where
  covers := key.template.covers
  key := key.template.key
  alignment := key.template.alignment
  input := {
    system := key.system
    fresh := fresh
    running := running.children
    parent := running.parent
  }
  pending := none
  piCcsInput := proof.piCcsInput
  priorState := proof.priorState
  piCcsSchedule := key.template.piCcsSchedule
  piRlcMachine := key.template.piRlcMachine
  profile := key.template.profile
  challengeSetSize := key.template.challengeSetSize

namespace SelectedRunning

/-- Forget only verifier-owned structure and norm stages from the canonical
result. Every commitment, public input, point, and evaluation remains in the
next running accumulator. -/
def ofResult
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (result :
      FixedActive.FoldResult shape publicRingColumns publicFits verifierRows) :
    SelectedRunning shape publicRingColumns publicFits verifierRows where
  parent := {
    commitment := result.parent.commitment
    publicInput := result.parent.publicInput
    point := result.parent.point
    evaluations := result.parent.evaluations
  }
  children := fun child => {
    commitment := (result.children child).commitment
    publicInput := (result.children child).publicInput
    point := (result.children child).point
    evaluations := (result.children child).evaluations
  }

end SelectedRunning

/-- HyperNova's one-message NIFS verifier instantiated by the exact canonical
ConcretePhi81 checker and evaluator. -/
def nifsVerifier
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth} :
    Verifier
      (SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
      (SelectedRunning shape publicRingColumns publicFits verifierRows)
      (SelectedFresh shape publicRingColumns publicFits verifierRows)
      (SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) where
  verify := fun key running fresh proof =>
    (FixedActive.Evaluator.run
      (FixedActive.Canonical.Checker.evaluatorChecker
        (context key running fresh proof))
      proof.certificate).map SelectedRunning.ofResult

/-- Exact operational meaning of the selected NIFS verifier. Successful
execution is equivalent to concrete physical acceptance and the uniquely
computed payload result. -/
theorem nifsVerifier_eq_some_iff
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    nifsVerifier.verify key running fresh proof = some output ↔
      ConcretePhi81.Accepted
          (context key running fresh proof).materialize proof.certificate ∧
        SelectedRunning.ofResult
            (FixedActive.resultOf
              (context key running fresh proof).materialize
              proof.certificate) =
          output := by
  let checker :=
    FixedActive.Canonical.Checker.evaluatorChecker
      (context key running fresh proof)
  cases executed :
      FixedActive.Evaluator.run checker proof.certificate with
  | none =>
      have notAccepted :
          ¬ ConcretePhi81.Accepted
            (context key running fresh proof).materialize
            proof.certificate := by
        intro accepted
        have succeeds :
            FixedActive.Evaluator.run checker proof.certificate =
              some
                (FixedActive.resultOf
                  (context key running fresh proof).materialize
                  proof.certificate) :=
          (FixedActive.Evaluator.run_eq_some_iff_accepted
            checker proof.certificate _).2 ⟨accepted, rfl⟩
        rw [executed] at succeeds
        contradiction
      simp [nifsVerifier, checker, executed, notAccepted]
  | some result =>
      have meaning :
          ConcretePhi81.Accepted
              (context key running fresh proof).materialize
              proof.certificate ∧
            FixedActive.resultOf
                (context key running fresh proof).materialize
                proof.certificate =
              result :=
        (FixedActive.Evaluator.run_eq_some_iff_accepted
          checker proof.certificate result).1 executed
      simp only [nifsVerifier, checker, executed, Option.map_some,
        Option.some.injEq]
      constructor
      · intro equal
        exact ⟨meaning.1, congrArg SelectedRunning.ofResult meaning.2 |>.trans
          equal⟩
      · rintro ⟨_accepted, equal⟩
        exact (congrArg SelectedRunning.ofResult meaning.2).symm.trans equal

/-- Construct the fixed-one lowering vocabulary around the selected concrete
NIFS. Application semantics, terminal relations, widths, and physical
footprints remain explicit deployment inputs. -/
def selected
    {shape : SemanticShape}
    {TranscriptState : Type}
    {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (keys : Fin 1 ->
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (defaultRunning :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (machine :
      Machine
        (SelectedKey shape TranscriptState publicRingColumns publicFits
          verifierRows)
        Digest AppState Witness
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        Encoded 1)
    (terminalRelations :
      TerminalRelations
        (SelectedKey shape TranscriptState publicRingColumns publicFits
          verifierRows)
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        RunningWitness
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
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
    SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows
  Digest := Digest
  State := AppState
  Witness := Witness
  Running :=
    SelectedRunning shape publicRingColumns publicFits verifierRows
  Fresh := SelectedFresh shape publicRingColumns publicFits verifierRows
  NifsProof :=
    SelectedProof shape TranscriptState publicRingColumns publicFits
      verifierRows
  Encoded := Encoded
  RunningWitness := RunningWitness
  FreshWitness := FreshWitness
  stateDecidableEq := inferInstance
  encodedDecidableEq := inferInstance
  setup := {
    verifierKeys := keys
    nifs := nifsVerifier
    defaultRunning := defaultRunning
  }
  machine := machine
  terminalRelations := terminalRelations
  terminalChecks := terminalChecks
  widths := widths
  footprints := footprints

@[simp] theorem selected_setup_nifs
    {shape : SemanticShape}
    {TranscriptState : Type}
    {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (keys : Fin 1 ->
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (defaultRunning :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (machine :
      Machine
        (SelectedKey shape TranscriptState publicRingColumns publicFits
          verifierRows)
        Digest AppState Witness
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        Encoded 1)
    (terminalRelations :
      TerminalRelations
        (SelectedKey shape TranscriptState publicRingColumns publicFits
          verifierRows)
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        RunningWitness
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        FreshWitness 1)
    (terminalChecks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        terminalRelations)
    (widths : Widths)
    (footprints : Footprints) :
    (selected keys defaultRunning machine terminalRelations terminalChecks
      widths footprints).setup.nifs = nifsVerifier :=
  rfl

/-- The selected lowering call executes exactly the canonical concrete NIFS
adapter. -/
@[simp] theorem callEval_nifsVerify
    {shape : SemanticShape}
    {TranscriptState : Type}
    {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (keys : Fin 1 ->
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (defaultRunning :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (machine :
      Machine
        (SelectedKey shape TranscriptState publicRingColumns publicFits
          verifierRows)
        Digest AppState Witness
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        Encoded 1)
    (terminalRelations :
      TerminalRelations
        (SelectedKey shape TranscriptState publicRingColumns publicFits
          verifierRows)
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        RunningWitness
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        FreshWitness 1)
    (terminalChecks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        terminalRelations)
    (widths : Widths)
    (footprints : Footprints)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    callEval
        (selected keys defaultRunning machine terminalRelations terminalChecks
          widths footprints)
        Call.nifsVerify
        (.cons running (.cons fresh (.cons proof .nil))) =
      match nifsVerifier.verify
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof with
      | none => none
      | some folded => some (.cons folded .nil) :=
  by
    simp [callEval, selected]
    cases verifierResult :
        nifsVerifier.verify
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof <;>
      simp

/-- Exact operational contract of the selected lowering call.

The output is not caller-authored: successful `callEval` is equivalent to
acceptance by the concrete fixed-active checker and equality with the
deterministically computed parent-and-children payload. -/
theorem callEval_nifsVerify_eq_some_iff
    {shape : SemanticShape}
    {TranscriptState : Type}
    {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (keys : Fin 1 ->
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (defaultRunning :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (machine :
      Machine
        (SelectedKey shape TranscriptState publicRingColumns publicFits
          verifierRows)
        Digest AppState Witness
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        Encoded 1)
    (terminalRelations :
      TerminalRelations
        (SelectedKey shape TranscriptState publicRingColumns publicFits
          verifierRows)
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        RunningWitness
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        FreshWitness 1)
    (terminalChecks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        terminalRelations)
    (widths : Widths)
    (footprints : Footprints)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    callEval
        (selected keys defaultRunning machine terminalRelations terminalChecks
          widths footprints)
        Call.nifsVerify
        (.cons running (.cons fresh (.cons proof .nil))) =
        some (.cons output .nil) ↔
      ConcretePhi81.Accepted
          (context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate ∧
        SelectedRunning.ofResult
            (FixedActive.resultOf
              (context
                (keys
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                running fresh proof).materialize
              proof.certificate) =
          output := by
  rw [callEval_nifsVerify]
  cases verifierResult :
      nifsVerifier.verify
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
        running fresh proof with
  | none =>
      have impossible :
          ¬ (ConcretePhi81.Accepted
                (context
                  (keys
                    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                  running fresh proof).materialize
                proof.certificate ∧
              SelectedRunning.ofResult
                  (FixedActive.resultOf
                    (context
                      (keys
                        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                      running fresh proof).materialize
                    proof.certificate) =
                output) := by
        intro right
        have succeeds :=
          (nifsVerifier_eq_some_iff
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof output).2 right
        rw [verifierResult] at succeeds
        contradiction
      constructor
      · intro equal
        simp at equal
      · intro right
        exact (impossible right).elim
  | some result =>
      have meaning :=
        (nifsVerifier_eq_some_iff
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof result).1 verifierResult
      constructor
      · intro equal
        have resultEqual : result = output := by
          simp only [Option.some.injEq] at equal
          cases equal
          rfl
        exact ⟨meaning.1, meaning.2.trans resultEqual⟩
      · rintro ⟨_accepted, outputEqual⟩
        have resultEqual : result = output :=
          meaning.2.symm.trans outputEqual
        cases resultEqual
        simp

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
