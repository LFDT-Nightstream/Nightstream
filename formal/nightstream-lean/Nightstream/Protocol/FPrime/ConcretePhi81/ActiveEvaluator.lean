import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

/-!
Executable fixed-active outer F-prime evaluator.

Owns: explicit Boolean refinement owners for the three nontrivial outer
equalities; a dependent concrete NIFS checker; the raw typed certificate;
fail-closed execution; exact physical characterization; semantic
soundness-or-named-failure.

Does not own: implementations of the equality or NIFS checkers, elimination
of Split-NC bad events, minimization of the inherited raw NIFS message
carrier, semantic closure, honest completeness, the concrete-to-paper bridge,
base steps, Rust, R1CS, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `Certificate` contains only a typed selected slot and the
raw certificate for the context computed at that slot. The slot is bound to
the raw prior counter by a direct Nat check. Large/function-valued equality
surfaces are decided only by named checker functions with exactness theorems;
there is no ambient `DecidableEq` requirement. Successful execution computes
the complete output through `ActiveSemantics.outputOf`. Any verifier-derived
cache still present inside the inherited NIFS certificate, including its
challenge vector, gains no authority here: `Checkers.nifs` must replay and
prove exact physical acceptance. Removing those cached fields is a later NIFS
message-minimization step. Semantic soundness additionally keeps every
private PiDEC child opening as an explicit extraction/binding premise; public
child recomposition alone is not treated as knowledge authority.

| Stage path | Executable owner | Meaning |
|---|---|---|
| `fprime.active.iteration` | `outerCheck` | direct Nat positivity |
| `fprime.active.prior_slot` | `outerCheck` | raw counter equals typed slot plus one |
| `fprime.active.prior_link` | `Checkers.priorLinkCheck` | exact prior public-input equality |
| `fprime.active.structure` | `Checkers.freshStructureCheck` | exact fresh-to-expected structure equality |
| `fprime.active.dispatch` | `Checkers.dispatchCheck` | exact fixed-function dispatch equality |
| `fprime.active.nifs` | `Checkers.nifs` | physical fixed-active NIFS acceptance |
| `fprime.active.nifs.children` | `run_sound_or_outputUnbound_or_piCcsBadEvent` | explicit private child-opening authority |
| `fprime.active.output` | `run` | canonical output computation |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

universe uOuterKey uAppState uWitness uDigest uTranscriptState

section

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows slotCount : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Raw active certificate. No output, completed context, equality proof, or
range proof is prover-supplied. -/
structure Certificate
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) where
  selected : Fin slotCount
  nifs :
    FixedActive.Certificate
      (contextAt setup input selected)

/-- Executable owners for equality checks whose carriers are intentionally
not required to have global `DecidableEq` instances. -/
structure Checkers
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount) where
  priorLinkCheck :
    Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount ->
      Bool
  priorLinkExact :
    forall input,
      priorLinkCheck input = true <->
        input.fresh.publicInput =
          machine.encodeInstance
            (machine.hash (Paper.priorHashPreimage input.toPaper))
  freshStructureCheck :
    Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount ->
      Fin slotCount -> Bool
  freshStructureExact :
    forall input selected,
      freshStructureCheck input selected = true <->
        input.fresh.constraintSystem =
          setup.expectedStructure input.verifierKey selected
  dispatchCheck :
    Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount ->
      Bool
  dispatchExact :
    forall input,
      dispatchCheck input = true <->
        machine.control input.zi input.witness =
          Paper.ProgramCounter.ofIndex functionIndex
  nifs :
    forall input selected,
      FixedActive.Evaluator.Checker
        (contextAt setup input selected)

namespace Checkers

@[simp] theorem priorLinkCheck_eq_true_iff
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    checkers.priorLinkCheck input = true <->
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)) :=
  checkers.priorLinkExact input

@[simp] theorem freshStructureCheck_eq_true_iff
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount) :
    checkers.freshStructureCheck input selected = true <->
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey selected :=
  checkers.freshStructureExact input selected

@[simp] theorem dispatchCheck_eq_true_iff
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    checkers.dispatchCheck input = true <->
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex :=
  checkers.dispatchExact input

end Checkers

/-- Exact meaning of the five outer checks before NIFS execution. -/
structure OuterChecks
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
    (selected : Fin slotCount) : Prop where
  iterationPositive : 0 < input.iteration
  priorSlot : input.priorPc = selected.val + 1
  priorPublicInput :
    input.fresh.publicInput =
      machine.encodeInstance
        (machine.hash (Paper.priorHashPreimage input.toPaper))
  expectedStructure :
    input.fresh.constraintSystem =
      setup.expectedStructure input.verifierKey selected
  dispatch :
    machine.control input.zi input.witness =
      Paper.ProgramCounter.ofIndex functionIndex

/-- One Boolean parent for the exact outer-check family. Only Nat propositions
use `decide`; the remaining decisions are owned by `Checkers`. -/
def outerCheck
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount) : Bool :=
  decide (0 < input.iteration) &&
    (decide (input.priorPc = selected.val + 1) &&
      (checkers.priorLinkCheck input &&
        (checkers.freshStructureCheck input selected &&
          checkers.dispatchCheck input)))

theorem outerCheck_eq_true_iff
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount) :
    outerCheck checkers input selected = true <->
      OuterChecks setup machine functionIndex input selected := by
  constructor
  · intro checked
    have parts :
        0 < input.iteration /\
          input.priorPc = selected.val + 1 /\
          input.fresh.publicInput =
            machine.encodeInstance
              (machine.hash (Paper.priorHashPreimage input.toPaper)) /\
          input.fresh.constraintSystem =
            setup.expectedStructure input.verifierKey selected /\
          machine.control input.zi input.witness =
            Paper.ProgramCounter.ofIndex functionIndex := by
      simpa only [outerCheck, Bool.and_eq_true, decide_eq_true_eq,
        checkers.priorLinkCheck_eq_true_iff,
        checkers.freshStructureCheck_eq_true_iff,
        checkers.dispatchCheck_eq_true_iff] using checked
    exact {
      iterationPositive := parts.1
      priorSlot := parts.2.1
      priorPublicInput := parts.2.2.1
      expectedStructure := parts.2.2.2.1
      dispatch := parts.2.2.2.2
    }
  · intro holds
    simp only [outerCheck, Bool.and_eq_true, decide_eq_true_eq,
      checkers.priorLinkCheck_eq_true_iff,
      checkers.freshStructureCheck_eq_true_iff,
      checkers.dispatchCheck_eq_true_iff]
    exact ⟨holds.iterationPositive, holds.priorSlot,
      holds.priorPublicInput, holds.expectedStructure, holds.dispatch⟩

/-- Complete physical meaning of one successful evaluator result. -/
structure PhysicalChecks
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
    (certificate : Certificate setup input)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount) : Prop where
  outer :
    OuterChecks setup machine functionIndex input certificate.selected
  nifsAccepted :
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Accepted
      (contextAt setup input certificate.selected) certificate.nifs
  resultExact :
    FixedActive.resultOf
        (contextAt setup input certificate.selected) certificate.nifs =
      selectedNext
  outputExact :
    output =
      outputOf machine input certificate.selected selectedNext

/-- Fail-closed active evaluator. The only returned value is canonical. -/
def run
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (certificate : Certificate setup input) :
    Option
      (Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount) :=
  if outerCheck checkers input certificate.selected then
    (FixedActive.Evaluator.run
      (checkers.nifs input certificate.selected) certificate.nifs).map
        (outputOf machine input certificate.selected)
  else
    none

/-- Exact characterization of executable success by named physical checks. -/
theorem run_eq_some_iff_physicalChecks
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (certificate : Certificate setup input)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount) :
    run checkers input certificate = some output <->
      exists selectedNext :
          Slot shape publicRingColumns publicFits verifierRows,
        PhysicalChecks setup machine functionIndex input certificate
          selectedNext output := by
  cases outerChecked : outerCheck checkers input certificate.selected with
  | false =>
      constructor
      · intro executed
        simp [run, outerChecked] at executed
      · rintro ⟨selectedNext, physical⟩
        have checked :
            outerCheck checkers input certificate.selected = true :=
          (outerCheck_eq_true_iff checkers input certificate.selected).2
            physical.outer
        simp [outerChecked] at checked
  | true =>
      have outer :
          OuterChecks setup machine functionIndex input certificate.selected :=
        (outerCheck_eq_true_iff checkers input certificate.selected).1
          outerChecked
      cases nifsExecuted :
          FixedActive.Evaluator.run
            (checkers.nifs input certificate.selected) certificate.nifs with
      | none =>
          constructor
          · intro executed
            simp [run, outerChecked, nifsExecuted] at executed
          · rintro ⟨selectedNext, physical⟩
            have executed :
                FixedActive.Evaluator.run
                    (checkers.nifs input certificate.selected)
                    certificate.nifs =
                  some selectedNext :=
              (FixedActive.Evaluator.run_eq_some_iff_accepted
                (checkers.nifs input certificate.selected) certificate.nifs
                selectedNext).2
                ⟨physical.nifsAccepted, physical.resultExact⟩
            rw [nifsExecuted] at executed
            contradiction
      | some result =>
          have nifsMeaning :=
            (FixedActive.Evaluator.run_eq_some_iff_accepted
              (checkers.nifs input certificate.selected) certificate.nifs
              result).1 nifsExecuted
          constructor
          · intro executed
            have outputEq :
                output =
                  outputOf machine input certificate.selected result := by
              simpa [run, outerChecked, nifsExecuted] using executed.symm
            exact ⟨result, {
              outer := outer
              nifsAccepted := nifsMeaning.1
              resultExact := nifsMeaning.2
              outputExact := outputEq
            }⟩
          · rintro ⟨selectedNext, physical⟩
            have selectedNextEq : selectedNext = result :=
              physical.resultExact.symm.trans nifsMeaning.2
            subst selectedNext
            simpa [run, outerChecked, nifsExecuted] using
              physical.outputExact.symm

/-- Physical success refines the independent semantics or exposes exactly the
existing output-binding or Split-NC failure. No bad-event exclusion is hidden
in this theorem. -/
theorem run_sound_or_outputUnbound_or_piCcsBadEvent
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (certificate : Certificate setup input)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount)
    (data : Data shape)
    (semanticInput :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticInput
        (contextAt setup input certificate.selected) data)
    (childOpenings :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildOpenings
        (contextAt setup input certificate.selected) data certificate.nifs)
    (executed : run checkers input certificate = some output) :
    ActiveSemantics.Holds setup machine functionIndex input output \/
      ¬
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.OutputBound
          (contextAt setup input certificate.selected) data certificate.nifs) \/
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsBadEvent
        (contextAt setup input certificate.selected) data certificate.nifs := by
  rcases (run_eq_some_iff_physicalChecks checkers input certificate output).1
      executed with ⟨selectedNext, physical⟩
  have nifsExecuted :
      FixedActive.Evaluator.run
          (checkers.nifs input certificate.selected) certificate.nifs =
        some selectedNext :=
    (FixedActive.Evaluator.run_eq_some_iff_accepted
      (checkers.nifs input certificate.selected) certificate.nifs
      selectedNext).2
      ⟨physical.nifsAccepted, physical.resultExact⟩
  rcases FixedActive.Evaluator.run_sound noZeroDivisors semanticInput
      childOpenings nifsExecuted with
    transition | outputUnbound | bad
  · exact Or.inl ⟨certificate.selected, selectedNext, {
      iterationPositive := physical.outer.iterationPositive
      priorSlot := physical.outer.priorSlot
      priorPublicInput := physical.outer.priorPublicInput
      expectedStructure := physical.outer.expectedStructure
      selectedNifs := transition
      dispatch := physical.outer.dispatch
    }, physical.outputExact⟩
  · exact Or.inr (Or.inl outputUnbound)
  · exact Or.inr (Or.inr bad)

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
