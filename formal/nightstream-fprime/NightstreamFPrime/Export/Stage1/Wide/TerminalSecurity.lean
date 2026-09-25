import NightstreamFPrime.Export.Stage1.SecurityInstance
import NightstreamFPrime.Export.Stage1.Wide.OpeningBinding
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputWitnessConsumer

/-! Owns what one accepted recursive terminal of a wide-sampler package says
about its decoded step: the complete next preimage, the HyperNova step, the
wide-key NIFS output, the PiDEC parent opening and the accepted predecessor.
Each conclusion either holds or gives the named state-hash collision.

A `Target` fixes the application, compiled sampler plan, carrier fit, Ajtai
key and verifier context digest. The selected package supplies one value.
Source openings for a recursive predecessor stay an explicit premise; their
extraction and every probability bound belong to the history layer. -/

namespace NightstreamFPrime.Export.Stage1.Wide

open NightstreamFPrime.Spec NightstreamFPrime.Layout NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint
open Spec.HyperNova.Construction2.Paper Spec.HyperNova.NonInteractiveMultiFold

section
-- The structure command reduces field types; the width stays symbolic.
attribute [local irreducible] RetainedLayout.logicalWidth

/-- One wide-sampler package statement. The context digest is the verifier
key digest that the terminal public check binds; its length is the only
premise that the collision reduction needs. -/
structure Target where
  program : RetainedLayout.Program
  compiled : PiRlcWideSampler.RangePlan.Compiled
  fits : PerApplicationFixedPoint.FitsTwoPow28 program
  ajtai : AjtaiKey (logicalWidth := RetainedLayout.logicalWidth program)
    (publicFits := FixedPoint.publicFits program)
  context : KeyDigest
  contextLength : context.length = 4

end

/-- The selected package's target. Its prepared stream supplies the context. -/
def selected (compiled : PiRlcWideSampler.RangePlan.Compiled) (parts : AuthorityStream.Parts) :
    Target where
  program := SetupBinding.application
  compiled := compiled
  fits := SetupBinding.fits
  ajtai := SetupBinding.productionAjtaiKey
  context := SetupBinding.contextKey (SetupBinding.descriptor parts)
  contextLength := SetupBinding.contextKey_length _

namespace Target

variable (target : Target)

abbrev logicalWidth : Nat := RetainedLayout.logicalWidth target.program

abbrev publicFits := FixedPoint.publicFits target.program

abbrev relation := FixedPoint.relation target.program target.compiled target.fits

/-- The NIFS extraction statement reads only the width, relation and key. -/
abbrev security : SecurityInstance where
  logicalWidth := target.logicalWidth
  publicFits := target.publicFits
  relation := target.relation
  ajtai := target.ajtai

abbrev Payload := OpeningBinding.TerminalPayload target.program

abbrev Envelope := Lifecycle.Stage1.Terminal.ProofEnvelope
  (logicalWidth := target.logicalWidth) (publicFits := target.publicFits)

/-- Acceptance of one terminal statement by this package's verifier. -/
abbrev Holds (statement : TerminalStatement AppState) (proof : target.Envelope) : Prop :=
  Lifecycle.Stage1.Terminal.HoldsFor target.relation target.ajtai target.context target.program
    statement proof

/-- The logical fresh assignment of one terminal opening. -/
abbrev assignment (payload : target.Payload) :=
  ProductionRelation.Plan.logicalAssignment payload.freshWitness

/-- The step input decoded from the fresh opening, with no honest-witness premise. -/
def decodedInput (payload : target.Payload) :=
  FixedPointSoundness.input target.program (target.assignment payload) target.relation

/-- The step output decoded from the same opening. -/
def decodedOutput (payload : target.Payload) :=
  FixedPointSoundness.output target.program (target.assignment payload) target.logicalWidth
    target.publicFits

/-- The next preimage whose hash the accepted rows constrain. -/
noncomputable def decodedNext (payload : target.Payload) :=
  ContextBinding.decodedNext target.program (target.assignment payload) target.compiled target.fits
    target.ajtai

/-- The preimage that the terminal public check hashes. -/
def preimage (statement : TerminalStatement AppState) (payload : target.Payload) :=
  OpeningBinding.terminalPreimage target.program target.context statement payload

/-- The named state-hash collision at one terminal opening. -/
def Collision (statement : TerminalStatement AppState) (payload : target.Payload) : Prop :=
  PiCCSSecurity.StateHashCollision (target.decodedNext payload) (target.preimage statement payload)

/-- The accepted terminal identifies the complete next preimage, including
its natural iteration, or gives the named state-hash collision. -/
theorem terminal_implies_preimageOrCollision (statement : TerminalStatement AppState)
    (payload : target.Payload) (terminal : target.Holds statement (.recursive payload)) :
    target.decodedNext payload = target.preimage statement payload ∨
      target.Collision statement payload := by
  rcases (Lifecycle.Stage1.Terminal.holdsFor_recursive_iff target.relation target.ajtai target.context
    target.program statement payload).mp terminal with
    ⟨valid, pcValid, positive, publicLink, _runningValid, freshValid⟩
  let actual := target.decodedNext payload
  let expected := target.preimage statement payload
  obtain ⟨rows, publicEqual⟩ := OpeningBinding.freshHolds_implies_rowsAndPublic target.program
    target.compiled target.fits target.ajtai payload.fresh payload.freshWitness freshValid
  change payload.fresh.publicInputs ⟨0, by decide⟩ = encHash (stateHash expected) at publicLink
  have accepted := PublicBinding.step target.program (target.assignment payload) target.compiled
    target.fits target.ajtai (stateHash expected) (StateEncoding.stateHash_length expected)
    (publicEqual.trans publicLink) rows
  have outputHash : FixedPointSoundness.digest target.program (target.assignment payload) =
      stateHash actual := accepted.1.2.2.1
  have hashes : stateHash expected = stateHash actual := accepted.2.symm.trans outputHash
  change actual = expected ∨ PiCCSSecurity.StateHashCollision actual expected
  by_cases encodedEqual : serializePreimage actual = serializePreimage expected
  · apply Or.inl
    let prior := FixedPointSoundness.priorState target.program (target.assignment payload)
    have keyLength : (actual.verifierKeys functionIndex).length =
        (expected.verifierKeys functionIndex).length := by
      change (StateDecoder.keyDigest prior).length = target.context.length
      simp only [StateDecoder.keyDigest, StateDecoder.slice, List.length_ofFn, target.contextLength]
      rfl
    have counterWord := StateEncoding.serializePreimage_eq_implies_iteration_word_eq
      actual expected keyLength encodedEqual
    have priorBound := (StateDecoder.preimage_wellFormed target.logicalWidth
      target.publicFits prior).2.1
    have iteration := StateEncoding.natWord_successor_eq_below_modulus
      (StateDecoder.iteration prior) statement.iteration priorBound positive valid.1 counterWord
    have actualWellFormed : StateEncoding.WellFormed actual := by
      refine ⟨?_, ?_, rfl⟩
      · refine ⟨?_, ?_, ?_⟩ <;>
          simp [actual, decodedNext, ContextBinding.decodedNext, nextHashPreimage,
            Lifecycle.Stage1.Wide.Relation.setup, FixedPointSoundness.contextKey,
            FixedPointSoundness.input, FixedPointSoundness.output, StateDecoder.keyDigest,
            StateDecoder.slice, PilotProduction.digestWords, PilotValues.digestWords,
            Lifecycle.Stage1.Application.stateWordCount]
      · change StateDecoder.iteration prior + 1 < goldilocksModulus
        rw [iteration]
        exact valid.1
    have expectedWellFormed : StateEncoding.WellFormed expected := by
      refine ⟨⟨?_, valid.2.1, valid.2.2⟩, valid.1, ?_⟩
      · change target.context.length = PilotProduction.digestWords
        rw [target.contextLength]
        rfl
      · change payload.pc = 1
        change 1 ≤ payload.pc ∧ payload.pc ≤ 1 at pcValid
        omega
    exact StateEncoding.serializePreimage_injective actualWellFormed expectedWellFormed encodedEqual
  · exact Or.inr ⟨encodedEqual, hashes.symm⟩

/-- The decoded HyperNova step reaches the advertised terminal preimage. -/
theorem terminal_implies_matchingStepOrCollision (statement : TerminalStatement AppState)
    (payload : target.Payload) (terminal : target.Holds statement (.recursive payload)) :
    (Lifecycle.Stage1.Wide.Relation.StepHoldsFor target.relation target.ajtai target.context
        target.program (target.decodedInput payload) (target.decodedOutput payload) ∧
      target.decodedNext payload = target.preimage statement payload) ∨
      target.Collision statement payload := by
  rcases terminal_implies_preimageOrCollision target statement payload terminal with same | collision
  · rcases OpeningBinding.terminal_implies_stepOrCollision target.program target.compiled target.fits
        target.ajtai target.context target.contextLength statement payload terminal with step | collision
    · exact Or.inl ⟨step, same⟩
    · exact False.elim (collision.1 (congrArg serializePreimage same))
  · exact Or.inr collision

/-- A recursive decoded step calls the wide-key NIFS verifier on the decoded
input and proof, and its output is the terminal's running claim. The base
step makes no NIFS call. -/
theorem terminal_implies_nifsOrBaseOrCollision (statement : TerminalStatement AppState)
    (payload : target.Payload) (terminal : target.Holds statement (.recursive payload)) :
    let input := target.decodedInput payload
    (input.iteration = 0 ∨
      (0 < input.iteration ∧
        Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key target.relation target.ajtai)
          (input.running functionIndex) input.fresh input.nifsProof =
            some (payload.running functionIndex))) ∨
      target.Collision statement payload := by
  rcases terminal_implies_matchingStepOrCollision target statement payload terminal with
    ⟨step, same⟩ | collision
  · apply Or.inl
    unfold Lifecycle.Stage1.Wide.Relation.StepHoldsFor at step
    rcases step.2.2.2 with base | recursive
    · exact Or.inl base.1
    · rcases recursive with ⟨priorPcValid, positive, _priorPublic, selectedNifs, _unchanged⟩
      have selected : selectedIndex priorPcValid = functionIndex := by
        apply Fin.ext
        have bound := (selectedIndex priorPcValid).isLt
        change (selectedIndex priorPcValid).val < 1 at bound
        change (selectedIndex priorPcValid).val = 0
        omega
      rw [selected] at selectedNifs
      have checked : Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key target.relation target.ajtai)
          ((target.decodedInput payload).running functionIndex) (target.decodedInput payload).fresh
          (target.decodedInput payload).nifsProof =
            some ((target.decodedOutput payload).runningNext functionIndex) := by
        simpa [Accepts, Lifecycle.Stage1.Wide.Relation.setup,
          Lifecycle.Stage1.Wide.Relation.nifsVerifier] using selectedNifs
      have outputSame : (target.decodedOutput payload).runningNext functionIndex =
          payload.running functionIndex :=
        congrArg (fun preimage => preimage.running functionIndex) same
      exact Or.inr ⟨positive, checked.trans (congrArg some outputSame)⟩
  · exact Or.inr collision

/-- The terminal's running witnesses open the recomposed PiDEC parent of the
decoded recursive proof, unless the step is the base step or the named
collision occurs. No output-match or child-opening premise is added. -/
theorem terminal_implies_parentOrBaseOrCollision (statement : TerminalStatement AppState)
    (payload : target.Payload) (terminal : target.Holds statement (.recursive payload)) :
    let input := target.decodedInput payload
    let key := PiRLC.Wide.Key.key target.relation target.ajtai
    input.iteration = 0 ∨
      (0 < input.iteration ∧ ∃ attempt,
        key.piDecAttempt (input.running functionIndex) input.fresh input.nifsProof = some attempt ∧
        CE.Holds (semantics target.ajtai) productionGlobalParams attempt.parent
          ((PaperAlgebra.piDecAlgebra target.ajtai).recomposeAssignment
            (payload.runningWitness functionIndex))) ∨
      target.Collision statement payload := by
  let input := target.decodedInput payload
  let key := PiRLC.Wide.Key.key target.relation target.ajtai
  rcases terminal_implies_nifsOrBaseOrCollision target statement payload terminal with
    (base | ⟨positive, accepted⟩) | collision
  · exact Or.inl base
  · have checks := (Nifs.PaperNonInteractive.verify_eq_some_iff key
      (input.running functionIndex) input.fresh input.nifsProof
      (payload.running functionIndex)).mp accepted
    rcases (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff key
      (input.running functionIndex) input.fresh input.nifsProof).mp checks.2.1 with
      ⟨attempt, attemptEq, _attemptAccepted⟩
    rcases (Lifecycle.Stage1.Terminal.holdsFor_recursive_iff target.relation target.ajtai
      target.context target.program statement payload).mp terminal with
      ⟨_valid, _pcValid, _positive, _publicLink, runningValid, _freshValid⟩
    exact Or.inr (Or.inl ⟨positive, attempt, attemptEq,
      Spec.Folding.PiDEC.OutputWitnessConsumer.accepted_outputWitness_extracts_parent key
        (input.running functionIndex) input.fresh input.nifsProof (payload.running functionIndex)
        attempt attemptEq accepted (payload.runningWitness functionIndex)
        (runningValid functionIndex)⟩)
  · exact Or.inr (Or.inr collision)

end Target

private theorem step_implies_predecessor {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : KeyDigest) (application : Lifecycle.Stage1.Application.Program)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (runningWitness : Lifecycle.Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : Lifecycle.Stage1.Terminal.FreshWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (valid : Lifecycle.Stage1.Terminal.StatementValid
      { iteration := input.iteration, z0 := input.z0, zi := input.zi })
    (step : Lifecycle.Stage1.Wide.Relation.StepHoldsFor relation ajtai context application input output)
    (sources : 0 < input.iteration → Lifecycle.TerminalHolds relation ajtai
      (input.running functionIndex) runningWitness input.fresh freshWitness) :
    Lifecycle.Stage1.Terminal.HoldsFor relation ajtai context application
      { iteration := input.iteration, z0 := input.z0, zi := input.zi }
      (if input.iteration = 0 then .bottom else .recursive {
        running := input.running
        runningWitness := fun _ => runningWitness
        fresh := input.fresh
        freshWitness := freshWitness
        pc := input.priorPc }) := by
  unfold Lifecycle.Stage1.Wide.Relation.StepHoldsFor at step
  rcases step.2.2.2 with base | recursive
  · rw [if_pos base.1]
    exact (Lifecycle.Stage1.Terminal.holdsFor_bottom_iff relation ajtai context application _).mpr
      ⟨valid, base.1, base.2.1.symm⟩
  · rcases recursive with ⟨pcValid, positive, publicLink, _fold, _unchanged⟩
    rw [if_neg (Nat.ne_of_gt positive)]
    apply (Lifecycle.Stage1.Terminal.holdsFor_recursive_iff relation ajtai context application _ _).mpr
    refine ⟨valid, pcValid, positive, publicLink, ?_, ?_⟩
    · intro slot
      have selected : slot = functionIndex := by
        apply Fin.ext
        have bound := slot.isLt
        change slot.val < 1 at bound
        change slot.val = 0
        omega
      rw [selected]
      exact (sources positive).1
    · exact (sources positive).2

namespace Target

variable (target : Target)

/-- The predecessor statement that the decoded step advertises. -/
theorem predecessor_valid (payload : target.Payload) :
    let input := target.decodedInput payload
    Lifecycle.Stage1.Terminal.StatementValid
        { iteration := input.iteration, z0 := input.z0, zi := input.zi } := by
  let prior := FixedPointSoundness.priorState target.program (target.assignment payload)
  refine ⟨?_, ?_, ?_⟩
  · exact (StateDecoder.preimage_wellFormed target.logicalWidth target.publicFits
      prior).2.1
  · exact StateDecoder.initialState_length prior
  · exact StateDecoder.currentState_length prior

/-- Valid openings for the exact decoded source instances construct an
accepted predecessor. The success fields identify its counter, initial
state, recovered application transition and terminal proof. The opening
premise is used only when the decoded predecessor counter is positive. -/
theorem terminal_implies_predecessorOrCollision (statement : TerminalStatement AppState)
    (payload : target.Payload)
    (runningWitness : Lifecycle.Stage1.Terminal.RunningWitness
      (logicalWidth := target.logicalWidth) (publicFits := target.publicFits))
    (freshWitness : Lifecycle.Stage1.Terminal.FreshWitness
      (logicalWidth := target.logicalWidth) (publicFits := target.publicFits))
    (terminal : target.Holds statement (.recursive payload))
    (sourceHolds : 0 < (target.decodedInput payload).iteration → Lifecycle.TerminalHolds
      target.relation target.ajtai ((target.decodedInput payload).running functionIndex)
      runningWitness (target.decodedInput payload).fresh freshWitness) :
    let input := target.decodedInput payload
    (input.iteration + 1 = statement.iteration ∧
      input.z0 = statement.z0 ∧
      statement.zi = target.program.step input.zi input.witness ∧
      target.Holds { iteration := input.iteration, z0 := input.z0, zi := input.zi }
        (if input.iteration = 0 then .bottom else .recursive {
          running := input.running
          runningWitness := fun _ => runningWitness
          fresh := input.fresh
          freshWitness := freshWitness
          pc := input.priorPc })) ∨
      target.Collision statement payload := by
  rcases terminal_implies_matchingStepOrCollision target statement payload terminal with
    ⟨step, same⟩ | collision
  · apply Or.inl
    have iteration : (target.decodedInput payload).iteration + 1 = statement.iteration :=
      congrArg (fun preimage => preimage.iteration) same
    have initial : (target.decodedInput payload).z0 = statement.z0 :=
      congrArg (fun preimage => preimage.z0) same
    have current : (target.decodedOutput payload).zNext = statement.zi :=
      congrArg (fun preimage => preimage.current) same
    have applicationStep : (target.decodedOutput payload).zNext =
        target.program.step (target.decodedInput payload).zi (target.decodedInput payload).witness :=
      step.2.1
    exact ⟨iteration, initial, current.symm.trans applicationStep,
      step_implies_predecessor target.relation target.ajtai target.context target.program
        (target.decodedInput payload) (target.decodedOutput payload) runningWitness freshWitness
        (predecessor_valid target payload) step sourceHolds⟩
  · exact Or.inr collision

/-- Reverse iteration one uses the base step, recovers its application
witness and accepts the unique bottom predecessor. It needs no NIFS call,
extracted source witness or source-opening premise. -/
theorem terminal_one_implies_baseOrCollision (statement : TerminalStatement AppState)
    (payload : target.Payload) (first : statement.iteration = 1)
    (terminal : target.Holds statement (.recursive payload)) :
    (target.Holds { iteration := 0, z0 := statement.z0, zi := statement.z0 } .bottom ∧
      statement.zi = target.program.step statement.z0 (target.decodedInput payload).witness) ∨
      target.Collision statement payload := by
  rcases terminal_implies_matchingStepOrCollision target statement payload terminal with
    ⟨step, same⟩ | collision
  · apply Or.inl
    have iteration : (target.decodedInput payload).iteration + 1 = statement.iteration :=
      congrArg (fun preimage => preimage.iteration) same
    have zero : (target.decodedInput payload).iteration = 0 := by omega
    have initial : (target.decodedInput payload).z0 = statement.z0 :=
      congrArg (fun preimage => preimage.z0) same
    have current : (target.decodedOutput payload).zNext = statement.zi :=
      congrArg (fun preimage => preimage.current) same
    have applicationStep : (target.decodedOutput payload).zNext =
        target.program.step (target.decodedInput payload).zi (target.decodedInput payload).witness :=
      step.2.1
    unfold Lifecycle.Stage1.Wide.Relation.StepHoldsFor at step
    have baseState : (target.decodedInput payload).z0 = (target.decodedInput payload).zi := by
      rcases step.2.2.2 with base | recursive
      · exact base.2.1
      · rcases recursive with ⟨_pcValid, positive, _⟩
        exact False.elim ((Nat.ne_of_gt positive) zero)
    have priorState : (target.decodedInput payload).zi = statement.z0 := baseState.symm.trans initial
    refine ⟨?_, ?_⟩
    · apply (Lifecycle.Stage1.Terminal.holdsFor_bottom_iff target.relation target.ajtai target.context
        target.program _).mpr
      refine ⟨⟨?_, terminal.1.2.1, terminal.1.2.1⟩, rfl, rfl⟩
      have bound := terminal.1.1
      change 0 < goldilocksModulus
      omega
    · exact current.symm.trans (by simpa only [priorState] using applicationStep)
  · exact Or.inr collision

end Target

end NightstreamFPrime.Export.Stage1.Wide
