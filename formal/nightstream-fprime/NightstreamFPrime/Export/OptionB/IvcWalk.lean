import NightstreamFPrime.Export.OptionB.TerminalBridge
import NightstreamFPrime.Lifecycle.OptionB.PriorLink
import NightstreamFPrime.Spec.HyperNova.ReverseStages

/-! The reverse walk of HyperNova Lemma 17 (Appendix H.3) for option B. A visit is a terminal
statement, a proof and one witness bound per slot. At a recursive visit, the fresh witness decodes
to the last step (`TerminalBridge.Exports.last`). When that step is recursive, it folds the slot
that its prior program counter selects, and the visit's attempt is that fold, linked to the
step's prior hash preimage (`Lifecycle.OptionB.PriorLink`), with the slot's witness at the slot's
bound (`attemptOf`). An answer that extracts the fold opens the slot's input claim and the fresh
instance, and it moves the walk to the prior state (`back`).

A visit is relaxed when the bridge's premises hold, every slot claim opens at its slot's bound,
and each bound is within `B` plus `T(b − 1)` per light fold that the slot can still take
(`Relaxed`). A light predecessor opens only at its successor's bound plus `T(b − 1)`, so the
bounds are per slot. `walk_step`: at a relaxed visit without a collision, an answer that
extracts gives a relaxed prior visit one iteration back, or the visit's step is the base step.
`covered`: so a false acceptance has a collision or a failed extraction at one of its first
`iteration` visits. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.IvcWalk

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthReverse
  (Attempt Succeeds Extracted chainBound)
open NightstreamFPrime.Spec.HyperNova.ReverseStages
  (visitAfter StageFails visitAfter_cons stageFails_cons)
open NightstreamFPrime.Spec.Folding.Schedule (extractedBefore nextDepth productionLightFoldCap)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Export.OptionB.StepRelation
open NightstreamFPrime.Export.OptionB.TerminalBridge

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

abbrev Payload (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  TerminalProof (SlotData logicalWidth publicFits)
    (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
    (StepFresh logicalWidth publicFits)
    (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2

/-- A visit of the walk: a terminal statement, a proof and one witness bound per slot. -/
structure Visit (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  statement : TerminalStatement AppState
  proof : Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits)
  bounds : Fin 2 → Nat

abbrev WalkAttempt (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Attempt (PriorPreimage logicalWidth publicFits) (CircuitKey logicalWidth publicFits)
    (SlotClaim logicalWidth publicFits) (StepFresh logicalWidth publicFits) StepProof
    (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- An extractor's answer: a witness of the fold's input claim and one of its fresh instance. -/
abbrev Answer (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits) ×
    Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)

/-- The walk's visit for an IVC adversary's output: every slot at bound `B`. -/
def initial (output : TerminalStatement AppState ×
    Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Visit logicalWidth publicFits :=
  ⟨output.1, output.2, fun _ => productionGlobalParams.bigB⟩

/-- A slot's bound is `B`, or the slot's depth is a field value and the bound leaves room for
the light folds that the slot can still take. -/
def DepthWithin (bound depth : Nat) : Prop :=
  bound ≤ productionGlobalParams.bigB ∨
    (depth < goldilocksModulus ∧ bound ≤ productionGlobalParams.bigB +
      (productionLightFoldCap - depth) *
        (productionGlobalParams.expansionT * (productionGlobalParams.b - 1)))

/-- Each witness of `advice` is valid for `program` at the state that the earlier witnesses
reach from `state`. -/
def ValidFrom (program : Stage1.Application.Program) : AppState → List AppWitness → Prop
  | _, [] => True
  | state, witness :: rest =>
      program.valid state witness ∧ ValidFrom program (program.step state witness) rest

theorem validFrom_append (program : Stage1.Application.Program) :
    ∀ (state : AppState) (first second : List AppWitness),
      ValidFrom program state (first ++ second) ↔
        ValidFrom program state first ∧
          ValidFrom program (first.foldl program.step state) second
  | _, [], _ => by simp [ValidFrom]
  | state, witness :: rest, second => by
      simp only [List.cons_append, ValidFrom, List.foldl_cons, validFrom_append program _ rest second,
        and_assoc]

/-- `iteration` valid steps of `program`, each with a witness of the program's word count, lead
from `z0` to `zi`. -/
def Reaches (program : Stage1.Application.Program) (statement : TerminalStatement AppState) :
    Prop :=
  ∃ advice : List AppWitness, advice.length = statement.iteration ∧
    (∀ witness ∈ advice, witness.length = program.witnessWordCount) ∧
    ValidFrom program statement.z0 advice ∧ advice.foldl program.step statement.z0 = statement.zi

/-- The decoded steps, from the last back, are valid steps of `program` that lead from `z0` to
`zi` of `statement` and end at the base step. -/
def Leads (program : Stage1.Application.Program) :
    TerminalStatement AppState → List (Decoded logicalWidth publicFits) → Prop
  | statement, [] => statement.iteration = 0 ∧ statement.zi = statement.z0
  | statement, decoded :: older =>
      program.valid decoded.input.zi decoded.input.witness ∧
        program.step decoded.input.zi decoded.input.witness = statement.zi ∧
        decoded.input.z0 = statement.z0 ∧ decoded.input.iteration + 1 = statement.iteration ∧
        Leads program ⟨decoded.input.iteration, decoded.input.z0, decoded.input.zi⟩ older

/-- The witnesses of decoded steps that lead to `statement`, each of the program's word count,
reach it. -/
theorem reaches_of_leads (program : Stage1.Application.Program) :
    ∀ (statement : TerminalStatement AppState) (steps : List (Decoded logicalWidth publicFits)),
      Leads program statement steps →
      (∀ decoded ∈ steps, decoded.input.witness.length = program.witnessWordCount) →
      Reaches program statement
  | _, [], ⟨zero, same⟩, _ => ⟨[], zero.symm, by simp, trivial, same.symm⟩
  | statement, decoded :: older, ⟨valid, stepped, z0Eq, iterationEq, earlier⟩, lengths => by
      obtain ⟨advice, length, adviceLengths, validFrom, folded⟩ :=
        reaches_of_leads program _ older earlier fun other inside =>
          lengths other (List.mem_cons_of_mem _ inside)
      refine ⟨advice ++ [decoded.input.witness], ?_, ?_, ?_, ?_⟩
      · simp only [List.length_append, List.length_singleton, length]
        exact iterationEq
      · intro witness inside
        rcases List.mem_append.mp inside with old | new
        · exact adviceLengths witness old
        · rw [List.mem_singleton.mp new]
          exact lengths decoded (by simp)
      · rw [validFrom_append]
        dsimp only at validFrom folded
        rw [← z0Eq, folded]
        exact ⟨validFrom, valid, trivial⟩
      · dsimp only at folded
        rw [List.foldl_append, ← z0Eq, folded]
        exact stepped

theorem cap_lt_modulus : productionLightFoldCap < goldilocksModulus := by
  decide

/-- A light fold: the slot's bound before the fold leaves room for one more light fold. -/
theorem depthWithin_light {bound depth payloadDepth : Nat} (guard : depth < productionLightFoldCap)
    (words : natWord (depth + 1) = natWord payloadDepth) (within : DepthWithin bound payloadDepth) :
    DepthWithin (bound + productionGlobalParams.expansionT * (productionGlobalParams.b - 1))
      depth := by
  have depthBound : depth < goldilocksModulus := lt_trans guard cap_lt_modulus
  have room : productionLightFoldCap - depth = (productionLightFoldCap - (depth + 1)) + 1 := by
    omega
  refine Or.inr ⟨depthBound, ?_⟩
  rw [room, Nat.add_mul, Nat.one_mul]
  rcases within with small | ⟨payloadBound, large⟩
  · have := Nat.zero_le ((productionLightFoldCap - (depth + 1)) *
      (productionGlobalParams.expansionT * (productionGlobalParams.b - 1)))
    omega
  · have same := natWord_injective (by have := cap_lt_modulus; omega) payloadBound words
    rw [same]
    omega

/-- A slot that the step does not fold keeps its bound. -/
theorem depthWithin_kept {bound depth payloadDepth : Nat} (depthBound : depth < goldilocksModulus)
    (words : natWord depth = natWord payloadDepth) (within : DepthWithin bound payloadDepth) :
    DepthWithin bound depth := by
  rcases within with small | ⟨payloadBound, large⟩
  · exact Or.inl small
  · rw [natWord_injective depthBound payloadBound words]
    exact Or.inr ⟨payloadBound, large⟩

theorem circuitOf_selected {pc : Nat} (valid : InRange 2 pc) :
    circuitOf pc = selectedIndex valid := by
  obtain ⟨lower, upper⟩ := valid
  apply Fin.ext
  simp only [selectedIndex, circuitOf]
  split <;> simp <;> omega

section Walk

variable {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  {step : AppState → AppWitness → AppState}
  {cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  (exports : Exports relation ajtai circuitKeys step cubeFits)

/-- A slot claim opens at norm `bound` and its evaluations are true: the CE relation of the
slot's key with `bound` in place of `B`. -/
def ClaimHolds (key : CircuitKey logicalWidth publicFits) (bound : Nat)
    (claim : SlotClaim logicalWidth publicFits)
    (witness : Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Prop :=
  let statement := toClaim (key.system ajtai) claim
  Opening.Holds (PaperAlgebra.semantics ajtai) bound statement.commitment statement.publicInput
      witness ∧
    (PaperAlgebra.semantics ajtai).evaluationPointValid statement.constraintSystem
      statement.point ∧
    (PaperAlgebra.semantics ajtai).evaluations statement.constraintSystem witness
      statement.point = statement.evaluations

/-- At bound `B`, `ClaimHolds` is the terminal's decider check. -/
theorem claimHolds_bigB (key : CircuitKey logicalWidth publicFits)
    (claim : SlotClaim logicalWidth publicFits)
    (witness : Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    ClaimHolds ajtai key productionGlobalParams.bigB claim witness ↔
      CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (toClaim (key.system ajtai) claim) witness :=
  Iff.rfl

/-- A fresh instance holds under the relation of `key`. -/
def FreshHolds (key : CircuitKey logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (witness : Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Prop :=
  CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
    (freshStatement key.relation fresh) witness

/-- The NIFS success event of the walk: the bounded-depth fold verifier with the light cap,
option B's prior link, and `ClaimHolds`. -/
def WalkSucceeds (attempt : WalkAttempt logicalWidth publicFits) : Prop :=
  Succeeds productionGlobalParams productionLightFoldCap (slotFolds ajtai) PriorLink
    (ClaimHolds ajtai) attempt

def WalkExtracted (attempt : WalkAttempt logicalWidth publicFits)
    (answer : Answer logicalWidth publicFits) : Prop :=
  Extracted productionGlobalParams (ClaimHolds ajtai) (FreshHolds ajtai) attempt answer

/-- The prior hash preimage of a decoded step (HyperNova Construction 2, step 4(b)). -/
def priorOf (input : StepInput logicalWidth publicFits) : PriorPreimage logicalWidth publicFits where
  verifierKeys := circuitKeys
  iteration := input.iteration
  z0 := input.z0
  current := input.zi
  running := input.running
  pc := input.priorPc

/-- The fold of a recursive visit's last step: the circuit that ran last folds the slot that the
step's prior program counter selects, under that slot's key. -/
def foldOf (payload : Payload logicalWidth publicFits) : StepFold logicalWidth publicFits :=
  let decoded := exports.last payload
  let slot := circuitOf decoded.input.priorPc
  { circuit := circuitOf payload.pc
    key := circuitKeys slot
    input := (decoded.input.running slot).1
    fresh := decoded.input.fresh
    proof := decoded.input.nifsProof
    output := decoded.output.runningNext slot }

/-- The attempt of a visit whose last step is recursive. -/
def attemptOf (visit : Visit logicalWidth publicFits) : Option (WalkAttempt logicalWidth publicFits) :=
  match visit.proof with
  | .bottom => none
  | .recursive payload =>
      let input := (exports.last payload).input
      let slot := circuitOf input.priorPc
      if 0 < input.iteration then
        some ⟨⟨priorOf circuitKeys input, input.running slot, foldOf ajtai circuitKeys exports payload⟩,
          visit.bounds slot, payload.runningWitness slot⟩
      else none

/-- The prior visit: the last step's input as the statement and the slots, the answer's witnesses
for the folded slot and the fresh instance, and the folded slot's recovered bound. -/
def back (visit : Visit logicalWidth publicFits) (answer : Answer logicalWidth publicFits) :
    Visit logicalWidth publicFits :=
  match visit.proof with
  | .bottom => visit
  | .recursive payload =>
      let input := (exports.last payload).input
      let slot := circuitOf input.priorPc
      if 0 < input.iteration then
        { statement := ⟨input.iteration, input.z0, input.zi⟩
          proof := .recursive
            ⟨input.running, Function.update payload.runningWitness slot answer.1, input.fresh,
              answer.2, input.priorPc⟩
          bounds := Function.update visit.bounds slot
            (extractedBefore productionGlobalParams
              (BoundedDepth.circuitStep (circuitOf payload.pc)) (visit.bounds slot)) }
      else visit

/-- The bridge's collision at a recursive visit. -/
def WalkCollision (visit : Visit logicalWidth publicFits) : Prop :=
  match visit.proof with
  | .bottom => False
  | .recursive payload => Collision exports visit.statement payload

/-- The walk's invariant. -/
def Relaxed (visit : Visit logicalWidth publicFits) : Prop :=
  match visit.proof with
  | .bottom => visit.statement.iteration = 0 ∧ visit.statement.zi = visit.statement.z0
  | .recursive payload =>
      Linked (ajtai := ajtai) (circuitKeys := circuitKeys) visit.statement payload ∧
        (∀ slot, ClaimHolds ajtai (circuitKeys slot) (visit.bounds slot) (payload.running slot).1
          (payload.runningWitness slot)) ∧
        ∀ slot, DepthWithin (visit.bounds slot) (payload.running slot).2

/-- A decoded step's counters and slot depths are field values, and its states have four
words. -/
structure Shaped (decoded : Decoded logicalWidth publicFits) : Prop where
  iteration : decoded.input.iteration < goldilocksModulus
  priorPc : decoded.input.priorPc < goldilocksModulus
  z0 : decoded.input.z0.length = 4
  zi : decoded.input.zi.length = 4
  priorDepth : ∀ slot, (decoded.input.running slot).2 < goldilocksModulus
  nextDepth : ∀ slot, (decoded.output.runningNext slot).2 < goldilocksModulus

theorem decode_shaped (circuit : Fin 2)
    (witness : Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Fin 2 → SlotData logicalWidth publicFits) :
    Shaped (exports.decode circuit witness running) := by
  unfold Exports.decode
  split
  · exact ⟨Fin.isLt _, Fin.isLt _, by simp [lightInput], by simp [lightInput],
      lightInput_depth exports.program running _,
      DigestSelect.nextSlots_depth _ _ _ _ (lightInput_depth exports.program running _)
        (Fin.isLt _)⟩
  · have represents := refreshRepresents exports.program relation ajtai circuitKeys 1
      (refreshEnv exports.program relation ajtai exports.refreshLayout
        (Layout.ProductionRelation.Plan.logicalAssignment witness))
    exact ⟨represents.iterationBound, represents.priorPcBound, by simp [refreshInput],
      by simp [refreshInput], represents.priorDepth,
      DigestSelect.nextSlots_depth _ _ _ _ represents.priorDepth (Fin.isLt _)⟩

theorem initial_relaxed (control : AppState → AppWitness → Fin 2) (cap : Nat)
    (output : TerminalStatement AppState ×
      Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits))
    (accepted : Terminal.HoldsFor cap ajtai circuitKeys step control output.1 output.2) :
    Relaxed ajtai circuitKeys (initial output) := by
  obtain ⟨statement, proof⟩ := output
  cases proof with
  | bottom =>
      obtain ⟨_, zero, same⟩ :=
        (Terminal.holdsFor_bottom_iff cap ajtai circuitKeys step control statement).1 accepted
      exact ⟨zero, same⟩
  | recursive payload =>
      obtain ⟨_, _, _, _, running, _⟩ :=
        (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control statement payload).1
          accepted
      exact ⟨linked_of_holdsFor control cap statement payload accepted,
        fun slot => (claimHolds_bigB ajtai _ _ _).2 (running slot), fun _ => Or.inl le_rfl⟩

/-- One step of the walk at a relaxed recursive visit without a collision: the decoded last
step is a valid step of the program into the visit's statement. It is the base step, or an
answer that extracts the visit's attempt gives a relaxed prior visit whose statement is the
step's input. -/
theorem walk_step (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (statement : TerminalStatement AppState) (payload : Payload logicalWidth publicFits)
    (bounds : Fin 2 → Nat) (answer : Answer logicalWidth publicFits)
    (relaxed : Relaxed ajtai circuitKeys ⟨statement, .recursive payload, bounds⟩)
    (safe : ¬ Collision exports statement payload)
    (extracts : ∀ attempt,
      attemptOf ajtai circuitKeys exports ⟨statement, .recursive payload, bounds⟩ = some attempt →
        WalkSucceeds ajtai attempt → WalkExtracted ajtai attempt answer) :
    exports.program.valid (exports.last payload).input.zi (exports.last payload).input.witness ∧
      exports.program.step (exports.last payload).input.zi (exports.last payload).input.witness =
        statement.zi ∧
      (exports.last payload).input.z0 = statement.z0 ∧
      (exports.last payload).input.iteration + 1 = statement.iteration ∧
      (((exports.last payload).input.iteration = 0 ∧
          (exports.last payload).input.z0 = (exports.last payload).input.zi) ∨
        (0 < (exports.last payload).input.iteration ∧
          Relaxed ajtai circuitKeys
            (back ajtai circuitKeys exports ⟨statement, .recursive payload, bounds⟩ answer) ∧
          (back ajtai circuitKeys exports ⟨statement, .recursive payload, bounds⟩ answer).statement =
            ⟨(exports.last payload).input.iteration, (exports.last payload).input.z0,
              (exports.last payload).input.zi⟩)) := by
  obtain ⟨linked, holds, within⟩ := relaxed
  rcases bridge_linked exports keysShaped statement payload linked with ⟨stepHolds, ends⟩ |
    collision
  swap
  · exact absurd collision safe
  have shaped := decode_shaped ajtai circuitKeys exports (circuitOf payload.pc)
    payload.freshWitness payload.running
  obtain ⟨dispatch, _, zNext, _, branch⟩ :=
    (stepHolds_iff _ ajtai circuitKeys step _ _ _ _).1 stepHolds
  have stepped : exports.program.step (exports.last payload).input.zi
      (exports.last payload).input.witness = statement.zi := by
    rw [exports.programStep, ← zNext, ends.zNext]
  refine ⟨valid_of_chosen dispatch, stepped, ends.z0, ends.iteration, ?_⟩
  rcases branch with ⟨iterationZero, initialEq, _⟩ | recursive
  · exact Or.inl ⟨iterationZero, initialEq⟩
  obtain ⟨priorPcValid, positive, freshPublic, guard, folded, depthNext, unchanged⟩ := recursive
  have slotEq := circuitOf_selected priorPcValid
  rw [← slotEq] at guard folded depthNext unchanged
  let input := (exports.last payload).input
  let output := (exports.last payload).output
  let slot := circuitOf input.priorPc
  have attemptEq : attemptOf ajtai circuitKeys exports ⟨statement, .recursive payload, bounds⟩ =
      some ⟨⟨priorOf circuitKeys input, input.running slot, foldOf ajtai circuitKeys exports payload⟩,
        bounds slot, payload.runningWitness slot⟩ := by
    simp only [attemptOf, positive, ↓reduceIte]
    rfl
  have accepted := (accepts_iff productionLightFoldCap ajtai (circuitOf payload.pc)
    (circuitKeys slot) (input.running slot) input.fresh input.nifsProof
    (output.runningNext slot)).2 ⟨guard, folded, depthNext⟩
  have succeeds : WalkSucceeds ajtai ⟨⟨priorOf circuitKeys input, input.running slot,
      foldOf ajtai circuitKeys exports payload⟩, bounds slot, payload.runningWitness slot⟩ := by
    refine ⟨⟨freshPublic, ⟨priorPcValid, congrArg input.running slotEq.symm⟩,
      ⟨keysShaped, shaped.z0, shaped.zi, shaped.iteration, shaped.priorPc, shaped.priorDepth⟩⟩,
      rfl, accepted, ?_, ?_⟩
    · change ClaimHolds ajtai (circuitKeys slot) (bounds slot) (output.runningNext slot).1
        (payload.runningWitness slot)
      rw [ends.claims slot]
      exact holds slot
    · show bounds slot ≤ productionGlobalParams.bigB + productionLightFoldCap *
        (productionGlobalParams.expansionT * (productionGlobalParams.b - 1))
      rcases within slot with small | ⟨_, large⟩
      · exact small.trans (Nat.le_add_right _ _)
      · exact large.trans (Nat.add_le_add_left
          (Nat.mul_le_mul_right _ (Nat.sub_le _ _)) _)
  obtain ⟨extractedClaim, extractedFresh⟩ := extracts _ attemptEq succeeds
  have backEq : back ajtai circuitKeys exports ⟨statement, .recursive payload, bounds⟩ answer =
      { statement := ⟨input.iteration, input.z0, input.zi⟩
        proof := .recursive ⟨input.running, Function.update payload.runningWitness slot answer.1,
          input.fresh, answer.2, input.priorPc⟩
        bounds := Function.update bounds slot (extractedBefore productionGlobalParams
          (BoundedDepth.circuitStep (circuitOf payload.pc)) (bounds slot)) } := by
    simp only [back, positive, ↓reduceIte]
    rfl
  refine Or.inr ⟨positive, ?_, by rw [backEq]⟩
  rw [backEq]
  refine ⟨⟨⟨shaped.iteration, shaped.z0, shaped.zi⟩, priorPcValid, positive, freshPublic,
      extractedFresh⟩, ?_, ?_⟩
  · intro other
    by_cases same : other = slot
    · subst same
      simp only [Function.update_self]
      exact extractedClaim
    · simp only [Function.update_of_ne same]
      rw [← unchanged other same, ends.claims other]
      exact holds other
  · intro other
    by_cases same : other = slot
    · subst same
      simp only [Function.update_self]
      cases kind : BoundedDepth.circuitStep (circuitOf payload.pc) with
      | light =>
          rw [kind] at depthNext
          simp only [extractedBefore]
          exact depthWithin_light (guard kind) (by rw [← ends.depths slot, depthNext]; rfl)
            (within slot)
      | refresh =>
          exact Or.inl le_rfl
    · simp only [Function.update_of_ne same]
      rw [← unchanged other same]
      exact depthWithin_kept (shaped.nextDepth other) (ends.depths other) (within other)

/-- The steps that the walk decodes, from the terminal back: one per recursive visit, up to the
base step or the last answer. -/
def trail : Visit logicalWidth publicFits → List (Answer logicalWidth publicFits) →
    List (Decoded logicalWidth publicFits)
  | visit, [] =>
      match visit.proof with
      | .bottom => []
      | .recursive payload => [exports.last payload]
  | visit, answer :: rest =>
      match visit.proof with
      | .bottom => []
      | .recursive payload =>
          exports.last payload ::
            if 0 < (exports.last payload).input.iteration then
              trail (back ajtai circuitKeys exports visit answer) rest
            else []

/-- Each step of the trail decodes a witness of the program's word count. -/
theorem trail_lengths :
    ∀ (visit : Visit logicalWidth publicFits) (answers : List (Answer logicalWidth publicFits)),
      ∀ decoded ∈ trail ajtai circuitKeys exports visit answers,
        decoded.input.witness.length = exports.program.witnessWordCount
  | visit, [], decoded, inside => by
      unfold trail at inside
      split at inside
      · simp at inside
      · rw [List.mem_singleton.mp inside]
        exact exports.last_witness_length _
  | visit, answer :: rest, decoded, inside => by
      unfold trail at inside
      split at inside
      · simp at inside
      · rcases List.mem_cons.mp inside with first | later
        · rw [first]
          exact exports.last_witness_length _
        · split at later
          · exact trail_lengths _ rest decoded later
          · simp at later

/-- With no collision and no failed stage among its first visits, the steps that the walk
decodes from a relaxed visit lead to its statement, when its iteration is at most the number of
answers. -/
theorem walk_leads (keysShaped : ∀ key, (circuitKeys key).digest.length = 4) :
    ∀ (answers : List (Answer logicalWidth publicFits)) (visit : Visit logicalWidth publicFits),
      Relaxed ajtai circuitKeys visit → visit.statement.iteration ≤ answers.length →
      (∀ j < answers.length,
        ¬ WalkCollision ajtai circuitKeys exports
            (visitAfter (back ajtai circuitKeys exports) visit (answers.take j)) ∧
          ¬ StageFails (attemptOf ajtai circuitKeys exports) (back ajtai circuitKeys exports)
            (WalkSucceeds ajtai) (WalkExtracted ajtai) visit answers j) →
      Leads exports.program visit.statement (trail ajtai circuitKeys exports visit answers)
  | [], ⟨statement, .bottom, _⟩, relaxed, _, _ => relaxed
  | _ :: _, ⟨statement, .bottom, _⟩, relaxed, _, _ => relaxed
  | [], ⟨statement, .recursive payload, _⟩, relaxed, small, _ => by
      have positive : 0 < statement.iteration := relaxed.1.positive
      have zero : statement.iteration ≤ 0 := small
      omega
  | answer :: rest, ⟨statement, .recursive payload, bounds⟩, relaxed, small, clean => by
      obtain ⟨safe, noFailure⟩ := clean 0 (by simp)
      have extracts : ∀ attempt, attemptOf ajtai circuitKeys exports
          ⟨statement, .recursive payload, bounds⟩ = some attempt →
          WalkSucceeds ajtai attempt → WalkExtracted ajtai attempt answer :=
        fun attempt attemptEq wins => Classical.byContradiction fun missing =>
          noFailure ⟨attempt, answer, attemptEq, rfl, wins, missing⟩
      obtain ⟨valid, stepped, z0Eq, iterationEq, base | ⟨positive, relaxedPrior, statementEq⟩⟩ :=
        walk_step ajtai circuitKeys exports keysShaped statement payload bounds answer relaxed safe
          extracts
      · have stops : ¬ 0 < (exports.last payload).input.iteration := by omega
        simp only [trail, stops, ↓reduceIte]
        exact ⟨valid, stepped, z0Eq, iterationEq, base.1, base.2.symm⟩
      · have later := walk_leads keysShaped rest _ relaxedPrior
          (by
            rw [statementEq]
            show (exports.last payload).input.iteration ≤ rest.length
            have bound : statement.iteration ≤ rest.length + 1 := by simpa using small
            omega) fun j inside => by
            have shifted := clean (j + 1) (by simp; omega)
            rw [visitAfter_cons, stageFails_cons] at shifted
            exact shifted
        rw [statementEq] at later
        simp only [trail, positive, ↓reduceIte]
        exact ⟨valid, stepped, z0Eq, iterationEq, later⟩

/-- An accepted terminal whose decoded steps do not lead to its statement has, at one of its
first `iteration` visits, a collision or a stage whose attempt succeeds and whose answer does not
extract. -/
theorem covered (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (control : AppState → AppWitness → Fin 2) (cap : Nat)
    (output : TerminalStatement AppState ×
      Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits))
    (answers : List (Answer logicalWidth publicFits))
    (depthBound : output.1.iteration ≤ answers.length)
    (accepted : Terminal.HoldsFor cap ajtai circuitKeys step control output.1 output.2)
    (unled : ¬ Leads exports.program output.1
      (trail ajtai circuitKeys exports (initial output) answers)) :
    ∃ j < answers.length,
      WalkCollision ajtai circuitKeys exports
          (visitAfter (back ajtai circuitKeys exports) (initial output) (answers.take j)) ∨
        StageFails (attemptOf ajtai circuitKeys exports) (back ajtai circuitKeys exports)
          (WalkSucceeds ajtai) (WalkExtracted ajtai) (initial output) answers j := by
  refine Classical.byContradiction fun none => unled ?_
  exact walk_leads ajtai circuitKeys exports keysShaped answers (initial output)
    (initial_relaxed ajtai circuitKeys control cap output accepted) depthBound
    fun j inside => not_or.mp fun found => none ⟨j, inside, found⟩

end Walk

end NightstreamFPrime.Export.OptionB.IvcWalk
