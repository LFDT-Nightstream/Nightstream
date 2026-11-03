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

/-- `iteration` steps of `step` lead from `z0` to `zi`. -/
def Reaches (step : AppState → AppWitness → AppState) (statement : TerminalStatement AppState) :
    Prop :=
  ∃ advice : List AppWitness, advice.length = statement.iteration ∧
    advice.foldl step statement.z0 = statement.zi

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
    (witness : Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Shaped (exports.decode circuit witness) := by
  unfold Exports.decode
  split
  · have represents := lightRepresents exports.lightProgram ajtai circuitKeys 0
      (lightEnv exports.lightProgram relation ajtai exports.lightLayout
        (Layout.ProductionRelation.Plan.logicalAssignment witness))
    exact ⟨represents.iterationBound, represents.priorPcBound, by simp [lightInput],
      by simp [lightInput], represents.priorDepth, represents.nextDepth⟩
  · have represents := refreshRepresents exports.refreshProgram ajtai circuitKeys 1
      (refreshEnv exports.refreshProgram relation ajtai exports.refreshLayout
        (Layout.ProductionRelation.Plan.logicalAssignment witness))
    exact ⟨represents.iterationBound, represents.priorPcBound, by simp [refreshInput],
      by simp [refreshInput], represents.priorDepth, represents.nextDepth⟩

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

/-- One step of the walk at a relaxed recursive visit without a collision: the last step is the
base step and the statement is reached, or an answer that extracts the visit's attempt gives a
relaxed prior visit one iteration back, from whose current state one step reaches this one. -/
theorem walk_step (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (statement : TerminalStatement AppState) (payload : Payload logicalWidth publicFits)
    (bounds : Fin 2 → Nat) (answer : Answer logicalWidth publicFits)
    (relaxed : Relaxed ajtai circuitKeys ⟨statement, .recursive payload, bounds⟩)
    (safe : ¬ Collision exports statement payload)
    (extracts : ∀ attempt,
      attemptOf ajtai circuitKeys exports ⟨statement, .recursive payload, bounds⟩ = some attempt →
        WalkSucceeds ajtai attempt → WalkExtracted ajtai attempt answer) :
    Reaches step statement ∨
      (Relaxed ajtai circuitKeys
          (back ajtai circuitKeys exports ⟨statement, .recursive payload, bounds⟩ answer) ∧
        (back ajtai circuitKeys exports ⟨statement, .recursive payload, bounds⟩
          answer).statement.iteration + 1 = statement.iteration ∧
        (back ajtai circuitKeys exports ⟨statement, .recursive payload, bounds⟩
          answer).statement.z0 = statement.z0 ∧
        ∃ witness, step (back ajtai circuitKeys exports ⟨statement, .recursive payload, bounds⟩
          answer).statement.zi witness = statement.zi) := by
  obtain ⟨linked, holds, within⟩ := relaxed
  rcases bridge_linked exports keysShaped statement payload linked with ⟨stepHolds, ends⟩ |
    collision
  swap
  · exact absurd collision safe
  have shaped := decode_shaped ajtai circuitKeys exports (circuitOf payload.pc)
    payload.freshWitness
  obtain ⟨_, _, zNext, _, base | recursive⟩ :=
    (stepHolds_iff _ ajtai circuitKeys step _ _ _ _).1 stepHolds
  · obtain ⟨iterationZero, initialEq, _⟩ := base
    left
    refine ⟨[(exports.last payload).input.witness], ?_, ?_⟩
    · simp [← ends.iteration, iterationZero]
    · simp only [List.foldl_cons, List.foldl_nil]
      rw [← ends.z0, initialEq, ← ends.zNext, zNext]
  obtain ⟨priorPcValid, positive, freshPublic, guard, folded, depthNext, unchanged⟩ := recursive
  have slotEq := circuitOf_selected priorPcValid
  rw [← slotEq] at guard folded depthNext unchanged
  right
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
  rw [backEq]
  refine ⟨⟨⟨⟨shaped.iteration, shaped.z0, shaped.zi⟩, priorPcValid, positive, freshPublic,
      extractedFresh⟩, ?_, ?_⟩, ends.iteration, ends.z0, input.witness, ?_⟩
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
  · rw [← ends.zNext, zNext]

/-- With no collision and no failed stage among its first visits, a relaxed visit whose
iteration is at most the number of answers is reached. -/
theorem walk_reaches (keysShaped : ∀ key, (circuitKeys key).digest.length = 4) :
    ∀ (answers : List (Answer logicalWidth publicFits)) (visit : Visit logicalWidth publicFits),
      Relaxed ajtai circuitKeys visit → visit.statement.iteration ≤ answers.length →
      (∀ j < answers.length,
        ¬ WalkCollision ajtai circuitKeys exports
            (visitAfter (back ajtai circuitKeys exports) visit (answers.take j)) ∧
          ¬ StageFails (attemptOf ajtai circuitKeys exports) (back ajtai circuitKeys exports)
            (WalkSucceeds ajtai) (WalkExtracted ajtai) visit answers j) →
      Reaches step visit.statement
  | _, ⟨statement, .bottom, _⟩, relaxed, _, _ => ⟨[], relaxed.1.symm, relaxed.2.symm⟩
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
      rcases walk_step ajtai circuitKeys exports keysShaped statement payload bounds answer
          relaxed safe extracts with reached | ⟨relaxedPrior, iterationEq, z0Eq, witness, stepEq⟩
      · exact reached
      obtain ⟨advice, length, folded⟩ := walk_reaches keysShaped rest _ relaxedPrior
        (by simp at small; omega) fun j inside => by
          have shifted := clean (j + 1) (by simp; omega)
          rw [visitAfter_cons, stageFails_cons] at shifted
          exact shifted
      refine ⟨advice ++ [witness], by simp [length]; omega, ?_⟩
      rw [List.foldl_append, ← z0Eq, folded]
      exact stepEq

/-- A false acceptance has, at one of its first `iteration` visits, a collision or a stage whose
attempt succeeds and whose answer does not extract. -/
theorem covered (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (control : AppState → AppWitness → Fin 2) (cap : Nat)
    (output : TerminalStatement AppState ×
      Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits))
    (answers : List (Answer logicalWidth publicFits))
    (depthBound : output.1.iteration ≤ answers.length)
    (accepted : Terminal.HoldsFor cap ajtai circuitKeys step control output.1 output.2)
    (unreached : ¬ Reaches step output.1) :
    ∃ j < answers.length,
      WalkCollision ajtai circuitKeys exports
          (visitAfter (back ajtai circuitKeys exports) (initial output) (answers.take j)) ∨
        StageFails (attemptOf ajtai circuitKeys exports) (back ajtai circuitKeys exports)
          (WalkSucceeds ajtai) (WalkExtracted ajtai) (initial output) answers j := by
  refine Classical.byContradiction fun none => unreached ?_
  exact walk_reaches ajtai circuitKeys exports keysShaped answers (initial output)
    (initial_relaxed ajtai circuitKeys control cap output accepted) depthBound
    fun j inside => not_or.mp fun found => none ⟨j, inside, found⟩

end Walk

end NightstreamFPrime.Export.OptionB.IvcWalk
