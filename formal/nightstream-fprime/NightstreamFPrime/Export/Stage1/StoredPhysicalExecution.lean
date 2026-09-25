import NightstreamFPrime.Export.Stage1.StoredExecutionSupport
import NightstreamFPrime.Export.Stage1.StoredPhysicalPlan

/-!
The stored physical event dispatcher used by the replay tool. Bounds checks,
recipe order, compact-row rejection, and field operations retain their owners.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredPhysicalExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open StoredExecutionSupport (Agree CompactRowSupported)

def requireWrite (values : Array F) (start count : Nat) : Except String Unit :=
  if start + count ≤ values.size then .ok () else .error "physical write exceeds its selected bound"

def compactResult (target : Nat) : Option (Array F) → Except String (Array F)
  | some result => .ok result
  | none => .error s!"compact row failed at {target}"

def compact (templates : Array CompactRowTemplate) (target : Nat)
    (invocation : CompactRowInvocation) (values : Array F) : Except String (Array F) := do
  let some template := templates[invocation.templateIndex]?
    | throw "missing canonical compact template"
  unless compactInputColumn invocation.inputRanges template.outputInput == target do
    throw "compact output differs from its scheduled target"
  requireWrite values target 1
  requireWrite values invocation.localStart template.localColumnCount
  compactResult target (StoredCompactRowExecution.execute
    (compactInputColumn invocation.inputRanges) invocation.localStart template values)

def hashInvocation (pilot : CircuitPackage) (chain : HashChain) (ordinal : Nat) :
    PermutationInvocation := {
  phase := chain.phase
  rowStart := chain.rowStart + ordinal * pilot.poseidon.recipesPerPermutation
  witnessStart := invocationLocalStart pilot chain ordinal
  inputs := List.ofFn fun lane : Fin 8 =>
    Rows.sparseCombination (invocationInput pilot chain ordinal lane.val) }

def executeEvent (pilot : CircuitPackage) (templates : Array CompactRowTemplate)
    (event : StoredPhysicalPlan.Event) (values : Array F) : Except String (Array F) := do
  match event with
  | .hash chain ordinal =>
      let invocation := hashInvocation pilot chain ordinal
      requireWrite values invocation.witnessStart 592
      return StoredPermutationExecution.execute invocation values
  | .permutation invocation =>
      requireWrite values invocation.witnessStart 592
      return StoredPermutationExecution.execute invocation values
  | .compact target invocation => compact templates target invocation values
  | .batch batch =>
      requireWrite values batch.start (batch.recipes.length + batch.hints.length)
      let result := StoredWitnessExecution.executeRecipes values batch.start batch.recipes
      return StoredWitnessExecution.executeHints result
        (batch.start + batch.recipes.length) batch.hints
  | .instruction instruction =>
      requireWrite values instruction.target 1
      return StoredInstructionExecution.execute instruction values

def ResultAgree (allowed : Nat → Prop) : Except String (Array F) → Except String (Array F) → Prop
  | .ok left, .ok right => Agree allowed left right
  | .error left, .error right => left = right
  | _, _ => False

def PermutationSupported (allowed : Nat → Prop) (invocation : PermutationInvocation) : Prop :=
  ∀ lane, lane < 8 → (invocationInputCombination invocation lane).toR1CS.VarsSatisfy allowed

/-- Read support covers evaluated rows and hints, including the C check after
each compact write. Constraint support alone is not sufficient. -/
def EventSupported (allowed : Nat → Prop) (pilot : CircuitPackage)
    (templates : Array CompactRowTemplate) : StoredPhysicalPlan.Event → Prop
  | .hash chain ordinal => PermutationSupported allowed (hashInvocation pilot chain ordinal)
  | .permutation invocation => PermutationSupported allowed invocation
  | .compact _ invocation => ∀ template, templates[invocation.templateIndex]? = some template →
      template.outputRecipe.VarsSatisfy
        (fun input => allowed (compactInputColumn invocation.inputRanges input)) ∧
      ∀ row ∈ template.rows, CompactRowSupported allowed
        (compactInputColumn invocation.inputRanges) invocation.localStart row
  | .batch batch => (∀ recipe ∈ batch.recipes, recipe.VarsSatisfy allowed) ∧
      ∀ hint ∈ batch.hints, hint.source.VarsSatisfy allowed
  | .instruction instruction => instruction.a.toR1CS.VarsSatisfy allowed ∧
      instruction.b.toR1CS.VarsSatisfy allowed

theorem option_result_agree (allowed : Nat → Prop) (target : Nat)
    (left right : Option (Array F)) (agree : Option.Rel (Agree allowed) left right) :
    ResultAgree allowed
      (compactResult target left) (compactResult target right) := by
  cases agree with
  | none => rfl
  | some same => exact same

theorem compact_agree (allowed : Nat → Prop) (templates : Array CompactRowTemplate)
    (target : Nat) (invocation : CompactRowInvocation) (left right : Array F)
    (support : EventSupported allowed (PilotData.circuitPackage ()) templates
      (.compact target invocation))
    (agree : Agree allowed left right) :
    ResultAgree allowed (compact templates target invocation left)
      (compact templates target invocation right) := by
  cases found : templates[invocation.templateIndex]? with
  | none =>
      simp only [compact, found]
      exact rfl
  | some template =>
      have supported := support template found
      have steps := StoredExecutionSupport.compact_agree allowed
        (compactInputColumn invocation.inputRanges) invocation.localStart template
        left right supported.1 supported.2 agree
      have results := option_result_agree allowed target _ _ steps
      simp only [compact, found, requireWrite, agree.1]
      split_ifs <;> first | exact results | rfl

/-- The actual dispatcher gives identical errors or equal retained values.
The proof uses read support, not satisfaction of the emitted constraints. -/
theorem executeEvent_agree (allowed : Nat → Prop) (pilot : CircuitPackage)
    (templates : Array CompactRowTemplate) (event : StoredPhysicalPlan.Event)
    (left right : Array F) (support : EventSupported allowed pilot templates event)
    (agree : Agree allowed left right) :
    ResultAgree allowed (executeEvent pilot templates event left)
      (executeEvent pilot templates event right) := by
  cases event with
  | hash chain ordinal =>
      have next := StoredExecutionSupport.permutation_agree allowed left right
        (hashInvocation pilot chain ordinal) support agree
      simp only [executeEvent, requireWrite, agree.1]
      split_ifs <;> simp_all [ResultAgree]
  | permutation invocation =>
      have next := StoredExecutionSupport.permutation_agree allowed left right invocation support agree
      simp only [executeEvent, requireWrite, agree.1]
      split_ifs <;> simp_all [ResultAgree]
  | compact target invocation => exact compact_agree allowed templates target invocation left right support agree
  | batch batch =>
      have recipes := StoredExecutionSupport.recipes_agree allowed left right batch.start
        batch.recipes support.1 agree
      have hints := StoredExecutionSupport.hints_agree allowed _ _
        (batch.start + batch.recipes.length) batch.hints support.2 recipes
      simp only [executeEvent, requireWrite, agree.1]
      split_ifs <;> simp_all [ResultAgree]
  | instruction instruction =>
      have next := StoredExecutionSupport.instruction_agree allowed left right instruction
        support.1 support.2 agree
      simp only [executeEvent, requireWrite, agree.1]
      split_ifs <;> simp_all [ResultAgree]

private def loopStep
    (runner : StoredPhysicalPlan.Event → Array F → Except String (Array F))
    (state : Option Nat × Array F) (event : StoredPhysicalPlan.Event) :
    Except String (Option Nat × Array F) := do
  if let some target := state.1 then
    unless target < event.target do
      throw "physical event targets are not strictly increasing"
  let values ← runner event state.2
  return (some event.target, values)

/-- Execute the existing array order, retaining the strict target-order check. -/
def runWith (runner : StoredPhysicalPlan.Event → Array F → Except String (Array F))
    (events : Array StoredPhysicalPlan.Event) (values : Array F) : Except String (Array F) :=
  (events.foldlM (loopStep runner) (none, values)).map Prod.snd

private def LoopResultAgree (allowed : Nat → Prop) :
    Except String (Option Nat × Array F) → Except String (Option Nat × Array F) → Prop
  | .ok left, .ok right => left.1 = right.1 ∧ Agree allowed left.2 right.2
  | .error left, .error right => left = right
  | _, _ => False

private theorem loopStep_agree (allowed : Nat → Prop)
    (runLeft runRight : StoredPhysicalPlan.Event → Array F → Except String (Array F))
    (event : StoredPhysicalPlan.Event) (previous : Option Nat) (left right : Array F)
    (results : ResultAgree allowed (runLeft event left) (runRight event right)) :
    LoopResultAgree allowed (loopStep runLeft (previous, left) event)
      (loopStep runRight (previous, right) event) := by
  cases leftResult : runLeft event left with
  | error leftError =>
      cases rightResult : runRight event right with
      | error rightError =>
          have equal : leftError = rightError := by
            simpa only [leftResult, rightResult, ResultAgree] using results
          subst rightError
          cases previous <;> simp only [loopStep, leftResult, rightResult]
          · rfl
          · split_ifs <;> rfl
      | ok after => simp only [leftResult, rightResult, ResultAgree] at results
  | ok afterLeft =>
      cases rightResult : runRight event right with
      | error rightError => simp only [leftResult, rightResult, ResultAgree] at results
      | ok afterRight =>
          have same : Agree allowed afterLeft afterRight := by
            simpa only [leftResult, rightResult, ResultAgree] using results
          cases previous <;>
            simp only [loopStep, leftResult, rightResult]
          · exact ⟨rfl, same⟩
          · split_ifs <;> first | exact ⟨rfl, same⟩ | rfl

private theorem list_loop_agree (allowed : Nat → Prop)
    (runLeft runRight : StoredPhysicalPlan.Event → Array F → Except String (Array F))
    (events : List StoredPhysicalPlan.Event)
    (runners : ∀ event ∈ events, ∀ left right,
      Agree allowed left right →
      ResultAgree allowed (runLeft event left) (runRight event right))
    (previous : Option Nat) (left right : Array F) (agree : Agree allowed left right) :
    LoopResultAgree allowed
      (events.foldlM (loopStep runLeft) (previous, left))
      (events.foldlM (loopStep runRight) (previous, right)) := by
  induction events generalizing previous left right with
  | nil => exact ⟨rfl, agree⟩
  | cons event rest inductionHypothesis =>
      have next := loopStep_agree allowed runLeft runRight event previous left right
        (runners event (by simp) left right agree)
      have remaining : ∀ current ∈ rest, ∀ left right,
          Agree allowed left right →
          ResultAgree allowed (runLeft current left) (runRight current right) := by
        intro current member
        exact runners current (by simp [member])
      rw [List.foldlM_cons, List.foldlM_cons]
      cases leftResult : loopStep runLeft (previous, left) event with
      | error leftError =>
          cases rightResult : loopStep runRight (previous, right) event with
          | error rightError =>
              simpa only [leftResult, rightResult, LoopResultAgree] using! next
          | ok after => simp only [leftResult, rightResult, LoopResultAgree] at next
      | ok afterLeft =>
          cases rightResult : loopStep runRight (previous, right) event with
          | error rightError => simp only [leftResult, rightResult, LoopResultAgree] at next
          | ok afterRight =>
              have same : afterLeft.1 = afterRight.1 ∧
                  Agree allowed afterLeft.2 afterRight.2 := by
                simpa only [leftResult, rightResult, LoopResultAgree] using next
              rcases afterLeft with ⟨previousLeft, valuesLeft⟩
              rcases afterRight with ⟨previousRight, valuesRight⟩
              rcases same with ⟨rfl, sameValues⟩
              exact inductionHypothesis remaining _ valuesLeft valuesRight sameValues

/-- Replacing event execution preserves both rejection and retained values.
Only events in the supplied array need a replacement proof. -/
theorem runWith_agree (allowed : Nat → Prop)
    (runLeft runRight : StoredPhysicalPlan.Event → Array F → Except String (Array F))
    (events : Array StoredPhysicalPlan.Event) (left right : Array F)
    (runners : ∀ event ∈ events, ∀ left right,
      Agree allowed left right →
      ResultAgree allowed (runLeft event left) (runRight event right))
    (agree : Agree allowed left right) :
    ResultAgree allowed (runWith runLeft events left) (runWith runRight events right) := by
  have folded := list_loop_agree allowed runLeft runRight events.toList
    (fun event member => runners event (Array.mem_toList_iff.mp member)) none left right agree
  rw [Array.foldlM_toList, Array.foldlM_toList] at folded
  unfold runWith
  cases leftResult : events.foldlM (loopStep runLeft) (none, left) with
  | error leftError =>
      cases rightResult : events.foldlM (loopStep runRight) (none, right) with
      | error rightError =>
          simpa only [leftResult, rightResult, LoopResultAgree, ResultAgree] using! folded
      | ok after => simp only [leftResult, rightResult, LoopResultAgree] at folded
  | ok afterLeft =>
      cases rightResult : events.foldlM (loopStep runRight) (none, right) with
      | error rightError => simp only [leftResult, rightResult, LoopResultAgree] at folded
      | ok afterRight =>
          have same : afterLeft.1 = afterRight.1 ∧ Agree allowed afterLeft.2 afterRight.2 := by
            simpa only [leftResult, rightResult, LoopResultAgree] using folded
          exact same.2

end NightstreamFPrime.Export.Stage1.StoredPhysicalExecution
