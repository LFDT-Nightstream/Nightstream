import NightstreamFPrime.Export.Stage1.StoredPhysicalExecution
import NightstreamFPrime.Export.Stage1.StoredPiRLCCombination

/-!
Stored event execution for CCS witness construction. Only canonical PiRLC
product events omit their R1CS scratch rows. Other event execution and all
outer guards keep their existing definitions. Canonical schedule support
is a separate proof, not a caller-selected correctness assumption.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredDirectPhysicalExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Package
open PiRLCCombinationScratchGeometry (scratchStart scratchEnd)
open StoredExecutionSupport (Agree)
open StoredPhysicalExecution (ResultAgree EventSupported)

abbrev Outside (column : Nat) : Prop := column < scratchStart ∨ scratchEnd ≤ column

def compact (templates : Array CompactRowTemplate) (target : Nat)
    (invocation : CompactRowInvocation) (values : Array F) : Except String (Array F) := do
  let some template := templates[invocation.templateIndex]?
    | throw "missing canonical compact template"
  unless compactInputColumn invocation.inputRanges template.outputInput == target do
    throw "compact output differs from its scheduled target"
  StoredPhysicalExecution.requireWrite values target 1
  StoredPhysicalExecution.requireWrite values invocation.localStart template.localColumnCount
  StoredPhysicalExecution.compactResult target (StoredCompactOutput.execute
    (compactInputColumn invocation.inputRanges) invocation.localStart template values)

def executeEvent (pilot : CircuitPackage) (templates : Array CompactRowTemplate)
    (event : StoredPhysicalPlan.Event) (values : Array F) : Except String (Array F) :=
  match event with
  | .compact target invocation =>
      if invocation.templateIndex < 2 * ringDegree then compact templates target invocation values
      else StoredPhysicalExecution.executeEvent pilot templates event values
  | _ => StoredPhysicalExecution.executeEvent pilot templates event values

/-- These are the precise two cases supplied by the canonical event owners. -/
def EventSafe (context : PerApplicationCachedShift.Context) (pilot : CircuitPackage)
    (templates : Array CompactRowTemplate) (event : StoredPhysicalPlan.Event) : Prop :=
  match event with
  | .compact _ invocation =>
      if invocation.templateIndex < 2 * ringDegree then
        ∃ descriptor : PiRLCProductSchedule.Descriptor,
          invocation = PerApplicationCachedShift.shiftCompactRowInvocation context
            descriptor.compactInvocation
      else EventSupported Outside pilot templates event
  | _ => EventSupported Outside pilot templates event

/-- The product output constructor refines the actual guarded dispatcher. -/
theorem compact_agree (context : PerApplicationCachedShift.Context)
    (templates : Array CompactRowTemplate)
    (canonical : templates.toList = Data.compactRowTemplates ())
    (target : Nat) (descriptor : PiRLCProductSchedule.Descriptor)
    (left right : Array F) (agree : Agree Outside left right) :
    let invocation := PerApplicationCachedShift.shiftCompactRowInvocation context
      descriptor.compactInvocation
    ResultAgree Outside (StoredPhysicalExecution.compact templates target invocation left)
      (compact templates target invocation right) := by
  let invocation := PerApplicationCachedShift.shiftCompactRowInvocation context
    descriptor.compactInvocation
  let template := PiRLCCombinationTemplates.template
    (PiRLCCombinationInvocations.firstSource descriptor.source.val) descriptor.lane
  have selected : templates[invocation.templateIndex]? = some template := by
    have listEq := congrArg (fun values : List CompactRowTemplate =>
      values[descriptor.compactInvocation.templateIndex]?) canonical
    simpa only [Array.getElem?_toList,
      PiRLCCombinationInvocationOrigin.descriptor_template] using! listEq
  have step := StoredPiRLCCombination.shifted_agree context descriptor left right agree
  have result := StoredPhysicalExecution.option_result_agree Outside target _ _ step
  change ResultAgree Outside (StoredPhysicalExecution.compact templates target invocation left)
    (compact templates target invocation right)
  simp only [StoredPhysicalExecution.compact, compact, selected,
    StoredPhysicalExecution.requireWrite, agree.1]
  split_ifs <;> first | exact result | rfl

/-- A proved event contract gives exact rejection agreement or the retained
array correspondence. No completed R1CS witness is an input to this step. -/
theorem executeEvent_agree (context : PerApplicationCachedShift.Context)
    (pilot : CircuitPackage) (templates : Array CompactRowTemplate)
    (canonical : templates.toList = Data.compactRowTemplates ())
    (event : StoredPhysicalPlan.Event) (left right : Array F)
    (safe : EventSafe context pilot templates event) (agree : Agree Outside left right) :
    ResultAgree Outside (StoredPhysicalExecution.executeEvent pilot templates event left)
      (executeEvent pilot templates event right) := by
  cases event with
  | compact target invocation =>
      by_cases product : invocation.templateIndex < 2 * ringDegree
      · simp only [EventSafe, if_pos product] at safe
        rcases safe with ⟨descriptor, rfl⟩
        simp only [executeEvent, if_pos product, StoredPhysicalExecution.executeEvent]
        exact compact_agree context templates canonical target descriptor left right agree
      · simp only [EventSafe, if_neg product] at safe
        simp only [executeEvent, if_neg product]
        exact StoredPhysicalExecution.executeEvent_agree Outside pilot templates
          (.compact target invocation) left right safe agree
  | hash chain ordinal =>
      exact StoredPhysicalExecution.executeEvent_agree Outside pilot templates
        (.hash chain ordinal) left right safe agree
  | permutation invocation =>
      exact StoredPhysicalExecution.executeEvent_agree Outside pilot templates
        (.permutation invocation) left right safe agree
  | batch batch =>
      exact StoredPhysicalExecution.executeEvent_agree Outside pilot templates
        (.batch batch) left right safe agree
  | instruction instruction =>
      exact StoredPhysicalExecution.executeEvent_agree Outside pilot templates
        (.instruction instruction) left right safe agree

end NightstreamFPrime.Export.Stage1.StoredDirectPhysicalExecution
