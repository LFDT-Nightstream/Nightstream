import NightstreamFPrime.Export.Stage1.PiRLCFirst54ScratchGeometry
import NightstreamFPrime.Export.Stage1.StoredPhysicalExecution
import NightstreamFPrime.Export.Stage1.PerApplicationPreservation

/-! First54 recipes and all compact A/B/C reads avoid PiRLC product scratch. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54ReadSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Sampling
open PiRLCCombinationWitnessReadSupport (Outside)
open PiRLCCombinationScratchGeometry (scratchStart)
open PiRLCFirst54ScratchGeometry
open PiRLCFirst54Invocations
open PiRLCFirst54Templates
open PerApplicationPreservation
open StoredExecutionSupport (CompactRowSupported)

private theorem combination_supported (inputCount localCount : Nat)
    (inputColumn : Nat → Nat) (localStart : Nat) (combination : TemplateCombination)
    (within : CombinationWithin inputCount localCount combination)
    (inputs : ∀ input, inputColumn input < scratchStart)
    (locals : localStart + localCount ≤ scratchStart) :
    (CompactRows.instantiateCombination inputColumn localStart combination).VarsSatisfy Outside := by
  intro term member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  have referenced := within source sourceMember
  left
  cases selected : source.column with
  | input input => exact inputs input
  | «local» index =>
      simp only [RefWithin, selected] at referenced
      change localStart + index < scratchStart
      omega

private theorem constraintTemplate_supported (inputCount outputInput : Nat)
    (recipe : Expr) (scope : (Expr.var outputInput - recipe).VarsBelow inputCount)
    (inputColumn : Nat → Nat) (localStart : Nat)
    (inputs : ∀ input, inputColumn input < scratchStart)
    (locals : localStart + R1CS.constraintFreshCount (Expr.var outputInput - recipe) ≤ scratchStart) :
    (CompactRows.compactConstraintTemplate inputCount outputInput recipe).outputRecipe.VarsSatisfy
        (fun input => Outside (inputColumn input)) ∧
      ∀ row ∈ (CompactRows.compactConstraintTemplate inputCount outputInput recipe).rows,
        CompactRowSupported Outside inputColumn localStart row := by
  constructor
  · have recipeScope : recipe.VarsBelow inputCount := scope.2.2
    exact Expr.VarsSatisfy.mono recipe
      ((Expr.varsSatisfy_lt_iff_varsBelow recipe inputCount).mpr recipeScope)
      (fun input _ => Or.inl (inputs input))
  · intro row member
    have within := compactConstraintTemplate_rowWithin inputCount outputInput recipe row scope member
    exact ⟨combination_supported _ _ _ _ _ within.1 inputs locals,
      combination_supported _ _ _ _ _ within.2.1 inputs locals,
      combination_supported _ _ _ _ _ within.2.2 inputs locals⟩

private theorem template_eq (templates : Array CompactRowTemplate)
    (canonical : templates.toList = Data.compactRowTemplates ())
    (invocation : CompactRowInvocation) (expected actual : CompactRowTemplate)
    (selected : packageTemplates[invocation.templateIndex]? = some expected)
    (found : templates[invocation.templateIndex]? = some actual) : actual = expected := by
  have same : templates[invocation.templateIndex]? = packageTemplates[invocation.templateIndex]? := by
    rw [← Array.getElem?_toList, canonical, Data.compactRowTemplates_eq]
    rfl
  rw [same, selected] at found
  exact (Option.some.inj found).symm

theorem position_supported (context : PerApplicationCachedShift.Context)
    (templates : Array CompactRowTemplate) (canonical : templates.toList = Data.compactRowTemplates ())
    (source round : Nat) (slot : Fin First54Step.slotCount)
    (sourceBound : source < 17) (roundBound : round < 64) (target : Nat) :
    StoredPhysicalExecution.EventSupported Outside (PilotData.circuitPackage ()) templates
      (.compact target (PerApplicationCachedShift.shiftCompactRowInvocation context
        (positionInvocation source round slot.val))) := by
  intro template found
  have inputs := position_inputs_before context source round slot sourceBound roundBound
  have locals := shifted_position_local_end context source round slot sourceBound roundBound
  cases round with
  | zero =>
      have equal := template_eq templates canonical (positionInvocation source 0 slot.val)
        (firstPositionTemplate slot) template (positionInvocation_zero_template source slot) found
      subst template
      apply constraintTemplate_supported _ _ _ (firstPosition_constraint_varsBelow slot) _ _ inputs
      rw [firstPosition_constraintFreshCount]
      omega
  | succ previous =>
      have equal := template_eq templates canonical (positionInvocation source (previous + 1) slot.val)
        (laterPositionTemplate slot) template
        (positionInvocation_succ_template source previous slot) found
      subst template
      apply constraintTemplate_supported _ _ _ (laterPosition_constraint_varsBelow slot) _ _ inputs
      rw [laterPosition_constraintFreshCount]
      split_ifs <;> omega

theorem value_supported (context : PerApplicationCachedShift.Context)
    (templates : Array CompactRowTemplate) (canonical : templates.toList = Data.compactRowTemplates ())
    (source round : Nat) (slot : Fin First54ValueStep.outputCount)
    (sourceBound : source < 17) (roundBound : round < 64) (target : Nat) :
    StoredPhysicalExecution.EventSupported Outside (PilotData.circuitPackage ()) templates
      (.compact target (PerApplicationCachedShift.shiftCompactRowInvocation context
        (valueInvocation source round slot.val))) := by
  intro template found
  have inputs := value_inputs_before context source round slot sourceBound roundBound
  have locals := shifted_value_local_end context source round slot sourceBound roundBound
  cases round with
  | zero =>
      have equal := template_eq templates canonical (valueInvocation source 0 slot.val)
        (firstValueTemplate slot) template (valueInvocation_zero_template source slot) found
      subst template
      apply constraintTemplate_supported _ _ _ (firstValue_constraint_varsBelow slot) _ _ inputs
      rw [firstValue_constraintFreshCount]
      exact locals
  | succ previous =>
      have equal := template_eq templates canonical (valueInvocation source (previous + 1) slot.val)
        (laterValueTemplate slot) template (valueInvocation_succ_template source previous slot) found
      subst template
      apply constraintTemplate_supported _ _ _ (laterValue_constraint_varsBelow slot) _ _ inputs
      rw [laterValue_constraintFreshCount]
      exact locals

/-- The canonical First54 block discharges recipe and row read support. -/
theorem canonical_supported (context : PerApplicationCachedShift.Context)
    (templates : Array CompactRowTemplate) (canonical : templates.toList = Data.compactRowTemplates ())
    (invocation : CompactRowInvocation)
    (member : invocation ∈ PackagePlan.canonicalFirst54Block.expand) (target : Nat) :
    StoredPhysicalExecution.EventSupported Outside (PilotData.circuitPackage ()) templates
      (.compact target (PerApplicationCachedShift.shiftCompactRowInvocation context invocation)) := by
  rw [PackagePlan.canonicalFirst54Block_expand] at member
  rcases List.mem_flatMap.mp member with ⟨source, sourceMember, selected⟩
  have sourceBound : source < 17 := List.mem_range.mp sourceMember
  rcases List.mem_flatMap.mp selected with ⟨round, roundMember, selected⟩
  have roundBound : round < 64 := List.mem_range.mp roundMember
  rcases List.mem_append.mp selected with position | value
  · rcases List.mem_map.mp position with ⟨slot, _, rfl⟩
    exact position_supported context templates canonical source round slot sourceBound roundBound target
  · rcases List.mem_map.mp value with ⟨slot, _, rfl⟩
    exact value_supported context templates canonical source round slot sourceBound roundBound target

end NightstreamFPrime.Export.Stage1.PiRLCFirst54ReadSupport
