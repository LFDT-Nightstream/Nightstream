import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support

/-!
Owns variable-support propagation for the PiCCS statement-binding leaf.

The assertions read fixed prior/output state words, the four
verifier-context words, the packed prior words, the prior child digits, and the
leaf's own hinted sign columns. This module changes no circuit and selects no
layout.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem sub_varsSatisfy (left right : Expr) (allowed : Nat → Prop)
    (leftSupport : left.VarsSatisfy allowed)
    (rightSupport : right.VarsSatisfy allowed) :
    (left - right).VarsSatisfy allowed := by
  exact ⟨leftSupport, ⟨trivial, rightSupport⟩⟩

private theorem stateAssertions_varsSatisfy
    (state : Nat → Expr) (allowed : Nat → Prop)
    (support : ∀ word ∈ StateBinding.fixedWords,
      (state word.index).VarsSatisfy allowed) :
    ∀ expression ∈ StateBinding.stateAssertions state,
      expression.VarsSatisfy allowed := by
  intro expression member
  rw [StateBinding.stateAssertions, List.mem_map] at member
  rcases member with ⟨word, wordMember, rfl⟩
  exact sub_varsSatisfy _ _ allowed (support word wordMember) trivial

private theorem contextAssertions_varsSatisfy
    (state : Nat → Expr) (expected : Fin 4 → Expr)
    (allowed : Nat → Prop)
    (stateSupport : ∀ lane : Fin 4,
      (state (StateBinding.contextWordStart + lane.val)).VarsSatisfy allowed)
    (expectedSupport : ∀ lane : Fin 4,
      (expected lane).VarsSatisfy allowed) :
    ∀ expression ∈ StateBinding.contextAssertions state expected,
      expression.VarsSatisfy allowed := by
  intro expression member
  rw [StateBinding.contextAssertions, List.mem_map] at member
  rcases member with ⟨lane, _laneMember, rfl⟩
  exact sub_varsSatisfy _ _ allowed (stateSupport lane)
    (expectedSupport lane)

/-- Exact support of the frozen statement-binding child selected by the
PiCCS parent. -/
theorem statementBindingConstraints_varsSatisfy
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (parentOffset childOffset : Nat) (allowed : Nat → Prop)
    (support : ExternalInputsSupported interface parentOffset allowed)
    (signs : ∀ word lane, allowed (childOffset + (StateBinding.signIndex word lane).val)) :
    ∀ expression ∈ flatConstraints (Circuit.ops
        (statementBindingCircuit (atOffset interface parentOffset)).main
          childOffset),
      expression.VarsSatisfy allowed := by
  intro expression member
  unfold statementBindingCircuit at member
  rw [FormalCircuit.withConstantFootprint_main,
    StatementBinding.flatConstraints_eq_stateAssertions] at member
  rw [StateBinding.assertions, List.mem_append] at member
  rcases member with member | childMember
  swap
  · apply StateBinding.childAssertions_closed (fun expression => expression.VarsSatisfy allowed)
      (fun _ => trivial) (fun _ _ left right => ⟨left, right⟩)
      (fun _ _ left right => ⟨left, right⟩) _ _ _ _ _ expression childMember
    · intro word
      simpa [statementBindingInterface, atOffset] using support.priorStatePacked word
    · intro word lane child
      simpa [statementBindingInterface, atOffset] using
        support.runningPublicInput
          (Fin.cast runningCount_eq_radixChildCount.symm child) (packedColumn word lane)
    · exact signs
  rw [StateBinding.stateWordAssertions, List.mem_append] at member
  rcases member with priorMember | remainingMember
  · apply stateAssertions_varsSatisfy _ allowed _ expression priorMember
    intro word wordMember
    simpa [statementBindingInterface, atOffset] using
      support.priorStateFixed word wordMember
  · rw [List.mem_append] at remainingMember
    rcases remainingMember with middleMember | outputContextMember
    · rw [List.mem_append] at middleMember
      rcases middleMember with outputMember | priorContextMember
      · apply stateAssertions_varsSatisfy _ allowed _ expression outputMember
        intro word wordMember
        simpa [statementBindingInterface, atOffset] using
          support.outputStateFixed word wordMember
      · apply contextAssertions_varsSatisfy _ _ allowed _ _ expression
          priorContextMember
        · intro lane
          simpa [statementBindingInterface, atOffset] using
            support.priorStateContext lane
        · intro lane
          simpa [statementBindingInterface, atOffset] using
            support.expectedContext lane
    · apply contextAssertions_varsSatisfy _ _ allowed _ _ expression
        outputContextMember
      · intro lane
        simpa [statementBindingInterface, atOffset] using
          support.outputStateContext lane
      · intro lane
        simpa [statementBindingInterface, atOffset] using
          support.expectedContext lane

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal
