import NightstreamFPrime.Layout.OptionB.LightRlcBounds
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Completeness.Core
import NightstreamFPrime.Lifecycle.OptionB.LightCommitment

/-! Completeness of the light Π_RLC layout (assembler A3b). The sampler and the commitment,
public-input and evaluation combinations compute their outputs from their inputs and check no
relation, so when the inputs lie below the parent offset, the four leaves can be filled one after
another (`completeFrom`). -/

namespace NightstreamFPrime.Layout.OptionB.LightRlcCompleteness

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.CompletenessSupport
open NightstreamFPrime.Lifecycle.OptionB.LightRlcLayout

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (piccs : LightPiCCSPhase.Leaves logicalWidth degreeBound publicFits)
  (interface : LightRlcLayout.Interface logicalWidth publicFits) (offset : Nat)

/-- Completeness of the light Π_RLC layout after a prefix that ends at the layout's offset. -/
theorem completeFrom {initial : Env} {base : Nat} (before : Sequence.Prefix initial base)
    (startEq : base + localLength before.operations = offset)
    (below : LightRlcBounds.InputsBelow piccs interface offset) :
    ∃ after : Sequence.Prefix initial base,
      after.operations = before.operations ++ opsAt piccs interface offset ∧
      base + localLength after.operations = finalOffset piccs interface offset ∧
      Sequence.PreservesPrefix before after := by
  have assumptionsAt := LightRlcBounds.assumptions piccs interface offset below
  obtain ⟨built1, agrees1, rows1⟩ := Sampler.complete (samplerInterface piccs) 0 before.current
    offset (assumptionsAt before.current).sampler
  obtain ⟨p1, ops1, end1, preserves1, -⟩ := appendBuiltAt before "light.pirlc.sampler"
    (samplerCircuit piccs) offset startEq
    (fun expression member => by
      have scope := Sampler.scope (samplerInterface piccs) 0 offset
        (assumptionsAt before.current).sampler expression member
      rw [← Sampler.localLength_eq (samplerInterface piccs) 0 offset] at scope
      exact scope)
    built1 agrees1 rows1
  obtain ⟨built2, agrees2, rows2⟩ := CombinationStep.complete
    (LightCommitment.stepInterface (commitmentInterface interface offset)) p1.current
    (commitmentOffset piccs offset) (assumptionsAt p1.current).commitment
  obtain ⟨p2, ops2, end2, preserves2, -⟩ := appendBuiltAt p1 "light.pirlc.commitment"
    (commitmentCircuit interface offset) (commitmentOffset piccs offset) end1
    (fun expression member => by
      have scope := CombinationStep.flatConstraints_varsBelow _ _ p1.current
        (assumptionsAt p1.current).commitment expression member
      rw [← CombinationStep.localLength_eq
        (LightCommitment.stepInterface (commitmentInterface interface offset))
        (commitmentOffset piccs offset)] at scope
      exact scope)
    built2 agrees2 rows2
  obtain ⟨built3, agrees3, rows3⟩ := CombinationStep.complete
    (LightPublicInput.stepInterface (publicInputInterface interface offset)) p2.current
    (publicInputOffset piccs interface offset) (assumptionsAt p2.current).publicInput
  obtain ⟨p3, ops3, end3, preserves3, -⟩ := appendBuiltAt p2 "light.pirlc.public_input"
    (publicInputCircuit interface offset) (publicInputOffset piccs interface offset) end2
    (fun expression member => by
      have scope := CombinationStep.flatConstraints_varsBelow _ _ p2.current
        (assumptionsAt p2.current).publicInput expression member
      rw [← CombinationStep.localLength_eq
        (LightPublicInput.stepInterface (publicInputInterface interface offset))
        (publicInputOffset piccs interface offset)] at scope
      exact scope)
    built3 agrees3 rows3
  obtain ⟨built4, agrees4, rows4⟩ := CombinationStep.complete
    (LightEvaluation.stepInterface (evaluationInterface piccs offset)) p3.current
    (evaluationOffset piccs interface offset) (assumptionsAt p3.current).evaluation
  obtain ⟨p4, ops4, end4, preserves4, -⟩ := appendBuiltAt p3 "light.pirlc.evaluation"
    (evaluationCircuit piccs offset) (evaluationOffset piccs interface offset) end3
    (fun expression member => by
      have scope := CombinationStep.flatConstraints_varsBelow _ _ p3.current
        (assumptionsAt p3.current).evaluation expression member
      rw [← CombinationStep.localLength_eq
        (LightEvaluation.stepInterface (evaluationInterface piccs offset))
        (evaluationOffset piccs interface offset)] at scope
      exact scope)
    built4 agrees4 rows4
  refine ⟨p4, ?_, end4, preserves1.trans (preserves2.trans (preserves3.trans preserves4))⟩
  rw [ops4, ops3, ops2, ops1]
  simp only [opsAt, List.append_assoc, List.cons_append, List.nil_append]

end NightstreamFPrime.Layout.OptionB.LightRlcCompleteness
