import NightstreamFPrime.Lifecycle.PiRLC.Wide.CompletionValues
import NightstreamFPrime.Lifecycle.PiRLC.Wide.Completeness

/-! Carry exact sampler completion values through the existing four
combination children and output binding. No protocol premise is added. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.Wide.Formal

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

def RangesCompleted (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) : Prop :=
  Batch.RangesCompleted (samplerInterface (atOffset interface offset)) (samplerOffset offset) Batch.sourceCount env

theorem rangesCompleted_of_agree
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) (before after : Env)
    (assumptions : Assumptions relation interface offset before)
    (completed : RangesCompleted interface offset before)
    (agrees : ∀ index, index < commitmentOffset offset → after index = before index) :
    RangesCompleted interface offset after := by
  apply Batch.rangesCompleted_of_agree _ _ _ assumptions.sampler before after completed
  intro index below
  apply agrees
  change index < offset + 55403
  change index < offset + 54485 at below
  omega

/-- Valid phase inputs construct a complete witness with exact range hints. -/
theorem completePrefix_constructive_with_values
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits) (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = opsAt relation interface offset ∧
      Semantics.PhaseHolds relation ajtai interface offset completed.current ∧
      RangesCompleted interface offset completed.current := by
  let shared := atOffset interface offset
  obtain ⟨p1, o1, s1, v1⟩ := completeInputPrefix relation interface env offset
  obtain ⟨sampled, sampleAgreement, sampleRows, sampledValues⟩ := ProjectedBatch.complete_with_values
    (samplerInterface shared) env (samplerOffset offset) assumptions.sampler
  have sampleScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (samplerCircuit shared).main (samplerOffset offset)),
      expression.VarsBelow (samplerOffset offset +
        localLength (Circuit.ops (samplerCircuit shared).main (samplerOffset offset))) := by
    change ∀ expression ∈ flatConstraints (ProjectedBatch.operations (samplerInterface shared) (samplerOffset offset)),
      expression.VarsBelow (samplerOffset offset + localLength (ProjectedBatch.operations _ _))
    rw [ProjectedBatch.localLength_eq]
    exact ProjectedBatch.scope _ _ assumptions.sampler
  obtain ⟨p2, o2, n2, _, v2⟩ := Sequence.appendBuiltAt_current p1 "pirlc.wide.sampler"
    (samplerCircuit shared) (samplerOffset offset) s1 sampleScope sampled
    (by simpa only [v1] using! sampleAgreement) sampleRows
  have s2 : offset + localLength p2.operations = commitmentOffset offset := by
    rw [n2]
    change samplerOffset offset + localLength (ProjectedBatch.operations _ _) = _
    rw [ProjectedBatch.localLength_eq]
    rfl
  obtain ⟨p3, o3, s3, p23⟩ := appendCommitment relation interface env offset assumptions p2 s2
  obtain ⟨p4, o4, s4, p34⟩ := appendPublicInput relation interface env offset assumptions p3 s3
  obtain ⟨p5, o5, s5, p45⟩ := appendEvalK relation interface env offset assumptions p4 s4
  obtain ⟨p6, o6, s6, p56⟩ := appendEvalA relation interface env offset assumptions p5 s5
  obtain ⟨p7, o7, _, p67⟩ := appendOutputBinding relation interface env offset p6 s6
  have operations : p7.operations = opsAt relation interface offset := by
    rw [o7, o6, o5, o4, o3, o2, o1]
    rfl
  have preserved := (((p23.trans p34).trans p45).trans p56).trans p67
  have completed : RangesCompleted interface offset p7.current := by
    apply rangesCompleted_of_agree relation interface offset p2.current p7.current
      (assumptionsAt assumptions p2.current)
    · rwa [v2]
    · intro index below
      exact preserved.values index (by rwa [s2])
  have rows : holdsFlat p7.current (Circuit.ops (main relation interface) offset) := by
    rw [main_ops, ← operations]
    exact p7.rows
  have specification := soundness relation interface offset p7.current
    (assumptionsAt assumptions p7.current) (holdsFlat_implies_holds p7.current _ rows)
  exact ⟨p7, operations,
    Semantics.spec_implies_phaseHolds relation ajtai interface offset p7.current specification, completed⟩

end NightstreamFPrime.Lifecycle.PiRLC.Wide.Formal
