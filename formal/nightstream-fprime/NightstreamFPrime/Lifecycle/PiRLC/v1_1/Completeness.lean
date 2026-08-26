import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics

/-!
Completes the seven-child PiRLC v1.1 logical assembler. This file owns only
ordered child-witness composition and the sole parent `FormalCircuit`. It
adds no protocol predicate, transcript action, or verifier row.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def assumptionsAt
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {interface : Interface logicalWidth publicFits}
    {offset : Nat} {env : Env}
    (assumptions : Assumptions relation interface offset env)
    (current : Env) : Assumptions relation interface offset current where
  sampler := { initialBelow := assumptions.sampler.initialBelow }
  commitment := {
    challengeBelow := assumptions.commitment.challengeBelow
    inputBelow := assumptions.commitment.inputBelow }
  publicInput := {
    challengeBelow := assumptions.publicInput.challengeBelow
    inputBelow := assumptions.publicInput.inputBelow }
  eval_K := {
    challengeBelow := assumptions.eval_K.challengeBelow
    inputBelow := assumptions.eval_K.inputBelow }
  eval_A := {
    challengeBelow := assumptions.eval_A.challengeBelow
    inputBelow := assumptions.eval_A.inputBelow }

private theorem completeInputPrefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations =
        [childOp "pirlc.v1_1.input_binding"
          (inputBindingCircuit relation (atOffset interface offset))
          (inputBindingOffset offset)] ∧
      offset + localLength completed.operations = samplerOffset offset ∧
      completed.current = env := by
  let empty := Sequence.empty env offset
  let shared := atOffset interface offset
  have startEq : offset + localLength empty.operations =
      inputBindingOffset offset := by
    simp [empty, Sequence.empty, inputBindingOffset, localLength]
  have childScope := InputBinding.flatConstraints_varsBelow relation
    (inputBindingInterface shared) (inputBindingOffset offset)
  have childRows : holdsFlat env
      (Circuit.ops (inputBindingCircuit relation shared).main
        (inputBindingOffset offset)) := by
    intro expression member
    cases member
  have childAgrees : AgreesOutside env env (inputBindingOffset offset)
      (localLength (Circuit.ops (inputBindingCircuit relation shared).main
        (inputBindingOffset offset))) := by
    intro _ _
    rfl
  rcases Sequence.appendBuiltAt empty "pirlc.v1_1.input_binding"
      (inputBindingCircuit relation shared) (inputBindingOffset offset)
      startEq childScope env childAgrees childRows with
    ⟨completed, operationsEq, nextEq, _preserves, _rows⟩
  have operations : completed.operations =
      [childOp "pirlc.v1_1.input_binding"
        (inputBindingCircuit relation shared) (inputBindingOffset offset)] := by
    simpa [empty, Sequence.empty] using operationsEq
  have endEq : offset + localLength completed.operations =
      samplerOffset offset := by
    calc
      offset + localLength completed.operations =
          inputBindingOffset offset +
            localLength (Circuit.ops
              (inputBindingCircuit relation shared).main
              (inputBindingOffset offset)) := nextEq
      _ = samplerOffset offset := by
        unfold inputBindingCircuit
        rw [InputBinding.localLength_eq]
        rfl
  have lengthZero : localLength completed.operations = 0 := by
    simpa [samplerOffset] using endEq
  have currentEq : completed.current = env := by
    funext index
    exact completed.agrees index (by rw [lengthZero]; omega)
  exact ⟨completed, by simpa [shared] using operations, endEq, currentEq⟩

private theorem appendSampler
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (samplerRelation : SamplerChain.RelationHolds
      (samplerInterface (atOffset interface offset)) (samplerOffset offset) env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations = samplerOffset offset)
    (currentEq : before.current = env) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "pirlc.v1_1.sampler_chain"
          (samplerCircuit (atOffset interface offset)) (samplerOffset offset)] ∧
      offset + localLength after.operations = commitmentOffset offset ∧
      Sequence.PreservesPrefix before after := by
  let shared := atOffset interface offset
  have childAssumptions : SamplerChain.Assumptions
      (samplerInterface shared) (samplerOffset offset) before.current := by
    simpa [shared, currentEq] using assumptions.sampler
  have childRelation : SamplerChain.RelationHolds
      (samplerInterface shared) (samplerOffset offset) before.current := by
    simpa [shared, currentEq] using samplerRelation
  rcases SamplerChain.completeness (samplerInterface shared)
      (samplerOffset offset) before.current childAssumptions childRelation with
    ⟨built, builtAgrees, builtRows⟩
  have builtAssumptions : SamplerChain.Assumptions
      (samplerInterface shared) (samplerOffset offset) built := by
    exact { initialBelow := assumptions.sampler.initialBelow }
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (samplerCircuit shared).main (samplerOffset offset)),
      expression.VarsBelow
        (samplerOffset offset + localLength
          (Circuit.ops (samplerCircuit shared).main (samplerOffset offset))) := by
    unfold samplerCircuit SamplerChain.circuit
    rw [SamplerChain.localLength_eq]
    exact SamplerChain.flatConstraints_varsBelow_of_rows
      (samplerInterface shared) (samplerOffset offset) built builtAssumptions
      builtRows
  rcases Sequence.appendBuiltAt before "pirlc.v1_1.sampler_chain"
      (samplerCircuit shared) (samplerOffset offset) startEq childScope built
      builtAgrees builtRows with
    ⟨after, operationsEq, nextEq, preserves, _rows⟩
  have endEq : offset + localLength after.operations = commitmentOffset offset := by
    calc
      offset + localLength after.operations =
          samplerOffset offset + localLength
            (Circuit.ops (samplerCircuit shared).main (samplerOffset offset)) :=
        nextEq
      _ = commitmentOffset offset := by
        unfold samplerCircuit SamplerChain.circuit
        rw [SamplerChain.localLength_eq]
        rfl
  exact ⟨after, by simpa [shared] using operationsEq, endEq, preserves⟩

private theorem appendCommitment
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations = commitmentOffset offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "pirlc.v1_1.commitment_combination"
          (commitmentCircuit (atOffset interface offset))
          (commitmentOffset offset)] ∧
      offset + localLength after.operations = publicInputOffset offset ∧
      Sequence.PreservesPrefix before after := by
  let shared := atOffset interface offset
  have childAssumptions :=
    (assumptionsAt assumptions before.current).commitment
  rcases CommitmentCombination.complete (commitmentInterface shared)
      (commitmentOffset offset) before.current childAssumptions with
    ⟨built, builtAgrees, builtRows⟩
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (commitmentCircuit shared).main (commitmentOffset offset)),
      expression.VarsBelow
        (commitmentOffset offset + localLength
          (Circuit.ops (commitmentCircuit shared).main
            (commitmentOffset offset))) := by
    unfold commitmentCircuit
    rw [CommitmentCombination.localLength_eq]
    exact CommitmentCombination.flatConstraints_varsBelow
      (commitmentInterface shared) (commitmentOffset offset) before.current
      childAssumptions
  rcases Sequence.appendBuiltAt before "pirlc.v1_1.commitment_combination"
      (commitmentCircuit shared) (commitmentOffset offset) startEq childScope
      built builtAgrees builtRows with
    ⟨after, operationsEq, nextEq, preserves, _rows⟩
  have endEq : offset + localLength after.operations = publicInputOffset offset := by
    calc
      offset + localLength after.operations =
          commitmentOffset offset + localLength
            (Circuit.ops (commitmentCircuit shared).main
              (commitmentOffset offset)) := nextEq
      _ = publicInputOffset offset := by
        unfold commitmentCircuit
        rw [CommitmentCombination.localLength_eq]
        rfl
  exact ⟨after, by simpa [shared] using operationsEq, endEq, preserves⟩

private theorem appendPublicInput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations = publicInputOffset offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "pirlc.v1_1.public_input_combination"
          (publicInputCircuit (atOffset interface offset))
          (publicInputOffset offset)] ∧
      offset + localLength after.operations = evalKOffset offset ∧
      Sequence.PreservesPrefix before after := by
  let shared := atOffset interface offset
  have childAssumptions :=
    (assumptionsAt assumptions before.current).publicInput
  rcases PublicInputCombination.complete (publicInputInterface shared)
      (publicInputOffset offset) before.current childAssumptions with
    ⟨built, builtAgrees, builtRows⟩
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (publicInputCircuit shared).main (publicInputOffset offset)),
      expression.VarsBelow
        (publicInputOffset offset + localLength
          (Circuit.ops (publicInputCircuit shared).main
            (publicInputOffset offset))) := by
    unfold publicInputCircuit
    rw [PublicInputCombination.localLength_eq]
    exact PublicInputCombination.flatConstraints_varsBelow
      (publicInputInterface shared) (publicInputOffset offset) before.current
      childAssumptions
  rcases Sequence.appendBuiltAt before "pirlc.v1_1.public_input_combination"
      (publicInputCircuit shared) (publicInputOffset offset) startEq childScope
      built builtAgrees builtRows with
    ⟨after, operationsEq, nextEq, preserves, _rows⟩
  have endEq : offset + localLength after.operations = evalKOffset offset := by
    calc
      offset + localLength after.operations =
          publicInputOffset offset + localLength
            (Circuit.ops (publicInputCircuit shared).main
              (publicInputOffset offset)) := nextEq
      _ = evalKOffset offset := by
        unfold publicInputCircuit
        rw [PublicInputCombination.localLength_eq]
        rfl
  exact ⟨after, by simpa [shared] using operationsEq, endEq, preserves⟩

private theorem appendEvalK
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations = evalKOffset offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "pirlc.v1_1.eval_K_combination"
          (evalKCircuit (atOffset interface offset)) (evalKOffset offset)] ∧
      offset + localLength after.operations = evalAOffset offset ∧
      Sequence.PreservesPrefix before after := by
  let shared := atOffset interface offset
  have childAssumptions := (assumptionsAt assumptions before.current).eval_K
  rcases EvalKCombination.complete (evalKInterface shared) (evalKOffset offset)
      before.current childAssumptions with ⟨built, builtAgrees, builtRows⟩
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (evalKCircuit shared).main (evalKOffset offset)),
      expression.VarsBelow
        (evalKOffset offset + localLength
          (Circuit.ops (evalKCircuit shared).main (evalKOffset offset))) := by
    unfold evalKCircuit
    rw [EvalKCombination.localLength_eq]
    exact EvalKCombination.flatConstraints_varsBelow (evalKInterface shared)
      (evalKOffset offset) before.current childAssumptions
  rcases Sequence.appendBuiltAt before "pirlc.v1_1.eval_K_combination"
      (evalKCircuit shared) (evalKOffset offset) startEq childScope built
      builtAgrees builtRows with
    ⟨after, operationsEq, nextEq, preserves, _rows⟩
  have endEq : offset + localLength after.operations = evalAOffset offset := by
    calc
      offset + localLength after.operations =
          evalKOffset offset + localLength
            (Circuit.ops (evalKCircuit shared).main (evalKOffset offset)) := nextEq
      _ = evalAOffset offset := by
        unfold evalKCircuit
        rw [EvalKCombination.localLength_eq]
        rfl
  exact ⟨after, by simpa [shared] using operationsEq, endEq, preserves⟩

private theorem appendEvalA
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations = evalAOffset offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "pirlc.v1_1.eval_A_combination"
          (evalACircuit (atOffset interface offset)) (evalAOffset offset)] ∧
      offset + localLength after.operations = outputBindingOffset offset ∧
      Sequence.PreservesPrefix before after := by
  let shared := atOffset interface offset
  have childAssumptions := (assumptionsAt assumptions before.current).eval_A
  rcases EvalACombination.complete (evalAInterface shared) (evalAOffset offset)
      before.current childAssumptions with ⟨built, builtAgrees, builtRows⟩
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (evalACircuit shared).main (evalAOffset offset)),
      expression.VarsBelow
        (evalAOffset offset + localLength
          (Circuit.ops (evalACircuit shared).main (evalAOffset offset))) := by
    unfold evalACircuit
    rw [EvalACombination.localLength_eq]
    exact EvalACombination.flatConstraints_varsBelow (evalAInterface shared)
      (evalAOffset offset) before.current childAssumptions
  rcases Sequence.appendBuiltAt before "pirlc.v1_1.eval_A_combination"
      (evalACircuit shared) (evalAOffset offset) startEq childScope built
      builtAgrees builtRows with
    ⟨after, operationsEq, nextEq, preserves, _rows⟩
  have endEq : offset + localLength after.operations = outputBindingOffset offset := by
    calc
      offset + localLength after.operations =
          evalAOffset offset + localLength
            (Circuit.ops (evalACircuit shared).main (evalAOffset offset)) := nextEq
      _ = outputBindingOffset offset := by
        unfold evalACircuit
        rw [EvalACombination.localLength_eq]
        rfl
  exact ⟨after, by simpa [shared] using operationsEq, endEq, preserves⟩

theorem completeSamplerPrefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (phase : Semantics.PhaseHolds relation ajtai interface offset env) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = (opsAt relation interface offset).take 2 ∧
      offset + localLength completed.operations = commitmentOffset offset := by
  rcases completeInputPrefix relation interface env offset with
    ⟨p1, o1, s1, currentEq⟩
  rcases appendSampler relation interface env offset assumptions phase.sampler p1 s1
      currentEq with ⟨p2, o2, s2, _preserves⟩
  refine ⟨p2, ?_, s2⟩
  rw [o2, o1]
  simp [opsAt]

theorem completeCombinationPrefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (phase : Semantics.PhaseHolds relation ajtai interface offset env) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = (opsAt relation interface offset).take 6 ∧
      offset + localLength completed.operations = outputBindingOffset offset := by
  rcases completeSamplerPrefix relation ajtai interface env offset assumptions phase with
    ⟨p2, o2, s2⟩
  rcases appendCommitment relation interface env offset assumptions p2 s2 with
    ⟨p3, o3, s3, _p2to3⟩
  rcases appendPublicInput relation interface env offset assumptions p3 s3 with
    ⟨p4, o4, s4, _p3to4⟩
  rcases appendEvalK relation interface env offset assumptions p4 s4 with
    ⟨p5, o5, s5, _p4to5⟩
  rcases appendEvalA relation interface env offset assumptions p5 s5 with
    ⟨p6, o6, s6, _p5to6⟩
  refine ⟨p6, ?_, s6⟩
  rw [o6, o5, o4, o3, o2]
  simp [opsAt]

private theorem appendOutputBinding
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations =
      outputBindingOffset offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "pirlc.v1_1.output_binding"
          (outputBindingCircuit relation (atOffset interface offset) offset)
          (outputBindingOffset offset)] ∧
      offset + localLength after.operations = finalOffset offset ∧
      Sequence.PreservesPrefix before after := by
  let shared := atOffset interface offset
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (outputBindingCircuit relation shared offset).main
        (outputBindingOffset offset)),
      expression.VarsBelow
        (outputBindingOffset offset + localLength
          (Circuit.ops (outputBindingCircuit relation shared offset).main
            (outputBindingOffset offset))) := by
    unfold outputBindingCircuit
    rw [OutputBinding.localLength_eq]
    exact OutputBinding.flatConstraints_varsBelow relation
      (outputBindingInterface shared offset) (outputBindingOffset offset)
  have childRows : holdsFlat before.current
      (Circuit.ops (outputBindingCircuit relation shared offset).main
        (outputBindingOffset offset)) := by
    intro expression member
    cases member
  have childAgrees : AgreesOutside before.current before.current
      (outputBindingOffset offset)
      (localLength
        (Circuit.ops (outputBindingCircuit relation shared offset).main
          (outputBindingOffset offset))) := by
    intro _ _
    rfl
  rcases Sequence.appendBuiltAt before "pirlc.v1_1.output_binding"
      (outputBindingCircuit relation shared offset) (outputBindingOffset offset)
      startEq childScope before.current childAgrees childRows with
    ⟨after, operationsEq, nextEq, preserves, _rows⟩
  have endEq : offset + localLength after.operations = finalOffset offset := by
    calc
      offset + localLength after.operations =
          outputBindingOffset offset + localLength
            (Circuit.ops (outputBindingCircuit relation shared offset).main
              (outputBindingOffset offset)) := nextEq
      _ = finalOffset offset := by
        unfold outputBindingCircuit
        rw [OutputBinding.localLength_eq]
        rfl
  exact ⟨after, by simpa [shared] using operationsEq, endEq, preserves⟩

theorem completePrefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (phase : Semantics.PhaseHolds relation ajtai interface offset env) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = opsAt relation interface offset := by
  rcases completeCombinationPrefix relation ajtai interface env offset assumptions
      phase with ⟨p6, o6, s6⟩
  rcases appendOutputBinding relation interface env offset p6 s6 with
    ⟨p7, o7, _s7, _p6to7⟩
  refine ⟨p7, ?_⟩
  rw [o7, o6]
  simp [opsAt]

theorem completeness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (phase : Semantics.PhaseHolds relation ajtai interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main relation interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main relation interface) offset) := by
  rcases completePrefix relation ajtai interface env offset assumptions phase with
    ⟨completed, operationsEq⟩
  refine ⟨completed.current, ?_, ?_⟩
  · change AgreesOutside env completed.current offset
      (localLength (opsAt relation interface offset))
    rw [← operationsEq]
    exact completed.agrees
  · change holdsFlat completed.current (opsAt relation interface offset)
    rw [← operationsEq]
    exact completed.rows

/-- The sole proved logical PiRLC v1.1 circuit. -/
def circuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits) : FormalCircuit where
  main := main relation interface
  assumptions := Assumptions relation interface
  spec := Semantics.PhaseHolds relation ajtai interface
  privateCount := fun _ => logicalPrivateCount
  rowCount := fun _ => logicalRowCount
  privateCount_eq := localLength_eq relation interface
  rowCount_eq := flatConstraints_length relation interface
  soundness := by
    intro env offset assumptions rows
    exact Semantics.spec_implies_phaseHolds relation ajtai interface offset env
      (soundness relation interface offset env assumptions rows)
  completeness := completeness relation ajtai interface

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal
