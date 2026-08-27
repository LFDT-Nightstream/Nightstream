import NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics

/-!
Completes the six-child PiDEC v1.1 logical assembler. This file owns only
ordered child-witness composition and the sole parent `FormalCircuit`. It adds
no protocol predicate or verifier row.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

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
  let shared := atOffset interface offset
  have phaseAt : ∀ before : Sequence.Prefix env offset,
      Semantics.PhaseHolds relation ajtai interface offset before.current := by
    intro before
    apply Semantics.phaseHolds_of_agree relation ajtai interface offset
      env before.current assumptions
    · intro index below
      exact (before.agrees index (Or.inl below)).symm
    · exact phase
  have specAt : ∀ before : Sequence.Prefix env offset,
      SpecHolds relation interface offset before.current := by
    intro before
    exact Semantics.phaseHolds_implies_spec relation ajtai interface offset
      before.current (phaseAt before)

  let p0 := Sequence.empty env offset
  have inputStart : offset + localLength p0.operations =
      inputBindingOffset offset := by
    simp [p0, Sequence.empty, inputBindingOffset, localLength]
  have inputLength : localLength
      (Circuit.ops (inputBindingCircuit relation shared).main
        (inputBindingOffset offset)) = 0 := by
    calc
      _ = (inputBindingCircuit relation shared).privateCount
          (inputBindingOffset offset) :=
        (inputBindingCircuit relation shared).privateCount_eq _
      _ = 0 := rfl
  have inputScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (inputBindingCircuit relation shared).main
        (inputBindingOffset offset)),
      expression.VarsBelow
        (inputBindingOffset offset + localLength
          (Circuit.ops (inputBindingCircuit relation shared).main
            (inputBindingOffset offset))) := by
    rw [inputLength]
    simpa [inputBindingOffset] using
      InputBinding.flatConstraints_varsBelow relation
        (inputBindingInterface shared) (inputBindingOffset offset)
  rcases Sequence.appendAt p0 "pidec.v1_1.input_binding"
      (inputBindingCircuit relation shared) (inputBindingOffset offset)
      inputStart inputScope trivial (specAt p0).inputBinding with
    ⟨p1, o1, s1, _preserves1, _rows1⟩

  have publicStart : offset + localLength p1.operations =
      publicInputOffset offset := by
    calc
      offset + localLength p1.operations =
          inputBindingOffset offset + localLength
            (Circuit.ops (inputBindingCircuit relation shared).main
              (inputBindingOffset offset)) := s1
      _ = publicInputOffset offset := by
        rw [inputLength]
        rfl
  have publicAssumptions := publicInputAssumptions assumptions p1.current
  have publicLength : localLength
      (Circuit.ops (publicInputCircuit shared).main
        (publicInputOffset offset)) = 54 := by
    calc
      _ = (publicInputCircuit shared).privateCount
          (publicInputOffset offset) :=
        (publicInputCircuit shared).privateCount_eq _
      _ = PublicInputSplit.logicalPrivateCount logicalWidth publicFits := rfl
      _ = 54 := PublicInputSplit.logicalPrivateCount_eq logicalWidth publicFits
  have publicScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (publicInputCircuit shared).main (publicInputOffset offset)),
      expression.VarsBelow
        (publicInputOffset offset + localLength
          (Circuit.ops (publicInputCircuit shared).main
            (publicInputOffset offset))) := by
    rw [publicLength]
    simpa [publicInputOffset, PublicInputSplit.logicalPrivateCount_eq] using
      PublicInputSplit.flatConstraints_varsBelow
        (publicInputInterface shared) (publicInputOffset offset)
        p1.current publicAssumptions
  rcases Sequence.appendAt p1 "pidec.v1_1.public_input_split"
      (publicInputCircuit shared) (publicInputOffset offset)
      publicStart publicScope publicAssumptions (specAt p1).publicInput with
    ⟨p2, o2, s2, _preserves2, _rows2⟩

  have commitmentStart : offset + localLength p2.operations =
      commitmentOffset offset := by
    calc
      offset + localLength p2.operations =
          publicInputOffset offset + localLength
            (Circuit.ops (publicInputCircuit shared).main
              (publicInputOffset offset)) := s2
      _ = commitmentOffset offset := by
        rw [publicLength]
        rfl
  have commitmentAssumptionsAt := commitmentAssumptions assumptions p2.current
  have commitmentLength : localLength
      (Circuit.ops (commitmentCircuit shared).main
        (commitmentOffset offset)) = 0 := by
    calc
      _ = (commitmentCircuit shared).privateCount
          (commitmentOffset offset) :=
        (commitmentCircuit shared).privateCount_eq _
      _ = 0 := rfl
  have commitmentScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (commitmentCircuit shared).main (commitmentOffset offset)),
      expression.VarsBelow
        (commitmentOffset offset + localLength
          (Circuit.ops (commitmentCircuit shared).main
            (commitmentOffset offset))) := by
    rw [commitmentLength]
    exact CommitmentRecomposition.flatConstraints_varsBelow
      (commitmentInterface shared) (commitmentOffset offset)
      p2.current commitmentAssumptionsAt
  rcases Sequence.appendAt p2 "pidec.v1_1.commitment_recomposition"
      (commitmentCircuit shared) (commitmentOffset offset)
      commitmentStart commitmentScope commitmentAssumptionsAt
      (specAt p2).commitment with
    ⟨p3, o3, s3, _preserves3, _rows3⟩

  have evalKStart : offset + localLength p3.operations = evalKOffset offset := by
    calc
      offset + localLength p3.operations =
          commitmentOffset offset + localLength
            (Circuit.ops (commitmentCircuit shared).main
              (commitmentOffset offset)) := s3
      _ = evalKOffset offset := by
        rw [commitmentLength]
        rfl
  have evalKAssumptionsAt := evalKAssumptions assumptions p3.current
  have evalKLength : localLength
      (Circuit.ops (evalKCircuit shared).main (evalKOffset offset)) = 0 := by
    calc
      _ = (evalKCircuit shared).privateCount (evalKOffset offset) :=
        (evalKCircuit shared).privateCount_eq _
      _ = 0 := rfl
  have evalKScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (evalKCircuit shared).main (evalKOffset offset)),
      expression.VarsBelow
        (evalKOffset offset + localLength
          (Circuit.ops (evalKCircuit shared).main (evalKOffset offset))) := by
    rw [evalKLength]
    exact EvalKRecomposition.flatConstraints_varsBelow
      (evalKInterface shared) (evalKOffset offset) p3.current
      evalKAssumptionsAt
  rcases Sequence.appendAt p3 "pidec.v1_1.eval_K_recomposition"
      (evalKCircuit shared) (evalKOffset offset) evalKStart evalKScope
      evalKAssumptionsAt (specAt p3).eval_K with
    ⟨p4, o4, s4, _preserves4, _rows4⟩

  have evalAStart : offset + localLength p4.operations = evalAOffset offset := by
    calc
      offset + localLength p4.operations =
          evalKOffset offset + localLength
            (Circuit.ops (evalKCircuit shared).main (evalKOffset offset)) := s4
      _ = evalAOffset offset := by
        rw [evalKLength]
        rfl
  have evalAAssumptionsAt := evalAAssumptions assumptions p4.current
  have evalALength : localLength
      (Circuit.ops (evalACircuit shared).main (evalAOffset offset)) = 0 := by
    calc
      _ = (evalACircuit shared).privateCount (evalAOffset offset) :=
        (evalACircuit shared).privateCount_eq _
      _ = 0 := rfl
  have evalAScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (evalACircuit shared).main (evalAOffset offset)),
      expression.VarsBelow
        (evalAOffset offset + localLength
          (Circuit.ops (evalACircuit shared).main (evalAOffset offset))) := by
    rw [evalALength]
    exact EvalARecomposition.flatConstraints_varsBelow
      (evalAInterface shared) (evalAOffset offset) p4.current
      evalAAssumptionsAt
  rcases Sequence.appendAt p4 "pidec.v1_1.eval_A_recomposition"
      (evalACircuit shared) (evalAOffset offset) evalAStart evalAScope
      evalAAssumptionsAt (specAt p4).eval_A with
    ⟨p5, o5, s5, _preserves5, _rows5⟩

  have outputStart : offset + localLength p5.operations =
      outputBindingOffset offset := by
    calc
      offset + localLength p5.operations =
          evalAOffset offset + localLength
            (Circuit.ops (evalACircuit shared).main (evalAOffset offset)) := s5
      _ = outputBindingOffset offset := by
        rw [evalALength]
        rfl
  have outputLength : localLength
      (Circuit.ops (outputBindingCircuit relation shared).main
        (outputBindingOffset offset)) = 0 := by
    calc
      _ = (outputBindingCircuit relation shared).privateCount
          (outputBindingOffset offset) :=
        (outputBindingCircuit relation shared).privateCount_eq _
      _ = 0 := rfl
  have outputScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (outputBindingCircuit relation shared).main
        (outputBindingOffset offset)),
      expression.VarsBelow
        (outputBindingOffset offset + localLength
          (Circuit.ops (outputBindingCircuit relation shared).main
            (outputBindingOffset offset))) := by
    rw [outputLength]
    simpa using OutputBinding.flatConstraints_varsBelow relation
      (outputBindingInterface shared) (outputBindingOffset offset)
  rcases Sequence.appendAt p5 "pidec.v1_1.output_binding"
      (outputBindingCircuit relation shared) (outputBindingOffset offset)
      outputStart outputScope trivial (specAt p5).outputBinding with
    ⟨p6, o6, _s6, _preserves6, _rows6⟩

  refine ⟨p6, ?_⟩
  rw [o6, o5, o4, o3, o2, o1]
  simp [p0, Sequence.empty, opsAt, shared, childOp]

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

/-- The sole proved logical PiDEC v1.1 circuit. -/
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

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal
