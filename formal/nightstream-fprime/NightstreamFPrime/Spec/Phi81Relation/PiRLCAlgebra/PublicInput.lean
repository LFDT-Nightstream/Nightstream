import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

/-! SuperNeo v1.2 Theorem 11. -/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def publicBlock {shape : Shape}
    (input : PublicInput shape) (block : Fin shape.publicRingColumns) : RingF :=
  fun lane => input ⟨block.val * ringDegree + lane.val, by
    have blockLt := block.isLt
    have laneLt : lane.val < 54 := by
      change lane.val < 54
      exact lane.isLt
    change block.val * 54 + lane.val < 54 * shape.publicRingColumns
    omega⟩

def publicBlockIndex (shape : Shape) (column : Fin shape.publicWidth) :
    Fin shape.publicRingColumns :=
  ⟨column.val / ringDegree, by
    have columnLt := column.isLt
    simp only [Shape.publicWidth, ringDegree] at columnLt ⊢
    omega⟩

def publicLaneIndex {shape : Shape} (column : Fin shape.publicWidth) :
    Fin ringDegree :=
  ⟨column.val % ringDegree, Nat.mod_lt _ (by simp [ringDegree])⟩

def carrierBlock {shape : Shape} (block : Fin shape.publicRingColumns) :
    Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) :=
  ⟨block.val, by
    change block.val < Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth shape.logicalWidth)
    rw [Phi81CarrierLayout.blockCount_carrierWidth]
    have fits := shape.publicFits
    rw [Phi81CarrierLayout.carrierWidth_eq] at fits
    have publicBlocksFit :
        shape.publicRingColumns <=
          Phi81ColumnLayout.blockCount shape.logicalWidth := by
      simp only [ringDegree] at fits ⊢
      omega
    exact Nat.lt_of_lt_of_le block.isLt publicBlocksFit⟩

theorem assignmentBlock_projectPublicInput
    {shape : Shape} (assignment : Assignment shape)
    (block : Fin shape.publicRingColumns) :
    CarrierAction.assignmentBlock assignment (carrierBlock block) =
      publicBlock (projectPublicInput assignment) block := by
  funext lane
  unfold CarrierAction.assignmentBlock publicBlock projectPublicInput
  apply congrArg assignment
  apply Fin.ext
  rfl

def publicAct {shape : Shape}
    (challenge : RingF) (input : PublicInput shape) : PublicInput shape :=
  fun column =>
    ringFMul challenge (publicBlock input (publicBlockIndex shape column))
      (publicLaneIndex column)

def publicAdd {shape : Shape}
    (left right : PublicInput shape) : PublicInput shape :=
  fun column => left column + right column

def publicZero {shape : Shape} : PublicInput shape := fun _ => 0

theorem projectPublicInput_act
    {shape : Shape} (challenge : RingF) (assignment : Assignment shape) :
    projectPublicInput (CarrierAction.act challenge assignment) =
      publicAct challenge (projectPublicInput assignment) := by
  funext column
  unfold projectPublicInput CarrierAction.act publicAct
  simp only []
  rfl

theorem projectPublicInput_add
    {shape : Shape} (left right : Assignment shape) :
    projectPublicInput (BaseLinear.assignmentAdd left right) =
      publicAdd (projectPublicInput left) (projectPublicInput right) := by
  rfl

theorem projectPublicInput_zero {shape : Shape} :
    projectPublicInput (BaseLinear.assignmentZero : Assignment shape) =
      publicZero := by
  rfl

def combinePublicInputs {shape : Shape} :
    {count : Nat} ->
      (Fin count -> RingF) ->
      (Fin count -> PublicInput shape) -> PublicInput shape
  | 0, _, _ => publicZero
  | _ + 1, challenges, inputs =>
      publicAdd
        (publicAct (challenges 0) (inputs 0))
        (combinePublicInputs
          (fun index => challenges index.succ)
          (fun index => inputs index.succ))

theorem projectPublicInput_combine
    {shape : Shape} {count : Nat}
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape) :
    projectPublicInput (PiRLCFinite.combineAssignments challenges assignments) =
      combinePublicInputs challenges
        (fun index => projectPublicInput (assignments index)) := by
  induction count with
  | zero => exact projectPublicInput_zero
  | succ count inductionHypothesis =>
      rw [PiRLCFinite.combineAssignments, combinePublicInputs,
        projectPublicInput_add, projectPublicInput_act]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => assignments index.succ)]

theorem relation_publicInput_hom
    {shape : Shape} {Commitment : Type}
    (commit : Assignment shape -> Commitment)
    {count : Nat} (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape) :
    (relationSemantics commit).projectPublicInput
        (PiRLCFinite.combineAssignments challenges assignments) =
      combinePublicInputs challenges
        (fun index =>
          (relationSemantics commit).projectPublicInput (assignments index)) := by
  exact projectPublicInput_combine challenges assignments

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput
