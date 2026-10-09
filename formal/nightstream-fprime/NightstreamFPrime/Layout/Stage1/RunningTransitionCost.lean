import NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
import NightstreamFPrime.Layout.Polynomial.Horner

/-!
Owns the exact R1CS footprint and endpoints of the Stage 1 running transition.
The proof is structural and does not normalize the complete running vector.
-/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def logicalColumnCount : Nat :=
  phaseOffset + RunningTransition.exactPrivateCount

/-- Every running-transition constraint is one rank-one row, so the
transition allocates no fresh column. -/
def exactFreshCount : Nat := 0

/-- The running transition ends after its logical and R1CS fresh columns. -/
def physicalEnd : Nat := logicalColumnCount + exactFreshCount

structure RunningMulFree {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits) : Prop where
  point : ∀ coordinate, KExprLinear (running.point coordinate)
  commitment : ∀ source row coefficient,
    R1CS.mulCount (running.commitment source row coefficient) = 0
  publicInput : ∀ source column,
    R1CS.mulCount (running.publicInput source column) = 0
  eval_K : ∀ source coefficient,
    KExprLinear ((running.evaluation source).eval_K coefficient)
  eval_A : ∀ source matrix coefficient,
    KExprLinear ((running.evaluation source).eval_A matrix coefficient)

private theorem serializeKExpr_mulFree (value : KExpr)
    (linear : KExprLinear value) :
    ∀ expression ∈ StatementAbsorption.serializeKExpr value,
      R1CS.mulCount expression = 0 := by
  intro expression member
  simp only [StatementAbsorption.serializeKExpr, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact linear.c0_mulCount
  · exact linear.c1_mulCount

/-- Every hashed running word is affine: the fields are multiplication-free
and each packed parent word is a constant-weighted sum of child inputs. -/
private theorem serializeRunningExpr_affine {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (linear : RunningMulFree running) :
    ∀ expression ∈ RunningWords.serializeRunningExpr running,
      R1CS.IsAffine expression := by
  intro expression member
  rcases RunningWords.serializeRunningExpr_mem member with
    ⟨source, row, coefficient, rfl⟩ | ⟨source, coefficient, kMember⟩ |
      ⟨source, matrix, coefficient, kMember⟩ | ⟨coordinate, kMember⟩ | ⟨word, rfl⟩
  · exact isAffine_of_mulCount_zero _ (linear.commitment source row coefficient)
  · exact isAffine_of_mulCount_zero _
      (serializeKExpr_mulFree _ (linear.eval_K source coefficient) _ kMember)
  · exact isAffine_of_mulCount_zero _
      (serializeKExpr_mulFree _ (linear.eval_A source matrix coefficient) _ kMember)
  · exact isAffine_of_mulCount_zero _
      (serializeKExpr_mulFree _ (linear.point coordinate) _ kMember)
  · exact RunningWords.packWordExpr_parent_closed R1CS.IsAffine
      R1CS.isAffine_const (fun _ _ => R1CS.IsAffine.add)
      (fun weight _ => R1CS.IsAffine.const_mul weight) running
      (fun source column =>
        isAffine_of_mulCount_zero _ (linear.publicInput source column)) word

theorem runningWord_isAffine {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (linear : RunningMulFree running) (index : RunningTransition.WordIndex) :
    R1CS.IsAffine (RunningTransition.runningWord running index) := by
  have indexBound : index.val <
      (RunningWords.serializeRunningExpr running).length := by
    rw [RunningWords.serializeRunningExpr_length]
    exact index.isLt
  rw [RunningTransition.runningWord,
    List.getD_eq_get _ _ ⟨index.val, indexBound⟩]
  exact serializeRunningExpr_affine running linear _
    (List.get_mem _ ⟨index.val, indexBound⟩)

def recursiveMulFree
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    RunningMulFree (recursiveRunningExpr logicalWidth publicFits) := by
  refine {
    point := recursivePointLinear relation
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_ }
  · intro source row coefficient
    rfl
  · intro source column
    rfl
  · intro source coefficient
    refine ⟨rfl, rfl, ?_, ?_⟩ <;>
      simp [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
        PiDECInputs.message, PiDECInputs.childEvalK, Nonconstant]
  · intro source matrix coefficient
    refine ⟨rfl, rfl, ?_, ?_⟩ <;>
      simp [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
        PiDECInputs.message, PiDECInputs.childEvalA, Nonconstant]

def logicalConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  flatConstraints
    (RunningTransition.operations (interface logicalWidth publicFits) phaseOffset)

theorem logicalConstraints_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    logicalConstraints logicalWidth publicFits =
      RunningTransition.flagConstraint (interface logicalWidth publicFits)
          phaseOffset ::
        RunningTransition.constraints (interface logicalWidth publicFits)
          phaseOffset := by
  exact RunningTransition.flatConstraints_operations _ _

private theorem runningWord_affine {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (linear : RunningMulFree running) (index : RunningTransition.WordIndex) :
    R1CS.IsAffine
      (RunningTransition.runningWord running index - Expr.const
        (RunningTransition.defaultWord
          (logicalWidth := logicalWidth) (publicFits := publicFits) index)) :=
  R1CS.IsAffine.add (runningWord_isAffine running linear index)
    (R1CS.IsAffine.const_mul _ (R1CS.isAffine_const _))

private theorem baseFlag_affine
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    R1CS.IsAffine
      (RunningTransition.baseFlag (interface logicalWidth publicFits)
        phaseOffset) :=
  R1CS.IsAffine.add (R1CS.isAffine_const _)
    (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var _))

/-- The flag recipe and every assertion are one rank-one row. -/
private theorem constraintFreshCount_eq_zero
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ expression ∈ logicalConstraints logicalWidth publicFits,
      R1CS.constraintFreshCount expression = 0 := by
  have iterationAffine : R1CS.IsAffine iterationExpr := R1CS.isAffine_var _
  intro expression member
  rw [logicalConstraints_eq] at member
  simp only [RunningTransition.constraints, List.mem_cons,
    List.mem_append] at member
  rcases member with rfl | rfl | muxMember | stateMember
  · exact R1CS.constraintFreshCount_recipe_eq_zero _ _
      (R1CS.IsDirectRecipe.mul _ iterationAffine (R1CS.isAffine_var _))
  · exact R1CS.constraintFreshCount_rankOne_eq_zero _ _ _
      (R1CS.isAffine_const _) iterationAffine
      (baseFlag_affine logicalWidth publicFits)
  · rcases List.mem_ofFn.mp muxMember with ⟨index, rfl⟩
    exact R1CS.constraintFreshCount_rankOne_eq_zero _ _ _
      (R1CS.IsAffine.add (R1CS.isAffine_var _)
        (R1CS.IsAffine.const_mul _ (R1CS.isAffine_const _)))
      (R1CS.isAffine_var _)
      (runningWord_affine _ (recursiveMulFree relation) index)
  · rcases List.mem_ofFn.mp stateMember with ⟨index, rfl⟩
    exact R1CS.constraintFreshCount_rankOne_eq_zero _ _ _
      (R1CS.isAffine_const _) (baseFlag_affine logicalWidth publicFits)
      (R1CS.IsAffine.add (R1CS.isAffine_var _)
        (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var _)))

/-- Structural lowering uses exactly the fresh columns declared by the footprint. -/
theorem totalFreshCount_eq_exactFreshCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount (logicalConstraints logicalWidth publicFits) =
      exactFreshCount :=
  R1CS.totalFreshCount_eq_zero_of_noFresh _
    (constraintFreshCount_eq_zero relation)

theorem totalFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount (logicalConstraints logicalWidth publicFits) = 0 :=
  totalFreshCount_eq_exactFreshCount relation

theorem logicalConstraints_length_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (logicalConstraints logicalWidth publicFits).length = 27800 := by
  exact RunningTransition.flatConstraints_length_eq _ _

theorem totalRowCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalRowCount (logicalConstraints logicalWidth publicFits) =
      27800 := by
  rw [R1CS.totalRowCount_eq_fresh_add_length,
    totalFreshCount_eq relation, logicalConstraints_length_eq]

end NightstreamFPrime.Layout.Stage1.RunningTransitionLayout
