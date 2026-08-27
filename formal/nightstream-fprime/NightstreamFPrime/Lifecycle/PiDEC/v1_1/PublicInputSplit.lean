import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.SignedSplitScalar
import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.PublicInput

/-!
Paper authority: SuperNeo v1.1, Section 7.5, verifier Step 2.

Obligation: reject a parent public input with any coordinate outside
`B = 2^16`, and constrain the exact signed-binary split of all 270 production
public coordinates into sixteen children.

Inputs:
- one 270-field parent public input;
- one 270-field public input for each of the 16 child claims.

Outputs:
- 16 child public inputs proved equal to verifier-owned `split_b(parent)`.

Constraint groups:
- P1: one `SignedSplitScalar` child for each public coordinate;
- P2: no copy or boundary rows.

Parent coverage:
- `PiDEC.PaperVerifier.Accepted.parentBounded`;
- verifier-computed child public inputs in `PiDEC.PaperVerifier.children`.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

def coordinateCount (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Nat :=
  (FullShape logicalWidth publicFits).publicWidth

theorem coordinateCount_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    coordinateCount logicalWidth publicFits = 270 := by
  norm_num [coordinateCount, FullShape, fullShape,
    Phi81Relation.Shape.publicWidth, publicRingColumns, ringDegree]

def logicalPrivateCount
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Nat :=
  coordinateCount logicalWidth publicFits *
    SignedSplitScalar.exactPrivateCount

def logicalRowCount
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Nat :=
  coordinateCount logicalWidth publicFits * SignedSplitScalar.exactRowCount

theorem logicalPrivateCount_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    logicalPrivateCount logicalWidth publicFits = 270 := by
  rw [logicalPrivateCount, coordinateCount_eq]
  rfl

theorem logicalRowCount_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    logicalRowCount logicalWidth publicFits = 4860 := by
  rw [logicalRowCount, coordinateCount_eq]
  norm_num [SignedSplitScalar.exactRowCount]

structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  parent : Nat → Fin (coordinateCount logicalWidth publicFits) → Expr
  digit : Nat → Radix.ChildIndex →
    Fin (coordinateCount logicalWidth publicFits) → Expr

def evalParent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    PaperAlgebra.PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  fun coordinate => (interface.parent offset coordinate).eval env

def evalChildren
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    Radix.ChildIndex → PaperAlgebra.PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  fun child coordinate => (interface.digit offset child coordinate).eval env

def sourceOffset (offset source : Nat) : Nat :=
  offset + source * SignedSplitScalar.exactPrivateCount

def childInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset source : Nat)
    (sourceLt : source < coordinateCount logicalWidth publicFits) :
    SignedSplitScalar.Interface where
  parent := fun _ => interface.parent offset ⟨source, sourceLt⟩
  digit := fun _ child => interface.digit offset child ⟨source, sourceLt⟩

def childName (source : Nat) : String :=
  "pidec.v1_1.public_split.coordinate_" ++ toString source

def childOp
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset source : Nat) : Op :=
  if sourceLt : source < coordinateCount logicalWidth publicFits then
    Sequence.childOp (childName source)
      (SignedSplitScalar.circuit
        (childInterface interface offset source sourceLt))
      (sourceOffset offset source)
  else .assertZero 0

def opsPrefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset count : Nat) : List Op :=
  (List.range count).map (childOp interface offset)

def opsAt
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) : List Op :=
  opsPrefix interface offset (coordinateCount logicalWidth publicFits)

def main
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : Circuit Unit := fun offset =>
  ((), offset + logicalPrivateCount logicalWidth publicFits,
    opsAt interface offset)

@[simp] private theorem childOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset source : Nat)
    (sourceLt : source < coordinateCount logicalWidth publicFits) :
    (childOp interface offset source).localLength =
      SignedSplitScalar.exactPrivateCount := by
  rw [childOp, dif_pos sourceLt, Sequence.childOp_localLength]
  exact SignedSplitScalar.localLength_eq
    (childInterface interface offset source sourceLt)
    (sourceOffset offset source)

@[simp] private theorem childOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset source : Nat)
    (sourceLt : source < coordinateCount logicalWidth publicFits) :
    (childOp interface offset source).rowCount =
      SignedSplitScalar.exactRowCount := by
  rw [childOp, dif_pos sourceLt]
  rfl

@[simp] private theorem opsPrefix_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset count : Nat)
    (bounded : count ≤ coordinateCount logicalWidth publicFits) :
    localLength (opsPrefix interface offset count) =
      count * SignedSplitScalar.exactPrivateCount := by
  unfold opsPrefix localLength
  rw [List.map_map]
  calc
    (List.map (Op.localLength ∘ childOp interface offset)
        (List.range count)).sum =
        (List.map (Op.localLength ∘ childOp interface offset)
          (List.range count)).length • SignedSplitScalar.exactPrivateCount := by
      apply List.sum_eq_card_nsmul
      intro operation member
      rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
      rw [Function.comp_apply, childOp_localLength]
      exact lt_of_lt_of_le (List.mem_range.mp sourceMember) bounded
    _ = count * SignedSplitScalar.exactPrivateCount := by simp

theorem localLength_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) =
      logicalPrivateCount logicalWidth publicFits := by
  change localLength (opsAt interface offset) = _
  exact opsPrefix_localLength interface offset _ (Nat.le_refl _)

theorem flatConstraints_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length =
      logicalRowCount logicalWidth publicFits := by
  rw [flatConstraints_length_eq_rowCount]
  change rowCount (opsAt interface offset) = _
  unfold opsAt opsPrefix rowCount logicalRowCount
  rw [List.map_map]
  calc
    (List.map (Op.rowCount ∘ childOp interface offset)
        (List.range (coordinateCount logicalWidth publicFits))).sum =
        (List.map (Op.rowCount ∘ childOp interface offset)
          (List.range (coordinateCount logicalWidth publicFits))).length •
            SignedSplitScalar.exactRowCount := by
      apply List.sum_eq_card_nsmul
      intro operation member
      rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
      rw [Function.comp_apply, childOp_rowCount]
      exact List.mem_range.mp sourceMember
    _ = coordinateCount logicalWidth publicFits *
        SignedSplitScalar.exactRowCount := by simp

structure Assumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (_env : Env) : Prop where
  parentBelow : ∀ coordinate, (interface.parent offset coordinate).VarsBelow offset
  digitBelow : ∀ child coordinate,
    (interface.digit offset child coordinate).VarsBelow offset

theorem childAssumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset source : Nat)
    (sourceLt : source < coordinateCount logicalWidth publicFits)
    (env : Env) (assumptions : Assumptions interface offset env) :
    SignedSplitScalar.Assumptions
      (childInterface interface offset source sourceLt)
      (sourceOffset offset source) env := by
  constructor
  · exact Expr.VarsBelow.mono _ (assumptions.parentBelow ⟨source, sourceLt⟩)
      (by simp [sourceOffset])
  · intro child
    exact Expr.VarsBelow.mono _
      (assumptions.digitBelow child ⟨source, sourceLt⟩)
      (by simp [sourceOffset])

abbrev RelationHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : Prop :=
  ∀ coordinate : Fin (coordinateCount logicalWidth publicFits),
    SignedSplitScalar.SpecHolds
      (childInterface interface offset coordinate.val coordinate.isLt)
      (sourceOffset offset coordinate.val) env

theorem parentBounded
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (relation : RelationHolds interface offset env) :
    PiDECAlgebra.PublicInput.parentBounded (evalParent interface offset env) := by
  intro coordinate
  exact SignedSplitScalar.spec_parentBounded (relation coordinate)

theorem children_eq_splitPublicInput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (relation : RelationHolds interface offset env) :
    evalChildren interface offset env =
      PiDECAlgebra.PublicInput.splitPublicInput
        (evalParent interface offset env) := by
  funext child coordinate
  exact congrFun (SignedSplitScalar.spec_digits_eq_splitScalar
    (relation coordinate)) child

theorem relationHolds_of_parentBounded_children_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (bounded : PiDECAlgebra.PublicInput.parentBounded
      (evalParent interface offset env))
    (children : evalChildren interface offset env =
      PiDECAlgebra.PublicInput.splitPublicInput
        (evalParent interface offset env)) :
    RelationHolds interface offset env := by
  intro coordinate
  let child := childInterface interface offset coordinate.val coordinate.isLt
  have exactDigits :
      SignedSplitScalar.digitValues child
          (sourceOffset offset coordinate.val) env =
        Radix.splitScalar
          (SignedSplitScalar.parentValue child
            (sourceOffset offset coordinate.val) env) := by
    funext index
    have value := congrFun (congrFun children index) coordinate
    simpa [child, childInterface, SignedSplitScalar.digitValues,
      SignedSplitScalar.parentValue, evalChildren, evalParent,
      PiDECAlgebra.PublicInput.splitPublicInput] using value
  refine ⟨Radix.UniformSignedDigits.honestSign
    (SignedSplitScalar.parentValue child
      (sourceOffset offset coordinate.val) env), ?_⟩
  rw [exactDigits]
  apply Radix.UniformSignedDigits.honest_complete
  simpa [child, childInterface, SignedSplitScalar.parentValue,
    evalParent] using bounded coordinate

private theorem childHolds_of_rows
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    RelationHolds interface offset env := by
  intro coordinate
  have member : childOp interface offset coordinate.val ∈ opsAt interface offset := by
    apply List.mem_map.mpr
    exact ⟨coordinate.val, List.mem_range.mpr coordinate.isLt, rfl⟩
  have call := rows _ member
  rw [childOp, dif_pos coordinate.isLt] at call
  change SignedSplitScalar.Assumptions
      (childInterface interface offset coordinate.val coordinate.isLt)
      (sourceOffset offset coordinate.val) env →
    SignedSplitScalar.SpecHolds
      (childInterface interface offset coordinate.val coordinate.isLt)
      (sourceOffset offset coordinate.val) env at call
  exact call (childAssumptions interface offset coordinate.val coordinate.isLt
    env assumptions)

theorem soundness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    RelationHolds interface offset env :=
  childHolds_of_rows interface offset env assumptions rows

private theorem childScope
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset source : Nat)
    (sourceLt : source < coordinateCount logicalWidth publicFits)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops
        (SignedSplitScalar.circuit
          (childInterface interface offset source sourceLt)).main
        (sourceOffset offset source)),
      expression.VarsBelow
        (sourceOffset offset source + SignedSplitScalar.exactPrivateCount) := by
  exact SignedSplitScalar.flatConstraints_varsBelow
    (childInterface interface offset source sourceLt)
    (sourceOffset offset source) env
    (childAssumptions interface offset source sourceLt env assumptions)

theorem flatConstraints_varsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow
        (offset + logicalPrivateCount logicalWidth publicFits) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset), _
  intro expression member
  rcases List.mem_flatMap.mp member with
    ⟨operation, operationMember, expressionMember⟩
  rcases List.mem_map.mp operationMember with ⟨source, sourceMember, rfl⟩
  have sourceLt := List.mem_range.mp sourceMember
  apply Expr.VarsBelow.mono expression
    (childScope interface offset source sourceLt env assumptions expression (by
      simpa [childOp, dif_pos sourceLt, Sequence.childOp] using expressionMember))
  have scaled := Nat.mul_le_mul_right SignedSplitScalar.exactPrivateCount
    (Nat.succ_le_iff.mpr sourceLt)
  simpa [sourceOffset, logicalPrivateCount, Nat.succ_mul, Nat.add_assoc] using
    Nat.add_le_add_left scaled offset

private theorem childSpec_of_prefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset source : Nat)
    (sourceLt : source < coordinateCount logicalWidth publicFits)
    (env : Env) (assumptions : Assumptions interface offset env)
    (relation : RelationHolds interface offset env)
    (before : Sequence.Prefix env offset) :
    SignedSplitScalar.SpecHolds
      (childInterface interface offset source sourceLt)
      (sourceOffset offset source) before.current := by
  rcases relation ⟨source, sourceLt⟩ with ⟨sign, accepted⟩
  refine ⟨sign, ?_⟩
  have agrees : ∀ index, index < offset → before.current index = env index := by
    intro index below
    exact before.agrees index (Or.inl below)
  have parentEq :
      SignedSplitScalar.parentValue
          (childInterface interface offset source sourceLt)
          (sourceOffset offset source) before.current =
        SignedSplitScalar.parentValue
          (childInterface interface offset source sourceLt)
          (sourceOffset offset source) env := by
    exact Expr.eval_eq_of_agree_below _ offset _ _
      (assumptions.parentBelow ⟨source, sourceLt⟩) agrees
  have digitsEq :
      SignedSplitScalar.digitValues
          (childInterface interface offset source sourceLt)
          (sourceOffset offset source) before.current =
        SignedSplitScalar.digitValues
          (childInterface interface offset source sourceLt)
          (sourceOffset offset source) env := by
    funext child
    exact Expr.eval_eq_of_agree_below _ offset _ _
      (assumptions.digitBelow child ⟨source, sourceLt⟩) agrees
  simpa only [parentEq, digitsEq] using accepted

private theorem completePrefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (relation : RelationHolds interface offset env)
    (count : Nat) (bounded : count ≤ coordinateCount logicalWidth publicFits) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = opsPrefix interface offset count := by
  induction count with
  | zero => exact ⟨Sequence.empty env offset, rfl⟩
  | succ count inductionHypothesis =>
      have countLt : count < coordinateCount logicalWidth publicFits := by omega
      rcases inductionHypothesis (Nat.le_of_lt countLt) with
        ⟨before, beforeOperations⟩
      have startEq : offset + localLength before.operations =
          sourceOffset offset count := by
        rw [beforeOperations, opsPrefix_localLength interface offset count
          (Nat.le_of_lt countLt)]
        rfl
      have currentParentAssumptions :
          Assumptions interface offset before.current :=
        ⟨assumptions.parentBelow, assumptions.digitBelow⟩
      have currentAssumptions := childAssumptions interface offset count countLt
        before.current currentParentAssumptions
      have currentSpec := childSpec_of_prefix interface offset count countLt env
        assumptions relation before
      have scope : ∀ expression ∈ flatConstraints
          (Circuit.ops
            (SignedSplitScalar.circuit
              (childInterface interface offset count countLt)).main
            (sourceOffset offset count)),
          expression.VarsBelow
            (sourceOffset offset count +
              localLength (Circuit.ops
                (SignedSplitScalar.circuit
                  (childInterface interface offset count countLt)).main
                (sourceOffset offset count))) := by
        have exactScope := childScope interface offset count countLt
          before.current currentParentAssumptions
        have childLength : localLength
            (Circuit.ops
              (SignedSplitScalar.circuit
                (childInterface interface offset count countLt)).main
              (sourceOffset offset count)) =
            SignedSplitScalar.exactPrivateCount := by
          exact SignedSplitScalar.localLength_eq
            (childInterface interface offset count countLt)
            (sourceOffset offset count)
        simpa only [childLength] using exactScope
      rcases Sequence.appendAt before (childName count)
          (SignedSplitScalar.circuit
            (childInterface interface offset count countLt))
          (sourceOffset offset count) startEq scope currentAssumptions currentSpec with
        ⟨completed, completedOperations, _, _, _⟩
      refine ⟨completed, ?_⟩
      rw [completedOperations, beforeOperations]
      simp [opsPrefix, List.range_succ, childOp, dif_pos countLt]

theorem completeness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (relation : RelationHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  rcases completePrefix interface offset env assumptions relation
      (coordinateCount logicalWidth publicFits) (Nat.le_refl _) with
    ⟨completed, operationsEq⟩
  refine ⟨completed.current, ?_, ?_⟩
  · have agrees := completed.agrees
    rw [operationsEq] at agrees
    exact agrees
  · change holdsFlat completed.current
      (opsPrefix interface offset (coordinateCount logicalWidth publicFits))
    rw [← operationsEq]
    exact completed.rows

def circuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := RelationHolds interface
  privateCount := fun _ => logicalPrivateCount logicalWidth publicFits
  rowCount := fun _ => logicalRowCount logicalWidth publicFits
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length interface
  soundness := by
    intro env offset assumptions rows
    exact soundness interface offset env assumptions rows
  completeness := by
    intro env offset assumptions relation
    exact completeness interface offset env assumptions relation

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit
