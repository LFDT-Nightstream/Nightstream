import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fintype.BigOperators
import NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep

/-!
Owns the structural production footprint of one PiRLC combination step.

The proof counts the fixed 54-lane Phi81 convolution once, transports that
cost through the circuit's exact block/lane/cell index equivalence, and never
reduces a complete 17-source family in the kernel.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

/-- Exact production expression shapes. Sampler challenges are centered
output variables, inputs are existing variables, and the accumulator is
multiplication-free. -/
structure ProductionInputs {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) : Prop where
  challenge : ∀ lane, ∃ index,
    interface.challenge offset lane = Expr.var index - 2
  priorMulCount : ∀ block lane cell,
    R1CS.mulCount (interface.prior offset block lane cell) = 0
  value : ∀ block lane cell, ∃ index,
    interface.value offset block lane cell = Expr.var index

private theorem challengeMulCount
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (inputs : ProductionInputs interface offset) (lane : Fin ringDegree) :
    R1CS.mulCount (interface.challenge offset lane) = 1 := by
  rcases inputs.challenge lane with ⟨index, equality⟩
  rw [equality]
  rfl

private theorem valueMulCount
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (inputs : ProductionInputs interface offset) (block : Fin blockCount)
    (lane : Fin ringDegree) (cell : Fin cellCount) :
    R1CS.mulCount (interface.value offset block lane cell) = 0 := by
  rcases inputs.value block lane cell with ⟨index, equality⟩
  rw [equality]
  rfl

def rawTermCount : List Nat → Nat → Nat
  | [], _ => 0
  | source :: rest, degree =>
      (if source ≤ degree ∧ degree - source < ringDegree then 1 else 0) +
        rawTermCount rest degree

def termCount (degree : Nat) : Nat :=
  rawTermCount (List.range ringDegree) degree

private theorem foldlMulCount
    (indices : List Nat) (degree : Nat)
    (challenge value : Fin ringDegree → Expr) (initial : Expr)
    (indicesBound : ∀ source ∈ indices, source < ringDegree)
    (challengeCount : ∀ lane, R1CS.mulCount (challenge lane) = 1)
    (valueCount : ∀ lane, R1CS.mulCount (value lane) = 0) :
    R1CS.mulCount
        (indices.foldl (fun accumulated source =>
          if source ≤ degree ∧ degree - source < ringDegree then
            accumulated +
              NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.exprCoeff
                challenge source *
              NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.exprCoeff
                value (degree - source)
          else accumulated) initial) =
      R1CS.mulCount initial + 2 * rawTermCount indices degree := by
  induction indices generalizing initial with
  | nil => simp [rawTermCount]
  | cons source rest inductionHypothesis =>
      have sourceBound : source < ringDegree :=
        indicesBound source (by simp)
      have restBound : ∀ current ∈ rest, current < ringDegree := by
        intro current member
        exact indicesBound current (by simp [member])
      simp only [List.foldl_cons]
      by_cases included : source ≤ degree ∧
          degree - source < ringDegree
      · rw [if_pos included, inductionHypothesis _ restBound]
        simp [rawTermCount, included,
          NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.exprCoeff,
          sourceBound, challengeCount, valueCount, R1CS.mulCount]
        omega
      · rw [if_neg included, inductionHypothesis _ restBound]
        simp [rawTermCount, included]

private theorem rawExprMulCount
    (challenge value : Fin ringDegree → Expr) (degree : Nat)
    (challengeCount : ∀ lane, R1CS.mulCount (challenge lane) = 1)
    (valueCount : ∀ lane, R1CS.mulCount (value lane) = 0) :
    R1CS.mulCount
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.rawExpr
          challenge value degree) =
      2 * termCount degree := by
  change R1CS.mulCount
      ((List.range ringDegree).foldl (fun accumulated source =>
        if source ≤ degree ∧ degree - source < ringDegree then
          accumulated +
            NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.exprCoeff
              challenge source *
            NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.exprCoeff
              value (degree - source)
        else accumulated) 0) =
    2 * rawTermCount (List.range ringDegree) degree
  simpa only [R1CS.mulCount, Nat.zero_add] using
    (foldlMulCount (List.range ringDegree) degree challenge value 0
      (by simp) challengeCount valueCount)

def foldedDegree (lane : Fin ringDegree) : Nat :=
  if lane.val < ringMiddleDegree then lane.val + ringDegree
  else lane.val + ringMiddleDegree

def twiceTermCount (lane : Fin ringDegree) : Nat :=
  if lane.val + 81 ≤ 106 then termCount (lane.val + 81) else 0

def laneMulCount (lane : Fin ringDegree) : Nat :=
  2 * (termCount lane.val + termCount (foldedDegree lane) +
    twiceTermCount lane) + 1

def laneFreshCount (lane : Fin ringDegree) : Nat := laneMulCount lane + 1

@[simp] private theorem mulCountSub (left right : Expr) :
    R1CS.mulCount (left - right) =
      R1CS.mulCount left + R1CS.mulCount right + 1 := by
  change R1CS.mulCount
    (.add left (.mul (.const (-1)) right)) =
      R1CS.mulCount left + R1CS.mulCount right + 1
  simp only [R1CS.mulCount]
  omega

private theorem mulExprMulCount
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (inputs : ProductionInputs interface offset) (block : Fin blockCount)
    (lane : Fin ringDegree) (cell : Fin cellCount) :
    R1CS.mulCount
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.mulExpr
          (interface.challenge offset)
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.ringExpr
            interface.value offset block cell) lane) =
      laneMulCount lane := by
  have challengeCount := challengeMulCount interface offset inputs
  have valueCount : ∀ current,
      R1CS.mulCount
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.ringExpr
          interface.value offset block cell current) = 0 := by
    intro current
    exact valueMulCount interface offset inputs block current cell
  unfold NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.mulExpr
    laneMulCount foldedDegree twiceTermCount
  split <;> split <;>
    simp [R1CS.mulCount, mulCountSub,
      rawExprMulCount _ _ _ challengeCount valueCount]
  all_goals omega

private theorem challengeFunctionExists
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (inputs : ProductionInputs interface offset) :
    ∃ indices : Fin ringDegree → Nat,
      interface.challenge offset =
        fun lane => Expr.var (indices lane) - 2 := by
  classical
  let indices := fun lane => Classical.choose (inputs.challenge lane)
  refine ⟨indices, ?_⟩
  funext lane
  exact Classical.choose_spec (inputs.challenge lane)

private theorem valueFunctionExists
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (inputs : ProductionInputs interface offset) (block : Fin blockCount)
    (cell : Fin cellCount) :
    ∃ indices : Fin ringDegree → Nat,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.ringExpr
          interface.value offset block cell =
        fun lane => Expr.var (indices lane) := by
  classical
  let indices := fun lane => Classical.choose (inputs.value block lane cell)
  refine ⟨indices, ?_⟩
  funext lane
  exact Classical.choose_spec (inputs.value block lane cell)

private theorem lowerAffineMulExprEqNone
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (inputs : ProductionInputs interface offset) (block : Fin blockCount)
    (lane : Fin ringDegree) (cell : Fin cellCount) :
    R1CS.lowerAffine
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.mulExpr
        (interface.challenge offset)
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.ringExpr
          interface.value offset block cell) lane) = none := by
  rcases challengeFunctionExists interface offset inputs with
    ⟨challengeIndices, challengeEq⟩
  rcases valueFunctionExists interface offset inputs block cell with
    ⟨valueIndices, valueEq⟩
  rw [challengeEq, valueEq]
  fin_cases lane <;> rfl

private theorem directConstraintRecipeEqNone
    (output : Nat) (prior : Expr)
    (challenge value : Fin ringDegree → Expr) (lane : Fin ringDegree)
    (notAffine : R1CS.lowerAffine
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.mulExpr
        challenge value lane) = none) :
    R1CS.directConstraint
      (Expr.var output - (prior +
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.mulExpr
          challenge value lane)) = none := by
  change R1CS.directConstraint
    (.add (.var output)
      (.mul (.const (-1)) (.add prior
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.mulExpr
          challenge value lane)))) = none
  cases priorAffine : R1CS.lowerAffine prior <;>
    simp [R1CS.directConstraint, R1CS.directRecipeRow,
      R1CS.affineConstraint, R1CS.lowerAffine, priorAffine, notAffine]

theorem constraintFreshCountEqLane
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (inputs : ProductionInputs interface offset)
    (index : Fin (Logical.privateCount blockCount cellCount)) :
    R1CS.constraintFreshCount
      (Expr.var (offset + index.val) -
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.recipe
          interface offset index) =
        laneFreshCount
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.laneOf
            index) := by
  let block :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.blockOf index
  let lane :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.laneOf index
  let cell :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.cellOf index
  rw [show
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.recipe
        interface offset index =
      interface.prior offset block lane cell +
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.mulExpr
          (interface.challenge offset)
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.ringExpr
            interface.value offset block cell) lane by rfl]
  unfold R1CS.constraintFreshCount
  rw [directConstraintRecipeEqNone _ _ _ _ _
    (lowerAffineMulExprEqNone interface offset inputs block lane cell)]
  change R1CS.mulCount
    (.add (.var (offset + index.val))
      (.mul (.const (-1))
        (.add (interface.prior offset block lane cell)
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.mulExpr
            (interface.challenge offset)
            (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.ringExpr
              interface.value offset block cell) lane)))) =
      laneFreshCount lane
  simp only [R1CS.mulCount]
  rw [inputs.priorMulCount block lane cell,
    mulExprMulCount interface offset inputs block lane cell]
  simp [laneFreshCount]

theorem recipeConstraintsOfFn {count : Nat} (start : Nat)
    (recipes : Fin count → Expr) :
    recipeConstraints start (List.ofFn recipes) =
      List.ofFn fun index => Expr.var (start + index.val) - recipes index := by
  induction count generalizing start with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [List.ofFn_succ, recipeConstraints]
      apply congrArg₂ List.cons
      · simp
      · rw [inductionHypothesis]
        apply congrArg List.ofFn
        funext index
        congr 2
        simp only [Fin.val_succ]
        omega

private theorem totalFreshCountOfFn {count : Nat}
    (constraints : Fin count → Expr) (cost : Fin count → Nat)
    (pointwise : ∀ index,
      R1CS.constraintFreshCount (constraints index) = cost index) :
    R1CS.totalFreshCount (List.ofFn constraints) =
      (List.ofFn cost).sum := by
  unfold R1CS.totalFreshCount
  rw [List.map_ofFn]
  apply congrArg List.sum
  apply congrArg List.ofFn
  funext index
  exact pointwise index

private theorem laneFreshCountSum :
    (List.ofFn laneFreshCount).sum = 8100 := by
  rfl

private theorem listSumOfFn {count : Nat} (values : Fin count → Nat) :
    (List.ofFn values).sum = ∑ index, values index := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [List.ofFn_succ, Fin.sum_univ_succ, inductionHypothesis]

private theorem indexedLaneFreshCountSum
    (blockCount cellCount : Nat) [NeZero cellCount] :
    (List.ofFn fun index : Fin (Logical.privateCount blockCount cellCount) =>
      laneFreshCount
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.laneOf
          index)).sum = blockCount * cellCount * 8100 := by
  rw [listSumOfFn]
  let outerEquiv :
      Fin (Logical.privateCount blockCount cellCount) ≃
        Fin blockCount × Fin (ringDegree * cellCount) := by
    unfold Logical.privateCount
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.privateCount
    exact finProdFinEquiv.symm
  let innerEquiv :
      Fin (ringDegree * cellCount) ≃ Fin ringDegree × Fin cellCount :=
    finProdFinEquiv.symm
  calc
    ∑ index : Fin (Logical.privateCount blockCount cellCount),
        laneFreshCount
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.laneOf
            index) =
      ∑ pair : Fin blockCount × Fin (ringDegree * cellCount),
        laneFreshCount (innerEquiv pair.2).1 := by
          apply Fintype.sum_equiv outerEquiv
          intro index
          rfl
    _ = ∑ block : Fin blockCount,
        ∑ inner : Fin (ringDegree * cellCount),
          laneFreshCount (innerEquiv inner).1 := by
            rw [Fintype.sum_prod_type]
    _ = ∑ block : Fin blockCount,
        ∑ pair : Fin ringDegree × Fin cellCount,
          laneFreshCount pair.1 := by
            apply Finset.sum_congr rfl
            intro block _
            apply Fintype.sum_equiv innerEquiv
            intro inner
            rfl
    _ = ∑ block : Fin blockCount,
        ∑ lane : Fin ringDegree,
          ∑ cell : Fin cellCount, laneFreshCount lane := by
            simp_rw [Fintype.sum_prod_type]
    _ = ∑ block : Fin blockCount,
        cellCount * (∑ lane : Fin ringDegree, laneFreshCount lane) := by
          apply Finset.sum_congr rfl
          intro block _
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro lane _
          simp
    _ = blockCount * cellCount * 8100 := by
      have laneSum : ∑ lane : Fin ringDegree, laneFreshCount lane = 8100 := by
        rw [← listSumOfFn]
        exact laneFreshCountSum
      rw [laneSum]
      simp [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]

theorem physicalFreshColumnCountEqProduction
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (inputs : ProductionInputs interface offset) :
    physicalFreshColumnCount interface offset =
      Logical.privateCount blockCount cellCount * 150 := by
  unfold physicalFreshColumnCount logicalConstraints
  rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.flatConstraints_operations]
  change R1CS.totalFreshCount
    (recipeConstraints offset (List.ofFn
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.recipe
        interface offset))) = _
  rw [recipeConstraintsOfFn]
  rw [totalFreshCountOfFn _ (fun index => laneFreshCount
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.laneOf index))
    (constraintFreshCountEqLane interface offset inputs)]
  rw [indexedLaneFreshCountSum]
  unfold Logical.privateCount
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.privateCount
    ringDegree
  rw [show 8100 = 54 * 150 by rfl]
  simp [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]

end NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep
