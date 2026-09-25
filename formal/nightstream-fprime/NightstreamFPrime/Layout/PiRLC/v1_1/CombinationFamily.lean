import NightstreamFPrime.Layout.PiRLC.v1_1.CombinationCost
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily

/-!
Owns physical composition for one exact 17-source PiRLC combination family.

The family preserves the logical `K + k` child order and sums the exact
computed step footprints. Concrete commitment, public-input, `Eval_K`, and
`Eval_A` owners prove fixed production totals over this generic layout.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

namespace Logical

abbrev Interface (blockCount cellCount : Nat) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.Interface
    blockCount cellCount
abbrev sourceCount :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.sourceCount
abbrev stepSize :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.stepSize
abbrev logicalPrivateCount :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.logicalPrivateCount
abbrev main {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.main interface
abbrev circuit {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.circuit interface
abbrev stepOffset :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.stepOffset
abbrev stepInterface {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset source : Nat) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.stepInterface
    interface offset source
abbrev stepOp {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset source : Nat) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.stepOp
    interface offset source
abbrev Assumptions {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.Assumptions
    interface offset env
abbrev CanonicalHolds {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.CanonicalHolds
    interface offset env
abbrev soundness {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.soundness interface
abbrev completeness {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.completeness interface
abbrev localLength_eq {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.localLength_eq
    interface
abbrev flatConstraints_varsBelow {blockCount cellCount : Nat}
    [NeZero cellCount] (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.flatConstraints_varsBelow
    interface
abbrev main_ops {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.main_ops interface
abbrev opsAt_eq {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.opsAt_eq interface

end Logical

def childConstraints {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount)
    (offset source : Nat) : List Expr :=
  CombinationStep.logicalConstraints
    (Logical.stepInterface interface offset source)
    (Logical.stepOffset offset source blockCount cellCount)

def childConstraintLists {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) :
    List (List Expr) :=
  (List.range Logical.sourceCount).map (childConstraints interface offset)

def orderedConstraints {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) :
    List Expr :=
  (childConstraintLists interface offset).flatten

def logicalConstraints {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) :
    List Expr :=
  flatConstraints (Circuit.ops (Logical.main interface) offset)

/-- Exact production expression shapes for all 17 sources in one family. -/
structure ProductionInputs {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) : Prop where
  challenge : ∀ source lane, ∃ index,
    interface.challenge offset source lane = Expr.var index - 2
  input : ∀ source block lane cell, ∃ index,
    interface.input offset source block lane cell = Expr.var index

theorem stepProductionInputs
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset source : Nat)
    (sourceLt : source < Logical.sourceCount)
    (inputs : ProductionInputs interface offset) :
    CombinationStep.ProductionInputs
      (Logical.stepInterface interface offset source)
      (Logical.stepOffset offset source blockCount cellCount) := by
  constructor
  · intro lane
    rcases inputs.challenge ⟨source, sourceLt⟩ lane with ⟨index, equality⟩
    refine ⟨index, ?_⟩
    simpa [Logical.stepInterface,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.stepInterface,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.challengeAt,
      sourceLt] using equality
  · intro block lane cell
    change R1CS.mulCount
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.priorAt
        offset source block lane cell) = 0
    unfold NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.priorAt
    split <;> rfl
  · intro block lane cell
    rcases inputs.input ⟨source, sourceLt⟩ block lane cell with
      ⟨index, equality⟩
    refine ⟨index, ?_⟩
    simpa [Logical.stepInterface,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.stepInterface,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.inputAt,
      sourceLt] using equality

private theorem childOp_flatConstraints {blockCount cellCount : Nat}
    [NeZero cellCount] (interface : Logical.Interface blockCount cellCount)
    (offset source : Nat) :
    (Logical.stepOp interface offset source).flatConstraints =
      childConstraints interface offset source := by
  rfl

private theorem flatConstraints_childOps {blockCount cellCount : Nat}
    [NeZero cellCount] (interface : Logical.Interface blockCount cellCount)
    (offset : Nat) (sources : List Nat) :
    flatConstraints (sources.map (Logical.stepOp interface offset)) =
      (sources.map (childConstraints interface offset)).flatten := by
  induction sources with
  | nil => rfl
  | cons source rest inductionHypothesis =>
      simp only [List.map_cons, flatConstraints, List.flatMap_cons,
        List.flatten_cons, childOp_flatConstraints]
      exact congrArg (fun tail =>
        childConstraints interface offset source ++ tail) inductionHypothesis

theorem logicalConstraints_eq_ordered {blockCount cellCount : Nat}
    [NeZero cellCount] (interface : Logical.Interface blockCount cellCount)
    (offset : Nat) :
    logicalConstraints interface offset =
      orderedConstraints interface offset := by
  unfold logicalConstraints
  rw [Logical.main_ops, Logical.opsAt_eq]
  unfold orderedConstraints childConstraintLists
  exact flatConstraints_childOps interface offset _

def physicalFreshDeltas {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) :
    List Nat :=
  (List.range Logical.sourceCount).map fun source =>
    CombinationStep.physicalFreshColumnCount
      (Logical.stepInterface interface offset source)
      (Logical.stepOffset offset source blockCount cellCount)

def physicalRowDeltas {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) :
    List Nat :=
  (List.range Logical.sourceCount).map fun source =>
    CombinationStep.physicalRowCount
      (Logical.stepInterface interface offset source)
      (Logical.stepOffset offset source blockCount cellCount)

private theorem totalFreshCount_flatten (lists : List (List Expr)) :
    R1CS.totalFreshCount lists.flatten =
      (lists.map R1CS.totalFreshCount).sum := by
  induction lists with
  | nil => rfl
  | cons first rest inductionHypothesis =>
      simp only [List.flatten_cons, R1CS.totalFreshCount_append,
        List.map_cons, List.sum_cons, inductionHypothesis]

private theorem totalRowCount_flatten (lists : List (List Expr)) :
    R1CS.totalRowCount lists.flatten =
      (lists.map R1CS.totalRowCount).sum := by
  induction lists with
  | nil => rfl
  | cons first rest inductionHypothesis =>
      simp only [List.flatten_cons, R1CS.totalRowCount_append,
        List.map_cons, List.sum_cons, inductionHypothesis]

theorem totalFreshCount_eq_deltas {blockCount cellCount : Nat}
    [NeZero cellCount] (interface : Logical.Interface blockCount cellCount)
    (offset : Nat) :
    R1CS.totalFreshCount (logicalConstraints interface offset) =
      (physicalFreshDeltas interface offset).sum := by
  rw [logicalConstraints_eq_ordered]
  unfold orderedConstraints
  rw [totalFreshCount_flatten]
  unfold childConstraintLists physicalFreshDeltas
  rfl

theorem totalRowCount_eq_deltas {blockCount cellCount : Nat}
    [NeZero cellCount] (interface : Logical.Interface blockCount cellCount)
    (offset : Nat) :
    R1CS.totalRowCount (logicalConstraints interface offset) =
      (physicalRowDeltas interface offset).sum := by
  rw [logicalConstraints_eq_ordered]
  unfold orderedConstraints
  rw [totalRowCount_flatten]
  unfold childConstraintLists physicalRowDeltas
  rfl

def physicalFreshColumnCount {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) : Nat :=
  (physicalFreshDeltas interface offset).sum

private theorem sumMapEqMulLength {Index : Type}
    (indices : List Index) (cost : Index → Nat) (expected : Nat)
    (each : ∀ index ∈ indices, cost index = expected) :
    (indices.map cost).sum = indices.length * expected := by
  induction indices with
  | nil => simp
  | cons index rest inductionHypothesis =>
      have head := each index (by simp)
      have tail : ∀ current ∈ rest, cost current = expected := by
        intro current member
        exact each current (by simp [member])
      simp only [List.map_cons, List.sum_cons, List.length_cons, head,
        inductionHypothesis tail]
      rw [Nat.add_one, Nat.succ_mul]
      omega

theorem physicalFreshColumnCountEqProduction
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (inputs : ProductionInputs interface offset) :
    physicalFreshColumnCount interface offset =
      Logical.logicalPrivateCount blockCount cellCount * 150 := by
  unfold physicalFreshColumnCount physicalFreshDeltas
  rw [sumMapEqMulLength _ _
    (CombinationStep.Logical.privateCount blockCount cellCount * 150)]
  · change Logical.sourceCount *
      (CombinationStep.Logical.privateCount blockCount cellCount * 150) =
        (Logical.sourceCount *
          CombinationStep.Logical.privateCount blockCount cellCount) * 150
    exact (Nat.mul_assoc _ _ _).symm
  · intro source member
    exact CombinationStep.physicalFreshColumnCountEqProduction
      (Logical.stepInterface interface offset source)
      (Logical.stepOffset offset source blockCount cellCount)
      (stepProductionInputs interface offset source
        (List.mem_range.mp member) inputs)

def physicalRowCount {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) : Nat :=
  (physicalRowDeltas interface offset).sum

theorem physicalRowCountEqFreshAddLogical
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) :
    physicalRowCount interface offset =
      physicalFreshColumnCount interface offset +
        Logical.logicalPrivateCount blockCount cellCount := by
  have lengthEq :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.flatConstraints_length
      interface offset
  change (logicalConstraints interface offset).length =
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.logicalRowCount
      blockCount cellCount at lengthEq
  calc
    physicalRowCount interface offset =
        R1CS.totalRowCount (logicalConstraints interface offset) :=
      (totalRowCount_eq_deltas interface offset).symm
    _ = R1CS.totalFreshCount (logicalConstraints interface offset) +
        (logicalConstraints interface offset).length :=
      R1CS.totalRowCount_eq_fresh_add_length _
    _ = physicalFreshColumnCount interface offset +
        Logical.logicalPrivateCount blockCount cellCount := by
      rw [totalFreshCount_eq_deltas, lengthEq]
      rfl

theorem physicalRowCountEqProduction
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (inputs : ProductionInputs interface offset) :
    physicalRowCount interface offset =
      Logical.logicalPrivateCount blockCount cellCount * 151 := by
  rw [physicalRowCountEqFreshAddLogical,
    physicalFreshColumnCountEqProduction interface offset inputs]
  omega

def physicalPrivateColumnCount {blockCount cellCount : Nat}
    [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) : Nat :=
  Logical.logicalPrivateCount blockCount cellCount +
    physicalFreshColumnCount interface offset

def plan {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) :
    R1CS.LoweringPlan where
  constraints := logicalConstraints interface offset
  firstFresh := offset + Logical.logicalPrivateCount blockCount cellCount

def PhysicalHolds {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (env : Env) : Prop :=
  R1CS.RowsHold env (plan interface offset).rows

theorem physicalRowCount_eq {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) :
    (plan interface offset).rowCount = physicalRowCount interface offset := by
  rw [R1CS.LoweringPlan.rowCount_eq]
  change R1CS.totalRowCount (logicalConstraints interface offset) = _
  rw [totalRowCount_eq_deltas]
  rfl

theorem physical_implies_canonical {blockCount cellCount : Nat}
    [NeZero cellCount] (interface : Logical.Interface blockCount cellCount)
    (offset : Nat) (env : Env)
    (assumptions : Logical.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) :
    Logical.CanonicalHolds interface offset env := by
  apply Logical.soundness interface offset env assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold env (logicalConstraints interface offset)
  exact R1CS.LoweringPlan.sound (plan interface offset) env physical

theorem physical_complete {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (env : Env) (assumptions : Logical.Assumptions interface offset env)
    (specification : Logical.CanonicalHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (physicalPrivateColumnCount interface offset) ∧
      PhysicalHolds interface offset completed := by
  rcases Logical.completeness interface offset env assumptions specification with
    ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have logicalAgreesFixed : AgreesOutside env logicalEnv offset
      (Logical.logicalPrivateCount blockCount cellCount) := by
    rw [Logical.localLength_eq] at logicalAgrees
    exact logicalAgrees
  have logicalAssumptions : Logical.Assumptions interface offset logicalEnv :=
    ⟨assumptions.challengeBelow, assumptions.inputBelow⟩
  have scope : ∀ expression ∈ logicalConstraints interface offset,
      expression.VarsBelow
        (offset + Logical.logicalPrivateCount blockCount cellCount) := by
    exact Logical.flatConstraints_varsBelow interface offset logicalEnv
      logicalAssumptions
  rcases R1CS.lowerConstraints_complete logicalEnv
      (logicalConstraints interface offset)
      (offset + Logical.logicalPrivateCount blockCount cellCount) scope
      logicalRows with
    ⟨completed, physicalAgrees, rows⟩
  refine ⟨completed, ?_, rows⟩
  have combined := logicalAgreesFixed.append physicalAgrees
  rw [totalFreshCount_eq_deltas] at combined
  exact combined

end NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily
