import NightstreamFPrime.Layout.R1CS.Completeness
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep

/-!
Owns physical R1CS lowering for one generic PiRLC combination step.

The logical step keeps its exact `next = prior + rho * value` rows. Because
the external expressions are generic, this owner exposes the exact computed
physical footprint rather than claiming a constant production cost.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

namespace Logical

abbrev Interface (blockCount cellCount : Nat) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.Interface
    blockCount cellCount
abbrev privateCount :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.privateCount
abbrev operations {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.operations
    interface offset
abbrev Assumptions {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.Assumptions
    interface offset env
abbrev SpecHolds {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.SpecHolds
    interface offset env
abbrev soundness {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.soundness interface
abbrev completeness {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.completeness interface
abbrev localLength_eq {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.localLength_eq interface
abbrev flatConstraints_varsBelow {blockCount cellCount : Nat}
    [NeZero cellCount] (interface : Interface blockCount cellCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.flatConstraints_varsBelow
    interface
end Logical

def logicalConstraints {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) :
    List Expr :=
  flatConstraints (Logical.operations interface offset)

def physicalFreshColumnCount {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) : Nat :=
  R1CS.totalFreshCount (logicalConstraints interface offset)

def physicalRowCount {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) : Nat :=
  R1CS.totalRowCount (logicalConstraints interface offset)

def physicalPrivateColumnCount {blockCount cellCount : Nat}
    [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) : Nat :=
  Logical.privateCount blockCount cellCount +
    physicalFreshColumnCount interface offset

def plan {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) :
    R1CS.LoweringPlan where
  constraints := logicalConstraints interface offset
  firstFresh := offset + Logical.privateCount blockCount cellCount

def PhysicalHolds {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (env : Env) : Prop :=
  R1CS.RowsHold env (plan interface offset).rows

theorem physicalRowCount_eq {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat) :
    (plan interface offset).rowCount =
      physicalRowCount interface offset := by
  exact R1CS.LoweringPlan.rowCount_eq _

/-- Physical rows imply the exact generic step equation. -/
theorem physical_implies_spec {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (env : Env) (assumptions : Logical.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) :
    Logical.SpecHolds interface offset env := by
  apply Logical.soundness interface env offset assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold env (logicalConstraints interface offset)
  exact R1CS.LoweringPlan.sound (plan interface offset) env physical

/-- Constructive completion over the exact logical and computed R1CS-fresh
intervals. -/
theorem physical_complete {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Logical.Interface blockCount cellCount) (offset : Nat)
    (env : Env) (assumptions : Logical.Assumptions interface offset env)
    (specification : Logical.SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (physicalPrivateColumnCount interface offset) ∧
      PhysicalHolds interface offset completed := by
  rcases Logical.completeness interface env offset assumptions specification with
    ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have logicalAgreesFixed : AgreesOutside env logicalEnv offset
      (Logical.privateCount blockCount cellCount) := by
    rw [Logical.localLength_eq] at logicalAgrees
    exact logicalAgrees
  have logicalAssumptions : Logical.Assumptions interface offset logicalEnv :=
    ⟨assumptions.challengeBelow, assumptions.priorBelow,
      assumptions.valueBelow⟩
  have scope : ∀ expression ∈ logicalConstraints interface offset,
      expression.VarsBelow
        (offset + Logical.privateCount blockCount cellCount) := by
    exact Logical.flatConstraints_varsBelow interface offset logicalEnv
      logicalAssumptions
  rcases R1CS.lowerConstraints_complete logicalEnv
      (logicalConstraints interface offset)
      (offset + Logical.privateCount blockCount cellCount) scope logicalRows with
    ⟨completed, physicalAgrees, rows⟩
  refine ⟨completed, ?_, rows⟩
  exact logicalAgreesFixed.append physicalAgrees

end NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep
