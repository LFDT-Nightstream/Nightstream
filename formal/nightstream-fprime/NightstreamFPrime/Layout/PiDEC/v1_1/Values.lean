import NightstreamFPrime.Layout.PiDEC.v1_1.Lowering
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.Values

/-! Production values for the derived PiDEC physical footprints. Core
composition and lowering use owner counts and do not import this module. -/

namespace NightstreamFPrime.Layout.PiDEC.v1_1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiDEC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem Leaves.SignedSplitScalar.freshColumnCount_value :
    Leaves.SignedSplitScalar.freshColumnCount = 66 := by rfl

theorem Leaves.SignedSplitScalar.physicalRowCount_value :
    Leaves.SignedSplitScalar.physicalRowCount = 84 := by rfl

theorem Leaves.SignedSplitScalar.physicalPrivateColumnCount_value :
    Lifecycle.PiDEC.v1_1.SignedSplitScalar.exactPrivateCount +
      Leaves.SignedSplitScalar.freshColumnCount = 67 := by rfl

theorem PublicInputSplit.freshColumnCount_value :
    PublicInputSplit.freshColumnCount = 17820 := by rfl

theorem PublicInputSplit.physicalRowCount_value :
    PublicInputSplit.physicalRowCount = 22680 := by rfl

theorem CommitmentRecomposition.physicalRowCount_value :
    CommitmentRecomposition.physicalRowCount = 1188 := by rfl

theorem EvalKRecomposition.physicalRowCount_value :
    EvalKRecomposition.physicalRowCount = 108 := by rfl

theorem EvalARecomposition.physicalRowCount_value :
    EvalARecomposition.physicalRowCount = 1512 := by rfl

theorem exactFreshDeltas_value :
    exactFreshDeltas = [0, 17820, 0, 0, 0, 0] := by rfl

theorem exactRowDeltas_value :
    exactRowDeltas = [0, 22680, 1188, 108, 1512, 0] := by rfl

theorem exactPhysicalColumnDeltas_value :
    exactPhysicalColumnDeltas = [0, 18090, 0, 0, 0, 0] := by rfl

theorem exactFreshCount_value : exactFreshCount = 17820 := by rfl

theorem exactRowCount_value : exactRowCount = 25488 := by rfl

theorem exactPrivateCount_value :
    Formal.logicalPrivateCount + exactFreshCount = 18090 := by
  rw [exactFreshCount_value]
  change Lifecycle.PiDEC.v1_1.PublicInputSplit.exactPrivateCount + 17820 = 18090
  rw [Lifecycle.PiDEC.v1_1.PublicInputSplit.exactPrivateCount_eq]

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

theorem cumulativeFootprints_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    cumulativePhysicalRows relation interface offset =
        [0, 22680, 23868, 23976, 25488, 25488] ∧
      cumulativePhysicalColumns relation interface offset =
        [0, 18090, 18090, 18090, 18090, 18090] ∧
      cumulativeJointDomains relation interface offset =
        [0, 22680, 23868, 23976, 25488, 25488] := by
  norm_num [cumulativePhysicalRows,
    physicalRowDeltas_eq relation interface offset inputs, exactRowDeltas_value,
    cumulativePhysicalColumns,
    physicalColumnDeltas_eq relation interface offset inputs,
    exactPhysicalColumnDeltas_value, cumulativeFrom, cumulativeJointDomains,
    List.zipWith_cons_cons, List.zipWith_nil_left]

end NightstreamFPrime.Layout.PiDEC.v1_1
