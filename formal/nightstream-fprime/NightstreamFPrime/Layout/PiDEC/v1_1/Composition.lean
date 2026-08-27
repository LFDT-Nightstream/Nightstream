import NightstreamFPrime.Layout.PiDEC.v1_1.Leaves.InputBinding
import NightstreamFPrime.Layout.PiDEC.v1_1.Leaves.OutputBinding
import NightstreamFPrime.Layout.PiDEC.v1_1.PublicInputSplit
import NightstreamFPrime.Layout.PiDEC.v1_1.CommitmentRecomposition
import NightstreamFPrime.Layout.PiDEC.v1_1.EvalKRecomposition
import NightstreamFPrime.Layout.PiDEC.v1_1.EvalARecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.Completeness

/-!
Owns physical composition and the exact six-child footprint ledger for the
PiDEC v1_1 phase.

The parent order is operational input binding, public split, commitment,
separate Pad `Eval_K`, separate 14-matrix `Eval_A`, and output binding. The
parent and both boundary views add no row.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiDEC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def logicalConstraints
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops (Formal.main relation interface) offset)

def childConstraints (child : FormalCircuit) (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops child.main offset)

def childConstraintLists
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List (List Expr) :=
  let shared := Formal.atOffset interface offset
  [childConstraints (Formal.inputBindingCircuit relation shared)
      (Formal.inputBindingOffset offset),
   childConstraints (Formal.publicInputCircuit shared)
      (Formal.publicInputOffset offset),
   childConstraints (Formal.commitmentCircuit shared)
      (Formal.commitmentOffset offset),
   childConstraints (Formal.evalKCircuit shared)
      (Formal.evalKOffset offset),
   childConstraints (Formal.evalACircuit shared)
      (Formal.evalAOffset offset),
   childConstraints (Formal.outputBindingCircuit relation shared)
      (Formal.outputBindingOffset offset)]

def orderedConstraints
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Expr :=
  (childConstraintLists relation interface offset).flatten

private theorem childOp_flatConstraints (name : String)
    (child : FormalCircuit) (offset : Nat) :
    (Formal.childOp name child offset).flatConstraints =
      childConstraints child offset := by
  rfl

/-- Exact equality between phase rows and the six opaque child lists. -/
theorem logicalConstraints_eq_ordered
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) :
    logicalConstraints relation interface offset =
      orderedConstraints relation interface offset := by
  unfold logicalConstraints
  rw [Formal.main_ops]
  unfold Formal.opsAt orderedConstraints childConstraintLists
  simp only [flatConstraints, List.flatMap_cons, List.flatMap_nil,
    List.flatten_cons, List.flatten_nil, childOp_flatConstraints,
    List.append_nil]

/-- The executable nonempty PiDEC child sequence. Input and output binding are
typed zero-row views and therefore do not occur in this list. -/
def nonBoundaryConstraints
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Expr :=
  let shared := Formal.atOffset interface offset
  childConstraints (Formal.publicInputCircuit shared)
      (Formal.publicInputOffset offset) ++
    childConstraints (Formal.commitmentCircuit shared)
      (Formal.commitmentOffset offset) ++
    childConstraints (Formal.evalKCircuit shared)
      (Formal.evalKOffset offset) ++
    childConstraints (Formal.evalACircuit shared)
      (Formal.evalAOffset offset)

/-- Removing the two typed zero-row boundary children changes no logical
constraint and keeps the four nonempty children in exact parent order. -/
theorem logicalConstraints_eq_nonBoundary
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) :
    logicalConstraints relation interface offset =
      nonBoundaryConstraints interface offset := by
  rw [logicalConstraints_eq_ordered]
  unfold orderedConstraints childConstraintLists nonBoundaryConstraints
  simp only [List.flatten_cons, List.flatten_nil, List.append_nil]
  have inputNil :
      childConstraints
        (Formal.inputBindingCircuit relation (Formal.atOffset interface offset))
        (Formal.inputBindingOffset offset) = [] := by
    rfl
  have outputNil :
      childConstraints
        (Formal.outputBindingCircuit relation (Formal.atOffset interface offset))
        (Formal.outputBindingOffset offset) = [] := by
    rfl
  rw [inputNil, outputNil]
  simp only [List.nil_append, List.append_nil, List.append_assoc]

/-- Expression-shape evidence required by the fixed production footprint. -/
structure InputShapes
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (parentOffset : Nat) : Prop where
  publicInput : ∀ childOffset,
    PublicInputSplit.InputsLinear
      (Formal.publicInputInterface (Formal.atOffset interface parentOffset))
      childOffset
  commitment : ∀ childOffset,
    CommitmentRecomposition.InputsLinear
      (Formal.commitmentInterface (Formal.atOffset interface parentOffset))
      childOffset
  eval_K : ∀ childOffset,
    EvalKRecomposition.InputsLinear
      (Formal.evalKInterface (Formal.atOffset interface parentOffset))
      childOffset
  eval_A : ∀ childOffset,
    EvalARecomposition.InputsLinear
      (Formal.evalAInterface (Formal.atOffset interface parentOffset))
      childOffset

def physicalFreshDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Nat :=
  (childConstraintLists relation interface offset).map R1CS.totalFreshCount

def physicalRowDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Nat :=
  (childConstraintLists relation interface offset).map R1CS.totalRowCount

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

theorem totalFreshCount_eq_deltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) :
    R1CS.totalFreshCount (logicalConstraints relation interface offset) =
      (physicalFreshDeltas relation interface offset).sum := by
  rw [logicalConstraints_eq_ordered]
  unfold orderedConstraints physicalFreshDeltas
  exact totalFreshCount_flatten _

theorem totalRowCount_eq_deltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) :
    R1CS.totalRowCount (logicalConstraints relation interface offset) =
      (physicalRowDeltas relation interface offset).sum := by
  rw [logicalConstraints_eq_ordered]
  unfold orderedConstraints physicalRowDeltas
  exact totalRowCount_flatten _

private theorem inputFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    R1CS.totalFreshCount
      (childConstraints
        (Formal.inputBindingCircuit relation (Formal.atOffset interface offset))
        (Formal.inputBindingOffset offset)) = 0 :=
  Leaves.InputBinding.freshColumnCount_eq relation
    (Formal.inputBindingInterface (Formal.atOffset interface offset))
    (Formal.inputBindingOffset offset)

private theorem inputRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    R1CS.totalRowCount
      (childConstraints
        (Formal.inputBindingCircuit relation (Formal.atOffset interface offset))
        (Formal.inputBindingOffset offset)) = 0 :=
  Leaves.InputBinding.physicalRowCount_eq relation
    (Formal.inputBindingInterface (Formal.atOffset interface offset))
    (Formal.inputBindingOffset offset)

private theorem publicFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalFreshCount
      (childConstraints (Formal.publicInputCircuit (Formal.atOffset interface offset))
        (Formal.publicInputOffset offset)) = 3564 :=
  PublicInputSplit.totalFreshCount_eq
    (Formal.publicInputInterface (Formal.atOffset interface offset))
    (Formal.publicInputOffset offset)
    (inputs.publicInput (Formal.publicInputOffset offset))

private theorem publicRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalRowCount
      (childConstraints (Formal.publicInputCircuit (Formal.atOffset interface offset))
        (Formal.publicInputOffset offset)) = 4536 :=
  PublicInputSplit.totalRowCount_eq
    (Formal.publicInputInterface (Formal.atOffset interface offset))
    (Formal.publicInputOffset offset)
    (inputs.publicInput (Formal.publicInputOffset offset))

private theorem commitmentFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalFreshCount
      (childConstraints (Formal.commitmentCircuit (Formal.atOffset interface offset))
        (Formal.commitmentOffset offset)) = 0 :=
  CommitmentRecomposition.freshColumnCount_eq
    (Formal.commitmentInterface (Formal.atOffset interface offset))
    inputs.commitment (Formal.commitmentOffset offset)

private theorem commitmentRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalRowCount
      (childConstraints (Formal.commitmentCircuit (Formal.atOffset interface offset))
        (Formal.commitmentOffset offset)) = 972 :=
  CommitmentRecomposition.physicalRowCount_eq
    (Formal.commitmentInterface (Formal.atOffset interface offset))
    inputs.commitment (Formal.commitmentOffset offset)

private theorem evalKFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalFreshCount
      (childConstraints (Formal.evalKCircuit (Formal.atOffset interface offset))
        (Formal.evalKOffset offset)) = 0 :=
  EvalKRecomposition.freshColumnCount_eq
    (Formal.evalKInterface (Formal.atOffset interface offset))
    inputs.eval_K (Formal.evalKOffset offset)

private theorem evalKRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalRowCount
      (childConstraints (Formal.evalKCircuit (Formal.atOffset interface offset))
        (Formal.evalKOffset offset)) = 108 :=
  EvalKRecomposition.physicalRowCount_eq
    (Formal.evalKInterface (Formal.atOffset interface offset))
    inputs.eval_K (Formal.evalKOffset offset)

private theorem evalAFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalFreshCount
      (childConstraints (Formal.evalACircuit (Formal.atOffset interface offset))
        (Formal.evalAOffset offset)) = 0 :=
  EvalARecomposition.freshColumnCount_eq
    (Formal.evalAInterface (Formal.atOffset interface offset))
    inputs.eval_A (Formal.evalAOffset offset)

private theorem evalARows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalRowCount
      (childConstraints (Formal.evalACircuit (Formal.atOffset interface offset))
        (Formal.evalAOffset offset)) = 1512 :=
  EvalARecomposition.physicalRowCount_eq
    (Formal.evalAInterface (Formal.atOffset interface offset))
    inputs.eval_A (Formal.evalAOffset offset)

private theorem outputFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    R1CS.totalFreshCount
      (childConstraints
        (Formal.outputBindingCircuit relation (Formal.atOffset interface offset))
        (Formal.outputBindingOffset offset)) = 0 :=
  Leaves.OutputBinding.freshColumnCount_eq relation
    (Formal.outputBindingInterface (Formal.atOffset interface offset))
    (Formal.outputBindingOffset offset)

private theorem outputRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    R1CS.totalRowCount
      (childConstraints
        (Formal.outputBindingCircuit relation (Formal.atOffset interface offset))
        (Formal.outputBindingOffset offset)) = 0 :=
  Leaves.OutputBinding.physicalRowCount_eq relation
    (Formal.outputBindingInterface (Formal.atOffset interface offset))
    (Formal.outputBindingOffset offset)

theorem physicalFreshDeltas_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    physicalFreshDeltas relation interface offset =
      [0, 3564, 0, 0, 0, 0] := by
  unfold physicalFreshDeltas childConstraintLists
  simp only [List.map_cons, List.map_nil]
  rw [inputFresh_eq, publicFresh_eq relation interface offset inputs,
    commitmentFresh_eq relation interface offset inputs,
    evalKFresh_eq relation interface offset inputs,
    evalAFresh_eq relation interface offset inputs, outputFresh_eq]

theorem physicalRowDeltas_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    physicalRowDeltas relation interface offset =
      [0, 4536, 972, 108, 1512, 0] := by
  unfold physicalRowDeltas childConstraintLists
  simp only [List.map_cons, List.map_nil]
  rw [inputRows_eq, publicRows_eq relation interface offset inputs,
    commitmentRows_eq relation interface offset inputs,
    evalKRows_eq relation interface offset inputs,
    evalARows_eq relation interface offset inputs, outputRows_eq]

def logicalPrivateDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Nat :=
  (Formal.opsAt relation interface offset).map Op.localLength

theorem logicalPrivateDeltas_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    logicalPrivateDeltas relation interface offset =
      [0, 54, 0, 0, 0, 0] := by
  unfold logicalPrivateDeltas Formal.opsAt Formal.childOp Sequence.childOp
  simp only [List.map_cons, List.map_nil, Op.localLength,
    FormalCircuit.asSubcircuit_localLength]
  unfold Formal.inputBindingCircuit Formal.publicInputCircuit
    Formal.commitmentCircuit Formal.evalKCircuit Formal.evalACircuit
    Formal.outputBindingCircuit
  unfold NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.circuit
  rw [NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.localLength_eq,
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.localLength_eq,
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition.localLength_eq,
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition.localLength_eq,
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalARecomposition.localLength_eq,
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputBinding.localLength_eq,
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.logicalPrivateCount_eq]

def physicalColumnDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Nat :=
  List.zipWith (· + ·) (logicalPrivateDeltas relation interface offset)
    (physicalFreshDeltas relation interface offset)

theorem physicalColumnDeltas_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    physicalColumnDeltas relation interface offset =
      [0, 3618, 0, 0, 0, 0] := by
  unfold physicalColumnDeltas
  rw [logicalPrivateDeltas_eq,
    physicalFreshDeltas_eq relation interface offset inputs]
  rfl

def cumulativeFrom : Nat → List Nat → List Nat
  | _, [] => []
  | total, delta :: rest =>
      let next := total + delta
      next :: cumulativeFrom next rest

def cumulativePhysicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Nat :=
  cumulativeFrom 0 (physicalRowDeltas relation interface offset)

def cumulativePhysicalColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Nat :=
  cumulativeFrom 0 (physicalColumnDeltas relation interface offset)

def cumulativeJointDomains
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Nat :=
  List.zipWith max (cumulativePhysicalRows relation interface offset)
    (cumulativePhysicalColumns relation interface offset)

theorem cumulativeFootprints_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    cumulativePhysicalRows relation interface offset =
        [0, 4536, 5508, 5616, 7128, 7128] ∧
      cumulativePhysicalColumns relation interface offset =
        [0, 3618, 3618, 3618, 3618, 3618] ∧
      cumulativeJointDomains relation interface offset =
        [0, 4536, 5508, 5616, 7128, 7128] := by
  rw [cumulativePhysicalRows,
    physicalRowDeltas_eq relation interface offset inputs,
    cumulativePhysicalColumns,
    physicalColumnDeltas_eq relation interface offset inputs]
  norm_num [cumulativeFrom, cumulativeJointDomains, cumulativePhysicalRows,
    cumulativePhysicalColumns,
    physicalRowDeltas_eq relation interface offset inputs,
    physicalColumnDeltas_eq relation interface offset inputs]

theorem totalFreshCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalFreshCount (logicalConstraints relation interface offset) =
      3564 := by
  rw [totalFreshCount_eq_deltas,
    physicalFreshDeltas_eq relation interface offset inputs]
  rfl

theorem totalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalRowCount (logicalConstraints relation interface offset) =
      7128 := by
  rw [totalRowCount_eq_deltas,
    physicalRowDeltas_eq relation interface offset inputs]
  rfl

theorem physicalPrivateColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    localLength (Circuit.ops (Formal.main relation interface) offset) +
      R1CS.totalFreshCount (logicalConstraints relation interface offset) =
      3618 := by
  rw [Formal.localLength_eq, totalFreshCount_eq relation interface offset inputs]
  rfl

def footprint
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (inputs : ∀ offset, InputShapes relation interface offset) :
    R1CS.CircuitFootprint (Formal.circuit relation ajtai interface) where
  freshColumnCount := fun _ => 3564
  physicalRowCount := fun _ => 7128
  freshColumnCount_eq := fun offset =>
    totalFreshCount_eq relation interface offset (inputs offset)
  physicalRowCount_eq := fun offset =>
    totalRowCount_eq relation interface offset (inputs offset)

end NightstreamFPrime.Layout.PiDEC.v1_1
