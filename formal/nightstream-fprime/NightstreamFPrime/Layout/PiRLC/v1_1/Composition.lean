import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.InputBinding
import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.OutputBinding
import NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain
import NightstreamFPrime.Layout.PiRLC.v1_1.CommitmentCombination
import NightstreamFPrime.Layout.PiRLC.v1_1.PublicInputCombination
import NightstreamFPrime.Layout.PiRLC.v1_1.EvalKCombination
import NightstreamFPrime.Layout.PiRLC.v1_1.EvalACombination
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Completeness

/-!
Owns physical composition and the exact seven-child footprint ledger for the
PiRLC v1_1 phase.

The parent order is input view, 17-sampler chain, commitment, public input,
separate `Eval_K`, separate `Eval_A`, and output view. The two views own zero
rows. The parent adds no copy, boundary, or assertion row.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
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
   childConstraints (Formal.samplerCircuit shared)
      (Formal.samplerOffset offset),
   childConstraints (Formal.commitmentCircuit shared)
      (Formal.commitmentOffset offset),
   childConstraints (Formal.publicInputCircuit shared)
      (Formal.publicInputOffset offset),
   childConstraints (Formal.evalKCircuit shared)
      (Formal.evalKOffset offset),
   childConstraints (Formal.evalACircuit shared)
      (Formal.evalAOffset offset),
   childConstraints (Formal.outputBindingCircuit relation shared offset)
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

/-- Exact equality between the phase rows and the seven opaque child lists. -/
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

/-- Exact expression-shape evidence required for the fixed production
footprint. The canonical package must construct this record. -/
structure InputShapes
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : Prop where
  sampler : SamplerChain.InputsAffine
    (Formal.samplerInterface (Formal.atOffset interface offset))
    (Formal.samplerOffset offset)
  commitmentFresh : CommitmentCombination.physicalFreshColumnCount
    (Formal.commitmentInterface (Formal.atOffset interface offset))
    (Formal.commitmentOffset offset) = 3029400
  commitmentRows : CommitmentCombination.physicalRowCount
    (Formal.commitmentInterface (Formal.atOffset interface offset))
    (Formal.commitmentOffset offset) = 3049596
  publicInputFresh : PublicInputCombination.physicalFreshColumnCount
    (Formal.publicInputInterface (Formal.atOffset interface offset))
    (Formal.publicInputOffset offset) = 688500
  publicInputRows : PublicInputCombination.physicalRowCount
    (Formal.publicInputInterface (Formal.atOffset interface offset))
    (Formal.publicInputOffset offset) = 693090
  evalKFresh : EvalKCombination.physicalFreshColumnCount
    (Formal.evalKInterface (Formal.atOffset interface offset))
    (Formal.evalKOffset offset) = 275400
  evalKRows : EvalKCombination.physicalRowCount
    (Formal.evalKInterface (Formal.atOffset interface offset))
    (Formal.evalKOffset offset) = 277236
  evalAFresh : EvalACombination.physicalFreshColumnCount
    (Formal.evalAInterface (Formal.atOffset interface offset))
    (Formal.evalAOffset offset) = 3855600
  evalARows : EvalACombination.physicalRowCount
    (Formal.evalAInterface (Formal.atOffset interface offset))
    (Formal.evalAOffset offset) = 3881304

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
        (Formal.inputBindingOffset offset)) = 0 := by
  exact Leaves.InputBinding.freshColumnCount_eq relation
    (Formal.inputBindingInterface (Formal.atOffset interface offset))
    (Formal.inputBindingOffset offset)

private theorem inputRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    R1CS.totalRowCount
      (childConstraints
        (Formal.inputBindingCircuit relation (Formal.atOffset interface offset))
        (Formal.inputBindingOffset offset)) = 0 := by
  exact Leaves.InputBinding.physicalRowCount_eq relation
    (Formal.inputBindingInterface (Formal.atOffset interface offset))
    (Formal.inputBindingOffset offset)

private theorem samplerFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalFreshCount
      (childConstraints (Formal.samplerCircuit (Formal.atOffset interface offset))
        (Formal.samplerOffset offset)) = 743631 := by
  exact SamplerChain.totalFreshCount_eq
    (Formal.samplerInterface (Formal.atOffset interface offset))
    (Formal.samplerOffset offset) inputs.sampler

private theorem samplerRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalRowCount
      (childConstraints (Formal.samplerCircuit (Formal.atOffset interface offset))
        (Formal.samplerOffset offset)) = 1008848 := by
  exact SamplerChain.totalRowCount_eq
    (Formal.samplerInterface (Formal.atOffset interface offset))
    (Formal.samplerOffset offset) inputs.sampler

private theorem commitmentFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalFreshCount
      (childConstraints
        (Formal.commitmentCircuit (Formal.atOffset interface offset))
        (Formal.commitmentOffset offset)) = 3029400 := by
  change R1CS.totalFreshCount
    (CommitmentCombination.logicalConstraints
      (Formal.commitmentInterface (Formal.atOffset interface offset))
      (Formal.commitmentOffset offset)) = 3029400
  rw [CommitmentCombination.totalFreshCount_eq]
  exact inputs.commitmentFresh

private theorem commitmentRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalRowCount
      (childConstraints
        (Formal.commitmentCircuit (Formal.atOffset interface offset))
        (Formal.commitmentOffset offset)) = 3049596 := by
  change R1CS.totalRowCount
    (CommitmentCombination.logicalConstraints
      (Formal.commitmentInterface (Formal.atOffset interface offset))
      (Formal.commitmentOffset offset)) = 3049596
  rw [CommitmentCombination.totalRowCount_eq]
  exact inputs.commitmentRows

private theorem publicInputFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalFreshCount
      (childConstraints
        (Formal.publicInputCircuit (Formal.atOffset interface offset))
        (Formal.publicInputOffset offset)) = 688500 := by
  change R1CS.totalFreshCount
    (PublicInputCombination.logicalConstraints
      (Formal.publicInputInterface (Formal.atOffset interface offset))
      (Formal.publicInputOffset offset)) = 688500
  rw [PublicInputCombination.totalFreshCount_eq]
  exact inputs.publicInputFresh

private theorem publicInputRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalRowCount
      (childConstraints
        (Formal.publicInputCircuit (Formal.atOffset interface offset))
        (Formal.publicInputOffset offset)) = 693090 := by
  change R1CS.totalRowCount
    (PublicInputCombination.logicalConstraints
      (Formal.publicInputInterface (Formal.atOffset interface offset))
      (Formal.publicInputOffset offset)) = 693090
  rw [PublicInputCombination.totalRowCount_eq]
  exact inputs.publicInputRows

private theorem evalKFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalFreshCount
      (childConstraints (Formal.evalKCircuit (Formal.atOffset interface offset))
        (Formal.evalKOffset offset)) = 275400 := by
  change R1CS.totalFreshCount
    (EvalKCombination.logicalConstraints
      (Formal.evalKInterface (Formal.atOffset interface offset))
      (Formal.evalKOffset offset)) = 275400
  rw [EvalKCombination.totalFreshCount_eq]
  exact inputs.evalKFresh

private theorem evalKRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalRowCount
      (childConstraints (Formal.evalKCircuit (Formal.atOffset interface offset))
        (Formal.evalKOffset offset)) = 277236 := by
  change R1CS.totalRowCount
    (EvalKCombination.logicalConstraints
      (Formal.evalKInterface (Formal.atOffset interface offset))
      (Formal.evalKOffset offset)) = 277236
  rw [EvalKCombination.totalRowCount_eq]
  exact inputs.evalKRows

private theorem evalAFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalFreshCount
      (childConstraints (Formal.evalACircuit (Formal.atOffset interface offset))
        (Formal.evalAOffset offset)) = 3855600 := by
  change R1CS.totalFreshCount
    (EvalACombination.logicalConstraints
      (Formal.evalAInterface (Formal.atOffset interface offset))
      (Formal.evalAOffset offset)) = 3855600
  rw [EvalACombination.totalFreshCount_eq]
  exact inputs.evalAFresh

private theorem evalARows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalRowCount
      (childConstraints (Formal.evalACircuit (Formal.atOffset interface offset))
        (Formal.evalAOffset offset)) = 3881304 := by
  change R1CS.totalRowCount
    (EvalACombination.logicalConstraints
      (Formal.evalAInterface (Formal.atOffset interface offset))
      (Formal.evalAOffset offset)) = 3881304
  rw [EvalACombination.totalRowCount_eq]
  exact inputs.evalARows

private theorem outputFresh_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    R1CS.totalFreshCount
      (childConstraints
        (Formal.outputBindingCircuit relation (Formal.atOffset interface offset)
          offset) (Formal.outputBindingOffset offset)) = 0 := by
  exact Leaves.OutputBinding.freshColumnCount_eq relation
    (Formal.outputBindingInterface (Formal.atOffset interface offset) offset)
    (Formal.outputBindingOffset offset)

private theorem outputRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    R1CS.totalRowCount
      (childConstraints
        (Formal.outputBindingCircuit relation (Formal.atOffset interface offset)
          offset) (Formal.outputBindingOffset offset)) = 0 := by
  exact Leaves.OutputBinding.physicalRowCount_eq relation
    (Formal.outputBindingInterface (Formal.atOffset interface offset) offset)
    (Formal.outputBindingOffset offset)

theorem physicalFreshDeltas_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    physicalFreshDeltas relation interface offset =
      [0, 743631, 3029400, 688500, 275400, 3855600, 0] := by
  unfold physicalFreshDeltas childConstraintLists
  simp only [List.map_cons, List.map_nil]
  rw [inputFresh_eq, samplerFresh_eq _ _ _ inputs,
    commitmentFresh_eq _ _ _ inputs, publicInputFresh_eq _ _ _ inputs,
    evalKFresh_eq _ _ _ inputs, evalAFresh_eq _ _ _ inputs,
    outputFresh_eq]

theorem physicalRowDeltas_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    physicalRowDeltas relation interface offset =
      [0, 1008848, 3049596, 693090, 277236, 3881304, 0] := by
  unfold physicalRowDeltas childConstraintLists
  simp only [List.map_cons, List.map_nil]
  rw [inputRows_eq, samplerRows_eq _ _ _ inputs,
    commitmentRows_eq _ _ _ inputs, publicInputRows_eq _ _ _ inputs,
    evalKRows_eq _ _ _ inputs, evalARows_eq _ _ _ inputs,
    outputRows_eq]

def logicalPrivateDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Nat :=
  (Formal.opsAt relation interface offset).map Op.localLength

theorem logicalPrivateDeltas_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    logicalPrivateDeltas relation interface offset =
      [0, 263568, 20196, 4590, 1836, 25704, 0] := by
  unfold logicalPrivateDeltas Formal.opsAt
  simp only [List.map_cons, List.map_nil, Formal.childOp_privateCount]
  rw [show (Formal.inputBindingCircuit relation (Formal.atOffset interface offset)).privateCount
        (Formal.inputBindingOffset offset) = 0 by rfl,
    show (Formal.samplerCircuit (Formal.atOffset interface offset)).privateCount
        (Formal.samplerOffset offset) = 263568 by rfl,
    show (Formal.commitmentCircuit (Formal.atOffset interface offset)).privateCount
        (Formal.commitmentOffset offset) = 20196 by
      exact CommitmentCombination.logicalPrivateCount_eq,
    show (Formal.publicInputCircuit (Formal.atOffset interface offset)).privateCount
        (Formal.publicInputOffset offset) = 4590 by
      exact PublicInputCombination.logicalPrivateCount_eq,
    show (Formal.evalKCircuit (Formal.atOffset interface offset)).privateCount
        (Formal.evalKOffset offset) = 1836 by
      exact EvalKCombination.logicalPrivateCount_eq,
    show (Formal.evalACircuit (Formal.atOffset interface offset)).privateCount
        (Formal.evalAOffset offset) = 25704 by
      exact EvalACombination.logicalPrivateCount_eq,
    show (Formal.outputBindingCircuit relation (Formal.atOffset interface offset)
        offset).privateCount (Formal.outputBindingOffset offset) = 0 by rfl]

def physicalColumnDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List Nat :=
  List.zipWith (· + ·) (logicalPrivateDeltas relation interface offset)
    (physicalFreshDeltas relation interface offset)

theorem physicalColumnDeltas_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    physicalColumnDeltas relation interface offset =
      [0, 1007199, 3049596, 693090, 277236, 3881304, 0] := by
  unfold physicalColumnDeltas
  rw [logicalPrivateDeltas_eq,
    physicalFreshDeltas_eq_production relation interface offset inputs]
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

theorem cumulativeFootprints_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    cumulativePhysicalRows relation interface offset =
        [0, 1008848, 4058444, 4751534, 5028770, 8910074, 8910074] ∧
      cumulativePhysicalColumns relation interface offset =
        [0, 1007199, 4056795, 4749885, 5027121, 8908425, 8908425] ∧
      cumulativeJointDomains relation interface offset =
        [0, 1008848, 4058444, 4751534, 5028770, 8910074, 8910074] := by
  rw [cumulativePhysicalRows,
    physicalRowDeltas_eq_production relation interface offset inputs,
    cumulativePhysicalColumns,
    physicalColumnDeltas_eq_production relation interface offset inputs]
  norm_num [cumulativeFrom, cumulativeJointDomains, cumulativePhysicalRows,
    cumulativePhysicalColumns,
    physicalRowDeltas_eq_production relation interface offset inputs,
    physicalColumnDeltas_eq_production relation interface offset inputs]

theorem totalFreshCount_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalFreshCount (logicalConstraints relation interface offset) =
      8592531 := by
  calc
    _ = (physicalFreshDeltas relation interface offset).sum :=
      totalFreshCount_eq_deltas relation interface offset
    _ = [0, 743631, 3029400, 688500, 275400, 3855600, 0].sum :=
      congrArg List.sum
        (physicalFreshDeltas_eq_production relation interface offset inputs)
    _ = 8592531 := by norm_num

theorem totalRowCount_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    R1CS.totalRowCount (logicalConstraints relation interface offset) =
      8910074 := by
  calc
    _ = (physicalRowDeltas relation interface offset).sum :=
      totalRowCount_eq_deltas relation interface offset
    _ = [0, 1008848, 3049596, 693090, 277236, 3881304, 0].sum :=
      congrArg List.sum
        (physicalRowDeltas_eq_production relation interface offset inputs)
    _ = 8910074 := by norm_num

theorem physicalPrivateColumnCount_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    Formal.logicalPrivateCount +
      R1CS.totalFreshCount (logicalConstraints relation interface offset) =
      8908425 := by
  rw [totalFreshCount_eq_production relation interface offset inputs]
  rfl

end NightstreamFPrime.Layout.PiRLC.v1_1
