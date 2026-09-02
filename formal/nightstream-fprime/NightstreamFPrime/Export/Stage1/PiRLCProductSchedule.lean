import Batteries.Data.Fin.Coding
import Mathlib.Data.List.OfFn
import NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations

/-!
Owns the constant-time production selector for the four PiRLC product
families. The selector uses the same family, source, block, lane, and cell
order as the proof-oriented compact-invocation lists.

This module does not construct matrix rows or select retained assignment
columns.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCProductSchedule

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations

/-- The four product families in exact SuperNeo order. -/
inductive Family where
  | commitment
  | publicInput
  | evalK
  | evalA
deriving Repr, DecidableEq

namespace Family

def blockCount : Family → Nat
  | .commitment => 22
  | .publicInput => 5
  | .evalK => 1
  | .evalA => 14

def cellCount : Family → Nat
  | .commitment => 1
  | .publicInput => 1
  | .evalK => 2
  | .evalA => 2

def valueStride : Family → Nat
  | .commitment => 1
  | .publicInput => 1
  | .evalK => 2
  | .evalA => 2

def logicalStart : Family → Nat
  | .commitment => PiRLCStarts.commitmentLogicalStart
  | .publicInput => PiRLCStarts.publicInputLogicalStart
  | .evalK => PiRLCStarts.evalKLogicalStart
  | .evalA => PiRLCStarts.evalALogicalStart

def privateCount (family : Family) : Nat :=
  CombinationStep.privateCount family.blockCount family.cellCount

def invocationCount (family : Family) : Nat :=
  sourceCount * family.privateCount

@[simp] theorem commitment_invocationCount :
    invocationCount .commitment = 20196 := by
  norm_num [invocationCount, privateCount, blockCount, cellCount,
    CombinationStep.privateCount, sourceCount, ringDegree]

@[simp] theorem publicInput_invocationCount :
    invocationCount .publicInput = 4590 := by
  norm_num [invocationCount, privateCount, blockCount, cellCount,
    CombinationStep.privateCount, sourceCount, ringDegree]

@[simp] theorem evalK_invocationCount : invocationCount .evalK = 1836 := by
  norm_num [invocationCount, privateCount, blockCount, cellCount,
    CombinationStep.privateCount, sourceCount, ringDegree]

@[simp] theorem evalA_invocationCount : invocationCount .evalA = 25704 := by
  norm_num [invocationCount, privateCount, blockCount, cellCount,
    CombinationStep.privateCount, sourceCount, ringDegree]

end Family

/-- One fully decoded product invocation. Dependent fields carry all bounds
needed by the source-layout formulas. -/
structure Descriptor where
  family : Family
  source : Fin sourceCount
  block : Fin family.blockCount
  lane : Fin ringDegree
  cell : Fin family.cellCount
deriving Repr

namespace Descriptor

def withLane (descriptor : Descriptor) (lane : Fin ringDegree) : Descriptor :=
  match descriptor with
  | ⟨family, source, block, _productLane, cell⟩ =>
      ⟨family, source, block, lane, cell⟩

def previousSource (descriptor : Descriptor)
    (notFirst : descriptor.source.val ≠ 0) : Descriptor :=
  match descriptor with
  | ⟨family, source, block, lane, cell⟩ =>
      ⟨family,
        ⟨source.val - 1, by
          have sourceBound := source.isLt
          omega⟩,
        block, lane, cell⟩

def challengeColumn (descriptor : Descriptor) (lane : Fin ringDegree) : Nat :=
  challengeSourceStart descriptor.source.val + lane.val

def valueColumn (descriptor : Descriptor) (lane : Fin ringDegree) : Nat :=
  match descriptor with
  | ⟨.commitment, source, block, _productLane, cell⟩ =>
      commitmentValueSourceStart source.val block.val cell.val + lane.val
  | ⟨.publicInput, source, block, _productLane, cell⟩ =>
      publicInputValueSourceStart source.val block.val cell.val + lane.val
  | ⟨.evalK, source, block, _productLane, cell⟩ =>
      evalKValueSourceStart source.val block.val cell.val + lane.val * 2
  | ⟨.evalA, source, block, _productLane, cell⟩ =>
      evalAValueSourceStart source.val block.val cell.val + lane.val * 2

@[simp] theorem withLane_valueColumn (descriptor : Descriptor)
    (lane : Fin ringDegree) :
    (descriptor.withLane lane).valueColumn
        (descriptor.withLane lane).lane =
      descriptor.valueColumn lane := by
  rcases descriptor with ⟨family, source, block, productLane, cell⟩
  cases family <;> rfl

def logicalIndex (descriptor : Descriptor) : Nat :=
  PiRLCCombinationInvocations.logicalIndex descriptor.family.cellCount
    descriptor.block.val descriptor.lane.val descriptor.cell.val

def outputColumn (descriptor : Descriptor) : Nat :=
  descriptor.family.logicalStart +
    descriptor.source.val *
      PiRLCCombinationInvocations.stepSize descriptor.family.blockCount
        descriptor.family.cellCount +
    descriptor.logicalIndex

def priorColumn (descriptor : Descriptor) : Nat :=
  descriptor.family.logicalStart +
    (descriptor.source.val - 1) *
      PiRLCCombinationInvocations.stepSize descriptor.family.blockCount
        descriptor.family.cellCount +
    descriptor.logicalIndex

@[simp] theorem previousSource_outputColumn (descriptor : Descriptor)
    (notFirst : descriptor.source.val ≠ 0) :
    (descriptor.previousSource notFirst).outputColumn =
      descriptor.priorColumn := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp [previousSource, outputColumn, priorColumn, logicalIndex]

def priorColumn? (descriptor : Descriptor) : Option Nat :=
  if descriptor.source.val = 0 then none
  else some descriptor.priorColumn

def challengeExpr (descriptor : Descriptor) : Fin ringDegree →
    NightstreamFPrime.Circuit.Expr :=
  fun lane => NightstreamFPrime.Circuit.Expr.var
    (descriptor.challengeColumn lane) - 2

def valueExpr (descriptor : Descriptor) : Fin ringDegree →
    NightstreamFPrime.Circuit.Expr :=
  fun lane => NightstreamFPrime.Circuit.Expr.var (descriptor.valueColumn lane)

def referenceValueExpr (descriptor : Descriptor) : Fin ringDegree →
    NightstreamFPrime.Circuit.Expr :=
  match descriptor with
  | ⟨.commitment, source, block, _lane, cell⟩ =>
      sourceValue 1 source.val block.val cell.val commitmentValueSourceStart
  | ⟨.publicInput, source, block, _lane, cell⟩ =>
      sourceValue 1 source.val block.val cell.val publicInputValueSourceStart
  | ⟨.evalK, source, block, _lane, cell⟩ =>
      sourceValue 2 source.val block.val cell.val evalKValueSourceStart
  | ⟨.evalA, source, block, _lane, cell⟩ =>
      sourceValue 2 source.val block.val cell.val evalAValueSourceStart

theorem challengeExpr_eq_sourceChallenge (descriptor : Descriptor) :
    descriptor.challengeExpr = sourceChallenge descriptor.source.val := by
  funext lane
  rfl

theorem valueExpr_eq_reference (descriptor : Descriptor) :
    descriptor.valueExpr = descriptor.referenceValueExpr := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;> funext current <;>
    simp [valueExpr, valueColumn, referenceValueExpr, sourceValue,
      commitmentValueSourceStart, publicInputValueSourceStart,
      evalKValueSourceStart, evalAValueSourceStart]

def priorExpr (descriptor : Descriptor) : NightstreamFPrime.Circuit.Expr :=
  if descriptor.source.val = 0 then 0
  else NightstreamFPrime.Circuit.Expr.var descriptor.priorColumn

def outputExpr (descriptor : Descriptor) : NightstreamFPrime.Circuit.Expr :=
  NightstreamFPrime.Circuit.Expr.var descriptor.outputColumn

/-- The exact source expression constrained by this product invocation. -/
def sourceConstraint (descriptor : Descriptor) :
    NightstreamFPrime.Circuit.Expr :=
  match descriptor with
  | ⟨.commitment, source, block, lane, cell⟩ =>
      PiRLCCombinationInvocations.sourceConstraint
        PiRLCStarts.commitmentLogicalStart 22 1 1 source.val block.val cell.val
          commitmentValueSourceStart lane
  | ⟨.publicInput, source, block, lane, cell⟩ =>
      PiRLCCombinationInvocations.sourceConstraint
        PiRLCStarts.publicInputLogicalStart 5 1 1 source.val block.val cell.val
          publicInputValueSourceStart lane
  | ⟨.evalK, source, block, lane, cell⟩ =>
      PiRLCCombinationInvocations.sourceConstraint
        PiRLCStarts.evalKLogicalStart 1 2 2 source.val block.val cell.val
          evalKValueSourceStart lane
  | ⟨.evalA, source, block, lane, cell⟩ =>
      PiRLCCombinationInvocations.sourceConstraint
        PiRLCStarts.evalALogicalStart 14 2 2 source.val block.val cell.val
          evalAValueSourceStart lane

/-- The direct descriptor expressions are definitionally the current
canonical PiRLC source constraint in all four families. -/
theorem sourceConstraint_eq_direct (descriptor : Descriptor) :
    descriptor.sourceConstraint =
      descriptor.outputExpr -
        (descriptor.priorExpr +
          CombinationStep.mulExpr descriptor.challengeExpr
            descriptor.valueExpr descriptor.lane) := by
  rw [challengeExpr_eq_sourceChallenge, valueExpr_eq_reference]
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp only [sourceConstraint,
      PiRLCCombinationInvocations.sourceConstraint,
      PiRLCCombinationInvocations.sourceOutput,
      PiRLCCombinationInvocations.sourcePrior,
      referenceValueExpr, outputExpr, priorExpr, outputColumn, priorColumn,
      logicalIndex, Family.logicalStart, Family.blockCount, Family.cellCount]

end Descriptor

/-- Constant-time source-major decoding inside one family. -/
def familyDescriptor (family : Family)
    (index : Fin family.invocationCount) : Descriptor :=
  let decoded : Fin sourceCount × Fin family.privateCount :=
    Fin.decodeProd index
  let coordinates := CombinationStep.coordinates decoded.2
  { family := family
    source := decoded.1
    block := coordinates.1
    lane := coordinates.2.1
    cell := coordinates.2.2 }

/-- The existing compact invocation selected by one decoded descriptor. -/
def Descriptor.compactInvocation (descriptor : Descriptor) :
    CompactRowInvocation :=
  match descriptor with
  | ⟨.commitment, source, block, lane, cell⟩ =>
      invocation PiRLCStarts.commitmentLogicalStart
        PiRLCStarts.commitmentRowStart PiRLCStarts.commitmentFreshStart
          22 1 1 source.val block.val lane.val cell.val
            commitmentValueSourceStart
  | ⟨.publicInput, source, block, lane, cell⟩ =>
      invocation PiRLCStarts.publicInputLogicalStart
        PiRLCStarts.publicInputRowStart PiRLCStarts.publicInputFreshStart
          5 1 1 source.val block.val lane.val cell.val
            publicInputValueSourceStart
  | ⟨.evalK, source, block, lane, cell⟩ =>
      invocation PiRLCStarts.evalKLogicalStart PiRLCStarts.evalKRowStart
        PiRLCStarts.evalKFreshStart
          1 2 2 source.val block.val lane.val cell.val
            evalKValueSourceStart
  | ⟨.evalA, source, block, lane, cell⟩ =>
      invocation PiRLCStarts.evalALogicalStart PiRLCStarts.evalARowStart
        PiRLCStarts.evalAFreshStart
          14 2 2 source.val block.val lane.val cell.val
            evalAValueSourceStart

def familyCompactInvocation (family : Family) :
    Fin family.invocationCount → CompactRowInvocation :=
  fun index => (familyDescriptor family index).compactInvocation

/-- Proof-oriented materialization of one direct family selector. -/
def familyCompactInvocations (family : Family) :
    List CompactRowInvocation :=
  List.ofFn (familyCompactInvocation family)

private theorem ofFn_decodeProd_eq_range_flatMap {Alpha : Type}
    (m n : Nat) (value : Nat → Fin n → Alpha) :
    List.ofFn (fun index : Fin (m * n) =>
        let decoded : Fin m × Fin n := Fin.decodeProd index
        value decoded.1.val decoded.2) =
      (List.range m).flatMap fun outer =>
        List.ofFn fun inner : Fin n => value outer inner := by
  rw [List.ofFn_mul]
  simp only [List.flatten_eq_flatMap]
  rw [List.ofFn_eq_map]
  rw [List.flatMap_map]
  rw [← List.map_coe_finRange_eq_range]
  rw [List.flatMap_map]
  apply List.flatMap_congr
  intro outer _member
  simp only [id_eq]
  apply congrArg List.ofFn
  funext inner
  let combined : Fin (m * n) :=
    ⟨outer.val * n + inner.val, by
      calc
        outer.val * n + inner.val < (outer.val + 1) * n := by
          simpa [Nat.add_mul] using Nat.add_lt_add_left inner.isLt (outer.val * n)
        _ ≤ m * n := Nat.mul_le_mul_right n outer.isLt⟩
  change
    value (Fin.decodeProd combined).1.val (Fin.decodeProd combined).2 =
      value outer.val inner
  have combined_eq : combined = Fin.encodeProd (outer, inner) := by
    apply Fin.ext
    simp [combined, Fin.encodeProd, Nat.mul_comm]
  rw [combined_eq, Fin.decodeProd_encodeProd]

theorem commitmentCompactInvocations_eq :
    familyCompactInvocations .commitment = commitmentInvocations := by
  unfold familyCompactInvocations familyCompactInvocation
    commitmentInvocations familyInvocations
  simp only [Family.invocationCount, Family.privateCount, Family.blockCount,
    Family.cellCount]
  simp only [familyDescriptor, Descriptor.compactInvocation]
  simpa only using
    (ofFn_decodeProd_eq_range_flatMap sourceCount
      (CombinationStep.privateCount 22 1)
      (fun source index =>
        let coordinates := CombinationStep.coordinates index
        invocation PiRLCStarts.commitmentLogicalStart
          PiRLCStarts.commitmentRowStart PiRLCStarts.commitmentFreshStart
            22 1 1 source coordinates.1.val coordinates.2.1.val
              coordinates.2.2.val commitmentValueSourceStart))

theorem publicInputCompactInvocations_eq :
    familyCompactInvocations .publicInput = publicInputInvocations := by
  unfold familyCompactInvocations familyCompactInvocation
    publicInputInvocations familyInvocations
  simp only [Family.invocationCount, Family.privateCount, Family.blockCount,
    Family.cellCount]
  simp only [familyDescriptor, Descriptor.compactInvocation]
  simpa only using
    (ofFn_decodeProd_eq_range_flatMap sourceCount
      (CombinationStep.privateCount 5 1)
      (fun source index =>
        let coordinates := CombinationStep.coordinates index
        invocation PiRLCStarts.publicInputLogicalStart
          PiRLCStarts.publicInputRowStart PiRLCStarts.publicInputFreshStart
            5 1 1 source coordinates.1.val coordinates.2.1.val
              coordinates.2.2.val publicInputValueSourceStart))

theorem evalKCompactInvocations_eq :
    familyCompactInvocations .evalK = evalKInvocations := by
  unfold familyCompactInvocations familyCompactInvocation
    evalKInvocations familyInvocations
  simp only [Family.invocationCount, Family.privateCount, Family.blockCount,
    Family.cellCount]
  simp only [familyDescriptor, Descriptor.compactInvocation]
  simpa only using
    (ofFn_decodeProd_eq_range_flatMap sourceCount
      (CombinationStep.privateCount 1 2)
      (fun source index =>
        let coordinates := CombinationStep.coordinates index
        invocation PiRLCStarts.evalKLogicalStart PiRLCStarts.evalKRowStart
          PiRLCStarts.evalKFreshStart 1 2 2 source coordinates.1.val
            coordinates.2.1.val coordinates.2.2.val evalKValueSourceStart))

theorem evalACompactInvocations_eq :
    familyCompactInvocations .evalA = evalAInvocations := by
  unfold familyCompactInvocations familyCompactInvocation
    evalAInvocations familyInvocations
  simp only [Family.invocationCount, Family.privateCount, Family.blockCount,
    Family.cellCount]
  simp only [familyDescriptor, Descriptor.compactInvocation]
  simpa only using
    (ofFn_decodeProd_eq_range_flatMap sourceCount
      (CombinationStep.privateCount 14 2)
      (fun source index =>
        let coordinates := CombinationStep.coordinates index
        invocation PiRLCStarts.evalALogicalStart PiRLCStarts.evalARowStart
          PiRLCStarts.evalAFreshStart 14 2 2 source coordinates.1.val
            coordinates.2.1.val coordinates.2.2.val evalAValueSourceStart))

/-- Total count expressed as the exact four-family sum. -/
def invocationCount : Nat :=
  Family.invocationCount .commitment +
    (Family.invocationCount .publicInput +
      (Family.invocationCount .evalK + Family.invocationCount .evalA))

@[simp] theorem invocationCount_eq : invocationCount = 52326 := by
  norm_num [invocationCount]

/-- Constant-time semantic descriptor selector in exact family order. -/
def descriptor : Fin invocationCount → Descriptor :=
  Fin.append (familyDescriptor .commitment) <|
    Fin.append (familyDescriptor .publicInput) <|
      Fin.append (familyDescriptor .evalK) (familyDescriptor .evalA)

def Descriptor.familyIndex (descriptor : Descriptor) :
    Fin descriptor.family.invocationCount :=
  match descriptor with
  | ⟨_family, source, block, lane, cell⟩ =>
      Fin.encodeProd (source, CombinationStep.indexOf block lane cell)

theorem familyDescriptor_familyIndex (descriptor : Descriptor) :
    familyDescriptor descriptor.family descriptor.familyIndex = descriptor := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp [Descriptor.familyIndex, familyDescriptor,
      CombinationStep.coordinates, CombinationStep.indexOf]

@[simp] theorem familyIndex_familyDescriptor (family : Family)
    (index : Fin family.invocationCount) :
    (familyDescriptor family index).familyIndex = index := by
  unfold familyDescriptor Descriptor.familyIndex
  dsimp only
  rw [show CombinationStep.indexOf
      (CombinationStep.coordinates (Fin.decodeProd index).2).1
      (CombinationStep.coordinates (Fin.decodeProd index).2).2.1
      (CombinationStep.coordinates (Fin.decodeProd index).2).2.2 =
        (Fin.decodeProd index).2 by
    unfold CombinationStep.coordinates CombinationStep.indexOf
    dsimp only
    rw [show finProdFinEquiv
        ((finProdFinEquiv.symm
          (finProdFinEquiv.symm (Fin.decodeProd index).2).2).1,
         (finProdFinEquiv.symm
          (finProdFinEquiv.symm (Fin.decodeProd index).2).2).2) =
          (finProdFinEquiv.symm (Fin.decodeProd index).2).2 by
      exact finProdFinEquiv.apply_symm_apply _]
    exact finProdFinEquiv.apply_symm_apply _]
  exact Fin.encodeProd_decodeProd index

/-- Constant-time inverse of `descriptor` in exact family order. -/
def Descriptor.invocation (descriptor : Descriptor) : Fin invocationCount :=
  match descriptor with
  | ⟨.commitment, source, block, lane, cell⟩ =>
      Fin.castAdd
        (Family.invocationCount .publicInput +
          (Family.invocationCount .evalK + Family.invocationCount .evalA))
        ({ family := .commitment, source, block, lane, cell } : Descriptor).familyIndex
  | ⟨.publicInput, source, block, lane, cell⟩ =>
      Fin.natAdd (Family.invocationCount .commitment) <|
        Fin.castAdd
          (Family.invocationCount .evalK + Family.invocationCount .evalA)
          ({ family := .publicInput, source, block, lane, cell } : Descriptor).familyIndex
  | ⟨.evalK, source, block, lane, cell⟩ =>
      Fin.natAdd (Family.invocationCount .commitment) <|
        Fin.natAdd (Family.invocationCount .publicInput) <|
          Fin.castAdd (Family.invocationCount .evalA)
            ({ family := .evalK, source, block, lane, cell } : Descriptor).familyIndex
  | ⟨.evalA, source, block, lane, cell⟩ =>
      Fin.natAdd (Family.invocationCount .commitment) <|
        Fin.natAdd (Family.invocationCount .publicInput) <|
          Fin.natAdd (Family.invocationCount .evalK)
            ({ family := .evalA, source, block, lane, cell } : Descriptor).familyIndex

@[simp] theorem descriptor_invocation (descriptor : Descriptor) :
    PiRLCProductSchedule.descriptor descriptor.invocation = descriptor := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family
  · simp [Descriptor.invocation, PiRLCProductSchedule.descriptor]
    exact familyDescriptor_familyIndex
      ({ family := .commitment, source, block, lane, cell } : Descriptor)
  · simp [Descriptor.invocation, PiRLCProductSchedule.descriptor]
    exact familyDescriptor_familyIndex
      ({ family := .publicInput, source, block, lane, cell } : Descriptor)
  · simp [Descriptor.invocation, PiRLCProductSchedule.descriptor]
    exact familyDescriptor_familyIndex
      ({ family := .evalK, source, block, lane, cell } : Descriptor)
  · simp [Descriptor.invocation, PiRLCProductSchedule.descriptor]
    exact familyDescriptor_familyIndex
      ({ family := .evalA, source, block, lane, cell } : Descriptor)

/-- The four-family decoder and encoder are inverse in the index direction.
Thus no physical product invocation is omitted or duplicated. -/
@[simp] theorem invocation_descriptor (index : Fin invocationCount) :
    (descriptor index).invocation = index := by
  unfold descriptor
  refine Fin.addCases (fun commitment => ?_) (fun remaining => ?_) index
  · simp only [Fin.append_left]
    change Fin.castAdd _ (familyDescriptor .commitment commitment).familyIndex =
      Fin.castAdd _ commitment
    rw [familyIndex_familyDescriptor]
    rfl
  · simp only [Fin.append_right]
    refine Fin.addCases (fun publicInput => ?_) (fun remaining => ?_) remaining
    · simp only [Fin.append_left]
      change Fin.natAdd (Family.invocationCount .commitment)
          (Fin.castAdd _ (familyDescriptor .publicInput publicInput).familyIndex) =
        Fin.natAdd (Family.invocationCount .commitment)
          (Fin.castAdd _ publicInput)
      rw [familyIndex_familyDescriptor]
      rfl
    · simp only [Fin.append_right]
      refine Fin.addCases (fun evalK => ?_) (fun evalA => ?_) remaining
      · simp only [Fin.append_left]
        change Fin.natAdd (Family.invocationCount .commitment)
            (Fin.natAdd (Family.invocationCount .publicInput)
              (Fin.castAdd _ (familyDescriptor .evalK evalK).familyIndex)) =
          Fin.natAdd (Family.invocationCount .commitment)
            (Fin.natAdd (Family.invocationCount .publicInput)
              (Fin.castAdd _ evalK))
        rw [familyIndex_familyDescriptor]
        rfl
      · simp only [Fin.append_right]
        change Fin.natAdd (Family.invocationCount .commitment)
            (Fin.natAdd (Family.invocationCount .publicInput)
              (Fin.natAdd (Family.invocationCount .evalK)
                (familyDescriptor .evalA evalA).familyIndex)) =
          Fin.natAdd (Family.invocationCount .commitment)
            (Fin.natAdd (Family.invocationCount .publicInput)
              (Fin.natAdd (Family.invocationCount .evalK) evalA))
        rw [familyIndex_familyDescriptor]
        rfl

/-- Constant-time selector in exact family order. -/
def compactInvocation : Fin invocationCount → CompactRowInvocation :=
  Fin.append (familyCompactInvocation .commitment) <|
    Fin.append (familyCompactInvocation .publicInput) <|
      Fin.append (familyCompactInvocation .evalK)
        (familyCompactInvocation .evalA)

private theorem map_append {Alpha Beta : Type} {m n : Nat}
    (map : Alpha → Beta) (left : Fin m → Alpha) (right : Fin n → Alpha) :
    (fun index => map (Fin.append left right index)) =
      Fin.append (fun index => map (left index))
        (fun index => map (right index)) := by
  funext index
  refine Fin.addCases (fun leftIndex => ?_) (fun rightIndex => ?_) index
  · simp
  · simp

/-- Decoding and then constructing the compact record is exactly the fast
compact-record selector. -/
theorem compactInvocation_eq_descriptor :
    compactInvocation = fun index => (descriptor index).compactInvocation := by
  unfold compactInvocation familyCompactInvocation descriptor
  rw [← map_append, ← map_append, ← map_append]
  rfl

/-- Proof-oriented materialization of the total constant-time selector. -/
def compactInvocations : List CompactRowInvocation :=
  List.ofFn compactInvocation

/-- The fast selector has exactly the current canonical invocation order and
contents. -/
theorem compactInvocations_eq : compactInvocations = invocations := by
  unfold compactInvocations compactInvocation invocationCount
  rw [List.ofFn_fin_append, List.ofFn_fin_append, List.ofFn_fin_append]
  change
    familyCompactInvocations .commitment ++
        (familyCompactInvocations .publicInput ++
          (familyCompactInvocations .evalK ++
            familyCompactInvocations .evalA)) = invocations
  rw [commitmentCompactInvocations_eq, publicInputCompactInvocations_eq,
    evalKCompactInvocations_eq, evalACompactInvocations_eq]
  simp only [invocations, List.append_assoc]

end NightstreamFPrime.Export.Stage1.PiRLCProductSchedule
