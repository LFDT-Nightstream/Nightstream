import Batteries.Data.Fin.Coding
import NightstreamFPrime.Layout.MatrixProgram
import NightstreamFPrime.Layout.ProductionRelation.Phi81ProductPlan
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep

/-!
Owns the generic compact opcode for an invocation-major family of direct
Phi81 product rows. The wire data fixes the family order and all retained
operands. A consumer decodes that data; it does not select a Stage 1 schedule.

This module does not select concrete PiRLC families or package rows.
-/

namespace NightstreamFPrime.Layout.MatrixProgram.Phi81Product

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- One encoded source-major product family. Lane count is the fixed Phi81
ring degree; only source, block, and cell counts vary. -/
structure Family where
  sourceCount : Nat
  blockCount : Nat
  cellCount : Nat
deriving Repr, DecidableEq

def Family.privateCount (family : Family) : Nat :=
  CombinationStep.privateCount family.blockCount family.cellCount

def Family.invocationCount (family : Family) : Nat :=
  family.sourceCount * family.privateCount

/-- Decoded coordinates retain the exact family-local bounds. -/
structure Descriptor where
  family : Family
  familyOffset : Nat
  source : Fin family.sourceCount
  coordinate : Fin family.privateCount

def Descriptor.coordinates (descriptor : Descriptor) :
    Fin descriptor.family.blockCount × Fin ringDegree ×
      Fin descriptor.family.cellCount :=
  CombinationStep.coordinates descriptor.coordinate

def Descriptor.block (descriptor : Descriptor) :
    Fin descriptor.family.blockCount :=
  descriptor.coordinates.1

def Descriptor.lane (descriptor : Descriptor) : Fin ringDegree :=
  descriptor.coordinates.2.1

def Descriptor.cell (descriptor : Descriptor) :
    Fin descriptor.family.cellCount :=
  descriptor.coordinates.2.2

def Descriptor.localInvocation (descriptor : Descriptor) : Nat :=
  (Fin.encodeProd (descriptor.source, descriptor.coordinate)).val

def Descriptor.invocation (descriptor : Descriptor) : Nat :=
  descriptor.familyOffset + descriptor.localInvocation

/-- Global invocation slot with only the product lane replaced. -/
def Descriptor.invocationAtLane (descriptor : Descriptor)
    (lane : Fin ringDegree) : Nat :=
  descriptor.familyOffset +
    (Fin.encodeProd (descriptor.source,
      CombinationStep.indexOf descriptor.block lane descriptor.cell)).val

def Family.descriptor? (family : Family) (familyOffset index : Nat) :
    Option Descriptor :=
  if bound : index < family.invocationCount then
    let decoded : Fin family.sourceCount × Fin family.privateCount :=
      Fin.decodeProd ⟨index, bound⟩
    some {
      family
      familyOffset
      source := decoded.1
      coordinate := decoded.2 }
  else
    none

@[simp] theorem Family.descriptor?_encode (family : Family)
    (familyOffset : Nat) (source : Fin family.sourceCount)
    (coordinate : Fin family.privateCount) :
    family.descriptor? familyOffset
        (Fin.encodeProd (source, coordinate)).val =
      some {
        family
        familyOffset
        source
        coordinate } := by
  unfold descriptor? invocationCount
  rw [dif_pos (Fin.encodeProd (source, coordinate)).isLt]
  simp

/-- Select one family without materializing the descriptor list. -/
def descriptorFrom? : List Family → Nat → Nat → Option Descriptor
  | [], _, _ => none
  | family :: rest, familyOffset, index =>
      if index < family.invocationCount then
        family.descriptor? familyOffset index
      else
        descriptorFrom? rest (familyOffset + family.invocationCount)
          (index - family.invocationCount)

@[simp] theorem descriptorFrom?_head (family : Family)
    (rest : List Family) (familyOffset : Nat)
    (source : Fin family.sourceCount)
    (coordinate : Fin family.privateCount) :
    descriptorFrom? (family :: rest) familyOffset
        (Fin.encodeProd (source, coordinate)).val =
      some {
        family
        familyOffset
        source
        coordinate } := by
  have bound : (Fin.encodeProd (source, coordinate)).val <
      family.invocationCount :=
    (Fin.encodeProd (source, coordinate)).isLt
  change (if (Fin.encodeProd (source, coordinate)).val <
      family.invocationCount then
        family.descriptor? familyOffset
          (Fin.encodeProd (source, coordinate)).val
      else
        descriptorFrom? rest (familyOffset + family.invocationCount)
          ((Fin.encodeProd (source, coordinate)).val -
            family.invocationCount)) = _
  rw [if_pos bound]
  exact Family.descriptor?_encode family familyOffset source coordinate

theorem descriptorFrom?_tail (family : Family) (rest : List Family)
    (familyOffset index : Nat) :
    descriptorFrom? (family :: rest) familyOffset
        (family.invocationCount + index) =
      descriptorFrom? rest (familyOffset + family.invocationCount) index := by
  change (if family.invocationCount + index < family.invocationCount then
      family.descriptor? familyOffset (family.invocationCount + index)
    else
      descriptorFrom? rest (familyOffset + family.invocationCount)
        (family.invocationCount + index - family.invocationCount)) = _
  rw [if_neg (by omega)]
  rw [Nat.add_sub_cancel_left]

def descriptor? (families : List Family) (index : Nat) : Option Descriptor :=
  descriptorFrom? families 0 index

def invocationCount (families : List Family) : Nat :=
  (families.map Family.invocationCount).sum

/-- A challenge is either the legacy retained digit or an explicit sparse
form. Sparse forms carry the centered value and allocate no copied field. -/
inductive Challenge where
  | retained (block : RetainedBlock) (slotStart sourceStride : Nat)
  | direct (forms : Array WireForm) (sourceStride : Nat)
deriving Repr, DecidableEq

def Challenge.form? {columns : Nat} (challenge : Challenge) (one : Fin columns)
    (source lane : Nat) : Option (SparseForm columns) :=
  match challenge with
  | .retained block slotStart sourceStride => do
      let digit ← block.form? columns (slotStart + source * sourceStride + lane)
      pure (SparseForm.add digit (SparseForm.singleton one (-2)))
  | .direct forms sourceStride => do
      let encoded ← forms[source * sourceStride + lane]?
      encoded.semantic? columns

/-- Decoding the old operand keeps the same centered form and entry order. -/
theorem Challenge.retained_form {columns : Nat} (block : RetainedBlock)
    (slotStart sourceStride source lane : Nat) (one : Fin columns) (digit : SparseForm columns)
    (loaded : block.form? columns (slotStart + source * sourceStride + lane) = some digit) :
    (Challenge.retained block slotStart sourceStride).form? one source lane =
      some (SparseForm.add digit (SparseForm.singleton one (-2))) := by
  simp only [Challenge.form?, loaded]
  rfl

/-- The direct operand preserves every coefficient and its stored position. -/
theorem Challenge.direct_form {columns count : Nat}
    (forms : Fin count → SparseForm columns) (one : Fin columns)
    (sourceStride source lane : Nat) (index : Fin count)
    (position : source * sourceStride + lane = index.val) :
    (Challenge.direct (Array.ofFn fun i => WireForm.ofSemantic (forms i)) sourceStride).form?
      one source lane = some (forms index) := by
  simp [Challenge.form?, position, WireForm.semantic?_ofSemantic]

/-- Complete wire operands for one direct Phi81 product family block. -/
structure Block where
  families : List Family
  oneColumn : Nat
  challenge : Challenge
  input : SourceSubstitution
  output : RetainedBlock
  group : RetainedBlock
deriving Repr, DecidableEq

def Block.invocationCount (block : Block) : Nat :=
  Phi81Product.invocationCount block.families

def Family.ringCount (family : Family) : Nat :=
  family.sourceCount * (family.blockCount * family.cellCount)

def ringCount (families : List Family) : Nat :=
  (families.map Family.ringCount).sum

/-- Decode a full ring while retaining the original lane/cell source order. -/
def Family.ringDescriptor? (family : Family) (familyOffset index : Nat) :
    Option Descriptor :=
  if bound : index < family.ringCount then
    let sourceAndCell : Fin family.sourceCount ×
        Fin (family.blockCount * family.cellCount) := Fin.decodeProd ⟨index, bound⟩
    let blockAndCell : Fin family.blockCount × Fin family.cellCount :=
      Fin.decodeProd sourceAndCell.2
    some {
      family, familyOffset
      source := sourceAndCell.1
      coordinate := CombinationStep.indexOf blockAndCell.1 ⟨0, by decide⟩ blockAndCell.2 }
  else none

def ringDescriptorFrom? : List Family → Nat → Nat → Option Descriptor
  | [], _, _ => none
  | family :: rest, familyOffset, index =>
      if index < family.ringCount then
        family.ringDescriptor? familyOffset index
      else
        ringDescriptorFrom? rest (familyOffset + family.invocationCount)
          (index - family.ringCount)

def ringDescriptor? (families : List Family) (index : Nat) : Option Descriptor :=
  ringDescriptorFrom? families 0 index

@[simp] theorem Family.ringDescriptor?_encode (family : Family)
    (familyOffset : Nat) (source : Fin family.sourceCount)
    (block : Fin family.blockCount) (cell : Fin family.cellCount) :
    family.ringDescriptor? familyOffset
        (Fin.encodeProd (source, Fin.encodeProd (block, cell))).val =
      some {
        family := family
        familyOffset := familyOffset
        source := source
        coordinate := CombinationStep.indexOf block ⟨0, by decide⟩ cell } := by
  unfold Family.ringDescriptor? Family.ringCount
  rw [dif_pos (Fin.encodeProd (source, Fin.encodeProd (block, cell))).isLt]
  simp

@[simp] theorem ringDescriptorFrom?_head (family : Family)
    (rest : List Family) (familyOffset : Nat) (source : Fin family.sourceCount)
    (block : Fin family.blockCount) (cell : Fin family.cellCount) :
    ringDescriptorFrom? (family :: rest) familyOffset
        (Fin.encodeProd (source, Fin.encodeProd (block, cell))).val =
      some {
        family := family
        familyOffset := familyOffset
        source := source
        coordinate := CombinationStep.indexOf block ⟨0, by decide⟩ cell } := by
  have bound : (Fin.encodeProd (source, Fin.encodeProd (block, cell))).val <
      family.ringCount := (Fin.encodeProd (source, Fin.encodeProd (block, cell))).isLt
  simp only [ringDescriptorFrom?, bound, if_pos]
  exact family.ringDescriptor?_encode familyOffset source block cell

theorem ringDescriptorFrom?_tail (family : Family) (rest : List Family)
    (familyOffset index : Nat) :
    ringDescriptorFrom? (family :: rest) familyOffset
        (family.ringCount + index) =
      ringDescriptorFrom? rest (familyOffset + family.invocationCount) index := by
  change (if family.ringCount + index < family.ringCount then _ else _) = _
  rw [if_neg (by omega), Nat.add_sub_cancel_left]

def Block.rowCount (block : Block) : Nat :=
  ringCount block.families * 108

/-- Load a fixed finite function. Any missing element rejects the complete
function. -/
def loadFin? {Alpha : Type} :
    (count : Nat) → (Fin count → Option Alpha) →
      Option (Fin count → Alpha)
  | 0, _ => some Fin.elim0
  | count + 1, load => do
      let head ← load 0
      let tail ← loadFin? count (fun index => load index.succ)
      pure (Fin.cases head tail)

theorem loadFin?_of_some {Alpha : Type} (count : Nat)
    (load : Fin count → Option Alpha) (value : Fin count → Alpha)
    (loaded : ∀ index, load index = some (value index)) :
    loadFin? count load = some value := by
  induction count with
  | zero =>
      simp only [loadFin?]
      apply congrArg some
      funext index
      exact Fin.elim0 index
  | succ count inductionHypothesis =>
      rw [loadFin?]
      rw [loaded 0]
      rw [inductionHypothesis
        (fun index => load index.succ)
        (fun index => value index.succ)
        (fun index => loaded index.succ)]
      apply congrArg some
      funext index
      refine Fin.cases ?_ (fun _ => ?_) index <;> rfl

def Block.oneColumn? (block : Block) (logicalWidth : Nat) :
    Option (Fin logicalWidth) :=
  if bound : block.oneColumn < logicalWidth then
    some ⟨block.oneColumn, bound⟩
  else
    none

def Block.challengeState? (block : Block) {logicalWidth : Nat} (one : Fin logicalWidth)
    (descriptor : Descriptor) : Option (Phi81ProductPlan.State logicalWidth) :=
  loadFin? ringDegree fun lane => block.challenge.form? one descriptor.source.val lane.val

def Block.inputState? (block : Block) (logicalWidth : Nat)
    (descriptor : Descriptor) :
    Option (Phi81ProductPlan.State logicalWidth) :=
  loadFin? ringDegree fun lane =>
    block.input.form? logicalWidth (descriptor.invocationAtLane lane)

def Block.quotientState? (block : Block) (logicalWidth : Nat)
    (descriptor : Descriptor) :
    Option (Phi81ProductPlan.State logicalWidth) :=
  loadFin? ringDegree fun lane =>
    block.group.form? logicalWidth (descriptor.invocationAtLane lane)

def Block.outputState? (block : Block) (logicalWidth : Nat)
    (descriptor : Descriptor) :
    Option (Phi81ProductPlan.State logicalWidth) :=
  loadFin? ringDegree fun lane =>
    block.output.form? logicalWidth (descriptor.invocationAtLane lane)

def Block.priorState? (block : Block) (logicalWidth : Nat)
    (descriptor : Descriptor) :
    Option (Phi81ProductPlan.State logicalWidth) :=
  if descriptor.source.val = 0 then some (fun _ => SparseForm.empty)
  else loadFin? ringDegree fun lane =>
    block.output.form? logicalWidth
      (descriptor.invocationAtLane lane - descriptor.family.privateCount)

/-- Reconstruct one complete ring equation from its canonical retained data. -/
def Block.interface? (block : Block) (logicalWidth : Nat)
    (descriptor : Descriptor) :
    Option (Phi81ProductPlan.Interface logicalWidth) := do
  let oneColumn ← block.oneColumn? logicalWidth
  let left ← block.challengeState? oneColumn descriptor
  let input ← block.inputState? logicalWidth descriptor
  let quotient ← block.quotientState? logicalWidth descriptor
  let prior ← block.priorState? logicalWidth descriptor
  let output ← block.outputState? logicalWidth descriptor
  pure { oneColumn, left, right := input, quotient, prior, output }

/-- Ring-major order, then every fixed evaluation point in increasing order. -/
def Block.row? (block : Block) (logicalWidth ordinal : Nat) :
    Option (RowForms logicalWidth) :=
  if ordinal < block.rowCount then do
    let descriptor ← ringDescriptor? block.families (ordinal / 108)
    let interface ← block.interface? logicalWidth descriptor
    let row ← (Phi81ProductPlan.rows interface)[ordinal % 108]?
    pure row.meaningfulForm
  else none

theorem Block.row?_of_loaded (block : Block) (logicalWidth ordinal : Nat)
    (bound : ordinal < block.rowCount)
    (descriptor : Descriptor)
    (selected : ringDescriptor? block.families (ordinal / 108) = some descriptor)
    (interface : Phi81ProductPlan.Interface logicalWidth)
    (loaded : block.interface? logicalWidth descriptor = some interface)
    (row : ProductSumPlan.Row logicalWidth)
    (rowSelected : (Phi81ProductPlan.rows interface)[ordinal % 108]? = some row) :
    block.row? logicalWidth ordinal = some row.meaningfulForm := by
  simp [Block.row?, bound, selected, loaded, rowSelected]

end NightstreamFPrime.Layout.MatrixProgram.Phi81Product
