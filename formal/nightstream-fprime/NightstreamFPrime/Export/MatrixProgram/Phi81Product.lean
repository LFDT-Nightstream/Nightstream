import Batteries.Data.Fin.Coding
import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Layout.ProductionRelation.Phi81ProductPlan
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep

/-!
Owns the generic compact opcode for an invocation-major family of direct
Phi81 product rows. The wire data fixes the family order and all retained
operands. A consumer decodes that data; it does not select a Stage 1 schedule.

This module does not select concrete PiRLC families or package rows.
-/

namespace NightstreamFPrime.Export.MatrixProgram.Phi81Product

open NightstreamFPrime.Export.Codec
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

def Family.format : Format Family where
  encode := fun family => .array [
    .atom family.sourceCount,
    .atom family.blockCount,
    .atom family.cellCount]
  decode
    | .array [.atom sourceCount, .atom blockCount, .atom cellCount] =>
        .ok ⟨sourceCount, blockCount, cellCount⟩
    | _ => .error "invalid Phi81 product family"
  decode_encode := by
    intro family
    cases family
    rfl

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

/-- Complete wire operands for one direct Phi81 product family block. -/
structure Block where
  families : List Family
  oneColumn : Nat
  challenge : RetainedBlock
  challengeSlotStart : Nat
  challengeSourceStride : Nat
  input : SourceSubstitution
  output : RetainedBlock
  group : RetainedBlock
deriving Repr, DecidableEq

def Block.format : Format Block where
  encode := fun block => .array [
    (list Family.format).encode block.families,
    .atom block.oneColumn,
    RetainedBlock.format.encode block.challenge,
    .atom block.challengeSlotStart,
    .atom block.challengeSourceStride,
    SourceSubstitution.format.encode block.input,
    RetainedBlock.format.encode block.output,
    RetainedBlock.format.encode block.group]
  decode
    | .array [families, .atom oneColumn, challenge,
        .atom challengeSlotStart, .atom challengeSourceStride,
        input, output, group] => do
      pure {
        families := ← (list Family.format).decode families
        oneColumn
        challenge := ← RetainedBlock.format.decode challenge
        challengeSlotStart
        challengeSourceStride
        input := ← SourceSubstitution.format.decode input
        output := ← RetainedBlock.format.decode output
        group := ← RetainedBlock.format.decode group }
    | _ => .error "invalid Phi81 product block"
  decode_encode := by
    rintro ⟨families, oneColumn, challenge, challengeSlotStart,
      challengeSourceStride, input, output, group⟩
    simp only
    rw [(list Family.format).decode_encode,
      RetainedBlock.format.decode_encode,
      SourceSubstitution.format.decode_encode,
      RetainedBlock.format.decode_encode,
      RetainedBlock.format.decode_encode]
    rfl

def Block.invocationCount (block : Block) : Nat :=
  Phi81Product.invocationCount block.families

def Block.rowCount (block : Block) : Nat :=
  block.invocationCount * 34

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

def Block.challengeState? (block : Block) (logicalWidth : Nat)
    (descriptor : Descriptor) :
    Option (Phi81ProductPlan.State logicalWidth) :=
  loadFin? ringDegree fun lane =>
    block.challenge.form? logicalWidth
      (block.challengeSlotStart +
        descriptor.source.val * block.challengeSourceStride + lane.val)

def Block.inputState? (block : Block) (logicalWidth : Nat)
    (descriptor : Descriptor) :
    Option (Phi81ProductPlan.State logicalWidth) :=
  loadFin? ringDegree fun lane =>
    block.input.form? logicalWidth (descriptor.invocationAtLane lane)

def Block.groupOutput? (block : Block) (logicalWidth : Nat)
    (descriptor : Descriptor) :
    Option (Fin 33 → SparseForm logicalWidth) :=
  loadFin? 33 fun group =>
    block.group.form? logicalWidth (descriptor.invocation * 33 + group.val)

/-- Reconstruct the exact direct product-row interface for one invocation.
Every retained lookup and the constant column fail closed. -/
def Block.interface? (block : Block) (logicalWidth : Nat)
    (descriptor : Descriptor) :
    Option (ProductSumPlan.Interface logicalWidth) := do
  let oneColumn ← block.oneColumn? logicalWidth
  let challenge ← block.challengeState? logicalWidth descriptor
  let input ← block.inputState? logicalWidth descriptor
  let groupOutput ← block.groupOutput? logicalWidth descriptor
  let prior ← if descriptor.source.val = 0 then
      some SparseForm.empty
    else
      block.output.form? logicalWidth
        (descriptor.invocation - descriptor.family.privateCount)
  let output ← block.output.form? logicalWidth descriptor.invocation
  let left : Phi81ProductPlan.State logicalWidth := fun lane =>
    SparseForm.add (challenge lane)
      (SparseForm.singleton oneColumn (-2))
  pure {
    oneColumn
    terms := Phi81ProductPlan.terms left input descriptor.lane
    groupOutput
    prior
    output }

/-- Select one compact product row without expanding the family. -/
def Block.row? (block : Block) (logicalWidth ordinal : Nat) :
    Option (RowForms logicalWidth) :=
  if ordinal < block.rowCount then do
    let descriptor ← descriptor? block.families (ordinal / 34)
    let interface ← block.interface? logicalWidth descriptor
    let row ← (ProductSumPlan.rows interface)[ordinal % 34]?
    pure row.meaningfulForm
  else
    none

theorem Block.row?_of_loaded (block : Block) (logicalWidth ordinal : Nat)
    (bound : ordinal < block.rowCount)
    (descriptor : Descriptor)
    (selected : descriptor? block.families (ordinal / 34) = some descriptor)
    (interface : ProductSumPlan.Interface logicalWidth)
    (loaded : block.interface? logicalWidth descriptor = some interface)
    (row : ProductSumPlan.Row logicalWidth)
    (rowSelected : (ProductSumPlan.rows interface)[ordinal % 34]? =
      some row) :
    block.row? logicalWidth ordinal = some row.meaningfulForm := by
  simp [Block.row?, bound, selected, loaded, rowSelected]

end NightstreamFPrime.Export.MatrixProgram.Phi81Product
