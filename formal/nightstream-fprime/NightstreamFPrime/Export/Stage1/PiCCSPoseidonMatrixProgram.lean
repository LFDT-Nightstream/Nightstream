import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan
import NightstreamFPrime.Export.Stage1.PiCCSPayloadMatrix

/-!
Owns the compact matrix program for the complete PiCCS Poseidon2 block. Lean
supplies one action tag per invocation, parent affine words and retained S-boxes,
and the squeeze-binding pin rows.

This module does not select PiCCS actions or close package conformance.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

abbrev Program := Lifecycle.Stage1.Application.Program

def invocationTag : PoseidonActionSchedule.Kind → PoseidonInput.InvocationTag
  | .absorb _ => .absorb
  | .squeezeFirst _ => .squeezeFirst
  | .squeezeSecond => .squeezeSecond

def tagAt (invocation : Fin PiCCSPoseidonPlan.invocationCount) :
    PoseidonInput.InvocationTag :=
  invocationTag (PiCCSActionPayloadBlock.kindAt invocation)

def tags (_delay : Unit := ()) : PoseidonInput.TagTable :=
  PoseidonInput.TagTable.ofSemantic tagAt

def materializedKinds (_delay : Unit := ()) :
    List PoseidonActionSchedule.Kind :=
  PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.statementActions ++
    (PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.challengeActions ++
      (PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.roundActions ++
        PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.outputActions))

def directTags (_delay : Unit := ()) : PoseidonInput.TagTable where
  tags := ((materializedKinds ()).map invocationTag).toArray

theorem materializedKinds_eq (delay : Unit := ()) :
    materializedKinds delay = List.ofFn PiCCSActionPayloadBlock.kindAt := by
  unfold materializedKinds
  exact PiCCSActionPayloadBlock.kindAt_materializes.symm

theorem directTags_eq_tags (delay : Unit := ()) :
    directTags delay = tags delay := by
  unfold directTags tags PoseidonInput.TagTable.ofSemantic
  apply congrArg PoseidonInput.TagTable.mk
  rw [materializedKinds_eq, List.map_ofFn, List.toArray_ofFn]
  rfl

@[csimp] theorem tags_eq_directTags : @tags = @directTags := by
  funext delay
  exact (directTags_eq_tags (delay := delay)).symm

theorem directTag_lookup
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) :
    (directTags ()).tag? invocation.val = some (tagAt invocation) := by
  rw [directTags_eq_tags]
  exact PoseidonInput.TagTable.tag?_ofSemantic tagAt invocation

def directBindingForm
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount)
    (component : Fin 2) : SparseForm logicalWidth :=
  match (directTags ()).tag? invocation.val with
  | some .squeezeFirst =>
      SparseForm.add
        (PiCCSPoseidonPlan.payloadForm (PiCCSPayloadWiring.form geometry) invocation
          ⟨component.val, Nat.lt_trans component.isLt (by
            norm_num [Spec.Poseidon2.width])⟩)
        (SparseForm.scale (-1)
          (PiCCSPoseidonPlan.bindingActual
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry) invocation component))
  | _ => .empty

theorem directBindingForm_eq_bindingForm
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount)
    (component : Fin 2) :
    directBindingForm geometry invocation component =
      PiCCSPoseidonPlan.bindingForm (PiCCSPayloadWiring.form geometry)
      (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry) invocation component := by
  unfold directBindingForm PiCCSPoseidonPlan.bindingForm
  rw [directTag_lookup]
  unfold tagAt invocationTag
  cases PiCCSActionPayloadBlock.kindAt invocation <;> rfl

def directBindingInterface
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    PinFamilyPlan.Interface logicalWidth PiCCSPoseidonPlan.bindingRowCount where
  oneColumn := PiCCSOrdinaryRetainedGeometry.oneColumn geometry
  value := fun row =>
    let decoded : Fin PiCCSPoseidonPlan.invocationCount × Fin 2 :=
      Fin.decodeProd row
    directBindingForm geometry decoded.1 decoded.2

theorem directBindingInterface_eq_bindingInterface
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    directBindingInterface geometry =
      PiCCSPoseidonPlan.bindingInterface (PiCCSPayloadWiring.form geometry)
      (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry) := by
  unfold directBindingInterface PiCCSPoseidonPlan.bindingInterface
  apply congrArg
    (PinFamilyPlan.Interface.mk (PiCCSOrdinaryRetainedGeometry.oneColumn geometry))
  funext row
  exact directBindingForm_eq_bindingForm geometry
    (Fin.decodeProd row).1 (Fin.decodeProd row).2

def previousRule (program : Program) : PoseidonInput.Rule where
  region := ⟨1, 7603, 0, 8⟩
  term := .external
    (RetainedBlock.ofSemantic (PiCCSPoseidonPlan.retainedBlock program)
      (PiCCSPoseidonPlan.retainedStart program)) 78 86

def payloadRule (program : Program) : PoseidonInput.Rule where
  region := ⟨0, 7604, 0, 4⟩
  term := .taggedAffine (PiCCSPayloadMatrix.table ())
    (PiCCSOrdinaryMatrixProgram.substitution program) tags .absorb 4

def inputProgram (program : Program) : PoseidonInput.Program where
  rules := [previousRule program, payloadRule program]

def poseidonBlock {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    Poseidon.Block :=
  Poseidon.Block.ofSemantic (PiCCSPoseidonPlan.schedule program)
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) (inputProgram program)

def bindingBlock {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) : Pin.Block :=
  Pin.Block.ofSemantic (PiCCSPoseidonPlan.bindingInterface (PiCCSPayloadWiring.form geometry)
      (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))

def directBindingBlock {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) : Pin.Block :=
  Pin.Block.ofSemantic (directBindingInterface geometry)

theorem directBindingBlock_eq_bindingBlock
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    directBindingBlock geometry = bindingBlock geometry := by
  unfold directBindingBlock bindingBlock
  rw [directBindingInterface_eq_bindingInterface]

@[csimp] theorem bindingBlock_eq_directBindingBlock :
    @bindingBlock = @directBindingBlock := by
  funext program logicalWidth geometry
  exact (directBindingBlock_eq_bindingBlock geometry).symm

/-- PiCCS permutation rows precede the squeeze-binding rows. -/
def matrixProgram {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.Program where
  blocks := [.poseidon (poseidonBlock geometry), .pin (bindingBlock geometry)]

@[simp] theorem poseidonBlock_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (poseidonBlock geometry).rowCount = 714776 := by
  calc
    (poseidonBlock geometry).rowCount =
        PiCCSPoseidonPlan.invocationCount * 94 := by
      exact Poseidon.Block.ofSemantic_rowCount
        (PiCCSPoseidonPlan.schedule program)
        (PiCCSPoseidonPlan.retainedStart program)
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) (inputProgram program)
    _ = 714776 := by
      norm_num [PiCCSPoseidonPlan.invocationCount_eq]

@[simp] theorem bindingBlock_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (bindingBlock geometry).rowCount = 15208 := by
  calc
    (bindingBlock geometry).rowCount = PiCCSPoseidonPlan.bindingRowCount := by
      exact Pin.Block.ofSemantic_rowCount
        (PiCCSPoseidonPlan.bindingInterface (PiCCSPayloadWiring.form geometry)
      (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
    _ = 15208 := by
      norm_num [PiCCSPoseidonPlan.bindingRowCount,
        PiCCSPoseidonPlan.invocationCount_eq]

@[simp] theorem matrixProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 729984 := by
  rw [show matrixProgram geometry = MatrixProgram.Program.mk
      [.poseidon (poseidonBlock geometry), .pin (bindingBlock geometry)] by
    rfl]
  rw [MatrixProgram.Program.two_rowCount]
  change (poseidonBlock geometry).rowCount +
    (bindingBlock geometry).rowCount = 729984
  rw [poseidonBlock_rowCount, bindingBlock_rowCount]

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram
