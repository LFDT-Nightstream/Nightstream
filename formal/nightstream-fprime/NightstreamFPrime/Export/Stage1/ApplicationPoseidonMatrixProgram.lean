import NightstreamFPrime.Export.Stage1.ApplicationPoseidonRetainedGeometry
import NightstreamFPrime.Layout.MatrixProgram.Program
import NightstreamFPrime.Layout.MatrixProgram.Poseidon
import NightstreamFPrime.Layout.MatrixProgram.Pin

/-! Encode the checked application's three permutations with existing matrix opcodes. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixProgram

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open ApplicationPoseidonRetainedBlock ApplicationPoseidonRetainedGeometry

variable {application : Stage1.Application.Program} {certificate : Certificate application}
  {columns : Nat}

def schedule (application : Stage1.Application.Program) (certificate : Certificate application) :
    PoseidonRetainedFamily.Schedule (ApplicationDirectSource.sourceWidth application) 3 where
  block := block application certificate
  slotCount_eq := block_slotCount application certificate

def constants (position : Fin 24) : Option F :=
  if position.val < 8 then
    some (Stage1.Poseidon2HashChainV1Prefix.constantState.getD position.val 0)
  else if position.val = 16 then some 1 else none

def constantRule : PoseidonInput.Rule where
  region := ⟨0, 3, 0, 8⟩
  term := .optionalConstant (PoseidonInput.OptionalConstantTable.ofSemantic constants) 8

def previousRule (application : Stage1.Application.Program) (certificate : Certificate application) :
    PoseidonInput.Rule where
  region := ⟨1, 2, 0, 8⟩
  term := .external (RetainedBlock.ofSemantic (block application certificate)
    (localStart application)) 78 86

def priorRule (application : Stage1.Application.Program) : PoseidonInput.Rule where
  region := ⟨0, 1, 0, 4⟩
  term := .retained (RetainedBlock.ofSemantic (ApplicationRetainedBlocks.inputBlock application)
    (inputStart application)) 0 0 1

def messageRule (application : Stage1.Application.Program) : PoseidonInput.Rule where
  region := ⟨1, 1, 0, 4⟩
  term := .retained (RetainedBlock.ofSemantic (ApplicationRetainedBlocks.witnessBlock application)
    (witnessStart application)) 0 0 1

def inputProgram (application : Stage1.Application.Program) (certificate : Certificate application) :
    PoseidonInput.Program where
  rules := [previousRule application certificate, constantRule,
    priorRule application, messageRule application]

def poseidonBlock (geometry : Geometry application certificate columns) : Poseidon.Block :=
  Poseidon.Block.ofSemantic (schedule application certificate) (localStart application)
    (oneColumn geometry) (inputProgram application certificate)

def bindingBlock (geometry : Geometry application certificate columns) : Pin.Block :=
  Pin.Block.ofSemantic (Stage1.Poseidon2HashChainCompact.pins (interface geometry))

def matrixProgram (geometry : Geometry application certificate columns) : MatrixProgram.Program where
  blocks := [.poseidon (poseidonBlock geometry), .pin (bindingBlock geometry)]

theorem matrixProgram_rowCount (geometry : Geometry application certificate columns) :
    (matrixProgram geometry).rowCount = 262 := by
  rw [show matrixProgram geometry = MatrixProgram.Program.mk
    [.poseidon (poseidonBlock geometry), .pin (bindingBlock geometry)] by rfl]
  rw [MatrixProgram.Program.two_rowCount]
  change (poseidonBlock geometry).rowCount + (bindingBlock geometry).rowCount = 262
  simp only [poseidonBlock, Poseidon.Block.ofSemantic_rowCount, bindingBlock,
    Pin.Block.ofSemantic_rowCount]

theorem bindingBlock_row? (geometry : Geometry application certificate columns) (row : Fin 4) :
    (bindingBlock geometry).row? columns row.val =
      some (PinFamilyPlan.forms (Stage1.Poseidon2HashChainCompact.pins (interface geometry)) row) :=
  Pin.Block.row?_ofSemantic _ row

end NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixProgram
