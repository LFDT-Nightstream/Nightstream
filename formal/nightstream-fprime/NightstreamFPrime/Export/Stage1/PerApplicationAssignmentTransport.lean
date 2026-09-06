import NightstreamFPrime.Export.Package
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentBlocks
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPreservation

/-!
Owns the sealed executable transport for the final 14-matrix assignment.
The transport keeps the existing 33 retained block plans and adds only the
recipes that cannot be recovered from their source runs: Phi81 group totals,
First54 accepted-symbol products, PiCCS payload expressions, and the four
verifier-owned output-digest words.

Every expression variable is renamed to its final physical package column.
The package does not carry expanded low-norm coordinates or assignment values.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PerApplicationCanonicalAssignment
open PerApplicationAssignmentPlan

abbrev Program := Lifecycle.Stage1.Application.Program

/-- One compact product-family shape. Product invocations are source-major,
then block-major, lane-major, and cell-major. -/
structure Phi81FamilyShape where
  sourceCount : Nat
  blockCount : Nat
  cellCount : Nat
deriving Repr, DecidableEq

def Phi81FamilyShape.format : Format Phi81FamilyShape where
  encode := fun shape => .array [
    .atom shape.sourceCount,
    .atom shape.blockCount,
    .atom shape.cellCount]
  decode
    | .array [.atom sourceCount, .atom blockCount, .atom cellCount] =>
        .ok { sourceCount, blockCount, cellCount }
    | _ => .error "invalid Phi81 assignment family shape"
  decode_encode := by
    intro shape
    cases shape
    rfl

/-- Compact executable recipe for all 52,326 by 33 Phi81 group values.
The scalar fields protocol-bind the fixed three-convolution Phi81 reduction;
the block fields name its retained inputs and outputs. -/
structure Phi81GroupRecipe where
  ringDegree : Nat
  middleDegree : Nat
  foldOffset : Nat
  twiceCutoff : Nat
  rawConvolutionCount : Nat
  rawTermCount : Nat
  groupWidth : Nat
  groupCount : Nat
  familyShapes : List Phi81FamilyShape
  challengeBlock : BlockKind
  challengeSlotBase : Nat
  challengeSourceStride : Nat
  challengeShift : Nat
  valueSources : List AffineRuns.Run
  groupOutputBlock : BlockKind
deriving Repr, DecidableEq

def Phi81GroupRecipe.format : Format Phi81GroupRecipe where
  encode := fun recipe => .array [
    .atom recipe.ringDegree,
    .atom recipe.middleDegree,
    .atom recipe.foldOffset,
    .atom recipe.twiceCutoff,
    .atom recipe.rawConvolutionCount,
    .atom recipe.rawTermCount,
    .atom recipe.groupWidth,
    .atom recipe.groupCount,
    (Codec.list Phi81FamilyShape.format).encode recipe.familyShapes,
    BlockKind.format.encode recipe.challengeBlock,
    .atom recipe.challengeSlotBase,
    .atom recipe.challengeSourceStride,
    .atom recipe.challengeShift,
    AffineRuns.format.encode recipe.valueSources,
    BlockKind.format.encode recipe.groupOutputBlock]
  decode
    | .array [.atom ringDegree, .atom middleDegree, .atom foldOffset,
        .atom twiceCutoff, .atom rawConvolutionCount, .atom rawTermCount,
        .atom groupWidth, .atom groupCount, familyShapes, challengeBlock,
        .atom challengeSlotBase, .atom challengeSourceStride,
        .atom challengeShift, valueSources, groupOutputBlock] => do
      pure {
        ringDegree,
        middleDegree,
        foldOffset,
        twiceCutoff,
        rawConvolutionCount,
        rawTermCount,
        groupWidth,
        groupCount,
        familyShapes :=
          ← (Codec.list Phi81FamilyShape.format).decode familyShapes,
        challengeBlock := ← BlockKind.format.decode challengeBlock,
        challengeSlotBase,
        challengeSourceStride,
        challengeShift,
        valueSources := ← AffineRuns.format.decode valueSources,
        groupOutputBlock := ← BlockKind.format.decode groupOutputBlock }
    | _ => .error "invalid Phi81 assignment group recipe"
  decode_encode := by
    intro recipe
    cases recipe
    simp only [(Codec.list Phi81FamilyShape.format).decode_encode,
      BlockKind.format.decode_encode, AffineRuns.decode_encode]
    rfl

/-- The four fixed product families in semantic order. -/
def phi81FamilyShapes : List Phi81FamilyShape :=
  [ ⟨17, 22, 1⟩
  , ⟨17, 5, 1⟩
  , ⟨17, 1, 2⟩
  , ⟨17, 14, 2⟩ ]

/-- Physical source columns used to construct the shared PiRLC operand values. -/
def phi81ValueSources (program : Program) : List AffineRuns.Run :=
  AffineRuns.compress <| List.ofFn fun invocation :
      Fin PiRLCProductSchedule.invocationCount =>
    let descriptor := PiRLCProductSchedule.descriptor invocation
    (PiRLCProductPlan.valueColumn program descriptor descriptor.lane).val

def directPhi81ValueSources (program : Program) : List AffineRuns.Run :=
  AffineRuns.compressIndexedTR fun invocation :
      Fin PiRLCProductSchedule.invocationCount =>
    let descriptor := PiRLCProductSchedule.descriptor invocation
    (PiRLCProductPlan.valueColumn program descriptor descriptor.lane).val

@[csimp] theorem phi81ValueSources_eq_direct :
    phi81ValueSources = directPhi81ValueSources := by
  funext program
  exact (AffineRuns.compressIndexedTR_eq_compress_ofFn _).symm

theorem phi81ValueSources_at (program : Program)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    AffineRuns.sourceAt (phi81ValueSources program) invocation.val =
      (PiRLCProductPlan.valueColumn program
        (PiRLCProductSchedule.descriptor invocation)
        (PiRLCProductSchedule.descriptor invocation).lane).val := by
  rw [AffineRuns.sourceAt_eq_expand_getD, phi81ValueSources,
    AffineRuns.expand_compress]
  exact Lifecycle.PriorStateHash.ofFn_getD _ invocation 0

/-- The complete generic Phi81 recipe. No per-invocation expressions are
materialized. -/
def phi81GroupRecipe (program : Program) : Phi81GroupRecipe where
  ringDegree := 54
  middleDegree := 27
  foldOffset := 81
  twiceCutoff := 106
  rawConvolutionCount := 3
  rawTermCount := 162
  groupWidth := 5
  groupCount := 33
  familyShapes := phi81FamilyShapes
  challengeBlock := .first54Value
  challengeSlotBase := 3402
  challengeSourceStride := 3456
  challengeShift := 2
  valueSources := phi81ValueSources program
  groupOutputBlock := .productGroup

/-- Compact executable recipe for the 1,088 shared First54 products. -/
structure First54ProductRecipe where
  candidateCount : Nat
  rejectBlock : BlockKind
  symbolBlock : BlockKind
  outputBlock : BlockKind
deriving Repr, DecidableEq

def First54ProductRecipe.format : Format First54ProductRecipe where
  encode := fun recipe => .array [
    .atom recipe.candidateCount,
    BlockKind.format.encode recipe.rejectBlock,
    BlockKind.format.encode recipe.symbolBlock,
    BlockKind.format.encode recipe.outputBlock]
  decode
    | .array [.atom candidateCount, rejectBlock, symbolBlock, outputBlock] =>
        do
          pure {
            candidateCount,
            rejectBlock := ← BlockKind.format.decode rejectBlock,
            symbolBlock := ← BlockKind.format.decode symbolBlock,
            outputBlock := ← BlockKind.format.decode outputBlock }
    | _ => .error "invalid First54 assignment product recipe"
  decode_encode := by
    intro recipe
    cases recipe
    simp only [BlockKind.format.decode_encode]
    rfl

def first54ProductRecipe : First54ProductRecipe where
  candidateCount := 1088
  rejectBlock := .first54Reject
  symbolBlock := .first54Symbol
  outputBlock := .first54Product

def physicalExpr (program : Program) (expression : Expr) : Expr :=
  CompactRows.renameExpr (PerApplicationPackage.shiftColumn program) <|
    PermutationOutput.Readout.rewriteExpr PiCCSTranscriptReadout.phaseStart
      NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport.transcriptInvocationCount <|
        CompactRows.renameExpr NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan expression

/-- Invocation-major, rate-lane-major PiCCS payload recipes. -/
def payloadExpressions (program : Program) : List Expr :=
  List.ofFn fun index : Fin PiCCSActionPayloadBlock.payloadCount =>
    physicalExpr program (PiCCSActionPayloadBlock.payloadExpression index)

@[simp] theorem payloadExpressions_length (program : Program) :
    (payloadExpressions program).length = 30416 := by
  rw [payloadExpressions, List.length_ofFn]
  exact PiCCSActionPayloadBlock.payloadCount_eq

/-- Apply the physical source map to the canonical ordered payload words. -/
def materializedPayloadExpressions (program : Program) : List Expr :=
  (PiCCSActionPayloadBlock.materializedPayloadExpressions ()).map
    (physicalExpr program)

theorem materializedPayloadExpressions_eq (program : Program) :
    materializedPayloadExpressions program = payloadExpressions program := by
  rw [materializedPayloadExpressions,
    PiCCSActionPayloadBlock.materializedPayloadExpressions_eq, ← List.ofFn_comp']
  rfl

/-- The exact four constrained Pilot output-digest expressions. -/
def outputDigestExpression (program : Program) (lane : Fin 4) : Expr :=
  physicalExpr program <|
    PilotProduction.outputInterface.digest
      (Lifecycle.Pilot.outputOffset PilotProduction.interface
        PilotProduction.witnessOffset) lane

def outputDigestExpressions (program : Program) : List Expr :=
  List.ofFn (outputDigestExpression program)

@[simp] theorem outputDigestExpressions_length (program : Program) :
    (outputDigestExpressions program).length = 4 := by
  simp [outputDigestExpressions]

/-- Schema of the assignment-transport child in the sealed package. -/
def schema : Nat := 1

/-- Complete package-carried transport plan. -/
structure Plan where
  blocks : List PerApplicationAssignmentBlocks.BlockPlan
  phi81 : Phi81GroupRecipe
  first54 : First54ProductRecipe
  payloadBlock : BlockKind
  payloadExpressions : List Expr
  outputDigestBlock : BlockKind
  outputDigestExpressions : List Expr
deriving Repr, DecidableEq

def Plan.format : Format Plan where
  encode := fun plan => .array [
    .atom schema,
    PerApplicationAssignmentBlocks.format.encode plan.blocks,
    Phi81GroupRecipe.format.encode plan.phi81,
    First54ProductRecipe.format.encode plan.first54,
    BlockKind.format.encode plan.payloadBlock,
    (Codec.list NightstreamFPrime.Export.Package.exprFormat).encode
      plan.payloadExpressions,
    BlockKind.format.encode plan.outputDigestBlock,
    (Codec.list NightstreamFPrime.Export.Package.exprFormat).encode
      plan.outputDigestExpressions]
  decode
    | .array [.atom 1, blocks, phi81, first54, payloadBlock,
        payloadExpressions, outputDigestBlock, outputDigestExpressions] => do
      pure {
        blocks := ← PerApplicationAssignmentBlocks.format.decode blocks,
        phi81 := ← Phi81GroupRecipe.format.decode phi81,
        first54 := ← First54ProductRecipe.format.decode first54,
        payloadBlock := ← BlockKind.format.decode payloadBlock,
        payloadExpressions :=
          ← (Codec.list NightstreamFPrime.Export.Package.exprFormat).decode
            payloadExpressions,
        outputDigestBlock := ← BlockKind.format.decode outputDigestBlock,
        outputDigestExpressions :=
          ← (Codec.list NightstreamFPrime.Export.Package.exprFormat).decode
            outputDigestExpressions }
    | _ => .error "invalid per-application assignment transport plan"
  decode_encode := by
    intro plan
    cases plan
    simp only [schema, PerApplicationAssignmentBlocks.format.decode_encode,
      Phi81GroupRecipe.format.decode_encode,
      First54ProductRecipe.format.decode_encode,
      BlockKind.format.decode_encode,
      (Codec.list NightstreamFPrime.Export.Package.exprFormat).decode_encode]
    rfl

def canonical (program : Program) : Plan where
  blocks := PerApplicationAssignmentBlocks.canonical program
  phi81 := phi81GroupRecipe program
  first54 := first54ProductRecipe
  payloadBlock := .piCcsPayload
  payloadExpressions := materializedPayloadExpressions program
  outputDigestBlock := .pilotOutputDigest
  outputDigestExpressions := outputDigestExpressions program

@[simp] theorem canonical_payloadExpressions (program : Program) :
    (canonical program).payloadExpressions = payloadExpressions program :=
  materializedPayloadExpressions_eq program

@[simp] theorem canonical_blocks_length (program : Program) :
    (canonical program).blocks.length = 33 := by
  exact PerApplicationAssignmentBlocks.canonical_length program

@[simp] theorem canonical_outputDigestExpressions_length (program : Program) :
    (canonical program).outputDigestExpressions.length = 4 := by
  change (outputDigestExpressions program).length = 4
  exact outputDigestExpressions_length program

end NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
