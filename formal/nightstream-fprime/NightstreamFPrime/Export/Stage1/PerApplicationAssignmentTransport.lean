import NightstreamFPrime.Export.Package
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentBlocks
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPreservation

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

structure RingFamilyShape where
  sourceCount : Nat
  blockCount : Nat
  cellCount : Nat
deriving Repr, DecidableEq

def RingFamilyShape.format : Format RingFamilyShape where
  encode := fun shape => .array [
    .atom shape.sourceCount,
    .atom shape.blockCount,
    .atom shape.cellCount]
  decode
    | .array [.atom sourceCount, .atom blockCount, .atom cellCount] =>
        .ok { sourceCount, blockCount, cellCount }
    | _ => .error "invalid ring assignment family shape"
  decode_encode := by
    intro shape
    cases shape
    rfl

structure RingQuotientRecipe where
  ringDegree : Nat
  quotientCount : Nat
  familyShapes : List RingFamilyShape
  challengeBlock : BlockKind
  challengeSlotBase : Nat
  challengeSourceStride : Nat
  challengeShift : Nat
  valueSources : List AffineRuns.Run
  quotientOutputBlock : BlockKind
deriving Repr, DecidableEq

def RingQuotientRecipe.format : Format RingQuotientRecipe where
  encode := fun recipe => .array [
    .atom recipe.ringDegree,
    .atom recipe.quotientCount,
    (Codec.list RingFamilyShape.format).encode recipe.familyShapes,
    BlockKind.format.encode recipe.challengeBlock,
    .atom recipe.challengeSlotBase,
    .atom recipe.challengeSourceStride,
    .atom recipe.challengeShift,
    AffineRuns.format.encode recipe.valueSources,
    BlockKind.format.encode recipe.quotientOutputBlock]
  decode
    | .array [.atom ringDegree, .atom quotientCount, familyShapes, challengeBlock,
        .atom challengeSlotBase, .atom challengeSourceStride,
        .atom challengeShift, valueSources, quotientOutputBlock] => do
      pure {
        ringDegree, quotientCount,
        familyShapes := ← (Codec.list RingFamilyShape.format).decode familyShapes,
        challengeBlock := ← BlockKind.format.decode challengeBlock,
        challengeSlotBase, challengeSourceStride, challengeShift,
        valueSources := ← AffineRuns.format.decode valueSources,
        quotientOutputBlock := ← BlockKind.format.decode quotientOutputBlock }
    | _ => .error "invalid ring assignment quotient recipe"
  decode_encode := by
    intro recipe
    cases recipe
    simp only [(Codec.list RingFamilyShape.format).decode_encode,
      BlockKind.format.decode_encode, AffineRuns.decode_encode]
    rfl

def ringFamilyShapes : List RingFamilyShape :=
  [ ⟨17, 18, 1⟩
  , ⟨17, 5, 1⟩
  , ⟨17, 1, 2⟩
  , ⟨17, 4, 2⟩ ]

def ringValueSources (program : Program) : List AffineRuns.Run :=
  AffineRuns.compress <| List.ofFn fun invocation :
      Fin PiRLCProductSchedule.invocationCount =>
    let descriptor := PiRLCProductSchedule.descriptor invocation
    (PiRLCProductPlan.valueColumn program descriptor descriptor.lane).val

def directRingValueSources (program : Program) : List AffineRuns.Run :=
  AffineRuns.compressIndexedTR fun invocation :
      Fin PiRLCProductSchedule.invocationCount =>
    let descriptor := PiRLCProductSchedule.descriptor invocation
    (PiRLCProductPlan.valueColumn program descriptor descriptor.lane).val

@[csimp] theorem ringValueSources_eq_direct :
    ringValueSources = directRingValueSources := by
  funext program
  exact (AffineRuns.compressIndexedTR_eq_compress_ofFn _).symm

theorem ringValueSources_at (program : Program)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    AffineRuns.sourceAt (ringValueSources program) invocation.val =
      (PiRLCProductPlan.valueColumn program
        (PiRLCProductSchedule.descriptor invocation)
        (PiRLCProductSchedule.descriptor invocation).lane).val := by
  rw [AffineRuns.sourceAt_eq_expand_getD, ringValueSources,
    AffineRuns.expand_compress]
  exact Lifecycle.PriorStateHash.ofFn_getD _ invocation 0

def ringQuotientRecipe (program : Program) : RingQuotientRecipe where
  ringDegree := Spec.ringDegree
  quotientCount := Spec.ringDegree
  familyShapes := ringFamilyShapes
  challengeBlock := .challengeWords
  challengeSlotBase := 0
  challengeSourceStride := Spec.ringDegree
  challengeShift := 2
  valueSources := ringValueSources program
  quotientOutputBlock := .productGroup

def physicalExpr (program : Program) (expression : Expr) : Expr :=
  CompactRows.renameExpr (PerApplicationPackage.shiftColumn program) <|
    PermutationOutput.Readout.rewriteExpr PiCCSTranscriptReadout.transcriptStart
      NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport.transcriptInvocationCount <|
        CompactRows.renameExpr NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan expression

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

def schema : Nat := 4

structure Plan where
  blocks : List PerApplicationAssignmentBlocks.BlockPlan
  ringQuotient : RingQuotientRecipe
  outputDigestBlock : BlockKind
  outputDigestExpressions : List Expr
deriving Repr, DecidableEq

def Plan.format : Format Plan where
  encode := fun plan => .array [
    .atom schema,
    PerApplicationAssignmentBlocks.format.encode plan.blocks,
    RingQuotientRecipe.format.encode plan.ringQuotient,
    BlockKind.format.encode plan.outputDigestBlock,
    (Codec.list NightstreamFPrime.Export.Package.exprFormat).encode
      plan.outputDigestExpressions]
  decode
    | .array [.atom 4, blocks, ringQuotient,
        outputDigestBlock, outputDigestExpressions] => do
      pure {
        blocks := ← PerApplicationAssignmentBlocks.format.decode blocks,
        ringQuotient := ← RingQuotientRecipe.format.decode ringQuotient,
        outputDigestBlock := ← BlockKind.format.decode outputDigestBlock,
        outputDigestExpressions :=
          ← (Codec.list NightstreamFPrime.Export.Package.exprFormat).decode
            outputDigestExpressions }
    | _ => .error "invalid per-application assignment transport plan"
  decode_encode := by
    intro plan
    cases plan
    simp only [schema, PerApplicationAssignmentBlocks.format.decode_encode,
      RingQuotientRecipe.format.decode_encode,
      BlockKind.format.decode_encode,
      (Codec.list NightstreamFPrime.Export.Package.exprFormat).decode_encode]
    rfl

def canonical (program : Program) : Plan where
  blocks := PerApplicationAssignmentBlocks.canonical program
  ringQuotient := ringQuotientRecipe program
  outputDigestBlock := .pilotOutputDigest
  outputDigestExpressions := outputDigestExpressions program

@[simp] theorem canonical_blocks_length (program : Program) :
    (canonical program).blocks.length = 26 := by
  exact PerApplicationAssignmentBlocks.canonical_length program

@[simp] theorem canonical_outputDigestExpressions_length (program : Program) :
    (canonical program).outputDigestExpressions.length = 4 := by
  change (outputDigestExpressions program).length = 4
  exact outputDigestExpressions_length program

end NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
