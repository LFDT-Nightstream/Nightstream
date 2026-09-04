import NightstreamFPrime.Export.Package
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentBlocks
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPreservation

/-!
Owns the sealed executable transport for the final 14-matrix assignment.
The transport keeps the existing 45 retained block plans and adds only the
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
  valueBlock : BlockKind
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
    BlockKind.format.encode recipe.valueBlock,
    BlockKind.format.encode recipe.groupOutputBlock]
  decode
    | .array [.atom ringDegree, .atom middleDegree, .atom foldOffset,
        .atom twiceCutoff, .atom rawConvolutionCount, .atom rawTermCount,
        .atom groupWidth, .atom groupCount, familyShapes, challengeBlock,
        .atom challengeSlotBase, .atom challengeSourceStride,
        .atom challengeShift, valueBlock, groupOutputBlock] => do
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
        valueBlock := ← BlockKind.format.decode valueBlock,
        groupOutputBlock := ← BlockKind.format.decode groupOutputBlock }
    | _ => .error "invalid Phi81 assignment group recipe"
  decode_encode := by
    intro recipe
    cases recipe
    simp only [(Codec.list Phi81FamilyShape.format).decode_encode,
      BlockKind.format.decode_encode]
    rfl

/-- The four fixed product families in semantic order. -/
def phi81FamilyShapes : List Phi81FamilyShape :=
  [ ⟨17, 22, 1⟩
  , ⟨17, 5, 1⟩
  , ⟨17, 1, 2⟩
  , ⟨17, 14, 2⟩ ]

/-- The complete generic Phi81 recipe. No per-invocation expressions are
materialized. -/
def phi81GroupRecipe : Phi81GroupRecipe where
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
  valueBlock := .productInput
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

/-- Map an original Stage 1 source column to the final physical package
assignment. This is the same two-step pullback used by the Lean semantics. -/
def physicalColumn (program : Program) (column : Nat) : Nat :=
  PerApplicationPackage.shiftColumn program
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan column)

def physicalExpr (program : Program) (expression : Expr) : Expr :=
  CompactRows.renameExpr (physicalColumn program) expression

/-- Invocation-major, rate-lane-major PiCCS payload recipes. -/
def payloadExpressions (program : Program) : List Expr :=
  List.ofFn fun index : Fin PiCCSActionPayloadBlock.payloadCount =>
    physicalExpr program (PiCCSActionPayloadBlock.payloadExpression index)

@[simp] theorem payloadExpressions_length (program : Program) :
    (payloadExpressions program).length = 30416 := by
  rw [payloadExpressions, List.length_ofFn]
  exact PiCCSActionPayloadBlock.payloadCount_eq

/-- Materialize each action expansion once. In particular, the PiCCS output
absorb is chunked once instead of once per random payload lookup. -/
def materializedPayloadKinds (_delay : Unit := ()) :
    List PoseidonActionSchedule.Kind :=
  PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.statementActions ++
    (PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.challengeActions ++
      (PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.roundActions ++
        PoseidonActionSchedule.kinds PiCCSActionPayloadBlock.outputActions))

theorem materializedPayloadKinds_eq :
    materializedPayloadKinds () =
      List.ofFn PiCCSActionPayloadBlock.kindAt := by
  simpa only [materializedPayloadKinds] using
    PiCCSActionPayloadBlock.kindAt_materializes.symm

/-- Linear-time payload materialization for the package emitter. -/
def materializedPayloadExpressions (program : Program) : List Expr :=
  List.flatten <| (materializedPayloadKinds ()).map fun kind =>
    List.ofFn fun lane : Fin Spec.Poseidon2.rate =>
      physicalExpr program
        (PiCCSActionPayloadBlock.payloadExprForKind kind lane)

private theorem ofFn_decodeProd_eq_nested {Alpha : Type}
    (m n : Nat) (value : Fin m → Fin n → Alpha) :
    List.ofFn (fun index : Fin (m * n) =>
      let decoded : Fin m × Fin n := Fin.decodeProd index
      value decoded.1 decoded.2) =
      List.flatten (List.ofFn fun outer : Fin m =>
        List.ofFn fun inner : Fin n => value outer inner) := by
  rw [List.ofFn_mul]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext outer
  apply congrArg List.ofFn
  funext inner
  let combined : Fin (m * n) :=
    ⟨outer.val * n + inner.val, by
      calc
        outer.val * n + inner.val < (outer.val + 1) * n := by
          simpa [Nat.add_mul] using
            Nat.add_lt_add_left inner.isLt (outer.val * n)
        _ ≤ m * n := Nat.mul_le_mul_right n outer.isLt⟩
  change value (Fin.decodeProd combined).1 (Fin.decodeProd combined).2 =
    value outer inner
  have combined_eq : combined = Fin.encodeProd (outer, inner) := by
    apply Fin.ext
    simp [combined, Fin.encodeProd, Nat.mul_comm]
  rw [combined_eq, Fin.decodeProd_encodeProd]

/-- The emitter's linear materialization is exactly the canonical
random-access payload-expression list. -/
theorem materializedPayloadExpressions_eq (program : Program) :
    materializedPayloadExpressions program = payloadExpressions program := by
  unfold materializedPayloadExpressions
  rw [materializedPayloadKinds_eq]
  rw [← List.ofFn_comp']
  symm
  unfold payloadExpressions
  simpa only [PiCCSActionPayloadBlock.payloadCount,
    PiCCSActionPayloadBlock.payloadExpression,
    PiCCSActionPayloadBlock.payloadExpr] using
    (ofFn_decodeProd_eq_nested PiCCSActionPayloadBlock.invocationCount
      Spec.Poseidon2.rate
      (fun invocation lane => physicalExpr program
        (PiCCSActionPayloadBlock.payloadExprForKind
          (PiCCSActionPayloadBlock.kindAt invocation) lane)))

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
  phi81 := phi81GroupRecipe
  first54 := first54ProductRecipe
  payloadBlock := .piCcsPayload
  payloadExpressions := payloadExpressions program
  outputDigestBlock := .pilotOutputDigest
  outputDigestExpressions := outputDigestExpressions program

@[simp] theorem canonical_blocks_length (program : Program) :
    (canonical program).blocks.length = 45 := by
  exact PerApplicationAssignmentBlocks.canonical_length program

@[simp] theorem canonical_outputDigestExpressions_length (program : Program) :
    (canonical program).outputDigestExpressions.length = 4 := by
  change (outputDigestExpressions program).length = 4
  exact outputDigestExpressions_length program

end NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
