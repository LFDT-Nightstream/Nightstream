import Nightstream.Implementation.R1CS.ProjectionProgram
import Nightstream.Implementation.R1CS.LinearOutputs
import Nightstream.Protocol.Terminal.CE

/-!
Contract: independent semantic layout for the direct terminal-CE circuit.

Generated artifacts provide only column ownership and exact rows. This module
decodes those columns into the witness, commitment, public projection,
evaluation point, ring evaluations, constant terms, and NC sidecar consumed by
`TerminalCE.ClaimHolds`. No generated field states that the claim is valid.
-/

namespace Nightstream.Implementation.R1CS.TerminalCeCompiler

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionProgram

/-- Fixed host-validated shape and input columns of one terminal CE claim. -/
structure Layout where
  normBound : Nat
  expectedPublicWidth : Option Nat
  structureRows : Nat
  structureColumns : Nat
  witnessRows : Nat
  witnessColumns : Nat
  witnessCols : List Nat
  normFirstAllocatedColumn : Nat
  commitmentCols : List Nat
  commitmentD : Nat
  commitmentKappa : Nat
  publicCols : List Nat
  publicRows : Nat
  publicWidth : Nat
  publicInputLen : Nat
  pointCols : List KColumns
  evaluationCols : List (List Nat)
  constantTermCols : List KColumns
  ncPointCols : List KColumns
  ncEvaluationCols : List Nat
  ncEvaluationLanes : Nat
deriving DecidableEq, Repr

/-- Shape conditions checked by Rust before emitting any authoritative rows. -/
structure ShapeValid (layout : Layout) : Prop where
  witnessSize :
    layout.witnessCols.length = layout.witnessRows * layout.witnessColumns
  commitmentSize :
    layout.commitmentCols.length = layout.commitmentD * layout.commitmentKappa
  publicSize : layout.publicCols.length = layout.publicRows * layout.publicWidth
  publicRowsPositive : 0 < layout.publicRows
  publicProjectionWithinStructure :
    layout.publicInputLen ≤ layout.structureColumns
  publicWidthPinned :
    match layout.expectedPublicWidth with
    | none => True
    | some width => layout.publicInputLen = width
  constantTermSize :
    layout.constantTermCols.length = layout.evaluationCols.length
  evaluationRowsNonempty :
    ∀ row ∈ layout.evaluationCols, 2 ≤ row.length
  evaluationRowsEven :
    ∀ row ∈ layout.evaluationCols, row.length % 2 = 0
  ncEvaluationSize :
    layout.ncEvaluationCols.length = 2 * layout.ncEvaluationLanes

abbrev F := ProjectionProgram.F
abbrev K := ProjectionProgram.K

def fieldAt (assignment : Nat → Nat) (column : Nat) : F :=
  residue (assignment column)

def kAt (assignment : Nat → Nat) (columns : KColumns) : K :=
  columns.value assignment

def valuesAt (assignment : Nat → Nat) (columns : List Nat) : List F :=
  columns.map (fieldAt assignment)

def kValuesAt (assignment : Nat → Nat) (columns : List KColumns) : List K :=
  columns.map (kAt assignment)

/-- Row-major packed terminal witness. -/
structure Witness where
  rows : Nat
  columns : Nat
  values : List F
deriving DecidableEq, Repr

/-- Implementation-side NC channel that is accumulator-digest authority. -/
structure Sidecar where
  point : List K
  evaluations : List K
deriving DecidableEq, Repr

/-- Verifier-owned structure shape for this fixed generated profile. -/
structure Structure where
  rows : Nat
  columns : Nat
deriving DecidableEq, Repr

def decodeWitness (layout : Layout) (assignment : Nat → Nat) : Witness where
  rows := layout.witnessRows
  columns := layout.witnessColumns
  values := valuesAt assignment layout.witnessCols

def decodeCommitment (layout : Layout) (assignment : Nat → Nat) : List F :=
  valuesAt assignment layout.commitmentCols

def requiredPublicColumns (layout : Layout) : Nat :=
  (layout.publicInputLen + layout.publicRows - 1) / layout.publicRows

def publicColumn (layout : Layout) (row column : Nat) : Nat :=
  layout.publicCols.getD (row * layout.publicWidth + column) 0

def witnessColumn (layout : Layout) (row column : Nat) : Nat :=
  layout.witnessCols.getD (row * layout.witnessColumns + column) 0

/-- Canonical verifier traversal is column-major even though storage is row-major. -/
def publicOutputColumns (layout : Layout) : List Nat :=
  (List.range layout.publicWidth).flatMap fun column =>
    (List.range layout.publicRows).map fun row =>
      publicColumn layout row column

def decodePublicInput (layout : Layout) (assignment : Nat → Nat) : List F :=
  valuesAt assignment (publicOutputColumns layout)

def decodePoint (layout : Layout) (assignment : Nat → Nat) : List K :=
  kValuesAt assignment layout.pointCols

def pairs (values : List F) : List K :=
  match values with
  | c0 :: c1 :: tail => ⟨c0, c1⟩ :: pairs tail
  | _ => []

def decodeEvaluations (layout : Layout) (assignment : Nat → Nat) : List (List K) :=
  layout.evaluationCols.map fun row => pairs (valuesAt assignment row)

def decodeConstantTerms (layout : Layout) (assignment : Nat → Nat) : List K :=
  kValuesAt assignment layout.constantTermCols

def decodeSidecar (layout : Layout) (assignment : Nat → Nat) : Sidecar where
  point := kValuesAt assignment layout.ncPointCols
  evaluations := pairs (valuesAt assignment layout.ncEvaluationCols)

def context (layout : Layout) : Nightstream.Protocol.TerminalCE.Context Structure where
  relation := ⟨layout.structureRows, layout.structureColumns⟩
  normBound := layout.normBound
  expectedPublicWidth := layout.expectedPublicWidth

def claim (layout : Layout) (assignment : Nat → Nat) :
    Nightstream.Protocol.TerminalCE.Claim
      (List F) (List K) (List K) (List F) K Sidecar where
  commitment := decodeCommitment layout assignment
  publicWidth := layout.publicInputLen
  publicInput := decodePublicInput layout assignment
  point := decodePoint layout assignment
  evaluations := decodeEvaluations layout assignment
  constantTerms := decodeConstantTerms layout assignment
  sidecar := decodeSidecar layout assignment

/-! ## Exact checked-program compiler view -/

structure Schedule where
  commitmentStart : Nat
  commitmentEnd : Nat
  publicInputStart : Nat
  publicInputEnd : Nat
  normStart : Nat
  normEnd : Nat
  evaluationsStart : Nat
  evaluationsEnd : Nat
  constantTermStart : Nat
  constantTermEnd : Nat
  ncChannelStart : Nat
  ncChannelEnd : Nat
deriving DecidableEq, Repr

def instructionSlice (instructions : List Instruction) (start finish : Nat) :
    List Instruction :=
  (instructions.drop start).take (finish - start)

def decodeOutputCheck (output : Nat) (row : Row) : LinearOutputs.Check :=
  if row.a.head? = some (output, 1) then
    ⟨output, negateTerms row.a.tail, .forward⟩
  else
    ⟨output, row.a.dropLast, .reverse⟩

def outputChecks (outputs : List Nat) (instructions : List Instruction) :
    List LinearOutputs.Check :=
  (outputs.zip (checks instructions)).map fun pair =>
    decodeOutputCheck pair.1 pair.2

def projectionChecks (layout : Layout) : List LinearOutputs.Check :=
  (List.range layout.publicWidth).flatMap fun column =>
    (List.range layout.publicRows).map fun row =>
      ⟨publicColumn layout row column,
        if column < requiredPublicColumns layout then
          [(witnessColumn layout row column, 1)] else [],
        if column < requiredPublicColumns layout then
          .reverse else .forward⟩

def projectedPublic (layout : Layout) (assignment : Nat → Nat) : List F :=
  (projectionChecks layout).map fun check =>
    residue (check.expected assignment)

def centeredUnitInstructions (column output : Nat) : List Instruction :=
  [.define ⟨output,
      .product [(column, 1), (0, 1)] [(column, 1)]⟩,
   .check ⟨[(output, 1)],
      [(column, 1), (0, goldilocksP - 1)], []⟩]

/-- Exact `b = 2` direct-CE norm compiler used by the supported profile. -/
def normInstructionsFrom : Nat → List Nat → List Instruction
  | _, [] => []
  | output, column :: tail =>
      centeredUnitInstructions column output ++
        normInstructionsFrom (output + 1) tail

def normInstructions (layout : Layout) : List Instruction :=
  normInstructionsFrom layout.normFirstAllocatedColumn layout.witnessCols

def CenteredUnit (value : Nat) : Prop :=
  value = 0 ∨ value = 1 ∨ value = goldilocksP - 1

def NormHolds (layout : Layout) (assignment : Nat → Nat) : Prop :=
  ∀ column ∈ layout.witnessCols, CenteredUnit (assignment column)

def checkCenteredUnit (value : Nat) : Bool :=
  decide (value = 0) || decide (value = 1) ||
    decide (value = goldilocksP - 1)

def checkNorm (layout : Layout) (assignment : Nat → Nat) : Bool :=
  layout.witnessCols.all fun column => checkCenteredUnit (assignment column)

def flattenKColumns (columns : List KColumns) : List Nat :=
  columns.flatMap fun value => [value.c0, value.c1]

def semanticColumns (layout : Layout) : List Nat :=
  layout.witnessCols ++ layout.commitmentCols ++ layout.publicCols ++
    flattenKColumns layout.pointCols ++ layout.evaluationCols.flatten ++
    flattenKColumns layout.constantTermCols ++
    flattenKColumns layout.ncPointCols ++ layout.ncEvaluationCols

def constantTermChecks (layout : Layout) : List LinearOutputs.Check :=
  (layout.constantTermCols.zip layout.evaluationCols).flatMap fun pair =>
    [⟨pair.1.c0, [(pair.2.getD 0 0, 1)], .forward⟩,
     ⟨pair.1.c1, [(pair.2.getD 1 0, 1)], .forward⟩]

structure Program where
  layout : Layout
  schedule : Schedule
  inputColumns : List Nat
  instructions : List Instruction

namespace Program

def definitions (program : Program) : List Definition :=
  CheckedProgram.definitions program.instructions

def final (program : Program) (assignment : Nat → Nat) : Nat → Nat :=
  run assignment program.definitions

def commitmentInstructions (program : Program) : List Instruction :=
  instructionSlice program.instructions program.schedule.commitmentStart
    program.schedule.commitmentEnd

def publicInstructions (program : Program) : List Instruction :=
  instructionSlice program.instructions program.schedule.publicInputStart
    program.schedule.publicInputEnd

def normInstructionsSlice (program : Program) : List Instruction :=
  instructionSlice program.instructions program.schedule.normStart
    program.schedule.normEnd

def evaluationInstructions (program : Program) : List Instruction :=
  instructionSlice program.instructions program.schedule.evaluationsStart
    program.schedule.evaluationsEnd

def constantTermInstructions (program : Program) : List Instruction :=
  instructionSlice program.instructions program.schedule.constantTermStart
    program.schedule.constantTermEnd

def ncInstructions (program : Program) : List Instruction :=
  instructionSlice program.instructions program.schedule.ncChannelStart
    program.schedule.ncChannelEnd

def commitmentChecks (program : Program) : List LinearOutputs.Check :=
  outputChecks program.layout.commitmentCols program.commitmentInstructions

def evaluationChecks (program : Program) : List LinearOutputs.Check :=
  outputChecks program.layout.evaluationCols.flatten program.evaluationInstructions

def ncChecks (program : Program) : List LinearOutputs.Check :=
  outputChecks program.layout.ncEvaluationCols program.ncInstructions

def expectedFields (program : Program) (assignment : Nat → Nat)
    (outputChecks : Program → List LinearOutputs.Check) : List F :=
  (outputChecks program).map fun check =>
    residue (check.expected (program.final assignment))

def expectedCommitment (program : Program) (assignment : Nat → Nat) : List F :=
  expectedFields program assignment commitmentChecks

def splitByLengths : List Nat → List α → List (List α)
  | [], _ => []
  | length :: lengths, values =>
      values.take length :: splitByLengths lengths (values.drop length)

def expectedEvaluations (program : Program) (assignment : Nat → Nat) :
    List (List K) :=
  (splitByLengths (program.layout.evaluationCols.map List.length)
      (expectedFields program assignment evaluationChecks)).map pairs

def expectedNcEvaluations (program : Program) (assignment : Nat → Nat) :
    List K :=
  pairs (expectedFields program assignment ncChecks)

def semantics (program : Program) :
    Nightstream.Protocol.TerminalCE.Semantics
      Structure (Nat → Nat) (List F) (List K) (List K) (List F) K Sidecar where
  commit := expectedCommitment program
  projectPublicInput := fun width assignment =>
    if width = program.layout.publicInputLen then
      some (projectedPublic program.layout assignment)
    else none
  normBounded := fun bound assignment =>
    decide (bound = program.layout.normBound) &&
      checkNorm program.layout assignment
  evaluationPointValid := fun relation point =>
    decide (relation = (context program.layout).relation) &&
      decide (point.length = program.layout.pointCols.length)
  evaluations := fun relation assignment point =>
    if relation = (context program.layout).relation ∧
        point.length = program.layout.pointCols.length then
      some (expectedEvaluations program assignment)
    else none
  constantTerm := fun evaluation => evaluation.headD K.zero
  sidecarValid := fun relation assignment sidecar =>
    decide (relation = (context program.layout).relation) &&
      decide (sidecar.point.length = program.layout.ncPointCols.length) &&
      decide (sidecar.evaluations = expectedNcEvaluations program assignment)

end Program

def ClaimHolds (program : Program) (assignment : Nat → Nat) : Prop :=
  Nightstream.Protocol.TerminalCE.ClaimHolds program.semantics
    (context program.layout) (claim program.layout assignment) assignment

end Nightstream.Implementation.R1CS.TerminalCeCompiler
