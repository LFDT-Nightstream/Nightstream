import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Terminal
import Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
import Nightstream.Implementation.Lowering.Goldilocks.InstructionReceipts

/-!
Contract: deterministic physical placement and proof-free manifest for the
selected SuperNeo terminal R1CS.

Assurance tier: model-level.

Owns: separate physical owners for fourteen running claims and one fresh
claim; exact placement of committed assignments, public statements, and
auxiliary columns; exact receipt construction; and proof-free manifest
serialization.

Does not own: terminal statement values, private witnesses, a selected
benchmark setup, assignment construction, Spartan, WHIR, Rust, or Ajtai
binding security.

The running relation and its evaluation points are verifier-owned setup data.
They determine row coefficients. A proof cannot select them.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Layout

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev RelationShape
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length) :=
  NativeCcsPhi81.shape program domain publicRingColumns publicFits

/-- A short structural path for terminal claim `index`. -/
def claimPath : Nat → OwnerPath
  | 0 => .root
  | index + 1 => .rest (claimPath index)

/-- Each running child has one structural receipt owner. -/
def runningOwner (child : Fin productionGlobalParams.k) : PhysicalOwner :=
  .typed (.instruction (claimPath child.val))

/-- The fresh claim follows all running claims. -/
def freshOwner : PhysicalOwner :=
  .typed (.instruction (claimPath productionGlobalParams.k))

/-- One local coordinate under a claim owner. -/
def localColumn (owner : PhysicalOwner) (coordinate : Nat) : ColumnId :=
  { owner := owner, bundleIndex := 0, coordinateIndex := coordinate }

/-- Number of public statement coordinates for one running CE claim. -/
def runningStatementWidth
    (shape : Phi81Relation.Shape) (verifierRows : Nat) : Nat :=
  verifierRows * ringDegree + shape.publicWidth +
    2 * (shape.matrixCount * ringDegree)

/-- Number of input coordinates for one running CE claim. -/
def runningInputWidth
    (shape : Phi81Relation.Shape) (verifierRows : Nat) : Nat :=
  shape.carrierWidth + runningStatementWidth shape verifierRows

/-- Total allocated coordinates for one running CE claim. -/
def runningWidth
    (shape : Phi81Relation.Shape) (verifierRows : Nat) : Nat :=
  runningInputWidth shape verifierRows + shape.carrierWidth

/-- Number of public statement coordinates for the fresh CCS claim. -/
def freshStatementWidth
    (shape : Phi81Relation.Shape) (verifierRows : Nat) : Nat :=
  verifierRows * ringDegree + shape.publicWidth

/-- Number of input coordinates for the fresh CCS claim. -/
def freshInputWidth
    (shape : Phi81Relation.Shape) (verifierRows : Nat) : Nat :=
  shape.carrierWidth + freshStatementWidth shape verifierRows

/-- Total allocated coordinates for the fresh CCS claim. -/
def freshWidth
    (program : NativeCcsProgram.Program)
    (shape : Phi81Relation.Shape) (verifierRows : Nat) : Nat :=
  freshInputWidth shape verifierRows + shape.carrierWidth +
    program.rows.length

/-- A contiguous owned column block. -/
def columnBlock
    (owner : PhysicalOwner)
    (start count : Nat)
    (ownership : Ownership) : List OwnedColumn :=
  List.ofFn fun coordinate : Fin count =>
    { id := localColumn owner (start + coordinate.val)
      ownership := ownership }

@[simp] theorem columnBlock_length
    (owner : PhysicalOwner)
    (start count : Nat)
    (ownership : Ownership) :
    (columnBlock owner start count ownership).length = count := by
  simp [columnBlock]

theorem columnBlock_owned
    (owner : PhysicalOwner)
    (start count : Nat)
    (ownership : Ownership)
    (column : OwnedColumn)
    (member : column ∈ columnBlock owner start count ownership) :
    column.id.owner = owner := by
  rcases List.mem_ofFn.mp member with ⟨coordinate, rfl⟩
  rfl

/-- Committed assignment, public CE statement, then norm squares. -/
def runningAllocations
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat)
    (child : Fin productionGlobalParams.k) : List OwnedColumn :=
  let owner := runningOwner child
  columnBlock owner 0 shape.carrierWidth .committedColumn ++
    columnBlock owner shape.carrierWidth
      (runningStatementWidth shape verifierRows) .publicColumn ++
    columnBlock owner (runningInputWidth shape verifierRows)
      shape.carrierWidth .auxiliaryColumn

@[simp] theorem runningAllocations_length
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat)
    (child : Fin productionGlobalParams.k) :
    (runningAllocations shape verifierRows child).length =
      runningWidth shape verifierRows := by
  simp [runningAllocations, runningWidth, runningInputWidth]
  omega

theorem runningAllocations_owned
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat)
    (child : Fin productionGlobalParams.k)
    (column : OwnedColumn)
    (member : column ∈ runningAllocations shape verifierRows child) :
    column.id.owner = runningOwner child := by
  simp only [runningAllocations, List.mem_append] at member
  rcases member with (committed | publicMember) | auxiliary
  · exact columnBlock_owned _ _ _ _ _ committed
  · exact columnBlock_owned _ _ _ _ _ publicMember
  · exact columnBlock_owned _ _ _ _ _ auxiliary

/-- Committed assignment, public CCS statement, squares, then residuals. -/
def freshAllocations
    (program : NativeCcsProgram.Program)
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat) : List OwnedColumn :=
  columnBlock freshOwner 0 shape.carrierWidth .committedColumn ++
    columnBlock freshOwner shape.carrierWidth
      (freshStatementWidth shape verifierRows) .publicColumn ++
    (columnBlock freshOwner (freshInputWidth shape verifierRows)
        shape.carrierWidth .auxiliaryColumn ++
      columnBlock freshOwner
        (freshInputWidth shape verifierRows + shape.carrierWidth)
        program.rows.length .auxiliaryColumn)

@[simp] theorem freshAllocations_length
    (program : NativeCcsProgram.Program)
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat) :
    (freshAllocations program shape verifierRows).length =
      freshWidth program shape verifierRows := by
  simp [freshAllocations, freshWidth, freshInputWidth]
  omega

theorem freshAllocations_owned
    (program : NativeCcsProgram.Program)
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat)
    (column : OwnedColumn)
    (member : column ∈ freshAllocations program shape verifierRows) :
    column.id.owner = freshOwner := by
  simp only [freshAllocations, List.mem_append] at member
  rcases member with (committed | publicMember) | (square | residual)
  · exact columnBlock_owned _ _ _ _ _ committed
  · exact columnBlock_owned _ _ _ _ _ publicMember
  · exact columnBlock_owned _ _ _ _ _ square
  · exact columnBlock_owned _ _ _ _ _ residual

/-- Deterministic frame for one running CE claim. -/
def runningFrame
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows)
    (child : Fin productionGlobalParams.k) :
    Running.Frame
      (RelationShape program domain publicRingColumns publicFits)
      verifierRows :=
  let shape := RelationShape program domain publicRingColumns publicFits
  let owner := runningOwner child
  {
    owner := owner
    firstOrdinal := 0
    one := oneColumn
    key := key
    witness := fun coordinate =>
      localColumn owner coordinate.val
    commitment := fun verifierRow output =>
      localColumn owner
        (shape.carrierWidth +
          verifierRow.val * ringDegree + output.val)
    publicColumn := fun coordinate =>
      localColumn owner
        (shape.carrierWidth + verifierRows * ringDegree + coordinate.val)
    evaluationLow := fun matrix lane =>
      localColumn owner
        (shape.carrierWidth + verifierRows * ringDegree +
          shape.publicWidth + matrix.val * ringDegree + lane.val)
    evaluationHigh := fun matrix lane =>
      localColumn owner
        (shape.carrierWidth + verifierRows * ringDegree +
          shape.publicWidth + shape.matrixCount * ringDegree +
          matrix.val * ringDegree + lane.val)
    square := fun coordinate =>
      localColumn owner
        (runningInputWidth shape verifierRows + coordinate.val)
  }

/-- Deterministic frame for the fresh CCS claim. -/
def freshFrame
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows) :
    Fresh.Frame program domain publicRingColumns publicFits verifierRows :=
  let shape := RelationShape program domain publicRingColumns publicFits
  {
    owner := freshOwner
    firstOrdinal := 0
    one := oneColumn
    key := key
    witness := fun coordinate =>
      localColumn freshOwner coordinate.val
    commitment := fun verifierRow output =>
      localColumn freshOwner
        (shape.carrierWidth +
          verifierRow.val * ringDegree + output.val)
    publicColumn := fun coordinate =>
      localColumn freshOwner
        (shape.carrierWidth + verifierRows * ringDegree + coordinate.val)
    square := fun coordinate =>
      localColumn freshOwner
        (freshInputWidth shape verifierRows + coordinate.val)
    residual := fun source =>
      localColumn freshOwner
        (freshInputWidth shape verifierRows +
          shape.carrierWidth + source.val)
  }

/-- Complete deterministic terminal frame. -/
def frame
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows) :
    Terminal.Frame program domain publicRingColumns publicFits verifierRows where
  running := runningFrame key
  fresh := freshFrame key

/-- One receipt owns one complete running claim. -/
def runningReceipt
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows)
    (child : Fin productionGlobalParams.k) :
    InstructionReceipt where
  owner := runningOwner child
  kind := .call
  allocations :=
    runningAllocations
      (RelationShape program domain publicRingColumns publicFits)
      verifierRows child
  rows := Running.rows (runningFrame key child) (statements child)
  allocationsOwned := by
    intro column member
    exact runningAllocations_owned _ _ child column member
  rowsOwned := by
    intro row member
    exact Running.rows_owned (runningFrame key child)
      (statements child) row member

/-- One receipt owns the complete fresh claim. -/
def freshReceipt
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows) :
    InstructionReceipt where
  owner := freshOwner
  kind := .call
  allocations :=
    freshAllocations program
      (RelationShape program domain publicRingColumns publicFits)
      verifierRows
  rows := Fresh.rows valid (freshFrame key)
  allocationsOwned := by
    intro column member
    exact freshAllocations_owned _ _ _ column member
  rowsOwned := by
    intro row member
    exact Fresh.rows_owned valid (freshFrame key) row member

/-- Exact receipt order: public one, fourteen running claims, then fresh. -/
def receipts
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows) :
    List InstructionReceipt :=
  InstructionReceipt.prelude ::
    ((List.finRange productionGlobalParams.k).map
      (runningReceipt key statements) ++ [freshReceipt valid key])

/-- Proof-free terminal program emitted directly from the exact receipts. -/
def program
    {source : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain source}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth source.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid source)
    (key :
      Commitment.Key
        (RelationShape source domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements source domain publicRingColumns publicFits
        verifierRows) :
    CanonicalManifest.Program where
  one := oneColumn
  receipts :=
    (receipts valid key statements).map
      CanonicalManifest.ManifestReceipt.ofReceipt

@[simp] theorem receipts_length
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows) :
    (receipts valid key statements).length =
      productionGlobalParams.k + 2 := by
  simp [receipts]

@[simp] theorem program_one
    {source : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain source}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth source.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid source)
    (key :
      Commitment.Key
        (RelationShape source domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements source domain publicRingColumns publicFits
        verifierRows) :
    (program valid key statements).one = oneColumn :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Layout
