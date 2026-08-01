import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Physical

/-!
Contract: receipt-order scoping for the selected SuperNeo terminal R1CS.

Assurance tier: model-level.

Owns: proof that every terminal row reads only the public constant-one
column or columns allocated by its own claim receipt, and construction of the
exact receipt-conserved physical program.

Does not own: terminal witness values, honest assignment construction,
semantic soundness, a selected benchmark statement, Spartan, WHIR, or Rust.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Scoping

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
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

private theorem localColumn_mem_block
    (owner : PhysicalOwner)
    (start count coordinate : Nat)
    (ownership : Ownership)
    (lower : start ≤ coordinate)
    (upper : coordinate < start + count) :
    Layout.localColumn owner coordinate ∈
      (Layout.columnBlock owner start count ownership).map
        fun column => column.id := by
  apply List.mem_map.mpr
  let position : Fin count :=
    ⟨coordinate - start, by omega⟩
  refine ⟨
    ({ id := Layout.localColumn owner (start + position.val)
       ownership := ownership } : OwnedColumn),
    List.mem_ofFn.mpr ⟨position, rfl⟩,
    ?_
  ⟩
  change
    Layout.localColumn owner (start + (coordinate - start)) =
      Layout.localColumn owner coordinate
  rw [Nat.add_sub_of_le lower]

private theorem runningLocalColumn_mem
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat)
    (child : Fin productionGlobalParams.k)
    (coordinate : Nat)
    (below : coordinate < Layout.runningWidth shape verifierRows) :
    Layout.localColumn (Layout.runningOwner child) coordinate ∈
      (Layout.runningAllocations shape verifierRows child).map
        fun column => column.id := by
  simp only [Layout.runningAllocations, List.map_append,
    List.mem_append]
  by_cases committed : coordinate < shape.carrierWidth
  · exact Or.inl (Or.inl
      (localColumn_mem_block _ 0 shape.carrierWidth coordinate
        .committedColumn (by omega) (by omega)))
  · by_cases statement :
      coordinate <
        shape.carrierWidth +
          Layout.runningStatementWidth shape verifierRows
    · exact Or.inl (Or.inr
        (localColumn_mem_block _ shape.carrierWidth
          (Layout.runningStatementWidth shape verifierRows) coordinate
          .publicColumn (by omega) statement))
    · exact Or.inr
        (localColumn_mem_block _
          (Layout.runningInputWidth shape verifierRows)
          shape.carrierWidth coordinate .auxiliaryColumn
          (by
            unfold Layout.runningInputWidth
            omega)
          below)

private theorem freshLocalColumn_mem
    (program : NativeCcsProgram.Program)
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat)
    (coordinate : Nat)
    (below : coordinate < Layout.freshWidth program shape verifierRows) :
    Layout.localColumn Layout.freshOwner coordinate ∈
      (Layout.freshAllocations program shape verifierRows).map
        fun column => column.id := by
  simp only [Layout.freshAllocations, List.map_append,
    List.mem_append]
  by_cases committed : coordinate < shape.carrierWidth
  · exact Or.inl (Or.inl
      (localColumn_mem_block _ 0 shape.carrierWidth coordinate
        .committedColumn (by omega) (by omega)))
  · by_cases statement :
      coordinate <
        shape.carrierWidth +
          Layout.freshStatementWidth shape verifierRows
    · exact Or.inl (Or.inr
        (localColumn_mem_block _ shape.carrierWidth
          (Layout.freshStatementWidth shape verifierRows) coordinate
          .publicColumn (by omega) statement))
    · by_cases square :
        coordinate <
          Layout.freshInputWidth shape verifierRows + shape.carrierWidth
      · exact Or.inr (Or.inl
          (localColumn_mem_block _
            (Layout.freshInputWidth shape verifierRows)
            shape.carrierWidth coordinate .auxiliaryColumn
            (by
              unfold Layout.freshInputWidth
              omega)
            square))
      · exact Or.inr (Or.inr
          (localColumn_mem_block _
            (Layout.freshInputWidth shape verifierRows + shape.carrierWidth)
            program.rows.length coordinate .auxiliaryColumn
            (by omega)
            below))

private theorem runningWitness_mem
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
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    (Layout.runningFrame key child).witness coordinate ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id := by
  change
    Layout.localColumn (Layout.runningOwner child) coordinate.val ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id
  apply runningLocalColumn_mem
  unfold Layout.runningWidth Layout.runningInputWidth
  omega

private theorem runningCommitment_mem
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
    (child : Fin productionGlobalParams.k)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    (Layout.runningFrame key child).commitment verifierRow output ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id := by
  change
    Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRow.val * ringDegree + output.val) ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id
  apply runningLocalColumn_mem
  unfold Layout.runningWidth Layout.runningInputWidth
    Layout.runningStatementWidth
  have pairLt := (Ajtai.pairIndex verifierRow output).isLt
  simp only [Ajtai.pairIndex] at pairLt
  omega

private theorem runningPublic_mem
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
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).publicWidth) :
    (Layout.runningFrame key child).publicColumn coordinate ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id := by
  change
    Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree + coordinate.val) ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id
  apply runningLocalColumn_mem
  unfold Layout.runningWidth Layout.runningInputWidth
    Layout.runningStatementWidth
  omega

private theorem runningEvaluationLow_mem
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
    (child : Fin productionGlobalParams.k)
    (matrix :
      Fin (RelationShape program domain publicRingColumns publicFits).matrixCount)
    (lane : Fin ringDegree) :
    (Layout.runningFrame key child).evaluationLow matrix lane ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id := by
  change
    Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree +
          (RelationShape program domain publicRingColumns publicFits).publicWidth +
          matrix.val * ringDegree + lane.val) ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id
  apply runningLocalColumn_mem
  unfold Layout.runningWidth Layout.runningInputWidth
    Layout.runningStatementWidth
  have pairLt := (Ajtai.pairIndex matrix lane).isLt
  simp only [Ajtai.pairIndex] at pairLt
  omega

private theorem runningEvaluationHigh_mem
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
    (child : Fin productionGlobalParams.k)
    (matrix :
      Fin (RelationShape program domain publicRingColumns publicFits).matrixCount)
    (lane : Fin ringDegree) :
    (Layout.runningFrame key child).evaluationHigh matrix lane ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id := by
  change
    Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree +
          (RelationShape program domain publicRingColumns publicFits).publicWidth +
          (RelationShape program domain publicRingColumns publicFits).matrixCount *
            ringDegree +
          matrix.val * ringDegree + lane.val) ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id
  apply runningLocalColumn_mem
  unfold Layout.runningWidth Layout.runningInputWidth
    Layout.runningStatementWidth
  have pairLt := (Ajtai.pairIndex matrix lane).isLt
  simp only [Ajtai.pairIndex] at pairLt
  omega

private theorem runningSquare_mem
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
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    (Layout.runningFrame key child).square coordinate ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id := by
  change
    Layout.localColumn (Layout.runningOwner child)
        (Layout.runningInputWidth
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows + coordinate.val) ∈
      (Layout.runningAllocations
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows child).map fun column => column.id
  apply runningLocalColumn_mem
  unfold Layout.runningWidth
  omega

private theorem freshWitness_mem
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
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    (Layout.freshFrame key).witness coordinate ∈
      (Layout.freshAllocations program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).map fun column => column.id := by
  change
    Layout.localColumn Layout.freshOwner coordinate.val ∈
      (Layout.freshAllocations program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).map fun column => column.id
  apply freshLocalColumn_mem
  unfold Layout.freshWidth Layout.freshInputWidth
  omega

private theorem freshCommitment_mem
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
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    (Layout.freshFrame key).commitment verifierRow output ∈
      (Layout.freshAllocations program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).map fun column => column.id := by
  change
    Layout.localColumn Layout.freshOwner
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRow.val * ringDegree + output.val) ∈
      (Layout.freshAllocations program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).map fun column => column.id
  apply freshLocalColumn_mem
  unfold Layout.freshWidth Layout.freshInputWidth
    Layout.freshStatementWidth
  have pairLt := (Ajtai.pairIndex verifierRow output).isLt
  simp only [Ajtai.pairIndex] at pairLt
  omega

private theorem freshPublic_mem
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
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).publicWidth) :
    (Layout.freshFrame key).publicColumn coordinate ∈
      (Layout.freshAllocations program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).map fun column => column.id := by
  change
    Layout.localColumn Layout.freshOwner
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree + coordinate.val) ∈
      (Layout.freshAllocations program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).map fun column => column.id
  apply freshLocalColumn_mem
  unfold Layout.freshWidth Layout.freshInputWidth
    Layout.freshStatementWidth
  omega

private theorem freshSquare_mem
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
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    (Layout.freshFrame key).square coordinate ∈
      (Layout.freshAllocations program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).map fun column => column.id := by
  change
    Layout.localColumn Layout.freshOwner
        (Layout.freshInputWidth
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows + coordinate.val) ∈
      (Layout.freshAllocations program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).map fun column => column.id
  apply freshLocalColumn_mem
  unfold Layout.freshWidth
  omega

private theorem freshResidual_mem
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
    (source : Fin program.rows.length) :
    (Layout.freshFrame key).residual source ∈
      (Layout.freshAllocations program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).map fun column => column.id := by
  change
    Layout.localColumn Layout.freshOwner
        (Layout.freshInputWidth
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows +
          (RelationShape program domain publicRingColumns publicFits).carrierWidth +
          source.val) ∈
      (Layout.freshAllocations program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).map fun column => column.id
  apply freshLocalColumn_mem
  unfold Layout.freshWidth
  omega

/-- Every running-claim row reads only the verifier-owned constant column or
one column allocated by that claim's exact receipt. -/
theorem runningRows_supported
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
    (child : Fin productionGlobalParams.k)
    (owned : OwnedRow)
    (member :
      owned ∈
        Running.rows (Layout.runningFrame key child) (statements child))
    (column : ColumnId)
    (mentioned : column ∈ owned.columnIds) :
    column = oneColumn ∨
      column ∈
        (Layout.runningAllocations
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows child).map fun allocated => allocated.id := by
  simp only [Running.rows, List.mem_append] at member
  rcases member with ajtaiMember |
      (projectionMember | (normMember | evaluationMember))
  · rcases List.mem_ofFn.mp ajtaiMember with ⟨coordinate, rfl⟩
    rcases
        Ajtai.row_supported
          (Running.ajtaiFrame (Layout.runningFrame key child))
          coordinate column mentioned with
      one | witness | commitment
    · exact Or.inl one
    · rcases witness with ⟨witnessCoordinate, rfl⟩
      exact Or.inr (runningWitness_mem key child witnessCoordinate)
    · rw [commitment]
      exact Or.inr
        (runningCommitment_mem key child
          (Ajtai.verifierRowAt coordinate) (Ajtai.outputAt coordinate))
  · rcases List.mem_ofFn.mp projectionMember with ⟨coordinate, rfl⟩
    rcases
        Projection.row_supported
          (Running.projectionFrame (Layout.runningFrame key child))
          coordinate column mentioned with
      one | witness | publicColumn
    · exact Or.inl one
    · rw [witness]
      exact Or.inr
        (runningWitness_mem key child
          ((RelationShape program domain publicRingColumns publicFits).publicColumn
            coordinate))
    · rw [publicColumn]
      exact Or.inr (runningPublic_mem key child coordinate)
  · rcases List.mem_ofFn.mp normMember with ⟨position, rfl⟩
    rcases
        Norm.rowAt_supported
          (Running.normFrame (Layout.runningFrame key child))
          position column mentioned with
      witness | square
    · rw [witness]
      exact Or.inr
        (runningWitness_mem key child (Norm.coordinateAt position))
    · rw [square]
      exact Or.inr
        (runningSquare_mem key child (Norm.coordinateAt position))
  · rcases List.mem_ofFn.mp evaluationMember with ⟨position, rfl⟩
    rcases
        FixedPointEvaluation.rowAt_supported
          (Running.evaluationFrame (Layout.runningFrame key child))
          (statements child).constraintSystem (statements child).point
          position column mentioned with
      one | witness | low | high
    · exact Or.inl one
    · rcases witness with ⟨witnessCoordinate, rfl⟩
      exact Or.inr (runningWitness_mem key child witnessCoordinate)
    · rw [low]
      exact Or.inr
        (runningEvaluationLow_mem key child
          (FixedPointEvaluation.matrixAt position)
          (FixedPointEvaluation.laneAt position))
    · rw [high]
      exact Or.inr
        (runningEvaluationHigh_mem key child
          (FixedPointEvaluation.matrixAt position)
          (FixedPointEvaluation.laneAt position))

/-- Every fresh-claim row reads only the verifier-owned constant column or one
column allocated by the fresh claim's exact receipt. -/
theorem freshRows_supported
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
    (owned : OwnedRow)
    (member : owned ∈ Fresh.rows valid (Layout.freshFrame key))
    (column : ColumnId)
    (mentioned : column ∈ owned.columnIds) :
    column = oneColumn ∨
      column ∈
        (Layout.freshAllocations program
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows).map fun allocated => allocated.id := by
  simp only [Fresh.rows, List.mem_append] at member
  rcases member with ajtaiMember |
      (projectionMember | (normMember | ccsMember))
  · rcases List.mem_ofFn.mp ajtaiMember with ⟨coordinate, rfl⟩
    rcases
        Ajtai.row_supported
          (Fresh.ajtaiFrame (Layout.freshFrame key))
          coordinate column mentioned with
      one | witness | commitment
    · exact Or.inl one
    · rcases witness with ⟨witnessCoordinate, rfl⟩
      exact Or.inr (freshWitness_mem key witnessCoordinate)
    · rw [commitment]
      exact Or.inr
        (freshCommitment_mem key
          (Ajtai.verifierRowAt coordinate) (Ajtai.outputAt coordinate))
  · rcases List.mem_ofFn.mp projectionMember with ⟨coordinate, rfl⟩
    rcases
        Projection.row_supported
          (Fresh.projectionFrame (Layout.freshFrame key))
          coordinate column mentioned with
      one | witness | publicColumn
    · exact Or.inl one
    · rw [witness]
      exact Or.inr
        (freshWitness_mem key
          ((RelationShape program domain publicRingColumns publicFits).publicColumn
            coordinate))
    · rw [publicColumn]
      exact Or.inr (freshPublic_mem key coordinate)
  · rcases List.mem_ofFn.mp normMember with ⟨position, rfl⟩
    rcases
        Norm.rowAt_supported
          (Fresh.normFrame (Layout.freshFrame key))
          position column mentioned with
      witness | square
    · rw [witness]
      exact Or.inr (freshWitness_mem key (Norm.coordinateAt position))
    · rw [square]
      exact Or.inr (freshSquare_mem key (Norm.coordinateAt position))
  · rcases List.mem_ofFn.mp ccsMember with ⟨position, rfl⟩
    rcases
        FreshCcs.rowAt_supported_by_frame valid
          (Fresh.ccsFrame (Layout.freshFrame key))
          position column mentioned with
      witness | residual
    · rcases witness with ⟨witnessCoordinate, rfl⟩
      exact Or.inr (freshWitness_mem key witnessCoordinate)
    · rw [residual]
      exact Or.inr
        (freshResidual_mem key (FreshCcs.sourceAt position))

/-- One running receipt is scoped once the public constant-one column is
available. All other dependencies are allocated by that same receipt. -/
theorem runningReceipt_wellScoped
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
    (child : Fin productionGlobalParams.k)
    (available : List ColumnId)
    (oneAvailable : oneColumn ∈ available) :
    (Layout.runningReceipt key statements child).WellScopedAfter available := by
  intro column referenced
  rcases List.mem_flatMap.mp referenced with
    ⟨owned, rowMember, columnMember⟩
  have mentioned : column ∈ owned.columnIds := by
    simpa [InstructionReceipt.rowColumns, OwnedRow.columnIds,
      Row.columnIds] using columnMember
  rcases
      runningRows_supported key statements child owned rowMember
        column mentioned with
    one | allocated
  · exact Or.inl (one ▸ oneAvailable)
  · exact Or.inr (by
      simpa [Layout.runningReceipt, InstructionReceipt.columnIds]
        using allocated)

/-- The fresh receipt is scoped once the public constant-one column is
available. All other dependencies are allocated by that same receipt. -/
theorem freshReceipt_wellScoped
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
    (available : List ColumnId)
    (oneAvailable : oneColumn ∈ available) :
    (Layout.freshReceipt valid key).WellScopedAfter available := by
  intro column referenced
  rcases List.mem_flatMap.mp referenced with
    ⟨owned, rowMember, columnMember⟩
  have mentioned : column ∈ owned.columnIds := by
    simpa [InstructionReceipt.rowColumns, OwnedRow.columnIds,
      Row.columnIds] using columnMember
  rcases freshRows_supported valid key owned rowMember column mentioned with
    one | allocated
  · exact Or.inl (one ▸ oneAvailable)
  · exact Or.inr (by
      simpa [Layout.freshReceipt, InstructionReceipt.columnIds]
        using allocated)

private theorem claimReceipts_wellScoped
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
    ∀ (children : List (Fin productionGlobalParams.k))
      (available : List ColumnId),
      oneColumn ∈ available →
      ReceiptsWellScoped available
        (children.map (Layout.runningReceipt key statements) ++
          [Layout.freshReceipt valid key])
  | [], available, oneAvailable => by
      constructor
      · exact freshReceipt_wellScoped valid key available oneAvailable
      · trivial
  | child :: rest, available, oneAvailable => by
      constructor
      · exact
          runningReceipt_wellScoped key statements child
            available oneAvailable
      · apply claimReceipts_wellScoped valid key statements rest
          (available ++
            (Layout.runningReceipt key statements child).columnIds)
        exact List.mem_append_left _ oneAvailable

/-- The complete terminal receipt stream is scoped in execution order. The
prelude allocates the only cross-receipt dependency, the constant-one
column; each claim then owns every other column that its rows mention. -/
theorem receipts_wellScoped
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
    ReceiptsWellScoped [] (Layout.receipts valid key statements) := by
  unfold Layout.receipts
  constructor
  · intro column referenced
    simp [InstructionReceipt.referencedColumns,
      InstructionReceipt.prelude_rows] at referenced
  · apply claimReceipts_wellScoped valid key statements
    simp [InstructionReceipt.prelude_columnIds]

/-- Complete physical evidence for the exact proof-free terminal manifest.

This structure is terminal-specific because the terminal relation is not a
`Typed.Program`. It does not fabricate a typed source only to reuse the
generic `ReceiptProgram` index. -/
structure PhysicalCertificate
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
        verifierRows) : Prop where
  preludeMember :
    InstructionReceipt.prelude ∈ Layout.receipts valid key statements
  ownersNodup :
    ((Layout.receipts valid key statements).map
      fun receipt => receipt.owner).Nodup
  allocationIdsNodup :
    ((Layout.receipts valid key statements).flatMap
      InstructionReceipt.columnIds).Nodup
  rowIdsNodup :
    ((Layout.receipts valid key statements).flatMap
      InstructionReceipt.rowIds).Nodup
  wellScoped :
    ReceiptsWellScoped [] (Layout.receipts valid key statements)

/-- The selected terminal manifest has one complete kernel-checked physical
certificate. -/
def physicalCertificate
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
    PhysicalCertificate valid key statements where
  preludeMember := by
    simp [Layout.receipts]
  ownersNodup := Physical.receiptOwners_nodup valid key statements
  allocationIdsNodup := Physical.allocationIds_nodup valid key statements
  rowIdsNodup := Physical.rowIds_nodup valid key statements
  wellScoped := receipts_wellScoped valid key statements

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Scoping
