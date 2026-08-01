import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Program

/-!
Contract: physical uniqueness for the selected SuperNeo terminal R1CS.

Assurance tier: model-level.

Owns: injective terminal claim owners, receipt-owner uniqueness, receipt-local
allocation uniqueness, and global allocation uniqueness for the exact
terminal receipt stream.

Does not own: row scoping, honest assignment construction, a selected
benchmark statement, Spartan, WHIR, Rust, or Ajtai binding security.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Physical

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

private theorem nodup_ofFn_of_injective
    {alpha : Type} :
    ∀ {count : Nat}
      (function : Fin count → alpha),
      Function.Injective function →
      (List.ofFn function).Nodup
  | 0, function, injective => by
      simp
  | _ + 1, function, injective => by
      rw [List.ofFn_succ, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index (injective equal)
      · exact nodup_ofFn_of_injective
          (fun index => function index.succ)
          (fun first second equal =>
            Fin.succ_inj.mp (injective equal))

private theorem nodup_map_of_injective
    {alpha beta : Type}
    (function : alpha → beta)
    (injective : Function.Injective function) :
    ∀ {items : List alpha}, items.Nodup → (items.map function).Nodup
  | [], _ => List.nodup_nil
  | head :: tail, nodup => by
      simp only [List.map_cons, List.nodup_cons] at nodup ⊢
      refine ⟨?_, nodup_map_of_injective function injective nodup.2⟩
      intro member
      rcases List.mem_map.mp member with ⟨item, itemMember, equal⟩
      exact nodup.1 (injective equal.symm ▸ itemMember)

/-- Unary structural paths retain the exact claim index. -/
theorem claimPath_injective : Function.Injective Layout.claimPath := by
  intro first second equal
  induction first generalizing second with
  | zero =>
      cases second with
      | zero => rfl
      | succ second => cases equal
  | succ first inductionHypothesis =>
      cases second with
      | zero => cases equal
      | succ second =>
          simp only [Layout.claimPath] at equal
          exact congrArg Nat.succ
            (inductionHypothesis (OwnerPath.rest.inj equal))

/-- Different running children have different physical owners. -/
theorem runningOwner_injective : Function.Injective Layout.runningOwner := by
  intro first second equal
  apply Fin.ext
  apply claimPath_injective
  exact Typed.Owner.instruction.inj
    (PhysicalOwner.typed.inj equal)

/-- The fresh owner follows, and cannot equal, any running child owner. -/
theorem runningOwner_ne_fresh
    (child : Fin productionGlobalParams.k) :
    Layout.runningOwner child ≠ Layout.freshOwner := by
  intro equal
  have pathEqual :
      Layout.claimPath child.val =
        Layout.claimPath productionGlobalParams.k :=
    Typed.Owner.instruction.inj (PhysicalOwner.typed.inj equal)
  have valueEqual := claimPath_injective pathEqual
  omega

/-- The exact terminal receipt list has one structural owner per position. -/
theorem receiptOwners_nodup
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
    ((Layout.receipts valid key statements).map
      fun receipt => receipt.owner).Nodup := by
  have runningNodup :
      ((List.finRange productionGlobalParams.k).map
        Layout.runningOwner).Nodup :=
    nodup_map_of_injective Layout.runningOwner runningOwner_injective
      (by
        change (List.ofFn fun child : Fin productionGlobalParams.k => child).Nodup
        exact nodup_ofFn_of_injective _ fun _ _ equal => equal)
  have freshNotRunning :
      Layout.freshOwner ∉
        (List.finRange productionGlobalParams.k).map Layout.runningOwner := by
    intro member
    rcases List.mem_map.mp member with ⟨child, _childMember, equal⟩
    exact runningOwner_ne_fresh child equal
  simp only [Layout.receipts, List.map_cons, List.map_append, List.map_map,
    Layout.freshReceipt, List.map_nil, List.nodup_cons]
  refine ⟨?_, ?_⟩
  · intro member
    rcases List.mem_append.mp member with runningMember | freshMember
    · rcases List.mem_map.mp runningMember with
        ⟨child, _childMember, equal⟩
      change Layout.runningOwner child = .prelude at equal
      cases equal
    · have equal : (.prelude : PhysicalOwner) = Layout.freshOwner := by
        simpa using freshMember
      cases equal
  rw [List.nodup_append]
  exact ⟨
    runningNodup,
    (by simp),
    by
      intro running runningMember fresh freshMember equal
      have freshEqual : fresh = Layout.freshOwner := by
        simpa using freshMember
      rcases List.mem_map.mp runningMember with
        ⟨child, _childMember, runningEqual⟩
      exact runningOwner_ne_fresh child
        (runningEqual.trans (equal.trans freshEqual))
  ⟩

private theorem columnBlockIds_nodup
    (owner : PhysicalOwner)
    (start count : Nat)
    (ownership : Ownership) :
    ((Layout.columnBlock owner start count ownership).map
      fun column => column.id).Nodup := by
  rw [Layout.columnBlock, List.map_ofFn]
  apply nodup_ofFn_of_injective
  intro first second equal
  apply Fin.ext
  have coordinateEqual :=
    congrArg (fun id : ColumnId => id.coordinateIndex) equal
  exact Nat.add_left_cancel coordinateEqual

private theorem columnBlocks_disjoint
    (owner : PhysicalOwner)
    (firstStart firstCount secondStart secondCount : Nat)
    (firstOwnership secondOwnership : Ownership)
    (ordered : firstStart + firstCount ≤ secondStart) :
    ∀ firstId,
      firstId ∈
          ((Layout.columnBlock owner firstStart firstCount firstOwnership).map
            fun column => column.id) →
        ∀ secondId,
          secondId ∈
              ((Layout.columnBlock owner secondStart secondCount
                secondOwnership).map fun column => column.id) →
            firstId ≠ secondId := by
  intro firstId firstMember secondId secondMember equal
  rcases List.mem_map.mp firstMember with
    ⟨firstColumn, firstColumnMember, firstEqual⟩
  rcases List.mem_map.mp secondMember with
    ⟨secondColumn, secondColumnMember, secondEqual⟩
  rcases List.mem_ofFn.mp firstColumnMember with ⟨first, rfl⟩
  rcases List.mem_ofFn.mp secondColumnMember with ⟨second, rfl⟩
  have idsEqual :
      (Layout.localColumn owner (firstStart + first.val)) =
        Layout.localColumn owner (secondStart + second.val) :=
    firstEqual.trans (equal.trans secondEqual.symm)
  have coordinateEqual :=
    congrArg (fun column : ColumnId => column.coordinateIndex) idsEqual
  simp only [Layout.localColumn] at coordinateEqual
  have firstBelow : firstStart + first.val < secondStart := by
    omega
  omega

private theorem threeColumnBlocks_nodup
    (owner : PhysicalOwner)
    (firstCount secondCount thirdCount : Nat)
    (firstOwnership secondOwnership thirdOwnership : Ownership) :
    ((Layout.columnBlock owner 0 firstCount firstOwnership ++
      (Layout.columnBlock owner firstCount secondCount secondOwnership ++
        Layout.columnBlock owner (firstCount + secondCount)
          thirdCount thirdOwnership)).map fun column => column.id).Nodup := by
  simp only [List.map_append]
  rw [List.nodup_append]
  refine ⟨columnBlockIds_nodup _ _ _ _, ?_, ?_⟩
  · rw [List.nodup_append]
    exact ⟨
      columnBlockIds_nodup _ _ _ _,
      columnBlockIds_nodup _ _ _ _,
      columnBlocks_disjoint owner firstCount secondCount
        (firstCount + secondCount) thirdCount
        secondOwnership thirdOwnership (by omega)
    ⟩
  · intro firstId firstMember tailId tailMember
    rcases List.mem_append.mp tailMember with secondMember | thirdMember
    · exact
        columnBlocks_disjoint owner 0 firstCount
          firstCount secondCount
          firstOwnership secondOwnership (by omega)
          firstId firstMember tailId secondMember
    · exact
        columnBlocks_disjoint owner 0 firstCount
          (firstCount + secondCount) thirdCount
          firstOwnership thirdOwnership (by omega)
          firstId firstMember tailId thirdMember

private theorem fourColumnBlocks_nodup
    (owner : PhysicalOwner)
    (firstCount secondCount thirdCount fourthCount : Nat)
    (firstOwnership secondOwnership thirdOwnership fourthOwnership :
      Ownership) :
    ((Layout.columnBlock owner 0 firstCount firstOwnership ++
      (Layout.columnBlock owner firstCount secondCount secondOwnership ++
        (Layout.columnBlock owner (firstCount + secondCount)
          thirdCount thirdOwnership ++
        Layout.columnBlock owner (firstCount + secondCount + thirdCount)
          fourthCount fourthOwnership))).map fun column => column.id).Nodup := by
  simp only [List.map_append]
  rw [List.nodup_append]
  refine ⟨columnBlockIds_nodup _ _ _ _, ?_, ?_⟩
  · rw [List.nodup_append]
    refine ⟨columnBlockIds_nodup _ _ _ _, ?_, ?_⟩
    · rw [List.nodup_append]
      exact ⟨
        columnBlockIds_nodup _ _ _ _,
        columnBlockIds_nodup _ _ _ _,
        columnBlocks_disjoint owner
          (firstCount + secondCount) thirdCount
          (firstCount + secondCount + thirdCount) fourthCount
          thirdOwnership fourthOwnership (by omega)
      ⟩
    · intro secondId secondMember tailId tailMember
      rcases List.mem_append.mp tailMember with thirdMember | fourthMember
      · exact
          columnBlocks_disjoint owner firstCount secondCount
            (firstCount + secondCount) thirdCount
            secondOwnership thirdOwnership (by omega)
            secondId secondMember tailId thirdMember
      · exact
          columnBlocks_disjoint owner firstCount secondCount
            (firstCount + secondCount + thirdCount) fourthCount
            secondOwnership fourthOwnership (by omega)
            secondId secondMember tailId fourthMember
  · intro firstId firstMember tailId tailMember
    rcases List.mem_append.mp tailMember with secondMember | tailMember
    · exact
        columnBlocks_disjoint owner 0 firstCount firstCount secondCount
          firstOwnership secondOwnership (by omega)
          firstId firstMember tailId secondMember
    · rcases List.mem_append.mp tailMember with thirdMember | fourthMember
      · exact
          columnBlocks_disjoint owner 0 firstCount
            (firstCount + secondCount) thirdCount
            firstOwnership thirdOwnership (by omega)
            firstId firstMember tailId thirdMember
      · exact
          columnBlocks_disjoint owner 0 firstCount
            (firstCount + secondCount + thirdCount) fourthCount
            firstOwnership fourthOwnership (by omega)
            firstId firstMember tailId fourthMember

/-- One running receipt allocates each physical column exactly once. -/
theorem runningAllocationIds_nodup
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat)
    (child : Fin productionGlobalParams.k) :
    ((Layout.runningAllocations shape verifierRows child).map
      fun column => column.id).Nodup := by
  unfold Layout.runningAllocations
  simpa [Layout.runningInputWidth] using
    threeColumnBlocks_nodup
      (Layout.runningOwner child)
      shape.carrierWidth
      (Layout.runningStatementWidth shape verifierRows)
      shape.carrierWidth
      .committedColumn .publicColumn .auxiliaryColumn

/-- The fresh receipt allocates each physical column exactly once. -/
theorem freshAllocationIds_nodup
    (program : NativeCcsProgram.Program)
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat) :
    ((Layout.freshAllocations program shape verifierRows).map
      fun column => column.id).Nodup := by
  unfold Layout.freshAllocations
  simpa [Layout.freshInputWidth] using
    fourColumnBlocks_nodup
      Layout.freshOwner
      shape.carrierWidth
      (Layout.freshStatementWidth shape verifierRows)
      shape.carrierWidth
      program.rows.length
      .committedColumn .publicColumn .auxiliaryColumn .auxiliaryColumn

private def RowOrdinalsWithin
    (rows : List OwnedRow)
    (first count : Nat) : Prop :=
  ∀ owned, owned ∈ rows →
    first ≤ owned.id.ordinal ∧ owned.id.ordinal < first + count

private theorem ofFnRows_ordinalsWithin
    {count first : Nat}
    (function : Fin count → OwnedRow)
    (ordinal :
      ∀ position,
        (function position).id.ordinal = first + position.val) :
    RowOrdinalsWithin (List.ofFn function) first count := by
  intro owned member
  rcases List.mem_ofFn.mp member with ⟨position, rfl⟩
  rw [ordinal]
  omega

private theorem ajtaiRows_ordinalsWithin
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Ajtai.Frame shape verifierRows) :
    RowOrdinalsWithin (Ajtai.rows frame) frame.firstOrdinal
      (verifierRows * ringDegree) := by
  apply ofFnRows_ordinalsWithin
  intro position
  rfl

private theorem projectionRows_ordinalsWithin
    {shape : Phi81Relation.Shape}
    (frame : Projection.Frame shape) :
    RowOrdinalsWithin (Projection.rows frame) frame.firstOrdinal
      shape.publicWidth := by
  apply ofFnRows_ordinalsWithin
  intro position
  rfl

private theorem normRows_ordinalsWithin
    {shape : Phi81Relation.Shape}
    (frame : Norm.Frame shape) :
    RowOrdinalsWithin (Norm.rows frame) frame.firstOrdinal
      (2 * shape.carrierWidth) := by
  apply ofFnRows_ordinalsWithin
  intro position
  unfold Norm.rowAt
  split <;> rfl

private theorem evaluationRows_ordinalsWithin
    {shape : Phi81Relation.Shape}
    (frame : FixedPointEvaluation.Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape) :
    RowOrdinalsWithin
      (FixedPointEvaluation.rows frame system point)
      frame.firstOrdinal
      (2 * (shape.matrixCount * ringDegree)) := by
  apply ofFnRows_ordinalsWithin
  intro position
  unfold FixedPointEvaluation.rowAt
  split <;> rfl

private theorem freshCcsRows_ordinalsWithin
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (valid : NativeCcsCompiler.Valid program)
    (frame : FreshCcs.Frame program domain publicRingColumns publicFits) :
    RowOrdinalsWithin (FreshCcs.rows valid frame) frame.firstOrdinal
      (2 * program.rows.length) := by
  apply ofFnRows_ordinalsWithin
  intro position
  unfold FreshCcs.rowAt
  split <;> rfl

private theorem rowBlocks_disjoint
    (firstRows secondRows : List OwnedRow)
    (firstStart firstCount secondStart secondCount : Nat)
    (firstWithin :
      RowOrdinalsWithin firstRows firstStart firstCount)
    (secondWithin :
      RowOrdinalsWithin secondRows secondStart secondCount)
    (ordered : firstStart + firstCount ≤ secondStart) :
    ∀ firstId,
      firstId ∈ firstRows.map (fun owned => owned.id) →
        ∀ secondId,
          secondId ∈ secondRows.map (fun owned => owned.id) →
            firstId ≠ secondId := by
  intro firstId firstMember secondId secondMember equal
  rcases List.mem_map.mp firstMember with
    ⟨firstOwned, firstOwnedMember, rfl⟩
  rcases List.mem_map.mp secondMember with
    ⟨secondOwned, secondOwnedMember, rfl⟩
  have firstBounds := firstWithin firstOwned firstOwnedMember
  have secondBounds := secondWithin secondOwned secondOwnedMember
  have ordinalEqual :
      firstOwned.id.ordinal = secondOwned.id.ordinal :=
    congrArg (fun id : RowId => id.ordinal) equal
  omega

private theorem fourRowBlocks_nodup
    (firstRows secondRows thirdRows fourthRows : List OwnedRow)
    (firstStart firstCount secondCount thirdCount fourthCount : Nat)
    (firstNodup :
      (firstRows.map fun owned => owned.id).Nodup)
    (secondNodup :
      (secondRows.map fun owned => owned.id).Nodup)
    (thirdNodup :
      (thirdRows.map fun owned => owned.id).Nodup)
    (fourthNodup :
      (fourthRows.map fun owned => owned.id).Nodup)
    (firstWithin :
      RowOrdinalsWithin firstRows firstStart firstCount)
    (secondWithin :
      RowOrdinalsWithin secondRows (firstStart + firstCount) secondCount)
    (thirdWithin :
      RowOrdinalsWithin thirdRows
        (firstStart + firstCount + secondCount) thirdCount)
    (fourthWithin :
      RowOrdinalsWithin fourthRows
        (firstStart + firstCount + secondCount + thirdCount) fourthCount) :
    ((firstRows ++ (secondRows ++ (thirdRows ++ fourthRows))).map
      fun owned => owned.id).Nodup := by
  simp only [List.map_append]
  rw [List.nodup_append]
  refine ⟨firstNodup, ?_, ?_⟩
  · rw [List.nodup_append]
    refine ⟨secondNodup, ?_, ?_⟩
    · rw [List.nodup_append]
      exact ⟨
        thirdNodup,
        fourthNodup,
        rowBlocks_disjoint thirdRows fourthRows
          (firstStart + firstCount + secondCount) thirdCount
          (firstStart + firstCount + secondCount + thirdCount) fourthCount
          thirdWithin fourthWithin (by omega)
      ⟩
    · intro secondId secondMember tailId tailMember
      rcases List.mem_append.mp tailMember with
        thirdMember | fourthMember
      · exact
          rowBlocks_disjoint secondRows thirdRows
            (firstStart + firstCount) secondCount
            (firstStart + firstCount + secondCount) thirdCount
            secondWithin thirdWithin (by omega)
            secondId secondMember tailId thirdMember
      · exact
          rowBlocks_disjoint secondRows fourthRows
            (firstStart + firstCount) secondCount
            (firstStart + firstCount + secondCount + thirdCount) fourthCount
            secondWithin fourthWithin (by omega)
            secondId secondMember tailId fourthMember
  · intro firstId firstMember tailId tailMember
    rcases List.mem_append.mp tailMember with secondMember | tailMember
    · exact
        rowBlocks_disjoint firstRows secondRows
          firstStart firstCount
          (firstStart + firstCount) secondCount
          firstWithin secondWithin (by omega)
          firstId firstMember tailId secondMember
    · rcases List.mem_append.mp tailMember with thirdMember | fourthMember
      · exact
          rowBlocks_disjoint firstRows thirdRows
            firstStart firstCount
            (firstStart + firstCount + secondCount) thirdCount
            firstWithin thirdWithin (by omega)
            firstId firstMember tailId thirdMember
      · exact
          rowBlocks_disjoint firstRows fourthRows
            firstStart firstCount
            (firstStart + firstCount + secondCount + thirdCount) fourthCount
            firstWithin fourthWithin (by omega)
            firstId firstMember tailId fourthMember

/-- One running receipt assigns a unique positional identity to every row. -/
theorem runningRowIds_nodup
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Running.Frame shape verifierRows)
    (statement :
      Phi81Relation.CEStatement shape (CommitmentValue verifierRows)) :
    ((Running.rows frame statement).map fun owned => owned.id).Nodup := by
  unfold Running.rows
  apply fourRowBlocks_nodup
  · exact Ajtai.rowIds_nodup (Running.ajtaiFrame frame)
  · exact Projection.rowIds_nodup (Running.projectionFrame frame)
  · exact Norm.rowIds_nodup (Running.normFrame frame)
  · exact FixedPointEvaluation.rowIds_nodup
      (Running.evaluationFrame frame)
      statement.constraintSystem statement.point
  · exact ajtaiRows_ordinalsWithin (Running.ajtaiFrame frame)
  · exact projectionRows_ordinalsWithin (Running.projectionFrame frame)
  · exact normRows_ordinalsWithin (Running.normFrame frame)
  · exact evaluationRows_ordinalsWithin
      (Running.evaluationFrame frame)
      statement.constraintSystem statement.point

/-- The fresh receipt assigns a unique positional identity to every row. -/
theorem freshRowIds_nodup
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (frame :
      Fresh.Frame program domain publicRingColumns publicFits verifierRows) :
    ((Fresh.rows valid frame).map fun owned => owned.id).Nodup := by
  unfold Fresh.rows
  apply fourRowBlocks_nodup
  · exact Ajtai.rowIds_nodup (Fresh.ajtaiFrame frame)
  · exact Projection.rowIds_nodup (Fresh.projectionFrame frame)
  · exact Norm.rowIds_nodup (Fresh.normFrame frame)
  · exact FreshCcs.rowIds_nodup valid (Fresh.ccsFrame frame)
  · exact ajtaiRows_ordinalsWithin (Fresh.ajtaiFrame frame)
  · exact projectionRows_ordinalsWithin (Fresh.projectionFrame frame)
  · exact normRows_ordinalsWithin (Fresh.normFrame frame)
  · exact freshCcsRows_ordinalsWithin valid (Fresh.ccsFrame frame)

private theorem flattenedIds_nodup
    {Receipt Owner Id : Type}
    (ownerOf : Receipt → Owner)
    (idOwner : Id → Owner)
    (ids : Receipt → List Id)
    (receipts : List Receipt)
    (ownersNodup : (receipts.map ownerOf).Nodup)
    (localNodup :
      ∀ receipt, receipt ∈ receipts → (ids receipt).Nodup)
    (idsOwned :
      ∀ receipt id, id ∈ ids receipt →
        idOwner id = ownerOf receipt) :
    (receipts.flatMap ids).Nodup := by
  induction receipts with
  | nil =>
      exact List.nodup_nil
  | cons head tail inductionHypothesis =>
      have ownerSplit :
          ownerOf head ∉ tail.map ownerOf ∧
            (tail.map ownerOf).Nodup := by
        simpa only [List.map_cons, List.nodup_cons] using ownersNodup
      rw [List.flatMap_cons, List.nodup_append]
      refine ⟨
        localNodup head List.mem_cons_self,
        inductionHypothesis ownerSplit.2
          (fun receipt member =>
            localNodup receipt (List.mem_cons_of_mem head member)),
        ?_
      ⟩
      intro headId headMember tailId tailMember idsEqual
      rcases List.mem_flatMap.mp tailMember with
        ⟨tailReceipt, tailReceiptMember, tailIdMember⟩
      have ownersEqual : ownerOf head = ownerOf tailReceipt := by
        calc
          ownerOf head = idOwner headId :=
            (idsOwned head headId headMember).symm
          _ = idOwner tailId := congrArg idOwner idsEqual
          _ = ownerOf tailReceipt :=
            idsOwned tailReceipt tailId tailIdMember
      exact False.elim (ownerSplit.1
        (List.mem_map.mpr
          ⟨tailReceipt, tailReceiptMember, ownersEqual.symm⟩))

/-- Every terminal allocation ID is globally unique. -/
theorem allocationIds_nodup
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
    ((Layout.receipts valid key statements).flatMap
      InstructionReceipt.columnIds).Nodup := by
  apply flattenedIds_nodup
    (fun receipt : InstructionReceipt => receipt.owner)
    (fun id : ColumnId => id.owner)
    InstructionReceipt.columnIds
    (Layout.receipts valid key statements)
    (receiptOwners_nodup valid key statements)
  · intro receipt member
    simp only [Layout.receipts, List.mem_cons, List.mem_append,
      List.mem_map, List.not_mem_nil, or_false] at member
    rcases member with rfl | member
    · simp [InstructionReceipt.columnIds, InstructionReceipt.prelude,
        preludeColumns]
    · rcases member with
        ⟨child, _childMember, runningEqual⟩ | freshMember
      · rw [← runningEqual]
        exact runningAllocationIds_nodup _ _ child
      · rcases freshMember with rfl
        exact freshAllocationIds_nodup _ _ _
  · intro receipt id member
    rcases List.mem_map.mp member with ⟨column, columnMember, rfl⟩
    exact receipt.allocationsOwned column columnMember

/-- Every terminal row occurrence ID is globally unique. -/
theorem rowIds_nodup
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
    ((Layout.receipts valid key statements).flatMap
      InstructionReceipt.rowIds).Nodup := by
  apply flattenedIds_nodup
    (fun receipt : InstructionReceipt => receipt.owner)
    (fun id : RowId => id.owner)
    InstructionReceipt.rowIds
    (Layout.receipts valid key statements)
    (receiptOwners_nodup valid key statements)
  · intro receipt member
    simp only [Layout.receipts, List.mem_cons, List.mem_append,
      List.mem_map, List.not_mem_nil, or_false] at member
    rcases member with rfl | member
    · simp [InstructionReceipt.rowIds, InstructionReceipt.prelude]
    · rcases member with
        ⟨child, _childMember, runningEqual⟩ | freshMember
      · rw [← runningEqual]
        exact runningRowIds_nodup
          (Layout.runningFrame key child) (statements child)
      · rcases freshMember with rfl
        exact freshRowIds_nodup valid (Layout.freshFrame key)
  · intro receipt id member
    rcases List.mem_map.mp member with ⟨row, rowMember, rfl⟩
    exact receipt.rowsOwned row rowMember

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Physical
