import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Profile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.SourceOwners

/-!
Contract: canonical obligation-10 physical encoding certificates for the
fixed-one step and terminal typed programs.

Owns:
- one source-aligned conserved receipt program;
- exact selected rows at every finite rewrite site;
- exact joined-column allocation for the step and exact empty terminal join;
- a fixed cost computed from the actual non-rewrite rows and all actual
  canonical allocations;
- exact decomposition of physical cost into that fixed program cost plus the
  selected finite-class cost;
- conservation, unique ownership, and finite-class minimum theorems.

Does not own: semantic soundness or honest witness construction for the full
physical row program; those are proved downstream by
`Canonical{Step,Terminal}Soundness` and
`Canonical{Step,Terminal}PhysicalCompleteness`.  It also does not own a
production codec/recipe instantiation, Rust emission, generated row equality,
or source-to-R1CS refinement.  Those concrete obligation-11 connections
remain open.  No global minimality claim is made outside the explicit finite
class in `Encoding.NormalForm`.

All costs are four-way `Cost` values in the fixed lexicographic order:
recurring rows, committed columns, public columns, auxiliary columns.

Emits constraints: no.  This is the checked certificate boundary consumed by
a later concrete compiler/refinement proof.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.SourceAlignment
open Nightstream.Implementation.Lowering.Goldilocks.PrimitiveNormalForm
open Nightstream.Implementation.Lowering.Goldilocks.NormalFormComposition
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalEncoding

universe u

/-- One distinguished receipt with all of its selected physical data fixed
exactly. -/
structure ExactReceipt
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source)
    (owner : PhysicalOwner)
    (kind : InstructionKind)
    (allocations : List OwnedColumn)
    (rows : List OwnedRow) where
  receipt : InstructionReceipt
  member : receipt ∈ program.physical.receipts
  ownerExact : receipt.owner = owner
  kindExact : receipt.kind = kind
  allocationsExact : receipt.allocations = allocations
  rowsExact : receipt.rows = rows

namespace ExactReceipt

/-- Source alignment makes the selected receipt equal to every other receipt
claiming the same structural owner. -/
theorem unique
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    {program : AlignedReceiptProgram source}
    {owner : PhysicalOwner}
    {kind : InstructionKind}
    {allocations : List OwnedColumn}
    {rows : List OwnedRow}
    (selected : ExactReceipt program owner kind allocations rows)
    (candidate : InstructionReceipt)
    (candidateMember : candidate ∈ program.physical.receipts)
    (candidateOwner : candidate.owner = owner) :
    candidate = selected.receipt := by
  exact program.physical.receipt_eq_of_owner_eq
    candidateMember selected.member
    (candidateOwner.trans selected.ownerExact.symm)

end ExactReceipt

/-! ## Receipt-derived cost decomposition -/

private theorem rowCost_eq_length (rows : List OwnedRow) :
    rowCost rows = ⟨rows.length, 0, 0, 0⟩ := by
  induction rows with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      change Cost.oneRow + rowCost tail =
        ⟨(head :: tail).length, 0, 0, 0⟩
      rw [inductionHypothesis]
      change
        ({ recurringRows := 1 + tail.length
           committedColumns := 0
           publicColumns := 0
           auxiliaryColumns := 0 } : Cost) =
          { recurringRows := tail.length + 1
            committedColumns := 0
            publicColumns := 0
            auxiliaryColumns := 0 }
      rw [Nat.add_comm 1 tail.length]

private theorem filteredRows_eq_filteredReceipts
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source)
    (owners : List PhysicalOwner) :
    program.toEncoding.rows.filter
        (fun row => decide (row.id.owner ∈ owners)) =
      (program.physical.receipts.filter
        (fun receipt => decide (receipt.owner ∈ owners))).flatMap
          (fun receipt => receipt.rows) := by
  change
    (program.physical.receipts.flatMap
      (fun receipt => receipt.rows)).filter
        (fun row => decide (row.id.owner ∈ owners)) =
      (program.physical.receipts.filter
        (fun receipt => decide (receipt.owner ∈ owners))).flatMap
          (fun receipt => receipt.rows)
  induction program.physical.receipts with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.filter_append,
        List.filter_cons, inductionHypothesis]
      by_cases selected : head.owner ∈ owners
      · have rowsSelected :
            head.rows.filter
                (fun row => decide (row.id.owner ∈ owners)) =
              head.rows := by
          apply List.filter_eq_self.mpr
          intro row member
          have owned := head.rowsOwned row member
          simp [owned, selected]
        rw [rowsSelected]
        simp [selected]
      · have rowsRejected :
            head.rows.filter
                (fun row => decide (row.id.owner ∈ owners)) =
              [] := by
          apply List.filter_eq_nil_iff.mpr
          intro row member
          have owned := head.rowsOwned row member
          simp [owned, selected]
        rw [rowsRejected]
        simp [selected]

private theorem rows_length_eq_fixed_add_selected
    (rows : List OwnedRow)
    (selected : OwnedRow -> Bool) :
    rows.length =
      (rows.filter (fun row => !selected row)).length +
      (rows.filter selected).length := by
  induction rows with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      cases selectedHead : selected head <;>
        simp [selectedHead, inductionHypothesis] <;>
        omega

private theorem physicalCost_eq_fixed_add_rows
    (columns : List OwnedColumn)
    (rows fixedRows : List OwnedRow)
    (variableRows : Nat)
    (lengthExact :
      rows.length = fixedRows.length + variableRows) :
    physicalCost columns rows =
      physicalCost columns fixedRows +
        ⟨variableRows, 0, 0, 0⟩ := by
  unfold physicalCost
  rw [rowCost_eq_length, rowCost_eq_length, lengthExact]
  cases columnCost columns with
  | mk recurring committed publicCount auxiliary =>
      change
        ({ recurringRows := recurring + (fixedRows.length + variableRows)
           committedColumns := committed
           publicColumns := publicCount
           auxiliaryColumns := auxiliary } : Cost) =
          { recurringRows :=
              (recurring + fixedRows.length) + variableRows
            committedColumns := committed
            publicColumns := publicCount
            auxiliaryColumns := auxiliary }
      rw [Nat.add_assoc]

private theorem map_owner_filter
    (receipts : List InstructionReceipt)
    (owners : List PhysicalOwner) :
    (receipts.filter
      (fun receipt => decide (receipt.owner ∈ owners))).map
        (fun receipt => receipt.owner) =
      (receipts.map
        (fun receipt => receipt.owner)).filter
          (fun owner => decide (owner ∈ owners)) := by
  induction receipts with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      by_cases selected : head.owner ∈ owners <;>
        simp [selected, inductionHypothesis]

private theorem receiptLists_eq_of_ownerMaps_eq
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source)
    (left right : List InstructionReceipt)
    (leftMember :
      ∀ receipt, receipt ∈ left ->
        receipt ∈ program.physical.receipts)
    (rightMember :
      ∀ receipt, receipt ∈ right ->
        receipt ∈ program.physical.receipts)
    (ownersExact :
      left.map (fun receipt => receipt.owner) =
        right.map (fun receipt => receipt.owner)) :
    left = right := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil =>
          rfl
      | cons head tail =>
          simp only [List.map_nil, List.map_cons] at ownersExact
          cases ownersExact
  | cons head tail inductionHypothesis =>
      cases right with
      | nil =>
          simp only [List.map_cons, List.map_nil] at ownersExact
          cases ownersExact
      | cons rightHead rightTail =>
          simp only [List.map_cons, List.cons.injEq] at ownersExact
          have headsEqual :
              head = rightHead :=
            program.physical.receipt_eq_of_owner_eq
              (leftMember head (by simp))
              (rightMember rightHead (by simp))
              ownersExact.1
          subst rightHead
          congr
          apply inductionHypothesis
          · intro receipt member
            exact leftMember receipt
              (List.mem_cons_of_mem head member)
          · intro receipt member
            exact rightMember receipt
              (List.mem_cons_of_mem head member)
          · exact ownersExact.2

/-! ## Step certificate -/

def stepVariableOwners : List PhysicalOwner :=
  [.typed (.branch SourceOwners.stepBranchPath),
    .typed (.instruction SourceOwners.stepBaseAssertionPath),
    .typed (.instruction SourceOwners.stepRecursiveAssertionPath)]

/-- Rows outside the exact finite step rewrite positions. -/
def stepFixedRows
    {parameters : Parameters}
    (program :
      AlignedReceiptProgram
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.program
          parameters)) : List OwnedRow :=
  program.toEncoding.rows.filter
    (fun row => !(decide (row.id.owner ∈ stepVariableOwners)))

/-- The common step cost is computed from every actual canonical allocation
and exactly the rows outside the finite rewrite positions.  Canonical mux and
direct-assertion candidates allocate no candidate-specific columns, so all
canonical allocations belong to this common part. -/
def stepFixedCost
    {parameters : Parameters}
    (program :
      AlignedReceiptProgram
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.program
          parameters)) : Cost :=
  physicalCost program.toEncoding.columns (stepFixedRows program)

/-- Selected canonical encoding of the exact fixed-one step source program.

The cost equation is a proof, not a caller-supplied numeric measurement:
`stepFixedCost` unfolds to the physical receipt program itself and the local
term unfolds to the finite candidate program. -/
structure Step
    (parameters : Parameters) where
  profile :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Profile
      parameters
  specifications : NormalForm.StepSpecifications
  sitesAligned :
    SourceOwners.StepNormalFormAligned parameters specifications
  program :
    AlignedReceiptProgram
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.program
        parameters)
  branchJoin :
    ExactReceipt program
      (.typed (.branch SourceOwners.stepBranchPath))
      .branchJoin
      (schemaOwnedColumns
        (branchJoinColumns SourceOwners.stepBranchPath
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
            parameters)))
      (specifications.joinCoordinates.flatMap
        (fun specification =>
          BranchJoin.Candidate.rows
            .selectedMux specification))
  baseEndpoint :
    ExactReceipt program
      (.typed (.instruction SourceOwners.stepBaseAssertionPath))
      .assertion
      []
      (GatedAssertion.Candidate.rows
        .direct specifications.baseEndpoint)
  recursivePriorLink :
    ExactReceipt program
      (.typed (.instruction
        SourceOwners.stepRecursiveAssertionPath))
      .assertion
      []
      (GatedAssertion.Candidate.rows
        .direct specifications.recursivePriorLink)

namespace Step

def encoding
    {parameters : Parameters}
    (canonical : Step parameters) :
    Nightstream.Implementation.Lowering.Goldilocks.Encoding
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.program
        parameters) :=
  canonical.program.toEncoding

/-- The source-aligned receipt owners are definitionally the exact fixed-one
step owner list. -/
theorem receiptOwnersExact
    {parameters : Parameters}
    (canonical : Step parameters) :
    canonical.program.physical.receipts.map
        (fun receipt => receipt.owner) =
      SourceOwners.stepOwners parameters := by
  rw [canonical.program.ownersExact,
    SourceOwners.stepProgramOwnersExact]

/-- Every typed source owner has exactly one physical step receipt. -/
theorem everySourceOwnerHasExactlyOneReceipt
    {parameters : Parameters}
    (canonical : Step parameters)
    (owner : PhysicalOwner)
    (expected : owner ∈ SourceOwners.stepOwners parameters) :
    ∃ receipt,
      receipt ∈ canonical.program.physical.receipts ∧
        receipt.owner = owner ∧
          ∀ candidate,
            candidate ∈ canonical.program.physical.receipts ->
            candidate.owner = owner ->
            candidate = receipt := by
  apply canonical.program.expected_owner_has_exactly_one_receipt owner
  rw [SourceOwners.stepProgramOwnersExact]
  exact expected

private def selectedReceipts
    {parameters : Parameters}
    (canonical : Step parameters) : List InstructionReceipt :=
  [canonical.baseEndpoint.receipt,
    canonical.recursivePriorLink.receipt,
    canonical.branchJoin.receipt]

private theorem selectedReceipts_exact
    {parameters : Parameters}
    (canonical : Step parameters) :
    canonical.program.physical.receipts.filter
        (fun receipt =>
          decide (receipt.owner ∈ stepVariableOwners)) =
      selectedReceipts canonical := by
  apply receiptLists_eq_of_ownerMaps_eq canonical.program
  · intro receipt member
    exact (List.mem_filter.mp member).1
  · intro receipt member
    simp only [selectedReceipts, List.mem_cons,
      List.not_mem_nil, or_false] at member
    rcases member with baseEqual | recursiveEqual | branchEqual
    · subst receipt
      exact canonical.baseEndpoint.member
    · subst receipt
      exact canonical.recursivePriorLink.member
    · subst receipt
      exact canonical.branchJoin.member
  · rw [map_owner_filter, canonical.receiptOwnersExact]
    simp only [selectedReceipts, List.map_cons, List.map_nil]
    rw [canonical.baseEndpoint.ownerExact,
      canonical.recursivePriorLink.ownerExact,
      canonical.branchJoin.ownerExact]
    simp [SourceOwners.stepOwners, SourceAlignment.inputOwners,
      SourceOwners.stepBodyOwners, stepVariableOwners,
      SourceOwners.stepBaseAssertionPath,
      SourceOwners.stepRecursiveAssertionPath,
      SourceOwners.stepBranchPath,
      SourceOwners.stepBaseStateEqualPath,
      SourceOwners.stepBaseDefaultPath,
      SourceOwners.stepRecursiveEncodedEqualPath,
      SourceOwners.stepRecursiveEncodePath,
      SourceOwners.stepRecursiveFreshPublicPath,
      SourceOwners.stepRecursiveHashPriorPath,
      SourceOwners.stepRecursiveNifsPath,
      SourceOwners.stepContinuationHashPath,
      SourceOwners.stepSelectorPath,
      SourceOwners.stepApplyPath]

private theorem selectedRows_length
    {parameters : Parameters}
    (canonical : Step parameters) :
    (canonical.program.toEncoding.rows.filter
      (fun row => decide (row.id.owner ∈ stepVariableOwners))).length =
        parameters.widths.running + 2 := by
  rw [filteredRows_eq_filteredReceipts]
  rw [canonical.selectedReceipts_exact]
  simp only [selectedReceipts, List.flatMap_cons, List.flatMap_nil,
    List.append_nil]
  rw [canonical.baseEndpoint.rowsExact,
    canonical.recursivePriorLink.rowsExact,
    canonical.branchJoin.rowsExact]
  simp only [GatedAssertion.Candidate.rows, List.length_append,
    List.length_cons, List.length_nil]
  have joinLength :
      (canonical.specifications.joinCoordinates.flatMap
        (fun specification =>
          BranchJoin.Candidate.rows
            .selectedMux specification)).length =
        canonical.specifications.joinCoordinates.length := by
    induction canonical.specifications.joinCoordinates with
    | nil =>
        rfl
    | cons head tail inductionHypothesis =>
        rw [List.flatMap_cons, List.length_append,
          inductionHypothesis]
        simp [BranchJoin.Candidate.rows, Nat.add_comm]
  rw [joinLength, canonical.sitesAligned.joinCoordinateCount]
  omega

/-- Cost conservation is derived from the receipt program and the exact
selected rows.  It is not a field supplied by the certificate. -/
theorem costConserved
    {parameters : Parameters}
    (canonical : Step parameters) :
    canonical.program.toEncoding.cost =
      stepFixedCost canonical.program +
        totalCost
          (NormalForm.stepClasses canonical.specifications)
          (canonicalSelection
            (NormalForm.stepClasses canonical.specifications)) := by
  rw [canonical.sitesAligned.canonicalLocalCost]
  apply physicalCost_eq_fixed_add_rows
  calc
    canonical.program.toEncoding.rows.length =
        (stepFixedRows canonical.program).length +
          (canonical.program.toEncoding.rows.filter
            (fun row =>
              decide (row.id.owner ∈ stepVariableOwners))).length := by
      exact rows_length_eq_fixed_add_selected
        canonical.program.toEncoding.rows
        (fun row => decide (row.id.owner ∈ stepVariableOwners))
    _ = (stepFixedRows canonical.program).length +
        (parameters.widths.running + 2) := by
      rw [canonical.selectedRows_length]

/-- Exact selected step cost: the receipt-computed fixed part plus one mux row
per running coordinate and the two direct assertion rows. -/
theorem exactCost
    {parameters : Parameters}
    (canonical : Step parameters) :
    canonical.encoding.cost =
      stepFixedCost canonical.program +
        ⟨parameters.widths.running + 2, 0, 0, 0⟩ := by
  rw [encoding, canonical.costConserved,
    canonical.sitesAligned.canonicalLocalCost]

/-- The selected physical step cost is no greater than the same fixed program
with any pointwise-admissible member of the exact finite rewrite class. -/
theorem minimum
    {parameters : Parameters}
    (canonical : Step parameters)
    (selection :
      Selection
        (NormalForm.stepClasses canonical.specifications))
    (admissible :
      Admissible
        (NormalForm.stepClasses canonical.specifications)
        selection) :
    Cost.LexLe
      canonical.encoding.cost
      (stepFixedCost canonical.program +
        totalCost
          (NormalForm.stepClasses canonical.specifications)
          selection) := by
  rw [encoding, canonical.costConserved]
  exact NormalForm.stepCanonicalMinimumWithFixedCost
    canonical.specifications
    (stepFixedCost canonical.program)
    selection admissible

/-- Every physical step column identity has exactly one receipt owner named
by the typed source. -/
theorem everyColumnHasExactlyOneSourceOwner
    {parameters : Parameters}
    (canonical : Step parameters)
    (column : ColumnId)
    (member : column ∈ canonical.encoding.columnIds) :
    ∃ receipt,
      receipt ∈ canonical.program.physical.receipts ∧
        receipt.owner ∈ SourceOwners.stepOwners parameters ∧
        column ∈ receipt.columnIds ∧
          ∀ candidate,
            candidate ∈ canonical.program.physical.receipts ->
            column ∈ candidate.columnIds ->
            candidate = receipt := by
  rcases canonical.program.column_identity_has_exactly_one_source_owner
      column member with
    ⟨receipt, receiptMember, expected, columnMember, unique⟩
  rw [SourceOwners.stepProgramOwnersExact] at expected
  exact ⟨receipt, receiptMember, expected, columnMember, unique⟩

/-- Every physical step row identity has exactly one receipt owner named by
the typed source. -/
theorem everyRowHasExactlyOneSourceOwner
    {parameters : Parameters}
    (canonical : Step parameters)
    (row : RowId)
    (member : row ∈ canonical.encoding.rowIds) :
    ∃ receipt,
      receipt ∈ canonical.program.physical.receipts ∧
        receipt.owner ∈ SourceOwners.stepOwners parameters ∧
        row ∈ receipt.rowIds ∧
          ∀ candidate,
            candidate ∈ canonical.program.physical.receipts ->
            row ∈ candidate.rowIds ->
            candidate = receipt := by
  rcases canonical.program.row_identity_has_exactly_one_source_owner
      row member with
    ⟨receipt, receiptMember, expected, rowMember, unique⟩
  rw [SourceOwners.stepProgramOwnersExact] at expected
  exact ⟨receipt, receiptMember, expected, rowMember, unique⟩

/-- No physical step columns exist outside emission receipts. -/
theorem columnsConserved
    {parameters : Parameters}
    (canonical : Step parameters) :
    canonical.encoding.columns =
      canonical.program.physical.receipts.flatMap
        (fun receipt => receipt.allocations) :=
  rfl

/-- No physical step rows exist outside emission receipts. -/
theorem rowsConserved
    {parameters : Parameters}
    (canonical : Step parameters) :
    canonical.encoding.rows =
      canonical.program.physical.receipts.flatMap
        (fun receipt => receipt.rows) :=
  rfl

/-- Physical step cost is the exact receipt fold, independently of the
normal-form decomposition view. -/
theorem costIsReceiptFold
    {parameters : Parameters}
    (canonical : Step parameters) :
    canonical.encoding.cost =
      Cost.sum
        (canonical.program.physical.receipts.map
          InstructionReceipt.cost) :=
  canonical.program.cost_eq_receipt_cost

/-- Bundled obligation-10 claim for the selected fixed-one step encoding.
This is deliberately silent about semantic R1CS refinement, which belongs to
obligation 11. -/
structure Claims
    {parameters : Parameters}
    (canonical : Step parameters) : Prop where
  ownersExact :
    canonical.program.physical.receipts.map
        (fun receipt => receipt.owner) =
      SourceOwners.stepOwners parameters
  everySourceOwner :
    ∀ owner, owner ∈ SourceOwners.stepOwners parameters ->
      ∃ receipt,
        receipt ∈ canonical.program.physical.receipts ∧
          receipt.owner = owner ∧
            ∀ candidate,
              candidate ∈ canonical.program.physical.receipts ->
              candidate.owner = owner ->
              candidate = receipt
  everyColumn :
    ∀ column, column ∈ canonical.encoding.columnIds ->
      ∃ receipt,
        receipt ∈ canonical.program.physical.receipts ∧
          receipt.owner ∈ SourceOwners.stepOwners parameters ∧
          column ∈ receipt.columnIds ∧
            ∀ candidate,
              candidate ∈ canonical.program.physical.receipts ->
              column ∈ candidate.columnIds ->
              candidate = receipt
  everyRow :
    ∀ row, row ∈ canonical.encoding.rowIds ->
      ∃ receipt,
        receipt ∈ canonical.program.physical.receipts ∧
          receipt.owner ∈ SourceOwners.stepOwners parameters ∧
          row ∈ receipt.rowIds ∧
            ∀ candidate,
              candidate ∈ canonical.program.physical.receipts ->
              row ∈ candidate.rowIds ->
              candidate = receipt
  columnsConserved :
    canonical.encoding.columns =
      canonical.program.physical.receipts.flatMap
        (fun receipt => receipt.allocations)
  rowsConserved :
    canonical.encoding.rows =
      canonical.program.physical.receipts.flatMap
        (fun receipt => receipt.rows)
  receiptCostExact :
    canonical.encoding.cost =
      Cost.sum
        (canonical.program.physical.receipts.map
          InstructionReceipt.cost)
  selectedCostExact :
    canonical.encoding.cost =
      stepFixedCost canonical.program +
        ⟨parameters.widths.running + 2, 0, 0, 0⟩
  finiteClassMinimum :
    ∀ selection :
        Selection
          (NormalForm.stepClasses canonical.specifications),
      Admissible
          (NormalForm.stepClasses canonical.specifications)
          selection ->
        Cost.LexLe
          canonical.encoding.cost
          (stepFixedCost canonical.program +
            totalCost
              (NormalForm.stepClasses canonical.specifications)
              selection)

/-- Conditional obligation-10 theorem for a supplied exact fixed-one step
encoding certificate.  Constructing that certificate is a separate
realization result. -/
theorem obligation10_of_certificate
    {parameters : Parameters}
    (canonical : Step parameters) :
    Claims canonical where
  ownersExact := canonical.receiptOwnersExact
  everySourceOwner :=
    canonical.everySourceOwnerHasExactlyOneReceipt
  everyColumn := canonical.everyColumnHasExactlyOneSourceOwner
  everyRow := canonical.everyRowHasExactlyOneSourceOwner
  columnsConserved := canonical.columnsConserved
  rowsConserved := canonical.rowsConserved
  receiptCostExact := canonical.costIsReceiptFold
  selectedCostExact := canonical.exactCost
  finiteClassMinimum := canonical.minimum

end Step

/-! ## Terminal certificate -/

def terminalVariableOwners : List PhysicalOwner :=
  [.typed (.instruction SourceOwners.terminalBaseAssertionPath),
    .typed (.instruction
      SourceOwners.terminalRecursivePriorAssertionPath),
    .typed (.instruction
      SourceOwners.terminalRecursiveRunningAssertionPath),
    .typed (.instruction
      SourceOwners.terminalRecursiveFreshAssertionPath)]

def terminalFixedRows
    {parameters : Parameters}
    (program :
      AlignedReceiptProgram
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program
          parameters)) : List OwnedRow :=
  program.toEncoding.rows.filter
    (fun row => !(decide (row.id.owner ∈ terminalVariableOwners)))

def terminalFixedCost
    {parameters : Parameters}
    (program :
      AlignedReceiptProgram
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program
          parameters)) : Cost :=
  physicalCost program.toEncoding.columns (terminalFixedRows program)

/-- Selected canonical encoding of the exact fixed-one terminal source
program. -/
structure Terminal
    (parameters : Parameters) where
  profile :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Profile
      parameters
  specifications : NormalForm.TerminalSpecifications
  sitesAligned :
    SourceOwners.TerminalNormalFormAligned specifications
  program :
    AlignedReceiptProgram
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program
        parameters)
  emptyJoin :
    ExactReceipt program
      (.typed (.branch SourceOwners.terminalBranchPath))
      .branchJoin
      []
      []
  baseEndpoint :
    ExactReceipt program
      (.typed (.instruction SourceOwners.terminalBaseAssertionPath))
      .assertion
      []
      (GatedAssertion.Candidate.rows
        .direct specifications.baseEndpoint)
  recursivePriorLink :
    ExactReceipt program
      (.typed (.instruction
        SourceOwners.terminalRecursivePriorAssertionPath))
      .assertion
      []
      (GatedAssertion.Candidate.rows
        .direct specifications.recursivePriorLink)
  runningRelation :
    ExactReceipt program
      (.typed (.instruction
        SourceOwners.terminalRecursiveRunningAssertionPath))
      .assertion
      []
      (GatedAssertion.Candidate.rows
        .direct specifications.runningRelation)
  freshRelation :
    ExactReceipt program
      (.typed (.instruction
        SourceOwners.terminalRecursiveFreshAssertionPath))
      .assertion
      []
      (GatedAssertion.Candidate.rows
        .direct specifications.freshRelation)

namespace Terminal

def encoding
    {parameters : Parameters}
    (canonical : Terminal parameters) :
    Nightstream.Implementation.Lowering.Goldilocks.Encoding
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program
        parameters) :=
  canonical.program.toEncoding

theorem receiptOwnersExact
    {parameters : Parameters}
    (canonical : Terminal parameters) :
    canonical.program.physical.receipts.map
        (fun receipt => receipt.owner) =
      SourceOwners.terminalOwners parameters := by
  rw [canonical.program.ownersExact,
    SourceOwners.terminalProgramOwnersExact]

theorem everySourceOwnerHasExactlyOneReceipt
    {parameters : Parameters}
    (canonical : Terminal parameters)
    (owner : PhysicalOwner)
    (expected : owner ∈ SourceOwners.terminalOwners parameters) :
    ∃ receipt,
      receipt ∈ canonical.program.physical.receipts ∧
        receipt.owner = owner ∧
          ∀ candidate,
            candidate ∈ canonical.program.physical.receipts ->
            candidate.owner = owner ->
            candidate = receipt := by
  apply canonical.program.expected_owner_has_exactly_one_receipt owner
  rw [SourceOwners.terminalProgramOwnersExact]
  exact expected

private def selectedReceipts
    {parameters : Parameters}
    (canonical : Terminal parameters) : List InstructionReceipt :=
  [canonical.baseEndpoint.receipt,
    canonical.recursivePriorLink.receipt,
    canonical.runningRelation.receipt,
    canonical.freshRelation.receipt]

private theorem selectedReceipts_exact
    {parameters : Parameters}
    (canonical : Terminal parameters) :
    canonical.program.physical.receipts.filter
        (fun receipt =>
          decide (receipt.owner ∈ terminalVariableOwners)) =
      selectedReceipts canonical := by
  apply receiptLists_eq_of_ownerMaps_eq canonical.program
  · intro receipt member
    exact (List.mem_filter.mp member).1
  · intro receipt member
    simp only [selectedReceipts, List.mem_cons,
      List.not_mem_nil, or_false] at member
    rcases member with
      baseEqual | priorEqual | runningEqual | freshEqual
    · subst receipt
      exact canonical.baseEndpoint.member
    · subst receipt
      exact canonical.recursivePriorLink.member
    · subst receipt
      exact canonical.runningRelation.member
    · subst receipt
      exact canonical.freshRelation.member
  · rw [map_owner_filter, canonical.receiptOwnersExact]
    simp only [selectedReceipts, List.map_cons, List.map_nil]
    rw [canonical.baseEndpoint.ownerExact,
      canonical.recursivePriorLink.ownerExact,
      canonical.runningRelation.ownerExact,
      canonical.freshRelation.ownerExact]
    simp [SourceOwners.terminalOwners, SourceAlignment.inputOwners,
      SourceOwners.terminalBodyOwners, terminalVariableOwners,
      SourceOwners.terminalSelectorPath,
      SourceOwners.terminalBranchPath,
      SourceOwners.terminalBaseStateEqualPath,
      SourceOwners.terminalBaseAssertionPath,
      SourceOwners.terminalRecursiveHashPriorPath,
      SourceOwners.terminalRecursiveFreshPublicPath,
      SourceOwners.terminalRecursiveEncodePath,
      SourceOwners.terminalRecursiveEncodedEqualPath,
      SourceOwners.terminalRecursivePriorAssertionPath,
      SourceOwners.terminalRecursiveRunningCheckPath,
      SourceOwners.terminalRecursiveRunningAssertionPath,
      SourceOwners.terminalRecursiveFreshCheckPath,
      SourceOwners.terminalRecursiveFreshAssertionPath]

private theorem selectedRows_length
    {parameters : Parameters}
    (canonical : Terminal parameters) :
    (canonical.program.toEncoding.rows.filter
      (fun row =>
        decide (row.id.owner ∈ terminalVariableOwners))).length = 4 := by
  rw [filteredRows_eq_filteredReceipts]
  rw [canonical.selectedReceipts_exact]
  simp only [selectedReceipts, List.flatMap_cons, List.flatMap_nil,
    List.append_nil]
  rw [canonical.baseEndpoint.rowsExact,
    canonical.recursivePriorLink.rowsExact,
    canonical.runningRelation.rowsExact,
    canonical.freshRelation.rowsExact]
  rfl

/-- Terminal cost conservation is derived from the conserved receipt program
and the exact four selected assertion rows. -/
theorem costConserved
    {parameters : Parameters}
    (canonical : Terminal parameters) :
    canonical.program.toEncoding.cost =
      terminalFixedCost canonical.program +
        totalCost
          (NormalForm.terminalClasses canonical.specifications)
          (canonicalSelection
            (NormalForm.terminalClasses canonical.specifications)) := by
  rw [NormalForm.terminalCanonicalLocalCost]
  apply physicalCost_eq_fixed_add_rows
  calc
    canonical.program.toEncoding.rows.length =
        (terminalFixedRows canonical.program).length +
          (canonical.program.toEncoding.rows.filter
            (fun row =>
              decide
                (row.id.owner ∈ terminalVariableOwners))).length := by
      exact rows_length_eq_fixed_add_selected
        canonical.program.toEncoding.rows
        (fun row =>
          decide (row.id.owner ∈ terminalVariableOwners))
    _ = (terminalFixedRows canonical.program).length + 4 := by
      rw [canonical.selectedRows_length]

/-- Exact selected terminal cost: the receipt-computed fixed part plus four
direct assertion rows. -/
theorem exactCost
    {parameters : Parameters}
    (canonical : Terminal parameters) :
    canonical.encoding.cost =
      terminalFixedCost canonical.program +
        ⟨4, 0, 0, 0⟩ := by
  rw [encoding, canonical.costConserved,
    NormalForm.terminalCanonicalLocalCost]

theorem minimum
    {parameters : Parameters}
    (canonical : Terminal parameters)
    (selection :
      Selection
        (NormalForm.terminalClasses canonical.specifications))
    (admissible :
      Admissible
        (NormalForm.terminalClasses canonical.specifications)
        selection) :
    Cost.LexLe
      canonical.encoding.cost
      (terminalFixedCost canonical.program +
        totalCost
          (NormalForm.terminalClasses canonical.specifications)
          selection) := by
  rw [encoding, canonical.costConserved]
  exact NormalForm.terminalCanonicalMinimumWithFixedCost
    canonical.specifications
    (terminalFixedCost canonical.program)
    selection admissible

theorem everyColumnHasExactlyOneSourceOwner
    {parameters : Parameters}
    (canonical : Terminal parameters)
    (column : ColumnId)
    (member : column ∈ canonical.encoding.columnIds) :
    ∃ receipt,
      receipt ∈ canonical.program.physical.receipts ∧
        receipt.owner ∈ SourceOwners.terminalOwners parameters ∧
        column ∈ receipt.columnIds ∧
          ∀ candidate,
            candidate ∈ canonical.program.physical.receipts ->
            column ∈ candidate.columnIds ->
            candidate = receipt := by
  rcases canonical.program.column_identity_has_exactly_one_source_owner
      column member with
    ⟨receipt, receiptMember, expected, columnMember, unique⟩
  rw [SourceOwners.terminalProgramOwnersExact] at expected
  exact ⟨receipt, receiptMember, expected, columnMember, unique⟩

theorem everyRowHasExactlyOneSourceOwner
    {parameters : Parameters}
    (canonical : Terminal parameters)
    (row : RowId)
    (member : row ∈ canonical.encoding.rowIds) :
    ∃ receipt,
      receipt ∈ canonical.program.physical.receipts ∧
        receipt.owner ∈ SourceOwners.terminalOwners parameters ∧
        row ∈ receipt.rowIds ∧
          ∀ candidate,
            candidate ∈ canonical.program.physical.receipts ->
            row ∈ candidate.rowIds ->
            candidate = receipt := by
  rcases canonical.program.row_identity_has_exactly_one_source_owner
      row member with
    ⟨receipt, receiptMember, expected, rowMember, unique⟩
  rw [SourceOwners.terminalProgramOwnersExact] at expected
  exact ⟨receipt, receiptMember, expected, rowMember, unique⟩

theorem columnsConserved
    {parameters : Parameters}
    (canonical : Terminal parameters) :
    canonical.encoding.columns =
      canonical.program.physical.receipts.flatMap
        (fun receipt => receipt.allocations) :=
  rfl

theorem rowsConserved
    {parameters : Parameters}
    (canonical : Terminal parameters) :
    canonical.encoding.rows =
      canonical.program.physical.receipts.flatMap
        (fun receipt => receipt.rows) :=
  rfl

theorem costIsReceiptFold
    {parameters : Parameters}
    (canonical : Terminal parameters) :
    canonical.encoding.cost =
      Cost.sum
        (canonical.program.physical.receipts.map
          InstructionReceipt.cost) :=
  canonical.program.cost_eq_receipt_cost

/-- Bundled obligation-10 claim for the selected fixed-one terminal
encoding. -/
structure Claims
    {parameters : Parameters}
    (canonical : Terminal parameters) : Prop where
  ownersExact :
    canonical.program.physical.receipts.map
        (fun receipt => receipt.owner) =
      SourceOwners.terminalOwners parameters
  everySourceOwner :
    ∀ owner, owner ∈ SourceOwners.terminalOwners parameters ->
      ∃ receipt,
        receipt ∈ canonical.program.physical.receipts ∧
          receipt.owner = owner ∧
            ∀ candidate,
              candidate ∈ canonical.program.physical.receipts ->
              candidate.owner = owner ->
              candidate = receipt
  everyColumn :
    ∀ column, column ∈ canonical.encoding.columnIds ->
      ∃ receipt,
        receipt ∈ canonical.program.physical.receipts ∧
          receipt.owner ∈ SourceOwners.terminalOwners parameters ∧
          column ∈ receipt.columnIds ∧
            ∀ candidate,
              candidate ∈ canonical.program.physical.receipts ->
              column ∈ candidate.columnIds ->
              candidate = receipt
  everyRow :
    ∀ row, row ∈ canonical.encoding.rowIds ->
      ∃ receipt,
        receipt ∈ canonical.program.physical.receipts ∧
          receipt.owner ∈ SourceOwners.terminalOwners parameters ∧
          row ∈ receipt.rowIds ∧
            ∀ candidate,
              candidate ∈ canonical.program.physical.receipts ->
              row ∈ candidate.rowIds ->
              candidate = receipt
  columnsConserved :
    canonical.encoding.columns =
      canonical.program.physical.receipts.flatMap
        (fun receipt => receipt.allocations)
  rowsConserved :
    canonical.encoding.rows =
      canonical.program.physical.receipts.flatMap
        (fun receipt => receipt.rows)
  receiptCostExact :
    canonical.encoding.cost =
      Cost.sum
        (canonical.program.physical.receipts.map
          InstructionReceipt.cost)
  selectedCostExact :
    canonical.encoding.cost =
      terminalFixedCost canonical.program +
        ⟨4, 0, 0, 0⟩
  finiteClassMinimum :
    ∀ selection :
        Selection
          (NormalForm.terminalClasses canonical.specifications),
      Admissible
          (NormalForm.terminalClasses canonical.specifications)
          selection ->
        Cost.LexLe
          canonical.encoding.cost
          (terminalFixedCost canonical.program +
            totalCost
              (NormalForm.terminalClasses canonical.specifications)
              selection)

/-- Conditional obligation-10 theorem for a supplied exact fixed-one terminal
encoding certificate.  Constructing that certificate is a separate
realization result. -/
theorem obligation10_of_certificate
    {parameters : Parameters}
    (canonical : Terminal parameters) :
    Claims canonical where
  ownersExact := canonical.receiptOwnersExact
  everySourceOwner :=
    canonical.everySourceOwnerHasExactlyOneReceipt
  everyColumn := canonical.everyColumnHasExactlyOneSourceOwner
  everyRow := canonical.everyRowHasExactlyOneSourceOwner
  columnsConserved := canonical.columnsConserved
  rowsConserved := canonical.rowsConserved
  receiptCostExact := canonical.costIsReceiptFold
  selectedCostExact := canonical.exactCost
  finiteClassMinimum := canonical.minimum

end Terminal

end CanonicalEncoding

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
