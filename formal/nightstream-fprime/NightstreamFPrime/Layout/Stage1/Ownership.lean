import NightstreamFPrime.Layout.Stage1.Lowering

/-!
Owns the complete top-level row and column partition of the Stage 1 layout.

Each logical child has one adjacent row interval. The five NextPreimage rows
have a separate layout owner. Final columns are partitioned into the validated
private prefix, application witness words, application-local columns, the
constant column, and the public suffix. Nested phase modules retain the finer
ownership inside the validated prefix.
-/

namespace NightstreamFPrime.Layout.Stage1.Ownership

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

structure Interval where
  start : Nat
  count : Nat
deriving DecidableEq, Repr

def Interval.finish (range : Interval) : Nat := range.start + range.count

def Interval.Contains (range : Interval) (index : Nat) : Prop :=
  range.start ≤ index ∧ index < range.finish

inductive RowOwner where
  | priorStateHash
  | outputHash
  | piCcs
  | piRlc
  | piDec
  | runningTransition
  | application
  | nextPreimage
deriving DecidableEq, Repr

/-- The selected application's exact physical row count, named once so
ownership proofs do not reduce its complete constraint list. -/
def applicationRowCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  R1CS.totalRowCount (Lowering.applicationConstraints program)

theorem applicationRowCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    R1CS.totalRowCount (Lowering.applicationConstraints program) =
      applicationRowCount program := by
  rfl

def rowRange (program : Lifecycle.Stage1.Application.Program) :
    RowOwner → Interval
  | .priorStateHash => ⟨0, 7312526⟩
  | .outputHash => ⟨7312526, 7311204⟩
  | .piCcs => ⟨14623730, 5313237⟩
  | .piRlc => ⟨19936967, 8910074⟩
  | .piDec => ⟨28847041, 25488⟩
  | .runningTransition => ⟨28872529, 345495⟩
  | .application =>
      ⟨29218024, applicationRowCount program⟩
  | .nextPreimage =>
      ⟨29218024 + applicationRowCount program, 5⟩

theorem rowRanges_adjacent
    (program : Lifecycle.Stage1.Application.Program) :
    (rowRange program .priorStateHash).finish =
        (rowRange program .outputHash).start ∧
      (rowRange program .outputHash).finish =
        (rowRange program .piCcs).start ∧
      (rowRange program .piCcs).finish =
        (rowRange program .piRlc).start ∧
      (rowRange program .piRlc).finish =
        (rowRange program .piDec).start ∧
      (rowRange program .piDec).finish =
        (rowRange program .runningTransition).start ∧
      (rowRange program .runningTransition).finish =
        (rowRange program .application).start ∧
      (rowRange program .application).finish =
        (rowRange program .nextPreimage).start := by
  constructor
  · norm_num [rowRange, Interval.finish]
  constructor
  · norm_num [rowRange, Interval.finish]
  constructor
  · norm_num [rowRange, Interval.finish]
  constructor
  · norm_num [rowRange, Interval.finish]
  constructor
  · norm_num [rowRange, Interval.finish]
  constructor
  · norm_num [rowRange, Interval.finish]
  · rfl

theorem rowRanges_finish
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    (rowRange program .nextPreimage).finish =
      Lowering.physicalRowCount relation program := by
  change 29218024 + applicationRowCount program + 5 =
    Lowering.physicalRowCount relation program
  rw [Lowering.physicalRowCount_eq]
  rw [applicationRowCount_eq]

def rowOwnerAt (program : Lifecycle.Stage1.Application.Program)
    (row : Nat) : RowOwner :=
  if row < 7312526 then .priorStateHash
  else if row < 14623730 then .outputHash
  else if row < 19936967 then .piCcs
  else if row < 28847041 then .piRlc
  else if row < 28872529 then .piDec
  else if row < 29218024 then .runningTransition
  else if row < 29218024 + applicationRowCount program then
    .application
  else .nextPreimage

def OwnsRow (program : Lifecycle.Stage1.Application.Program)
    (owner : RowOwner) (row : Nat) : Prop :=
  owner = rowOwnerAt program row

theorem selectedRow_in_range
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (row : Nat)
    (bound : row < Lowering.physicalRowCount relation program) :
    (rowRange program (rowOwnerAt program row)).Contains row := by
  rw [Lowering.physicalRowCount_eq, applicationRowCount_eq] at bound
  by_cases h1 : row < 7312526
  · simp [rowOwnerAt, h1, rowRange, Interval.Contains, Interval.finish]
  by_cases h2 : row < 14623730
  · simp [rowOwnerAt, h1, h2, rowRange, Interval.Contains, Interval.finish]
    omega
  by_cases h3 : row < 19936967
  · simp [rowOwnerAt, h1, h2, h3, rowRange, Interval.Contains,
      Interval.finish]
    omega
  by_cases h4 : row < 28847041
  · simp [rowOwnerAt, h1, h2, h3, h4, rowRange, Interval.Contains,
      Interval.finish]
    omega
  by_cases h5 : row < 28872529
  · simp [rowOwnerAt, h1, h2, h3, h4, h5, rowRange, Interval.Contains,
      Interval.finish]
    omega
  by_cases h6 : row < 29218024
  · simp [rowOwnerAt, h1, h2, h3, h4, h5, h6, rowRange,
      Interval.Contains, Interval.finish]
    omega
  by_cases h7 : row < 29218024 + applicationRowCount program
  · simp [rowOwnerAt, h1, h2, h3, h4, h5, h6, h7, rowRange,
      Interval.Contains, Interval.finish]
    omega
  · simp [rowOwnerAt, h1, h2, h3, h4, h5, h6, h7, rowRange,
      Interval.Contains, Interval.finish]
    omega

theorem row_has_unique_owner
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    (row : Nat) (bound : row < Lowering.physicalRowCount relation program) :
    ∃! owner, OwnsRow program owner row ∧
      (rowRange program owner).Contains row := by
  refine ⟨rowOwnerAt program row, ⟨rfl,
    selectedRow_in_range relation program row bound⟩, ?_⟩
  intro owner owns
  exact owns.1

inductive ColumnOwner where
  | validatedPrivatePrefix
  | applicationWitness
  | applicationLocal
  | constant
  | publicColumns
deriving DecidableEq, Repr

def columnRange (program : Lifecycle.Stage1.Application.Program) :
    ColumnOwner → Interval
  | .validatedPrivatePrefix => ⟨0, Spartan.privateColumnCount⟩
  | .applicationWitness =>
      ⟨Spartan.privateColumnCount, program.witnessWordCount⟩
  | .applicationLocal =>
      ⟨ApplicationInputs.localStart program,
        Lowering.applicationPrivateCount program⟩
  | .constant => ⟨Lowering.constantColumn program, 1⟩
  | .publicColumns =>
      ⟨Lowering.constantColumn program + 1, Lowering.publicColumnCount⟩

private theorem applicationLocal_finish_eq_constant
    (program : Lifecycle.Stage1.Application.Program) :
    ApplicationInputs.localStart program +
        Lowering.applicationPrivateCount program =
      Lowering.constantColumn program := by
  unfold Lowering.constantColumn Lowering.addedPrivateColumnCount
    ApplicationInputs.localStart ApplicationInputs.witnessStart
  rw [Spartan.constantColumn_eq_private]
  omega

theorem columnRanges_finish
    (program : Lifecycle.Stage1.Application.Program) :
    (columnRange program .publicColumns).finish =
      Lowering.totalColumnCount program := by
  change Lowering.constantColumn program + 1 +
    Lowering.publicColumnCount = Lowering.totalColumnCount program
  rw [Lowering.totalColumnCount_eq]
  rw [Lowering.constantColumn_eq_privateColumnCount]

def columnOwnerAt (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) : ColumnOwner :=
  if column < Spartan.privateColumnCount then .validatedPrivatePrefix
  else if column < ApplicationInputs.localStart program then .applicationWitness
  else if column < Lowering.constantColumn program then .applicationLocal
  else if column = Lowering.constantColumn program then .constant
  else .publicColumns

def OwnsColumn (program : Lifecycle.Stage1.Application.Program)
    (owner : ColumnOwner) (column : Nat) : Prop :=
  owner = columnOwnerAt program column

theorem selectedColumn_in_range
    (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) (bound : column < Lowering.totalColumnCount program) :
    (columnRange program (columnOwnerAt program column)).Contains column := by
  have localStartEq : ApplicationInputs.localStart program =
      Spartan.privateColumnCount + program.witnessWordCount := by
    rfl
  have constantEq : Lowering.constantColumn program =
      ApplicationInputs.localStart program +
        Lowering.applicationPrivateCount program := by
    exact (applicationLocal_finish_eq_constant program).symm
  rw [Lowering.totalColumnCount_eq] at bound
  by_cases h1 : column < Spartan.privateColumnCount
  · simp [columnOwnerAt, h1, columnRange, Interval.Contains, Interval.finish]
  by_cases h2 : column < ApplicationInputs.localStart program
  · simp [columnOwnerAt, h1, h2, columnRange, Interval.Contains,
      Interval.finish]
    rw [localStartEq] at h2
    omega
  by_cases h3 : column < Lowering.constantColumn program
  · simp [columnOwnerAt, h1, h2, h3, columnRange, Interval.Contains,
      Interval.finish]
    rw [constantEq] at h3
    omega
  by_cases h4 : column = Lowering.constantColumn program
  · have ownerEq : columnOwnerAt program column = .constant := by
      have prefixLe : Spartan.privateColumnCount ≤
          Lowering.constantColumn program := by
        rw [Lowering.constantColumn_eq_privateColumnCount]
        unfold Lowering.privateColumnCount Lowering.addedPrivateColumnCount
        omega
      have localLe : ApplicationInputs.localStart program ≤
          Lowering.constantColumn program := by
        rw [constantEq]
        omega
      simp [columnOwnerAt, h4, Nat.not_lt.mpr prefixLe,
        Nat.not_lt.mpr localLe]
    rw [ownerEq]
    change Lowering.constantColumn program ≤ column ∧
      column < Lowering.constantColumn program + 1
    omega
  · have ownerEq : columnOwnerAt program column = .publicColumns := by
      simp [columnOwnerAt, h1, h2, h3, h4]
    rw [ownerEq]
    change Lowering.constantColumn program + 1 ≤ column ∧
      column < Lowering.constantColumn program + 1 +
        Lowering.publicColumnCount
    rw [Lowering.constantColumn_eq_privateColumnCount] at h3 h4
    rw [Lowering.constantColumn_eq_privateColumnCount]
    omega

theorem column_has_unique_owner
    (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) (bound : column < Lowering.totalColumnCount program) :
    ∃! owner, OwnsColumn program owner column ∧
      (columnRange program owner).Contains column := by
  refine ⟨columnOwnerAt program column, ⟨rfl,
    selectedColumn_in_range program column bound⟩, ?_⟩
  intro owner owns
  exact owns.1

end NightstreamFPrime.Layout.Stage1.Ownership
