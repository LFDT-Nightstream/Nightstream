import NightstreamFPrime.Layout.PilotProduction

/-!
Owns the production pilot's public/private column partition and its exact
permutation into the direct Spartan order: private columns, the constant
column, then public columns. It does not own constraints or witness values.
-/

namespace NightstreamFPrime.Layout.PilotSpartan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout

/-- Materialized source-column count for executable package emission. -/
def SourceColumnCount : Nat := 12659088

/-- The recursive public input has 54 cells and `XOut` has four cells. -/
def publicColumnCount : Nat :=
  PriorStateHash.publicWidth + PilotProduction.digestWords

/-- All hash-preimage and recipe columns are private advice. -/
def privateColumnCount : Nat :=
  2 * PilotProduction.stateHashWords +
    2 * PilotProduction.hashWitnessCount

/-- Spartan inserts its constant column between private and public columns. -/
def constantColumn : Nat := privateColumnCount

def spartanColumnCount : Nat :=
  privateColumnCount + 1 + publicColumnCount

/-- Exact source-order boundaries of the fixed production pilot. -/
def priorPublicStart : Nat := PilotProduction.stateHashWords
def outputPreimageStart : Nat :=
  priorPublicStart + PriorStateHash.publicWidth
def outputDigestStart : Nat :=
  outputPreimageStart + PilotProduction.stateHashWords
def witnessStart : Nat :=
  outputDigestStart + PilotProduction.digestWords
def secondPrivateStart : Nat := PilotProduction.stateHashWords
def witnessPrivateStart : Nat := 2 * PilotProduction.stateHashWords
def firstPublicStart : Nat := privateColumnCount + 1
def secondPublicStart : Nat := firstPublicStart + PriorStateHash.publicWidth

theorem sourceColumnCount_eq : SourceColumnCount = 12659088 := by
  rfl

/-- The materialized count is exactly the proved semantic pilot layout count. -/
theorem sourceColumnCount_matches :
    SourceColumnCount =
      Pilot.physicalColumnCount PilotProduction.interface
        PilotProduction.witnessOffset := by
  rw [PilotProduction.physicalColumnCount_eq, sourceColumnCount_eq]

theorem publicColumnCount_eq : publicColumnCount =
    PriorStateHash.publicWidth + PilotProduction.digestWords := by
  rfl

theorem privateColumnCount_eq :
    privateColumnCount + publicColumnCount = SourceColumnCount := by
  rw [sourceColumnCount_eq]
  norm_num [privateColumnCount, publicColumnCount,
    PilotProduction.stateHashWords_eq,
    PilotProduction.hashWitnessCount_eq,
    PilotProduction.digestWords, PilotValues.digestWords,
    PriorStateHash.publicWidth_eq]

theorem spartanColumnCount_eq : spartanColumnCount = SourceColumnCount + 1 := by
  rw [sourceColumnCount_eq]
  norm_num [spartanColumnCount, privateColumnCount, publicColumnCount,
    PilotProduction.stateHashWords_eq,
    PilotProduction.hashWitnessCount_eq,
    PilotProduction.digestWords, PilotValues.digestWords,
    PriorStateHash.publicWidth_eq]

theorem sourceBoundaries_eq :
    priorPublicStart = PilotProduction.priorPublicInputStart ∧
      outputPreimageStart = PilotProduction.outputPreimageStart ∧
      outputDigestStart = PilotProduction.outputDigestStart ∧
      witnessStart = PilotProduction.witnessOffset := by
  exact ⟨rfl, rfl, rfl, rfl⟩

/-- Visibility in the Lean source-column order. -/
inductive Visibility where
  | privateInput
  | publicInput
deriving Repr, DecidableEq

def visibility (column : Nat) : Visibility :=
  if PilotProduction.priorPublicInputStart ≤ column ∧
      column < PilotProduction.outputPreimageStart then
    .publicInput
  else if PilotProduction.outputDigestStart ≤ column ∧
      column < PilotProduction.witnessOffset then
    .publicInput
  else
    .privateInput

/-- Map a Lean physical column to direct Spartan's private/constant/public
order. The image never contains `constantColumn`. -/
def sourceToSpartan (column : Nat) : Nat :=
  if column < priorPublicStart then
    column
  else if column < outputPreimageStart then
    firstPublicStart + (column - priorPublicStart)
  else if column < outputDigestStart then
    secondPrivateStart + (column - outputPreimageStart)
  else if column < witnessStart then
    secondPublicStart + (column - outputDigestStart)
  else
    witnessPrivateStart + (column - witnessStart)

/-- Partial inverse because the Spartan constant column has no Lean source
column. -/
def spartanToSource (column : Nat) : Option Nat :=
  if column < secondPrivateStart then
    some column
  else if column < witnessPrivateStart then
    some (outputPreimageStart + (column - secondPrivateStart))
  else if column < privateColumnCount then
    some (witnessStart + (column - witnessPrivateStart))
  else if column = constantColumn then
    none
  else if column < secondPublicStart then
    some (priorPublicStart + (column - firstPublicStart))
  else if column < spartanColumnCount then
    some (outputDigestStart + (column - secondPublicStart))
  else
    none

theorem sourceToSpartan_lt (column : Nat) (bound : column < SourceColumnCount) :
    sourceToSpartan column < spartanColumnCount := by
  rw [sourceColumnCount_eq] at bound
  unfold sourceToSpartan
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals norm_num [spartanColumnCount, privateColumnCount,
    publicColumnCount, priorPublicStart, outputPreimageStart,
    outputDigestStart, witnessStart, secondPrivateStart,
    witnessPrivateStart, firstPublicStart, secondPublicStart,
    PilotProduction.stateHashWords_eq,
    PilotProduction.hashWitnessCount_eq,
    PilotProduction.digestWords, PilotValues.digestWords,
    PriorStateHash.publicWidth_eq] at * <;> omega

theorem spartanToSource_sourceToSpartan (column : Nat)
    (bound : column < SourceColumnCount) :
    spartanToSource (sourceToSpartan column) = some column := by
  rw [sourceColumnCount_eq] at bound
  unfold sourceToSpartan spartanToSource
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals norm_num [privateColumnCount, constantColumn,
    spartanColumnCount, publicColumnCount, priorPublicStart,
    outputPreimageStart, outputDigestStart, witnessStart,
    secondPrivateStart, witnessPrivateStart, firstPublicStart,
    secondPublicStart, PilotProduction.stateHashWords_eq,
    PilotProduction.hashWitnessCount_eq,
    PilotProduction.digestWords, PilotValues.digestWords,
    PriorStateHash.publicWidth_eq] at * <;> omega

theorem sourceToSpartan_ne_constant (column : Nat)
    (bound : column < SourceColumnCount) :
    sourceToSpartan column ≠ constantColumn := by
  rw [sourceColumnCount_eq] at bound
  unfold sourceToSpartan constantColumn
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals norm_num [privateColumnCount, priorPublicStart,
    outputPreimageStart, outputDigestStart, witnessStart,
    secondPrivateStart, witnessPrivateStart, firstPublicStart,
    secondPublicStart, PilotProduction.stateHashWords_eq,
    PilotProduction.hashWitnessCount_eq,
    PilotProduction.digestWords, PilotValues.digestWords,
    PriorStateHash.publicWidth_eq] at * <;> omega

theorem sourceToSpartan_injective {left right : Nat}
    (leftBound : left < SourceColumnCount)
    (rightBound : right < SourceColumnCount)
    (equals : sourceToSpartan left = sourceToSpartan right) : left = right := by
  have leftInverse := spartanToSource_sourceToSpartan left leftBound
  have rightInverse := spartanToSource_sourceToSpartan right rightBound
  rw [equals, rightInverse] at leftInverse
  exact (Option.some.inj leftInverse).symm

def pullback (target : Env) : Env :=
  fun source => target (sourceToSpartan source)

def remapCombination (combination : R1CS.LinearCombination) :
    R1CS.LinearCombination :=
  ⟨combination.constant,
    combination.terms.map fun term => (sourceToSpartan term.1, term.2)⟩

theorem remapCombination_eval (target : Env)
    (combination : R1CS.LinearCombination) :
    (remapCombination combination).eval target =
      combination.eval (pullback target) := by
  unfold remapCombination R1CS.LinearCombination.eval pullback
  rw [List.map_map]
  congr 1

def remapRow (row : R1CS.Row) : R1CS.Row :=
  ⟨remapCombination row.a, remapCombination row.b,
    remapCombination row.c⟩

theorem remapRow_holds (target : Env) (row : R1CS.Row) :
    (remapRow row).Holds target ↔ row.Holds (pullback target) := by
  simp [R1CS.Row.Holds, remapRow, remapCombination_eval]

def remapRows (rows : List R1CS.Row) : List R1CS.Row :=
  rows.map remapRow

theorem remapRows_hold (target : Env) (rows : List R1CS.Row) :
    R1CS.RowsHold target (remapRows rows) ↔
      R1CS.RowsHold (pullback target) rows := by
  constructor
  · intro holds row member
    have remapped := holds (remapRow row) (by
      rw [remapRows, List.mem_map]
      exact ⟨row, member, rfl⟩)
    exact (remapRow_holds target row).mp remapped
  · intro holds row member
    rw [remapRows, List.mem_map] at member
    rcases member with ⟨source, sourceMember, rfl⟩
    exact (remapRow_holds target source).mpr (holds source sourceMember)

/-- The generic Spartan variable and row domains use the production cube. -/
def domainSize : Nat := 2 ^ cubeVariables

/-- Spartan moves the constant and public columns after the padded private
domain. -/
def paddedConstantColumn : Nat := domainSize

def paddedSpartanColumnCount : Nat :=
  domainSize + 1 + publicColumnCount

/-- Roles introduced by the generic Spartan padding step. Source-private rows
and columns retain their detailed owners from `Layout.Pilot`. -/
inductive PaddedColumnRole where
  | sourcePrivate (index : Nat)
  | privatePadding (index : Nat)
  | constant
  | publicInput (index : Nat)
deriving Repr, DecidableEq

def paddedColumnRole (column : Fin paddedSpartanColumnCount) :
    PaddedColumnRole :=
  if column.val < privateColumnCount then
    .sourcePrivate column.val
  else if column.val < domainSize then
    .privatePadding (column.val - privateColumnCount)
  else if column.val = paddedConstantColumn then
    .constant
  else
    .publicInput (column.val - (paddedConstantColumn + 1))

inductive PaddedRowRole where
  | source (index : Nat)
  | zeroPadding (index : Nat)
deriving Repr, DecidableEq

def sourceRows (_unit : Unit) : List R1CS.Row :=
  Pilot.physicalRows PilotProduction.interface PilotProduction.witnessOffset

def paddedRowRole (row : Fin domainSize) : PaddedRowRole :=
  if row.val < (sourceRows ()).length then
    .source row.val
  else
    .zeroPadding (row.val - (sourceRows ()).length)

theorem sourceRowCount_bounds :
    2 ^ 23 < (sourceRows ()).length ∧
      (sourceRows ()).length ≤ domainSize := by
  change 2 ^ 23 <
      Pilot.physicalRowCount PilotProduction.interface
        PilotProduction.witnessOffset ∧
    Pilot.physicalRowCount PilotProduction.interface
        PilotProduction.witnessOffset ≤ 2 ^ cubeVariables
  rw [PilotProduction.physicalRowCount_eq]
  norm_num [cubeVariables]

theorem privateColumnCount_bounds :
    2 ^ 23 < privateColumnCount ∧ privateColumnCount ≤ domainSize := by
  norm_num [privateColumnCount, domainSize, cubeVariables,
    PilotProduction.stateHashWords_eq,
    PilotProduction.hashWitnessCount_eq]

/-- Column index used by the padded Spartan matrices. -/
def spartanToPadded (column : Nat) : Nat :=
  if column < constantColumn then
    column
  else
    domainSize + (column - constantColumn)

def paddedPullback (target : Env) : Env :=
  fun column => target (spartanToPadded column)

def padCombination (combination : R1CS.LinearCombination) :
    R1CS.LinearCombination :=
  ⟨combination.constant,
    combination.terms.map fun term => (spartanToPadded term.1, term.2)⟩

private theorem padCombination_eval (target : Env)
    (combination : R1CS.LinearCombination) :
    (padCombination combination).eval target =
      combination.eval (paddedPullback target) := by
  unfold padCombination R1CS.LinearCombination.eval paddedPullback
  rw [List.map_map]
  congr 1

def padRow (row : R1CS.Row) : R1CS.Row :=
  ⟨padCombination row.a, padCombination row.b, padCombination row.c⟩

private theorem padRow_holds (target : Env) (row : R1CS.Row) :
    (padRow row).Holds target ↔ row.Holds (paddedPullback target) := by
  simp [R1CS.Row.Holds, padRow, padCombination_eval]

def padRows (rows : List R1CS.Row) : List R1CS.Row :=
  rows.map padRow

private theorem padRows_hold (target : Env) (rows : List R1CS.Row) :
    R1CS.RowsHold target (padRows rows) ↔
      R1CS.RowsHold (paddedPullback target) rows := by
  constructor
  · intro holds row member
    have padded := holds (padRow row) (by
      rw [padRows, List.mem_map]
      exact ⟨row, member, rfl⟩)
    exact (padRow_holds target row).mp padded
  · intro holds row member
    rw [padRows, List.mem_map] at member
    rcases member with ⟨source, sourceMember, rfl⟩
    exact (padRow_holds target source).mpr (holds source sourceMember)

def zeroRow : R1CS.Row :=
  ⟨R1CS.LinearCombination.zero, R1CS.LinearCombination.zero,
    R1CS.LinearCombination.zero⟩

private theorem zeroRows_hold (target : Env) (count : Nat) :
    R1CS.RowsHold target (List.replicate count zeroRow) := by
  intro row member
  have equals : row = zeroRow := by
    simpa using (List.eq_of_mem_replicate member)
  subst row
  simp [zeroRow, R1CS.Row.Holds]

/-- Exact matrix rows seen by direct Spartan: remapped source rows followed by
semantically empty rows up to the fixed production domain. -/
def paddedRows (_unit : Unit) : List R1CS.Row :=
  padRows (remapRows (sourceRows ())) ++
    List.replicate (domainSize - (sourceRows ()).length) zeroRow

theorem paddedRows_length : (paddedRows ()).length = domainSize := by
  unfold paddedRows padRows remapRows
  rw [List.length_append, List.length_map, List.length_map,
    List.length_replicate, Nat.add_sub_of_le sourceRowCount_bounds.2]

/-- Generic Spartan padding preserves and reflects every Lean source row. -/
theorem paddedRows_hold (target : Env) :
    R1CS.RowsHold target (paddedRows ()) ↔
      Pilot.PhysicalHolds PilotProduction.interface
        PilotProduction.witnessOffset
        (pullback (paddedPullback target)) := by
  change R1CS.RowsHold target (paddedRows ()) ↔
    R1CS.RowsHold (pullback (paddedPullback target)) (sourceRows ())
  constructor
  · intro holds
    have split := (R1CS.rowsHold_append target _ _).mp holds
    exact (remapRows_hold (paddedPullback target) (sourceRows ())).mp
      ((padRows_hold target (remapRows (sourceRows ()))).mp split.1)
  · intro holds
    apply (R1CS.rowsHold_append target _ _).mpr
    exact ⟨
      (padRows_hold target (remapRows (sourceRows ()))).mpr
        ((remapRows_hold (paddedPullback target) (sourceRows ())).mpr holds),
      zeroRows_hold target _⟩

end NightstreamFPrime.Layout.PilotSpartan
