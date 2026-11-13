import NightstreamFPrime.Layout.PilotProduction

namespace NightstreamFPrime.Layout.PilotPackageOrder

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout

def SourceColumnCount : Nat := PilotValues.sourceColumnCount

def publicColumnCount : Nat :=
  PriorStateHash.publicWidth + PilotProduction.digestWords

def privateColumnCount : Nat :=
  PilotValues.privateColumnCount

def constantColumn : Nat := privateColumnCount

def packageColumnCount : Nat :=
  privateColumnCount + 1 + publicColumnCount

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

theorem sourceColumnCount_eq : SourceColumnCount = 5343330 := by
  rfl

theorem publicColumnCount_value : publicColumnCount = 324 := by
  rfl

theorem privateColumnCount_value : privateColumnCount = 5343006 := by
  rfl

theorem constantColumn_value : constantColumn = 5343006 := by
  rfl

theorem packageColumnCount_value : packageColumnCount = 5343331 := by
  rfl

theorem priorPublicStart_value : priorPublicStart = 28913 := by
  rfl

theorem outputPreimageStart_value : outputPreimageStart = 29233 := by
  rfl

theorem outputDigestStart_value : outputDigestStart = 58146 := by
  rfl

theorem witnessStart_value : witnessStart = 58150 := by
  rfl

theorem secondPrivateStart_value : secondPrivateStart = 28913 := by
  rfl

theorem witnessPrivateStart_value : witnessPrivateStart = 57826 := by
  rfl

theorem firstPublicStart_value : firstPublicStart = 5343007 := by
  rfl

theorem secondPublicStart_value : secondPublicStart = 5343327 := by
  rfl

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
  rw [privateColumnCount_value, publicColumnCount_value,
    sourceColumnCount_eq]

theorem packageColumnCount_eq : packageColumnCount = SourceColumnCount + 1 := by
  rw [packageColumnCount_value, sourceColumnCount_eq]

theorem sourceBoundaries_eq :
    priorPublicStart = PilotProduction.priorPublicInputStart ∧
      outputPreimageStart = PilotProduction.outputPreimageStart ∧
      outputDigestStart = PilotProduction.outputDigestStart ∧
      witnessStart = PilotProduction.witnessOffset := by
  exact ⟨rfl, rfl, rfl, rfl⟩

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

def sourceToPackage (column : Nat) : Nat :=
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

theorem sourceToPackage_witness (index : Nat) :
    sourceToPackage (witnessStart + index) =
      witnessPrivateStart + index := by
  unfold sourceToPackage
  rw [ite_eq_right (by
    rw [witnessStart_value, priorPublicStart_value]
    omega)]
  rw [ite_eq_right (by
    rw [witnessStart_value, outputPreimageStart_value]
    omega)]
  rw [ite_eq_right (by
    rw [witnessStart_value, outputDigestStart_value]
    omega)]
  rw [ite_eq_right (by omega)]
  omega

theorem sourceToPackage_pilotWitness (index : Nat) :
    sourceToPackage (PilotProduction.witnessOffset + index) =
      witnessPrivateStart + index := by
  rw [show PilotProduction.witnessOffset = witnessStart by rfl]
  exact sourceToPackage_witness index

def packageToSource (column : Nat) : Option Nat :=
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
  else if column < packageColumnCount then
    some (outputDigestStart + (column - secondPublicStart))
  else
    none

theorem sourceToPackage_lt (column : Nat) (bound : column < SourceColumnCount) :
    sourceToPackage column < packageColumnCount := by
  rw [sourceColumnCount_eq] at bound
  unfold sourceToPackage
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals norm_num [packageColumnCount_value, privateColumnCount_value,
    publicColumnCount_value, priorPublicStart_value,
    outputPreimageStart_value, outputDigestStart_value, witnessStart_value,
    secondPrivateStart_value, witnessPrivateStart_value,
    firstPublicStart_value, secondPublicStart_value] at *; omega

theorem packageToSource_sourceToPackage (column : Nat)
    (bound : column < SourceColumnCount) :
    packageToSource (sourceToPackage column) = some column := by
  rw [sourceColumnCount_eq] at bound
  unfold sourceToPackage packageToSource
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
  all_goals norm_num [privateColumnCount_value, constantColumn_value,
    packageColumnCount_value, publicColumnCount_value,
    priorPublicStart_value, outputPreimageStart_value,
    outputDigestStart_value, witnessStart_value,
    secondPrivateStart_value, witnessPrivateStart_value,
    firstPublicStart_value, secondPublicStart_value] at *
  all_goals omega

theorem sourceToPackage_ne_constant (column : Nat)
    (bound : column < SourceColumnCount) :
    sourceToPackage column ≠ constantColumn := by
  rw [sourceColumnCount_eq] at bound
  unfold sourceToPackage constantColumn
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals norm_num [privateColumnCount_value, priorPublicStart_value,
    outputPreimageStart_value, outputDigestStart_value, witnessStart_value,
    secondPrivateStart_value, witnessPrivateStart_value,
    firstPublicStart_value, secondPublicStart_value] at *; omega

theorem sourceToPackage_injective {left right : Nat}
    (leftBound : left < SourceColumnCount)
    (rightBound : right < SourceColumnCount)
    (equals : sourceToPackage left = sourceToPackage right) : left = right := by
  have leftInverse := packageToSource_sourceToPackage left leftBound
  have rightInverse := packageToSource_sourceToPackage right rightBound
  rw [equals, rightInverse] at leftInverse
  exact (Option.some.inj leftInverse).symm

def pullback (target : Env) : Env :=
  fun source => target (sourceToPackage source)

def remapCombination (combination : R1CS.LinearCombination) :
    R1CS.LinearCombination :=
  ⟨combination.constant,
    combination.terms.map fun term => (sourceToPackage term.1, term.2)⟩

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

def domainSize : Nat := 2 ^ cubeVariables

def paddedConstantColumn : Nat := domainSize

def paddedPackageColumnCount : Nat :=
  domainSize + 1 + publicColumnCount

inductive PaddedColumnRole where
  | sourcePrivate (index : Nat)
  | privatePadding (index : Nat)
  | constant
  | publicInput (index : Nat)
deriving Repr, DecidableEq

def paddedColumnRole (column : Fin paddedPackageColumnCount) :
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
    2 ^ 22 < (sourceRows ()).length ∧
      (sourceRows ()).length ≤ domainSize := by
  change 2 ^ 22 <
      Pilot.physicalRowCount PilotProduction.interface
        PilotProduction.witnessOffset ∧
    Pilot.physicalRowCount PilotProduction.interface
        PilotProduction.witnessOffset ≤ 2 ^ cubeVariables
  rw [PilotProduction.physicalRowCount_eq]
  norm_num [cubeVariables]

theorem privateColumnCount_bounds :
    2 ^ 22 < privateColumnCount ∧ privateColumnCount ≤ domainSize := by
  rw [privateColumnCount_value]
  norm_num [domainSize, cubeVariables]

def packageToPadded (column : Nat) : Nat :=
  if column < constantColumn then
    column
  else
    domainSize + (column - constantColumn)

def paddedPullback (target : Env) : Env :=
  fun column => target (packageToPadded column)

def padCombination (combination : R1CS.LinearCombination) :
    R1CS.LinearCombination :=
  ⟨combination.constant,
    combination.terms.map fun term => (packageToPadded term.1, term.2)⟩

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

def paddedRows (_unit : Unit) : List R1CS.Row :=
  padRows (remapRows (sourceRows ())) ++
    List.replicate (domainSize - (sourceRows ()).length) zeroRow

theorem paddedRows_length : (paddedRows ()).length = domainSize := by
  unfold paddedRows padRows remapRows
  rw [List.length_append, List.length_map, List.length_map,
    List.length_replicate, Nat.add_sub_of_le sourceRowCount_bounds.2]

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

end NightstreamFPrime.Layout.PilotPackageOrder
