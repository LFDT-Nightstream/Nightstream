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

abbrev SourceColumnCount : Nat :=
  Pilot.physicalColumnCount PilotProduction.interface
    PilotProduction.witnessOffset

/-- The recursive public input has 54 cells and `XOut` has four cells. -/
def publicColumnCount : Nat := 58

/-- All hash-preimage and recipe columns are private advice. -/
def privateColumnCount : Nat := 12144082

/-- Spartan inserts its constant column between private and public columns. -/
def constantColumn : Nat := privateColumnCount

def spartanColumnCount : Nat := privateColumnCount + 1 + publicColumnCount

/-- Exact source-order boundaries of the fixed production pilot. -/
def priorPublicStart : Nat := 40745
def outputPreimageStart : Nat := 40799
def outputDigestStart : Nat := 81544
def witnessStart : Nat := 81548
def secondPrivateStart : Nat := 40745
def witnessPrivateStart : Nat := 81490
def firstPublicStart : Nat := 12144083
def secondPublicStart : Nat := 12144137

theorem sourceColumnCount_eq : SourceColumnCount = 12144140 :=
  PilotProduction.physicalColumnCount_eq

theorem publicColumnCount_eq : publicColumnCount =
    PriorStateHash.publicWidth + PilotProduction.digestWords := by
  rfl

theorem privateColumnCount_eq :
    privateColumnCount + publicColumnCount = SourceColumnCount := by
  rw [sourceColumnCount_eq]
  rfl

theorem spartanColumnCount_eq : spartanColumnCount = SourceColumnCount + 1 := by
  rw [sourceColumnCount_eq]
  rfl

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
    witnessPrivateStart, firstPublicStart, secondPublicStart] at * <;> omega

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
    secondPublicStart] at * <;> omega

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
    secondPublicStart] at * <;> omega

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

end NightstreamFPrime.Layout.PilotSpartan
