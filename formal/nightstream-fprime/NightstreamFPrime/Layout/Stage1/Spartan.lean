import NightstreamFPrime.Layout.PilotSpartan
import NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDEC

/-!
Obligation: Permute the current pilot + PiCCS + PiRLC + PiDEC source layout into the
one production Spartan order: all private columns, the constant column, the
pilot's public columns, then the verifier-context public columns.

The first source interval keeps the proved pilot permutation. The four
verifier-context source words move to the public suffix. The appended proof
input, PiCCS-local, and PiRLC-local intervals are private and fill the interval
between the pilot-private columns and the relocated public columns. This
module changes no row and adds only the generic zero padding required by the
fixed `2^25` domain.
-/

namespace NightstreamFPrime.Layout.Stage1.Spartan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Layout
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Fixed completed pilot source interval. -/
def pilotSourceColumnCount : Nat := 12659088

/-- Fixed completed pilot private interval. -/
def pilotPrivateColumnCount : Nat := 12659030

/-- Caller-supplied pilot private inputs precede all generated witnesses. -/
def pilotInputPrivateColumnCount : Nat := 84950

/-- Caller-supplied PiCCS proof inputs. -/
def proofInputColumnCount : Nat := 29012

/-- Verifier-owned context words that follow the pilot source interval. -/
def expectedContextColumnCount : Nat := 4

/-- Source boundary between public context and private PiCCS proof inputs. -/
def proofInputSourceStart : Nat := 12659092

/-- Source boundary between proof inputs and PiCCS local witnesses. -/
def piCcsPhaseOffset : Nat := 12688104

/-- Target boundary after proof inputs and shifted pilot witnesses. -/
def piCcsLocalStart : Nat := 12688042

/-- Exact private proof-input plus PiCCS-local and PiRLC-local suffix. -/
def appendedPrivateColumnCount : Nat := 13055925

/-- All source columns before Spartan inserts its constant column. -/
def SourceColumnCount : Nat := 25715017

/-- Public columns owned by the closed pilot. -/
def pilotPublicColumnCount : Nat := 58

/-- Pilot public columns followed by four verifier-context words. -/
def publicColumnCount : Nat := 62

/-- The pilot-private prefix followed by all PiCCS columns. -/
def privateColumnCount : Nat := 25714955

def constantColumn : Nat := 25714955

def spartanColumnCount : Nat := 25715018

/-- First final public column owned by the verifier context. -/
def expectedContextPublicStart : Nat := 25715014

theorem appendedPrivateColumnCount_eq :
    appendedPrivateColumnCount = 13055925 := by
  rfl

theorem sourceColumnCount_eq : SourceColumnCount = 25715017 := by
  rfl

theorem pilotSourceColumnCount_matches :
    pilotSourceColumnCount = PilotSpartan.SourceColumnCount := by
  rw [PilotSpartan.sourceColumnCount_eq]
  rfl

theorem pilotPrivateColumnCount_matches :
    pilotPrivateColumnCount = PilotSpartan.privateColumnCount := by
  norm_num [PilotSpartan.privateColumnCount,
    pilotPrivateColumnCount,
    PilotProduction.stateHashWords_eq,
    PilotProduction.hashWitnessCount_eq]

theorem publicColumnCount_eq : publicColumnCount = 62 := by
  rfl

theorem privateColumnCount_eq : privateColumnCount = 25714955 := by
  rfl

theorem constantColumn_eq : constantColumn = 25714955 := by
  exact privateColumnCount_eq

theorem constantColumn_eq_private :
    constantColumn = privateColumnCount := by
  rfl

theorem spartanColumnCount_eq : spartanColumnCount = 25715018 := by
  rfl

theorem sourceColumnCount_decomposition :
    SourceColumnCount =
      pilotSourceColumnCount + expectedContextColumnCount +
        appendedPrivateColumnCount := by
  norm_num [SourceColumnCount, pilotSourceColumnCount,
    expectedContextColumnCount, appendedPrivateColumnCount]

theorem privateColumnCount_decomposition :
    privateColumnCount =
      pilotPrivateColumnCount + appendedPrivateColumnCount := by
  norm_num [privateColumnCount, pilotPrivateColumnCount,
    appendedPrivateColumnCount]

theorem sourceColumnCount_add_constant :
    spartanColumnCount = SourceColumnCount + 1 := by
  rw [sourceColumnCount_eq, spartanColumnCount_eq]

/-- Relocate one pilot-Spartan column into the combined Spartan layout. -/
def liftPilotColumn (column : Nat) : Nat :=
  if column < pilotInputPrivateColumnCount then
    column
  else if column < pilotPrivateColumnCount then
    column + proofInputColumnCount
  else
    privateColumnCount + (column - pilotPrivateColumnCount)

theorem liftPilotColumn_add_of_input (start offset : Nat)
    (bound : start + offset < pilotInputPrivateColumnCount) :
    liftPilotColumn (start + offset) = liftPilotColumn start + offset := by
  have startBound : start < pilotInputPrivateColumnCount := by omega
  unfold liftPilotColumn
  rw [if_pos bound, if_pos startBound]

theorem liftPilotColumn_add_of_private (start offset : Nat)
    (lower : pilotInputPrivateColumnCount ≤ start)
    (upper : start + offset < pilotPrivateColumnCount) :
    liftPilotColumn (start + offset) = liftPilotColumn start + offset := by
  have sumLower : pilotInputPrivateColumnCount ≤ start + offset := by omega
  have startUpper : start < pilotPrivateColumnCount := by omega
  unfold liftPilotColumn
  rw [if_neg (by omega : ¬ start + offset < pilotInputPrivateColumnCount),
    if_pos upper, if_neg (by omega : ¬ start < pilotInputPrivateColumnCount),
    if_pos startUpper]
  omega

theorem liftPilotColumn_add_of_public (start offset : Nat)
    (lower : pilotPrivateColumnCount ≤ start) :
    liftPilotColumn (start + offset) = liftPilotColumn start + offset := by
  have sumLower : pilotPrivateColumnCount ≤ start + offset := by omega
  have inputLePrivate :
      pilotInputPrivateColumnCount ≤ pilotPrivateColumnCount := by
    norm_num [pilotInputPrivateColumnCount, pilotPrivateColumnCount]
  unfold liftPilotColumn
  rw [if_neg (by omega : ¬ start + offset < pilotInputPrivateColumnCount),
    if_neg (by omega : ¬ start + offset < pilotPrivateColumnCount),
    if_neg (by omega : ¬ start < pilotInputPrivateColumnCount),
    if_neg (by omega : ¬ start < pilotPrivateColumnCount)]
  omega

theorem liftPilotColumn_lt (column : Nat)
    (bound : column < PilotSpartan.spartanColumnCount) :
    liftPilotColumn column < spartanColumnCount := by
  unfold liftPilotColumn
  split
  all_goals try split
  all_goals (norm_num [PilotSpartan.spartanColumnCount,
      PilotSpartan.constantColumn, PilotSpartan.privateColumnCount,
      PilotSpartan.publicColumnCount, privateColumnCount,
      spartanColumnCount, pilotPrivateColumnCount,
      pilotInputPrivateColumnCount, proofInputColumnCount,
      publicColumnCount,
      PilotProduction.stateHashWords_eq,
      PilotProduction.hashWitnessCount_eq,
      PilotProduction.digestWords, PilotValues.digestWords,
      PriorStateHash.publicWidth_eq] at *; omega)

theorem liftPilotColumn_ne_constant (column : Nat)
    (bound : column < PilotSpartan.spartanColumnCount)
    (notConstant : column ≠ PilotSpartan.constantColumn) :
    liftPilotColumn column ≠ constantColumn := by
  unfold liftPilotColumn
  split
  all_goals try split
  all_goals (norm_num [PilotSpartan.spartanColumnCount,
      PilotSpartan.constantColumn, PilotSpartan.privateColumnCount,
      PilotSpartan.publicColumnCount, privateColumnCount, constantColumn,
      publicColumnCount, pilotPrivateColumnCount,
      pilotInputPrivateColumnCount, proofInputColumnCount,
      PilotProduction.stateHashWords_eq,
      PilotProduction.hashWitnessCount_eq,
      PilotProduction.digestWords, PilotValues.digestWords,
      PriorStateHash.publicWidth_eq] at *; omega)

/-- Map the cumulative Lean source order into combined Spartan order. -/
def sourceToSpartan (column : Nat) : Nat :=
  if column < pilotSourceColumnCount then
    liftPilotColumn (PilotSpartan.sourceToSpartan column)
  else if column < proofInputSourceStart then
    expectedContextPublicStart + (column - pilotSourceColumnCount)
  else if column < piCcsPhaseOffset then
    pilotInputPrivateColumnCount + (column - proofInputSourceStart)
  else
    piCcsLocalStart + (column - piCcsPhaseOffset)

/-- Each verifier-context source word maps to its matching public lane. -/
theorem sourceToSpartan_expectedContext (lane : Fin 4) :
    sourceToSpartan
        (NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart +
          lane.val) =
      expectedContextPublicStart + lane.val := by
  have laneBound := lane.isLt
  unfold sourceToSpartan
  rw [if_neg (by
    norm_num [pilotSourceColumnCount,
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart])]
  rw [if_pos (by
    norm_num [proofInputSourceStart,
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart]
    omega)]
  norm_num [pilotSourceColumnCount,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart]

/-- The pilot prior-preimage interval remains in the first private Spartan
interval, so offsets inside that interval are preserved exactly. -/
theorem sourceToSpartan_add_of_pilotPriorPrivate (start offset : Nat)
    (upper : start + offset < PilotProduction.priorPublicInputStart) :
    sourceToSpartan (start + offset) = sourceToSpartan start + offset := by
  have startUpper : start < PilotProduction.priorPublicInputStart := by omega
  have sumPilot : start + offset < pilotSourceColumnCount := by
    norm_num [PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      pilotSourceColumnCount] at upper ⊢
    omega
  have startPilot : start < pilotSourceColumnCount := by omega
  have sumPrior : start + offset < PilotSpartan.priorPublicStart := by
    simpa [PilotSpartan.priorPublicStart] using upper
  have startPrior : start < PilotSpartan.priorPublicStart := by
    simpa [PilotSpartan.priorPublicStart] using startUpper
  unfold sourceToSpartan
  rw [if_pos sumPilot, if_pos startPilot]
  rw [PilotSpartan.sourceToSpartan, if_pos sumPrior,
    PilotSpartan.sourceToSpartan, if_pos startPrior]
  exact liftPilotColumn_add_of_input start offset (by
    norm_num [pilotInputPrivateColumnCount,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq]
      at upper ⊢
    omega)

/-- The 54-word pilot prior-public interval remains one contiguous public
Spartan interval, including the outer Stage 1 lift. -/
theorem sourceToSpartan_add_of_pilotPriorPublic (start offset : Nat)
    (lower : PilotProduction.priorPublicInputStart ≤ start)
    (upper : start + offset < PilotProduction.outputPreimageStart) :
    sourceToSpartan (start + offset) = sourceToSpartan start + offset := by
  have startUpper : start < PilotProduction.outputPreimageStart := by omega
  have sumPilot : start + offset < pilotSourceColumnCount := by
    norm_num [PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      PriorStateHash.publicWidth_eq, pilotSourceColumnCount] at upper ⊢
    omega
  have startPilot : start < pilotSourceColumnCount := by omega
  have sumNotPrior : ¬ start + offset < PilotSpartan.priorPublicStart := by
    simpa [PilotSpartan.priorPublicStart] using lower.trans (Nat.le_add_right _ _)
  have startNotPrior : ¬ start < PilotSpartan.priorPublicStart := by
    simpa [PilotSpartan.priorPublicStart] using lower
  have sumBeforeOutput :
      start + offset < PilotSpartan.outputPreimageStart := by
    simpa [PilotSpartan.outputPreimageStart, PilotSpartan.priorPublicStart]
      using upper
  have startBeforeOutput : start < PilotSpartan.outputPreimageStart := by
    simpa [PilotSpartan.outputPreimageStart, PilotSpartan.priorPublicStart]
      using startUpper
  have pilotAffine :
      PilotSpartan.sourceToSpartan (start + offset) =
        PilotSpartan.sourceToSpartan start + offset := by
    unfold PilotSpartan.sourceToSpartan
    rw [if_neg sumNotPrior, if_pos sumBeforeOutput,
      if_neg startNotPrior, if_pos startBeforeOutput]
    omega
  have mappedLower : pilotPrivateColumnCount ≤
      PilotSpartan.sourceToSpartan start := by
    unfold PilotSpartan.sourceToSpartan
    rw [if_neg startNotPrior, if_pos startBeforeOutput]
    norm_num [pilotPrivateColumnCount, PilotSpartan.firstPublicStart,
      PilotSpartan.privateColumnCount, PilotProduction.stateHashWords_eq,
      PilotProduction.hashWitnessCount_eq]
    omega
  unfold sourceToSpartan
  rw [if_pos sumPilot, if_pos startPilot, pilotAffine]
  exact liftPilotColumn_add_of_public _ _ mappedLower

/-- PiCCS proof inputs form one contiguous private interval before the
PiCCS-local witness suffix. -/
theorem sourceToSpartan_add_of_proofInput (start offset : Nat)
    (lower : proofInputSourceStart ≤ start)
    (upper : start + offset < piCcsPhaseOffset) :
    sourceToSpartan (start + offset) = sourceToSpartan start + offset := by
  have startUpper : start < piCcsPhaseOffset := by omega
  have pilotBeforeProof : pilotSourceColumnCount < proofInputSourceStart := by
    norm_num [pilotSourceColumnCount, proofInputSourceStart]
  have sumNotPilot : ¬ start + offset < pilotSourceColumnCount := by
    omega
  have startNotPilot : ¬ start < pilotSourceColumnCount := by omega
  have sumNotContext : ¬ start + offset < proofInputSourceStart := by omega
  have startNotContext : ¬ start < proofInputSourceStart := by omega
  unfold sourceToSpartan
  rw [if_neg sumNotPilot, if_neg sumNotContext, if_pos upper,
    if_neg startNotPilot, if_neg startNotContext, if_pos startUpper]
  omega

/-- A source column before the PiCCS phase maps either before every PiCCS
local witness or into the relocated public suffix after all private columns. -/
theorem sourceToSpartan_before_piCcsPhase (column : Nat)
    (before : column < piCcsPhaseOffset) :
    sourceToSpartan column < piCcsLocalStart ∨
      privateColumnCount < sourceToSpartan column := by
  by_cases pilot : column < pilotSourceColumnCount
  · rw [sourceToSpartan, if_pos pilot]
    let mapped := PilotSpartan.sourceToSpartan column
    have sourceBound : column < PilotSpartan.SourceColumnCount := by
      rw [← pilotSourceColumnCount_matches]
      exact pilot
    have mappedBound : mapped < PilotSpartan.spartanColumnCount :=
      PilotSpartan.sourceToSpartan_lt column sourceBound
    have mappedNotConstant : mapped ≠ PilotSpartan.constantColumn :=
      PilotSpartan.sourceToSpartan_ne_constant column sourceBound
    change liftPilotColumn mapped < piCcsLocalStart ∨
      privateColumnCount < liftPilotColumn mapped
    unfold liftPilotColumn
    by_cases input : mapped < pilotInputPrivateColumnCount
    · rw [if_pos input]
      exact Or.inl (by
        norm_num [piCcsLocalStart, pilotInputPrivateColumnCount] at *
        omega)
    · rw [if_neg input]
      by_cases privateBound : mapped < pilotPrivateColumnCount
      · rw [if_pos privateBound]
        exact Or.inl (by
          norm_num [piCcsLocalStart, proofInputColumnCount,
            pilotPrivateColumnCount] at *
          omega)
      · rw [if_neg privateBound]
        exact Or.inr (by
          have notBoundary : mapped ≠ pilotPrivateColumnCount := by
            rw [pilotPrivateColumnCount_matches]
            exact mappedNotConstant
          omega)
  · by_cases context : column < proofInputSourceStart
    · rw [sourceToSpartan, if_neg pilot, if_pos context]
      exact Or.inr (by
        norm_num [expectedContextPublicStart, privateColumnCount] at *
        omega)
    · rw [sourceToSpartan, if_neg pilot, if_neg context, if_pos before]
      exact Or.inl (by
        norm_num [pilotInputPrivateColumnCount, proofInputSourceStart,
          piCcsPhaseOffset, piCcsLocalStart] at *
        omega)

/-- The PiCCS-local interval is translated by one fixed displacement, so
relative offsets are preserved exactly. -/
theorem sourceToSpartan_add_of_piCcsLocal (start offset : Nat)
    (startLocal : piCcsPhaseOffset ≤ start) :
    sourceToSpartan (start + offset) = sourceToSpartan start + offset := by
  have pilotBefore : pilotSourceColumnCount ≤ piCcsPhaseOffset := by
    norm_num [pilotSourceColumnCount, piCcsPhaseOffset]
  have proofInputBefore : proofInputSourceStart ≤ piCcsPhaseOffset := by
    norm_num [proofInputSourceStart, piCcsPhaseOffset]
  have startAfterPilot : ¬ start < pilotSourceColumnCount := by omega
  have sumAfterPilot : ¬ start + offset < pilotSourceColumnCount := by omega
  have startAfterContext : ¬ start < proofInputSourceStart := by omega
  have sumAfterContext : ¬ start + offset < proofInputSourceStart := by omega
  have startAfterPhase : ¬ start < piCcsPhaseOffset := by omega
  have sumAfterPhase : ¬ start + offset < piCcsPhaseOffset := by omega
  unfold sourceToSpartan
  rw [if_neg sumAfterPilot, if_neg sumAfterContext, if_neg sumAfterPhase,
    if_neg startAfterPilot, if_neg startAfterContext,
    if_neg startAfterPhase]
  omega

theorem piCcsLocalStart_le_sourceToSpartan (column : Nat)
    (localBound : piCcsPhaseOffset ≤ column) :
    piCcsLocalStart ≤ sourceToSpartan column := by
  unfold sourceToSpartan
  rw [if_neg (by
    norm_num [pilotSourceColumnCount, piCcsPhaseOffset] at *
    omega), if_neg (by
      norm_num [proofInputSourceStart, piCcsPhaseOffset] at *
      omega), if_neg (by omega)]
  omega

theorem sourceToSpartan_lt_of_piCcsLocal (left right : Nat)
    (leftLocal : piCcsPhaseOffset ≤ left) (strict : left < right) :
    sourceToSpartan left < sourceToSpartan right := by
  have mapped := sourceToSpartan_add_of_piCcsLocal left (right - left)
    leftLocal
  have restored : left + (right - left) = right := by omega
  rw [restored] at mapped
  rw [mapped]
  omega

/-- A source variable before a PiCCS-local witness start maps either before
that Spartan witness start or into the relocated public suffix. -/
theorem sourceToSpartan_before_piCcsLocal (column start : Nat)
    (startLocal : piCcsPhaseOffset ≤ start) (before : column < start) :
    sourceToSpartan column < sourceToSpartan start ∨
      privateColumnCount < sourceToSpartan column := by
  by_cases beforePhase : column < piCcsPhaseOffset
  · rcases sourceToSpartan_before_piCcsPhase column beforePhase with
      mappedBefore | mappedPublic
    · exact Or.inl (by
        have startMapped : piCcsLocalStart ≤ sourceToSpartan start := by
          unfold sourceToSpartan
          rw [if_neg (by
            norm_num [pilotSourceColumnCount, piCcsPhaseOffset] at *
            omega), if_neg (by
              norm_num [proofInputSourceStart, piCcsPhaseOffset] at *
              omega), if_neg (by omega)]
          omega
        omega)
    · exact Or.inr mappedPublic
  · exact Or.inl (sourceToSpartan_lt_of_piCcsLocal column start
      (by omega) before)

/-- Partial inverse. The combined constant column has no source column. -/
def spartanToSource (column : Nat) : Option Nat :=
  if column < pilotInputPrivateColumnCount then
    PilotSpartan.spartanToSource column
  else if column < pilotInputPrivateColumnCount + proofInputColumnCount then
    some (proofInputSourceStart +
      (column - pilotInputPrivateColumnCount))
  else if column < piCcsLocalStart then
    PilotSpartan.spartanToSource (column - proofInputColumnCount)
  else if column < privateColumnCount then
    some (piCcsPhaseOffset + (column - piCcsLocalStart))
  else if column = constantColumn then
    none
  else if column < expectedContextPublicStart then
    PilotSpartan.spartanToSource
      (pilotPrivateColumnCount +
        (column - privateColumnCount))
  else if column < spartanColumnCount then
    some (pilotSourceColumnCount +
      (column - expectedContextPublicStart))
  else
    none

theorem sourceToSpartan_lt (column : Nat) (bound : column < SourceColumnCount) :
    sourceToSpartan column < spartanColumnCount := by
  unfold sourceToSpartan
  split
  · have pilotBound : column < PilotSpartan.SourceColumnCount := by
      rw [← pilotSourceColumnCount_matches]
      assumption
    exact liftPilotColumn_lt _
      (PilotSpartan.sourceToSpartan_lt column pilotBound)
  · split
    all_goals try split
    all_goals rw [sourceColumnCount_eq] at bound
    all_goals (norm_num [pilotSourceColumnCount,
      pilotInputPrivateColumnCount, proofInputSourceStart,
      piCcsPhaseOffset, piCcsLocalStart, expectedContextPublicStart,
      spartanColumnCount] at *; omega)

theorem sourceToSpartan_ne_constant (column : Nat)
    (bound : column < SourceColumnCount) :
    sourceToSpartan column ≠ constantColumn := by
  unfold sourceToSpartan
  split
  · have pilotBound : column < PilotSpartan.SourceColumnCount := by
      rw [← pilotSourceColumnCount_matches]
      assumption
    exact liftPilotColumn_ne_constant _
      (PilotSpartan.sourceToSpartan_lt column pilotBound)
      (PilotSpartan.sourceToSpartan_ne_constant column pilotBound)
  · split
    all_goals try split
    all_goals rw [sourceColumnCount_eq] at bound
    all_goals (norm_num [pilotSourceColumnCount,
      pilotInputPrivateColumnCount, proofInputSourceStart,
      piCcsPhaseOffset, piCcsLocalStart, expectedContextPublicStart,
      constantColumn] at *; omega)

theorem spartanToSource_sourceToSpartan (column : Nat)
    (bound : column < SourceColumnCount) :
    spartanToSource (sourceToSpartan column) = some column := by
  by_cases pilot : column < pilotSourceColumnCount
  · have pilotBound : column < PilotSpartan.SourceColumnCount := by
      rw [← pilotSourceColumnCount_matches]
      exact pilot
    have oldBound := PilotSpartan.sourceToSpartan_lt column pilotBound
    have oldNotConstant :=
      PilotSpartan.sourceToSpartan_ne_constant column pilotBound
    have oldInverse :=
      PilotSpartan.spartanToSource_sourceToSpartan column pilotBound
    let mapped := PilotSpartan.sourceToSpartan column
    have mappedBound : mapped < PilotSpartan.spartanColumnCount := oldBound
    have mappedNotConstant : mapped ≠ PilotSpartan.constantColumn :=
      oldNotConstant
    rw [sourceToSpartan, if_pos pilot]
    unfold liftPilotColumn
    by_cases inputPilot : mapped < pilotInputPrivateColumnCount
    · rw [if_pos inputPilot]
      unfold spartanToSource
      rw [if_pos inputPilot]
      exact oldInverse
    · rw [if_neg inputPilot]
      by_cases witnessPilot : mapped < pilotPrivateColumnCount
      · rw [if_pos witnessPilot]
        have notInput :
            ¬(mapped + proofInputColumnCount <
              pilotInputPrivateColumnCount) := by
          omega
        have notProof :
            ¬(mapped + proofInputColumnCount <
              pilotInputPrivateColumnCount + proofInputColumnCount) := by
          omega
        have shiftedBound :
            mapped + proofInputColumnCount < piCcsLocalStart := by
          norm_num [pilotPrivateColumnCount, proofInputColumnCount,
            piCcsLocalStart] at *
          omega
        unfold spartanToSource
        rw [if_neg notInput, if_neg notProof, if_pos shiftedBound]
        have restored :
            mapped + proofInputColumnCount - proofInputColumnCount =
              mapped := by
          omega
        rw [restored]
        exact oldInverse
      · rw [if_neg witnessPilot]
        have mappedNotPrivate : mapped ≠ pilotPrivateColumnCount := by
          intro equals
          apply mappedNotConstant
          rw [PilotSpartan.constantColumn,
            ← pilotPrivateColumnCount_matches]
          exact equals
        have mappedAbove : pilotPrivateColumnCount < mapped := by
          omega
        have notPilotPrivate :
            ¬(privateColumnCount +
                (mapped - pilotPrivateColumnCount) <
              pilotInputPrivateColumnCount) := by
          norm_num [pilotPrivateColumnCount, pilotInputPrivateColumnCount,
            privateColumnCount] at *
          omega
        have notProofInput :
            ¬(privateColumnCount +
                (mapped - pilotPrivateColumnCount) <
              pilotInputPrivateColumnCount + proofInputColumnCount) := by
          norm_num [privateColumnCount, pilotPrivateColumnCount,
            pilotInputPrivateColumnCount, proofInputColumnCount] at *
          omega
        have notShiftedPilot :
            ¬(privateColumnCount +
                (mapped - pilotPrivateColumnCount) < piCcsLocalStart) := by
          norm_num [privateColumnCount, pilotPrivateColumnCount,
            piCcsLocalStart] at *
          omega
        have notCombinedPrivate :
            ¬(privateColumnCount +
                (mapped - pilotPrivateColumnCount) < privateColumnCount) := by
          omega
        have notCombinedConstant :
            privateColumnCount +
                (mapped - pilotPrivateColumnCount) ≠
              constantColumn := by
          intro equals
          rw [constantColumn_eq_private] at equals
          have positive : 0 < mapped - pilotPrivateColumnCount :=
            Nat.sub_pos_of_lt mappedAbove
          omega
        have mappedBoundNumeric : mapped < 12659089 := by
          rw [PilotSpartan.spartanColumnCount_eq,
            PilotSpartan.sourceColumnCount_eq] at mappedBound
          norm_num at mappedBound
          exact mappedBound
        have combinedPilotPublicBound :
            privateColumnCount +
                (mapped - pilotPrivateColumnCount) <
              expectedContextPublicStart := by
          change 25714955 + (mapped - 12659030) < 25715014
          omega
        unfold spartanToSource
        rw [if_neg notPilotPrivate, if_neg notProofInput,
          if_neg notShiftedPilot, if_neg notCombinedPrivate,
          if_neg notCombinedConstant, if_pos combinedPilotPublicBound]
        have restored :
            pilotPrivateColumnCount +
                (privateColumnCount +
                  (mapped - pilotPrivateColumnCount) -
                  privateColumnCount) = mapped := by
          omega
        rw [restored]
        exact oldInverse
  · rw [sourceToSpartan, if_neg pilot]
    by_cases context : column < proofInputSourceStart
    · rw [if_pos context]
      have notPilotInput :
          ¬(expectedContextPublicStart +
              (column - pilotSourceColumnCount) <
            pilotInputPrivateColumnCount) := by
        norm_num [expectedContextPublicStart,
          pilotInputPrivateColumnCount]
        omega
      have notProofInput :
          ¬(expectedContextPublicStart +
              (column - pilotSourceColumnCount) <
            pilotInputPrivateColumnCount + proofInputColumnCount) := by
        norm_num [expectedContextPublicStart,
          pilotInputPrivateColumnCount, proofInputColumnCount]
        omega
      have notShiftedPilot :
          ¬(expectedContextPublicStart +
              (column - pilotSourceColumnCount) < piCcsLocalStart) := by
        norm_num [expectedContextPublicStart, piCcsLocalStart]
        omega
      have notCombinedPrivate :
          ¬(expectedContextPublicStart +
              (column - pilotSourceColumnCount) < privateColumnCount) := by
        norm_num [expectedContextPublicStart, privateColumnCount]
        omega
      have notCombinedConstant :
          expectedContextPublicStart +
              (column - pilotSourceColumnCount) ≠ constantColumn := by
        norm_num [expectedContextPublicStart, constantColumn]
        omega
      have notPilotPublic :
          ¬(expectedContextPublicStart +
              (column - pilotSourceColumnCount) <
            expectedContextPublicStart) := by
        omega
      have contextBound :
          expectedContextPublicStart +
              (column - pilotSourceColumnCount) < spartanColumnCount := by
        norm_num [expectedContextPublicStart, spartanColumnCount,
          proofInputSourceStart, pilotSourceColumnCount] at *
        omega
      unfold spartanToSource
      rw [if_neg notPilotInput, if_neg notProofInput,
        if_neg notShiftedPilot, if_neg notCombinedPrivate,
        if_neg notCombinedConstant, if_neg notPilotPublic,
        if_pos contextBound]
      apply congrArg some
      norm_num [pilotSourceColumnCount, expectedContextPublicStart] at pilot ⊢
      omega
    · rw [if_neg context]
      by_cases proofInput : column < piCcsPhaseOffset
      · rw [if_pos proofInput]
        have notPilotInput :
            ¬(pilotInputPrivateColumnCount +
                (column - proofInputSourceStart) <
              pilotInputPrivateColumnCount) := by
          omega
        have proofBound :
            pilotInputPrivateColumnCount +
                (column - proofInputSourceStart) <
              pilotInputPrivateColumnCount + proofInputColumnCount := by
          norm_num [proofInputSourceStart, piCcsPhaseOffset,
            proofInputColumnCount] at *
          omega
        unfold spartanToSource
        rw [if_neg notPilotInput, if_pos proofBound]
        apply congrArg some
        norm_num [proofInputSourceStart,
          pilotInputPrivateColumnCount] at context ⊢
        omega
      · rw [if_neg proofInput]
        have notPilotInput :
            ¬(piCcsLocalStart + (column - piCcsPhaseOffset) <
              pilotInputPrivateColumnCount) := by
          norm_num [piCcsLocalStart, pilotInputPrivateColumnCount]
          omega
        have notProofInput :
            ¬(piCcsLocalStart + (column - piCcsPhaseOffset) <
              pilotInputPrivateColumnCount + proofInputColumnCount) := by
          norm_num [piCcsLocalStart, pilotInputPrivateColumnCount,
            proofInputColumnCount]
          omega
        have notShiftedPilot :
            ¬(piCcsLocalStart + (column - piCcsPhaseOffset) <
              piCcsLocalStart) := by
          omega
        have localBound :
            piCcsLocalStart + (column - piCcsPhaseOffset) <
              privateColumnCount := by
          rw [sourceColumnCount_eq] at bound
          norm_num [piCcsLocalStart, piCcsPhaseOffset,
            privateColumnCount] at *
          omega
        unfold spartanToSource
        rw [if_neg notPilotInput, if_neg notProofInput,
          if_neg notShiftedPilot, if_pos localBound]
        apply congrArg some
        norm_num [piCcsLocalStart, piCcsPhaseOffset] at proofInput ⊢
        omega

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

/-- Copy one completed PiCCS-local source interval into its contiguous
Spartan interval. Every other target column keeps its prior value. -/
def copyMappedInterval (base source : Env) (sourceStart length : Nat) : Env :=
  fun target =>
    if sourceToSpartan sourceStart ≤ target ∧
        target < sourceToSpartan sourceStart + length then
      source (sourceStart + (target - sourceToSpartan sourceStart))
    else
      base target

theorem copyMappedInterval_agreesOutside (base source : Env)
    (sourceStart length : Nat) :
    AgreesOutside base (copyMappedInterval base source sourceStart length)
      (sourceToSpartan sourceStart) length := by
  intro target outside
  unfold copyMappedInterval
  rw [if_neg]
  intro inside
  rcases outside with before | after
  · omega
  · omega

/-- Pulling a mapped interval copy back to source order gives the completed
source environment exactly. The source completion must agree with the base
outside that interval, and the mapped interval must stay private. -/
theorem pullback_copyMappedInterval_eq (base source : Env)
    (sourceStart length : Nat)
    (startLocal : piCcsPhaseOffset ≤ sourceStart)
    (targetEndPrivate : sourceToSpartan sourceStart + length ≤
      privateColumnCount)
    (sourceAgrees : AgreesOutside (pullback base) source sourceStart length) :
    pullback (copyMappedInterval base source sourceStart length) = source := by
  funext column
  by_cases before : column < sourceStart
  · have mappedOutside : sourceToSpartan column <
          sourceToSpartan sourceStart ∨
        sourceToSpartan sourceStart + length ≤ sourceToSpartan column := by
      rcases sourceToSpartan_before_piCcsLocal column sourceStart startLocal
          before with mappedBefore | mappedPublic
      · exact Or.inl mappedBefore
      · exact Or.inr (by omega)
    have copiedBase : copyMappedInterval base source sourceStart length
        (sourceToSpartan column) = base (sourceToSpartan column) := by
      unfold copyMappedInterval
      rw [if_neg]
      intro inside
      rcases mappedOutside with mappedBefore | mappedAfter
      · omega
      · omega
    rw [pullback, copiedBase]
    exact (sourceAgrees column (Or.inl before)).symm
  · by_cases after : sourceStart + length ≤ column
    · have restored : sourceStart + (column - sourceStart) = column := by
        omega
      have mapped := sourceToSpartan_add_of_piCcsLocal sourceStart
        (column - sourceStart) startLocal
      rw [restored] at mapped
      have mappedAfter : sourceToSpartan sourceStart + length ≤
          sourceToSpartan column := by
        rw [mapped]
        omega
      have copiedBase : copyMappedInterval base source sourceStart length
          (sourceToSpartan column) = base (sourceToSpartan column) := by
        unfold copyMappedInterval
        rw [if_neg]
        intro inside
        omega
      rw [pullback, copiedBase]
      exact (sourceAgrees column (Or.inr after)).symm
    · have restored : sourceStart + (column - sourceStart) = column := by
        omega
      have mapped := sourceToSpartan_add_of_piCcsLocal sourceStart
        (column - sourceStart) startLocal
      rw [restored] at mapped
      have mappedInside : sourceToSpartan sourceStart ≤
          sourceToSpartan column ∧
          sourceToSpartan column < sourceToSpartan sourceStart + length := by
        rw [mapped]
        constructor <;> omega
      unfold pullback copyMappedInterval
      rw [if_pos mappedInside, mapped]
      congr 1
      omega

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

theorem remapRows_hold_copyMappedInterval (rows : List R1CS.Row)
    (base source : Env) (sourceStart length : Nat)
    (startLocal : piCcsPhaseOffset ≤ sourceStart)
    (targetEndPrivate : sourceToSpartan sourceStart + length ≤
      privateColumnCount)
    (sourceAgrees : AgreesOutside (pullback base) source sourceStart length)
    (holds : R1CS.RowsHold source rows) :
    R1CS.RowsHold (copyMappedInterval base source sourceStart length)
      (remapRows rows) := by
  apply (remapRows_hold _ _).mpr
  rw [pullback_copyMappedInterval_eq base source sourceStart length startLocal
    targetEndPrivate sourceAgrees]
  exact holds

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def sourceRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  PilotPiCCSPiRLCPiDEC.physicalRows relation

def remappedRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  remapRows (sourceRows relation)

theorem remappedRows_hold
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env) :
    R1CS.RowsHold target (remappedRows relation) ↔
      PilotPiCCSPiRLCPiDEC.PhysicalHolds relation (pullback target) := by
  exact remapRows_hold target (sourceRows relation)

theorem sourceColumnCount_matches
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PilotPiCCSPiRLCPiDEC.physicalColumnCount relation = SourceColumnCount := by
  rw [PilotPiCCSPiRLCPiDEC.physicalColumnCount_eq relation,
    sourceColumnCount_eq]

theorem sourceRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (sourceRows relation).length = 25564086 := by
  exact PilotPiCCSPiRLCPiDEC.physicalRowCount_eq relation

/-- The generic Spartan row and private-variable domains use the fixed cube. -/
def domainSize : Nat := 2 ^ cubeVariables

def paddedConstantColumn : Nat := domainSize

def paddedSpartanColumnCount : Nat :=
  domainSize + 1 + publicColumnCount

theorem domainSize_eq : domainSize = 33554432 := by
  norm_num [domainSize, cubeVariables]

theorem sourceRowCount_bounds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (sourceRows relation).length ≤ domainSize := by
  rw [sourceRowCount_eq relation, domainSize_eq]
  norm_num

theorem privateColumnCount_bound : privateColumnCount ≤ domainSize := by
  rw [privateColumnCount_eq, domainSize_eq]
  norm_num

/-- Move the unpadded constant and public suffix after the private domain. -/
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

def paddedRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  padRows (remappedRows relation) ++
    List.replicate
      (domainSize - (sourceRows relation).length) zeroRow

theorem paddedRows_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (paddedRows relation).length = domainSize := by
  unfold paddedRows padRows remappedRows remapRows
  rw [List.length_append, List.length_map, List.length_map,
    List.length_replicate,
    Nat.add_sub_of_le (sourceRowCount_bounds relation)]

/-- The padded direct-Spartan rows preserve and reflect the complete current
Stage 1 prefix through PiDEC. -/
theorem paddedRows_hold
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env) :
    R1CS.RowsHold target (paddedRows relation) ↔
      PilotPiCCSPiRLCPiDEC.PhysicalHolds relation
        (pullback (paddedPullback target)) := by
  change R1CS.RowsHold target (paddedRows relation) ↔
    R1CS.RowsHold (pullback (paddedPullback target))
      (sourceRows relation)
  constructor
  · intro holds
    have split := (R1CS.rowsHold_append target _ _).mp holds
    exact (remappedRows_hold relation (paddedPullback target)).mp
      ((padRows_hold target (remappedRows relation)).mp split.1)
  · intro holds
    apply (R1CS.rowsHold_append target _ _).mpr
    exact ⟨
      (padRows_hold target (remappedRows relation)).mpr
        ((remappedRows_hold relation (paddedPullback target)).mpr
          holds),
      zeroRows_hold target _⟩

end NightstreamFPrime.Layout.Stage1.Spartan
