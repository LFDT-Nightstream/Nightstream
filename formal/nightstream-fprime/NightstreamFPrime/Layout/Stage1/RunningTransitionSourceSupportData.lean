import NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
import NightstreamFPrime.Layout.Stage1.RunningTransitionPointBoundsDirect
import NightstreamFPrime.Layout.Stage1.Spartan

/-! Owns the compact source-support descriptors for the running transition. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open RunningTransitionInputs

def InRange (start count column : Nat) : Prop :=
  start ≤ column ∧ column < start + count

def stateStart : Nat :=
  PilotProduction.priorPreimageStart + iterationWordIndex

def stateCount : Nat := 11

def outputStart : Nat := PilotProduction.outputPreimageStart

def outputCount : Nat := PilotProduction.stateHashWords

def piDecStart : Nat := PiDECInputs.proofInputStart

def piDecCount : Nat := PiDECInputs.proofInputColumnCount

def physicalEnd : Nat := 27695988

/-- Exact PiDEC child-message and public-input fields consumed by the
recursive branch. -/
def PiDecField (column : Nat) : Prop :=
  (∃ (source : Radix.ChildIndex)
      (row : Fin productionProfile.commitmentWidth)
      (coefficient : Fin ringDegree),
    column = PiDECInputs.childCommitmentStart source +
      row.val * ringDegree + coefficient.val) ∨
  (∃ (source : Radix.ChildIndex) (coordinate : Fin 270),
    column = PiDECInputs.childPublicInputStart source + coordinate.val) ∨
  (∃ (source : Radix.ChildIndex)
      (coefficient : Fin productionShape.coefficientCount),
    column = PiDECInputs.childEvalKStart source + coefficient.val * 2 ∨
      column = PiDECInputs.childEvalKStart source + coefficient.val * 2 + 1) ∨
  (∃ (source : Radix.ChildIndex)
      (matrix : Fin productionShape.matrixCount)
      (coefficient : Fin productionShape.coefficientCount),
    column = PiDECInputs.childEvalAStart source +
        matrix.val * PiDECInputs.evalKWordsPerChild + coefficient.val * 2 ∨
      column = PiDECInputs.childEvalAStart source +
        matrix.val * PiDECInputs.evalKWordsPerChild + coefficient.val * 2 + 1)

def External (column : Nat) : Prop :=
  InRange stateStart stateCount column ∨
    InRange outputStart outputCount column ∨
      (∃ coordinate : Fin productionShape.cubeVariables,
        column = PiCCSStarts.roundTranscriptWitnessStart +
            coordinate.val * roundStride + roundSampleC0Offset ∨
          column = PiCCSStarts.roundTranscriptWitnessStart +
            coordinate.val * roundStride + roundSampleC1Offset) ∨
        PiDecField column

def Logical (column : Nat) : Prop :=
  External column ∨ column = phaseOffset

def Source (column : Nat) : Prop :=
  External column ∨ (phaseOffset ≤ column ∧ column < physicalEnd)

def Target (column : Nat) : Prop :=
  ∃ source, Source source ∧ column = Spartan.sourceToSpartan source

@[simp] theorem stateStart_eq : stateStart = 28 := by
  rfl

@[simp] theorem outputStart_eq : outputStart = 46207 := by
  rfl

@[simp] theorem outputCount_eq : outputCount = 45937 := by
  exact PilotProduction.stateHashWords_eq

@[simp] theorem piDecStart_eq : piDecStart = 27356704 := by
  rfl

@[simp] theorem piDecCount_eq : piDecCount = 45792 := by
  exact PiDECInputs.proofInputColumnCount_eq

/-- Every typed PiDEC child field lies in the one canonical proof-input
interval. This makes the retained resolver executable as a range decoder. -/
theorem piDecField_inRange {column : Nat} (field : PiDecField column) :
    InRange piDecStart piDecCount column := by
  rw [piDecStart_eq, piDecCount_eq]
  rcases field with commitment | publicInput | evalK | evalA
  · rcases commitment with ⟨source, row, coefficient, rfl⟩
    have sourceBound := source.isLt
    have rowBound := row.isLt
    have coefficientBound := coefficient.isLt
    change source.val < 16 at sourceBound
    change row.val < 18 at rowBound
    change coefficient.val < 54 at coefficientBound
    change InRange 27356704 45792
      (27356704 + source.val * 972 + row.val * 54 + coefficient.val)
    unfold InRange
    omega
  · rcases publicInput with ⟨source, coordinate, rfl⟩
    have sourceBound := source.isLt
    have coordinateBound := coordinate.isLt
    change source.val < 16 at sourceBound
    change coordinate.val < 270 at coordinateBound
    change InRange 27356704 45792
      (27398176 + source.val * 270 + coordinate.val)
    unfold InRange
    omega
  · rcases evalK with ⟨source, coefficient, low | high⟩
    all_goals subst column
    all_goals
      have sourceBound := source.isLt
      have coefficientBound := coefficient.isLt
      change source.val < 16 at sourceBound
      change coefficient.val < 54 at coefficientBound
      first
      | change InRange 27356704 45792
          (27372256 + source.val * 108 + coefficient.val * 2)
        unfold InRange
        omega
      | change InRange 27356704 45792
          (27372256 + source.val * 108 + coefficient.val * 2 + 1)
        unfold InRange
        omega
  · rcases evalA with ⟨source, matrix, coefficient, low | high⟩
    all_goals subst column
    all_goals
      have sourceBound := source.isLt
      have matrixBound := matrix.isLt
      have coefficientBound := coefficient.isLt
      change source.val < 16 at sourceBound
      change matrix.val < 14 at matrixBound
      change coefficient.val < 54 at coefficientBound
      first
      | change InRange 27356704 45792
          (27373984 + source.val * 1512 + matrix.val * 108 +
            coefficient.val * 2)
        unfold InRange
        omega
      | change InRange 27356704 45792
          (27373984 + source.val * 1512 + matrix.val * 108 +
            coefficient.val * 2 + 1)
        unfold InRange
        omega

/-- Every declared transition source has a valid image in the established
Spartan source permutation. -/
theorem source_lt_sourceColumnCount {column : Nat} (support : Source column) :
    column < Spartan.SourceColumnCount := by
  rw [Spartan.sourceColumnCount_eq]
  rcases support with external | fresh
  · rcases external with state | output | roundPoint | piDec
    · unfold InRange at state
      rw [stateStart_eq] at state
      norm_num [stateCount] at state
      omega
    · unfold InRange at output
      rw [outputStart_eq, outputCount_eq] at output
      omega
    · rcases roundPoint with ⟨coordinate, c0 | c1⟩
      all_goals subst column
      all_goals
        have coordinateBound := coordinate.isLt
        change coordinate.val < 28 at coordinateBound
        rw [PiCCSStarts.roundTranscriptWitnessStart_eq]
        norm_num [roundStride, roundSampleC0Offset, roundSampleC1Offset]
        omega
    · have inside := piDecField_inRange piDec
      unfold InRange at inside
      rw [piDecStart_eq, piDecCount_eq] at inside
      omega
  · change phaseOffset ≤ column ∧ column < physicalEnd at fresh
    norm_num [physicalEnd] at fresh ⊢
    omega

theorem logical_state {column : Nat}
    (inside : InRange stateStart stateCount column) : Logical column :=
  Or.inl (Or.inl inside)

theorem logical_output {column : Nat}
    (inside : InRange outputStart outputCount column) : Logical column :=
  Or.inl (Or.inr (Or.inl inside))

theorem logical_roundPointC0
    (coordinate : Fin productionShape.cubeVariables) :
    Logical (PiCCSStarts.roundTranscriptWitnessStart +
      coordinate.val * roundStride + roundSampleC0Offset) :=
  Or.inl (Or.inr (Or.inr (Or.inl
    ⟨coordinate, Or.inl rfl⟩)))

theorem logical_roundPointC1
    (coordinate : Fin productionShape.cubeVariables) :
    Logical (PiCCSStarts.roundTranscriptWitnessStart +
      coordinate.val * roundStride + roundSampleC1Offset) :=
  Or.inl (Or.inr (Or.inr (Or.inl
    ⟨coordinate, Or.inr rfl⟩)))

theorem logical_piDec {column : Nat}
    (field : PiDecField column) : Logical column :=
  Or.inl (Or.inr (Or.inr (Or.inr field)))

end NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport
