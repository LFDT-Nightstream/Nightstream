import NightstreamFPrime.Layout.Stage1.PiDECStarts
import NightstreamFPrime.Layout.Stage1.Spartan

/-!
Owns the exact pre-Spartan source families used by the nonempty PiDEC rows.
The four parent ranges are the final PiRLC combination steps. The remaining
ranges are the PiDEC proof inputs, logical split cells, and R1CS fresh cells.
-/

namespace NightstreamFPrime.Layout.Stage1.PiDECSourceSupport

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PiRLC.v1_1

def InRange (start count column : Nat) : Prop :=
  start ≤ column ∧ column < start + count

def parentCommitmentStart : Nat :=
  CombinationFamily.stepOffset PiRLCStarts.commitmentLogicalStart
    CombinationFamily.finalSource.val CommitmentCombination.blockCount
      CommitmentCombination.cellCount

def parentPublicInputStart : Nat :=
  CombinationFamily.stepOffset PiRLCStarts.publicInputLogicalStart
    CombinationFamily.finalSource.val PublicInputCombination.blockCount
      PublicInputCombination.cellCount

def parentEvalKStart : Nat :=
  CombinationFamily.stepOffset PiRLCStarts.evalKLogicalStart
    CombinationFamily.finalSource.val EvalKCombination.blockCount
      RingKCombination.cellCount

def parentEvalAStart : Nat :=
  CombinationFamily.stepOffset PiRLCStarts.evalALogicalStart
    CombinationFamily.finalSource.val EvalACombination.blockCount
      RingKCombination.cellCount

def Parent (column : Nat) : Prop :=
  InRange parentCommitmentStart PiDECInputs.commitmentWordsPerChild column ∨
    InRange parentPublicInputStart PiDECInputs.publicInputWordsPerChild column ∨
    InRange parentEvalKStart PiDECInputs.evalKWordsPerChild column ∨
    InRange parentEvalAStart PiDECInputs.evalAWordsPerChild column

def External (column : Nat) : Prop :=
  Parent column ∨
    InRange PiDECInputs.proofInputStart PiDECInputs.proofInputColumnCount column

/-- Logical cells allocated by the PiDEC phase. -/
def logicalCount : Nat := PiDEC.v1_1.Formal.logicalPrivateCount

def Logical (column : Nat) : Prop :=
  External column ∨
    InRange PiDECStarts.phaseLogicalStart logicalCount column

def freshCount : Nat := NightstreamFPrime.Layout.PiDEC.v1_1.exactFreshCount

def Source (column : Nat) : Prop :=
  Logical column ∨
    InRange PiDECStarts.phaseFreshStart freshCount column

def Target (column : Nat) : Prop :=
  ∃ source, Source source ∧ Spartan.sourceToSpartan source = column

@[simp] theorem parentCommitmentStart_eq :
    parentCommitmentStart = 20347399 := by
  rfl

@[simp] theorem parentPublicInputStart_eq :
    parentPublicInputStart = 20352907 := by
  rfl

@[simp] theorem parentEvalKStart_eq : parentEvalKStart = 20354905 := by
  rfl

@[simp] theorem parentEvalAStart_eq : parentEvalAStart = 20379205 := by
  rfl

@[simp] theorem parentStarts_eq :
    [parentCommitmentStart, parentPublicInputStart, parentEvalKStart,
      parentEvalAStart] = [20347399, 20352907, 20354905, 20379205] := by
  simp

theorem parentCommitment (column : Nat)
    (support : InRange parentCommitmentStart
      PiDECInputs.commitmentWordsPerChild column) : Parent column :=
  Or.inl support

theorem parentPublicInput (column : Nat)
    (support : InRange parentPublicInputStart
      PiDECInputs.publicInputWordsPerChild column) : Parent column :=
  Or.inr (Or.inl support)

theorem parentEvalK (column : Nat)
    (support : InRange parentEvalKStart PiDECInputs.evalKWordsPerChild column) :
    Parent column :=
  Or.inr (Or.inr (Or.inl support))

theorem parentEvalA (column : Nat)
    (support : InRange parentEvalAStart PiDECInputs.evalAWordsPerChild column) :
    Parent column :=
  Or.inr (Or.inr (Or.inr support))

theorem parent_source (column : Nat) (support : Parent column) :
    Source column :=
  Or.inl (Or.inl (Or.inl support))

theorem proof_source (column : Nat)
    (support : InRange PiDECInputs.proofInputStart
      PiDECInputs.proofInputColumnCount column) : Source column :=
  Or.inl (Or.inl (Or.inr support))

theorem logical_source (column : Nat)
    (support : InRange PiDECStarts.phaseLogicalStart logicalCount column) :
    Source column :=
  Or.inl (Or.inr support)

theorem fresh_source (column : Nat)
    (support : InRange PiDECStarts.phaseFreshStart freshCount column) :
    Source column :=
  Or.inr support

theorem source_target (column : Nat) (support : Source column) :
    Target (Spartan.sourceToSpartan column) :=
  ⟨column, support, rfl⟩

/-- The seven PiDEC source intervals are ordered within the affine Spartan
region. The proof inputs, logical split cells and fresh cells are contiguous. -/
theorem source_ranges_ordered :
    Spartan.piCcsPhaseOffset ≤ parentCommitmentStart ∧
    parentCommitmentStart + PiDECInputs.commitmentWordsPerChild ≤
      parentPublicInputStart ∧
    parentPublicInputStart + PiDECInputs.publicInputWordsPerChild ≤
      parentEvalKStart ∧
    parentEvalKStart + PiDECInputs.evalKWordsPerChild ≤ parentEvalAStart ∧
    parentEvalAStart + PiDECInputs.evalAWordsPerChild ≤ PiDECInputs.proofInputStart ∧
    PiDECInputs.proofInputStart + PiDECInputs.proofInputColumnCount =
      PiDECStarts.phaseLogicalStart ∧
    PiDECStarts.phaseLogicalStart + logicalCount = PiDECStarts.phaseFreshStart := by
  refine ⟨?_, ?_, ?_, ?_, ?_, rfl, rfl⟩ <;> decide

/-- The logical interval fits in the declared Stage 1 source capacity. -/
theorem logical_end_le_sourceColumnCount :
    PiDECStarts.phaseLogicalStart + logicalCount ≤ Spartan.SourceColumnCount := by
  decide

/-- The fresh interval fits in the declared Stage 1 source capacity. -/
theorem fresh_end_le_sourceColumnCount :
    PiDECStarts.phaseFreshStart + freshCount ≤ Spartan.SourceColumnCount := by
  decide

theorem source_lt_sourceColumnCount {column : Nat} (support : Source column) :
    column < Spartan.SourceColumnCount := by
  rcases support with ((parent | proof) | logical) | fresh
  · rcases parent with commitment | publicInput | evalK | evalA
    · exact Nat.lt_of_lt_of_le commitment.2 (by decide)
    · exact Nat.lt_of_lt_of_le publicInput.2 (by decide)
    · exact Nat.lt_of_lt_of_le evalK.2 (by decide)
    · exact Nat.lt_of_lt_of_le evalA.2 (by decide)
  · exact Nat.lt_of_lt_of_le proof.2 (by decide)
  · exact Nat.lt_of_lt_of_le logical.2 logical_end_le_sourceColumnCount
  · exact Nat.lt_of_lt_of_le fresh.2 fresh_end_le_sourceColumnCount

/-- The four protocol input families remain adjacent after Spartan mapping. -/
theorem mapped_input_ranges_contiguous :
    Spartan.sourceToSpartan PiDECInputs.commitmentInputStart +
        PiDECInputs.childCount * PiDECInputs.commitmentWordsPerChild =
      Spartan.sourceToSpartan PiDECInputs.evalKInputStart ∧
    Spartan.sourceToSpartan PiDECInputs.evalKInputStart +
        PiDECInputs.childCount * PiDECInputs.evalKWordsPerChild =
      Spartan.sourceToSpartan PiDECInputs.evalAInputStart ∧
    Spartan.sourceToSpartan PiDECInputs.evalAInputStart +
        PiDECInputs.childCount * PiDECInputs.evalAWordsPerChild =
      Spartan.sourceToSpartan PiDECInputs.publicInputStart ∧
    Spartan.sourceToSpartan PiDECInputs.publicInputStart +
        PiDECInputs.childCount * PiDECInputs.publicInputWordsPerChild =
      Spartan.sourceToSpartan PiDECInputs.phaseOffset := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [← Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by decide)]
    rfl
  · rw [← Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by decide)]
    rfl
  · rw [← Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by decide)]
    rfl
  · rw [← Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by decide)]
    simp only [PiDECInputs.publicInputStart, PiDECInputs.evalAInputStart,
      PiDECInputs.evalKInputStart, PiDECInputs.commitmentInputStart,
      PiDECInputs.phaseOffset, PiDECInputs.proofInputColumnCount,
      Nat.mul_add, Nat.add_assoc]

/-- The mapped logical interval ends no later than the next phase begins. -/
theorem mapped_logical_start_le_output :
    Spartan.sourceToSpartan PiDECInputs.phaseOffset ≤
      Spartan.sourceToSpartan RunningTransitionInputs.phaseOffset := by
  rcases Nat.eq_or_lt_of_le RunningTransitionInputs.piDecPhaseOffset_le with
    same | before
  · rw [same]
  · exact Nat.le_of_lt (Spartan.sourceToSpartan_lt_of_piCcsLocal _ _
      (by decide) before)

/-- The next phase starts inside the declared private source capacity. -/
theorem mapped_output_le_private :
    Spartan.sourceToSpartan RunningTransitionInputs.phaseOffset ≤
      Spartan.privateColumnCount := by
  decide

/-- Every retained source is at or after the first parent interval. -/
theorem parentStart_le_source {column : Nat} (support : Source column) :
    parentCommitmentStart ≤ column := by
  rcases source_ranges_ordered with
    ⟨_, commitmentPublic, publicEvalK, evalKEvalA, evalAProof,
      proofLogical, logicalFresh⟩
  have publicBound : parentCommitmentStart ≤ parentPublicInputStart :=
    Nat.le_trans (Nat.le_add_right _ _) commitmentPublic
  have evalKBound : parentCommitmentStart ≤ parentEvalKStart :=
    Nat.le_trans publicBound (Nat.le_trans (Nat.le_add_right _ _) publicEvalK)
  have evalABound : parentCommitmentStart ≤ parentEvalAStart :=
    Nat.le_trans evalKBound (Nat.le_trans (Nat.le_add_right _ _) evalKEvalA)
  have proofBound : parentCommitmentStart ≤ PiDECInputs.proofInputStart :=
    Nat.le_trans evalABound (Nat.le_trans (Nat.le_add_right _ _) evalAProof)
  have logicalBound : parentCommitmentStart ≤ PiDECStarts.phaseLogicalStart := by
    rw [← proofLogical]
    exact Nat.le_trans proofBound (Nat.le_add_right _ _)
  have freshBound : parentCommitmentStart ≤ PiDECStarts.phaseFreshStart := by
    rw [← logicalFresh]
    exact Nat.le_trans logicalBound (Nat.le_add_right _ _)
  rcases support with ((parent | proof) | logical) | fresh
  · rcases parent with commitment | publicInput | evalK | evalA
    · exact commitment.1
    · exact Nat.le_trans publicBound publicInput.1
    · exact Nat.le_trans evalKBound evalK.1
    · exact Nat.le_trans evalABound evalA.1
  · exact Nat.le_trans proofBound proof.1
  · exact Nat.le_trans logicalBound logical.1
  · exact Nat.le_trans freshBound fresh.1

/-- Parent input ranges lie within the PiRLC combination allocation. -/
theorem parent_within_piRlc {column : Nat} (support : Parent column) :
    PiRLCStarts.commitmentLogicalStart ≤ column ∧
      column < PiRLCStarts.phaseFreshStart := by
  rcases support with h | h | h | h <;>
    exact ⟨Nat.le_trans (by decide) h.1, Nat.lt_of_lt_of_le h.2 (by decide)⟩

end NightstreamFPrime.Layout.Stage1.PiDECSourceSupport
