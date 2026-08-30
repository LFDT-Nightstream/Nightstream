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

def Logical (column : Nat) : Prop :=
  External column ∨
    InRange PiDECStarts.phaseLogicalStart 270 column

def freshCount : Nat := 17820

def Source (column : Nat) : Prop :=
  Logical column ∨
    InRange PiDECStarts.phaseFreshStart freshCount column

def Target (column : Nat) : Prop :=
  ∃ source, Source source ∧ Spartan.sourceToSpartan source = column

@[simp] theorem parentCommitmentStart_eq :
    parentCommitmentStart = 19281871 := by
  rfl

@[simp] theorem parentPublicInputStart_eq :
    parentPublicInputStart = 19287163 := by
  rfl

@[simp] theorem parentEvalKStart_eq : parentEvalKStart = 19289161 := by
  rfl

@[simp] theorem parentEvalAStart_eq : parentEvalAStart = 19313461 := by
  rfl

@[simp] theorem parentStarts_eq :
    [parentCommitmentStart, parentPublicInputStart, parentEvalKStart,
      parentEvalAStart] = [19281871, 19287163, 19289161, 19313461] := by
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
    (support : InRange PiDECStarts.phaseLogicalStart 270 column) :
    Source column :=
  Or.inl (Or.inr support)

theorem fresh_source (column : Nat)
    (support : InRange PiDECStarts.phaseFreshStart freshCount column) :
    Source column :=
  Or.inr support

theorem source_target (column : Nat) (support : Source column) :
    Target (Spartan.sourceToSpartan column) :=
  ⟨column, support, rfl⟩

theorem source_lt_sourceColumnCount {column : Nat} (support : Source column) :
    column < Spartan.SourceColumnCount := by
  rcases support with ((parent | proof) | logical) | fresh
  · rcases parent with commitment | publicInput | evalK | evalA
    · exact Nat.lt_of_lt_of_le commitment.2 (by
        rw [parentCommitmentStart_eq]
        norm_num [PiDECInputs.commitmentWordsPerChild,
          Spartan.SourceColumnCount])
    · exact Nat.lt_of_lt_of_le publicInput.2 (by
        rw [parentPublicInputStart_eq]
        norm_num [PiDECInputs.publicInputWordsPerChild,
          Spartan.SourceColumnCount])
    · exact Nat.lt_of_lt_of_le evalK.2 (by
        rw [parentEvalKStart_eq]
        norm_num [PiDECInputs.evalKWordsPerChild,
          Spartan.SourceColumnCount])
    · exact Nat.lt_of_lt_of_le evalA.2 (by
        rw [parentEvalAStart_eq]
        norm_num [PiDECInputs.evalAWordsPerChild,
          Spartan.SourceColumnCount])
  · exact Nat.lt_of_lt_of_le proof.2 (by
      norm_num [PiDECInputs.proofInputStart,
        PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
        PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
        PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild,
        Spartan.SourceColumnCount])
  · exact Nat.lt_of_lt_of_le logical.2 (by
      norm_num [PiDECStarts.phaseLogicalStart, PiDECInputs.phaseOffset,
        PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
        PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
        PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
        PiDECInputs.publicInputWordsPerChild, Spartan.SourceColumnCount])
  · exact Nat.lt_of_lt_of_le fresh.2 (by
      norm_num [PiDECStarts.phaseFreshStart, PiDECStarts.phaseLogicalStart,
        PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
        PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
        PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
        PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild,
        freshCount, PiDEC.v1_1.Formal.logicalPrivateCount,
        Spartan.SourceColumnCount])

end NightstreamFPrime.Layout.Stage1.PiDECSourceSupport
