import NightstreamFPrime.Layout.Stage1.PiCCSInputs
import NightstreamFPrime.Layout.Stage1.PiCCSStarts
import NightstreamFPrime.Layout.Stage1.Spartan

/-!
Owns the compact source families selected for PiCCS ordinary-row lowering.

The source predicate is stated before Spartan remapping. `Target` is its exact
image under the established Spartan column permutation. This module does not
prove that PiCCS constraints use only these families.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport

def InRange (start count column : Nat) : Prop :=
  start ≤ column ∧ column < start + count

def External (column : Nat) : Prop :=
  InRange PilotProduction.priorPreimageStart PilotProduction.stateHashWords
      column ∨
    InRange PilotProduction.priorPublicInputStart 270 column ∨
    InRange PilotProduction.outputPreimageStart PilotProduction.stateHashWords
      column ∨
    InRange PiCCSInputs.expectedContextStart PiCCSInputs.expectedContextWords
      column ∨
    InRange PiCCSInputs.proofInputStart
      (PiCCSInputs.phaseOffset - PiCCSInputs.proofInputStart) column

def Logical (column : Nat) : Prop :=
  External column ∨
    PiCCSInputs.phaseOffset ≤ column ∧
      column < PiCCSStarts.outputBindingWitnessStart

def Source (column : Nat) : Prop :=
  Logical column ∨
    PiCCSStarts.initialClaimFreshStart ≤ column ∧
      column < PiRLCInputs.phaseOffset

def Target (column : Nat) : Prop :=
  ∃ source, Source source ∧ Spartan.sourceToSpartan source = column

theorem external_prior (column : Nat)
    (support : InRange PilotProduction.priorPreimageStart
      PilotProduction.stateHashWords column) : External column :=
  Or.inl support

theorem external_public (column : Nat)
    (support : InRange PilotProduction.priorPublicInputStart 270 column) :
    External column :=
  Or.inr (Or.inl support)

theorem external_output (column : Nat)
    (support : InRange PilotProduction.outputPreimageStart
      PilotProduction.stateHashWords column) : External column :=
  Or.inr (Or.inr (Or.inl support))

theorem external_context (column : Nat)
    (support : InRange PiCCSInputs.expectedContextStart
      PiCCSInputs.expectedContextWords column) : External column :=
  Or.inr (Or.inr (Or.inr (Or.inl support)))

theorem external_proof (column : Nat)
    (support : InRange PiCCSInputs.proofInputStart
      (PiCCSInputs.phaseOffset - PiCCSInputs.proofInputStart) column) :
    External column :=
  Or.inr (Or.inr (Or.inr (Or.inr support)))

theorem external_source (column : Nat) (support : External column) :
    Source column :=
  Or.inl (Or.inl support)

theorem local_source (column : Nat)
    (lower : PiCCSInputs.phaseOffset ≤ column)
    (upper : column < PiCCSStarts.outputBindingWitnessStart) :
    Source column :=
  Or.inl (Or.inr ⟨lower, upper⟩)

theorem fresh_source (column : Nat)
    (lower : PiCCSStarts.initialClaimFreshStart ≤ column)
    (upper : column < PiRLCInputs.phaseOffset) : Source column :=
  Or.inr ⟨lower, upper⟩

theorem source_target (column : Nat) (support : Source column) :
    Target (Spartan.sourceToSpartan column) :=
  ⟨column, support, rfl⟩

theorem source_lt_sourceColumnCount {column : Nat} (support : Source column) :
    column < Spartan.SourceColumnCount := by
  rcases support with (external | logicalRange) | fresh
  · rcases external with priorRange | publicRange | outputRange |
      contextRange | proofRange
    · exact Nat.lt_of_lt_of_le priorRange.2 (by
        rw [Spartan.sourceColumnCount_eq]
        norm_num [PilotProduction.priorPreimageStart,
          PilotProduction.stateHashWords_eq])
    · exact Nat.lt_of_lt_of_le publicRange.2 (by
        rw [Spartan.sourceColumnCount_eq]
        norm_num [PilotProduction.priorPublicInputStart,
          PilotProduction.priorPreimageStart,
          PilotProduction.stateHashWords_eq])
    · exact Nat.lt_of_lt_of_le outputRange.2 (by
        rw [Spartan.sourceColumnCount_eq]
        norm_num [PilotProduction.outputPreimageStart,
          PilotProduction.priorPublicInputStart,
          PilotProduction.priorPreimageStart,
          Lifecycle.PriorStateHash.publicWidth,
          PilotProduction.stateHashWords_eq, Spec.ringDegree,
          Lifecycle.PaperAlgebra.publicRingColumns])
    · exact Nat.lt_of_lt_of_le contextRange.2 (by
        rw [Spartan.sourceColumnCount_eq, PiCCSInputs.expectedContextStart_eq]
        norm_num [PiCCSInputs.expectedContextWords])
    · exact Nat.lt_of_lt_of_le proofRange.2 (by
        rw [Spartan.sourceColumnCount_eq, PiCCSInputs.phaseOffset_eq,
          PiCCSInputs.proofInputStart_eq]
        norm_num)
  · exact Nat.lt_of_lt_of_le logicalRange.2 (by
      rw [Spartan.sourceColumnCount_eq,
        PiCCSStarts.outputBindingWitnessStart_eq]
      norm_num)
  · exact Nat.lt_of_lt_of_le fresh.2 (by
      rw [Spartan.sourceColumnCount_eq]
      norm_num [PiRLCInputs.phaseOffset])

end NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport
