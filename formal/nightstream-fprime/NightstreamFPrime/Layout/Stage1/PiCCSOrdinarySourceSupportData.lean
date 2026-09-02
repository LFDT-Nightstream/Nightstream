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

/-- Exact caller-supplied PiCCS proof-input interval. -/
def proofInputCount : Nat :=
  PiCCSInputs.phaseOffset - PiCCSInputs.proofInputStart

/-- Statement, challenge, and round-transcript permutations before the first
ordinary PiCCS child. -/
def transcriptInvocationCount : Nat :=
  (PiCCSStarts.initialClaimLogicalStart - PiCCSInputs.phaseOffset) / 592

def transcriptOutputCount : Nat :=
  transcriptInvocationCount * NightstreamFPrime.Spec.Poseidon2.width

def ordinaryLogicalCount : Nat :=
  PiCCSStarts.outputBindingWitnessStart -
    PiCCSStarts.initialClaimLogicalStart

@[simp] theorem proofInputCount_eq : proofInputCount = 29288 := by
  rw [proofInputCount, PiCCSInputs.phaseOffset_eq,
    PiCCSInputs.proofInputStart_eq]

@[simp] theorem transcriptInvocationCount_eq :
    transcriptInvocationCount = 718 := by
  rw [transcriptInvocationCount, PiCCSInputs.phaseOffset_eq]
  norm_num [PiCCSStarts.initialClaimLogicalStart,
    PiCCSStarts.roundTranscriptWitnessStart_eq]

@[simp] theorem transcriptOutputCount_eq : transcriptOutputCount = 5744 := by
  rw [transcriptOutputCount, transcriptInvocationCount_eq]
  norm_num [NightstreamFPrime.Spec.Poseidon2.width]

@[simp] theorem ordinaryLogicalCount_eq : ordinaryLogicalCount = 79846 := by
  rw [ordinaryLogicalCount, PiCCSStarts.outputBindingWitnessStart_eq]
  norm_num [PiCCSStarts.initialClaimLogicalStart,
    PiCCSStarts.roundTranscriptWitnessStart_eq]

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

/-- One of the eight state lanes output by a pre-ordinary PiCCS transcript
permutation. Intermediate permutation recipes are not included. -/
def TranscriptOutput (column : Nat) : Prop :=
  ∃ (invocation : Fin transcriptInvocationCount)
      (lane : Fin NightstreamFPrime.Spec.Poseidon2.width),
    column = PiCCSInputs.phaseOffset + invocation.val * 592 + 584 + lane.val

def OrdinaryLogical (column : Nat) : Prop :=
  InRange PiCCSStarts.initialClaimLogicalStart ordinaryLogicalCount column

def Logical (column : Nat) : Prop :=
  External column ∨ TranscriptOutput column ∨ OrdinaryLogical column

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

theorem transcript_output_source (column : Nat)
    (support : TranscriptOutput column) : Source column :=
  Or.inl (Or.inr (Or.inl support))

theorem ordinary_logical_source (column : Nat)
    (support : OrdinaryLogical column) : Source column :=
  Or.inl (Or.inr (Or.inr support))

theorem local_source (column : Nat)
    (lower : PiCCSStarts.initialClaimLogicalStart ≤ column)
    (upper : column < PiCCSStarts.outputBindingWitnessStart) :
    Source column :=
  ordinary_logical_source column (by
    unfold OrdinaryLogical InRange ordinaryLogicalCount
    omega)

theorem fresh_source (column : Nat)
    (lower : PiCCSStarts.initialClaimFreshStart ≤ column)
    (upper : column < PiRLCInputs.phaseOffset) : Source column :=
  Or.inr ⟨lower, upper⟩

theorem source_target (column : Nat) (support : Source column) :
    Target (Spartan.sourceToSpartan column) :=
  ⟨column, support, rfl⟩

theorem source_lt_sourceColumnCount {column : Nat} (support : Source column) :
    column < Spartan.SourceColumnCount := by
  rcases support with logical | fresh
  · rcases logical with external | transcriptOrOrdinary
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
    · rcases transcriptOrOrdinary with transcript | ordinary
      · rcases transcript with ⟨invocation, lane, rfl⟩
        have invocationBound : invocation.val < 718 := by
          simpa only [transcriptInvocationCount_eq] using invocation.isLt
        have laneBound : lane.val < 8 := by
          simpa only [NightstreamFPrime.Spec.Poseidon2.width] using lane.isLt
        rw [Spartan.sourceColumnCount_eq, PiCCSInputs.phaseOffset_eq]
        omega
      · unfold OrdinaryLogical InRange at ordinary
        exact Nat.lt_of_lt_of_le ordinary.2 (by
          calc
            PiCCSStarts.initialClaimLogicalStart + ordinaryLogicalCount =
                PiCCSStarts.outputBindingWitnessStart := by
              rw [ordinaryLogicalCount_eq,
                PiCCSStarts.outputBindingWitnessStart_eq]
              norm_num [PiCCSStarts.initialClaimLogicalStart,
                PiCCSStarts.roundTranscriptWitnessStart_eq]
            _ ≤ Spartan.SourceColumnCount := by
              rw [Spartan.sourceColumnCount_eq,
                PiCCSStarts.outputBindingWitnessStart_eq]
              norm_num)
  · exact Nat.lt_of_lt_of_le fresh.2 (by
      rw [Spartan.sourceColumnCount_eq]
      norm_num [PiRLCInputs.phaseOffset])

end NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport
