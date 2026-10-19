import NightstreamFPrime.Layout.Stage1.PiCCSInputs
import NightstreamFPrime.Layout.Stage1.PiCCSStarts
import NightstreamFPrime.Layout.Stage1.PiRLCInputs
import NightstreamFPrime.Layout.Stage1.PiDECInputs
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

/-- Exact caller-supplied PiCCS interval: the prior child region, then the
proof inputs. -/
def callerInputCount : Nat :=
  PiCCSInputs.phaseOffset - PiCCSInputs.priorChildrenStart

/-- Every retained column before the first transcript permutation: the
caller-supplied interval, then the 270 hinted sign columns of the
statement-binding leaf. -/
def proofInputCount : Nat :=
  PiCCSStarts.statementWitnessStart - PiCCSInputs.priorChildrenStart

/-- Statement, challenge, and round-transcript permutations before the first
ordinary PiCCS child. -/
def transcriptInvocationCount : Nat :=
  (PiCCSStarts.initialClaimLogicalStart - PiCCSStarts.statementWitnessStart) / 1096

def transcriptOutputCount : Nat :=
  transcriptInvocationCount * NightstreamFPrime.Spec.Poseidon2.width

def ordinaryLogicalCount : Nat :=
  PiCCSStarts.outputBindingWitnessStart -
    PiCCSStarts.initialClaimLogicalStart

@[simp] theorem callerInputCount_eq : callerInputCount = 15192 := by
  rw [callerInputCount, PiCCSInputs.phaseOffset_eq,
    PiCCSInputs.priorChildrenStart_eq]

@[simp] theorem proofInputCount_eq : proofInputCount = 15462 := by
  rw [proofInputCount, PiCCSStarts.statementWitnessStart_eq,
    PiCCSInputs.priorChildrenStart_eq]

@[simp] theorem transcriptInvocationCount_eq :
    transcriptInvocationCount = 183 := by
  unfold transcriptInvocationCount PiCCSStarts.initialClaimLogicalStart
  rw [PiCCSStarts.roundTranscriptWitnessStart_eq, PiCCSStarts.statementWitnessStart_eq]

@[simp] theorem transcriptOutputCount_eq : transcriptOutputCount = 2928 := by
  rw [transcriptOutputCount, transcriptInvocationCount_eq]
  norm_num [NightstreamFPrime.Spec.Poseidon2.width]

@[simp] theorem ordinaryLogicalCount_eq : ordinaryLogicalCount = 29591 := by
  unfold ordinaryLogicalCount PiCCSStarts.initialClaimLogicalStart
  rw [PiCCSStarts.outputBindingWitnessStart_eq,
    PiCCSStarts.roundTranscriptWitnessStart_eq]

def External (column : Nat) : Prop :=
  InRange PilotProduction.priorPreimageStart PilotProduction.stateHashWords
      column ∨
    InRange PilotProduction.priorPublicInputStart 270 column ∨
    InRange PilotProduction.outputPreimageStart PilotProduction.stateHashWords
      column ∨
    InRange PiCCSInputs.expectedContextStart PiCCSInputs.expectedContextWords
      column ∨
    InRange PiCCSInputs.priorChildrenStart callerInputCount column

/-- One of the 270 hinted sign columns of the statement-binding leaf. -/
def StatementSign (column : Nat) : Prop :=
  InRange PiCCSStarts.statementBindingLogicalStart 270 column

/-- One of the eight state lanes output by a pre-ordinary PiCCS transcript
permutation. Intermediate permutation recipes are not included. -/
def TranscriptOutput (column : Nat) : Prop :=
  ∃ (invocation : Fin transcriptInvocationCount)
      (lane : Fin NightstreamFPrime.Spec.Poseidon2.width),
    column = PiCCSStarts.statementWitnessStart + invocation.val * 1096 + 1080 + lane.val

def OrdinaryLogical (column : Nat) : Prop :=
  InRange PiCCSStarts.initialClaimLogicalStart ordinaryLogicalCount column

def Logical (column : Nat) : Prop :=
  External column ∨ StatementSign column ∨ TranscriptOutput column ∨
    OrdinaryLogical column

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
    (support : InRange PiCCSInputs.priorChildrenStart callerInputCount column) :
    External column :=
  Or.inr (Or.inr (Or.inr (Or.inr support)))

theorem external_source (column : Nat) (support : External column) :
  Source column :=
  Or.inl (Or.inl support)

theorem statement_sign_source (column : Nat)
    (support : StatementSign column) : Source column :=
  Or.inl (Or.inr (Or.inl support))

theorem transcript_output_source (column : Nat)
    (support : TranscriptOutput column) : Source column :=
  Or.inl (Or.inr (Or.inr (Or.inl support)))

theorem ordinary_logical_source (column : Nat)
    (support : OrdinaryLogical column) : Source column :=
  Or.inl (Or.inr (Or.inr (Or.inr support)))

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
  have phaseValue := congrArg (fun starts : List Nat => starts[4]!)
    PiDECInputs.inputStarts_eq
  change PiDECInputs.phaseOffset = 11464602 at phaseValue
  have sourceLower := Spartan.sourceColumnCount_ge_piDecPhaseOffset
  rw [phaseValue] at sourceLower
  apply Nat.lt_of_lt_of_le ?_ sourceLower
  rcases support with logical | fresh
  · rcases logical with external | transcriptOrOrdinary
    · rcases external with priorRange | publicRange | outputRange |
        contextRange | proofRange
      · exact Nat.lt_of_lt_of_le priorRange.2 (by
          norm_num [PilotProduction.priorPreimageStart,
            PilotProduction.stateHashWords_eq])
      · exact Nat.lt_of_lt_of_le publicRange.2 (by
          norm_num [PilotProduction.priorPublicInputStart,
            PilotProduction.priorPreimageStart,
            PilotProduction.stateHashWords_eq])
      · exact Nat.lt_of_lt_of_le outputRange.2 (by
          norm_num [PilotProduction.outputPreimageStart,
            PilotProduction.priorPublicInputStart,
            PilotProduction.priorPreimageStart,
            Lifecycle.PriorStateHash.publicWidth,
            PilotProduction.stateHashWords_eq, Spec.ringDegree,
            Lifecycle.PaperAlgebra.publicRingColumns])
      · exact Nat.lt_of_lt_of_le contextRange.2 (by
          rw [PiCCSInputs.expectedContextStart_eq]
          norm_num [PiCCSInputs.expectedContextWords])
      · exact Nat.lt_of_lt_of_le proofRange.2 (by
          rw [callerInputCount_eq, PiCCSInputs.priorChildrenStart_eq]
          norm_num)
    · rcases transcriptOrOrdinary with sign | transcript | ordinary
      · unfold StatementSign InRange at sign
        exact Nat.lt_of_lt_of_le sign.2 (by
          unfold PiCCSStarts.statementBindingLogicalStart
          rw [PiCCSInputs.phaseOffset_eq]
          norm_num)
      · rcases transcript with ⟨invocation, lane, rfl⟩
        have invocationBound : invocation.val < 183 := by
          simpa only [transcriptInvocationCount_eq] using invocation.isLt
        have laneBound : lane.val < 16 := by
          simpa only [NightstreamFPrime.Spec.Poseidon2.width] using lane.isLt
        rw [PiCCSStarts.statementWitnessStart_eq]
        omega
      · unfold OrdinaryLogical InRange at ordinary
        exact Nat.lt_of_lt_of_le ordinary.2 (by
          calc
            PiCCSStarts.initialClaimLogicalStart + ordinaryLogicalCount =
                PiCCSStarts.outputBindingWitnessStart := by
              rw [ordinaryLogicalCount_eq,
                PiCCSStarts.outputBindingWitnessStart_eq]
              unfold PiCCSStarts.initialClaimLogicalStart
              rw [PiCCSStarts.roundTranscriptWitnessStart_eq]
            _ ≤ 11464602 := by
              rw [PiCCSStarts.outputBindingWitnessStart_eq]
              norm_num)
  · exact Nat.lt_of_lt_of_le fresh.2 (by
      norm_num [PiRLCInputs.phaseOffset])

end NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport
