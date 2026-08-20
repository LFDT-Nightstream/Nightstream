import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonReplaySequence

/-!
Contract: exact authoritative word order for the four production PiRLC
Poseidon2 replay runs.

Assurance tier: artifact-checked same-assignment replay authority for the
Nightstream b2/k16 profile.

Owns: exact order and length of all 918 input words and all 54 output words,
including the final two-word tail of each even run.

Does not own: initial or final state placement, final matrix-slice identity,
complete PiRLC semantics, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayAuthority

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompactTrace
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplaySequence
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayTransition
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonRunValues
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonValuePlacement

/-- A structural slice of one authoritative field-word stream. -/
def wordSlice (freshWord : Nat → F) (start count : Nat) : List Nat :=
  (List.range' start count).map fun ordinal => (freshWord ordinal).val

private theorem range'_add (start left right : Nat) :
    List.range' start (left + right) =
      List.range' start left ++ List.range' (start + left) right := by
  apply (List.range'_eq_append_iff).2
  refine ⟨left, by omega, rfl, ?_⟩
  simp

theorem wordSlice_append (freshWord : Nat → F)
    (start left right : Nat) :
    wordSlice freshWord start (left + right) =
      wordSlice freshWord start left ++
        wordSlice freshWord (start + left) right := by
  unfold wordSlice
  rw [range'_add, List.map_append]

private theorem wordSlice_eq_ofFn (freshWord : Nat → F)
    (start count : Nat) :
    wordSlice freshWord start count =
      List.ofFn fun index : Fin count =>
        (freshWord (start + index.val)).val := by
  induction count generalizing start with
  | zero => simp [wordSlice]
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ]
      unfold wordSlice
      rw [List.range'_succ]
      simp only [List.map_cons, Fin.val_zero, Nat.add_zero]
      apply congrArg ((freshWord start).val :: ·)
      change wordSlice freshWord (start + 1) count = _
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
        inductionHypothesis (start := start + 1)

/-- The nested 17-by-54 input frame is the same 918-word ordinal stream. -/
theorem inputReplayWords_eq_wordSlice
    (assignment : Fin productionFinalColumns → F) :
    inputReplayWords assignment =
      wordSlice (inputWordAt assignment) 0 918 := by
  calc
    inputReplayWords assignment =
        (List.ofFn fun source : Fin 17 =>
          List.ofFn fun lane : Fin 54 =>
            (inputWordAt assignment
              (source.val * 54 + lane.val)).val).flatten := by
      unfold inputReplayWords
      apply congrArg List.flatten
      apply congrArg List.ofFn
      funext source
      apply congrArg List.ofFn
      funext lane
      have bounded : source.val * 54 + lane.val < 918 := by
        have sourceBound := source.isLt
        change source.val < 17 at sourceBound
        have laneBound := lane.isLt
        change lane.val < 54 at laneBound
        omega
      simp [inputWordAt, ringDegree, bounded]
    _ = List.ofFn (fun ordinal : Fin (17 * 54) =>
          (inputWordAt assignment ordinal.val).val) := by
      exact (List.ofFn_mul (m := 17) (n := 54)
        (fun ordinal : Fin (17 * 54) =>
          (inputWordAt assignment ordinal.val).val)).symm
    _ = wordSlice (inputWordAt assignment) 0 918 := by
      simpa only [Nat.zero_add] using
        (wordSlice_eq_ofFn (inputWordAt assignment) 0 918).symm

/-- The lane-major output frame is the same 54-word ordinal stream. -/
theorem outputReplayWords_eq_wordSlice
    (assignment : Fin productionFinalColumns → F) :
    outputReplayWords assignment =
      wordSlice (outputWordAt assignment) 0 54 := by
  rw [wordSlice_eq_ofFn]
  unfold outputReplayWords
  apply congrArg List.ofFn
  funext ordinal
  simp [outputWordAt]

private theorem callInput_eq_fresh
    (run : Run) (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (transition : RunReplayTransition run assignment freshWord)
    (index : Nat) (bounded : index < run.raw.callCount)
    (lane : Fin 4) (ordinal : Nat)
    (fresh : (run.callSiteAt index).freshOrdinal lane = some ordinal) :
    callInputs run index assignment (rateLane lane) =
      (freshWord ordinal).val := by
  simpa [callInputs] using congrArg Fin.val
    (transition.fresh ⟨index, bounded⟩ lane ordinal fresh)

private theorem direct_call_words_exact
    (run : Run) (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (transition : RunReplayTransition run assignment freshWord)
    (firstClass : run.raw.firstClass = .direct)
    (firstFreshCount : run.raw.firstFreshCount = 4)
    (index : Nat) (bounded : index < run.raw.callCount) :
    callWordsAt run index assignment =
      wordSlice freshWord (index * 4) 4 := by
  have callShape :
      callWordsAt run index assignment =
        List.ofFn fun lane : Fin 4 =>
          callInputs run index assignment (rateLane lane) := by
    cases index <;> simp [callWordsAt, firstClass]
  rw [callShape, wordSlice_eq_ofFn]
  apply congrArg List.ofFn
  funext lane
  apply callInput_eq_fresh run assignment freshWord transition
    index bounded lane (index * 4 + lane.val)
  cases index with
  | zero =>
      simp [Run.callSiteAt, Run.leafClassAt, CallSite.freshOrdinal,
        firstClass]
  | succ prior =>
      simp [Run.callSiteAt, Run.leafClassAt, CallSite.freshOrdinal,
        firstFreshCount, Nat.succ_mul, Nat.add_assoc, Nat.add_comm,
        Nat.add_left_comm]

private def fin2 : Fin 4 := ⟨2, by decide⟩
private def fin3 : Fin 4 := ⟨3, by decide⟩

private theorem partial_first_call_words_exact
    (run : Run) (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (transition : RunReplayTransition run assignment freshWord)
    (firstClass : run.raw.firstClass = .partialStart)
    (positive : 0 < run.raw.callCount) :
    callWordsAt run 0 assignment = wordSlice freshWord 0 2 := by
  have word2 :
      callInputs run 0 assignment (rateLane fin2) = (freshWord 0).val := by
    apply callInput_eq_fresh run assignment freshWord transition
      0 positive fin2 0
    simp [Run.callSiteAt, Run.leafClassAt, CallSite.freshOrdinal,
      firstClass, fin2]
  have word3 :
      callInputs run 0 assignment (rateLane fin3) = (freshWord 1).val := by
    apply callInput_eq_fresh run assignment freshWord transition
      0 positive fin3 1
    simp [Run.callSiteAt, Run.leafClassAt, CallSite.freshOrdinal,
      firstClass, fin3]
  have callShape :
      callWordsAt run 0 assignment =
        [callInputs run 0 assignment (rateLane fin2),
          callInputs run 0 assignment (rateLane fin3)] := by
    unfold callWordsAt
    rw [firstClass]
    rfl
  rw [callShape, word2, word3]
  rfl

private theorem partial_next_call_words_exact
    (run : Run) (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (transition : RunReplayTransition run assignment freshWord)
    (firstFreshCount : run.raw.firstFreshCount = 2)
    (prior : Nat) (bounded : prior.succ < run.raw.callCount) :
    callWordsAt run prior.succ assignment =
      wordSlice freshWord (2 + prior * 4) 4 := by
  have callShape :
      callWordsAt run prior.succ assignment =
        List.ofFn fun lane : Fin 4 =>
          callInputs run prior.succ assignment (rateLane lane) := by
    simp [callWordsAt]
  rw [callShape, wordSlice_eq_ofFn]
  apply congrArg List.ofFn
  funext lane
  apply callInput_eq_fresh run assignment freshWord transition
    prior.succ bounded lane (2 + prior * 4 + lane.val)
  simp [Run.callSiteAt, Run.leafClassAt, CallSite.freshOrdinal,
    firstFreshCount]

private theorem direct_prefix_exact
    (run : Run) (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (transition : RunReplayTransition run assignment freshWord)
    (firstClass : run.raw.firstClass = .direct)
    (firstFreshCount : run.raw.firstFreshCount = 4)
    (count : Nat) (bounded : count ≤ run.raw.callCount) :
    callWordsPrefix run count assignment =
      wordSlice freshWord 0 (count * 4) := by
  induction count with
  | zero => simp [callWordsPrefix, wordSlice]
  | succ count inductionHypothesis =>
      rw [callWordsPrefix_succ]
      rw [inductionHypothesis (by omega)]
      rw [direct_call_words_exact run assignment freshWord transition
        firstClass firstFreshCount count (by omega)]
      have joined :=
        (wordSlice_append freshWord 0 (count * 4) 4).symm
      rw [Nat.zero_add] at joined
      calc
        wordSlice freshWord 0 (count * 4) ++
            wordSlice freshWord (count * 4) 4 =
          wordSlice freshWord 0 (count * 4 + 4) := joined
        _ = wordSlice freshWord 0 ((count + 1) * 4) :=
          congrArg (wordSlice freshWord 0) (by omega)

private theorem partial_prefix_succ_exact
    (run : Run) (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (transition : RunReplayTransition run assignment freshWord)
    (firstClass : run.raw.firstClass = .partialStart)
    (firstFreshCount : run.raw.firstFreshCount = 2)
    (count : Nat) (bounded : count.succ ≤ run.raw.callCount) :
    callWordsPrefix run count.succ assignment =
      wordSlice freshWord 0 (2 + count * 4) := by
  induction count with
  | zero =>
      simpa [callWordsPrefix] using
        partial_first_call_words_exact run assignment freshWord transition
          firstClass (by omega)
  | succ count inductionHypothesis =>
      rw [callWordsPrefix_succ]
      rw [inductionHypothesis (by omega)]
      rw [partial_next_call_words_exact run assignment freshWord transition
        firstFreshCount count (by omega)]
      have joined :=
        (wordSlice_append freshWord 0 (2 + count * 4) 4).symm
      rw [Nat.zero_add] at joined
      calc
        wordSlice freshWord 0 (2 + count * 4) ++
            wordSlice freshWord (2 + count * 4) 4 =
          wordSlice freshWord 0 (2 + count * 4 + 4) := joined
        _ = wordSlice freshWord 0 (2 + (count + 1) * 4) :=
          congrArg (wordSlice freshWord 0) (by omega)

/-- The only words not permuted by a direct-first run are its final two
words. A partial-first run consumes the complete frame in its calls. -/
def finalTailWords (run : Run) (freshWord : Nat → F) : List Nat :=
  match run.raw.firstClass with
  | .direct => wordSlice freshWord (run.raw.callCount * 4) 2
  | .partialStart => []

/-- Either exact input run consumes the same complete 918-word frame. -/
theorem input_run_words_exact
    (run : Run) (selected : run = evenInputRun ∨ run = oddInputRun)
    (assignment : Fin productionFinalColumns → F)
    (transition : RunReplayTransition run assignment (inputWordAt assignment)) :
    callWordsPrefix run run.raw.callCount assignment ++
        finalTailWords run (inputWordAt assignment) =
      inputReplayWords assignment := by
  rcases selected with rfl | rfl
  · rw [direct_prefix_exact evenInputRun assignment (inputWordAt assignment)
      transition (by rfl) (by rfl) evenInputRun.raw.callCount le_rfl]
    rw [inputReplayWords_eq_wordSlice]
    simpa [finalTailWords, evenInputRun,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenInput]
      using (wordSlice_append (inputWordAt assignment) 0 916 2).symm
  · rw [show oddInputRun.raw.callCount = 230 by rfl]
    rw [partial_prefix_succ_exact oddInputRun assignment
      (inputWordAt assignment) transition (by rfl) (by rfl) 229 (by rfl)]
    simpa [finalTailWords, oddInputRun,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddInput]
      using (inputReplayWords_eq_wordSlice assignment).symm

/-- Either exact output run consumes the same complete 54-word frame. -/
theorem output_run_words_exact
    (run : Run) (selected : run = evenOutputRun ∨ run = oddOutputRun)
    (assignment : Fin productionFinalColumns → F)
    (transition : RunReplayTransition run assignment (outputWordAt assignment)) :
    callWordsPrefix run run.raw.callCount assignment ++
        finalTailWords run (outputWordAt assignment) =
      outputReplayWords assignment := by
  rcases selected with rfl | rfl
  · rw [direct_prefix_exact evenOutputRun assignment (outputWordAt assignment)
      transition (by rfl) (by rfl) evenOutputRun.raw.callCount le_rfl]
    rw [outputReplayWords_eq_wordSlice]
    simpa [finalTailWords, evenOutputRun,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenOutput]
      using (wordSlice_append (outputWordAt assignment) 0 52 2).symm
  · rw [show oddOutputRun.raw.callCount = 14 by rfl]
    rw [partial_prefix_succ_exact oddOutputRun assignment
      (outputWordAt assignment) transition (by rfl) (by rfl) 13 (by rfl)]
    simpa [finalTailWords, oddOutputRun,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddOutput]
      using (outputReplayWords_eq_wordSlice assignment).symm

@[simp] theorem input_run_complete_words_length
    (run : Run) (selected : run = evenInputRun ∨ run = oddInputRun)
    (assignment : Fin productionFinalColumns → F)
    (transition : RunReplayTransition run assignment (inputWordAt assignment)) :
    (callWordsPrefix run run.raw.callCount assignment ++
      finalTailWords run (inputWordAt assignment)).length = 918 := by
  rw [input_run_words_exact run selected assignment transition]
  exact inputReplayWords_length assignment

@[simp] theorem output_run_complete_words_length
    (run : Run) (selected : run = evenOutputRun ∨ run = oddOutputRun)
    (assignment : Fin productionFinalColumns → F)
    (transition : RunReplayTransition run assignment (outputWordAt assignment)) :
    (callWordsPrefix run run.raw.callCount assignment ++
      finalTailWords run (outputWordAt assignment)).length = 54 := by
  rw [output_run_words_exact run selected assignment transition]
  exact outputReplayWords_length assignment

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayAuthority
