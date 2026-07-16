import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursiveAccumulatorArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryRecursiveAccumulatorCoreSound
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursiveAccumulatorCoreNative

/-!
Contract: compose the recursive accumulator input link, direct accumulator-v1
digest core, and output link.

| Branch | Exact rows | Mathematical obligation | Emits constraints? |
|---|---:|---|---|
| `running_link` | 4 | Bind the carried input handle to the PiCCS accumulator | yes |
| `digest_core` | 254,911 | Serialize the checked PiDEC parent and evaluate Poseidon2 | yes |
| `output_link` | 4 | Bind the recomputed handle to claimed and outgoing state | yes |

Owns: exact composition of all 254,919 recursive accumulator rows.
Does not own: PiCCS/PiDEC acceptance, PiRLC parent authority, or `y_zcol`
validation.
Authority boundary: the four-lane digest is compression only.  Its preimage
is the exact accumulator-v1 projection exposed by the checked core.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulator

def LinksHold (pairs : List (Nat × Nat)) (assignment : Nat → Nat) : Prop :=
  ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2

def linksCheck (pairs : List (Nat × Nat))
    (assignment : Nat → Nat) : Bool :=
  pairs.all fun pair => decide (assignment pair.1 = assignment pair.2)

theorem linksCheck_eq_true_iff (pairs : List (Nat × Nat))
    (assignment : Nat → Nat) :
    linksCheck pairs assignment = true ↔ LinksHold pairs assignment := by
  simp [linksCheck, LinksHold, List.all_eq_true, decide_eq_true_eq]

private theorem linked_values
    {pairs : List (Nat × Nat)} {assignment : Nat → Nat}
    (links : LinksHold pairs assignment) :
    pairs.map (fun pair => assignment pair.1) =
      pairs.map (fun pair => assignment pair.2) := by
  induction pairs with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      have headLink := links head (by simp)
      have tailLinks : LinksHold tail assignment := by
        intro pair member
        exact links pair (by simp [member])
      simp [headLink, inductionHypothesis tailLinks]

/-- Independent conclusions of all 254,919 exact rows. -/
structure Facts (assignment : Nat → Nat) : Prop where
  runningDigest : LinksHold
    FPrimeFullHistoryRecursiveAccumulatorRunningLink.pairs assignment
  core : FPrimeFullHistoryRecursiveAccumulatorCoreSound.Facts assignment
  outputDigest : LinksHold
    FPrimeFullHistoryRecursiveAccumulatorOutputLink.pairs assignment

def handle (assignment : Nat → Nat) : List Nat :=
  accumulatorDigestColumns.map assignment

def stateInputHandle (assignment : Nat → Nat) : List Nat :=
  stateInputAccumulatorDigestColumns.map assignment

def stateOutputHandle (assignment : Nat → Nat) : List Nat :=
  stateOutputAccumulatorDigestColumns.map assignment

theorem Facts.running_eq_stateInput
    {assignment : Nat → Nat} (facts : Facts assignment) :
    runningAccumulatorDigestColumns.map assignment =
      stateInputHandle assignment := by
  exact linked_values facts.runningDigest

theorem Facts.claimed_eq_handle
    {assignment : Nat → Nat} (facts : Facts assignment) :
    claimedAccumulatorDigestColumns.map assignment = handle assignment := by
  exact linked_values facts.outputDigest

theorem Facts.handle_eq_stateOutput
    {assignment : Nat → Nat} (_facts : Facts assignment) :
    handle assignment = stateOutputHandle assignment := by
  simp [handle, stateOutputHandle, accumulatorDigestColumns,
    FPrimeFullHistoryRecursiveAccumulator.recomputed_is_state_output]

/-- The installed handle is the pure Poseidon2 evaluation of the complete
accumulator-v1 projection of the checked recursive PiDEC parent. -/
theorem Facts.handle_recomputed
    {assignment : Nat → Nat} (facts : Facts assignment) :
    ∀ lane, lane < 4 →
      assignment (accumulatorDigestColumns.getD lane 0) =
        Poseidon2Sponge.runValueRounds
          FPrimeFullHistoryRecursiveAccumulatorCoreSound.digestTrace.rounds
          (FPrimeFullHistoryAccumulatorClaimSerialization.recursiveParentPreimage
            assignment)
          (fun _ => 0) lane := by
  intro lane laneLt
  have recomputed := facts.core.accumulatorDigest lane laneLt
  rw [FPrimeFullHistoryRecursiveAccumulatorCoreSound.digestTrace_outputColumns,
    FPrimeFullHistoryRecursiveAccumulatorCoreSound.digestTrace_inputColumns,
    facts.core.parentClaimSource] at recomputed
  simpa [accumulatorDigestColumns,
    FPrimeFullHistoryRecursiveAccumulator.recomputed_is_core_output]
    using recomputed

/-- Exact-row CIR-SOUND for the complete recursive accumulator owner. -/
theorem sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Facts assignment := by
  have pieces := (satisfies_flatten_iff rowPieces assignment).mp satisfies
  have runningRows := pieces runningDigestRows (by simp [rowPieces])
  have coreRowsSatisfy := pieces coreRows (by simp [rowPieces])
  have outputRows := pieces outputDigestRows (by simp [rowPieces])
  exact {
    runningDigest := EqualityPins.rows_sound canonical one runningRows
    core := FPrimeFullHistoryRecursiveAccumulatorCoreSound.sound
      canonical one coreRowsSatisfy
    outputDigest := EqualityPins.rows_sound canonical one outputRows
  }

/-- Independent executable checker for the three semantic owner pieces. -/
def nativeCheck (assignment : Nat → Nat) : Bool :=
  [ linksCheck FPrimeFullHistoryRecursiveAccumulatorRunningLink.pairs assignment
  , FPrimeFullHistoryRecursiveAccumulatorCoreNative.nativeCheck assignment
  , linksCheck FPrimeFullHistoryRecursiveAccumulatorOutputLink.pairs assignment
  ].all id

theorem nativeCheck_eq_true_iff (assignment : Nat → Nat) :
    nativeCheck assignment = true ↔ Facts assignment := by
  simp only [nativeCheck, List.all_cons, List.all_nil, id_eq,
    Bool.and_eq_true, and_true, linksCheck_eq_true_iff,
    FPrimeFullHistoryRecursiveAccumulatorCoreNative.nativeCheck_eq_true_iff]
  constructor
  · rintro ⟨runningDigest, core, outputDigest⟩
    exact ⟨runningDigest, core, outputDigest⟩
  · intro facts
    exact ⟨facts.runningDigest, facts.core, facts.outputDigest⟩

theorem nativeCheck_of_satisfies
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    nativeCheck assignment = true :=
  (nativeCheck_eq_true_iff assignment).2
    (sound canonical one satisfies)

/-- Native/compiler data sufficient to build every owner row.  No R1CS
satisfaction proposition or acceptance bit is supplied by the caller. -/
structure CompilerWitness (assignment : Nat → Nat) where
  runningDigest : LinksHold
    FPrimeFullHistoryRecursiveAccumulatorRunningLink.pairs assignment
  core : FPrimeFullHistoryRecursiveAccumulatorCoreSound.CompilerWitness assignment
  outputDigest : LinksHold
    FPrimeFullHistoryRecursiveAccumulatorOutputLink.pairs assignment

/-- Exact CIR-COMPLETE for the aggregate owner. -/
theorem complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : CompilerWitness assignment) :
    Satisfies rows assignment := by
  have runningRows : Satisfies runningDigestRows assignment :=
    EqualityPins.rows_complete canonical one witness.runningDigest
  have coreRowsSatisfy : Satisfies coreRows assignment :=
    FPrimeFullHistoryRecursiveAccumulatorCoreSound.complete
      canonical one witness.core
  have outputRows : Satisfies outputDigestRows assignment :=
    EqualityPins.rows_complete canonical one witness.outputDigest
  apply (satisfies_flatten_iff rowPieces assignment).mpr
  intro piece member
  change piece ∈ [runningDigestRows, coreRows, outputDigestRows] at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact runningRows
  · exact coreRowsSatisfy
  · exact outputRows

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorSound
