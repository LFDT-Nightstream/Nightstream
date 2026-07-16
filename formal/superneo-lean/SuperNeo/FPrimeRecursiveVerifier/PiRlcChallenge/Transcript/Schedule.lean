import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Semantics

/-!
Owns: the fixed fifteen-rho transcript schedule and exact cursor threading.

Does not own: incoming-cursor authority, concrete Poseidon2, or sampler row
refinement.

Emits constraints: no.

Authority boundary: the initial cursor must already include the authoritative
Pi_CCS output binding; this file preserves, but does not establish, that fact.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `runRhoSchedule`, `fixedRhoSchedule` | `challenge` | Threads one cursor through rho indices zero through fourteen | Authoritative initial cursor and supplied core | No — concrete Poseidon2/Rust refinement open |
| `RhoScheduleAccepts` | `challenge.sampler.acceptance_bound` | Every scheduled sample has at least 54 accepts | Digest trace for each sample | No — Rust refinement open |
| `FixedRhoScheduleSemantics` | `challenge` | Binds the fixed outputs and final cursor to the accepted schedule | Concrete schedule model above | No — Rust refinement open |
| `fixedRhoSchedule_has_fifteen_outputs`, `fixedRhoSchedule_outputs_have_fixed_length` | `challenge.sampler.selection` | Exactly fifteen outputs of length 54 | Accepted fixed schedule | No — Rust refinement open |

This file starts from an already-authoritative incoming cursor. Binding the
Pi_CCS output digest into that cursor and refining `Poseidon2Core` to the
concrete production permutation are separate composition obligations.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-- Observable data from the fixed schedule, retained for conformance checks. -/
structure RhoScheduleResult where
  cursor : SpongeCursor
  chunks : List (List Chunk)
  outputs : List (List Int)

/-- Replay `count` consecutive samples, threading the exact sponge cursor. -/
def runRhoSchedule (core : Poseidon2Core) :
    Nat → SpongeCursor → Nat → RhoScheduleResult
  | 0, cursor, _ =>
      { cursor := cursor, chunks := [], outputs := [] }
  | count + 1, cursor, rhoIndex =>
      let sample := rhoDigestTrace core cursor rhoIndex
      let rest := runRhoSchedule core count sample.cursor (rhoIndex + 1)
      { cursor := rest.cursor
        chunks := sample.chunks :: rest.chunks
        outputs := firstAcceptedSymbols sample.chunks :: rest.outputs }

/-- Every scheduled sample passes the fixed rejection-slack condition. -/
def RhoScheduleAccepts (core : Poseidon2Core) :
    Nat → SpongeCursor → Nat → Prop
  | 0, _, _ => True
  | count + 1, cursor, rhoIndex =>
      let sample := rhoDigestTrace core cursor rhoIndex
      EnoughAccepts sample.chunks ∧
        RhoScheduleAccepts core count sample.cursor (rhoIndex + 1)

/-- The one production schedule: rho indices zero through fourteen. -/
def fixedRhoSchedule
    (core : Poseidon2Core) (cursor : SpongeCursor) : RhoScheduleResult :=
  runRhoSchedule core rhoCount cursor 0

/-- Accepted semantics for the one production schedule and its final cursor. -/
def FixedRhoScheduleSemantics
    (core : Poseidon2Core) (initial : SpongeCursor)
    (result : RhoScheduleResult) : Prop :=
  result = fixedRhoSchedule core initial ∧
    RhoScheduleAccepts core rhoCount initial 0

@[simp] theorem runRhoSchedule_chunks_length
    (core : Poseidon2Core) (count : Nat)
    (cursor : SpongeCursor) (rhoIndex : Nat) :
    (runRhoSchedule core count cursor rhoIndex).chunks.length = count := by
  induction count generalizing cursor rhoIndex with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [runRhoSchedule, List.length_cons]
      rw [inductionHypothesis]

@[simp] theorem runRhoSchedule_outputs_length
    (core : Poseidon2Core) (count : Nat)
    (cursor : SpongeCursor) (rhoIndex : Nat) :
    (runRhoSchedule core count cursor rhoIndex).outputs.length = count := by
  induction count generalizing cursor rhoIndex with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [runRhoSchedule, List.length_cons]
      rw [inductionHypothesis]

theorem runRhoSchedule_outputs_have_fixed_length
    (core : Poseidon2Core) (count : Nat)
    (cursor : SpongeCursor) (rhoIndex : Nat)
    (hAccepts : RhoScheduleAccepts core count cursor rhoIndex) :
    ∀ output ∈ (runRhoSchedule core count cursor rhoIndex).outputs,
      output.length = outputLength := by
  induction count generalizing cursor rhoIndex with
  | zero => simp [runRhoSchedule]
  | succ count inductionHypothesis =>
      let sample := rhoDigestTrace core cursor rhoIndex
      have hHead : EnoughAccepts sample.chunks := by
        exact hAccepts.1
      have hTail : RhoScheduleAccepts core count sample.cursor (rhoIndex + 1) := by
        exact hAccepts.2
      intro output hMember
      simp only [runRhoSchedule, List.mem_cons] at hMember
      rcases hMember with hOutput | hOutput
      · subst output
        exact firstAcceptedSymbols_length sample.chunks hHead
      · exact inductionHypothesis sample.cursor (rhoIndex + 1) hTail output hOutput

theorem fixedRhoSchedule_has_fifteen_outputs
    (core : Poseidon2Core) (cursor : SpongeCursor) :
    (fixedRhoSchedule core cursor).outputs.length = 15 := by
  simp [fixedRhoSchedule, rhoCount]

theorem fixedRhoSchedule_outputs_have_fixed_length
    (core : Poseidon2Core) (cursor : SpongeCursor)
    (hAccepts : RhoScheduleAccepts core rhoCount cursor 0) :
    ∀ output ∈ (fixedRhoSchedule core cursor).outputs,
      output.length = outputLength := by
  exact runRhoSchedule_outputs_have_fixed_length
    core rhoCount cursor 0 hAccepts

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
