import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidates

/-!
Contract: semantic induction for the 64-candidate Lean-owned `Pi_RLC`
classification batch.

The induction proves that every candidate's prior expression is the exact
accepted-prefix count forced by all preceding candidate rows.  Thus the
candidate-local `< 64` premise is constructed internally rather than supplied
by a caller.

This layer does not yet prove first-accepted output selection or connect each
16-bit source slice back to the value-level Poseidon2 lane.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidates
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

def candidateOfNat (index : Nat) : Fin candidatesPerScalar :=
  ⟨index % candidatesPerScalar, Nat.mod_lt _ (by
    simp [candidatesPerScalar])⟩

theorem candidateOfNat_val
    {index : Nat} (bounded : index < candidatesPerScalar) :
    (candidateOfNat index).val = index := by
  exact Nat.mod_eq_of_lt bounded

theorem candidateOfNat_eq
    (candidate : Fin candidatesPerScalar) :
    candidateOfNat candidate.val = candidate := by
  apply Fin.ext
  exact candidateOfNat_val candidate.isLt

def acceptWire
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidate : Fin candidatesPerScalar) : Nat :=
  assignment
    (PiRlcCanonicalCandidate.acceptColumn
      (candidateLayout duplexBase u64Base candidateBase initial
        coordinate candidate))

def acceptedPrefix
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat) : Nat → Nat
  | 0 => 0
  | index + 1 =>
      acceptedPrefix duplexBase u64Base candidateBase initial coordinate
        assignment index +
      acceptWire duplexBase u64Base candidateBase initial coordinate
        assignment (candidateOfNat index)

theorem acceptWire_le_one
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase count initial) assignment)
    (coordinate : Fin count) (candidate : Fin candidatesPerScalar) :
    acceptWire duplexBase u64Base candidateBase initial coordinate
      assignment candidate ≤ 1 := by
  have sourceBits := sourceBitsBoolean prime duplexBase u64Base candidateBase
    count initial canonical constantWire u64Satisfied coordinate candidate
  have exactAcceptance :=
    PiRlcCanonicalCandidateSound.acceptance_sound prime canonical constantWire
      sourceBits
      (satisfies_candidate duplexBase u64Base candidateBase count initial
        assignment candidateSatisfied coordinate candidate)
  unfold acceptWire
  rw [exactAcceptance]
  split <;> simp

theorem acceptedPrefix_le
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase count initial) assignment)
    (coordinate : Fin count) :
    ∀ index,
      acceptedPrefix duplexBase u64Base candidateBase initial coordinate
        assignment index ≤ index := by
  intro index
  induction index with
  | zero => exact Nat.le_refl _
  | succ index inductionHypothesis =>
      simp only [acceptedPrefix]
      have bitBound := acceptWire_le_one prime duplexBase u64Base candidateBase
        count initial canonical constantWire u64Satisfied candidateSatisfied
        coordinate (candidateOfNat index)
      omega

private theorem prior_eval_zero
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat) :
    lcEval assignment
        (candidateLayout duplexBase u64Base candidateBase initial coordinate
          (candidateOfNat 0)).prior =
      0 := by
  simp [candidateLayout, prior, candidateOfNat, lcEval]

private theorem prior_eval_successor
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    {index : Nat} (positive : 0 < index)
    (bounded : index < candidatesPerScalar) :
    lcEval assignment
        (candidateLayout duplexBase u64Base candidateBase initial coordinate
          (candidateOfNat index)).prior =
      assignment
        (PiRlcCanonicalCandidate.cumulativeColumn
          (candidateLayout duplexBase u64Base candidateBase initial coordinate
            (candidateOfNat (index - 1)))) := by
  have currentValue : (candidateOfNat index).val = index :=
    candidateOfNat_val bounded
  have previousBound : index - 1 < candidatesPerScalar := by
    omega
  have previousValue : (candidateOfNat (index - 1)).val = index - 1 :=
    candidateOfNat_val previousBound
  have nonzero : index ≠ 0 := by omega
  unfold candidateLayout prior
  rw [if_neg (by simpa [currentValue] using nonzero)]
  unfold lcEval
  simp only [List.foldl, Nat.one_mul, Nat.zero_add]
  rw [Nat.mod_eq_of_lt (canonical _)]
  congr 1
  simp only [PiRlcCanonicalCandidate.cumulativeColumn, occurrenceBase,
    occurrenceIndex, currentValue, previousValue,
    candidatesPerScalar, PiRlcCanonicalCandidate.auxiliaryCount]
  omega

/-- Every candidate refines the verifier decision and symbol, and its
cumulative column equals the exact accepted-prefix count through that
candidate. -/
theorem candidate_refines_nat
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase count initial) assignment)
    (coordinate : Fin count)
    (index : Nat) (bounded : index < candidatesPerScalar) :
    ∃ sourceBits :
        PiRlcCanonicalCandidateSound.SourceBitsBoolean assignment
          (candidateLayout duplexBase u64Base candidateBase initial coordinate
            (candidateOfNat index)),
      PiRlcCanonicalCandidateSound.Refines assignment
          (candidateLayout duplexBase u64Base candidateBase initial coordinate
            (candidateOfNat index))
          sourceBits ∧
        assignment
            (PiRlcCanonicalCandidate.cumulativeColumn
              (candidateLayout duplexBase u64Base candidateBase initial
                coordinate (candidateOfNat index))) =
          acceptedPrefix duplexBase u64Base candidateBase initial coordinate
            assignment (index + 1) := by
  induction index using Nat.strongRecOn with
  | ind index inductionHypothesis =>
      have sourceBits := sourceBitsBoolean prime duplexBase u64Base
        candidateBase count initial canonical constantWire u64Satisfied
        coordinate (candidateOfNat index)
      have localSatisfied :=
        satisfies_candidate duplexBase u64Base candidateBase count initial
          assignment candidateSatisfied coordinate (candidateOfNat index)
      have priorEval :
          lcEval assignment
              (candidateLayout duplexBase u64Base candidateBase initial
                coordinate (candidateOfNat index)).prior =
            acceptedPrefix duplexBase u64Base candidateBase initial coordinate
              assignment index := by
        by_cases zero : index = 0
        · subst index
          exact prior_eval_zero duplexBase u64Base candidateBase initial
            coordinate assignment
        · have positive : 0 < index := by omega
          rw [prior_eval_successor duplexBase u64Base candidateBase initial
            coordinate assignment canonical positive bounded]
          have previousBound : index - 1 < candidatesPerScalar := by
            omega
          have previous :=
            inductionHypothesis (index - 1) (by omega) previousBound
          rcases previous with ⟨_, _, previousCumulative⟩
          have indexEq : index = index - 1 + 1 := by omega
          rw [indexEq]
          exact previousCumulative
      have prefixBound :=
        acceptedPrefix_le prime duplexBase u64Base candidateBase count initial
          canonical constantWire u64Satisfied candidateSatisfied coordinate
          index
      have priorBound :
          lcEval assignment
              (candidateLayout duplexBase u64Base candidateBase initial
                coordinate (candidateOfNat index)).prior <
            ProductionAlphabet.candidateBound := by
        rw [priorEval]
        change
          acceptedPrefix duplexBase u64Base candidateBase initial coordinate
              assignment index <
            64
        change index < 64 at bounded
        omega
      have refined := PiRlcCanonicalCandidateSound.sound prime canonical
        constantWire sourceBits priorBound localSatisfied
      refine ⟨sourceBits, refined, ?_⟩
      rw [refined.cumulative, priorEval, ← refined.accepted]
      rfl

theorem candidate_refines
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase count initial) assignment)
    (coordinate : Fin count) (candidate : Fin candidatesPerScalar) :
    ∃ sourceBits :
        PiRlcCanonicalCandidateSound.SourceBitsBoolean assignment
          (candidateLayout duplexBase u64Base candidateBase initial coordinate
            candidate),
      PiRlcCanonicalCandidateSound.Refines assignment
          (candidateLayout duplexBase u64Base candidateBase initial coordinate
            candidate)
          sourceBits ∧
        assignment
            (PiRlcCanonicalCandidate.cumulativeColumn
              (candidateLayout duplexBase u64Base candidateBase initial
                coordinate candidate)) =
          acceptedPrefix duplexBase u64Base candidateBase initial coordinate
            assignment (candidate.val + 1) := by
  have result := candidate_refines_nat prime duplexBase u64Base candidateBase
    count initial canonical constantWire u64Satisfied candidateSatisfied
    coordinate candidate.val candidate.isLt
  simpa [candidateOfNat_eq candidate] using result

/-- The source expression read immediately before a candidate equals the exact
number of verifier-accepted candidates in the preceding physical prefix. -/
theorem prior_refines
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase count initial) assignment)
    (coordinate : Fin count) (candidate : Fin candidatesPerScalar) :
    lcEval assignment
        (candidateLayout duplexBase u64Base candidateBase initial coordinate
          candidate).prior =
      acceptedPrefix duplexBase u64Base candidateBase initial coordinate
        assignment candidate.val := by
  by_cases zero : candidate.val = 0
  · have candidateZero : candidate = candidateOfNat 0 := by
      apply Fin.ext
      simpa [zero] using (candidateOfNat_val (by decide : 0 < candidatesPerScalar)).symm
    rw [candidateZero]
    exact prior_eval_zero duplexBase u64Base candidateBase initial coordinate
      assignment
  · have positive : 0 < candidate.val := by omega
    have bounded := candidate.isLt
    have candidateCurrent : candidateOfNat candidate.val = candidate :=
      candidateOfNat_eq candidate
    have currentPrior :=
      prior_eval_successor duplexBase u64Base candidateBase initial coordinate
        assignment canonical positive bounded
    have previousBound : candidate.val - 1 < candidatesPerScalar := by omega
    have previous := candidate_refines_nat prime duplexBase u64Base
      candidateBase count initial canonical constantWire u64Satisfied
      candidateSatisfied coordinate (candidate.val - 1) previousBound
    rcases previous with ⟨_, _, previousCumulative⟩
    have indexEq : candidate.val = candidate.val - 1 + 1 := by omega
    calc
      lcEval assignment
          (candidateLayout duplexBase u64Base candidateBase initial coordinate
            candidate).prior =
        lcEval assignment
          (candidateLayout duplexBase u64Base candidateBase initial coordinate
            (candidateOfNat candidate.val)).prior := by
              rw [candidateCurrent]
      _ = assignment
          (PiRlcCanonicalCandidate.cumulativeColumn
            (candidateLayout duplexBase u64Base candidateBase initial
              coordinate (candidateOfNat (candidate.val - 1)))) := currentPrior
      _ =
        acceptedPrefix duplexBase u64Base candidateBase initial coordinate
          assignment (candidate.val - 1 + 1) := previousCumulative
      _ = acceptedPrefix duplexBase u64Base candidateBase initial coordinate
          assignment candidate.val := congrArg
            (acceptedPrefix duplexBase u64Base candidateBase initial coordinate
              assignment)
            indexEq.symm

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesSound
