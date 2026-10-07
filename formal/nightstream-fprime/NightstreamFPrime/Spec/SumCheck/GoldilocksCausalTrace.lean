import NightstreamFPrime.Spec.SumCheck.GoldilocksCausal

/-!
Connects actual prefix-only prover messages to the existing fixed-phase
bad-challenge and acceptance predicates. An abort produces no certificate.
The verifier predicate retains every round equation and its exact terminal.
-/

namespace NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace

open GoldilocksRoots (ops)
open GoldilocksCausal
attribute [local instance] Classical.propDecidable

/-- Issue each message before using the current challenge to form the next
prefix. A failed message, including a rejected raw width, is represented by none. -/
def issued {degree : Nat} (strategy : Strategy degree) (fixed : List K) :
    List K → Option (List (FixedPolynomial K degree))
  | [] => some []
  | challenge :: rest =>
      match strategy fixed with
      | none => none
      | some message => (issued strategy (fixed ++ [challenge]) rest).map (message :: ·)

def BadChallengeEvent {degree : Nat} (q : List K → K) (strategy : Strategy degree)
    (initial : K) (challengeSetSize : Nat) (challenges : List K) : Prop :=
  ∃ certificate : FixedPhase.Certificate K degree,
    issued strategy [] challenges = some certificate.rounds ∧
      ∃ round, FixedPhase.BadChallenge ops q degree challengeSetSize initial
        challenges certificate round

def FalseAcceptance {degree : Nat} (q : List K → K) (strategy : Strategy degree)
    (initial : K) (challenges : List K) : Prop :=
  ∃ certificate : FixedPhase.Certificate K degree,
    issued strategy [] challenges = some certificate.rounds ∧
      FixedPhase.Accepted ops q initial challenges certificate ∧
        initial ≠ FixedPhase.semanticInitial ops q challenges.length

/-- Neither named event can occur on a trace where the prover aborted. -/
theorem aborted_not_events {degree : Nat} (q : List K → K) (strategy : Strategy degree)
    (initial : K) (challengeSetSize : Nat) (challenges : List K)
    (aborted : issued strategy [] challenges = none) :
    ¬ BadChallengeEvent q strategy initial challengeSetSize challenges ∧
      ¬ FalseAcceptance q strategy initial challenges := by
  constructor
  · rintro ⟨certificate, execution, _⟩
    rw [aborted] at execution
    cases execution
  · rintro ⟨certificate, execution, _⟩
    rw [aborted] at execution
    cases execution

private theorem representable_expectedFrom {degree totalRounds : Nat}
    (q : List K → K)
    (representable : FixedPhase.Sequential.RoundRepresentable ops q degree totalRounds) :
    ∀ (fixed challenges : List K), fixed.length + challenges.length = totalRounds →
      ∀ expected, expected ∈ HypercubeTruth.expectedPolynomialsFrom ops q fixed challenges →
        ∃ polynomial : FixedPolynomial K degree, FixedPhase.Represents ops polynomial expected
  | _, [], _, _, member => by
      simp only [HypercubeTruth.expectedPolynomialsFrom, List.not_mem_nil] at member
  | fixed, challenge :: challenges, length, expected, member => by
      simp only [HypercubeTruth.expectedPolynomialsFrom, List.mem_cons] at member
      rcases member with rfl | member
      · exact representable fixed challenges.length (by
          simp only [List.length_cons] at length
          omega)
      · exact representable_expectedFrom q representable (fixed ++ [challenge]) challenges
          (by
            simp only [List.length_append, List.length_singleton]
            simp only [List.length_cons] at length
            omega) expected member

end NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace
