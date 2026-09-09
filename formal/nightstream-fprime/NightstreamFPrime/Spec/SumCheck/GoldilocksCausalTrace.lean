import NightstreamFPrime.Spec.SumCheck.GoldilocksCausal

/-!
Connects actual prefix-only prover messages to the existing fixed-phase
collision and acceptance predicates. An abort produces no certificate.
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

private theorem issued_cons {degree : Nat} (strategy : Strategy degree)
    (fixed : List K) (challenge : K) (rest : List K) (result : List (FixedPolynomial K degree))
    (execution : issued strategy fixed (challenge :: rest) = some result) :
    ∃ message tail, strategy fixed = some message ∧
      issued strategy (fixed ++ [challenge]) rest = some tail ∧ result = message :: tail := by
  cases chosen : strategy fixed with
  | none =>
      simp only [issued, chosen] at execution
      cases execution
  | some message =>
      cases later : issued strategy (fixed ++ [challenge]) rest with
      | none =>
          simp only [issued, chosen, later, Option.map_none] at execution
          cases execution
      | some tail =>
          refine ⟨message, tail, rfl, rfl, ?_⟩
          have equal : some (message :: tail) = some result := by
            simpa only [issued, chosen, later, Option.map_some] using execution
          exact (Option.some.inj equal).symm

private theorem prefixHit_implies_collision {degree : Nat}
    (q : List K → K) (strategy : Strategy degree) (before : List K)
    (fixed : List K) (challenge : K) (after : List K)
    (beforeMessages : List (FixedPolynomial K degree)) (message : FixedPolynomial K degree)
    (afterMessages : List (FixedPolynomial K degree))
    (length : before.length = beforeMessages.length)
    (execution : issued strategy fixed (before ++ challenge :: after) =
      some (beforeMessages ++ message :: afterMessages))
    (hit : Hit q (fixed ++ before) after.length message challenge) :
    Collision q strategy fixed (before ++ challenge :: after) := by
  induction before generalizing fixed beforeMessages with
  | nil =>
      have empty : beforeMessages = [] := by
        cases beforeMessages with
        | nil => rfl
        | cons first rest => simp only [List.length_nil, List.length_cons] at length; omega
      subst beforeMessages
      have executionHead : issued strategy fixed (challenge :: after) =
          some (message :: afterMessages) := by simpa only [List.nil_append] using execution
      obtain ⟨observed, tail, chosen, _later, equal⟩ :=
        issued_cons strategy fixed challenge after (message :: afterMessages) executionHead
      have messageEqual := (List.cons.inj equal).1
      subst observed
      simp only [List.nil_append, Collision, chosen]
      exact Or.inl (by simpa only [List.append_nil] using hit)
  | cons first before ih =>
      cases beforeMessages with
      | nil => simp only [List.length_cons, List.length_nil] at length; omega
      | cons firstMessage beforeMessages =>
          have executionHead : issued strategy fixed (first :: (before ++ challenge :: after)) =
              some (firstMessage :: (beforeMessages ++ message :: afterMessages)) := by
            simpa only [List.cons_append] using execution
          obtain ⟨observed, tail, chosen, later, equal⟩ := issued_cons strategy fixed first
            (before ++ challenge :: after) (firstMessage :: (beforeMessages ++ message :: afterMessages))
            executionHead
          have tailEqual := (List.cons.inj equal).2
          have tailExecution : issued strategy (fixed ++ [first]) (before ++ challenge :: after) =
              some (beforeMessages ++ message :: afterMessages) :=
            later.trans (congrArg some tailEqual.symm)
          have tailLength : before.length = beforeMessages.length := Nat.succ.inj length
          have tailHit : Hit q ((fixed ++ [first]) ++ before) after.length message challenge := by
            simpa only [List.append_assoc, List.singleton_append] using hit
          simp only [List.cons_append, Collision, chosen]
          exact Or.inr (ih (fixed ++ [first]) beforeMessages tailLength tailExecution tailHit)

/-- The existing fixed-phase collision is a collision on the actual issued
message path. No sampled polynomial can be replaced by future-dependent advice. -/
theorem badChallenge_implies_collision {degree challengeSetSize : Nat}
    (q : List K → K) (strategy : Strategy degree) (initial : K)
    (challenges : List K) (certificate : FixedPhase.Certificate K degree)
    (execution : issued strategy [] challenges = some certificate.rounds)
    (bad : ∃ round, FixedPhase.BadChallenge ops q degree challengeSetSize initial
      challenges certificate round) :
    Collision q strategy [] challenges := by
  classical
  obtain ⟨before, challenge, after, beforeMessages, message, afterMessages,
      challengesEqual, messagesEqual, length, different, equal⟩ :=
    FixedPhase.badChallenge_implies_causal_decomposition ops q challengeSetSize
      initial challenges certificate bad
  have selectedExecution := execution
  rw [challengesEqual, messagesEqual] at selectedExecution
  have hit : Hit q ([] ++ before) after.length message challenge := by
    constructor
    · by_contra same
      apply different
      funext point
      by_contra unequal
      exact same ⟨point, by simpa only [expected, List.nil_append] using unequal⟩
    · simpa only [expected, List.nil_append] using equal
  rw [challengesEqual]
  exact prefixHit_implies_collision q strategy before [] challenge after
    beforeMessages message afterMessages length selectedExecution hit

/-- Actual verifier false acceptance on an issued certificate implies the
same causal collision event used by the uniform probability theorem. -/
theorem false_acceptance_implies_collision {degree challengeSetSize : Nat}
    (q : List K → K) (strategy : Strategy degree) (initial : K)
    (challenges : List K) (certificate : FixedPhase.Certificate K degree)
    (execution : issued strategy [] challenges = some certificate.rounds)
    (representable : FixedPhase.ExpectedRoundsRepresentable ops q degree challenges)
    (accepted : FixedPhase.Accepted ops q initial challenges certificate)
    (falseClaim : initial ≠ FixedPhase.semanticInitial ops q challenges.length) :
    Collision q strategy [] challenges :=
  badChallenge_implies_collision q strategy initial challenges certificate execution
    (FixedPhase.false_acceptance_implies_bad_challenge ops q challengeSetSize
      initial challenges certificate representable accepted falseClaim)

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

/-- Bound the existing bad-challenge event on the actual issued path. -/
theorem badChallengeProbability_le {degree totalRounds : Nat}
    (q : List K → K) (strategy : Strategy degree) (initial : K) (challengeSetSize : Nat)
    (representable : FixedPhase.Sequential.RoundRepresentable ops q degree totalRounds) :
    uniformAverage GoldilocksRoots.fullChallengeSet totalRounds (fun challenges =>
      if BadChallengeEvent q strategy initial challengeSetSize challenges then (1 : ℝ) else 0) ≤
        (totalRounds : ℝ) * degree / (goldilocksModulus ^ 2 : Nat) := by
  have inclusion : uniformAverage GoldilocksRoots.fullChallengeSet totalRounds (fun challenges =>
      if BadChallengeEvent q strategy initial challengeSetSize challenges then (1 : ℝ) else 0) ≤
      collisionProbability GoldilocksRoots.fullChallengeSet q strategy [] totalRounds := by
    apply uniformAverage_mono
    intro challenges _length
    by_cases bad : BadChallengeEvent q strategy initial challengeSetSize challenges
    · have event := bad
      obtain ⟨certificate, execution, badRound⟩ := bad
      have collision := badChallenge_implies_collision q strategy initial challenges certificate
        execution badRound
      simp only [if_pos event, if_pos collision, le_refl]
    · rw [if_neg bad]
      split_ifs <;> norm_num
  exact inclusion.trans (collisionProbability_le q strategy representable [] totalRounds (by simp))

/-- Causal false acceptance has the actual-field degree-times-rounds bound.
The probability includes aborted runs; it is not conditioned on acceptance. -/
theorem falseAcceptanceProbability_le {degree totalRounds : Nat}
    (q : List K → K) (strategy : Strategy degree) (initial : K)
    (representable : FixedPhase.Sequential.RoundRepresentable ops q degree totalRounds) :
    uniformAverage GoldilocksRoots.fullChallengeSet totalRounds (fun challenges =>
      if FalseAcceptance q strategy initial challenges then (1 : ℝ) else 0) ≤
        (totalRounds : ℝ) * degree / (goldilocksModulus ^ 2 : Nat) := by
  have inclusion : uniformAverage GoldilocksRoots.fullChallengeSet totalRounds (fun challenges =>
      if FalseAcceptance q strategy initial challenges then (1 : ℝ) else 0) ≤
      collisionProbability GoldilocksRoots.fullChallengeSet q strategy [] totalRounds := by
    apply uniformAverage_mono
    intro challenges length
    by_cases bad : FalseAcceptance q strategy initial challenges
    · have event := bad
      obtain ⟨certificate, execution, accepted, falseClaim⟩ := bad
      have expectedRepresentable : FixedPhase.ExpectedRoundsRepresentable ops q degree challenges := by
        intro expected member
        exact representable_expectedFrom q representable [] challenges (by simpa using length)
          expected member
      have collision := false_acceptance_implies_collision
        (challengeSetSize := goldilocksModulus ^ 2) q strategy initial challenges certificate
        execution expectedRepresentable accepted falseClaim
      simp only [if_pos event, if_pos collision, le_refl]
    · rw [if_neg bad]
      split_ifs <;> norm_num
  exact inclusion.trans (collisionProbability_le q strategy representable [] totalRounds (by simp))

end NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace
