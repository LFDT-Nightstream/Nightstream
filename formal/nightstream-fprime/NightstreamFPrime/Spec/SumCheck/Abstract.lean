
/-!
Semantic SumCheck assurance used by the SuperNeo folding reductions.
-/

namespace NightstreamFPrime.Spec.SumCheck

universe uChallenge uValue

structure Ops (Challenge : Type uChallenge) (Value : Type uValue) where
  zero : Challenge
  one : Challenge
  add : Value → Value → Value

structure Round (Challenge : Type uChallenge) (Value : Type uValue) where
  claimed : Challenge → Value
  expected : Challenge → Value
  challenge : Challenge
  degree : Nat

structure Instance (Challenge : Type uChallenge) (Value : Type uValue) where
  claimedInitial : Value
  trueInitial : Value
  terminal : Value
  rounds : List (Round Challenge Value)
  maxDegree : Nat
  challengeSetSize : Nat

def Chain
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (ops : Ops Challenge Value)
    (polynomial : Round Challenge Value → Challenge → Value) :
    Value → List (Round Challenge Value) → Value → Prop
  | target, [], terminal => target = terminal
  | target, round :: rest, terminal =>
      target = ops.add (polynomial round ops.zero) (polynomial round ops.one) ∧
      Chain ops polynomial (polynomial round round.challenge) rest terminal

def Accepted
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (ops : Ops Challenge Value)
    (transcript : Instance Challenge Value) : Prop :=
  (∀ round ∈ transcript.rounds, round.degree ≤ transcript.maxDegree) ∧
  Chain ops (fun round => round.claimed)
    transcript.claimedInitial transcript.rounds transcript.terminal

def checkChain
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    [DecidableEq Value]
    (ops : Ops Challenge Value) :
    Value → List (Round Challenge Value) → Value → Bool
  | target, [], terminal => decide (target = terminal)
  | target, round :: rest, terminal =>
      decide (target = ops.add (round.claimed ops.zero) (round.claimed ops.one)) &&
      checkChain ops (round.claimed round.challenge) rest terminal

def check
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    [DecidableEq Value]
    (ops : Ops Challenge Value)
    (transcript : Instance Challenge Value) : Bool :=
  transcript.rounds.all (fun round => decide (round.degree ≤ transcript.maxDegree)) &&
  checkChain ops transcript.claimedInitial transcript.rounds transcript.terminal

private theorem checkChain_eq_true_iff
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    [DecidableEq Value]
    (ops : Ops Challenge Value)
    (target terminal : Value)
    (rounds : List (Round Challenge Value)) :
    checkChain ops target rounds terminal = true ↔
      Chain ops (fun round => round.claimed) target rounds terminal := by
  induction rounds generalizing target with
  | nil => simp [checkChain, Chain]
  | cons head tail inductionHypothesis =>
      simp [checkChain, Chain, inductionHypothesis]

theorem check_eq_true_iff_accepted
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    [DecidableEq Value]
    (ops : Ops Challenge Value)
    (transcript : Instance Challenge Value) :
    check ops transcript = true ↔ Accepted ops transcript := by
  simp [check, Accepted, checkChain_eq_true_iff]

def TruthPath
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (ops : Ops Challenge Value)
    (transcript : Instance Challenge Value) : Prop :=
  Chain ops (fun round => round.expected)
    transcript.trueInitial transcript.rounds transcript.terminal

namespace Claim

def True
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (transcript : Instance Challenge Value) : Prop :=
  transcript.claimedInitial = transcript.trueInitial

end Claim

def Honest
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (transcript : Instance Challenge Value) : Prop :=
  Claim.True transcript ∧
  (∀ round ∈ transcript.rounds, round.claimed = round.expected) ∧
  ∀ round ∈ transcript.rounds, round.degree ≤ transcript.maxDegree

def BadChallenge
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (transcript : Instance Challenge Value)
    (round : Round Challenge Value) : Prop :=
  round ∈ transcript.rounds ∧
  round.degree ≤ transcript.maxDegree ∧
  round.claimed ≠ round.expected ∧
  round.claimed round.challenge = round.expected round.challenge

def errorNumerator
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (transcript : Instance Challenge Value) : Nat :=
  transcript.rounds.length * transcript.maxDegree

private theorem chain_of_round_agreement
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (ops : Ops Challenge Value)
    (rounds : List (Round Challenge Value))
    (target terminal : Value)
    (agreement : ∀ round ∈ rounds, round.claimed = round.expected)
    (truth : Chain ops (fun round => round.expected) target rounds terminal) :
    Chain ops (fun round => round.claimed) target rounds terminal := by
  induction rounds generalizing target with
  | nil =>
      exact truth
  | cons head tail inductionHypothesis =>
      simp only [Chain] at truth ⊢
      rcases truth with ⟨headSum, tailTruth⟩
      have headAgreement : head.claimed = head.expected :=
        agreement head (by simp)
      constructor
      · simpa only [headAgreement] using headSum
      · rw [headAgreement]
        exact inductionHypothesis _
          (fun round roundInTail => agreement round (by simp [roundInTail]))
          tailTruth

theorem complete
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (ops : Ops Challenge Value)
    (transcript : Instance Challenge Value)
    (truth : TruthPath ops transcript)
    (honest : Honest transcript) :
    Accepted ops transcript := by
  rcases honest with ⟨claimTrue, roundAgreement, degreeBound⟩
  constructor
  · exact degreeBound
  · rw [Claim.True] at claimTrue
    rw [claimTrue]
    exact chain_of_round_agreement ops transcript.rounds transcript.trueInitial
      transcript.terminal roundAgreement truth

private theorem chain_disagreement_implies_collision
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (ops : Ops Challenge Value)
    (rounds : List (Round Challenge Value))
    (claimedTarget trueTarget terminal : Value)
    (claimedPath :
      Chain ops (fun round => round.claimed) claimedTarget rounds terminal)
    (truePath :
      Chain ops (fun round => round.expected) trueTarget rounds terminal)
    (differentTargets : claimedTarget ≠ trueTarget) :
    ∃ round ∈ rounds,
      round.claimed ≠ round.expected ∧
      round.claimed round.challenge = round.expected round.challenge := by
  induction rounds generalizing claimedTarget trueTarget with
  | nil =>
      simp only [Chain] at claimedPath truePath
      exact False.elim (differentTargets (claimedPath.trans truePath.symm))
  | cons head tail inductionHypothesis =>
      simp only [Chain] at claimedPath truePath
      rcases claimedPath with ⟨claimedSum, claimedTail⟩
      rcases truePath with ⟨trueSum, trueTail⟩
      have differentPolynomials : head.claimed ≠ head.expected := by
        intro samePolynomial
        apply differentTargets
        calc
          claimedTarget =
              ops.add (head.claimed ops.zero) (head.claimed ops.one) := claimedSum
          _ = ops.add (head.expected ops.zero) (head.expected ops.one) := by
              rw [samePolynomial]
          _ = trueTarget := trueSum.symm
      by_cases collision :
          head.claimed head.challenge = head.expected head.challenge
      · exact ⟨head, by simp, differentPolynomials, collision⟩
      · rcases inductionHypothesis
          (head.claimed head.challenge)
          (head.expected head.challenge)
          claimedTail trueTail collision with
          ⟨round, roundInTail, differs, agrees⟩
        exact ⟨round, by simp [roundInTail], differs, agrees⟩

theorem false_acceptance_implies_bad_challenge
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (ops : Ops Challenge Value)
    (transcript : Instance Challenge Value)
    (accepted : Accepted ops transcript)
    (truth : TruthPath ops transcript)
    (falseClaim : ¬ Claim.True transcript) :
    ∃ round, BadChallenge transcript round := by
  rcases accepted with ⟨degreeBound, claimedPath⟩
  rcases chain_disagreement_implies_collision ops transcript.rounds
      transcript.claimedInitial transcript.trueInitial transcript.terminal
      claimedPath truth falseClaim with
    ⟨round, roundInTranscript, differs, agrees⟩
  exact ⟨round, roundInTranscript, degreeBound round roundInTranscript,
    differs, agrees⟩

end NightstreamFPrime.Spec.SumCheck
