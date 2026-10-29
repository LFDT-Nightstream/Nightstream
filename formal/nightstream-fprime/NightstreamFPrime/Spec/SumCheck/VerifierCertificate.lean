import NightstreamFPrime.Spec.SumCheck.Abstract
import NightstreamFPrime.Spec.SumCheck.Polynomial

namespace NightstreamFPrime.Spec.SumCheck.Finite

universe uField

namespace Ops

def toSymbolic
    {Field : Type uField}
    (ops : Ops Field) : SumCheck.Ops Field Field where
  zero := ops.zero
  one := ops.one
  add := ops.add

end Ops

structure Certificate (Field : Type uField) where
  rounds : List (Message Field)

def Chain
    {Field : Type uField}
    (ops : Ops Field)
    (maxDegree : Nat) :
    Field -> List (Message Field) -> List Field -> Field -> Prop
  | current, [], [], terminal => current = terminal
  | current, message :: messages, challenge :: challenges, terminal =>
      message.Canonical ops ∧
      message.degreeUpperBound ≤ maxDegree ∧
      current = ops.add
        (message.evaluate ops ops.zero)
        (message.evaluate ops ops.one) ∧
      Chain ops maxDegree
        (message.evaluate ops challenge) messages challenges terminal
  | _, _, _, _ => False

def Accepted
    {Field : Type uField}
    (ops : Ops Field)
    (maxDegree : Nat)
    (initial : Field)
    (challenges : List Field)
    (terminal : Field)
    (certificate : Certificate Field) : Prop :=
  Chain ops maxDegree initial certificate.rounds challenges terminal

theorem Chain.messages_length_eq_challenges_length
    {Field : Type uField}
    (ops : Ops Field)
    (maxDegree : Nat)
    (current terminal : Field)
    (messages : List (Message Field))
    (challenges : List Field)
    (chain : Chain ops maxDegree current messages challenges terminal) :
    messages.length = challenges.length := by
  induction messages generalizing current challenges with
  | nil =>
      cases challenges <;> simp [Chain] at chain ⊢
  | cons message messages inductionHypothesis =>
      cases challenges with
      | nil => simp [Chain] at chain
      | cons challenge challenges =>
          simp only [Chain] at chain
          simp only [List.length_cons, Nat.succ.injEq]
          exact inductionHypothesis
            (current := message.evaluate ops challenge)
            (challenges := challenges) chain.2.2.2

def checkChain
    {Field : Type uField}
    [DecidableEq Field]
    (ops : Ops Field)
    (maxDegree : Nat) :
    Field -> List (Message Field) -> List Field -> Field -> Bool
  | current, [], [], terminal => decide (current = terminal)
  | current, message :: messages, challenge :: challenges, terminal =>
      message.canonicalCheck ops &&
      decide (message.degreeUpperBound ≤ maxDegree) &&
      decide (current = ops.add
        (message.evaluate ops ops.zero)
        (message.evaluate ops ops.one)) &&
      checkChain ops maxDegree
        (message.evaluate ops challenge) messages challenges terminal
  | _, _, _, _ => false

def check
    {Field : Type uField}
    [DecidableEq Field]
    (ops : Ops Field)
    (maxDegree : Nat)
    (initial : Field)
    (challenges : List Field)
    (terminal : Field)
    (certificate : Certificate Field) : Bool :=
  checkChain ops maxDegree initial certificate.rounds challenges terminal

theorem checkChain_eq_true_iff
    {Field : Type uField}
    [DecidableEq Field]
    (ops : Ops Field)
    (maxDegree : Nat)
    (current terminal : Field)
    (messages : List (Message Field))
    (challenges : List Field) :
    checkChain ops maxDegree current messages challenges terminal = true ↔
      Chain ops maxDegree current messages challenges terminal := by
  induction messages generalizing current challenges with
  | nil =>
      cases challenges <;> simp [checkChain, Chain]
  | cons message messages inductionHypothesis =>
      cases challenges with
      | nil =>
          simp [checkChain, Chain]
      | cons challenge challenges =>
          simp [checkChain, Chain, inductionHypothesis, and_assoc]

theorem check_eq_true_iff_accepted
    {Field : Type uField}
    [DecidableEq Field]
    (ops : Ops Field)
    (maxDegree : Nat)
    (initial : Field)
    (challenges : List Field)
    (terminal : Field)
    (certificate : Certificate Field) :
    check ops maxDegree initial challenges terminal certificate = true ↔
      Accepted ops maxDegree initial challenges terminal certificate := by
  exact checkChain_eq_true_iff ops maxDegree initial terminal
    certificate.rounds challenges

theorem complete_of_canonical_chain
    {Field : Type uField}
    [DecidableEq Field]
    (ops : Ops Field)
    (maxDegree : Nat)
    (initial : Field)
    (challenges : List Field)
    (terminal : Field)
    (certificate : Certificate Field)
    (chain : Accepted ops maxDegree initial challenges terminal certificate) :
    check ops maxDegree initial challenges terminal certificate = true :=
  (check_eq_true_iff_accepted ops maxDegree initial challenges terminal
    certificate).2 chain

structure SemanticGhosts (Field : Type uField) where
  trueInitial : Field
  expected : List (Field -> Field)

private def toSymbolicRoundsFrom
    {Field : Type uField}
    (ops : Ops Field) :
    List (Field -> Field) -> List (Message Field) -> List Field ->
      List (SumCheck.Round Field Field)
  | [], [], [] => []
  | expected :: expecteds, message :: messages, challenge :: challenges =>
      {
        claimed := message.evaluate ops
        expected := expected
        challenge := challenge
        degree := message.degreeUpperBound
      } :: toSymbolicRoundsFrom ops expecteds messages challenges
  | _, _, _ => []

def toSymbolicInstance
    {Field : Type uField}
    (ops : Ops Field)
    (maxDegree challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (terminal : Field)
    (certificate : Certificate Field)
    (ghosts : SemanticGhosts Field) : SumCheck.Instance Field Field where
  claimedInitial := initial
  trueInitial := ghosts.trueInitial
  terminal := terminal
  rounds := toSymbolicRoundsFrom ops ghosts.expected
    certificate.rounds challenges
  maxDegree := maxDegree
  challengeSetSize := challengeSetSize

def ExpectedChain
    {Field : Type uField}
    (ops : Ops Field) :
    Field -> List (Field -> Field) -> List (Message Field) -> List Field ->
      Field -> Prop
  | current, [], [], [], terminal => current = terminal
  | current, expected :: expecteds, _ :: messages,
      challenge :: challenges, terminal =>
      current = ops.add (expected ops.zero) (expected ops.one) ∧
      ExpectedChain ops (expected challenge) expecteds messages challenges
        terminal
  | _, _, _, _, _ => False

def SemanticGhosts.Honest
    {Field : Type uField}
    (ghosts : SemanticGhosts Field)
    (ops : Ops Field)
    (_maxDegree _challengeSetSize : Nat)
    (_initial : Field)
    (challenges : List Field)
    (terminal : Field)
    (certificate : Certificate Field) : Prop :=
  ExpectedChain ops ghosts.trueInitial ghosts.expected certificate.rounds
    challenges terminal

private theorem symbolicDegrees_of_chains
    {Field : Type uField}
    (ops : Ops Field)
    (maxDegree : Nat)
    (claimedCurrent trueCurrent terminal : Field)
    (expected : List (Field -> Field))
    (messages : List (Message Field))
    (challenges : List Field)
    (claimedChain :
      Chain ops maxDegree claimedCurrent messages challenges terminal)
    (expectedChain :
      ExpectedChain ops trueCurrent expected messages challenges terminal) :
    ∀ round ∈ toSymbolicRoundsFrom ops expected messages challenges,
      round.degree ≤ maxDegree := by
  induction messages generalizing claimedCurrent trueCurrent expected challenges with
  | nil =>
      cases expected <;> cases challenges <;>
        simp [ExpectedChain, toSymbolicRoundsFrom] at expectedChain ⊢
  | cons message messages inductionHypothesis =>
      cases expected with
      | nil => simp [ExpectedChain] at expectedChain
      | cons expected expecteds =>
          cases challenges with
          | nil => simp [ExpectedChain] at expectedChain
          | cons challenge challenges =>
              simp only [Chain] at claimedChain
              rcases claimedChain with ⟨_, degree, _, claimedTail⟩
              simp only [ExpectedChain] at expectedChain
              rcases expectedChain with ⟨_, expectedTail⟩
              intro round roundIn
              simp only [toSymbolicRoundsFrom, List.mem_cons] at roundIn
              rcases roundIn with rfl | roundIn
              · exact degree
              · exact inductionHypothesis
                  (claimedCurrent := message.evaluate ops challenge)
                  (trueCurrent := expected challenge)
                  (expected := expecteds)
                  (challenges := challenges)
                  claimedTail expectedTail round roundIn

private theorem symbolicClaimedChain_of_chains
    {Field : Type uField}
    (ops : Ops Field)
    (maxDegree : Nat)
    (claimedCurrent trueCurrent terminal : Field)
    (expected : List (Field -> Field))
    (messages : List (Message Field))
    (challenges : List Field)
    (claimedChain :
      Chain ops maxDegree claimedCurrent messages challenges terminal)
    (expectedChain :
      ExpectedChain ops trueCurrent expected messages challenges terminal) :
    SumCheck.Chain ops.toSymbolic (fun round => round.claimed) claimedCurrent
      (toSymbolicRoundsFrom ops expected messages challenges)
      terminal := by
  induction messages generalizing claimedCurrent trueCurrent expected challenges with
  | nil =>
      cases expected <;> cases challenges
      · simpa [Chain, toSymbolicRoundsFrom, SumCheck.Chain] using claimedChain
      all_goals simp [ExpectedChain] at expectedChain
  | cons message messages inductionHypothesis =>
      cases expected with
      | nil => simp [ExpectedChain] at expectedChain
      | cons expected expecteds =>
          cases challenges with
          | nil => simp [ExpectedChain] at expectedChain
          | cons challenge challenges =>
              simp only [Chain] at claimedChain
              rcases claimedChain with ⟨_, _, sum, claimedTail⟩
              simp only [ExpectedChain] at expectedChain
              rcases expectedChain with ⟨_, expectedTail⟩
              simp only [toSymbolicRoundsFrom, SumCheck.Chain]
              constructor
              · exact sum
              · exact inductionHypothesis
                  (claimedCurrent := message.evaluate ops challenge)
                  (trueCurrent := expected challenge)
                  (expected := expecteds)
                  (challenges := challenges)
                  claimedTail expectedTail

private theorem symbolicTruthPath_of_expectedChain
    {Field : Type uField}
    (ops : Ops Field)
    (trueCurrent terminal : Field)
    (expected : List (Field -> Field))
    (messages : List (Message Field))
    (challenges : List Field)
    (expectedChain :
      ExpectedChain ops trueCurrent expected messages challenges terminal) :
    SumCheck.Chain ops.toSymbolic (fun round => round.expected) trueCurrent
      (toSymbolicRoundsFrom ops expected messages challenges) terminal := by
  induction messages generalizing trueCurrent expected challenges with
  | nil =>
      cases expected <;> cases challenges
      · simpa [ExpectedChain, toSymbolicRoundsFrom, SumCheck.Chain] using
          expectedChain
      all_goals simp [ExpectedChain] at expectedChain
  | cons message messages inductionHypothesis =>
      cases expected with
      | nil => simp [ExpectedChain] at expectedChain
      | cons expected expecteds =>
          cases challenges with
          | nil => simp [ExpectedChain] at expectedChain
          | cons challenge challenges =>
              simp only [ExpectedChain] at expectedChain
              rcases expectedChain with ⟨sum, tail⟩
              simp only [toSymbolicRoundsFrom, SumCheck.Chain]
              exact ⟨sum, inductionHypothesis
                (trueCurrent := expected challenge)
                (expected := expecteds)
                (challenges := challenges) tail⟩

theorem accepted_implies_symbolicAccepted_and_truthPath
    {Field : Type uField}
    (ops : Ops Field)
    (maxDegree challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (terminal : Field)
    (certificate : Certificate Field)
    (ghosts : SemanticGhosts Field)
    (accepted : Accepted ops maxDegree initial challenges terminal certificate)
    (honestGhosts : ghosts.Honest ops maxDegree challengeSetSize initial
      challenges terminal certificate) :
    SumCheck.Accepted ops.toSymbolic
        (toSymbolicInstance ops maxDegree challengeSetSize initial challenges
          terminal certificate ghosts) ∧
      SumCheck.TruthPath ops.toSymbolic
        (toSymbolicInstance ops maxDegree challengeSetSize initial challenges
          terminal certificate ghosts) := by
  constructor
  · constructor
    · exact symbolicDegrees_of_chains ops maxDegree initial ghosts.trueInitial
        terminal ghosts.expected certificate.rounds challenges accepted
        honestGhosts
    · exact symbolicClaimedChain_of_chains ops maxDegree initial
        ghosts.trueInitial terminal ghosts.expected certificate.rounds
        challenges accepted honestGhosts
  · exact symbolicTruthPath_of_expectedChain ops ghosts.trueInitial terminal
      ghosts.expected certificate.rounds challenges honestGhosts

end NightstreamFPrime.Spec.SumCheck.Finite
