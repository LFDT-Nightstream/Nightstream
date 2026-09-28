import NightstreamFPrime.Spec.SumCheck.Abstract
import NightstreamFPrime.Spec.SumCheck.Polynomial

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/SumCheck/VerifierCertificate.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespace renamed, otherwise unchanged. -/

/-!
Finite verifier-visible SumCheck certificates and their symbolic projection.

Owns: the raw round-message product, exact claimed-chain relation, executable
checker, checker equivalence, completeness from a canonical finite chain, and
a one-way bridge into the existing symbolic `SumCheck.Accepted` predicate.

Does not own: challenges, initial claims, terminal identities, maximum degree,
semantic truth polynomials, or challenge-set size as certificate data. It also
does not own root counting, Fiat--Shamir, PiCCS arithmetization, or any Rust or
R1CS representation.

Emits constraints: no.

Authority boundary: all verifier-owned values are explicit arguments to
`check`; `Certificate` contains only finite polynomial messages. The symbolic
bridge requires the semantic `trueInitial` and expected-polynomial chain in a
separate `SemanticGhosts` object whose truth path is independently supplied.

| Protocol phase | Mathematical obligation | Runtime owner | Semantic owner |
|---|---|---|---|
| every round | canonical coefficients and derived degree bound | `checkChain` | `Chain` |
| every round | `current = p(0) + p(1)` | `checkChain` | `Chain` |
| verifier response | next target is `p(challenge)` | verifier argument | `Chain` |
| terminal | final target equals verifier-computed terminal | verifier argument | `Chain` |
| symbolic projection | reconstruct `SumCheck.Round` without certificate ghosts | `toSymbolicInstance` | `SemanticGhosts.Honest` |
-/

namespace NightstreamFPrime.Spec.SumCheck.Finite

universe uField

namespace Ops

/-- Forget multiplication while retaining the operations used by the existing
symbolic SumCheck verifier. -/
def toSymbolic
    {Field : Type uField}
    (ops : Ops Field) : SumCheck.Ops Field Field where
  zero := ops.zero
  one := ops.one
  add := ops.add

end Ops

/-- Raw verifier-visible certificate. It carries no challenges, degree claims,
semantic polynomials, initial value, terminal value, or verifier parameters. -/
structure Certificate (Field : Type uField) where
  rounds : List (Message Field)

/-- Exact finite claimed-chain relation over parallel message/challenge lists. -/
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

/-- Acceptance under verifier-owned initial, challenge, terminal, and degree
parameters. -/
def Accepted
    {Field : Type uField}
    (ops : Ops Field)
    (maxDegree : Nat)
    (initial : Field)
    (challenges : List Field)
    (terminal : Field)
    (certificate : Certificate Field) : Prop :=
  Chain ops maxDegree initial certificate.rounds challenges terminal

/-- Logical acceptance consumes exactly one verifier challenge per finite
round message. -/
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

/-- Executable finite claimed-chain checker. -/
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

/-- Executable certificate verifier. -/
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

/-- Exact executable/logical correspondence for the finite claimed chain. -/
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

/-- The executable verifier accepts exactly the finite logical relation. -/
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

/-- Completeness of the executable verifier from one canonical finite chain. -/
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

/-! ## Projection to the existing symbolic model -/

/-- Semantic data deliberately absent from the verifier-visible certificate.
Expected polynomials are a finite list so their shape must align exactly with
the verifier-visible message and challenge lists. -/
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

/-- Reconstruct the symbolic transcript. All verifier-owned values are
arguments; only `trueInitial` and expected polynomials come from semantic
ghost data. -/
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

/-- Exact finite expected-polynomial chain. All three round-indexed lists must
be consumed in lockstep; mismatched shapes are false rather than truncated. -/
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

/-- Honest semantic ghosts are exactly the independently expected finite
chain. This is not verifier-visible certificate data. Verifier parameters and
the claimed initial are arguments only to keep the conformance surface aligned
with `toSymbolicInstance`; they do not define semantic truth. -/
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

/-- A finite accepted chain plus a separately honest semantic truth path
projects to the existing symbolic acceptance and truth predicates.

This theorem does not construct the ghosts, prove PiCCS terminal identities,
or establish a root-counting bound. -/
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
