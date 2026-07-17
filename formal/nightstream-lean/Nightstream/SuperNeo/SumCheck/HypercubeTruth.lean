import Nightstream.SuperNeo.SumCheck.VerifierCertificate

/-!
Canonical finite SumCheck truth path for an explicit hypercube polynomial.

Protocol: generic SumCheck.
Phase: semantic expected-round and terminal construction.
Constraint family: none; this is verifier-independent truth semantics.

Owns: recursive Boolean completion sums, the exact finite expected polynomial
at each challenge prefix, the terminal full-point evaluation, and construction
of `SemanticGhosts.Honest` from round/challenge shape equality.

Does not own: a protocol-specific polynomial, polynomial degree bounds,
verifier messages, challenge sampling, root counting, Fiat--Shamir, Rust,
R1CS, or counts.

Emits constraints: no.

Authority boundary: the caller supplies one explicit total polynomial on
coordinate lists. Expected rounds and the terminal are derived structurally
from that polynomial and verifier challenges; no prover-supplied expected
function or terminal identity enters the construction.

| SumCheck phase | Derived object | Exact mathematical meaning |
|---|---|---|
| initial | `sumCompletions q [] n` | sum of `q` over all `n` Boolean coordinates |
| product split | `sumCompletions_add` | split one cube into an explicit prefix cube followed by a suffix cube |
| round | `expectedPolynomialsFrom` | fix prior challenges, expose one variable, sum remaining Boolean suffixes |
| terminal | `q challenges` | evaluate the same explicit polynomial at the full challenge vector |
| truth path | `semanticGhosts_honest` | every expected sum/forwarding/terminal equation holds |
-/

namespace Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth

universe uField

/-- Sum an explicit polynomial over every Boolean completion of `prefix`.
The recursion order is coordinate-first, zero branch then one branch. -/
def sumCompletions
    {Field : Type uField}
    (ops : Ops Field)
    (polynomial : List Field -> Field) : List Field -> Nat -> Field
  | fixed, 0 => polynomial fixed
  | fixed, remaining + 1 =>
      ops.add
        (sumCompletions ops polynomial (fixed ++ [ops.zero]) remaining)
        (sumCompletions ops polynomial (fixed ++ [ops.one]) remaining)

/-- Split a Boolean-completion cube into an outer prefix domain followed by an
inner suffix domain. This is structural: both sides enumerate the same branch
tree in the same order, so no field or commutativity laws are required. -/
theorem sumCompletions_add
    {Field : Type uField}
    (ops : Ops Field)
    (polynomial : List Field -> Field)
    (fixed : List Field)
    (prefixVariables suffixVariables : Nat) :
    sumCompletions ops polynomial fixed
        (prefixVariables + suffixVariables) =
      sumCompletions ops
        (fun extended =>
          sumCompletions ops polynomial extended suffixVariables)
        fixed prefixVariables := by
  induction prefixVariables generalizing fixed with
  | zero => simp [sumCompletions]
  | succ prefixVariables inductionHypothesis =>
      simp only [Nat.succ_add, sumCompletions]
      rw [inductionHypothesis, inductionHypothesis]

/-- Exact expected round polynomials after fixing the preceding verifier
challenges. The list has one semantic polynomial per challenge. -/
def expectedPolynomialsFrom
    {Field : Type uField}
    (ops : Ops Field)
    (polynomial : List Field -> Field) :
    List Field -> List Field -> List (Field -> Field)
  | _, [] => []
  | fixed, challenge :: challenges =>
      (fun value =>
        sumCompletions ops polynomial (fixed ++ [value]) challenges.length) ::
      expectedPolynomialsFrom ops polynomial (fixed ++ [challenge]) challenges

/-- Expected polynomials from the empty challenge prefix. -/
def expectedPolynomials
    {Field : Type uField}
    (ops : Ops Field)
    (polynomial : List Field -> Field)
    (challenges : List Field) : List (Field -> Field) :=
  expectedPolynomialsFrom ops polynomial [] challenges

theorem expectedPolynomialsFrom_length
    {Field : Type uField}
    (ops : Ops Field)
    (polynomial : List Field -> Field)
    (fixed challenges : List Field) :
    (expectedPolynomialsFrom ops polynomial fixed challenges).length =
      challenges.length := by
  induction challenges generalizing fixed with
  | nil => rfl
  | cons challenge challenges inductionHypothesis =>
      simp [expectedPolynomialsFrom, inductionHypothesis]

private theorem expectedChainFrom
    {Field : Type uField}
    (ops : Ops Field)
    (polynomial : List Field -> Field)
    (fixed challenges : List Field)
    (messages : List (Message Field))
    (sameLength : messages.length = challenges.length) :
    ExpectedChain ops
      (sumCompletions ops polynomial fixed challenges.length)
      (expectedPolynomialsFrom ops polynomial fixed challenges)
      messages challenges (polynomial (fixed ++ challenges)) := by
  induction challenges generalizing fixed messages with
  | nil =>
      have messagesEmpty : messages = [] :=
        List.eq_nil_of_length_eq_zero (by simpa using sameLength)
      subst messages
      simp [sumCompletions, expectedPolynomialsFrom, ExpectedChain]
  | cons challenge challenges inductionHypothesis =>
      cases messages with
      | nil => simp at sameLength
      | cons message messages =>
          have tailLength : messages.length = challenges.length := by
            simpa using sameLength
          simp only [expectedPolynomialsFrom, sumCompletions, ExpectedChain]
          constructor
          · trivial
          · simpa [List.append_assoc] using
              inductionHypothesis (fixed := fixed ++ [challenge])
                (messages := messages) tailLength

/-- Canonical finite semantic ghosts for one explicit polynomial and challenge
vector. -/
def semanticGhosts
    {Field : Type uField}
    (ops : Ops Field)
    (polynomial : List Field -> Field)
    (challenges : List Field) : SemanticGhosts Field where
  trueInitial := sumCompletions ops polynomial [] challenges.length
  expected := expectedPolynomials ops polynomial challenges

/-- The structurally derived expected rounds form the exact semantic truth
path whenever the finite certificate has one message per challenge. -/
theorem semanticGhosts_honest
    {Field : Type uField}
    (ops : Ops Field)
    (polynomial : List Field -> Field)
    (maxDegree challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field)
    (sameLength : certificate.rounds.length = challenges.length) :
    (semanticGhosts ops polynomial challenges).Honest ops maxDegree
      challengeSetSize initial challenges (polynomial challenges)
      certificate := by
  simpa [semanticGhosts, expectedPolynomials] using
    expectedChainFrom ops polynomial [] challenges certificate.rounds sameLength

end Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth
