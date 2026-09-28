
/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/SumCheck.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespace renamed, otherwise unchanged. -/
/-!
Semantic SumCheck assurance used by the SuperNeo folding reductions.

The verifier follows only the prover's claimed polynomial chain. Truth follows a
separate expected chain whose initial target is the actual hypercube sum. The
main theorem therefore does not assume that acceptance implies truth: a false
accepted claim exposes a round polynomial that differs from the expected one
but collides with it at the verifier's sampled challenge.

This file proves the deterministic reduction. Turning the bounded-degree
collision into the paper's statistical `rounds * degree / |challengeSet|`
error is deliberately an explicit root-counting/sampling boundary, not a local
soundness assumption smuggled into the verifier model.
-/

namespace NightstreamFPrime.Spec.SumCheck

universe uChallenge uValue

/-- The two Boolean-cube points and addition needed by the round verifier. -/
structure Ops (Challenge : Type uChallenge) (Value : Type uValue) where
  zero : Challenge
  one : Challenge
  add : Value → Value → Value

/-- One prover polynomial, its semantic counterpart, and the public challenge. -/
structure Round (Challenge : Type uChallenge) (Value : Type uValue) where
  claimed : Challenge → Value
  expected : Challenge → Value
  challenge : Challenge
  degree : Nat

/-- A complete SumCheck claim and transcript at the semantic model boundary. -/
structure Instance (Challenge : Type uChallenge) (Value : Type uValue) where
  claimedInitial : Value
  /-- The actual sum of the target multilinear expression over its Boolean cube. -/
  trueInitial : Value
  /-- The actual terminal evaluation after every verifier challenge is fixed. -/
  terminal : Value
  rounds : List (Round Challenge Value)
  maxDegree : Nat
  /-- Cardinality of the verifier-owned strong challenge set. -/
  challengeSetSize : Nat

/-- Follow either the prover polynomials or the semantic polynomials round by round. -/
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

/-- What the executable verifier checks: claimed sums, challenge forwarding, and terminal equality. -/
def Accepted
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (ops : Ops Challenge Value)
    (transcript : Instance Challenge Value) : Prop :=
  (∀ round ∈ transcript.rounds, round.degree ≤ transcript.maxDegree) ∧
  Chain ops (fun round => round.claimed)
    transcript.claimedInitial transcript.rounds transcript.terminal

/-- Executable form of the claimed-chain verifier. -/
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

/-- Executable SumCheck verifier: only public degree and claimed-chain checks. -/
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

/-- The executable verifier accepts exactly the logical `Accepted` predicate. -/
theorem check_eq_true_iff_accepted
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    [DecidableEq Value]
    (ops : Ops Challenge Value)
    (transcript : Instance Challenge Value) :
    check ops transcript = true ↔ Accepted ops transcript := by
  simp [check, Accepted, checkChain_eq_true_iff]

/-- The independent semantic path beginning at the real hypercube sum. -/
def TruthPath
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (ops : Ops Challenge Value)
    (transcript : Instance Challenge Value) : Prop :=
  Chain ops (fun round => round.expected)
    transcript.trueInitial transcript.rounds transcript.terminal

namespace Claim

/-- Claim truth is equality with the actual sum, not verifier acceptance. -/
def True
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (transcript : Instance Challenge Value) : Prop :=
  transcript.claimedInitial = transcript.trueInitial

end Claim

/-- An honest transcript uses the semantic polynomial at every round. -/
def Honest
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (transcript : Instance Challenge Value) : Prop :=
  Claim.True transcript ∧
  (∀ round ∈ transcript.rounds, round.claimed = round.expected) ∧
  ∀ round ∈ transcript.rounds, round.degree ≤ transcript.maxDegree

/-- The exact bad event exposed by a false accepted SumCheck claim. -/
def BadChallenge
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (transcript : Instance Challenge Value)
    (round : Round Challenge Value) : Prop :=
  round ∈ transcript.rounds ∧
  round.degree ≤ transcript.maxDegree ∧
  round.claimed ≠ round.expected ∧
  round.claimed round.challenge = round.expected round.challenge

/-- Numerator in the standard Lund/Schwartz--Zippel union bound. -/
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

/-- Perfect completeness of the semantic SumCheck verifier. -/
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

/--
SumCheck soundness reduction: an accepted false claim cannot disappear into an
`accepted → valid` premise. It yields a concrete bounded-degree round where a
different prover polynomial collides with the semantic polynomial at the
verifier's challenge.
-/
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
