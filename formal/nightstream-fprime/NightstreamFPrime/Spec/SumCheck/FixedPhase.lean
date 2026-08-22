import NightstreamFPrime.Spec.SumCheck.FixedPolynomial
import NightstreamFPrime.Spec.SumCheck.HypercubeTruth

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/SumCheck/FixedPhase.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespace renamed, otherwise unchanged. -/

/-!
Ghost-free SumCheck verification with one exact coefficient width per phase.

Owns: the fixed-width certificate, its logical and executable claimed-chain
verifiers, semantic rounds derived from one explicit hypercube polynomial,
honest-certificate existence from representability, and the deterministic
projection of a false accepted claim to a symbolic bad challenge.

Does not own: challenge generation, root counting, Fiat--Shamir, a
protocol-specific polynomial, Rust, R1CS, or constraint costs.

Emits constraints: no.

Authority boundary: the certificate contains only fixed-width round
polynomials. Initial claims and challenges are verifier inputs; the terminal,
true initial sum, and expected rounds are recomputed from the explicit
polynomial. High zero coefficients remain part of the fixed layout and are
accepted; no canonical trimming rule is applied.

| Stage | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| certificate | every round has exactly `degree + 1` coefficients | checked by type | `Certificate` |
| claimed chain | `current = p(0) + p(1)` and next `current = p(r)` | checked | `Chain`, `checkChain` |
| terminal | final claim equals `q(challenges)` | computed | `Accepted` |
| semantic rounds | fix the challenge prefix and sum every Boolean suffix | computed | `expectedRounds` |
| completeness | representable expected rounds yield an honest certificate | derived | `exists_honest_certificate`, `complete` |
| representability | an honest fixed-width certificate witnesses every expected round's degree bound | derived | `expectedRoundsRepresentable_of_honest` |
| algebraic reduction | a false accepted initial claim exposes a distinct-function collision | derived | `false_acceptance_implies_algebraic_bad_challenge` |
| bounded reduction | expected-round representability upgrades the collision to two fixed-degree polynomials | security boundary | `false_acceptance_implies_bad_challenge` |
-/

namespace NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase

universe uField

/-- A verifier-visible certificate with one statically fixed polynomial width
for every round. It carries no semantic expected polynomial or terminal. -/
structure Certificate (Field : Type uField) (degree : Nat) where
  rounds : List (FixedPolynomial Field degree)

/-- Exact claimed-chain relation over fixed-width messages and verifier-owned
challenges. Mismatched round and challenge counts are rejected. -/
def Chain
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field) :
    Field -> List (FixedPolynomial Field degree) -> List Field -> Field -> Prop
  | current, [], [], terminal => current = terminal
  | current, polynomial :: polynomials, challenge :: challenges, terminal =>
      current = ops.add
        (polynomial.evaluate ops ops.zero)
        (polynomial.evaluate ops ops.one) ∧
      Chain ops (polynomial.evaluate ops challenge) polynomials challenges
        terminal
  | _, _, _, _ => False

/-- Acceptance recomputes the terminal from the explicit polynomial at the
full verifier challenge vector. -/
def Accepted
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree) : Prop :=
  Chain ops initial certificate.rounds challenges (q challenges)

/-- Logical acceptance consumes exactly one challenge per fixed-width round. -/
theorem Chain.rounds_length_eq_challenges_length
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (current terminal : Field)
    (rounds : List (FixedPolynomial Field degree))
    (challenges : List Field)
    (chain : Chain ops current rounds challenges terminal) :
    rounds.length = challenges.length := by
  induction rounds generalizing current challenges with
  | nil =>
      cases challenges <;> simp [Chain] at chain ⊢
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil => simp [Chain] at chain
      | cons challenge challenges =>
          simp only [Chain] at chain
          simp only [List.length_cons, Nat.succ.injEq]
          exact inductionHypothesis
            (current := polynomial.evaluate ops challenge)
            (challenges := challenges) chain.2

/-- Executable claimed-chain verifier. Static width replaces canonical-list
validation, so high zero slots are deliberately accepted. -/
def checkChain
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field) :
    Field -> List (FixedPolynomial Field degree) -> List Field -> Field -> Bool
  | current, [], [], terminal => decide (current = terminal)
  | current, polynomial :: polynomials, challenge :: challenges, terminal =>
      decide (current = ops.add
        (polynomial.evaluate ops ops.zero)
        (polynomial.evaluate ops ops.one)) &&
      checkChain ops (polynomial.evaluate ops challenge) polynomials challenges
        terminal
  | _, _, _, _ => false

/-- Executable fixed-phase verifier. -/
def check
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (q : List Field -> Field)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree) : Bool :=
  checkChain ops initial certificate.rounds challenges (q challenges)

/-- Exact executable/logical correspondence for the fixed claimed chain. -/
theorem checkChain_eq_true_iff
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (current terminal : Field)
    (rounds : List (FixedPolynomial Field degree))
    (challenges : List Field) :
    checkChain ops current rounds challenges terminal = true ↔
      Chain ops current rounds challenges terminal := by
  induction rounds generalizing current challenges with
  | nil =>
      cases challenges <;> simp [checkChain, Chain]
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil => simp [checkChain, Chain]
      | cons challenge challenges =>
          simp [checkChain, Chain, inductionHypothesis]

/-- The executable verifier accepts exactly the logical fixed-phase relation. -/
theorem check_eq_true_iff_accepted
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (q : List Field -> Field)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree) :
    check ops q initial challenges certificate = true ↔
      Accepted ops q initial challenges certificate := by
  exact checkChain_eq_true_iff ops initial (q challenges)
    certificate.rounds challenges

/-! ## Independent semantic rounds -/

/-- The true initial claim: sum the explicit polynomial over the Boolean cube
whose dimension is fixed by the verifier challenge vector. -/
def semanticInitial
    {Field : Type uField}
    (ops : Ops Field)
    (q : List Field -> Field)
    (variableCount : Nat) : Field :=
  HypercubeTruth.sumCompletions ops q [] variableCount

/-- Every expected round is derived from the same explicit polynomial and the
preceding verifier challenges. -/
def expectedRounds
    {Field : Type uField}
    (ops : Ops Field)
    (q : List Field -> Field)
    (challenges : List Field) : List (Field -> Field) :=
  HypercubeTruth.expectedPolynomials ops q challenges

@[simp] theorem expectedRounds_length
    {Field : Type uField}
    (ops : Ops Field)
    (q : List Field -> Field)
    (challenges : List Field) :
    (expectedRounds ops q challenges).length = challenges.length := by
  exact HypercubeTruth.expectedPolynomialsFrom_length ops q [] challenges

/-- A fixed polynomial represents one independently derived semantic round at
every field point, including its high fixed-width slots. -/
def Represents
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (polynomial : FixedPolynomial Field degree)
    (expected : Field -> Field) : Prop :=
  ∀ point, polynomial.evaluate ops point = expected point

/-- Lockstep representation of a finite semantic-round list. Shape mismatch is
false rather than silently truncated. -/
def Representations
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field) :
    List (FixedPolynomial Field degree) -> List (Field -> Field) -> Prop
  | [], [] => True
  | polynomial :: polynomials, expected :: expecteds =>
      Represents ops polynomial expected ∧
      Representations ops polynomials expecteds
  | _, _ => False

/-- Protocol-independent degree premise: every derived expected round admits
an exact polynomial at the chosen fixed width. -/
def ExpectedRoundsRepresentable
    {Field : Type uField}
    (ops : Ops Field)
    (q : List Field -> Field)
    (degree : Nat)
    (challenges : List Field) : Prop :=
  ∀ expected ∈ expectedRounds ops q challenges,
    ∃ polynomial : FixedPolynomial Field degree,
      Represents ops polynomial expected

/-- Certificate honesty contains no semantic fields: it compares certificate
rounds against expected rounds recomputed from `q`. -/
def Honest
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (challenges : List Field)
    (certificate : Certificate Field degree) : Prop :=
  Representations ops certificate.rounds (expectedRounds ops q challenges)

/-- Lockstep representations expose a fixed-width witness for every expected
function in the represented list. -/
theorem representations_member
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    {rounds : List (FixedPolynomial Field degree)}
    {expecteds : List (Field -> Field)}
    (representations : Representations ops rounds expecteds)
    (expected : Field -> Field)
    (member : expected ∈ expecteds) :
    ∃ polynomial : FixedPolynomial Field degree,
      polynomial ∈ rounds ∧ Represents ops polynomial expected := by
  induction rounds generalizing expecteds with
  | nil =>
      cases expecteds with
      | nil => simp at member
      | cons _ _ => simp [Representations] at representations
  | cons polynomial polynomials inductionHypothesis =>
      cases expecteds with
      | nil => simp at member
      | cons head tail =>
          simp only [Representations] at representations
          simp only [List.mem_cons] at member
          rcases member with rfl | member
          · exact ⟨polynomial, by simp, representations.1⟩
          · rcases inductionHypothesis representations.2 member with
              ⟨candidate, candidateIn, represents⟩
            exact ⟨candidate, by simp [candidateIn], represents⟩

/-- Certificate honesty is sufficient evidence for the semantic
fixed-degree premise consumed by the false-acceptance reduction. -/
theorem expectedRoundsRepresentable_of_honest
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (challenges : List Field)
    (certificate : Certificate Field degree)
    (honest : Honest ops q challenges certificate) :
    ExpectedRoundsRepresentable ops q degree challenges := by
  intro expected member
  rcases representations_member ops honest expected member with
    ⟨polynomial, _, represents⟩
  exact ⟨polynomial, represents⟩

private theorem representations_exist
    {Field : Type uField}
    (ops : Ops Field)
    (degree : Nat)
    (expecteds : List (Field -> Field))
    (representable : ∀ expected ∈ expecteds,
      ∃ polynomial : FixedPolynomial Field degree,
        Represents ops polynomial expected) :
    ∃ polynomials : List (FixedPolynomial Field degree),
      Representations ops polynomials expecteds := by
  induction expecteds with
  | nil => exact ⟨[], trivial⟩
  | cons expected expecteds inductionHypothesis =>
      obtain ⟨polynomial, represents⟩ := representable expected (by simp)
      obtain ⟨polynomials, representations⟩ := inductionHypothesis
        (fun tail tailIn => representable tail (by simp [tailIn]))
      exact ⟨polynomial :: polynomials, represents, representations⟩

/-- Fixed-width representability constructs a certificate whose rounds agree
with every independently derived semantic round. -/
theorem exists_honest_certificate
    {Field : Type uField}
    (ops : Ops Field)
    (q : List Field -> Field)
    (degree : Nat)
    (challenges : List Field)
    (representable : ExpectedRoundsRepresentable ops q degree challenges) :
    ∃ certificate : Certificate Field degree,
      Honest ops q challenges certificate := by
  obtain ⟨rounds, representations⟩ :=
    representations_exist ops degree (expectedRounds ops q challenges)
      representable
  exact ⟨⟨rounds⟩, representations⟩

private theorem chain_of_representationsFrom
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (fixed : List Field)
    (rounds : List (FixedPolynomial Field degree))
    (challenges : List Field)
    (representations : Representations ops rounds
      (HypercubeTruth.expectedPolynomialsFrom ops q fixed challenges)) :
    Chain ops
      (HypercubeTruth.sumCompletions ops q fixed challenges.length)
      rounds challenges (q (fixed ++ challenges)) := by
  induction challenges generalizing fixed rounds with
  | nil =>
      cases rounds with
      | nil => simp [Chain, HypercubeTruth.sumCompletions]
      | cons polynomial polynomials =>
          simp [HypercubeTruth.expectedPolynomialsFrom, Representations]
            at representations
  | cons challenge challenges inductionHypothesis =>
      cases rounds with
      | nil =>
          simp [HypercubeTruth.expectedPolynomialsFrom, Representations]
            at representations
      | cons polynomial polynomials =>
          simp only [HypercubeTruth.expectedPolynomialsFrom,
            Representations] at representations
          rcases representations with ⟨represents, tailRepresentations⟩
          simp only [HypercubeTruth.sumCompletions, Chain]
          constructor
          · rw [represents ops.zero, represents ops.one]
          · rw [represents challenge]
            simpa [List.append_assoc] using
              inductionHypothesis
                (fixed := fixed ++ [challenge])
                (rounds := polynomials) tailRepresentations

/-- Perfect completeness: true initial claim plus exact fixed-width semantic
rounds satisfies the ghost-free logical verifier. -/
theorem complete
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree)
    (initialIsTrue : initial = semanticInitial ops q challenges.length)
    (honest : Honest ops q challenges certificate) :
    Accepted ops q initial challenges certificate := by
  rw [initialIsTrue]
  simpa [Accepted, Honest, expectedRounds, semanticInitial] using
    chain_of_representationsFrom ops q [] certificate.rounds challenges honest

/-- Representability therefore yields an accepted honest certificate for the
true claim. The result is existential, so no choice function enters the
verifier or certificate. -/
theorem exists_honest_accepted_certificate
    {Field : Type uField}
    (ops : Ops Field)
    (q : List Field -> Field)
    (degree : Nat)
    (challenges : List Field)
    (representable : ExpectedRoundsRepresentable ops q degree challenges) :
    ∃ certificate : Certificate Field degree,
      Honest ops q challenges certificate ∧
      Accepted ops q (semanticInitial ops q challenges.length) challenges
        certificate := by
  obtain ⟨certificate, honest⟩ :=
    exists_honest_certificate ops q degree challenges representable
  exact ⟨certificate, honest,
    complete ops q _ challenges certificate rfl honest⟩

/-! ## Deterministic projection to the symbolic bad event -/

private def symbolicRoundsFrom
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field) :
    List Field -> List (FixedPolynomial Field degree) -> List Field ->
      List (SumCheck.Round Field Field)
  | fixed, polynomial :: polynomials, challenge :: challenges =>
      {
        claimed := polynomial.evaluate ops
        expected := fun value =>
          HypercubeTruth.sumCompletions ops q (fixed ++ [value])
            challenges.length
        challenge := challenge
        degree := degree
      } :: symbolicRoundsFrom ops q (fixed ++ [challenge]) polynomials
        challenges
  | _, _, _ => []

/-- Symbolic projection used only by the deterministic soundness reduction.
Every semantic field is recomputed from `q`; none is certificate data. -/
def symbolicInstance
    {Field : Type uField}
    (ops : Ops Field)
    (q : List Field -> Field)
    (degree challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree) : SumCheck.Instance Field Field where
  claimedInitial := initial
  trueInitial := semanticInitial ops q challenges.length
  terminal := q challenges
  rounds := symbolicRoundsFrom ops q [] certificate.rounds challenges
  maxDegree := degree
  challengeSetSize := challengeSetSize

/-- The algebraic collision exposed before any degree claim is made about the
independently derived expected polynomial. -/
def AlgebraicBadChallenge
    {Field : Type uField}
    (ops : Ops Field)
    (q : List Field -> Field)
    (degree challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree)
    (round : SumCheck.Round Field Field) : Prop :=
  SumCheck.BadChallenge
    (symbolicInstance ops q degree challengeSetSize initial challenges
      certificate) round

/-- Root-count-ready bad event. In addition to the symbolic collision, it
carries the certificate's claimed fixed polynomial and an independently
derived expected fixed polynomial. Expected representability is deliberately
absent from the certificate and must be proved from protocol semantics. -/
def BadChallenge
    {Field : Type uField}
    (ops : Ops Field)
    (q : List Field -> Field)
    (degree challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree)
    (round : SumCheck.Round Field Field) : Prop :=
  AlgebraicBadChallenge ops q degree challengeSetSize initial challenges
      certificate round ∧
    ∃ claimedPolynomial expectedPolynomial : FixedPolynomial Field degree,
      Represents ops claimedPolynomial round.claimed ∧
      Represents ops expectedPolynomial round.expected

private theorem symbolicExpected_mem_expectedPolynomialsFrom
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (fixed : List Field)
    (rounds : List (FixedPolynomial Field degree))
    (challenges : List Field)
    (round : SumCheck.Round Field Field)
    (roundIn : round ∈ symbolicRoundsFrom ops q fixed rounds challenges) :
    round.expected ∈
      HypercubeTruth.expectedPolynomialsFrom ops q fixed challenges := by
  induction rounds generalizing fixed challenges with
  | nil => simp [symbolicRoundsFrom] at roundIn
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil => simp [symbolicRoundsFrom] at roundIn
      | cons challenge challenges =>
          simp only [symbolicRoundsFrom, List.mem_cons] at roundIn
          simp only [HypercubeTruth.expectedPolynomialsFrom, List.mem_cons]
          rcases roundIn with rfl | roundIn
          · exact Or.inl rfl
          · exact Or.inr <| inductionHypothesis
              (fixed := fixed ++ [challenge])
              (challenges := challenges) roundIn

/-- Membership in the symbolic projection retains the exact certificate
polynomial that defines the claimed round function. -/
private theorem symbolicClaimed_representable
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (fixed : List Field)
    (rounds : List (FixedPolynomial Field degree))
    (challenges : List Field)
    (round : SumCheck.Round Field Field)
    (roundIn : round ∈ symbolicRoundsFrom ops q fixed rounds challenges) :
    ∃ polynomial : FixedPolynomial Field degree,
      Represents ops polynomial round.claimed := by
  induction rounds generalizing fixed challenges with
  | nil => simp [symbolicRoundsFrom] at roundIn
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil => simp [symbolicRoundsFrom] at roundIn
      | cons challenge challenges =>
          simp only [symbolicRoundsFrom, List.mem_cons] at roundIn
          rcases roundIn with rfl | roundIn
          · exact ⟨polynomial, fun _ => rfl⟩
          · exact inductionHypothesis
              (fixed := fixed ++ [challenge])
              (challenges := challenges) roundIn

private theorem symbolicDegreesFrom
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (fixed : List Field)
    (rounds : List (FixedPolynomial Field degree))
    (challenges : List Field) :
    ∀ round ∈ symbolicRoundsFrom ops q fixed rounds challenges,
      round.degree ≤ degree := by
  induction rounds generalizing fixed challenges with
  | nil => simp [symbolicRoundsFrom]
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil => simp [symbolicRoundsFrom]
      | cons challenge challenges =>
          intro round roundIn
          simp only [symbolicRoundsFrom, List.mem_cons] at roundIn
          rcases roundIn with rfl | roundIn
          · exact Nat.le_refl degree
          · exact inductionHypothesis (fixed := fixed ++ [challenge])
              (challenges := challenges) round roundIn

private theorem symbolicClaimedPathFrom
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (fixed : List Field)
    (current terminal : Field)
    (rounds : List (FixedPolynomial Field degree))
    (challenges : List Field)
    (chain : Chain ops current rounds challenges terminal) :
    SumCheck.Chain ops.toSymbolic (fun round => round.claimed) current
      (symbolicRoundsFrom ops q fixed rounds challenges) terminal := by
  induction rounds generalizing fixed current challenges with
  | nil =>
      cases challenges
      · simpa [Chain, symbolicRoundsFrom, SumCheck.Chain] using chain
      · simp [Chain] at chain
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil => simp [Chain] at chain
      | cons challenge challenges =>
          simp only [Chain] at chain
          simp only [symbolicRoundsFrom, SumCheck.Chain]
          exact ⟨chain.1,
            inductionHypothesis
              (fixed := fixed ++ [challenge])
              (current := polynomial.evaluate ops challenge)
              (challenges := challenges) chain.2⟩

private theorem symbolicTruthPathFrom
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (fixed : List Field)
    (rounds : List (FixedPolynomial Field degree))
    (challenges : List Field)
    (sameLength : rounds.length = challenges.length) :
    SumCheck.Chain ops.toSymbolic (fun round => round.expected)
      (HypercubeTruth.sumCompletions ops q fixed challenges.length)
      (symbolicRoundsFrom ops q fixed rounds challenges)
      (q (fixed ++ challenges)) := by
  induction rounds generalizing fixed challenges with
  | nil =>
      have challengesEmpty : challenges = [] :=
        List.eq_nil_of_length_eq_zero (by simpa using sameLength.symm)
      subst challenges
      simp [symbolicRoundsFrom, SumCheck.Chain,
        HypercubeTruth.sumCompletions]
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil => simp at sameLength
      | cons challenge challenges =>
          have tailLength : polynomials.length = challenges.length := by
            simpa using sameLength
          simp only [HypercubeTruth.sumCompletions,
            symbolicRoundsFrom, SumCheck.Chain]
          constructor
          · rfl
          · simpa [List.append_assoc] using
              inductionHypothesis
                (fixed := fixed ++ [challenge])
                (challenges := challenges) tailLength

/-- Logical fixed-phase acceptance projects to the generic symbolic claimed
path without adding canonical-shape premises. -/
theorem accepted_implies_symbolicAccepted
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree)
    (accepted : Accepted ops q initial challenges certificate) :
    SumCheck.Accepted ops.toSymbolic
      (symbolicInstance ops q degree challengeSetSize initial challenges
        certificate) := by
  constructor
  · exact symbolicDegreesFrom ops q [] certificate.rounds challenges
  · exact symbolicClaimedPathFrom ops q [] initial (q challenges)
      certificate.rounds challenges accepted

/-- The explicit hypercube polynomial always supplies the symbolic truth path
once accepted shape equality establishes one round per challenge. -/
theorem symbolicTruthPath
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree)
    (accepted : Accepted ops q initial challenges certificate) :
    SumCheck.TruthPath ops.toSymbolic
      (symbolicInstance ops q degree challengeSetSize initial challenges
        certificate) := by
  have sameLength := Chain.rounds_length_eq_challenges_length ops initial
    (q challenges) certificate.rounds challenges accepted
  simpa [SumCheck.TruthPath, symbolicInstance, semanticInitial] using
    symbolicTruthPathFrom ops q [] certificate.rounds challenges sameLength

/-- Algebraic part of the deterministic reduction. It exposes a collision with
the independently derived expected function, but deliberately makes no degree
claim about that expected function. -/
theorem false_acceptance_implies_algebraic_bad_challenge
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree)
    (accepted : Accepted ops q initial challenges certificate)
    (falseClaim : initial ≠ semanticInitial ops q challenges.length) :
    ∃ round,
      AlgebraicBadChallenge ops q degree challengeSetSize initial challenges
        certificate round := by
  have symbolicAccepted := accepted_implies_symbolicAccepted ops q
    challengeSetSize initial challenges certificate accepted
  have truthPath := symbolicTruthPath ops q challengeSetSize initial challenges
    certificate accepted
  have symbolicFalse :
      ¬ SumCheck.Claim.True
        (symbolicInstance ops q degree challengeSetSize initial challenges
          certificate) := by
    simpa [SumCheck.Claim.True, symbolicInstance] using falseClaim
  simpa [AlgebraicBadChallenge] using
    SumCheck.false_acceptance_implies_bad_challenge ops.toSymbolic
      (symbolicInstance ops q degree challengeSetSize initial challenges
        certificate) symbolicAccepted truthPath symbolicFalse

/-- Root-count-ready deterministic reduction. Expected-round representability
is an explicit semantic premise: it upgrades the algebraic collision to one
between the certificate's fixed-degree polynomial and an independently
derived fixed-degree expected polynomial. -/
theorem false_acceptance_implies_bad_challenge
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree)
    (representable : ExpectedRoundsRepresentable ops q degree challenges)
    (accepted : Accepted ops q initial challenges certificate)
    (falseClaim : initial ≠ semanticInitial ops q challenges.length) :
    ∃ round, BadChallenge ops q degree challengeSetSize initial challenges
      certificate round := by
  obtain ⟨round, algebraicBad⟩ :=
    false_acceptance_implies_algebraic_bad_challenge ops q challengeSetSize
      initial challenges certificate accepted falseClaim
  have roundIn :
      round ∈ symbolicRoundsFrom ops q [] certificate.rounds challenges := by
    exact algebraicBad.1
  have expectedIn : round.expected ∈ expectedRounds ops q challenges := by
    simpa [expectedRounds, HypercubeTruth.expectedPolynomials] using
      symbolicExpected_mem_expectedPolynomialsFrom ops q []
        certificate.rounds challenges round roundIn
  obtain ⟨expectedPolynomial, expectedRepresentation⟩ :=
    representable round.expected expectedIn
  obtain ⟨claimedPolynomial, claimedRepresentation⟩ :=
    symbolicClaimed_representable ops q [] certificate.rounds challenges
      round roundIn
  exact ⟨round, algebraicBad, claimedPolynomial, expectedPolynomial,
    claimedRepresentation, expectedRepresentation⟩

private theorem symbolicCollisionFrom_implies_causal_decomposition
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (fixed : List Field) :
    forall
      (rounds : List (FixedPolynomial Field degree))
      (challenges : List Field)
      (round : SumCheck.Round Field Field),
      round ∈ symbolicRoundsFrom ops q fixed rounds challenges ->
      round.claimed ≠ round.expected ->
      round.claimed round.challenge = round.expected round.challenge ->
      ∃ beforeChallenges challenge afterChallenges
          beforePolynomials claimedPolynomial afterPolynomials,
        challenges =
          beforeChallenges ++ challenge :: afterChallenges /\
        rounds =
          beforePolynomials ++ claimedPolynomial :: afterPolynomials /\
        beforeChallenges.length = beforePolynomials.length /\
        (fun point => claimedPolynomial.evaluate ops point) ≠
          (fun point =>
            HypercubeTruth.sumCompletions ops q
              (fixed ++ beforeChallenges ++ [point])
              afterChallenges.length) /\
        claimedPolynomial.evaluate ops challenge =
          HypercubeTruth.sumCompletions ops q
            (fixed ++ beforeChallenges ++ [challenge])
            afterChallenges.length
  | [], _, round, member, _, _ => by
      simp [symbolicRoundsFrom] at member
  | _ :: _, [], round, member, _, _ => by
      simp [symbolicRoundsFrom] at member
  | polynomial :: polynomials, challenge :: challenges, round, member,
      different, collision => by
      simp only [symbolicRoundsFrom, List.mem_cons] at member
      rcases member with rfl | tailMember
      · exact ⟨[], challenge, challenges, [], polynomial, polynomials,
          by simp, by simp, rfl, by simpa, by simpa⟩
      · obtain ⟨beforeChallenges, selectedChallenge, afterChallenges,
          beforePolynomials, claimedPolynomial, afterPolynomials,
          challengesEqual, polynomialsEqual, beforeLengths, functionsDifferent,
          valuesEqual⟩ :=
          symbolicCollisionFrom_implies_causal_decomposition ops q
            (fixed ++ [challenge]) polynomials challenges round tailMember
            different collision
        refine ⟨challenge :: beforeChallenges, selectedChallenge,
          afterChallenges, polynomial :: beforePolynomials,
          claimedPolynomial, afterPolynomials, ?_, ?_, ?_, ?_, ?_⟩
        · simp [challengesEqual]
        · simp [polynomialsEqual]
        · simpa using beforeLengths
        · simpa [List.append_assoc] using functionsDifferent
        · simpa [List.append_assoc] using valuesEqual

/-- Every fixed-phase bad challenge identifies one exact causal round:
the claimed polynomial and semantic polynomial are fixed by the prior
challenge prefix, are distinct as functions, and collide at the current
challenge. No future challenge appears in either polynomial. -/
theorem badChallenge_implies_causal_decomposition
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (certificate : Certificate Field degree)
    (bad :
      ∃ round,
        BadChallenge ops q degree challengeSetSize initial challenges
          certificate round) :
    ∃ beforeChallenges challenge afterChallenges
        beforePolynomials claimedPolynomial afterPolynomials,
      challenges =
        beforeChallenges ++ challenge :: afterChallenges /\
      certificate.rounds =
        beforePolynomials ++ claimedPolynomial :: afterPolynomials /\
      beforeChallenges.length = beforePolynomials.length /\
      (fun point => claimedPolynomial.evaluate ops point) ≠
        (fun point =>
          HypercubeTruth.sumCompletions ops q
            (beforeChallenges ++ [point]) afterChallenges.length) /\
      claimedPolynomial.evaluate ops challenge =
        HypercubeTruth.sumCompletions ops q
          (beforeChallenges ++ [challenge]) afterChallenges.length := by
  obtain ⟨round, algebraic, _claimedPolynomial, _expectedPolynomial,
      _claimedRepresents, _expectedRepresents⟩ := bad
  rcases algebraic with ⟨roundIn, _degreeBound, different, collision⟩
  simpa using
    symbolicCollisionFrom_implies_causal_decomposition ops q []
      certificate.rounds challenges round roundIn different collision

end NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase
