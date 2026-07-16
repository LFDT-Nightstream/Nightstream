import Nightstream.SuperNeo.Folding.Nifs

/-!
Typed events for the partial Fiat--Shamir carrier currently available in the
candidate SuperNeo NIFS model.

Owns: the verifier-visible prover messages currently represented by
`Nifs.Attempt`, their challenge carriers, a canonical partial phase order, and
diagnostic metadata naming known omissions.

Does not own: semantic `expected` SumCheck polynomials, truth-path witnesses,
concrete encodings, a random-oracle implementation, the concrete `Pi_RLC`
candidate stream or fixed-bound refinement, backend refinement, cost
accounting, or permission to remove checks. Generic first-accepted semantics
live in the sibling `PiRlcSampler` component rather than in this event carrier.

Emits backend obligations: no.

Authority boundary: `SumCheckMessage` deliberately excludes `trueInitial`,
`Round.expected`, and `Round.challenge`. Those are respectively semantic ghost
data or verifier-derived data, never prover-message authority. A typed seed
binds the public context; later challenges are carried only by the ordered
events below.

| Protocol | Phase | Constraint family | Mathematical obligation |
|---|---|---|---|
| NIFS | seed | public context | bind public parameters, verifier key, expected structure, and exact input statement |
| PiCCS | FE round | function-valued claim | absorb only the function-valued claim and untrusted declared-degree metadata before carrying that round's verifier challenge |
| PiCCS | NC round | function-valued claim | absorb only the function-valued claim and untrusted declared-degree metadata before carrying that round's verifier challenge |
| PiCCS -> PiRLC | prefix boundary | `piCcsPrefixDirectives` | absorb only the prover-sent evaluation arrays after both SumCheck chains and before deriving any PiRLC coefficient; reconstructed structure, commitment, public input, point, and stage are not rehashed |
| PiRLC | challenge vector | coordinate challenge | carry every coordinate under a distinct typed index from one verifier-owned batch rooted at the post-PiCCS prefix; the sampler refinement separately threads its internal state across coordinates |
| PiCCS | initial / terminal authority | derived claims | explicitly open because those verifier-read values are outside this partial schedule |
| PiCCS | verifier configuration | degree, set size, round counts | explicitly open until fixed parameters and exact FE/NC shapes are bound |
| PiCCS | pre-SumCheck coins | alpha and gamma | explicitly open because current `PiCCS.Attempt` has no carriers for them |
| PiCCS | round encoding | function-valued claim representation | explicitly open because `claimed : Challenge -> Value` has no canonical serialization or checked coefficient-degree witness |
| PiCCS | output point | challenge-to-point linkage | explicitly open because generic `Challenge` and `Point` have no refinement map |
| PiCCS | NC terminal | challenge/output sidecar | explicitly open because the generic split-NC attempt has no carrier connecting NC challenges to an additional terminal coordinate-column message |
| PiCCS | paper / production relation | joint-Q to SplitNc | explicitly open because two accepted chains are not yet proved equivalent to Section 7.3's one joint polynomial |
| PiCCS | output compression | projection sufficiency | explicitly open until context plus the absorbed payload uniquely determines every verifier-used output field |
| PiCCS | production split refinement | beta_a, beta_r, beta_m | explicitly open as production-refinement coins absent from the paper model and current carriers, not as a paper obligation |
| PiRLC | concrete sampler refinement | candidate stream, fixed bound, and strong-set decoding | generic first-accepted selection, exact 54-of-64 arithmetic, an abstract jointly owned four-block stream/state schedule, a pure production-shaped Poseidon2 transcript machine, coefficient-vector assembly/validity, and the reached-state replay bridge are proved separately; complete-carrier integration, the exact post-PiCCS start state, native/gadget/generated-row conformance, quotient-ring strong-set lifting, and lowering remain open |
| all | concrete transcript | canonical encoding and Poseidon2 tags | explicitly open because typed functions do not prove domain separation or concrete hash refinement |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.NonInteractive

universe uPublicParameters uVerifierKey uStructure uPublicInput uPoint
  uEvaluation uCommitment uScalar uChallenge uValue

/-- The two independently modeled SumCheck chains owned by PiCCS. -/
inductive SumCheckPhase where
  | fe
  | nc
deriving DecidableEq, Repr

/--
Exactly the verifier-visible portion of one SumCheck round message.

The semantic polynomial and carried challenge in `SumCheck.Round` are
intentionally absent.
-/
structure SumCheckMessage (Challenge : Type uChallenge) (Value : Type uValue) where
  claimed : Challenge -> Value
  /-- Untrusted metadata checked against `maxDegree`, not proof of the
  function-valued polynomial's actual degree. -/
  declaredDegree : Nat

/--
Domain-hardened seed context for one NIFS execution.

HyperNova Construction 3 begins from `hs = RO(pp, s)`. This canonical model
additionally binds the verifier key and exact public statement at the typed
seed boundary. Whether a concrete encoding realizes this stronger policy is a
later refinement theorem, not an assumption of this file.
-/
structure Context
    (PublicParameters : Type uPublicParameters)
    (VerifierKey : Type uVerifierKey)
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (arity : BatchArity params) where
  publicParameters : PublicParameters
  verifierKey : VerifierKey
  relationStructure : Structure
  statement : PiCCS.InputProduct
    Structure PublicInput Point Evaluation Commitment params arity

/-- Typed prover messages in their paper-level NIFS order. -/
inductive ProverMessage
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Challenge : Type uChallenge)
    (Value : Type uValue)
    (params : GlobalParams)
    (arity : BatchArity params) where
  | sumCheckRound
      (phase : SumCheckPhase)
      (roundIndex : Nat)
      (message : SumCheckMessage Challenge Value)
  | piCcsOutputEvaluations
      (evaluations : Fin arity.total -> Array Evaluation)

/--
An event skeleton has no prover-carried challenge value. Replay materializes
every verifier response from the chained prefix.
-/
inductive Directive
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Challenge : Type uChallenge)
    (Value : Type uValue)
    (params : GlobalParams)
    (arity : BatchArity params) where
  | absorb
      (message : ProverMessage
        Structure PublicInput Point Evaluation Commitment Challenge Value
          params arity)
  | sumCheckChallenge (phase : SumCheckPhase) (roundIndex : Nat)
  | piRlcChallenge (coordinate : Fin arity.total)

/-- A fully materialized typed event, including verifier response values. -/
inductive Event
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (Challenge : Type uChallenge)
    (Value : Type uValue)
    (params : GlobalParams)
    (arity : BatchArity params) where
  | absorb
      (message : ProverMessage
        Structure PublicInput Point Evaluation Commitment Challenge Value
          params arity)
  | sumCheckChallenge
      (phase : SumCheckPhase)
      (roundIndex : Nat)
      (challenge : Challenge)
  | piRlcChallenge
      (coordinate : Fin arity.total)
      (challenge : Scalar)

/-- Forget verifier response values while preserving the exact typed skeleton. -/
def eraseResponse
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params} :
    Event
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity ->
    Directive
      Structure PublicInput Point Evaluation Commitment Challenge Value
        params arity
  | .absorb message => .absorb message
  | .sumCheckChallenge phase roundIndex _ =>
      .sumCheckChallenge phase roundIndex
  | .piRlcChallenge coordinate _ => .piRlcChallenge coordinate

/-- List projection used to state replay exactness. -/
def eraseResponses
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (events : List (Event
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)) :
    List (Directive
      Structure PublicInput Point Evaluation Commitment Challenge Value
        params arity) :=
  events.map eraseResponse

/-- Known boundaries absent from the current partial transcript carrier.

This is diagnostic metadata only. Membership in or deletion from this enum is
not a proof that transcript coverage is complete. -/
inductive CoverageGap where
  | sumCheckInitialBinding
  | sumCheckTerminalBinding
  | sumCheckVerifierParameters
  | sumCheckRoundShape
  | piCcsAlpha
  | piCcsGamma
  | sumCheckPolynomialEncoding
  | sumCheckChallengePointLink
  | piCcsNcTerminalSidecar
  | piCcsJointQSplitRefinement
  | piCcsOutputProjectionSufficiency
  | piCcsSplitCoins
  /-- A pure production-shaped Poseidon2 machine now instantiates the jointly
  owned block schedule, fixes overwrite absorption and lane-major little-endian
  chunk order, and proves successful executions reach the same four-block
  successor state. It has not yet been proved equal to the native transcript,
  transcript gadget, generated R1CS trace, or the machine started at the exact
  post-PiCCS replay state. Those bridges are required before it can establish
  `PiRlcSampler.ResponseRefinesAt` for production acceptance. -/
  | piRlcBoundedSampler
  /-- The sampled 54-coordinate coefficient vector is proved pointwise in
  `[-2,2]`; distinct vectors are separated by a nonzero coordinate; every
  difference coordinate lies strictly inside the minimal threshold `5`; and
  the expansion arithmetic is `216`. Concrete centered embedding into
  Goldilocks, preservation under quotient-ring subtraction, and the Theorem-8
  low-norm invertibility lift remain open. -/
  | piRlcStrongSet
  | piRlcSamplingDistribution
  | concreteTranscriptEncoding
deriving DecidableEq, Repr

/-- Diagnostic list of known open boundaries. It is not a formal coverage
criterion and must never authorize a constraint removal. -/
def coverageGaps : List CoverageGap :=
  [.sumCheckInitialBinding, .sumCheckTerminalBinding,
    .sumCheckVerifierParameters, .sumCheckRoundShape,
    .piCcsAlpha, .piCcsGamma, .sumCheckPolynomialEncoding,
    .sumCheckChallengePointLink, .piCcsNcTerminalSidecar,
    .piCcsJointQSplitRefinement, .piCcsOutputProjectionSufficiency,
    .piCcsSplitCoins, .piRlcBoundedSampler, .piRlcStrongSet,
    .piRlcSamplingDistribution, .concreteTranscriptEncoding]

/-- Fail-closed status for this slice. There is intentionally no `complete`
constructor: a future complete surface must be a record of actual refinement
theorems, not an empty diagnostic list. -/
inductive CoverageStatus where
  | incomplete (gaps : List CoverageGap)
deriving Repr

/-- Current diagnostic status. -/
def coverageStatus : CoverageStatus := .incomplete coverageGaps

/-- The partial carrier is explicitly marked incomplete. -/
theorem coverageStatus_eq_incomplete :
    coverageStatus = .incomplete coverageGaps := by
  rfl

/-- The paper's pre-SumCheck point challenge remains explicitly open. -/
theorem piCcsAlpha_is_coverageGap :
    CoverageGap.piCcsAlpha ∈ coverageGaps := by
  simp [coverageGaps]

/-- The paper's pre-SumCheck mixing challenge remains explicitly open. -/
theorem piCcsGamma_is_coverageGap :
    CoverageGap.piCcsGamma ∈ coverageGaps := by
  simp [coverageGaps]

/-- Function-valued claimed polynomials still need a concrete encoding proof. -/
theorem sumCheckPolynomialEncoding_is_coverageGap :
    CoverageGap.sumCheckPolynomialEncoding ∈ coverageGaps := by
  simp [coverageGaps]

/-- SumCheck-derived coordinates are not yet identified with PiCCS points. -/
theorem sumCheckChallengePointLink_is_coverageGap :
    CoverageGap.sumCheckChallengePointLink ∈ coverageGaps := by
  simp [coverageGaps]

/-- Split NC still lacks a typed terminal challenge/output sidecar. -/
theorem piCcsNcTerminalSidecar_is_coverageGap :
    CoverageGap.piCcsNcTerminalSidecar ∈ coverageGaps := by
  simp [coverageGaps]

/-- The paper's one joint polynomial is not yet related to production SplitNc. -/
theorem piCcsJointQSplitRefinement_is_coverageGap :
    CoverageGap.piCcsJointQSplitRefinement ∈ coverageGaps := by
  simp [coverageGaps]

/-- Reduced output absorption has no uniqueness/sufficiency theorem yet. -/
theorem piCcsOutputProjectionSufficiency_is_coverageGap :
    CoverageGap.piCcsOutputProjectionSufficiency ∈ coverageGaps := by
  simp [coverageGaps]

/--
Production SplitNc's beta_a, beta_r, and beta_m coins remain a concrete
refinement gap. They are not asserted to be obligations of the paper model.
-/
theorem piCcsSplitCoins_is_coverageGap :
    CoverageGap.piCcsSplitCoins ∈ coverageGaps := by
  simp [coverageGaps]

/-- The bounded strong-set sampler remains a separate refinement obligation. -/
theorem piRlcBoundedSampler_is_coverageGap :
    CoverageGap.piRlcBoundedSampler ∈ coverageGaps := by
  simp [coverageGaps]

/-- Typed replay is not yet a concrete encoding/Poseidon2 refinement. -/
theorem concreteTranscriptEncoding_is_coverageGap :
    CoverageGap.concreteTranscriptEncoding ∈ coverageGaps := by
  simp [coverageGaps]

private def sumCheckDirectives
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (phase : SumCheckPhase) :
    Nat -> List (SumCheck.Round Challenge Value) ->
      List (Directive
        Structure PublicInput Point Evaluation Commitment Challenge Value
          params arity)
  | _, [] => []
  | roundIndex, round :: rest =>
      .absorb (.sumCheckRound phase roundIndex {
        claimed := round.claimed
        declaredDegree := round.degree
      }) ::
      .sumCheckChallenge phase roundIndex ::
      sumCheckDirectives phase (roundIndex + 1) rest

private def sumCheckEvents
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (phase : SumCheckPhase) :
    Nat -> List (SumCheck.Round Challenge Value) ->
      List (Event
        Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
          params arity)
  | _, [] => []
  | roundIndex, round :: rest =>
      .absorb (.sumCheckRound phase roundIndex {
        claimed := round.claimed
        declaredDegree := round.degree
      }) ::
      .sumCheckChallenge phase roundIndex round.challenge ::
      sumCheckEvents phase (roundIndex + 1) rest

private theorem eraseResponses_sumCheckEvents
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (phase : SumCheckPhase)
    (roundIndex : Nat)
    (rounds : List (SumCheck.Round Challenge Value)) :
    eraseResponses (sumCheckEvents
      (Structure := Structure)
      (PublicInput := PublicInput)
      (Point := Point)
      (Evaluation := Evaluation)
      (Commitment := Commitment)
      (Scalar := Scalar)
      (params := params)
      (arity := arity)
      phase roundIndex rounds) =
    sumCheckDirectives
      (Structure := Structure)
      (PublicInput := PublicInput)
      (Point := Point)
      (Evaluation := Evaluation)
      (Commitment := Commitment)
      (params := params)
      (arity := arity)
      phase roundIndex rounds := by
  induction rounds generalizing roundIndex with
  | nil => rfl
  | cons round rest inductionHypothesis =>
      simp only [sumCheckEvents, sumCheckDirectives, eraseResponses,
        List.map_cons, eraseResponse, List.cons.injEq, true_and]
      simpa only [eraseResponses] using
        inductionHypothesis (roundIndex + 1)

/-- The exact pre-PiRLC directive prefix: both represented SumCheck chains,
then the one represented PiCCS output-evaluation message. This boundary is
public so replay/sampler refinement can name the actually reached state. -/
def piCcsPrefixDirectives
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    List (Directive
      Structure PublicInput Point Evaluation Commitment Challenge Value
        params arity) :=
  sumCheckDirectives .fe 0 attempt.piCcs.fe.rounds ++
  sumCheckDirectives .nc 0 attempt.piCcs.nc.rounds ++
  [.absorb (.piCcsOutputEvaluations
    (fun coordinate => (attempt.piCcs.outputs coordinate).evaluations))]

/-- The PiRLC response suffix has one coordinate directive for every source. -/
def piRlcDirectives
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    (arity : BatchArity params) :
    List (Directive
      Structure PublicInput Point Evaluation Commitment Challenge Value
        params arity) :=
  (List.finRange arity.total).map fun coordinate =>
    .piRlcChallenge coordinate

/-- The one canonical verifier-owned schedule for the challenge carriers
currently present in `Nifs.Attempt`. -/
def schedule
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    List (Directive
      Structure PublicInput Point Evaluation Commitment Challenge Value
        params arity) :=
  piCcsPrefixDirectives attempt ++ piRlcDirectives arity

/-- The carried-event form of `piCcsPrefixDirectives`. -/
def piCcsPrefixEvents
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    List (Event
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :=
  sumCheckEvents .fe 0 attempt.piCcs.fe.rounds ++
  sumCheckEvents .nc 0 attempt.piCcs.nc.rounds ++
  [.absorb (.piCcsOutputEvaluations
    (fun coordinate => (attempt.piCcs.outputs coordinate).evaluations))]

/-- The carried PiRLC suffix using the attempt's coefficient vector. -/
def piRlcEvents
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    List (Event
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :=
  (List.finRange arity.total).map fun coordinate =>
    .piRlcChallenge coordinate (attempt.piRlc.challenges coordinate)

/-- The same canonical schedule populated with the attempt's carried values. -/
def canonicalEvents
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    List (Event
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :=
  piCcsPrefixEvents attempt ++ piRlcEvents attempt

/-! ## Formal blindness witnesses for the partial schedule -/

/-- Mutate every FE `SumCheck.Instance` envelope field while preserving the
round list. This helper exists to make the current replay omission executable
and reviewable; it is not a protocol operation. -/
def replaceFeEnvelope
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (claimedInitial trueInitial terminal : Value)
    (maxDegree challengeSetSize : Nat) :
    Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity :=
  { attempt with
    piCcs := { attempt.piCcs with
      fe := { attempt.piCcs.fe with
        claimedInitial := claimedInitial
        trueInitial := trueInitial
        terminal := terminal
        maxDegree := maxDegree
        challengeSetSize := challengeSetSize
      }
    }
  }

/-- The current canonical event list cannot observe FE initial/terminal or
verifier-parameter mutations. This is a formal gap witness, not a desirable
invariance. -/
theorem canonicalEvents_replaceFeEnvelope
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (claimedInitial trueInitial terminal : Value)
    (maxDegree challengeSetSize : Nat) :
    canonicalEvents
      (replaceFeEnvelope attempt claimedInitial trueInitial terminal
        maxDegree challengeSetSize) =
      canonicalEvents attempt := by
  rfl

/-- NC counterpart of `replaceFeEnvelope`. -/
def replaceNcEnvelope
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (claimedInitial trueInitial terminal : Value)
    (maxDegree challengeSetSize : Nat) :
    Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity :=
  { attempt with
    piCcs := { attempt.piCcs with
      nc := { attempt.piCcs.nc with
        claimedInitial := claimedInitial
        trueInitial := trueInitial
        terminal := terminal
        maxDegree := maxDegree
        challengeSetSize := challengeSetSize
      }
    }
  }

/-- The current canonical event list is equally blind to the NC envelope. -/
theorem canonicalEvents_replaceNcEnvelope
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (claimedInitial trueInitial terminal : Value)
    (maxDegree challengeSetSize : Nat) :
    canonicalEvents
      (replaceNcEnvelope attempt claimedInitial trueInitial terminal
        maxDegree challengeSetSize) =
      canonicalEvents attempt := by
  rfl

/-- Mutate every PiCCS output point while preserving the absorbed evaluation
payload. This helper witnesses the missing challenge-to-point binding. -/
def replacePiCcsOutputPoints
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (points : Fin arity.total -> Point) :
    Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity :=
  { attempt with
    piCcs := { attempt.piCcs with
      outputs := fun index =>
        { attempt.piCcs.outputs index with point := points index }
    }
  }

/-- The partial event projection is blind to every PiCCS output point. -/
theorem canonicalEvents_replacePiCcsOutputPoints
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (points : Fin arity.total -> Point) :
    canonicalEvents (replacePiCcsOutputPoints attempt points) =
      canonicalEvents attempt := by
  rfl

/-- Erasing the carried pre-PiRLC events recovers exactly the directive prefix
used to define the reached sampler state. -/
theorem eraseResponses_piCcsPrefixEvents
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    eraseResponses (piCcsPrefixEvents attempt) =
      piCcsPrefixDirectives attempt := by
  have feRounds := eraseResponses_sumCheckEvents
    (Structure := Structure)
    (PublicInput := PublicInput)
    (Point := Point)
    (Evaluation := Evaluation)
    (Commitment := Commitment)
    (Scalar := Scalar)
    (params := params)
    (arity := arity)
    .fe 0 attempt.piCcs.fe.rounds
  have ncRounds := eraseResponses_sumCheckEvents
    (Structure := Structure)
    (PublicInput := PublicInput)
    (Point := Point)
    (Evaluation := Evaluation)
    (Commitment := Commitment)
    (Scalar := Scalar)
    (params := params)
    (arity := arity)
    .nc 0 attempt.piCcs.nc.rounds
  simp only [eraseResponses] at feRounds ncRounds
  simp only [piCcsPrefixEvents, piCcsPrefixDirectives, eraseResponses,
    List.map_append,
    List.map_cons, List.map_nil, eraseResponse]
  rw [feRounds, ncRounds]

/-- Erasing carried responses recovers exactly the canonical event skeleton. -/
theorem eraseResponses_canonicalEvents
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    eraseResponses (canonicalEvents attempt) = schedule attempt := by
  have prefixEq := eraseResponses_piCcsPrefixEvents attempt
  simp only [eraseResponses] at prefixEq
  simp only [canonicalEvents, schedule, eraseResponses, List.map_append]
  rw [prefixEq]
  simp [piRlcEvents, piRlcDirectives, List.map_map, Function.comp_def,
    eraseResponse]

end Nightstream.SuperNeo.Folding.Nifs.NonInteractive
