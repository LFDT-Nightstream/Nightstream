import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum.CarriedBridge
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.MixingSoundness
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Interface

/-!
Semantic honesty and fixed-challenge completeness for mixed-width Split-NC FE
SumCheck.

Owns: phase-specific honest representations, exact-width honest-certificate
existence, the bridge to generic fixed-phase honesty, model-level
fixed-challenge completeness, and deterministic soundness into independent
FE truth or explicitly named compression/SumCheck bad events.

Does not own: the physical certificate/checker interface, transcript challenge
derivation, Fiat--Shamir security, Rust, R1CS, rows, removals, or costs.

Emits constraints: no.

Authority boundary: expected functions are recomputed from the independent FE
polynomial. Semantic reasoning consumes the physical interface but cannot add
certificate fields, widen serialized messages, or choose transcript values.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.sumcheck.honesty` | physical row/lane messages represent independently derived rounds | derived | `HonestAt` |
| `nifs.pi_ccs.fe.sumcheck.honesty.existence` | syntax-derived row and quadratic lane bounds construct an honest certificate | derived | `exists_honestAt` |
| `nifs.pi_ccs.fe.sumcheck.honesty.bridge` | phase-specific honesty implies generic fixed-phase honesty | derived | `honestAt_implies_fixedPhaseHonest` |
| `nifs.pi_ccs.fe.sumcheck.degree` | every semantic round has the verifier-owned uniform row ceiling | derived | `expectedRoundsRepresentable` |
| `nifs.pi_ccs.fe.sumcheck.completeness` | FE truth plus honest rounds is accepted | model-level | `complete_of_truth_and_honestAt` |
| `nifs.pi_ccs.fe.sumcheck.soundness` | false acceptance implies FE compression or fixed-degree round collision | security boundary | `accepted_implies_truth_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

/-- Exact phase-specific honesty at one fixed verifier challenge point.
Expected functions are recomputed from the independent polynomial; only the
certificate's physical row/lane polynomials appear on the claimed side. -/
def HonestAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (q : List K -> K)
    (point : Point shape domain)
    (certificate : Certificate input domain) : Prop :=
  let expected := FixedPhase.expectedRounds ops.toOps q point.coordinates
  FixedPhase.Representations ops.toOps
      (List.ofFn certificate.rowRounds)
      (expected.take shape.rowVariables) ∧
    FixedPhase.Representations ops.toOps
      (List.ofFn certificate.laneRounds)
      (expected.drop shape.rowVariables)

/-- Reindex an exactly sized list as the finite function carried by the
physical certificate. This changes no polynomial or coefficient width. -/
private def finFunctionOfList
    {Element : Type}
    {count : Nat}
    (values : List Element)
    (length : values.length = count) : Fin count -> Element :=
  fun index => values.get (Fin.cast length.symm index)

/-- Re-serializing an exactly sized list through its finite-function view is
the identity. -/
@[simp] private theorem ofFn_finFunctionOfList
    {Element : Type}
    {count : Nat}
    (values : List Element)
    (length : values.length = count) :
    List.ofFn (finFunctionOfList values length) = values := by
  apply List.ext_get
  · simp [length]
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_ofFn]
    rfl

/-- Construct the physical quadratic suffix after the row coordinates have
already been fixed. -/
private theorem laneRepresentations_exist
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (fixed challenges : List K)
    (lanePhase : shape.rowVariables <= fixed.length)
    (length : fixed.length + challenges.length =
      shape.rowVariables + domain.laneVariables) :
    exists rounds : List LaneMessage,
      rounds.length = challenges.length ∧
      FixedPhase.Representations ops.toOps rounds
        (HypercubeTruth.expectedPolynomialsFrom ops.toOps
          (InitialSum.sumcheckPolynomial profile data coins)
          fixed challenges) := by
  induction challenges generalizing fixed with
  | nil =>
      exact ⟨[], rfl, trivial⟩
  | cons challenge challenges inductionHypothesis =>
      rcases Degree.expectedLaneRound_quadratic profile data coins fixed
          challenges.length lanePhase (by
            simp only [List.length_cons] at length
            omega) with
        ⟨round, represents⟩
      have nextLanePhase :
          shape.rowVariables <= (fixed ++ [challenge]).length := by
        simp only [List.length_append, List.length_singleton]
        omega
      have nextLength :
          (fixed ++ [challenge]).length + challenges.length =
            shape.rowVariables + domain.laneVariables := by
        simp at length ⊢
        omega
      rcases inductionHypothesis (fixed := fixed ++ [challenge])
          nextLanePhase nextLength with
        ⟨rounds, roundsLength, representations⟩
      refine ⟨round :: rounds, by simp [roundsLength], ?_⟩
      simp only [HypercubeTruth.expectedPolynomialsFrom,
        FixedPhase.Representations]
      exact ⟨represents, representations⟩

/-- Construct row messages at the syntax-derived width and lane messages at
three slots, preserving the exact row/lane split of the semantic rounds. -/
private theorem mixedRepresentations_exist
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (fixed rowChallenges laneChallenges : List K)
    (rowLength : fixed.length + rowChallenges.length = shape.rowVariables)
    (laneLength : laneChallenges.length = domain.laneVariables) :
    exists rowRounds : List (RowMessage (PublicInput.ofSources data)),
      exists laneRounds : List LaneMessage,
        rowRounds.length = rowChallenges.length ∧
        laneRounds.length = laneChallenges.length ∧
        FixedPhase.Representations ops.toOps rowRounds
          ((HypercubeTruth.expectedPolynomialsFrom ops.toOps
            (InitialSum.sumcheckPolynomial profile data coins)
            fixed (rowChallenges ++ laneChallenges)).take
              rowChallenges.length) ∧
        FixedPhase.Representations ops.toOps laneRounds
          ((HypercubeTruth.expectedPolynomialsFrom ops.toOps
            (InitialSum.sumcheckPolynomial profile data coins)
            fixed (rowChallenges ++ laneChallenges)).drop
              rowChallenges.length) := by
  induction rowChallenges generalizing fixed with
  | nil =>
      have lanePhase : shape.rowVariables <= fixed.length := by
        simp at rowLength
        omega
      have totalLength :
          fixed.length + laneChallenges.length =
            shape.rowVariables + domain.laneVariables := by
        omega
      rcases laneRepresentations_exist profile data coins fixed laneChallenges
          lanePhase totalLength with
        ⟨laneRounds, laneRoundsLength, laneRepresentations⟩
      exact ⟨[], laneRounds, rfl, laneRoundsLength, trivial, by
        simpa using laneRepresentations⟩
  | cons challenge rowChallenges inductionHypothesis =>
      have rowPhase : fixed.length < shape.rowVariables := by
        simp only [List.length_cons] at rowLength
        omega
      have roundLength :
          fixed.length + 1 +
              (rowChallenges.length + laneChallenges.length) =
            shape.rowVariables + domain.laneVariables := by
        simp only [List.length_cons] at rowLength
        omega
      rcases Degree.expectedRowRound_bounded profile data coins fixed
          (rowChallenges.length + laneChallenges.length) rowPhase
          roundLength with
        ⟨rowRound, rowRepresents⟩
      have nextRowLength :
          (fixed ++ [challenge]).length + rowChallenges.length =
            shape.rowVariables := by
        simp at rowLength ⊢
        omega
      rcases inductionHypothesis (fixed := fixed ++ [challenge])
          nextRowLength with
        ⟨rowRounds, laneRounds, rowRoundsLength, laneRoundsLength,
          rowRepresentations, laneRepresentations⟩
      refine ⟨rowRound :: rowRounds, laneRounds,
        by simp [rowRoundsLength], laneRoundsLength, ?_, ?_⟩
      · simp only [List.cons_append,
          HypercubeTruth.expectedPolynomialsFrom, List.length_cons,
          List.take_succ_cons, FixedPhase.Representations]
        exact ⟨by simpa [List.length_append] using rowRepresents,
          rowRepresentations⟩
      · simpa only [List.cons_append,
          HypercubeTruth.expectedPolynomialsFrom, List.length_cons,
          List.drop_succ_cons] using laneRepresentations

/-- Every fixed verifier point admits an honest certificate whose row and lane
messages retain their independently proved physical widths.

This is fixed-challenge algebraic existence only. It neither constructs a
Fiat--Shamir transcript nor permits a uniform-width wire certificate. -/
theorem exists_honestAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain) :
    exists certificate : Certificate (PublicInput.ofSources data) domain,
      HonestAt (InitialSum.sumcheckPolynomial profile data coins)
        point certificate := by
  rcases mixedRepresentations_exist profile data coins []
      point.row.coordinates point.lane.coordinates (by
        simpa using point.row.dimension) point.lane.dimension with
    ⟨rowRounds, laneRounds, rowRoundsLength, laneRoundsLength,
      rowRepresentations, laneRepresentations⟩
  let certificate : Certificate (PublicInput.ofSources data) domain := {
    rowRounds := finFunctionOfList rowRounds (by
      simpa using rowRoundsLength.trans point.row.dimension)
    laneRounds := finFunctionOfList laneRounds (by
      simpa using laneRoundsLength.trans point.lane.dimension) }
  refine ⟨certificate, ?_⟩
  unfold HonestAt
  change FixedPhase.Representations ops.toOps
      (List.ofFn certificate.rowRounds)
      ((FixedPhase.expectedRounds ops.toOps
        (InitialSum.sumcheckPolynomial profile data coins)
        point.coordinates).take shape.rowVariables) ∧
    FixedPhase.Representations ops.toOps
      (List.ofFn certificate.laneRounds)
      ((FixedPhase.expectedRounds ops.toOps
        (InitialSum.sumcheckPolynomial profile data coins)
        point.coordinates).drop shape.rowVariables)
  simpa only [certificate, ofFn_finFunctionOfList, Point.coordinates,
    FixedPhase.expectedRounds, HypercubeTruth.expectedPolynomials,
    List.nil_append, point.row.dimension] using
      And.intro rowRepresentations laneRepresentations

private theorem representations_append
    {degree : Nat}
    {leftRounds rightRounds : List (FixedPolynomial K degree)}
    {leftExpected rightExpected : List (K -> K)}
    (left : FixedPhase.Representations ops.toOps
      leftRounds leftExpected)
    (right : FixedPhase.Representations ops.toOps
      rightRounds rightExpected) :
    FixedPhase.Representations ops.toOps
      (leftRounds ++ rightRounds) (leftExpected ++ rightExpected) := by
  induction leftRounds generalizing leftExpected with
  | nil =>
      cases leftExpected <;>
        simp [FixedPhase.Representations] at left ⊢
      exact right
  | cons polynomial polynomials inductionHypothesis =>
      cases leftExpected with
      | nil => simp [FixedPhase.Representations] at left
      | cons expected expecteds =>
          simp only [FixedPhase.Representations] at left
          simp only [List.cons_append, FixedPhase.Representations]
          exact ⟨left.1, inductionHypothesis left.2⟩

private theorem lane_representations_widen
    {shape : SemanticShape}
    (input : PublicInput shape)
    {rounds : List LaneMessage}
    {expected : List (K -> K)}
    (represented :
      FixedPhase.Representations ops.toOps rounds expected) :
    FixedPhase.Representations ops.toOps
      (rounds.map (laneToUniform input)) expected := by
  induction rounds generalizing expected with
  | nil =>
      cases expected <;>
        simp [FixedPhase.Representations] at represented ⊢
  | cons polynomial polynomials inductionHypothesis =>
      cases expected with
      | nil => simp [FixedPhase.Representations] at represented
      | cons function functions =>
          simp only [FixedPhase.Representations] at represented
          simp only [List.map_cons, FixedPhase.Representations]
          constructor
          · intro point
            rw [lane_evaluate_uniform input polynomial point,
              represented.1 point]
          · exact inductionHypothesis represented.2

/-- Phase-specific physical honesty implies honesty of the semantic uniform
proof view. This theorem is the only bridge that permits generic fixed-phase
reasoning about the mixed-width certificate. -/
theorem honestAt_implies_fixedPhaseHonest
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (q : List K -> K)
    (point : Point shape domain)
    (certificate : Certificate input domain)
    (honest : HonestAt q point certificate) :
    FixedPhase.Honest ops.toOps q point.coordinates
      { rounds := certificate.uniformRounds } := by
  unfold HonestAt at honest
  unfold FixedPhase.Honest
  have joined := representations_append honest.1
    (lane_representations_widen input honest.2)
  rw [Certificate.uniformRounds]
  simpa only [List.take_append_drop] using joined

/-- Every independently derived FE round admits the uniform fixed-degree
representation used only by the generic soundness reduction. Physical lane
messages remain three slots; this theorem does not widen serialization. -/
theorem expectedRoundsRepresentable
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain) :
    FixedPhase.ExpectedRoundsRepresentable ops.toOps
      (InitialSum.sumcheckPolynomial profile data coins)
      (Drow (PublicInput.ofSources data)) point.coordinates := by
  rcases exists_honestAt profile data coins point with
    ⟨certificate, honest⟩
  exact FixedPhase.expectedRoundsRepresentable_of_honest ops.toOps
    (InitialSum.sumcheckPolynomial profile data coins) point.coordinates
    { rounds := certificate.uniformRounds }
    (honestAt_implies_fixedPhaseHonest
      (InitialSum.sumcheckPolynomial profile data coins)
      point certificate honest)

/-- Model-level fixed-challenge completeness. This theorem neither constructs
Fiat--Shamir challenges nor claims production transcript conformance. -/
theorem complete_of_truth_and_honestAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (truth : Semantics.Fe.Truth data)
    (point : Point shape domain)
    (certificate : Certificate (PublicInput.ofSources data) domain)
    (honest : HonestAt
      (InitialSum.sumcheckPolynomial profile data coins) point certificate) :
    Accepted
      (initial profile (PublicInput.ofSources data) coins)
      (InitialSum.sumcheckPolynomial profile data coins point.coordinates)
      point certificate := by
  apply FixedPhase.complete ops.toOps
    (InitialSum.sumcheckPolynomial profile data coins)
    (initial profile (PublicInput.ofSources data) coins)
    point.coordinates
    { rounds := certificate.uniformRounds }
  · rw [InitialSum.CarriedBridge.initial_eq_sumcheckHypercubeSum_of_truth
      profile data coins truth]
    unfold InitialSum.sumcheckHypercubeSum FixedPhase.semanticInitial
    rw [point.coordinates_length]
  · exact honestAt_implies_fixedPhaseHonest
      (InitialSum.sumcheckPolynomial profile data coins) point certificate honest

/-- Exhaustive deterministic reasons why fixed-challenge FE SumCheck may
accept without the independent fresh-CCS and carried-evaluation obligations.
No transcript sampling or probability bound is asserted here. -/
inductive BadEvent
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain)
    (certificate : Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat) : Prop where
  | mixingRoot
      (root :
        Polynomial.Fe.MixingSoundness.MixingRoot profile data coins) :
      BadEvent profile data coins point certificate challengeSetSize
  | roundCollision
      (round : Nightstream.SuperNeo.SumCheck.Round K K)
      (collision :
        FixedPhase.BadChallenge ops.toOps
          (InitialSum.sumcheckPolynomial profile data coins)
          (Drow (PublicInput.ofSources data)) challengeSetSize
          (initial profile (PublicInput.ofSources data) coins)
          point.coordinates
          { rounds := certificate.uniformRounds } round) :
      BadEvent profile data coins point certificate challengeSetSize

/-- FE acceptance is sound up to the exact semantic-compression and
fixed-degree SumCheck collision events.

The proof first uses the independent identity
`initial - sum(Q) = mixedResidual`. If the mixed residual is zero, the
compression theorem yields FE truth or a mixing root. Otherwise the verifier's
initial claim is false, so generic fixed-phase soundness yields a bounded
round collision. -/
theorem accepted_implies_truth_or_badEvent
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain)
    (certificate : Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat)
    (accepted :
      Accepted
        (initial profile (PublicInput.ofSources data) coins)
        (InitialSum.sumcheckPolynomial profile data coins point.coordinates)
        point certificate) :
    Semantics.Fe.Truth data ∨
      BadEvent profile data coins point certificate challengeSetSize := by
  by_cases compressedZero :
      InitialSum.mixedResidual profile data coins = K.zero
  · rcases
      (Polynomial.Fe.MixingSoundness.mixedResidual_eq_zero_iff_truth_or_mixingRoot
        profile data coins).mp compressedZero with
      truth | mixingRoot
    · exact Or.inl truth
    · exact Or.inr (.mixingRoot mixingRoot)
  · apply Or.inr
    have semanticInitial_eq_hypercube :
        FixedPhase.semanticInitial ops.toOps
            (InitialSum.sumcheckPolynomial profile data coins)
            point.coordinates.length =
          InitialSum.hypercubeSum profile data coins := by
      unfold FixedPhase.semanticInitial
      rw [point.coordinates_length]
      change InitialSum.sumcheckHypercubeSum profile data coins =
        InitialSum.hypercubeSum profile data coins
      exact InitialSum.sumcheckHypercubeSum_eq_hypercubeSum
        profile data coins
    have falseClaim :
        initial profile (PublicInput.ofSources data) coins ≠
          FixedPhase.semanticInitial ops.toOps
            (InitialSum.sumcheckPolynomial profile data coins)
            point.coordinates.length := by
      intro initialEqualsSemantic
      apply compressedZero
      rw [
        ← InitialSum.CarriedBridge.initial_sub_hypercubeSum_eq_mixedResidual
          profile data coins]
      apply (FiniteSumAlgebra.sub_eq_zero_iff ops laws _ _).2
      exact initialEqualsSemantic.trans semanticInitial_eq_hypercube
    have semanticAccepted :
        FixedPhase.Accepted ops.toOps
          (InitialSum.sumcheckPolynomial profile data coins)
          (initial profile (PublicInput.ofSources data) coins)
          point.coordinates
          { rounds := certificate.uniformRounds } := by
      exact accepted
    rcases FixedPhase.false_acceptance_implies_bad_challenge ops.toOps
        (InitialSum.sumcheckPolynomial profile data coins)
        challengeSetSize
        (initial profile (PublicInput.ofSources data) coins)
        point.coordinates
        { rounds := certificate.uniformRounds }
        (expectedRoundsRepresentable profile data coins point)
        semanticAccepted falseClaim with
      ⟨round, collision⟩
    exact .roundCollision round collision

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe
