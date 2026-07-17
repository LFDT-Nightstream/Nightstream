import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe
import Nightstream.SuperNeo.SumCheck.FixedPhase.Sequential

/-!
Sequential honest-prover construction for the mixed-width Split-NC FE phase.

Owns: the row-then-lane honest construction against the actual physical
message widths, direct transcript-state handoff across the phase cut, exact
finite-index certificate materialization, and honesty at the challenge point
derived by replaying that same physical certificate.

Does not own: output-message construction, `yRing` source authority,
Poseidon2 encoding, Fiat--Shamir probability, NC, Rust, R1CS, rows, costs, or
row removal.

Emits constraints: no.

Authority boundary: row messages are derived before their row challenges at
the syntax-derived row width. The resulting transcript state and row
challenge prefix then determine the quadratic lane messages and lane
challenges. Uniform high-zero lane widening is used only by the semantic
checker and never enters transcript replay.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.prover.row` | derive row-width messages from the current row prefix | derived | `rowRoundRepresentable` |
| `nifs.pi_ccs.fe.prover.phase_cut` | start lane replay from the exact row successor state and prefix | direct dataflow | `exists_honest_certificate` |
| `nifs.pi_ccs.fe.prover.lane` | derive three-slot lane messages from the current lane prefix | derived | `laneRoundRepresentable` |
| `nifs.pi_ccs.fe.prover.serialization` | transcript replay sees only physical row then lane messages | direct dataflow | `sequentialRun_eq_runRoundsFrom`, `exists_honest_certificate` |
| `nifs.pi_ccs.fe.prover.honesty` | physical certificate is honest at its own derived point | derived | `exists_honest_certificate` |
| `nifs.pi_ccs.fe.prover.completeness` | FE truth yields transcript-bound semantic acceptance | derived | `complete_of_truth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Sequential

private abbrev ops := ConcreteCarrier.extensionOps

universe uState

/-- Reduce the full FE polynomial to its row domain by summing the complete
lane Boolean suffix. -/
private def rowPolynomial
    (q : List K -> K)
    (laneVariables : Nat)
    (rowCoordinates : List K) : K :=
  HypercubeTruth.sumCompletions ops.toOps q rowCoordinates laneVariables

/-- Fix the transcript-derived row prefix and expose only lane coordinates. -/
private def lanePolynomial
    (q : List K -> K)
    (rowChallenges laneCoordinates : List K) : K :=
  q (rowChallenges ++ laneCoordinates)

/-- Prefixing the input of a polynomial commutes with recursive Boolean
completion sums. -/
private theorem sumCompletions_prepend
    (q : List K -> K)
    (leading fixed : List K)
    (remaining : Nat) :
    HypercubeTruth.sumCompletions ops.toOps
        (fun suffix => q (leading ++ suffix)) fixed remaining =
      HypercubeTruth.sumCompletions ops.toOps q
        (leading ++ fixed) remaining := by
  induction remaining generalizing fixed with
  | zero =>
      simp [HypercubeTruth.sumCompletions]
  | succ remaining inductionHypothesis =>
      simp only [HypercubeTruth.sumCompletions]
      rw [inductionHypothesis, inductionHypothesis]
      simp only [List.append_assoc]

/-- The row prefix of the full expected-round list is exactly the expected
list for the lane-summed row polynomial. -/
private theorem expectedPolynomialsFrom_take_prefix
    (q : List K -> K)
    (fixed rowChallenges laneChallenges : List K) :
    (HypercubeTruth.expectedPolynomialsFrom ops.toOps q fixed
        (rowChallenges ++ laneChallenges)).take rowChallenges.length =
      HypercubeTruth.expectedPolynomialsFrom ops.toOps
        (rowPolynomial q laneChallenges.length) fixed rowChallenges := by
  induction rowChallenges generalizing fixed with
  | nil =>
      rfl
  | cons challenge rowChallenges inductionHypothesis =>
      simp only [List.cons_append,
        HypercubeTruth.expectedPolynomialsFrom, List.length_cons,
        List.take_succ_cons]
      congr 1
      · funext point
        simpa [rowPolynomial, List.length_append] using
          HypercubeTruth.sumCompletions_add ops.toOps q
            (fixed ++ [point]) rowChallenges.length laneChallenges.length
      · exact inductionHypothesis (fixed := fixed ++ [challenge])

/-- Dropping the row prefix from the full expected-round list leaves the
lane expected rounds at the fixed row challenge prefix. -/
private theorem expectedPolynomialsFrom_drop_prefix
    (q : List K -> K)
    (fixed rowChallenges laneChallenges : List K) :
    (HypercubeTruth.expectedPolynomialsFrom ops.toOps q fixed
        (rowChallenges ++ laneChallenges)).drop rowChallenges.length =
      HypercubeTruth.expectedPolynomialsFrom ops.toOps q
        (fixed ++ rowChallenges) laneChallenges := by
  induction rowChallenges generalizing fixed with
  | nil =>
      simp
  | cons challenge rowChallenges inductionHypothesis =>
      simp only [List.cons_append,
        HypercubeTruth.expectedPolynomialsFrom, List.length_cons,
        List.drop_succ_cons]
      simpa [List.append_assoc] using
        inductionHypothesis (fixed := fixed ++ [challenge])

/-- Expected rounds of the lane-only view equal the full polynomial's
expected rounds after fixing the row challenge prefix. -/
private theorem expectedPolynomialsFrom_lanePolynomial
    (q : List K -> K)
    (rowChallenges fixed laneChallenges : List K) :
    HypercubeTruth.expectedPolynomialsFrom ops.toOps
        (lanePolynomial q rowChallenges) fixed laneChallenges =
      HypercubeTruth.expectedPolynomialsFrom ops.toOps q
        (rowChallenges ++ fixed) laneChallenges := by
  induction laneChallenges generalizing fixed with
  | nil =>
      rfl
  | cons challenge laneChallenges inductionHypothesis =>
      simp only [HypercubeTruth.expectedPolynomialsFrom]
      congr 1
      · funext point
        simpa [lanePolynomial, List.append_assoc] using
          sumCompletions_prepend q rowChallenges
            (fixed ++ [point]) laneChallenges.length
      · simpa [List.append_assoc] using
          inductionHypothesis (fixed := fixed ++ [challenge])

/-- Generic fixed-width sequential replay is the concrete FE replay after
erasing only the static coefficient-width index. -/
theorem sequentialRun_eq_runRoundsFrom
    {State : Type uState}
    {degree : Nat}
    (machine : Transcript.Fe.Machine State)
    (state : State)
    (rounds : List (FixedPolynomial K degree)) :
    run
        (fun state polynomial =>
          Transcript.Fe.runRound machine state polynomial.toMessage)
        state rounds =
      Transcript.Fe.runRoundsFrom machine state
        (rounds.map FixedPolynomial.toMessage) := by
  induction rounds generalizing state with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      simp only [run, List.map_cons, Transcript.Fe.runRoundsFrom]
      rw [inductionHypothesis]

/-- Every row round of the lane-summed polynomial has the syntax-derived
physical row width, using only the current row prefix. -/
theorem rowRoundRepresentable
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) :
    RoundRepresentable ops.toOps
      (rowPolynomial
        (InitialSum.sumcheckPolynomial profile data coins)
        domain.laneVariables)
      (rowSumcheckDegreeBound (PublicInput.ofSources data))
      shape.rowVariables := by
  intro fixed remaining length
  have rowPhase : fixed.length < shape.rowVariables := by
    omega
  have totalLength :
      fixed.length + 1 + (remaining + domain.laneVariables) =
        shape.rowVariables + domain.laneVariables := by
    omega
  rcases Degree.expectedRowRound_bounded profile data coins fixed
      (remaining + domain.laneVariables) rowPhase totalLength with
    ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents point]
  simpa [rowPolynomial] using
    HypercubeTruth.sumCompletions_add ops.toOps
      (InitialSum.sumcheckPolynomial profile data coins)
      (fixed ++ [point]) remaining domain.laneVariables

/-- After fixing the exact row challenge prefix, every lane round has the
independently proved physical quadratic width. -/
theorem laneRoundRepresentable
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (rowChallenges : List K)
    (rowLength : rowChallenges.length = shape.rowVariables) :
    RoundRepresentable ops.toOps
      (lanePolynomial
        (InitialSum.sumcheckPolynomial profile data coins)
        rowChallenges)
      laneSumcheckDegreeBound domain.laneVariables := by
  intro fixed remaining length
  have lanePhase :
      shape.rowVariables <= (rowChallenges ++ fixed).length := by
    simp [rowLength]
  have totalLength :
      (rowChallenges ++ fixed).length + 1 + remaining =
        shape.rowVariables + domain.laneVariables := by
    simp only [List.length_append]
    omega
  rcases Degree.expectedLaneRound_quadratic profile data coins
      (rowChallenges ++ fixed) remaining lanePhase totalLength with
    ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents point]
  simpa [lanePolynomial, List.append_assoc] using
    (sumCompletions_prepend
      (InitialSum.sumcheckPolynomial profile data coins)
      rowChallenges (fixed ++ [point]) remaining).symm

/-- Construct one physical mixed-width FE certificate whose own replay
derives the challenge point at which its row and lane rounds are honest. -/
theorem exists_honest_certificate
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (coins : Coins shape domain) :
    ∃ certificate :
        SumCheck.Fe.Certificate (PublicInput.ofSources data) domain,
      SumCheck.Fe.HonestAt
        (InitialSum.sumcheckPolynomial profile data coins)
        (Transcript.Fe.derive machine initialState certificate).challengePoint
        certificate := by
  let q := InitialSum.sumcheckPolynomial profile data coins
  let qRow := rowPolynomial q domain.laneVariables
  rcases exists_honest_run ops.toOps qRow
      (rowSumcheckDegreeBound (PublicInput.ofSources data))
      shape.rowVariables
      (fun state polynomial =>
        Transcript.Fe.runRound machine state polynomial.toMessage)
      (by
        simpa [q, qRow] using rowRoundRepresentable profile data coins)
      (machine.enterFe initialState) with
    ⟨rowCertificate, rowChallenges, rowState, rowRoundsLength,
      rowChallengesLength, rowReplay, rowHonest⟩
  let qLane := lanePolynomial q rowChallenges
  rcases exists_honest_run ops.toOps qLane laneSumcheckDegreeBound
      domain.laneVariables
      (fun state polynomial =>
        Transcript.Fe.runRound machine state polynomial.toMessage)
      (by
        simpa [q, qLane] using laneRoundRepresentable profile data coins
          rowChallenges rowChallengesLength)
      rowState with
    ⟨laneCertificate, laneChallenges, finalState, laneRoundsLength,
      laneChallengesLength, laneReplay, laneHonest⟩
  let certificate :
      SumCheck.Fe.Certificate (PublicInput.ofSources data) domain := {
    rowRounds := functionOfExactList
      rowCertificate.rounds rowRoundsLength
    laneRounds := functionOfExactList
      laneCertificate.rounds laneRoundsLength
  }
  have rowRoundsExact :
      List.ofFn certificate.rowRounds = rowCertificate.rounds := by
    dsimp only [certificate]
    exact ofFn_functionOfExactList
      rowCertificate.rounds rowRoundsLength
  have laneRoundsExact :
      List.ofFn certificate.laneRounds = laneCertificate.rounds := by
    dsimp only [certificate]
    exact ofFn_functionOfExactList
      laneCertificate.rounds laneRoundsLength
  have rowRawExact :
      certificate.rowRawRounds =
        rowCertificate.rounds.map FixedPolynomial.toMessage := by
    unfold SumCheck.Fe.Certificate.rowRawRounds
    rw [rowRoundsExact]
  have laneRawExact :
      certificate.laneRawRounds =
        laneCertificate.rounds.map FixedPolynomial.toMessage := by
    unfold SumCheck.Fe.Certificate.laneRawRounds
    rw [laneRoundsExact]
  have rowTranscriptReplay :
      Transcript.Fe.runRoundsFrom machine (machine.enterFe initialState)
          certificate.rowRawRounds =
        (rowChallenges, rowState) := by
    rw [rowRawExact, ← sequentialRun_eq_runRoundsFrom]
    exact rowReplay
  have laneTranscriptReplay :
      Transcript.Fe.runRoundsFrom machine rowState
          certificate.laneRawRounds =
        (laneChallenges, finalState) := by
    rw [laneRawExact, ← sequentialRun_eq_runRoundsFrom]
    exact laneReplay
  have transcriptReplay :
      Transcript.Fe.runRoundsFrom machine (machine.enterFe initialState)
          certificate.rawRounds =
        (rowChallenges ++ laneChallenges, finalState) := by
    rw [SumCheck.Fe.Certificate.rawRounds,
      Transcript.Fe.runRoundsFrom_append]
    rw [rowTranscriptReplay]
    simp only
    rw [laneTranscriptReplay]
  have derivedCoordinates :
      (Transcript.Fe.derive machine initialState certificate).challengePoint.coordinates =
        rowChallenges ++ laneChallenges := by
    rw [Transcript.Fe.derive_point_coordinates]
    rw [transcriptReplay]
  have rowExpected :
      (FixedPhase.expectedRounds ops.toOps q
          (rowChallenges ++ laneChallenges)).take shape.rowVariables =
        FixedPhase.expectedRounds ops.toOps qRow rowChallenges := by
    rw [← rowChallengesLength]
    simpa [FixedPhase.expectedRounds, HypercubeTruth.expectedPolynomials,
      qRow, laneChallengesLength] using
        expectedPolynomialsFrom_take_prefix q []
          rowChallenges laneChallenges
  have laneExpected :
      (FixedPhase.expectedRounds ops.toOps q
          (rowChallenges ++ laneChallenges)).drop shape.rowVariables =
        FixedPhase.expectedRounds ops.toOps qLane laneChallenges := by
    rw [← rowChallengesLength]
    calc
      (FixedPhase.expectedRounds ops.toOps q
          (rowChallenges ++ laneChallenges)).drop rowChallenges.length =
          HypercubeTruth.expectedPolynomialsFrom ops.toOps q
            rowChallenges laneChallenges := by
        exact expectedPolynomialsFrom_drop_prefix q []
          rowChallenges laneChallenges
      _ = FixedPhase.expectedRounds ops.toOps qLane laneChallenges := by
        simpa [FixedPhase.expectedRounds, HypercubeTruth.expectedPolynomials,
          qLane] using
            (expectedPolynomialsFrom_lanePolynomial q
              rowChallenges [] laneChallenges).symm
  refine ⟨certificate, ?_⟩
  unfold SumCheck.Fe.HonestAt
  rw [derivedCoordinates]
  change FixedPhase.Representations ops.toOps
      (List.ofFn certificate.rowRounds)
      ((FixedPhase.expectedRounds ops.toOps q
        (rowChallenges ++ laneChallenges)).take shape.rowVariables) ∧
    FixedPhase.Representations ops.toOps
      (List.ofFn certificate.laneRounds)
      ((FixedPhase.expectedRounds ops.toOps q
        (rowChallenges ++ laneChallenges)).drop shape.rowVariables)
  rw [rowRoundsExact, laneRoundsExact, rowExpected, laneExpected]
  exact ⟨rowHonest, laneHonest⟩

/-- Honest FE truth yields transcript-bound semantic acceptance at the
terminal recomputed from the certificate's own derived point. -/
theorem complete_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (coins : Coins shape domain)
    (truth : Semantics.Fe.Truth data) :
    ∃ certificate :
        SumCheck.Fe.Certificate (PublicInput.ofSources data) domain,
      Transcript.Fe.Accepted machine initialState
        (initial profile (PublicInput.ofSources data) coins)
        (InitialSum.sumcheckPolynomial profile data coins
          (Transcript.Fe.derive machine initialState certificate).challengePoint.coordinates)
        certificate := by
  rcases exists_honest_certificate profile data machine initialState coins with
    ⟨certificate, honest⟩
  refine ⟨certificate, ?_⟩
  unfold Transcript.Fe.Accepted
  exact SumCheck.Fe.complete_of_truth_and_honestAt
    profile data coins truth
    (Transcript.Fe.derive machine initialState certificate).challengePoint
    certificate honest

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver
