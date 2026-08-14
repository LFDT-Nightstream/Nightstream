import Nightstream.Assurance.Nebula.IdealTranscript

/-!
Contract: explicit coupling boundary from the V2 Poseidon2/Fiat--Shamir
transcript to the exact two-repetition public-coin fingerprint experiment.

Owns the uniform four-coordinate table probability, the five named transcript
failure events, finite union accounting, and the transfer of the public-coin
Schwartz--Zippel bound through a non-circular coupling contract.

Does not prove that Poseidon2 is a random oracle, any failure-event budget,
frame extraction from generated rows, query counts, adaptive programming, or
Rust transcript conformance.

The coupling contract speaks only about challenge distributions, equality of
actual and ideal tables outside named failures, and per-failure probability
bounds. It does not assume memory balance, execution, or the final soundness
conclusion.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.Nebula.TranscriptSecurity

open Nightstream.Implementation.Nebula.ConcreteField
open Nightstream.Assurance.Nebula.FingerprintSecurity
open Nightstream.Assurance.Nebula.IdealTranscript
open Nightstream.Protocol.Nebula.Fingerprint

/-- Exact probability of an event under a uniform four-coordinate table. -/
noncomputable def uniformTableProbability
    (event : ChallengeTable ChallengeField → Prop) : ℚ≥0 := by
  classical
  exact
    ((Finset.univ.filter event).card : ℚ≥0) /
      Fintype.card (ChallengeTable ChallengeField)

noncomputable def AcceptsPolynomial
    (polynomial : MvPolynomial (Fin 2) ChallengeField)
    (table : ChallengeTable ChallengeField) : Prop :=
  tableEquiv ChallengeField table ∈ acceptingRepeatedPoints polynomial

private theorem fourthPower_eq_square_square
    {MonoidType : Type} [Monoid MonoidType] (value : MonoidType) :
    value ^ 4 = (value ^ 2) ^ 2 := by
  rw [show 4 = 2 * 2 by decide, pow_mul]

/-- Reindexing by the exact table equivalence preserves every probability
mass. The fixed-frame transcript therefore uses the same sample space as the
public-coin theorem. -/
theorem uniformTableProbability_accepts_eq_repeatedProbability
    (polynomial : MvPolynomial (Fin 2) ChallengeField) :
    uniformTableProbability (AcceptsPolynomial polynomial) =
      repeatedProbability polynomial := by
  classical
  have acceptingCard :
      (Finset.univ.filter (AcceptsPolynomial polynomial)).card =
        (acceptingRepeatedPoints polynomial).card := by
    have subtypeCard :
        Fintype.card
            { table : ChallengeTable ChallengeField //
              AcceptsPolynomial polynomial table } =
          Fintype.card
            { point : (Fin 2 → ChallengeField) ×
                (Fin 2 → ChallengeField) //
              point ∈ acceptingRepeatedPoints polynomial } :=
      Fintype.card_congr
        ((tableEquiv ChallengeField).subtypeEquiv fun _ => Iff.rfl)
    calc
      (Finset.univ.filter (AcceptsPolynomial polynomial)).card =
          Fintype.card
            { table : ChallengeTable ChallengeField //
              AcceptsPolynomial polynomial table } :=
        (Fintype.card_subtype (AcceptsPolynomial polynomial)).symm
      _ = Fintype.card
            { point : (Fin 2 → ChallengeField) ×
                (Fin 2 → ChallengeField) //
              point ∈ acceptingRepeatedPoints polynomial } :=
        subtypeCard
      _ = (Finset.univ.filter fun point =>
            point ∈ acceptingRepeatedPoints polynomial).card :=
        Fintype.card_subtype _
      _ = (acceptingRepeatedPoints polynomial).card := by
        rw [Finset.filter_mem_eq_inter]
        simp only [Finset.univ_inter]
  have tableCard :
      Fintype.card (ChallengeTable ChallengeField) =
        (Fintype.card ChallengeField ^ 2) ^ 2 := by
    rw [challengeTable_cardinality]
    exact fourthPower_eq_square_square _
  unfold uniformTableProbability repeatedProbability
  rw [acceptingCard, tableCard]
  simp only [Nat.cast_pow]

/-- Minimal probability laws needed by the coupling theorem. -/
structure ProbabilityModel (Outcome : Type) where
  probability : (Outcome → Prop) → ℚ≥0
  monotone : ∀ {left right : Outcome → Prop},
    (∀ outcome, left outcome → right outcome) →
      probability left ≤ probability right
  unionBound : ∀ (left right : Outcome → Prop),
    probability (fun outcome => left outcome ∨ right outcome) ≤
      probability left + probability right

/-- Closed V2 failure family for the memory-challenge transcript. -/
structure FailureEvents (Outcome : Type) where
  frameNotCommitted : Outcome → Prop
  poseidonCollision : Outcome → Prop
  poseidonMulticollision : Outcome → Prop
  latePreimage : Outcome → Prop
  adaptiveProgramming : Outcome → Prop

def FailureEvents.Any
    {Outcome : Type} (events : FailureEvents Outcome)
    (outcome : Outcome) : Prop :=
  events.frameNotCommitted outcome ∨
    events.poseidonCollision outcome ∨
    events.poseidonMulticollision outcome ∨
    events.latePreimage outcome ∨
    events.adaptiveProgramming outcome

structure Budget where
  frameNotCommitted : ℚ≥0
  poseidonCollision : ℚ≥0
  poseidonMulticollision : ℚ≥0
  latePreimage : ℚ≥0
  adaptiveProgramming : ℚ≥0

def Budget.total (budget : Budget) : ℚ≥0 :=
  budget.frameNotCommitted +
    (budget.poseidonCollision +
      (budget.poseidonMulticollision +
        (budget.latePreimage + budget.adaptiveProgramming)))

/-- A coupling between the actual transcript table and an ideal uniform
table. Uniformity is required for every ideal-table event, not only the
fingerprint acceptance event. -/
structure CouplingContract
    {Outcome : Type}
    (model : ProbabilityModel Outcome)
    (actualTable idealTable : Outcome → ChallengeTable ChallengeField)
    (events : FailureEvents Outcome)
    (budget : Budget) : Prop where
  idealUniform : ∀ event,
    model.probability (fun outcome => event (idealTable outcome)) =
      uniformTableProbability event
  agreesUnlessFailure : ∀ outcome,
    ¬ events.Any outcome → actualTable outcome = idealTable outcome
  frameNotCommittedBound :
    model.probability events.frameNotCommitted ≤ budget.frameNotCommitted
  poseidonCollisionBound :
    model.probability events.poseidonCollision ≤ budget.poseidonCollision
  poseidonMulticollisionBound :
    model.probability events.poseidonMulticollision ≤
      budget.poseidonMulticollision
  latePreimageBound :
    model.probability events.latePreimage ≤ budget.latePreimage
  adaptiveProgrammingBound :
    model.probability events.adaptiveProgramming ≤
      budget.adaptiveProgramming

theorem anyFailure_probability_le
    {Outcome : Type}
    {model : ProbabilityModel Outcome}
    {actualTable idealTable : Outcome → ChallengeTable ChallengeField}
    {events : FailureEvents Outcome}
    {budget : Budget}
    (contract :
      CouplingContract model actualTable idealTable events budget) :
    model.probability events.Any ≤ budget.total := by
  have lateTail :
      model.probability (fun outcome =>
        events.latePreimage outcome ∨ events.adaptiveProgramming outcome) ≤
      budget.latePreimage + budget.adaptiveProgramming :=
    (model.unionBound events.latePreimage events.adaptiveProgramming).trans
      (add_le_add contract.latePreimageBound
        contract.adaptiveProgrammingBound)
  have multicollisionTail :
      model.probability (fun outcome =>
        events.poseidonMulticollision outcome ∨
          events.latePreimage outcome ∨
          events.adaptiveProgramming outcome) ≤
      budget.poseidonMulticollision +
        (budget.latePreimage + budget.adaptiveProgramming) :=
    (model.unionBound events.poseidonMulticollision
      (fun outcome =>
        events.latePreimage outcome ∨ events.adaptiveProgramming outcome)).trans
      (add_le_add contract.poseidonMulticollisionBound lateTail)
  have collisionTail :
      model.probability (fun outcome =>
        events.poseidonCollision outcome ∨
          events.poseidonMulticollision outcome ∨
          events.latePreimage outcome ∨
          events.adaptiveProgramming outcome) ≤
      budget.poseidonCollision +
        (budget.poseidonMulticollision +
          (budget.latePreimage + budget.adaptiveProgramming)) :=
    (model.unionBound events.poseidonCollision
      (fun outcome =>
        events.poseidonMulticollision outcome ∨
          events.latePreimage outcome ∨
          events.adaptiveProgramming outcome)).trans
      (add_le_add contract.poseidonCollisionBound multicollisionTail)
  exact
    (model.unionBound events.frameNotCommitted
      (fun outcome =>
        events.poseidonCollision outcome ∨
          events.poseidonMulticollision outcome ∨
          events.latePreimage outcome ∨
          events.adaptiveProgramming outcome)).trans
      (add_le_add contract.frameNotCommittedBound collisionTail)

/-- The actual table can make any event occur only when the coupled ideal
table makes it occur or one named transcript failure occurs. -/
theorem actual_event_probability_le_uniform_add_failure
    {Outcome : Type}
    {model : ProbabilityModel Outcome}
    {actualTable idealTable : Outcome → ChallengeTable ChallengeField}
    {events : FailureEvents Outcome}
    {budget : Budget}
    (contract :
      CouplingContract model actualTable idealTable events budget)
    (event : ChallengeTable ChallengeField → Prop) :
    model.probability (fun outcome => event (actualTable outcome)) ≤
      uniformTableProbability event + budget.total := by
  have cover : ∀ outcome,
      event (actualTable outcome) →
        event (idealTable outcome) ∨ events.Any outcome := by
    intro outcome actual
    by_cases failure : events.Any outcome
    · exact Or.inr failure
    · left
      rw [← contract.agreesUnlessFailure outcome failure]
      exact actual
  calc
    model.probability (fun outcome => event (actualTable outcome)) ≤
        model.probability (fun outcome =>
          event (idealTable outcome) ∨ events.Any outcome) :=
      model.monotone cover
    _ ≤ model.probability (fun outcome => event (idealTable outcome)) +
          model.probability events.Any :=
      model.unionBound _ _
    _ = uniformTableProbability event +
          model.probability events.Any := by
      rw [contract.idealUniform]
    _ ≤ uniformTableProbability event + budget.total :=
      add_le_add (le_refl _) (anyFailure_probability_le contract)

/-- Concrete ROM transfer for one nonzero fingerprint polynomial. The
Schwartz--Zippel term is derived; only the named transcript budget is added. -/
theorem actual_fingerprint_probability_le_profile_add_transcript
    {Outcome : Type}
    {model : ProbabilityModel Outcome}
    {actualTable idealTable : Outcome → ChallengeTable ChallengeField}
    {events : FailureEvents Outcome}
    {budget : Budget}
    (contract :
      CouplingContract model actualTable idealTable events budget)
    {polynomial : MvPolynomial (Fin 2) ChallengeField}
    (nonzero : polynomial ≠ 0)
    (degreeBound : polynomial.totalDegree ≤ maxSegmentFactors) :
    model.probability (fun outcome =>
        AcceptsPolynomial polynomial (actualTable outcome)) ≤
      (maxSegmentFactors / Fintype.card ChallengeField : ℚ≥0) ^ 2 +
        budget.total := by
  calc
    model.probability (fun outcome =>
        AcceptsPolynomial polynomial (actualTable outcome)) ≤
        uniformTableProbability (AcceptsPolynomial polynomial) +
          budget.total :=
      actual_event_probability_le_uniform_add_failure contract _
    _ = repeatedProbability polynomial + budget.total := by
      rw [uniformTableProbability_accepts_eq_repeatedProbability]
    _ ≤ (maxSegmentFactors / Fintype.card ChallengeField : ℚ≥0) ^ 2 +
          budget.total :=
      add_le_add (repeatedProbability_le_profile nonzero degreeBound)
        (le_refl _)

end Nightstream.Assurance.Nebula.TranscriptSecurity
