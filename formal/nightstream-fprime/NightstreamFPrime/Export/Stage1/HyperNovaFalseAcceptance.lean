import NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity

/-!
False acceptance at the selected full-opening terminal boundary means that
the verifier accepts but no advice list has the advertised length and forward
application result. This is exactly the history conclusion in AdviceReturned.
It is distinct from bare NIFS Boolean acceptance and from extractor failure.

The same original mixed terminal law and its actual visited laws are retained.
No conditioning on invalid inputs is used. The bound keeps the hash-collision
mass explicit and takes HyperNova errata Assumption 1 at each visit
(`HyperNovaVisitedSecurity.NifsKnowledgeSound`). Honest rejection is
separate.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaFalseAcceptance

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open HyperNovaHistory (Statement Envelope Payload SourceResult)
open HyperNovaHistoryProbability (Sample Accepted AdviceReturned)
open HyperNovaVisitedLaw (Visit goodActive visitedLaw guardedDraw observedDraw)
open HyperNovaFirstFailure (MarkedHashCollision MarkedSourceFailure)
open HyperNovaVisitedSecurity (event_ne_top)
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup)

attribute [local instance] Classical.propDecidable

/-- Acceptance with no history meeting the existing exact length and forward
evaluation contract. The initial law may mix valid and invalid statements. -/
def FalseAcceptance (input : Statement × Envelope) : Prop :=
  PerApplicationTerminal.Holds application fits productionSetup input.1 input.2 ∧
    ¬ ∃ advice : List AppWitness,
      advice.length = input.1.iteration ∧
      advice.foldl application.step input.1.z0 = input.1.zi

/-- Invalid accepted statements cannot have a returned valid history. This
uses the event definitions only, with no output-soundness premise. -/
theorem falseAcceptance_not_adviceReturned (sample : Sample)
    (invalid : FalseAcceptance (sample.1, sample.2.1)) :
    ¬ AdviceReturned sample := by
  rintro ⟨advice, _unused, _returned, length, evaluated⟩
  exact invalid.2 ⟨advice, length, evaluated⟩

private theorem marked_hash_mass
    (source : Statement → Payload → PMF SourceResult) (contexts : PMF Visit) :
    (contexts.bind (guardedDraw source)).toOuterMeasure
      {draw | MarkedHashCollision draw.1} =
      contexts.toOuterMeasure {visit | MarkedHashCollision visit} := by
  have marginal : (contexts.bind (guardedDraw source)).map Prod.fst = contexts := by
    rw [PMF.map_bind]
    have each (visit : Visit) : (guardedDraw source visit).map Prod.fst = PMF.pure visit := by
      by_cases good : goodActive visit
      · rw [guardedDraw, if_pos good, PMF.map_comp]
        exact PMF.map_const _ _
      · rw [guardedDraw, if_neg good, PMF.pure_map]
    simp_rw [each]
    exact PMF.bind_pure _
  calc
    _ = ((contexts.bind (guardedDraw source)).map Prod.fst).toOuterMeasure
        {visit | MarkedHashCollision visit} := (PMF.toOuterMeasure_map_apply _ _ _).symm
    _ = _ := congrArg
      (fun distribution => distribution.toOuterMeasure {visit | MarkedHashCollision visit})
      marginal

private theorem false_acceptance_mass_le_first_failures
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (bounded : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    initial.toOuterMeasure {input | FalseAcceptance input} ≤
      ∑ j : Fin depth,
        ((visitedLaw source initial j.val).toOuterMeasure {visit | MarkedHashCollision visit} +
          (((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
            {draw | MarkedSourceFailure draw})) := by
  let distribution := HyperNovaHistoryLaw.law source initial
  let event (j : Fin depth) : Set Sample :=
    {sample | MarkedHashCollision (observedDraw j.val sample).1 ∨
      MarkedSourceFailure (observedDraw j.val sample)}
  have inclusion : {sample | FalseAcceptance (sample.1, sample.2.1)} ∩
      distribution.support ⊆ ⋃ j : Fin depth, event j := by
    rintro sample ⟨invalid, supported⟩
    have supportedInitial : (sample.1, sample.2.1) ∈ initial.support := by
      rw [← HyperNovaHistoryLaw.initial_marginal source initial]
      exact (PMF.mem_support_map_iff _ _ _).mpr ⟨sample, supported, rfl⟩
    rcases HyperNovaFirstFailure.accepted_failure_exists_first depth sample
      (bounded (sample.1, sample.2.1) supportedInitial) invalid.1
      (falseAcceptance_not_adviceReturned sample invalid) with ⟨j, below, failure⟩
    exact Set.mem_iUnion.mpr ⟨⟨j, below⟩, failure⟩
  have initialMass : initial.toOuterMeasure {input | FalseAcceptance input} =
      distribution.toOuterMeasure {sample | FalseAcceptance (sample.1, sample.2.1)} := by
    rw [← HyperNovaHistoryLaw.initial_marginal source initial, PMF.toOuterMeasure_map_apply]
    rfl
  rw [initialMass]
  calc
    _ ≤ distribution.toOuterMeasure (⋃ j : Fin depth, event j) :=
      distribution.toOuterMeasure_mono inclusion
    _ ≤ ∑ j : Fin depth, distribution.toOuterMeasure (event j) :=
      MeasureTheory.measure_iUnion_fintype_le _ _
    _ ≤ _ := by
      apply Finset.sum_le_sum
      intro j _member
      have eventMass : distribution.toOuterMeasure (event j) =
          (((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
            {draw | MarkedHashCollision draw.1 ∨ MarkedSourceFailure draw}) := by
        rw [← HyperNovaVisitedLaw.visitedDraw_marginal source initial j.val,
          PMF.toOuterMeasure_map_apply]
        rfl
      rw [eventMass]
      have unionBound := MeasureTheory.measure_union_le
        (μ := ((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure)
        {draw | MarkedHashCollision draw.1} {draw | MarkedSourceFailure draw}
      rw [marked_hash_mass] at unionBound
      exact unionBound

/-- The actual false-acceptance mass is bounded by the first marked failures
under the same mixed law. Only the advertised iteration bound is assumed;
all finite-measure facts follow from the PMFs. No valid-source premise is used. -/
theorem probability_le_first_failures
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (bounded : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    (initial.toOuterMeasure {input | FalseAcceptance input}).toReal ≤
      ∑ j : Fin depth,
        (((visitedLaw source initial j.val).toOuterMeasure {visit | MarkedHashCollision visit}).toReal +
          (((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
            {draw | MarkedSourceFailure draw}).toReal) := by
  have bound := false_acceptance_mass_le_first_failures source initial depth bounded
  have finiteTerm (j : Fin depth) :
      (visitedLaw source initial j.val).toOuterMeasure {visit | MarkedHashCollision visit} +
        ((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
          {draw | MarkedSourceFailure draw} ≠ ∞ :=
    ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩
  have finiteSum := ENNReal.sum_ne_top.mpr (fun j (_ : j ∈ Finset.univ) => finiteTerm j)
  have realBound := ENNReal.toReal_mono finiteSum bound
  rw [ENNReal.toReal_sum (fun j (_ : j ∈ Finset.univ) => finiteTerm j)] at realBound
  have realTerm (j : Fin depth) := ENNReal.toReal_add
    (event_ne_top (visitedLaw source initial j.val) {visit | MarkedHashCollision visit})
    (event_ne_top ((visitedLaw source initial j.val).bind (guardedDraw source))
      {draw | MarkedSourceFailure draw})
  simpa only [realTerm] using realBound

/-- False-acceptance bound for the selected terminal verifier on the original
mixed law under Assumption 1. Each visit contributes its marked hash-collision
mass and the assumed NIFS knowledge error. No output-soundness premise or
numerical advantage is supplied. This is not a bound on bare NIFS Boolean
acceptance. -/
theorem probability_bound
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (depthBound : ∀ input ∈ initial.support, input.1.iteration ≤ depth)
    (error : Fin depth → ℝ)
    (knowledge : HyperNovaVisitedSecurity.NifsKnowledgeSound source initial depth error) :
    (initial.toOuterMeasure {input | FalseAcceptance input}).toReal ≤
      ∑ j : Fin depth,
        (((visitedLaw source initial j.val).toOuterMeasure {visit | MarkedHashCollision visit}).toReal +
          error j) :=
  (probability_le_first_failures source initial depth depthBound).trans
    (Finset.sum_le_sum fun j _ => add_le_add le_rfl (knowledge j))

end NightstreamFPrime.Export.Stage1.HyperNovaFalseAcceptance
