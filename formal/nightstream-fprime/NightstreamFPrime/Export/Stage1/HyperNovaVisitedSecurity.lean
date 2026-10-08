import NightstreamFPrime.Export.Stage1.HyperNovaFirstFailure

/-!
Owns the history security bound of the selected HyperNova IVC under HyperNova
errata Assumption 1, plain-model part: the Poseidon2 NIFS is knowledge sound
at each visited step.

Inputs: a source extractor (a kernel from each visited statement and payload
to a source result), the initial terminal law, a depth bound and a per-visit
error.

Outputs:
- `NifsKnowledgeSound`: Assumption 1 at the actual visited laws. After a real
  NIFS acceptance at visit `j` (`goodActive`), the extractor returns no
  checked source witness with probability at most `error j`;
- `history_probability_bound`: the accepted terminal mass is at most the
  returned-history mass plus, at each visit, the marked hash-collision mass
  and `error j`.

Assumption 1 is not a theorem here. `Lifecycle.RandomOracleKnowledge` proves
a random-oracle analogue for one fold at error `knowledgeError Q`, which
motivates the value of `error`; no Lean statement derives one from the other
(TRUST_BOUNDARY.md lists the differences). The step circuit recomputes the
previous fold's challenges with Poseidon2, so the history uses the concrete
hash, which no random-oracle model covers.

Does not own: the extractor's work (HyperNova Definition 7 requires an
expected polynomial-time extractor), numerical hardness, or query
applicability.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity

open scoped BigOperators ENNReal
open HyperNovaHistory (Statement Envelope Payload SourceResult)
open HyperNovaVisitedLaw (visitedLaw guardedDraw)
open HyperNovaFirstFailure (MarkedSourceFailure)

/-- Every event of a PMF has finite mass. -/
theorem event_ne_top {Sample : Type*} (distribution : PMF Sample) (event : Set Sample) :
    distribution.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact distribution.tsum_coe_indicator_ne_top event

private theorem first_failure_real_bound
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (bounded : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    (initial.toOuterMeasure {input |
      PerApplicationTerminal.Holds Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Setup.productionSetup input.1 input.2}).toReal ≤
      ((HyperNovaHistoryLaw.law source initial).toOuterMeasure
        {sample | HyperNovaHistoryProbability.AdviceReturned sample}).toReal +
        ∑ j : Fin depth,
          (((visitedLaw source initial j.val).toOuterMeasure
              {visit | HyperNovaFirstFailure.MarkedHashCollision visit}).toReal +
            (((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
              {draw | MarkedSourceFailure draw}).toReal) := by
  have bound := HyperNovaFirstFailure.accepted_probability_le_first_failures source initial depth bounded
  have finiteTerm (j : Fin depth) :
      (visitedLaw source initial j.val).toOuterMeasure
          {visit | HyperNovaFirstFailure.MarkedHashCollision visit} +
        ((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
          {draw | MarkedSourceFailure draw} ≠ ∞ :=
    ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩
  have finiteSum := ENNReal.sum_ne_top.mpr (fun j (_ : j ∈ Finset.univ) => finiteTerm j)
  have realBound := ENNReal.toReal_mono
    (ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, finiteSum⟩) bound
  rw [ENNReal.toReal_add (event_ne_top _ _) finiteSum,
    ENNReal.toReal_sum (fun j (_ : j ∈ Finset.univ) => finiteTerm j)] at realBound
  have realTerm (j : Fin depth) := ENNReal.toReal_add
    (event_ne_top (visitedLaw source initial j.val)
      {visit | HyperNovaFirstFailure.MarkedHashCollision visit})
    (event_ne_top ((visitedLaw source initial j.val).bind (guardedDraw source))
      {draw | MarkedSourceFailure draw})
  simpa only [realTerm] using realBound

/-- HyperNova errata Assumption 1, plain-model part, at the visits of one
history: after a real NIFS acceptance at visit `j`, the source extractor
returns no checked source witness with probability at most `error j`. -/
def NifsKnowledgeSound (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (depth : Nat) (error : Fin depth → ℝ) : Prop :=
  ∀ j : Fin depth,
    (((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
      {draw | MarkedSourceFailure draw}).toReal ≤ error j

/-- History security under Assumption 1. The accepted terminal mass is at
most the returned-history mass plus, at each visit, the marked
hash-collision mass and the assumed NIFS knowledge error. -/
theorem history_probability_bound
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (depthBound : ∀ input ∈ initial.support, input.1.iteration ≤ depth)
    (error : Fin depth → ℝ) (knowledge : NifsKnowledgeSound source initial depth error) :
    (initial.toOuterMeasure {input |
      PerApplicationTerminal.Holds Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Setup.productionSetup input.1 input.2}).toReal ≤
      ((HyperNovaHistoryLaw.law source initial).toOuterMeasure
        {sample | HyperNovaHistoryProbability.AdviceReturned sample}).toReal +
        ∑ j : Fin depth,
          (((visitedLaw source initial j.val).toOuterMeasure
              {visit | HyperNovaFirstFailure.MarkedHashCollision visit}).toReal + error j) :=
  (first_failure_real_bound source initial depth depthBound).trans
    (add_le_add le_rfl (Finset.sum_le_sum fun j _ => add_le_add le_rfl (knowledge j)))

end NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity
