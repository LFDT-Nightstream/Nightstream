/-!
Kernel obstruction to applying a fixed-witness probability bound to an
outcome-adaptive existential witness.

Owns: one two-outcome countermodel in which every independently fixed witness
has one bad outcome, while choosing the witness after seeing the outcome makes
the existential bad event certain.

Does not own: `Pi_CCS`, a random oracle, a SumCheck or root bound, an
asymptotic statement, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

SuperNeo Appendix D.4 fixes the first successful run's witness before drawing
the fresh second-run verifier coins.  A theorem for each such fixed witness
does not automatically bound a predicate that existentially chooses a witness
inside the second-run outcome.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.Frozen.NonInteractiveAdaptiveWitnessObstruction

/-- The complete two-point verifier-coin space. -/
def outcomes : List Bool := [false, true]

/-- One bad outcome for each independently fixed witness. -/
def fixedWitnessBad (outcome witness : Bool) : Bool :=
  outcome == witness

/-- The outcome-adaptive event chooses either witness after observing the
outcome. -/
def adaptiveWitnessBad (outcome : Bool) : Bool :=
  fixedWitnessBad outcome false || fixedWitnessBad outcome true

def fixedWitnessBadCount (witness : Bool) : Nat :=
  (outcomes.filter fun outcome => fixedWitnessBad outcome witness).length

def adaptiveWitnessBadCount : Nat :=
  (outcomes.filter adaptiveWitnessBad).length

/-- The adaptive Boolean event is exactly existential witness choice. -/
theorem adaptiveWitnessBad_iff_exists (outcome : Bool) :
    adaptiveWitnessBad outcome = true <->
      exists witness : Bool, fixedWitnessBad outcome witness = true := by
  cases outcome <;> decide

/-- Every witness fixed before the outcome has probability numerator one over
the common denominator two. -/
theorem fixedWitnessBadCount_eq_one (witness : Bool) :
    fixedWitnessBadCount witness = 1 := by
  cases witness <;> decide

/-- Outcome-adaptive witness choice covers both outcomes. -/
theorem adaptiveWitnessBadCount_eq_two :
    adaptiveWitnessBadCount = 2 := by
  decide

/-- Headline countermodel: a common one-outcome bound holds for every fixed
witness but fails for the existential event over the same sample space. -/
theorem fixed_witness_bound_does_not_bound_adaptive_existential :
    (forall witness, fixedWitnessBadCount witness <= 1) /\
      adaptiveWitnessBadCount = 2 /\
      1 < adaptiveWitnessBadCount := by
  refine ⟨?_, adaptiveWitnessBadCount_eq_two, ?_⟩
  · intro witness
    rw [fixedWitnessBadCount_eq_one]
    decide
  · rw [adaptiveWitnessBadCount_eq_two]
    decide

end Nightstream.Protocol.FPrime.Frozen.NonInteractiveAdaptiveWitnessObstruction
