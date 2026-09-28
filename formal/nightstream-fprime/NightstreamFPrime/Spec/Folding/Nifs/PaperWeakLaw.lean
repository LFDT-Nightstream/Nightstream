import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalLaw
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalProgram
import NightstreamFPrime.Spec.Folding.Nifs.SuffixCoinCoupling

/-!
The finite endpoint law of the existing coordinate extractor. Rejected base
calls have their own atom; accepted bases retain every first-hit endpoint,
including repeated challenges that the terminal program rejects. The PMF is
derived from the stopped-search law and is a probability description, not an
implementation that samples a table of all possible verifier coins.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperWeakLaw

open scoped BigOperators
open NightstreamFPrime.Spec.Folding
open PiRLC.CoordinateRetry PiRLC.CoordinateOracle PiRLC.CoordinateOracleStar
open PiRLC.CoordinateTerminalLaw PiRLC.CoordinateTerminalProgram
open PiRLC.PaperForkExtractionWork

/-- `none` is the rejected-base exit; a present endpoint retains the original
base vector, base response and all independently stopped coordinate responses. -/
abbrev Endpoint (Index Challenge Assignment : Type*) :=
  Option ((Index → Challenge) × Option Assignment × (Index → (Challenge × Option Assignment)))

variable {Index Challenge Assignment : Type*} [Fintype Index] [DecidableEq Index]
  [Fintype Challenge] [Nonempty Challenge] [Fintype Assignment]

/-- The base vector is uniform. The conditional endpoint mass is the existing
sum over all rejected-prefix traces, not an assumed fork-success rate. -/
noncomputable def mass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) : Endpoint Index Challenge Assignment → ℝ
  | none => 1 - (line oracle check).rate
  | some (vector, initial, outputs) =>
      endpointMass oracle check vector initial outputs / Fintype.card (Index → Challenge)

theorem mass_nonnegative (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (endpoint : Endpoint Index Challenge Assignment) : 0 ≤ mass oracle check endpoint := by
  cases endpoint with
  | none => exact sub_nonneg.mpr (Line.rate_le_one _)
  | some endpoint =>
      exact div_nonneg (endpointMass_nonnegative oracle check endpoint.1 endpoint.2.1 endpoint.2.2)
        (Nat.cast_nonneg _)

private theorem present_mass_total (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) :
    (∑ endpoint : (Index → Challenge) × Option Assignment × (Index → (Challenge × Option Assignment)),
      mass oracle check (some endpoint)) = (line oracle check).rate := by
  calc
    _ = ∑ vector, (∑ initial, ∑ outputs,
        endpointMass oracle check vector initial outputs) / Fintype.card (Index → Challenge) := by
      simp only [Fintype.sum_prod_type, mass, Finset.sum_div]
    _ = ∑ vector, (line oracle check).acceptance vector / Fintype.card (Index → Challenge) := by
      simp_rw [endpointMass_total]
    _ = _ := rfl

/-- No abort or repeated-challenge endpoint is removed during normalization. -/
theorem mass_total (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) :
    ∑ endpoint, mass oracle check endpoint = 1 := by
  rw [Fintype.sum_option, present_mass_total]
  change (1 - (line oracle check).rate) + (line oracle check).rate = 1
  ring

/-- Normalized finite law of the actual coordinate-search endpoints. -/
noncomputable def law (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) : PMF (Endpoint Index Challenge Assignment) :=
  PMF.ofFintype (fun endpoint => ENNReal.ofReal (mass oracle check endpoint)) (by
    rw [← ENNReal.ofReal_sum_of_nonneg (fun endpoint _ => mass_nonnegative oracle check endpoint),
      mass_total]
    simp)

theorem law_toReal (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (endpoint : Endpoint Index Challenge Assignment) :
    (law oracle check endpoint).toReal = mass oracle check endpoint := by
  exact ENNReal.toReal_ofReal (mass_nonnegative oracle check endpoint)

/-- Positive law mass gives the literal positive endpoint premise used by the
actual-return extractor theorem. -/
theorem law_some_positive_iff (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (initial : Option Assignment)
    (outputs : Index → (Challenge × Option Assignment)) :
    0 < (law oracle check (some (vector, initial, outputs))).toReal ↔
      0 < endpointMass oracle check vector initial outputs := by
  rw [law_toReal]
  exact div_pos_iff_of_pos_right (by exact_mod_cast Fintype.card_pos (α := Index → Challenge))

/-- Every observable uses the same rejected-base atom and endpoint mass. -/
theorem law_mean (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (value : Endpoint Index Challenge Assignment → ℝ) :
    (∑ endpoint, (law oracle check endpoint).toReal * value endpoint) =
      (1 - (line oracle check).rate) * value none +
        𝔼 vector, ∑ initial, ∑ outputs,
          endpointMass oracle check vector initial outputs * value (some (vector, initial, outputs)) := by
  simp only [law_toReal, Fintype.sum_option, Fintype.sum_prod_type, mass,
    Fintype.expect_eq_sum_div_card, Finset.sum_div, add_div, div_mul_eq_mul_div]

section Terminal

variable {Scalar : Type*} {member : Scalar → Prop} {count : Nat}
  [DecidableEq Scalar] [Fintype {scalar // member scalar}] [Nonempty {scalar // member scalar}]

/-- Apply the existing inverse-difference program to its observed endpoint.
The rejected-base exit has the same one-step terminal cost as its work law. -/
def terminalResult (program : Primitives Scalar Assignment) :
    Endpoint (Fin count) {scalar // member scalar} Assignment → Result (Option (List Assignment))
  | none => ⟨none, 1⟩
  | some (vector, initial, outputs) => finish program vector initial outputs

def terminalValue (program : Primitives Scalar Assignment)
    (endpoint : Endpoint (Fin count) {scalar // member scalar} Assignment) : Option (List Assignment) :=
  (terminalResult program endpoint).value

/-- Distribution of the literal list returned by the terminal program. -/
noncomputable def returnLaw
    (oracle : Oracle (Fin count) {scalar // member scalar} Assignment)
    (check : (Fin count → {scalar // member scalar}) → Assignment → Bool)
    (program : Primitives Scalar Assignment) : PMF (Option (List Assignment)) :=
  (law oracle check).map (terminalValue program)

/-- Successful-return mass under the induced output distribution. -/
noncomputable def successProbability
    (oracle : Oracle (Fin count) {scalar // member scalar} Assignment)
    (check : (Fin count → {scalar // member scalar}) → Assignment → Bool)
    (program : Primitives Scalar Assignment) : ℝ :=
  ((returnLaw oracle check program).map Option.isSome true).toReal

theorem successProbability_eq_returningProbability
    (oracle : Oracle (Fin count) {scalar // member scalar} Assignment)
    (check : (Fin count → {scalar // member scalar}) → Assignment → Bool)
    (program : Primitives Scalar Assignment) :
    successProbability oracle check program = returningProbability oracle check
      (fun vector initial outputs => (finish program vector initial outputs).value.isSome) := by
  have mean := SuffixCoinCoupling.mean_map (law oracle check)
    (fun endpoint => (terminalValue program endpoint).isSome)
    (fun success : Bool => if success then (1 : ℝ) else 0)
  have mapped :
      (∑ endpoint, (law oracle check endpoint).toReal *
        (if (terminalValue program endpoint).isSome then (1 : ℝ) else 0)) =
        successProbability oracle check program := by
    simpa [successProbability, returnLaw, PMF.map_comp, Function.comp_def] using mean
  rw [← mapped, law_mean]
  simp only [terminalValue, terminalResult, Option.isSome_none, Bool.false_eq_true,
    ↓reduceIte, mul_zero, zero_add]
  unfold returningProbability
  apply Finset.expect_congr rfl
  intro vector _
  apply Finset.sum_congr rfl
  intro initial _
  apply Finset.sum_congr rfl
  intro outputs _
  split_ifs <;> simp

/-- The same endpoint PMF gives exactly the existing terminal-work mean.
Query work, including rejected retries, remains in `expectedQueryWork`. -/
theorem terminal_work_mean
    (oracle : Oracle (Fin count) {scalar // member scalar} Assignment)
    (check : (Fin count → {scalar // member scalar}) → Assignment → Bool)
    (program : Primitives Scalar Assignment) :
    (∑ endpoint, (law oracle check endpoint).toReal * ((terminalResult program endpoint).work : ℝ)) =
      expectedTerminalWork oracle check (fun vector initial outputs =>
        (finish program vector initial outputs).work) := by
  rw [law_mean]
  simp only [terminalResult, Nat.cast_one, mul_one, expectedTerminalWork]

end Terminal

end NightstreamFPrime.Spec.Folding.Nifs.PaperWeakLaw
