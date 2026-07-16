import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule

/-!
Coefficient-level strong-set obligations for production-shaped `Pi_RLC`.

Protocol: SuperNeo `Pi_RLC` inside the candidate noninteractive NIFS.
Phase: assembly and validation of one sampled scalar challenge.
Constraint family: centered coefficient range, pairwise difference range, and
the paper's expansion-factor arithmetic.

Owns: the exact 54-coordinate scalar carrier; its centered integer semantics;
the proof that every sampled coefficient lies in `[-2, 2]`; the proof that
every pairwise coefficient difference lies in `[-4, 4]`; nonzero-coordinate
separation for distinct scalar vectors; the minimal threshold `5` sufficient
for the coefficient-level infinity-norm premise; and `2 * 54 * 2 = 216`.

Does not own: embedding centered integers into Goldilocks, quotient-ring
equality or subtraction, low-norm invertibility in `R_F`, a proof that the
paper's analytic `b_inv` exceeds `4`, rotation-matrix materialization,
distribution or bias, Poseidon2, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: the scalar is the complete verifier-sampled coefficient
vector, not a prover-supplied digest or a Rust matrix. This file proves the
coefficient premises needed by Definition 17. It deliberately does not call
the result a ring strong set until a later theorem proves that the concrete
ring embedding preserves these values and instantiates Theorem 8.

| Protocol | Phase | Mathematical object | Exact obligation |
|---|---|---|---|
| `Pi_RLC` | scalar carrier | `Scalar` / `assembleCoefficients` | one scalar is exactly 54 sampled five-symbol coefficients |
| `Pi_RLC` | coefficient validity | `everyScalarValid` | every centered coefficient lies in `[-2, 2]` |
| `Pi_RLC` | pairwise difference | `coefficientDifference_bounds` | every coordinate difference lies in `[-4, 4]` |
| `Pi_RLC` | separation | `distinct_has_nonzero_difference` | distinct vectors differ at a nonzero centered coordinate |
| `Pi_RLC` | strong-set premise | `coefficientStrongPrecondition` | distinct vectors have nonzero difference and pointwise norm below `5` |
| `Pi_RLC` | sampler lift | `sampledChallenge_valid` | every successfully sampled scalar satisfies the coefficient predicate |
| `Pi_RLC` | expansion | `expansionFactor_value` | Theorem-9 arithmetic gives the bound `216` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

open ProductionAlphabet
open ProductionSchedule

universe uState

/-- One production scalar before its concrete quotient-ring embedding. -/
abbrev Scalar := Fin coefficientCount -> Coefficient

/-- Scalar assembly at the independent coefficient-vector layer is exact. -/
def assembleCoefficients (coefficients : Scalar) : Scalar := coefficients

@[simp] theorem assembleCoefficients_eq (coefficients : Scalar) :
    assembleCoefficients coefficients = coefficients := by
  rfl

/-- Semantic validity of one centered coefficient. -/
def CoefficientValid (coefficient : Coefficient) : Prop :=
  (-2 : Int) <= centeredValue coefficient /\
    centeredValue coefficient <= 2

/-- Semantic coefficient-level membership of one scalar. -/
def ScalarValid (scalar : Scalar) : Prop :=
  forall position, CoefficientValid (scalar position)

theorem everyCoefficientValid (coefficient : Coefficient) :
    CoefficientValid coefficient :=
  centeredValue_bounds coefficient

theorem everyScalarValid (scalar : Scalar) : ScalarValid scalar := by
  intro position
  exact everyCoefficientValid (scalar position)

/-- Centered integer difference at one scalar coordinate. -/
def coefficientDifference (left right : Scalar)
    (position : Fin coefficientCount) : Int :=
  centeredValue (left position) - centeredValue (right position)

/-- The five-symbol alphabet alone fixes the exact difference interval. -/
theorem coefficientDifference_bounds
    (left right : Scalar) (position : Fin coefficientCount) :
    (-4 : Int) <= coefficientDifference left right position /\
      coefficientDifference left right position <= 4 := by
  have leftBounds := centeredValue_bounds (left position)
  have rightBounds := centeredValue_bounds (right position)
  unfold coefficientDifference
  omega

/-- The smallest natural threshold strictly above every possible coefficient
difference magnitude. No approximate production `b_inv` value is needed for
this arithmetic fact. -/
def requiredInvertibilityThreshold : Nat := 5

/-- Pointwise form of `||left - right||_infinity < threshold`. Keeping the
definition pointwise avoids introducing an implementation-shaped maximum. -/
def DifferenceBelow (threshold : Nat) (left right : Scalar) : Prop :=
  forall position,
    -(threshold : Int) < coefficientDifference left right position /\
      coefficientDifference left right position < (threshold : Int)

theorem everyDifference_below_requiredThreshold (left right : Scalar) :
    DifferenceBelow requiredInvertibilityThreshold left right := by
  intro position
  have bounds := coefficientDifference_bounds left right position
  change (-5 : Int) < coefficientDifference left right position /\
    coefficientDifference left right position < 5
  omega

theorem centeredValue_injective : Function.Injective centeredValue := by
  intro left right equalValues
  apply Fin.ext
  unfold centeredValue at equalValues
  omega

/-- Extensional inequality of scalar vectors exposes a nonzero centered
coefficient difference. -/
theorem distinct_has_nonzero_difference
    {left right : Scalar} (different : left ≠ right) :
    exists position, coefficientDifference left right position ≠ 0 := by
  have differsAt : exists position, left position ≠ right position := by
    exact Classical.byContradiction fun noDifference =>
      different <| funext fun position =>
        Classical.byContradiction fun atPosition =>
          noDifference <| Exists.intro position atPosition
  obtain ⟨position, atPosition⟩ := differsAt
  refine ⟨position, ?_⟩
  intro zeroDifference
  apply atPosition
  apply centeredValue_injective
  unfold coefficientDifference at zeroDifference
  omega

/-- Exact coefficient-level part of Definition 17. The remaining theorem is
the concrete ring lift: centered embedding must preserve this difference and
Theorem 8 must prove every such nonzero ring element invertible. -/
def StrongCoefficientPrecondition : Prop :=
  forall {left right : Scalar}, left ≠ right ->
    DifferenceBelow requiredInvertibilityThreshold left right /\
      exists position, coefficientDifference left right position ≠ 0

theorem coefficientStrongPrecondition : StrongCoefficientPrecondition := by
  intro left right different
  exact ⟨everyDifference_below_requiredThreshold left right,
    distinct_has_nonzero_difference different⟩

/-- `StrongSetLaw` specialized only to coefficient-vector membership. This is
not yet the paper's quotient-ring `challengeValid` predicate. -/
def coefficientVectorLaw
    {State : Type uState}
    (machine : Machine State) :
    StrongSetLaw (specification machine assembleCoefficients) ScalarValid where
  validCoefficient := CoefficientValid
  accepted_coefficient_valid := by
    intro candidate _accepted
    exact everyCoefficientValid (verifier.symbol candidate)
  assembled_valid := by
    intro coefficients valid
    exact valid

/-- Every successful transcript-chained sample satisfies the independently
defined coefficient-vector predicate. -/
theorem sampledChallenge_valid
    {State : Type uState}
    {challengeCount : Nat}
    (machine : Machine State)
    (initial : State)
    (batch : BatchExecution
      (specification machine assembleCoefficients)
      challengeCount candidateBound initial)
    (coordinate : Fin challengeCount) :
    ScalarValid (challenge batch coordinate) :=
  challenge_valid (coefficientVectorLaw machine) batch coordinate

/-- Maximum centered coefficient norm in the production alphabet. -/
def maximumCoefficientNorm : Nat := 2

/-- Theorem-9 arithmetic expression `2 * phi(81) * maxNorm`, using the
independently fixed 54-coordinate carrier. Proving the analytic theorem and
its concrete ring hypotheses remains outside this module. -/
def expansionFactor : Nat :=
  2 * coefficientCount * maximumCoefficientNorm

theorem expansionFactor_value : expansionFactor = 216 := by
  decide

end Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet
