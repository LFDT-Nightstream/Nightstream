import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree.ConstraintPolynomial
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum

/-!
Complete per-phase degree contract for the independent Split-NC FE
polynomial.

Owns: composition of the source and sparse-CCS leaves into the fresh and
carried FE branches, separate row/lane degree bounds for the complete
polynomial, exact row-then-lane coordinate slicing, and closure of honest
Boolean-suffix SumCheck rounds.

Does not own: source MLE lemmas, sparse monomial construction, the generic
fixed-width coefficient carrier, transcript replay, prover-message
canonicalization, Poseidon2, Rust, R1CS, emitted rows, removals, or costs.

Emits constraints: no.

Authority boundary: row degree is derived from explicit sparse CCS exponents
and the quadratic carried branch. Lane degree is derived independently as
two. No declared CCS metadata, prover-supplied degree, Rust `d_sc`, or old
constraint total participates in these proofs.

| Child path | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `degree.source` | source `yRing` and padded-lane MLE affinity | no | `Degree.Source` |
| `degree.ccs` | sparse affine substitution and equality-gated CCS ceiling | no | `Degree.ConstraintPolynomial` |
| `degree.fresh` | fresh branch fits the syntax-derived row ceiling and lane degree one | no | this module |
| `degree.carried` | carried branch is quadratic in row and lane phases | no | this module |
| `degree.polynomial` | complete FE phase bounds | no | this module |
| `degree.sumcheck` | decoded slices and Boolean suffix sums preserve those bounds | no | this module |

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.degree.fresh.row` | equality-gated sparse CCS uses the row ceiling | derived | `freshAtPoint_row_bounded` |
| `nifs.pi_ccs.fe.degree.fresh.lane` | lane selector times a lane-constant fresh value is affine | derived | `freshAtPoint_lane_affine` |
| `nifs.pi_ccs.fe.degree.carried.row` | row selector times an affine carried value is quadratic | derived | `carriedAtPoint_row_quadratic` |
| `nifs.pi_ccs.fe.degree.carried.lane` | lane selector times an affine padded-lane value is quadratic | derived | `carriedAtPoint_lane_quadratic` |
| `nifs.pi_ccs.fe.degree.polynomial.row` | fresh plus carried fits `rowSumcheckDegreeBound` | derived | `qAtPoint_row_bounded` |
| `nifs.pi_ccs.fe.degree.polynomial.lane` | complete lane slice fits `laneSumcheckDegreeBound` | derived | `qAtPoint_lane_quadratic` |
| `nifs.pi_ccs.fe.degree.sumcheck.slice` | list decoder selects the correct phase theorem | derived | `sumcheckPolynomial_slice_bounded` |
| `nifs.pi_ccs.fe.degree.sumcheck.round` | honest suffix sums retain phase width | derived | `expectedRound_bounded` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree.Source
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree.ConstraintPolynomial

private theorem pointEquality_affine
    {variables : Nat}
    (beta : CubePoint K variables)
    (before after : List K)
    (length : before.length + 1 + after.length = variables) :
    Represents 1 fun point =>
      SumCheckTruthPath.pointEquality ops
        (cubeSlice before after length point) beta := by
  unfold SumCheckTruthPath.pointEquality
  apply pointEqualityCoordinates_affine
  rw [beta.dimension]
  exact length

private theorem liftedConstraintPolynomial_canonicalDegree
    {shape : SemanticShape}
    (data : Data shape) :
    (liftedConstraintPolynomial
      (PublicInput.ofSources data)).canonicalEqualityGatedDegreeBound =
      data.constraintPolynomial.canonicalEqualityGatedDegreeBound := by
  unfold liftedConstraintPolynomial PublicInput.ofSources
  exact
    ConstraintPolynomialLift.liftConstraintPolynomial_canonicalEqualityGatedDegreeBound
      K.embed data.constraintPolynomial

/-- Before the lane-constant selector is applied, the equality-gated fresh
CCS block has the exact canonical syntax-derived row ceiling. -/
private theorem freshRowCore_represents
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : Data shape)
    (coins : Coins shape domain)
    (before after : List K)
    (length : before.length + 1 + after.length = shape.rowVariables) :
    Represents
        data.constraintPolynomial.canonicalEqualityGatedDegreeBound
      fun point =>
        ops.mul
          (SumCheckTruthPath.pointEquality ops
            (cubeSlice before after length point) coins.betaR)
          (freshTermFromYRing (PublicInput.ofSources data) coins.gamma
            (sourceYRingAt data (cubeSlice before after length point))) := by
  let rowSelector : K -> K := fun point =>
    SumCheckTruthPath.pointEquality ops
      (cubeSlice before after length point) coins.betaR
  have rowSelectorRepresents : Represents 1 rowSelector :=
    pointEquality_affine coins.betaR before after length
  rcases polynomial_sum_exists
    (canonicalFinIndices shape.freshCount)
    (fun fresh =>
      TargetPolynomial.power ops.toOps coins.gamma fresh.val)
    (fun fresh point =>
      ops.mul (rowSelector point)
        (CCSResidualTable.evaluatePolynomial ops
          (liftedConstraintPolynomial (PublicInput.ofSources data))
          fun matrix =>
            sourceYRingAt data (cubeSlice before after length point)
              (Data.freshIndex fresh) matrix
              Phi81CoefficientKernel.constant))
    (by
      intro fresh _
      have represented := equalityGated_row_represents
        data (Data.freshIndex fresh)
        (liftedConstraintPolynomial (PublicInput.ofSources data))
        Phi81CoefficientKernel.constant
        rowSelector rowSelectorRepresents before after length
      rw [liftedConstraintPolynomial_canonicalDegree data] at represented
      exact represented) with
    ⟨sumPolynomial, sumRepresents⟩
  refine ⟨sumPolynomial, ?_⟩
  intro point
  rw [sumRepresents]
  letI : Std.Associative ops.mul := ⟨laws.mul_assoc⟩
  letI : Std.Commutative ops.mul := ⟨laws.mul_comm⟩
  calc
    FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.freshCount)
        (fun fresh =>
          ops.mul
            (TargetPolynomial.power ops.toOps coins.gamma fresh.val)
            (ops.mul (rowSelector point)
              (CCSResidualTable.evaluatePolynomial ops
                (liftedConstraintPolynomial (PublicInput.ofSources data))
                fun matrix =>
                  sourceYRingAt data
                    (cubeSlice before after length point)
                    (Data.freshIndex fresh) matrix
                    Phi81CoefficientKernel.constant))) =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.freshCount)
        (fun fresh =>
          ops.mul (rowSelector point)
            (ops.mul
              (TargetPolynomial.power ops.toOps coins.gamma fresh.val)
              (CCSResidualTable.evaluatePolynomial ops
                (liftedConstraintPolynomial (PublicInput.ofSources data))
                fun matrix =>
                  sourceYRingAt data
                    (cubeSlice before after length point)
                    (Data.freshIndex fresh) matrix
                    Phi81CoefficientKernel.constant))) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro fresh _
        ac_rfl
    _ = ops.mul (rowSelector point)
        (FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.freshCount)
          (fun fresh =>
            ops.mul
              (TargetPolynomial.power ops.toOps coins.gamma fresh.val)
              (CCSResidualTable.evaluatePolynomial ops
                (liftedConstraintPolynomial (PublicInput.ofSources data))
                fun matrix =>
                  sourceYRingAt data
                    (cubeSlice before after length point)
                    (Data.freshIndex fresh) matrix
                    Phi81CoefficientKernel.constant))) :=
      FiniteSumAlgebra.sumMap_mul_left ops laws _ _ _
    _ = ops.mul
        (SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) coins.betaR)
        (freshTermFromYRing (PublicInput.ofSources data) coins.gamma
          (sourceYRingAt data (cubeSlice before after length point))) := by
      rfl

/-- Each row-coordinate slice of the complete fresh branch fits the canonical
row ceiling derived from sparse syntax and the carried quadratic floor. -/
theorem freshAtPoint_row_bounded
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : Data shape)
    (coins : Coins shape domain)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = shape.rowVariables) :
    Represents (rowSumcheckDegreeBound (PublicInput.ofSources data))
      fun point =>
        InitialSum.freshAtPoint data coins {
          row := cubeSlice before after length point
          lane := lane } := by
  have core := freshRowCore_represents data coins before after length
  have canonicalLeRow :
      data.constraintPolynomial.canonicalEqualityGatedDegreeBound <=
        rowSumcheckDegreeBound (PublicInput.ofSources data) := by
    unfold rowSumcheckDegreeBound PublicInput.ofSources
    exact Nat.le_max_left _ _
  have widened := Represents.widen canonicalLeRow core
  let laneSelector := SumCheckTruthPath.pointEquality ops lane coins.betaA
  rcases Represents.scale laneSelector widened with
    ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  unfold InitialSum.freshAtPoint
  exact (laws.mul_assoc laneSelector
    (SumCheckTruthPath.pointEquality ops
      (cubeSlice before after length point) coins.betaR)
    (freshTermFromYRing (PublicInput.ofSources data) coins.gamma
      (sourceYRingAt data (cubeSlice before after length point)))).symm

/-- Each lane-coordinate slice of the fresh branch is affine: the CCS value
and row selector are constant during the lane phase. -/
theorem freshAtPoint_lane_affine
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : Data shape)
    (coins : Coins shape domain)
    (row : CubePoint K shape.rowVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    Represents 1 fun point =>
      InitialSum.freshAtPoint data coins {
        row := row
        lane := cubeSlice before after length point } := by
  have laneSelector := pointEquality_affine coins.betaA before after length
  let rowSelector := SumCheckTruthPath.pointEquality ops row coins.betaR
  let freshValue := freshTermFromYRing
    (PublicInput.ofSources data) coins.gamma (sourceYRingAt data row)
  have scaled := Represents.scale freshValue
    (Represents.scale rowSelector laneSelector)
  rcases scaled with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  unfold InitialSum.freshAtPoint
  letI : Std.Associative ops.mul := ⟨laws.mul_assoc⟩
  letI : Std.Commutative ops.mul := ⟨laws.mul_comm⟩
  change ops.mul freshValue
      (ops.mul rowSelector
        (SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) coins.betaA)) =
    ops.mul
      (ops.mul
        (SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) coins.betaA)
        rowSelector)
      freshValue
  ac_rfl

/-- An unweighted finite family of same-degree functions has a same-degree
representation under the protocol's explicit `sumMap`. -/
private theorem signedSum_represents
    {Index : Type}
    {degree : Nat}
    (indices : List Index)
    (value : Index -> K -> K)
    (represented : forall index, index ∈ indices ->
      Represents degree (value index)) :
    Represents degree fun point =>
      SignedJointIdentity.sumMap ops indices fun index => value index point := by
  rcases polynomial_sum_exists indices (fun _ => ops.one) value
    (by
      intro index member
      exact represented index member) with
    ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  change FiniteSumAlgebra.sumMap ops indices
      (fun index => ops.mul ops.one (value index point)) =
    FiniteSumAlgebra.sumMap ops indices (fun index => value index point)
  apply FiniteSumAlgebra.sumMap_congr
  intro index _
  rw [laws.one_mul]

/-- Gamma weighting is scalar multiplication, so it preserves a common
declared degree across an explicit finite family. -/
private theorem gammaSum_represents
    {Index : Type}
    {degree : Nat}
    (gamma : K)
    (indices : List Index)
    (exponent : Index -> Nat)
    (value : Index -> K -> K)
    (represented : forall index, index ∈ indices ->
      Represents degree (value index)) :
    Represents degree fun point =>
      SignedJointIdentity.sumMap ops indices fun index =>
        SignedJointIdentity.gammaTerm ops gamma (exponent index)
          (value index point) := by
  rcases polynomial_sum_exists indices
    (fun index => TargetPolynomial.power ops.toOps gamma (exponent index))
    value represented with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  rfl

/-- Fixing a lane point leaves the complete unshifted carried gamma block
affine in every row coordinate. -/
theorem carriedTermFromYRing_row_affine
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = shape.rowVariables) :
    Represents 1 fun point =>
      carriedTermFromYRing profile.laneCovers coins.gamma lane
        (sourceYRingAt data (cubeSlice before after length point)) := by
  unfold carriedTermFromYRing
  apply signedSum_represents
  intro matrix _
  apply gammaSum_represents coins.gamma
  intro running _
  exact paddedLaneEvaluation_row_affine profile.laneCovers data
    (Data.runningIndex running) matrix lane before after length

/-- Fixing a row point leaves the complete unshifted carried gamma block
affine in every lane coordinate. -/
theorem carriedTermFromYRing_lane_affine
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (row : CubePoint K shape.rowVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    Represents 1 fun point =>
      carriedTermFromYRing profile.laneCovers coins.gamma
        (cubeSlice before after length point) (sourceYRingAt data row) := by
  unfold carriedTermFromYRing
  apply signedSum_represents
  intro matrix _
  apply gammaSum_represents coins.gamma
  intro running _
  exact paddedLaneEvaluation_lane_affine profile.laneCovers
    (sourceYRingAt data row (Data.runningIndex running) matrix)
    before after length

/-- Each row-coordinate slice of the carried branch is quadratic, then
zero-extended to the complete FE row ceiling. -/
theorem carriedAtPoint_row_quadratic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = shape.rowVariables) :
    Represents (rowSumcheckDegreeBound (PublicInput.ofSources data))
      fun point =>
        InitialSum.carriedAtPoint profile data coins {
          row := cubeSlice before after length point
          lane := lane } := by
  have rowSelector := pointEquality_affine
    (PublicInput.ofSources data).priorPoint before after length
  have carried := carriedTermFromYRing_row_affine
    profile data coins lane before after length
  have product := Represents.mul rowSelector carried
  let laneSelector := SumCheckTruthPath.pointEquality ops lane coins.alpha
  let gammaShift := TargetPolynomial.power ops.toOps coins.gamma
    shape.sourceCount
  have scaled := Represents.scale (ops.mul laneSelector gammaShift) product
  have quadratic : Represents 2 fun point =>
      InitialSum.carriedAtPoint profile data coins {
        row := cubeSlice before after length point
        lane := lane } := by
    rcases scaled with ⟨polynomial, represents⟩
    refine ⟨polynomial, ?_⟩
    intro point
    rw [represents]
    unfold InitialSum.carriedAtPoint SignedJointIdentity.gammaTerm
    change ops.mul (ops.mul laneSelector gammaShift)
        (ops.mul
          (SumCheckTruthPath.pointEquality ops
            (cubeSlice before after length point)
            (PublicInput.ofSources data).priorPoint)
          (carriedTermFromYRing profile.laneCovers coins.gamma lane
            (sourceYRingAt data
              (cubeSlice before after length point)))) =
      ops.mul
        (ops.mul laneSelector
          (SumCheckTruthPath.pointEquality ops
            (cubeSlice before after length point)
            (PublicInput.ofSources data).priorPoint))
        (ops.mul gammaShift
          (carriedTermFromYRing profile.laneCovers coins.gamma lane
            (sourceYRingAt data
              (cubeSlice before after length point))))
    letI : Std.Associative ops.mul := ⟨laws.mul_assoc⟩
    letI : Std.Commutative ops.mul := ⟨laws.mul_comm⟩
    ac_rfl
  apply Represents.widen _ quadratic
  unfold rowSumcheckDegreeBound PublicInput.ofSources
  exact Nat.le_max_right _ _

/-- Each lane-coordinate slice of the carried branch has the exact quadratic
ceiling independently of the sparse CCS syntax. -/
theorem carriedAtPoint_lane_quadratic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (row : CubePoint K shape.rowVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    Represents laneSumcheckDegreeBound fun point =>
      InitialSum.carriedAtPoint profile data coins {
        row := row
        lane := cubeSlice before after length point } := by
  have laneSelector := pointEquality_affine coins.alpha before after length
  have carried := carriedTermFromYRing_lane_affine
    profile data coins row before after length
  have product := Represents.mul laneSelector carried
  let rowSelector := SumCheckTruthPath.pointEquality ops row
    (PublicInput.ofSources data).priorPoint
  let gammaShift := TargetPolynomial.power ops.toOps coins.gamma
    shape.sourceCount
  have scaled := Represents.scale (ops.mul rowSelector gammaShift) product
  rcases scaled with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  change polynomial.evaluate ops.toOps point =
    InitialSum.carriedAtPoint profile data coins {
      row := row
      lane := cubeSlice before after length point }
  rw [represents]
  unfold InitialSum.carriedAtPoint SignedJointIdentity.gammaTerm
  change ops.mul (ops.mul rowSelector gammaShift)
      (ops.mul
        (SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) coins.alpha)
        (carriedTermFromYRing profile.laneCovers coins.gamma
          (cubeSlice before after length point) (sourceYRingAt data row))) =
    ops.mul
      (ops.mul
        (SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) coins.alpha)
        rowSelector)
      (ops.mul gammaShift
        (carriedTermFromYRing profile.laneCovers coins.gamma
          (cubeSlice before after length point) (sourceYRingAt data row)))
  letI : Std.Associative ops.mul := ⟨laws.mul_assoc⟩
  letI : Std.Commutative ops.mul := ⟨laws.mul_comm⟩
  ac_rfl

/-- The complete FE polynomial fits the syntax-derived row ceiling. -/
theorem qAtPoint_row_bounded
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = shape.rowVariables) :
    Represents (rowSumcheckDegreeBound (PublicInput.ofSources data))
      fun point => qAtPoint profile data coins {
        row := cubeSlice before after length point
        lane := lane } := by
  have fresh := freshAtPoint_row_bounded data coins lane
    before after length
  have carried := carriedAtPoint_row_quadratic profile data coins lane
    before after length
  rcases Represents.add fresh carried with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  exact (InitialSum.qAtPoint_eq_fresh_add_carried profile data coins _).symm

/-- The complete FE polynomial has an exact quadratic lane ceiling. -/
theorem qAtPoint_lane_quadratic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (row : CubePoint K shape.rowVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    Represents laneSumcheckDegreeBound fun point =>
      qAtPoint profile data coins {
        row := row
        lane := cubeSlice before after length point } := by
  have freshAffine :=
    freshAtPoint_lane_affine data coins row before after length
  have fresh : Represents laneSumcheckDegreeBound fun point =>
      InitialSum.freshAtPoint data coins {
        row := row
        lane := cubeSlice before after length point } :=
    Represents.widen (by
      unfold laneSumcheckDegreeBound
      omega)
      freshAffine
  have carried := carriedAtPoint_lane_quadratic profile data coins row
    before after length
  rcases Represents.add fresh carried with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  exact (InitialSum.qAtPoint_eq_fresh_add_carried profile data coins _).symm

private theorem cubePoint_eq_of_coordinates_eq
    {variables : Nat}
    (left right : CubePoint K variables)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem ofCoordinates_eq_rowSlice
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (before after : List K)
    (beforeRow : before.length < shape.rowVariables)
    (totalLength : before.length + 1 + after.length =
      shape.rowVariables + domain.laneVariables)
    (point : K) :
    let rowAfter :=
      after.take (shape.rowVariables - before.length - 1)
    let laneCoordinates :=
      after.drop (shape.rowVariables - before.length - 1)
    let rowLength : before.length + 1 + rowAfter.length =
        shape.rowVariables := by
      dsimp only [rowAfter]
      rw [List.length_take]
      omega
    let laneLength : laneCoordinates.length = domain.laneVariables := by
      dsimp only [laneCoordinates]
      rw [List.length_drop]
      omega
    Point.ofCoordinates (before ++ point :: after) (by simp; omega) = {
      row := cubeSlice before rowAfter rowLength point
      lane := { coordinates := laneCoordinates, dimension := laneLength } } := by
  dsimp only
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates cubeSlice
    simp only
    rw [List.take_append]
    rw [List.take_of_length_le (Nat.le_of_lt beforeRow)]
    have remainingSucc :
        shape.rowVariables - before.length =
          (shape.rowVariables - before.length - 1) + 1 := by
      omega
    rw [remainingSucc]
    rfl
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates
    simp only
    rw [List.drop_append]
    rw [List.drop_eq_nil_of_le (Nat.le_of_lt beforeRow)]
    have remainingSucc :
        shape.rowVariables - before.length =
          (shape.rowVariables - before.length - 1) + 1 := by
      omega
    rw [remainingSucc]
    rfl

private theorem ofCoordinates_eq_laneSlice
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (before after : List K)
    (rowBefore : shape.rowVariables <= before.length)
    (totalLength : before.length + 1 + after.length =
      shape.rowVariables + domain.laneVariables)
    (point : K) :
    let rowCoordinates := before.take shape.rowVariables
    let laneBefore := before.drop shape.rowVariables
    let rowLength : rowCoordinates.length = shape.rowVariables := by
      dsimp only [rowCoordinates]
      rw [List.length_take]
      omega
    let laneLength : laneBefore.length + 1 + after.length =
        domain.laneVariables := by
      dsimp only [laneBefore]
      rw [List.length_drop]
      omega
    Point.ofCoordinates (before ++ point :: after) (by simp; omega) = {
      row := { coordinates := rowCoordinates, dimension := rowLength }
      lane := cubeSlice laneBefore after laneLength point } := by
  dsimp only
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates
    simp only
    exact List.take_append_of_le_length rowBefore
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates cubeSlice
    simp only
    exact List.drop_append_of_le_length rowBefore

private theorem sumcheckPolynomial_eq_qAtPoint_of_length
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (coordinates : List K)
    (length : coordinates.length =
      shape.rowVariables + domain.laneVariables) :
    InitialSum.sumcheckPolynomial profile data coins coordinates =
      qAtPoint profile data coins
        (Point.ofCoordinates coordinates length) := by
  unfold InitialSum.sumcheckPolynomial polynomial
  rw [dif_pos length]
  rfl

/-- An exact-arity slice before the row/lane boundary has the
syntax-derived row representation. -/
theorem sumcheckPolynomial_row_slice_bounded
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (before after : List K)
    (beforeRow : before.length < shape.rowVariables)
    (length : before.length + 1 + after.length =
      shape.rowVariables + domain.laneVariables) :
    Represents (rowSumcheckDegreeBound (PublicInput.ofSources data))
      fun point => InitialSum.sumcheckPolynomial profile data coins
        (before ++ point :: after) := by
  let rowAfter := after.take (shape.rowVariables - before.length - 1)
  let laneCoordinates :=
    after.drop (shape.rowVariables - before.length - 1)
  have rowLength : before.length + 1 + rowAfter.length =
      shape.rowVariables := by
    dsimp only [rowAfter]
    rw [List.length_take]
    omega
  have laneLength : laneCoordinates.length = domain.laneVariables := by
    dsimp only [laneCoordinates]
    rw [List.length_drop]
    omega
  let lane : CubePoint K domain.laneVariables := {
    coordinates := laneCoordinates
    dimension := laneLength }
  rcases qAtPoint_row_bounded profile data coins lane
    before rowAfter rowLength with ⟨slice, sliceRepresents⟩
  refine ⟨slice, ?_⟩
  intro point
  change slice.evaluate ops.toOps point =
    InitialSum.sumcheckPolynomial profile data coins
      (before ++ point :: after)
  rw [sumcheckPolynomial_eq_qAtPoint_of_length
    profile data coins (before ++ point :: after) (by
      simp only [List.length_append, List.length_cons]
      omega)]
  rw [ofCoordinates_eq_rowSlice before after beforeRow length]
  exact sliceRepresents point

/-- An exact-arity slice at or after the row/lane boundary has the independent
quadratic lane representation. -/
theorem sumcheckPolynomial_lane_slice_quadratic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (before after : List K)
    (rowBefore : shape.rowVariables <= before.length)
    (length : before.length + 1 + after.length =
      shape.rowVariables + domain.laneVariables) :
    Represents laneSumcheckDegreeBound fun point =>
      InitialSum.sumcheckPolynomial profile data coins
        (before ++ point :: after) := by
  let rowCoordinates := before.take shape.rowVariables
  let laneBefore := before.drop shape.rowVariables
  have rowLength : rowCoordinates.length = shape.rowVariables := by
    dsimp only [rowCoordinates]
    rw [List.length_take]
    omega
  have laneLength : laneBefore.length + 1 + after.length =
      domain.laneVariables := by
    dsimp only [laneBefore]
    rw [List.length_drop]
    omega
  let row : CubePoint K shape.rowVariables := {
    coordinates := rowCoordinates
    dimension := rowLength }
  rcases qAtPoint_lane_quadratic profile data coins row
    laneBefore after laneLength with ⟨slice, sliceRepresents⟩
  refine ⟨slice, ?_⟩
  intro point
  change slice.evaluate ops.toOps point =
    InitialSum.sumcheckPolynomial profile data coins
      (before ++ point :: after)
  rw [sumcheckPolynomial_eq_qAtPoint_of_length
    profile data coins (before ++ point :: after) (by
      simp only [List.length_append, List.length_cons]
      omega)]
  rw [ofCoordinates_eq_laneSlice before after rowBefore length]
  exact sliceRepresents point

/-- Every honest FE row-phase SumCheck polynomial retains the syntax-derived
row width after summing its Boolean suffix. -/
theorem expectedRowRound_bounded
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (fixed : List K)
    (remaining : Nat)
    (rowPhase : fixed.length < shape.rowVariables)
    (length : fixed.length + 1 + remaining =
      shape.rowVariables + domain.laneVariables) :
    Represents (rowSumcheckDegreeBound (PublicInput.ofSources data))
      fun point =>
        SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps
          (InitialSum.sumcheckPolynomial profile data coins)
          (fixed ++ [point]) remaining := by
  apply DegreeSupport.sumCompletions_represents
  intro vertex
  have suffixLength :
      (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex).length =
        remaining :=
    SumCheckTruthPath.VertexEncoding.fieldCoordinates_length ops vertex
  rcases sumcheckPolynomial_row_slice_bounded profile data coins fixed
    (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex)
    rowPhase (by
      rw [suffixLength]
      exact length) with ⟨slice, sliceRepresents⟩
  refine ⟨slice, ?_⟩
  intro point
  simpa only [List.append_assoc, List.singleton_append] using
    sliceRepresents point

/-- Every honest FE lane-phase SumCheck polynomial retains exactly three
coefficient slots after summing its Boolean suffix. -/
theorem expectedLaneRound_quadratic
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (fixed : List K)
    (remaining : Nat)
    (lanePhase : shape.rowVariables <= fixed.length)
    (length : fixed.length + 1 + remaining =
      shape.rowVariables + domain.laneVariables) :
    Represents laneSumcheckDegreeBound fun point =>
      SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps
        (InitialSum.sumcheckPolynomial profile data coins)
        (fixed ++ [point]) remaining := by
  apply DegreeSupport.sumCompletions_represents
  intro vertex
  have suffixLength :
      (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex).length =
        remaining :=
    SumCheckTruthPath.VertexEncoding.fieldCoordinates_length ops vertex
  rcases sumcheckPolynomial_lane_slice_quadratic profile data coins fixed
    (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex)
    lanePhase (by
      rw [suffixLength]
      exact length) with ⟨slice, sliceRepresents⟩
  refine ⟨slice, ?_⟩
  intro point
  simpa only [List.append_assoc, List.singleton_append] using
    sliceRepresents point

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree
