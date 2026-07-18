import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedParentProjection

/-!
Contract: define the compact delayed-parent projection residual that can be
batched into the production-shaped Π_CCS NC SumCheck.

Assurance tier: model-level.

Owns: the raw-child projection at the producer's univariate `beta`, its
column/lane lift by `eq(s, s_old) * B_beta(alpha)`, Boolean-cube normalization,
and the ordinary terminal-point formula that reuses the current raw `z`
evaluation.

Does not own: transcript timing or domain separation, Π_RLC `out_eval` row
refinement, independence of parent witnesses and Π_DEC child recomposition
before `beta`, the compact one-point projection identity, accumulator/state
binding, concrete SumCheck rows, or any row-removal permission.

Emits constraints: no.

Authority boundary: `rawChildren` is independent assignment authority.
`batchWeight`, `producerBeta`, and `oldS` are explicit verifier inputs; a
production refinement must prove when they are sampled and how they are bound.
A carried scalar evaluation or digest is not authority by itself.

| Stage path | Mathematical obligation | Explicit assumptions | Lean owner | Permits row removal? |
|---|---|---|---|---|
| `pi_ccs.nc.delayed_residual.raw_column` | radix-combine child diagonals, then evaluate lanes at producer `beta` | all coordinates fit the selected domains | `rawProjectionAtProducerBeta` | no |
| `pi_ccs.nc.delayed_residual.power_selector` | `B_beta(bit(lane)) = beta^lane` | lane is in the Boolean domain | `betaPowerSelector_cubePoint` | no |
| `pi_ccs.nc.delayed_residual.lift` | multiply raw `z(s,alpha)` by independent `batchWeight`, `eq(s,s_old)`, and `B_beta(alpha)` | `oldS.length = ellM` | `delayedResidualPolynomial` | no |
| `pi_ccs.nc.delayed_residual.source_coverage` | every raw child coordinate lies inside the selected column domain | `DelayedResidualShape.childrenFit` | `rawChildCoordinate_lt_columnDomain` | no |
| `pi_ccs.nc.delayed_residual.cube_normalization` | full Boolean sum equals the weighted active old-point projection | `DelayedResidualShape` | `delayedResidualCubeSum_eq_weightedCompactOldProjection` | no |
| `pi_ccs.nc.delayed_residual.terminal` | final SumCheck point reuses `radixCombinedRawZ(s',alpha')` | `DelayedResidualShape` and `TerminalPointShape` | `delayedResidualPolynomial_eq_terminalRhs` | no |

Open obligations: domain-separate and sample `batchWeight` independently;
bind `producerBeta` only after the compared parent and child data are fixed;
refine concrete terminal-list lengths and raw-child matrix coverage to
`TerminalPointShape` and `AssignmentsFitColumnDomain`;
instantiate the generic limb theorem on the claimed parent coefficient vector,
then connect Π_RLC's two retained `YZColLimb` parent-output evaluations to
that parent instance in `DelayedResidual.ProjectionIdentity`; separately
refine the generated padding-zero pins for the production input and parent
vectors; derive
parent-witness/child recomposition independently before `beta`; bind the
compact handle into the accumulator; and retain all current rows until those
links are proved.
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual

open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection
open Nightstream.SuperNeo.Concrete

private theorem fadd_assoc (left middle right : F) :
    (left + middle) + right = left + (middle + right) := by
  apply Fin.ext
  simp only [Fin.val_add]
  rw [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

private theorem fadd_comm (left right : F) :
    left + right = right + left := by
  apply Fin.ext
  simp only [Fin.val_add, Nat.add_comm]

private theorem fmul_assoc (left middle right : F) :
    (left * middle) * right = left * (middle * right) :=
  Fin.mul_assoc _ _ _

private theorem fmul_comm (left right : F) :
    left * right = right * left :=
  Fin.mul_comm _ _

private theorem fmul_add (left middle right : F) :
    left * (middle + right) = left * middle + left * right := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul]
  rw [Nat.mul_mod_mod, Nat.mul_add, ← Nat.add_mod]

private theorem fadd_mul (left middle right : F) :
    (left + middle) * right = left * right + middle * right := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul]
  rw [Nat.mod_mul_mod, Nat.add_mul, ← Nat.add_mod]

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨fadd_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨fadd_comm⟩

local instance : Std.Associative (fun (left right : F) => left * right) :=
  ⟨fmul_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left * right) :=
  ⟨fmul_comm⟩

private theorem k_add_assoc (left middle right : K) :
    K.add (K.add left middle) right =
      K.add left (K.add middle right) := by
  rcases left with ⟨left0, left1⟩
  rcases middle with ⟨middle0, middle1⟩
  rcases right with ⟨right0, right1⟩
  simp only [K.add, K.mk.injEq]
  exact ⟨fadd_assoc _ _ _, fadd_assoc _ _ _⟩

private theorem k_add_comm (left right : K) :
    K.add left right = K.add right left := by
  rcases left with ⟨left0, left1⟩
  rcases right with ⟨right0, right1⟩
  simp only [K.add, K.mk.injEq]
  exact ⟨fadd_comm _ _, fadd_comm _ _⟩

private theorem k_mul_assoc (left middle right : K) :
    K.mul (K.mul left middle) right =
      K.mul left (K.mul middle right) := by
  rcases left with ⟨left0, left1⟩
  rcases middle with ⟨middle0, middle1⟩
  rcases right with ⟨right0, right1⟩
  simp only [K.mul, K.mk.injEq]
  constructor <;> simp only [fmul_add, fadd_mul, fmul_assoc] <;> ac_rfl

private theorem k_mul_comm (left right : K) :
    K.mul left right = K.mul right left := by
  rcases left with ⟨left0, left1⟩
  rcases right with ⟨right0, right1⟩
  simp only [K.mul, K.mk.injEq]
  constructor <;> ac_rfl

private theorem k_mul_add (left middle right : K) :
    K.mul left (K.add middle right) =
      K.add (K.mul left middle) (K.mul left right) := by
  rcases left with ⟨left0, left1⟩
  rcases middle with ⟨middle0, middle1⟩
  rcases right with ⟨right0, right1⟩
  simp only [K.mul, K.add, K.mk.injEq]
  constructor <;> simp only [fmul_add] <;> ac_rfl

private theorem k_add_mul (left middle right : K) :
    K.mul (K.add left middle) right =
      K.add (K.mul left right) (K.mul middle right) := by
  rw [k_mul_comm, k_mul_add]
  congr 1 <;> rw [k_mul_comm]

private theorem fneg_add_cancel (value : F) :
    -value + value = 0 := by
  apply Fin.ext
  rw [Fin.val_add, Fin.val_neg]
  split
  · rename_i zero
    subst value
    rfl
  · rw [Nat.sub_add_cancel (Nat.le_of_lt value.isLt)]
    exact Nat.mod_self _

private theorem fsub_add_cancel (left right : F) :
    left - right + right = left := by
  rw [Fin.sub_eq_add_neg]
  calc
    left + -right + right = left + (-right + right) :=
      fadd_assoc left (-right) right
    _ = left + 0 := by rw [fneg_add_cancel]
    _ = left := Fin.add_zero left

local instance : Std.Associative (fun (left right : K) => K.add left right) :=
  ⟨k_add_assoc⟩

local instance : Std.Commutative (fun (left right : K) => K.add left right) :=
  ⟨k_add_comm⟩

local instance : Std.Associative (fun (left right : K) => K.mul left right) :=
  ⟨k_mul_assoc⟩

local instance : Std.Commutative (fun (left right : K) => K.mul left right) :=
  ⟨k_mul_comm⟩

private theorem sumRange_mul_left
    (factor : K) (count : Nat) (term : Nat → K) :
    K.mul factor (sumRange count term) =
      sumRange count (fun index => K.mul factor (term index)) := by
  induction count with
  | zero => exact mul_zero factor
  | succ count inductionHypothesis =>
      rw [sumRange, sumRange, k_mul_add, inductionHypothesis]

private theorem sumRange_mul_right
    (count : Nat) (term : Nat → K) (factor : K) :
    K.mul (sumRange count term) factor =
      sumRange count (fun index => K.mul (term index) factor) := by
  induction count with
  | zero => exact zero_mul factor
  | succ count inductionHypothesis =>
      rw [sumRange, sumRange, k_add_mul, inductionHypothesis]

private theorem sumRange_add
    (count : Nat) (left right : Nat → K) :
    sumRange count (fun index => K.add (left index) (right index)) =
      K.add (sumRange count left) (sumRange count right) := by
  induction count with
  | zero => exact (zero_add K.zero).symm
  | succ count inductionHypothesis =>
      rw [sumRange, sumRange, sumRange, inductionHypothesis]
      ac_rfl

private theorem sumRange_swap
    (outerCount innerCount : Nat) (term : Nat → Nat → K) :
    sumRange outerCount (fun outer =>
      sumRange innerCount (fun inner => term outer inner)) =
    sumRange innerCount (fun inner =>
      sumRange outerCount (fun outer => term outer inner)) := by
  induction outerCount with
  | zero =>
      rw [sumRange]
      symm
      apply sumRange_eq_zero
      intro inner _
      rfl
  | succ outerCount inductionHypothesis =>
      rw [sumRange, inductionHypothesis]
      symm
      calc
        sumRange innerCount (fun inner =>
            sumRange (outerCount + 1) (fun outer => term outer inner)) =
            sumRange innerCount (fun inner =>
              K.add
                (sumRange outerCount (fun outer => term outer inner))
                (term outerCount inner)) := by
          apply sumRange_congr
          intro inner _
          rw [sumRange]
        _ = K.add
            (sumRange innerCount (fun inner =>
              sumRange outerCount (fun outer => term outer inner)))
            (sumRange innerCount (fun inner => term outerCount inner)) :=
          sumRange_add innerCount
            (fun inner =>
              sumRange outerCount (fun outer => term outer inner))
            (fun inner => term outerCount inner)

private theorem sumRange_append
    (leftCount rightCount : Nat) (term : Nat → K) :
    sumRange (leftCount + rightCount) term =
      K.add (sumRange leftCount term)
        (sumRange rightCount fun index => term (leftCount + index)) := by
  induction rightCount with
  | zero =>
      rw [Nat.add_zero, sumRange]
      exact (add_zero _).symm
  | succ rightCount inductionHypothesis =>
      rw [Nat.add_succ, sumRange, sumRange, inductionHypothesis]
      rw [k_add_assoc]

private theorem k_sub_add_self (value : K) :
    K.add (K.sub K.one value) value = K.one := by
  rcases value with ⟨value0, value1⟩
  simp only [K.add, K.sub, K.one, K.mk.injEq]
  exact ⟨fsub_add_cancel _ _, fsub_add_cancel _ _⟩

/-! ## Multilinear power selector -/

private theorem testBit_add_twoPow_low
    {width mask bit : Nat}
    (maskLt : mask < 2 ^ width) (bitLt : bit < width) :
    Nat.testBit (mask + 2 ^ width) bit = Nat.testBit mask bit := by
  have modulo : (mask + 2 ^ width) % 2 ^ width = mask := by
    calc
      (mask + 2 ^ width) % 2 ^ width = mask % 2 ^ width := by
        simpa using Nat.add_mul_mod_self_left mask (2 ^ width) 1
      _ = mask := Nat.mod_eq_of_lt maskLt
  have projected := Nat.testBit_mod_two_pow
    (mask + 2 ^ width) width bit
  rw [modulo] at projected
  simp [bitLt] at projected
  exact projected.symm

private theorem testBit_add_twoPow_self
    {width mask : Nat} (maskLt : mask < 2 ^ width) :
    Nat.testBit (mask + 2 ^ width) width = true := by
  unfold Nat.testBit
  rw [Nat.shiftRight_eq_div_pow]
  have powerPositive : 0 < 2 ^ width := Nat.two_pow_pos width
  have quotient : (mask + 2 ^ width) / 2 ^ width = 1 := by
    calc
      (mask + 2 ^ width) / 2 ^ width = mask / 2 ^ width + 1 := by
        simpa using Nat.add_mul_div_right mask 1 powerPositive
      _ = 1 := by rw [Nat.div_eq_of_lt maskLt]
  rw [quotient]
  decide

private theorem powK_add (base : K) (left right : Nat) :
    powK base (left + right) =
      K.mul (powK base left) (powK base right) := by
  induction right with
  | zero =>
      rw [Nat.add_zero, powK]
      exact (mul_one _).symm
  | succ right inductionHypothesis =>
      rw [Nat.add_succ, powK, powK, inductionHypothesis, k_mul_assoc]

private def betaPowerWidth
    (producerBeta : K) (width lane : Nat) : K :=
  productRange width fun bit =>
    if Nat.testBit lane bit then
      powK producerBeta (2 ^ bit)
    else
      K.one

private theorem betaPowerWidth_eq_powK
    (producerBeta : K) (width lane : Nat)
    (laneLt : lane < 2 ^ width) :
    betaPowerWidth producerBeta width lane = powK producerBeta lane := by
  induction width generalizing lane with
  | zero =>
      have laneZero : lane = 0 := by omega
      subst lane
      rfl
  | succ width inductionHypothesis =>
      unfold betaPowerWidth
      rw [productRange]
      by_cases low : lane < 2 ^ width
      · rw [Nat.testBit_lt_two_pow low]
        change K.mul (betaPowerWidth producerBeta width lane) K.one = _
        rw [inductionHypothesis lane low, mul_one]
      · let lower := lane - 2 ^ width
        have powerLe : 2 ^ width ≤ lane := Nat.le_of_not_gt low
        have lowerLt : lower < 2 ^ width := by
          unfold lower
          rw [Nat.two_pow_succ] at laneLt
          omega
        have decompose : lower + 2 ^ width = lane := by
          unfold lower
          omega
        have prefixEqual :
            productRange width (fun bit =>
              if Nat.testBit lane bit then
                powK producerBeta (2 ^ bit)
              else K.one) =
            betaPowerWidth producerBeta width lower := by
          unfold betaPowerWidth
          apply productRange_congr
          intro bit bitLt
          rw [← decompose]
          rw [testBit_add_twoPow_low lowerLt bitLt]
        have highBit : Nat.testBit lane width = true := by
          rw [← decompose]
          exact testBit_add_twoPow_self lowerLt
        rw [prefixEqual, highBit, inductionHypothesis lower lowerLt]
        rw [if_pos (by rfl)]
        rw [← powK_add, decompose]

/-- Multilinear power selector
`B_beta(alpha) = product_j ((1-alpha_j) + alpha_j beta^(2^j))`.

Unlike a duplicated full `H_beta(s)` evaluation, this selector leaves the raw
packed value at the current `(s, alpha)` terminal point and therefore can reuse
the existing output evaluation. -/
def betaPowerSelector (producerBeta : K) (alpha : List K) : K :=
  productRange alpha.length fun bit =>
    K.add
      (K.sub K.one (alpha.getD bit K.zero))
      (K.mul (alpha.getD bit K.zero)
        (powK producerBeta (2 ^ bit)))

/-- On an in-domain Boolean lane, the multilinear selector is exactly the
producer's univariate monomial weight `beta^lane`. -/
theorem betaPowerSelector_cubePoint
    (producerBeta : K) (width lane : Nat)
    (laneLt : lane < 2 ^ width) :
    betaPowerSelector producerBeta (cubePoint width lane) =
      powK producerBeta lane := by
  unfold betaPowerSelector
  rw [cubePoint_length]
  calc
    productRange width (fun bit =>
        K.add
          (K.sub K.one
            ((cubePoint width lane).getD bit K.zero))
          (K.mul ((cubePoint width lane).getD bit K.zero)
            (powK producerBeta (2 ^ bit)))) =
        betaPowerWidth producerBeta width lane := by
      unfold betaPowerWidth
      apply productRange_congr
      intro bit bitLt
      rw [cubePoint_getD width lane bit bitLt]
      cases bitValue : Nat.testBit lane bit
      · change K.add (K.sub K.one K.zero)
          (K.mul K.zero (powK producerBeta (2 ^ bit))) = K.one
        rw [zero_mul, add_zero]
        rfl
      · change K.add (K.sub K.one K.one)
          (K.mul K.one (powK producerBeta (2 ^ bit))) =
            powK producerBeta (2 ^ bit)
        rw [one_mul]
        change K.add K.zero (powK producerBeta (2 ^ bit)) = _
        exact zero_add _
    _ = powK producerBeta lane :=
      betaPowerWidth_eq_powK producerBeta width lane laneLt

/-! ## Raw-child evaluation at the producer projection point -/

/-- One raw logical column after Π_DEC radix recomposition and univariate
evaluation of its packed lanes at the producer's Π_RLC `beta`. -/
def rawColumnAtProducerBeta
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (column : Nat) : K :=
  sumRange shape.laneDomain fun lane =>
    K.mul (radixWeightedRawDiagonal radix rawChildren column lane)
      (powK producerBeta lane)

/-- Multilinear column projection of the raw-child value evaluated at the
producer's univariate Π_RLC point. -/
def rawProjectionAtProducerBeta
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (s : List K) : K :=
  sumRange shape.columnDomain fun column =>
    K.mul
      (rawColumnAtProducerBeta
        shape radix rawChildren producerBeta column)
      (chi s column)

/-- The compact producer-point value can equivalently be evaluated from the
radix-combined `y_zcol` sidecar. -/
theorem rawProjectionAtProducerBeta_eq_yZcolEvaluation
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (s : List K) :
    rawProjectionAtProducerBeta
        shape radix rawChildren producerBeta s =
      sumRange shape.laneDomain fun lane =>
        K.mul
          (radixWeightedChildProjection shape radix rawChildren s lane)
          (powK producerBeta lane) := by
  unfold rawProjectionAtProducerBeta rawColumnAtProducerBeta
    radixWeightedChildProjection
  calc
    sumRange shape.columnDomain (fun column =>
        K.mul
          (sumRange shape.laneDomain fun lane =>
            K.mul
              (radixWeightedRawDiagonal
                radix rawChildren column lane)
              (powK producerBeta lane))
          (chi s column)) =
        sumRange shape.columnDomain (fun column =>
          sumRange shape.laneDomain fun lane =>
            K.mul
              (K.mul
                (radixWeightedRawDiagonal
                  radix rawChildren column lane)
                (powK producerBeta lane))
              (chi s column)) := by
      apply sumRange_congr
      intro column _
      exact sumRange_mul_right shape.laneDomain
        (fun lane =>
          K.mul
            (radixWeightedRawDiagonal radix rawChildren column lane)
            (powK producerBeta lane))
        (chi s column)
    _ = sumRange shape.columnDomain (fun column =>
          sumRange shape.laneDomain fun lane =>
            K.mul
              (K.mul
                (radixWeightedRawDiagonal
                  radix rawChildren column lane)
                (chi s column))
              (powK producerBeta lane)) := by
      apply sumRange_congr
      intro column _
      apply sumRange_congr
      intro lane _
      ac_rfl
    _ = sumRange shape.laneDomain (fun lane =>
          sumRange shape.columnDomain fun column =>
            K.mul
              (K.mul
                (radixWeightedRawDiagonal
                  radix rawChildren column lane)
                (chi s column))
              (powK producerBeta lane)) :=
      sumRange_swap shape.columnDomain shape.laneDomain _
    _ = sumRange shape.laneDomain (fun lane =>
          K.mul
            (sumRange shape.columnDomain fun column =>
              K.mul
                (radixWeightedRawDiagonal
                  radix rawChildren column lane)
                (chi s column))
            (powK producerBeta lane)) := by
      apply sumRange_congr
      intro lane _
      symm
      exact sumRange_mul_right shape.columnDomain
        (fun column =>
          K.mul
            (radixWeightedRawDiagonal radix rawChildren column lane)
            (chi s column))
        (powK producerBeta lane)
    _ = sumRange shape.laneDomain (fun lane =>
          K.mul
            (if lane < shape.laneDomain then
              sumRange shape.columnDomain fun column =>
                K.mul
                  (radixWeightedRawDiagonal
                    radix rawChildren column lane)
                  (chi s column)
            else K.zero)
            (powK producerBeta lane)) := by
      apply sumRange_congr
      intro lane laneLt
      rw [if_pos laneLt]

/-- Every lane at or above the active cyclotomic width is zero in the raw
diagonal, independently of the prover's child values. -/
theorem radixWeightedRawDiagonal_eq_zero_of_ringDegree_le
    (radix : F) (rawChildren : List (List F)) (column lane : Nat)
    (laneGe : ringDegree ≤ lane) :
    radixWeightedRawDiagonal radix rawChildren column lane = K.zero := by
  unfold radixWeightedRawDiagonal
  apply sumRange_eq_zero
  intro childIndex _
  have diagonalZero :
      directDiagonal (rawChildren.getD childIndex []) column lane = K.zero := by
    unfold directDiagonal
    rw [if_neg]
    intro live
    have reducedLt : column % ringDegree < ringDegree :=
      Nat.mod_lt column (by decide)
    omega
  rw [diagonalZero, mul_zero]

/-- Consequently every radix-combined `y_zcol` lane beyond `D = 54` is zero.
The production circuit still has to refine its explicit padding rows to this
semantic fact. -/
theorem radixWeightedChildProjection_eq_zero_of_ringDegree_le
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) (lane : Nat) (laneGe : ringDegree ≤ lane) :
    radixWeightedChildProjection shape radix rawChildren s lane = K.zero := by
  unfold radixWeightedChildProjection
  split
  · apply sumRange_eq_zero
    intro column _
    rw [radixWeightedRawDiagonal_eq_zero_of_ringDegree_le
      radix rawChildren column lane laneGe]
    exact zero_mul _
  · rfl

/-- Producer-point evaluation over exactly the `D = 54` active lanes. -/
def activeRawProjectionAtProducerBeta
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (s : List K) : K :=
  sumRange ringDegree fun lane =>
    K.mul
      (radixWeightedChildProjection shape radix rawChildren s lane)
      (powK producerBeta lane)

/-- The padded NC-lane evaluation equals the active Π_RLC evaluation when the
selected lane domain covers all `D` coefficients. -/
theorem rawProjectionAtProducerBeta_eq_active
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (s : List K)
    (lanesCoverRing : ringDegree ≤ shape.laneDomain) :
    rawProjectionAtProducerBeta shape radix rawChildren producerBeta s =
      activeRawProjectionAtProducerBeta
        shape radix rawChildren producerBeta s := by
  rw [rawProjectionAtProducerBeta_eq_yZcolEvaluation]
  unfold activeRawProjectionAtProducerBeta
  calc
    sumRange shape.laneDomain (fun lane =>
        K.mul
          (radixWeightedChildProjection shape radix rawChildren s lane)
          (powK producerBeta lane)) =
        sumRange (ringDegree + (shape.laneDomain - ringDegree)) (fun lane =>
          K.mul
            (radixWeightedChildProjection shape radix rawChildren s lane)
            (powK producerBeta lane)) := by
      rw [Nat.add_sub_of_le lanesCoverRing]
    _ = K.add
          (sumRange ringDegree fun lane =>
            K.mul
              (radixWeightedChildProjection
                shape radix rawChildren s lane)
              (powK producerBeta lane))
          (sumRange (shape.laneDomain - ringDegree) fun offset =>
            K.mul
              (radixWeightedChildProjection shape radix rawChildren s
                (ringDegree + offset))
              (powK producerBeta (ringDegree + offset))) :=
      sumRange_append ringDegree (shape.laneDomain - ringDegree) _
    _ = K.add
          (sumRange ringDegree fun lane =>
            K.mul
              (radixWeightedChildProjection
                shape radix rawChildren s lane)
              (powK producerBeta lane))
          K.zero := by
      congr 1
      apply sumRange_eq_zero
      intro offset _
      rw [radixWeightedChildProjection_eq_zero_of_ringDegree_le
        shape radix rawChildren s (ringDegree + offset) (by omega)]
      exact zero_mul _
    _ = sumRange ringDegree (fun lane =>
          K.mul
            (radixWeightedChildProjection shape radix rawChildren s lane)
            (powK producerBeta lane)) :=
      add_zero _

/-- At a Boolean column, the raw producer-point MLE selects exactly that
column's radix-combined packed-lane evaluation. -/
theorem rawProjectionAtProducerBeta_cubePoint
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) {column : Nat}
    (columnLt : column < shape.columnDomain) :
    rawProjectionAtProducerBeta shape radix rawChildren producerBeta
        (cubePoint shape.ellM column) =
      rawColumnAtProducerBeta
        shape radix rawChildren producerBeta column := by
  unfold rawProjectionAtProducerBeta
  calc
    sumRange shape.columnDomain (fun current =>
        K.mul
          (rawColumnAtProducerBeta
            shape radix rawChildren producerBeta current)
          (chi (cubePoint shape.ellM column) current)) =
        sumRange shape.columnDomain (fun current =>
          if current = column then
            rawColumnAtProducerBeta
              shape radix rawChildren producerBeta current
          else K.zero) := by
      apply sumRange_congr
      intro current currentLt
      rw [chi_cubePoint_eq_if shape.ellM column current columnLt currentLt]
      by_cases selected : current = column
      · subst current
        simp [mul_one]
      · rw [if_neg selected, if_neg (Ne.symm selected), mul_zero]
    _ = rawColumnAtProducerBeta
          shape radix rawChildren producerBeta column :=
      sumRange_select shape.columnDomain column
        (fun current =>
          rawColumnAtProducerBeta
            shape radix rawChildren producerBeta current)
        columnLt

/-! ## Production-shaped delayed lift -/

/-- Raw Π_DEC radix recomposition at the current NC multilinear point.

This is the value already needed by the ordinary NC terminal identity. The
delayed residual multiplies this value by selectors; it does not materialize a
second 54-coefficient univariate evaluation at the terminal point. -/
def radixCombinedRawZ
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s alpha : List K) : K :=
  dotChi shape
    (radixWeightedChildProjection shape radix rawChildren s)
    alpha

/-- The active `D = 54` prefix of the combined output evaluation. This is the
existing NC `y_eval` dot product after eliminating the semantically zero
padded lanes; it is not a second univariate evaluation at `producerBeta`. -/
def activeRadixCombinedRawZ
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s alpha : List K) : K :=
  sumRange ringDegree fun lane =>
    K.mul
      (radixWeightedChildProjection shape radix rawChildren s lane)
      (chi alpha lane)

/-- The padded raw `z` evaluation equals its active 54-lane `y_eval` prefix
when the selected NC lane domain covers the ring degree. -/
theorem radixCombinedRawZ_eq_active
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s alpha : List K)
    (lanesCoverRing : ringDegree ≤ shape.laneDomain) :
    radixCombinedRawZ shape radix rawChildren s alpha =
      activeRadixCombinedRawZ shape radix rawChildren s alpha := by
  unfold radixCombinedRawZ dotChi activeRadixCombinedRawZ
  calc
    sumRange shape.laneDomain (fun lane =>
        K.mul
          (radixWeightedChildProjection shape radix rawChildren s lane)
          (chi alpha lane)) =
        sumRange (ringDegree + (shape.laneDomain - ringDegree)) (fun lane =>
          K.mul
            (radixWeightedChildProjection shape radix rawChildren s lane)
            (chi alpha lane)) := by
      rw [Nat.add_sub_of_le lanesCoverRing]
    _ = K.add
          (sumRange ringDegree fun lane =>
            K.mul
              (radixWeightedChildProjection
                shape radix rawChildren s lane)
              (chi alpha lane))
          (sumRange (shape.laneDomain - ringDegree) fun offset =>
            K.mul
              (radixWeightedChildProjection shape radix rawChildren s
                (ringDegree + offset))
              (chi alpha (ringDegree + offset))) :=
      sumRange_append ringDegree (shape.laneDomain - ringDegree) _
    _ = K.add
          (sumRange ringDegree fun lane =>
            K.mul
              (radixWeightedChildProjection
                shape radix rawChildren s lane)
              (chi alpha lane))
          K.zero := by
      congr 1
      apply sumRange_eq_zero
      intro offset _
      rw [radixWeightedChildProjection_eq_zero_of_ringDegree_le
        shape radix rawChildren s (ringDegree + offset) (by omega)]
      exact zero_mul _
    _ = sumRange ringDegree (fun lane =>
          K.mul
            (radixWeightedChildProjection shape radix rawChildren s lane)
            (chi alpha lane)) :=
      add_zero _

/-- On an in-domain Boolean point, the shared raw `z` evaluation selects the
corresponding radix-recombined diagonal coordinate. -/
theorem radixCombinedRawZ_cubePoint
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    {column lane : Nat}
    (columnLt : column < shape.columnDomain)
    (laneLt : lane < shape.laneDomain) :
    radixCombinedRawZ shape radix rawChildren
        (cubePoint shape.ellM column)
        (cubePoint shape.ellD lane) =
      radixWeightedRawDiagonal radix rawChildren column lane := by
  unfold radixCombinedRawZ dotChi
  calc
    sumRange shape.laneDomain (fun current =>
        K.mul
          (radixWeightedChildProjection shape radix rawChildren
            (cubePoint shape.ellM column) current)
          (chi (cubePoint shape.ellD lane) current)) =
        sumRange shape.laneDomain (fun current =>
          if current = lane then
            radixWeightedRawDiagonal radix rawChildren column current
          else
            K.zero) := by
      apply sumRange_congr
      intro current currentLt
      rw [radixWeightedChildProjection_cubePoint
        shape radix rawChildren columnLt currentLt]
      rw [chi_cubePoint_eq_if shape.ellD lane current laneLt currentLt]
      by_cases selected : current = lane
      · subst current
        simp [mul_one]
      · rw [if_neg selected, if_neg (Ne.symm selected), mul_zero]
    _ = radixWeightedRawDiagonal radix rawChildren column lane :=
      sumRange_select shape.laneDomain lane
        (fun current =>
          radixWeightedRawDiagonal radix rawChildren column current)
        laneLt

/-- Shape and coverage premises needed to interpret the finite-domain theorem
as a statement about every production raw coordinate, rather than a truncated
semantic table. -/
structure DelayedResidualShape
    (shape : Shape) (oldS : List K)
    (rawChildren : List (List F)) : Prop where
  oldSLength : oldS.length = shape.ellM
  childrenFit : AssignmentsFitColumnDomain shape rawChildren
  lanesCoverRing : ringDegree ≤ shape.laneDomain

/-- Fixed arity of the final NC SumCheck point. Keeping this premise in the
terminal theorem prevents an algebraically self-consistent malformed list
from being mistaken for an evaluation of the intended `(ellM, ellD)`
polynomial. -/
structure TerminalPointShape
    (shape : Shape) (terminalS terminalAlpha : List K) : Prop where
  sLength : terminalS.length = shape.ellM
  alphaLength : terminalAlpha.length = shape.ellD

/-- The selected Boolean column domain covers every authoritative raw-child
coordinate. The Rust source-layout bridge must instantiate this lemma from
the concrete witness matrix dimensions. -/
theorem rawChildCoordinate_lt_columnDomain
    {shape : Shape} {oldS : List K} {rawChildren : List (List F)}
    (wellShaped : DelayedResidualShape shape oldS rawChildren)
    {assignment : List F} (assignmentMem : assignment ∈ rawChildren)
    {column : Nat} (columnLt : column < assignment.length) :
    column < shape.columnDomain :=
  Nat.lt_of_lt_of_le columnLt
    (wellShaped.childrenFit assignment assignmentMem)

/-- Direct Boolean-table form of the delayed residual. `batchWeight` is an
independent verifier-owned batching coefficient, not the NC range `gamma`
unless a later transcript proof reserves a distinct power for this purpose. -/
def delayedResidualOnCube
    (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    (column lane : Nat) : K :=
  K.mul batchWeight
    (K.mul (chi oldS column)
      (K.mul (powK producerBeta lane)
        (radixWeightedRawDiagonal radix rawChildren column lane)))

/-- Extension polynomial added as one independently weighted NC SumCheck
summand. The column equality selector recovers the old point, while the
multilinear power selector turns a Boolean lane into the producer's
univariate `beta^lane` weight. -/
def delayedResidualPolynomial
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    (s alpha : List K) : K :=
  K.mul batchWeight
    (K.mul (eqPoint s oldS)
      (K.mul (betaPowerSelector producerBeta alpha)
        (radixCombinedRawZ shape radix rawChildren s alpha)))

/-- The ordinary terminal-point formula for the delayed summand. Its final
factor is the same raw `z(s', alpha')` value already consumed by the NC
terminal identity, rather than a duplicated active-coefficient evaluation. -/
def delayedResidualTerminalRhs
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    (terminalS terminalAlpha : List K) : K :=
  K.mul batchWeight
    (K.mul (eqPoint terminalS oldS)
      (K.mul (betaPowerSelector producerBeta terminalAlpha)
        (activeRadixCombinedRawZ
          shape radix rawChildren terminalS terminalAlpha)))

/-- The extension polynomial restricts to the direct residual table at each
in-domain Boolean point. -/
theorem delayedResidualPolynomial_cubePoint
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    {column lane : Nat}
    (oldSLength : oldS.length = shape.ellM)
    (columnLt : column < shape.columnDomain)
    (laneLt : lane < shape.laneDomain) :
    delayedResidualPolynomial shape radix rawChildren
        producerBeta batchWeight oldS
        (cubePoint shape.ellM column)
        (cubePoint shape.ellD lane) =
      delayedResidualOnCube radix rawChildren
        producerBeta batchWeight oldS column lane := by
  unfold delayedResidualPolynomial delayedResidualOnCube
  rw [eqPoint_cubePoint_eq_chi shape.ellM column oldS oldSLength]
  rw [betaPowerSelector_cubePoint
    producerBeta shape.ellD lane laneLt]
  rw [radixCombinedRawZ_cubePoint
    shape radix rawChildren columnLt laneLt]

/-- The terminal formula is the same delayed polynomial evaluated at the final
SumCheck point; this uses raw-child semantics, not a prover-carried output. A
row-level refinement must still identify the production output wire with
`radixCombinedRawZ`. -/
theorem delayedResidualPolynomial_eq_terminalRhs
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    (terminalS terminalAlpha : List K)
    (wellShaped : DelayedResidualShape shape oldS rawChildren)
    (_terminalShape : TerminalPointShape shape terminalS terminalAlpha) :
    delayedResidualPolynomial shape radix rawChildren
        producerBeta batchWeight oldS terminalS terminalAlpha =
      delayedResidualTerminalRhs shape radix rawChildren
        producerBeta batchWeight oldS terminalS terminalAlpha := by
  unfold delayedResidualPolynomial delayedResidualTerminalRhs
  rw [radixCombinedRawZ_eq_active
    shape radix rawChildren terminalS terminalAlpha wellShaped.lanesCoverRing]

/-- Exact Boolean-cube sum of the delayed extension polynomial. -/
def delayedResidualCubeSum
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K) : K :=
  sumRange shape.columnDomain fun column =>
    sumRange shape.laneDomain fun lane =>
      delayedResidualPolynomial shape radix rawChildren
        producerBeta batchWeight oldS
        (cubePoint shape.ellM column)
        (cubePoint shape.ellD lane)

private theorem delayedResidualOnCube_sum
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K) :
    (sumRange shape.columnDomain fun column =>
      sumRange shape.laneDomain fun lane =>
        delayedResidualOnCube radix rawChildren
          producerBeta batchWeight oldS column lane) =
      K.mul batchWeight
        (rawProjectionAtProducerBeta
          shape radix rawChildren producerBeta oldS) := by
  unfold delayedResidualOnCube rawProjectionAtProducerBeta
    rawColumnAtProducerBeta
  calc
    sumRange shape.columnDomain (fun column =>
        sumRange shape.laneDomain (fun lane =>
          K.mul batchWeight
            (K.mul (chi oldS column)
              (K.mul (powK producerBeta lane)
                (radixWeightedRawDiagonal
                  radix rawChildren column lane))))) =
        sumRange shape.columnDomain (fun column =>
          K.mul
            batchWeight
            (K.mul
              (sumRange shape.laneDomain fun lane =>
                K.mul
                  (radixWeightedRawDiagonal
                    radix rawChildren column lane)
                  (powK producerBeta lane))
              (chi oldS column))) := by
      apply sumRange_congr
      intro column _
      symm
      rw [sumRange_mul_right, sumRange_mul_left]
      apply sumRange_congr
      intro lane _
      ac_rfl
    _ = K.mul batchWeight
          (sumRange shape.columnDomain fun column =>
            K.mul
              (sumRange shape.laneDomain fun lane =>
                K.mul
                  (radixWeightedRawDiagonal
                    radix rawChildren column lane)
                  (powK producerBeta lane))
              (chi oldS column)) :=
      (sumRange_mul_left batchWeight shape.columnDomain _).symm

/-- Boolean-cube normalization of the production-shaped delayed transfer.
`DelayedResidualShape` additionally records the coverage premises needed to
interpret the value as the complete production assignment rather than a
truncated table. -/
theorem delayedResidualCubeSum_eq_weightedOldProjection
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    (wellShaped : DelayedResidualShape shape oldS rawChildren) :
    delayedResidualCubeSum shape radix rawChildren
        producerBeta batchWeight oldS =
      K.mul batchWeight
        (rawProjectionAtProducerBeta
          shape radix rawChildren producerBeta oldS) := by
  unfold delayedResidualCubeSum
  calc
    sumRange shape.columnDomain (fun column =>
        sumRange shape.laneDomain fun lane =>
          delayedResidualPolynomial shape radix rawChildren
            producerBeta batchWeight oldS
            (cubePoint shape.ellM column)
            (cubePoint shape.ellD lane)) =
        sumRange shape.columnDomain (fun column =>
          sumRange shape.laneDomain fun lane =>
            delayedResidualOnCube radix rawChildren
              producerBeta batchWeight oldS column lane) := by
      apply sumRange_congr
      intro column columnLt
      apply sumRange_congr
      intro lane laneLt
      exact delayedResidualPolynomial_cubePoint
        shape radix rawChildren producerBeta batchWeight oldS
        wellShaped.oldSLength columnLt laneLt
    _ = K.mul batchWeight
          (rawProjectionAtProducerBeta
            shape radix rawChildren producerBeta oldS) :=
      delayedResidualOnCube_sum shape radix rawChildren
        producerBeta batchWeight oldS

/-- The cube sum is exactly the active `D = 54`, maximum-degree-53 compact
old-point projection used by Π_RLC. Padded NC lanes disappear by the proved
raw-diagonal zero property; production padding-row correspondence remains an
explicit refinement obligation. -/
theorem delayedResidualCubeSum_eq_weightedCompactOldProjection
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    (wellShaped : DelayedResidualShape shape oldS rawChildren) :
    delayedResidualCubeSum shape radix rawChildren
        producerBeta batchWeight oldS =
      K.mul batchWeight
        (activeRawProjectionAtProducerBeta
          shape radix rawChildren producerBeta oldS) := by
  rw [delayedResidualCubeSum_eq_weightedOldProjection
    shape radix rawChildren producerBeta batchWeight oldS wellShaped]
  rw [rawProjectionAtProducerBeta_eq_active
    shape radix rawChildren producerBeta oldS wellShaped.lanesCoverRing]

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
