import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Terminal.Identity

/-!
Contract: state the delayed old-point projection transfer required by the
optimized Π_CCS/Π_RLC/Π_DEC recursion.

Owns: radix-weighted raw-child projection, its child-sidecar linearity and
pointwise-decomposition bridges, the old-point relation/state-continuity
transfer, and mismatch exclusion.

Does not own: derivation from a domain-separated mixed summand in the
production NC SumCheck, Π_RLC output-to-next-parent refinement, state or
deterministic commitment binding, the PaperExact digit projection, concrete
rows, or row-removal permission.

Emits constraints: no.

Authority boundary: `rawChildren` is independent assignment authority. A
prover-carried digest or projection is insufficient; `DelayedProjectionStep`
separates the verifier-established old-point relation from equality to the
parent pair already bound in recursive state.

| Surface | Mathematical obligation | Refinement status | Permits row removal? |
|---|---|---|---|
| raw diagonal | `sum_i b^i Z_i[column, lane]` over raw assignments | optimized model-level | no |
| child sidecars | raw-first projection equals `sum_i b^i y_zcol_i` | optimized model-level | no |
| pointwise bridge | radix decomposition commutes with direct projection | model-level | no |
| old-point relation | checked result equals the raw-child projection | current NC SumCheck mixed-summand and fresh-mixing refinement open | no |
| delayed transfer | state parent equals the checked old-point result | state/commitment refinement open | no |
| next outputs | each new output is bound to its raw assignment projection | conditional `YZcolBound`; Π_RLC bridge open | no |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection

open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial
open Nightstream.Implementation.R1CS.PiCcsNc.Terminal
open Nightstream.SuperNeo.Concrete

/-- Parent NC pair carried across the delayed check boundary.

This value is semantic data only. Constructing it does not establish state or
commitment binding. -/
structure DelayedParent where
  sCol : List K
  yZcol : YZcol

/-- Pointwise radix decomposition of the authoritative raw child diagonals.

Child index zero has weight `b^0`; this is the zero-based form of the paper's
one-based `b^(i-1)` recomposition order. -/
def radixWeightedRawDiagonal
    (radix : F) (rawChildren : List (List F))
    (column lane : Nat) : K :=
  sumRange rawChildren.length fun childIndex =>
    K.mul (powK (K.embed radix) childIndex)
      (directDiagonal (rawChildren.getD childIndex []) column lane)

/-- Independently evaluate the old-point projection after pointwise radix
recomposition of the authoritative raw running children. -/
def radixWeightedChildProjection
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (sCol : List K) : YZcol :=
  fun lane =>
    if lane < shape.laneDomain then
      sumRange shape.columnDomain fun column =>
        K.mul (radixWeightedRawDiagonal radix rawChildren column lane)
          (chi sCol column)
    else
      K.zero

/-- Radix recomposition of independently evaluated child `y_zcol` sidecars. -/
def radixWeightedAuthoritativeYZcol
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (sCol : List K) : YZcol :=
  fun lane =>
    sumRange rawChildren.length fun childIndex =>
      K.mul (powK (K.embed radix) childIndex)
        (authoritativeYZcol shape
          (rawChildren.getD childIndex []) sCol lane)

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

local instance : Std.Associative (fun (left right : K) => K.add left right) :=
  ⟨k_add_assoc⟩

local instance : Std.Commutative (fun (left right : K) => K.add left right) :=
  ⟨k_add_comm⟩

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

/-- Projecting the pointwise radix recomposition is exactly the radix-weighted
sum of the authoritative child `y_zcol` lanes. This is the model-level
linearity bridge to the sidecars emitted by optimized Π_DEC. -/
theorem radixWeightedChildProjection_eq_weightedAuthoritativeYZcol
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (sCol : List K) {lane : Nat}
    (laneLt : lane < shape.laneDomain) :
    radixWeightedChildProjection shape radix rawChildren sCol lane =
      radixWeightedAuthoritativeYZcol
        shape radix rawChildren sCol lane := by
  unfold radixWeightedChildProjection radixWeightedAuthoritativeYZcol
    authoritativeYZcol radixWeightedRawDiagonal
  rw [if_pos laneLt]
  calc
    sumRange shape.columnDomain (fun column =>
        K.mul
          (sumRange rawChildren.length fun childIndex =>
            K.mul (powK (K.embed radix) childIndex)
              (directDiagonal
                (rawChildren.getD childIndex []) column lane))
          (chi sCol column)) =
        sumRange shape.columnDomain (fun column =>
          sumRange rawChildren.length fun childIndex =>
            K.mul
              (K.mul (powK (K.embed radix) childIndex)
                (directDiagonal
                  (rawChildren.getD childIndex []) column lane))
              (chi sCol column)) := by
      apply sumRange_congr
      intro column _
      exact sumRange_mul_right rawChildren.length
        (fun childIndex =>
          K.mul (powK (K.embed radix) childIndex)
            (directDiagonal
              (rawChildren.getD childIndex []) column lane))
        (chi sCol column)
    _ = sumRange rawChildren.length (fun childIndex =>
          sumRange shape.columnDomain fun column =>
            K.mul
              (K.mul (powK (K.embed radix) childIndex)
                (directDiagonal
                  (rawChildren.getD childIndex []) column lane))
              (chi sCol column)) :=
      sumRange_swap shape.columnDomain rawChildren.length
        (fun column childIndex =>
          K.mul
            (K.mul (powK (K.embed radix) childIndex)
              (directDiagonal
                (rawChildren.getD childIndex []) column lane))
            (chi sCol column))
    _ = sumRange rawChildren.length (fun childIndex =>
          K.mul (powK (K.embed radix) childIndex)
            (sumRange shape.columnDomain fun column =>
              K.mul
                (directDiagonal
                  (rawChildren.getD childIndex []) column lane)
                (chi sCol column))) := by
      apply sumRange_congr
      intro childIndex _
      calc
        sumRange shape.columnDomain (fun column =>
            K.mul
              (K.mul (powK (K.embed radix) childIndex)
                (directDiagonal
                  (rawChildren.getD childIndex []) column lane))
              (chi sCol column)) =
            sumRange shape.columnDomain (fun column =>
              K.mul (powK (K.embed radix) childIndex)
                (K.mul
                  (directDiagonal
                    (rawChildren.getD childIndex []) column lane)
                  (chi sCol column))) := by
          apply sumRange_congr
          intro column _
          rw [k_mul_assoc]
        _ = K.mul (powK (K.embed radix) childIndex)
            (sumRange shape.columnDomain fun column =>
              K.mul
                (directDiagonal
                  (rawChildren.getD childIndex []) column lane)
                (chi sCol column)) := by
          symm
          exact sumRange_mul_left
            (powK (K.embed radix) childIndex) shape.columnDomain
            (fun column =>
              K.mul
                (directDiagonal
                  (rawChildren.getD childIndex []) column lane)
                (chi sCol column))
    _ = sumRange rawChildren.length (fun childIndex =>
          K.mul (powK (K.embed radix) childIndex)
            (if lane < shape.laneDomain then
              sumRange shape.columnDomain fun column =>
                K.mul
                  (directDiagonal
                    (rawChildren.getD childIndex []) column lane)
                  (chi sCol column)
            else K.zero)) := by
      apply sumRange_congr
      intro childIndex _
      rw [if_pos laneLt]

/-- Exact pointwise Π_DEC radix recomposition premise for the direct packed
diagonal. A production refinement must derive this from the checked child
witness decomposition. -/
def PointwiseRadixDecomposition
    (shape : Shape) (radix : F) (parentAssignment : List F)
    (rawChildren : List (List F)) : Prop :=
  ∀ column lane,
    column < shape.columnDomain →
    lane < shape.laneDomain →
    directDiagonal parentAssignment column lane =
      radixWeightedRawDiagonal radix rawChildren column lane

/-- The existing independently evaluated parent projection agrees with the
raw-child projection whenever the raw assignment decomposes pointwise. -/
theorem authoritativeYZcol_eq_radixWeightedChildProjection
    {shape : Shape} {radix : F} {parentAssignment : List F}
    {rawChildren : List (List F)}
    (decomposition : PointwiseRadixDecomposition
      shape radix parentAssignment rawChildren)
    (sCol : List K) {lane : Nat}
    (laneLt : lane < shape.laneDomain) :
    authoritativeYZcol shape parentAssignment sCol lane =
      radixWeightedChildProjection
        shape radix rawChildren sCol lane := by
  unfold authoritativeYZcol radixWeightedChildProjection
  rw [if_pos laneLt, if_pos laneLt]
  apply sumRange_congr
  intro column columnLt
  rw [decomposition column lane columnLt laneLt]

/-- At a Boolean column point, the independently evaluated projection selects
the pointwise radix-weighted raw diagonal. -/
theorem radixWeightedChildProjection_cubePoint
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    {column lane : Nat}
    (columnLt : column < shape.columnDomain)
    (laneLt : lane < shape.laneDomain) :
    radixWeightedChildProjection shape radix rawChildren
        (cubePoint shape.ellM column) lane =
      radixWeightedRawDiagonal radix rawChildren column lane := by
  unfold radixWeightedChildProjection
  rw [if_pos laneLt]
  calc
    sumRange shape.columnDomain (fun current =>
        K.mul (radixWeightedRawDiagonal radix rawChildren current lane)
          (chi (cubePoint shape.ellM column) current)) =
        sumRange shape.columnDomain (fun current =>
          if current = column then
            radixWeightedRawDiagonal radix rawChildren current lane
          else
            K.zero) := by
      apply sumRange_congr
      intro current currentLt
      rw [chi_cubePoint_eq_if shape.ellM column current columnLt currentLt]
      by_cases selected : current = column
      · subst current
        simp [mul_one]
      · rw [if_neg selected, if_neg (Ne.symm selected), mul_zero]
    _ = radixWeightedRawDiagonal radix rawChildren column lane :=
      sumRange_select shape.columnDomain column
        (fun current =>
          radixWeightedRawDiagonal radix rawChildren current lane)
        columnLt

/-- Semantic result that the delayed old-point check must establish.

This need not be a second SumCheck: a production refinement may add a freshly,
domain-separated mixed residual to the current NC polynomial. This predicate
does not claim that the current rows or transcript derive that relation. -/
structure OldPointSumcheckRelation
    (shape : Shape) (radix : F) (checked : DelayedParent)
    (rawChildren : List (List F)) : Prop where
  pointLength : checked.sCol.length = shape.ellM
  childrenFit : AssignmentsFitColumnDomain shape rawChildren
  lane : ∀ lane, lane < shape.laneDomain →
    checked.yZcol lane =
      radixWeightedChildProjection
        shape radix rawChildren checked.sCol lane

/-- Pointwise raw decomposition is sufficient for the mathematical old-point
relation when the checked result is the authoritative parent projection. -/
theorem oldPointSumcheckRelation_of_pointwiseDecomposition
    {shape : Shape} {radix : F} {parentAssignment : List F}
    {rawChildren : List (List F)} {sCol : List K}
    (pointLength : sCol.length = shape.ellM)
    (childrenFit : AssignmentsFitColumnDomain shape rawChildren)
    (decomposition : PointwiseRadixDecomposition
      shape radix parentAssignment rawChildren) :
    OldPointSumcheckRelation shape radix
      { sCol := sCol
        yZcol := authoritativeYZcol shape parentAssignment sCol }
      rawChildren := by
  refine ⟨pointLength, childrenFit, ?_⟩
  intro lane laneLt
  exact authoritativeYZcol_eq_radixWeightedChildProjection
    decomposition sCol laneLt

/-- Exact summary property obtained after the delayed old-point result is
connected to the recursive-state parent. -/
structure DelayedParentProjectionBound
    (shape : Shape) (radix : F) (parent : DelayedParent)
    (rawChildren : List (List F)) : Prop where
  pointLength : parent.sCol.length = shape.ellM
  childrenFit : AssignmentsFitColumnDomain shape rawChildren
  lane : ∀ lane, lane < shape.laneDomain →
    parent.yZcol lane =
      radixWeightedChildProjection
        shape radix rawChildren parent.sCol lane

/-- One recursive delayed-projection transition.

`oldPointSumcheck` is the semantic verifier result for the state parent's old
point. `statePoint` and `stateLane` are the recursive-state continuity link.
`nextPointOutputs` records the independently authoritative outputs at the new
point; turning them into the next state parent still requires the open
Π_RLC/Π_DEC and commitment refinement. -/
structure DelayedProjectionStep
    (shape : Shape) (radix : F)
    (stateParent checkedOld : DelayedParent)
    (rawRunningChildren nextAssignments : List (List F))
    (nextSCol : List K) (nextOutputs : List YZcol) : Prop where
  statePoint : checkedOld.sCol = stateParent.sCol
  stateLane : ∀ lane, lane < shape.laneDomain →
    stateParent.yZcol lane = checkedOld.yZcol lane
  oldPointSumcheck : OldPointSumcheckRelation
    shape radix checkedOld rawRunningChildren
  nextPointOutputs : YZcolBound
    shape nextAssignments nextSCol nextOutputs

/-- State continuity transfers the verified old-point relation to the exact
state-parent bound. -/
theorem delayedParentProjectionBound_of_step
    {shape : Shape} {radix : F}
    {stateParent checkedOld : DelayedParent}
    {rawRunningChildren nextAssignments : List (List F)}
    {nextSCol : List K} {nextOutputs : List YZcol}
    (step : DelayedProjectionStep shape radix stateParent checkedOld
      rawRunningChildren nextAssignments nextSCol nextOutputs) :
    DelayedParentProjectionBound
      shape radix stateParent rawRunningChildren := by
  refine ⟨?_, step.oldPointSumcheck.childrenFit, ?_⟩
  · rw [← step.statePoint]
    exact step.oldPointSumcheck.pointLength
  · intro lane laneLt
    calc
      stateParent.yZcol lane = checkedOld.yZcol lane :=
        step.stateLane lane laneLt
      _ = radixWeightedChildProjection shape radix rawRunningChildren
          checkedOld.sCol lane :=
        step.oldPointSumcheck.lane lane laneLt
      _ = radixWeightedChildProjection shape radix rawRunningChildren
          stateParent.sCol lane := by rw [step.statePoint]

/-- A complete semantic step both closes the old state-parent projection and
carries authoritative new-point outputs toward the next recursive state. -/
theorem delayedProjectionStep_transfer
    {shape : Shape} {radix : F}
    {stateParent checkedOld : DelayedParent}
    {rawRunningChildren nextAssignments : List (List F)}
    {nextSCol : List K} {nextOutputs : List YZcol}
    (step : DelayedProjectionStep shape radix stateParent checkedOld
      rawRunningChildren nextAssignments nextSCol nextOutputs) :
    DelayedParentProjectionBound
        shape radix stateParent rawRunningChildren ∧
      YZcolBound shape nextAssignments nextSCol nextOutputs :=
  ⟨delayedParentProjectionBound_of_step step, step.nextPointOutputs⟩

/-- The bound exposes exact equality for every consumed in-domain parent lane. -/
theorem exactLane_of_delayedParentProjectionBound
    {shape : Shape} {radix : F} {parent : DelayedParent}
    {rawChildren : List (List F)}
    (bound : DelayedParentProjectionBound
      shape radix parent rawChildren)
    {lane : Nat} (laneLt : lane < shape.laneDomain) :
    parent.yZcol lane =
      radixWeightedChildProjection
        shape radix rawChildren parent.sCol lane :=
  bound.lane lane laneLt

/-- Equivalently, each bound parent lane is the radix-weighted sum of the
independently evaluated child `y_zcol` sidecars consumed by Π_DEC. -/
theorem exactWeightedAuthoritativeYZcolLane_of_bound
    {shape : Shape} {radix : F} {parent : DelayedParent}
    {rawChildren : List (List F)}
    (bound : DelayedParentProjectionBound
      shape radix parent rawChildren)
    {lane : Nat} (laneLt : lane < shape.laneDomain) :
    parent.yZcol lane =
      radixWeightedAuthoritativeYZcol
        shape radix rawChildren parent.sCol lane := by
  calc
    parent.yZcol lane =
        radixWeightedChildProjection
          shape radix rawChildren parent.sCol lane :=
      exactLane_of_delayedParentProjectionBound bound laneLt
    _ = radixWeightedAuthoritativeYZcol
          shape radix rawChildren parent.sCol lane :=
      radixWeightedChildProjection_eq_weightedAuthoritativeYZcol
        shape radix rawChildren parent.sCol laneLt

/-- A delayed check fails when one consumed state-parent lane differs from the
independently evaluated raw-child projection at the old point. -/
def DelayedParentProjectionMismatch
    (shape : Shape) (radix : F) (parent : DelayedParent)
    (rawChildren : List (List F)) : Prop :=
  ∃ lane, lane < shape.laneDomain ∧
    parent.yZcol lane ≠
      radixWeightedChildProjection
        shape radix rawChildren parent.sCol lane

/-- An exact delayed parent projection bound rules out every corresponding
in-domain mismatch. -/
theorem not_delayedParentProjectionMismatch_of_bound
    {shape : Shape} {radix : F} {parent : DelayedParent}
    {rawChildren : List (List F)}
    (bound : DelayedParentProjectionBound
      shape radix parent rawChildren) :
    ¬ DelayedParentProjectionMismatch
      shape radix parent rawChildren := by
  intro mismatch
  rcases mismatch with ⟨lane, laneLt, laneNe⟩
  exact laneNe (bound.lane lane laneLt)

/-- The delayed step closes the corresponding old-parent mismatch obligation;
the open Π_RLC/state bridge determines how its next outputs seed another step. -/
theorem not_delayedParentProjectionMismatch_of_step
    {shape : Shape} {radix : F}
    {stateParent checkedOld : DelayedParent}
    {rawRunningChildren nextAssignments : List (List F)}
    {nextSCol : List K} {nextOutputs : List YZcol}
    (step : DelayedProjectionStep shape radix stateParent checkedOld
      rawRunningChildren nextAssignments nextSCol nextOutputs) :
    ¬ DelayedParentProjectionMismatch
      shape radix stateParent rawRunningChildren :=
  not_delayedParentProjectionMismatch_of_bound
    (delayedParentProjectionBound_of_step step)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection
