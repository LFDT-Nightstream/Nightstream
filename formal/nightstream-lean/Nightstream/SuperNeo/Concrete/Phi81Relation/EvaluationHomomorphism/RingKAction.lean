import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear

/-!
Boolean-MLE compatibility with the concrete Phi81 `RingK` action.

Protocol: SuperNeo Theorem 5 and the evaluation branch of `Pi_RLC`.
Phase: fixed ring action on every Boolean row, followed by evaluation at the
verifier-owned extension-field point.
Constraint family: semantic evaluation only; this file emits no rows.

Owns: coefficientwise placement of a `RingF` challenge in `RingK`; exact
right-linearity of the concrete `ringKMul`; lane-wise evaluation of a table of
Phi81 ring rows; and the theorem that Boolean MLE commutes with multiplication
by any fixed `RingK` element.

Does not own: the action of a `RingF` challenge on the complete assignment,
the Phi81 coefficient-kernel action law, proof that coefficientwise embedding
preserves quotient-ring multiplication, commitments, transcripts, Rust,
R1CS, row removal, or counts.

Emits constraints: no.

Authority boundary: row values and the evaluation point are explicit. The
action result and every evaluated lane are definitions; no caller supplies a
linearity oracle. The challenge specialization uses only coefficientwise
`K.embed`, so this file does not silently assume the still-separate
`RingF -> RingK` quotient-multiplication theorem.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.evaluation_action.ringk.raw` | every raw-convolution coefficient is right-linear | derived | `rawMulCoeffK_zero`, `rawMulCoeffK_add`, `rawMulCoeffK_scale` |
| `nifs.pi_rlc.verify.evaluation_action.ringk.reduction` | Phi81 reduction preserves right-linearity | derived | `ringKMul_right_zero`, `ringKMul_right_add`, `ringKMul_right_scale` |
| `nifs.pi_rlc.verify.evaluation_action.mle` | row-wise fixed `RingK` action commutes with all 54 evaluated lanes | derived | `evaluateRows_action` |
| `nifs.pi_rlc.verify.evaluation_action.challenge_embed` | the MLE bridge accepts coefficientwise embedded `RingF` challenges | computed | `evaluateRows_embeddedChallenge_action` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Coefficientwise placement of one `RingF` challenge in the extension ring.
This definition alone does not assert preservation of quotient multiplication. -/
def embedChallenge (challenge : RingF) : RingK :=
  fun lane => K.embed (challenge lane)

/-- Coefficientwise multiplication of one `RingK` value by a `K` scalar. -/
def scale (scalar : K) (value : RingK) : RingK :=
  fun lane => K.mul scalar (value lane)

/-- Coefficientwise negation using the same operation as Boolean MLE. -/
def neg (value : RingK) : RingK :=
  fun lane => ConcreteCarrier.extensionOps.neg (value lane)

/-- Coefficientwise subtraction using the same derived operation as Boolean
MLE. -/
def sub (left right : RingK) : RingK :=
  fun lane => ConcreteCarrier.extensionOps.sub (left lane) (right lane)

/-- Ring-valued interpolation performed independently in all 54 lanes. -/
def interpolate (coordinate : K) (low high : RingK) : RingK :=
  ringKAdd low (scale coordinate (sub high low))

/-- One explicit `RingK` value at every Boolean row. -/
abbrev Rows (variables : Nat) := BooleanVertex variables -> RingK

/-- Evaluate all 54 coefficient tables at one typed extension-field point. -/
def evaluateRows {variables : Nat}
    (rows : Rows variables) (point : CubePoint K variables) : RingK :=
  fun lane =>
    (BooleanTable.tabulate (fun vertex => rows vertex lane)).evaluate
      ConcreteCarrier.extensionOps point

/-- Apply one fixed extension-ring element independently at every Boolean row. -/
def actRows {variables : Nat}
    (scalar : RingK) (rows : Rows variables) : Rows variables :=
  fun vertex => ringKMul scalar (rows vertex)

local instance : Std.Associative K.add :=
  ⟨ConcreteCarrier.extensionLaws.add_assoc⟩

local instance : Std.Commutative K.add :=
  ⟨ConcreteCarrier.extensionLaws.add_comm⟩

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨ConcreteCarrier.baseLaws.add_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨ConcreteCarrier.baseLaws.add_comm⟩

private theorem k_mul_comm (left right : K) :
    K.mul left right = K.mul right left :=
  ConcreteCarrier.extensionLaws.mul_comm left right

private theorem k_left_distrib (left middle right : K) :
    K.mul left (K.add middle right) =
      K.add (K.mul left middle) (K.mul left right) :=
  ConcreteCarrier.extensionLaws.left_distrib left middle right

private theorem k_one_mul (value : K) :
    K.mul ConcreteCarrier.extensionOps.one value = value :=
  ConcreteCarrier.extensionLaws.one_mul value

private theorem k_zero_add (value : K) : K.add K.zero value = value :=
  ConcreteCarrier.extensionLaws.zero_add value

private theorem k_mul_zero (value : K) : K.mul value K.zero = K.zero :=
  ConcreteCarrier.extensionLaws.mul_zero value

private theorem f_neg_add (left right : F) :
    -(left + right) = -left + -right :=
  ConcreteCarrier.baseLaws.neg_add left right

/-! ## Raw convolution linearity -/

private def sumTerms (indices : List Nat) (term : Nat -> K) : K :=
  match indices with
  | [] => K.zero
  | index :: rest => K.add (term index) (sumTerms rest term)

private theorem foldl_eq_add_sumTerms
    (indices : List Nat) (term : Nat -> K) (initial : K) :
    indices.foldl (fun accumulated index => K.add accumulated (term index)) initial =
      K.add initial (sumTerms indices term) := by
  induction indices generalizing initial with
  | nil => exact (ConcreteCarrier.extensionLaws.add_zero initial).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact ConcreteCarrier.extensionLaws.add_assoc initial (term index)
        (sumTerms indices term)

private theorem sumTerms_zero (indices : List Nat) :
    sumTerms indices (fun _ => K.zero) = K.zero := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [sumTerms, inductionHypothesis]
      exact ConcreteCarrier.extensionLaws.zero_add K.zero

private theorem sumTerms_add
    (indices : List Nat) (left right : Nat -> K) :
    sumTerms indices (fun index => K.add (left index) (right index)) =
      K.add (sumTerms indices left) (sumTerms indices right) := by
  induction indices with
  | nil => exact (ConcreteCarrier.extensionLaws.zero_add K.zero).symm
  | cons index indices inductionHypothesis =>
      simp only [sumTerms, inductionHypothesis]
      ac_rfl

private theorem sumTerms_scale
    (indices : List Nat) (scalar : K) (term : Nat -> K) :
    sumTerms indices (fun index => K.mul scalar (term index)) =
      K.mul scalar (sumTerms indices term) := by
  induction indices with
  | nil => exact (ConcreteCarrier.extensionLaws.mul_zero scalar).symm
  | cons index indices inductionHypothesis =>
      simp only [sumTerms, inductionHypothesis]
      exact (ConcreteCarrier.extensionLaws.left_distrib scalar
        (term index) (sumTerms indices term)).symm

private theorem ringKCoeff_zero (degree : Nat) :
    ringKCoeff ringKZero degree = K.zero := by
  unfold ringKCoeff
  split <;> rfl

private theorem ringKCoeff_add (left right : RingK) (degree : Nat) :
    ringKCoeff (ringKAdd left right) degree =
      K.add (ringKCoeff left degree) (ringKCoeff right degree) := by
  unfold ringKCoeff ringKAdd
  split
  · rfl
  · exact (ConcreteCarrier.extensionLaws.zero_add K.zero).symm

private theorem ringKCoeff_scale
    (scalar : K) (value : RingK) (degree : Nat) :
    ringKCoeff (scale scalar value) degree =
      K.mul scalar (ringKCoeff value degree) := by
  unfold ringKCoeff scale
  split
  · rfl
  · exact (ConcreteCarrier.extensionLaws.mul_zero scalar).symm

private def rawTerm
    (left right : RingK) (degree index : Nat) : K :=
  if index <= degree ∧ degree - index < ringDegree then
    K.mul (ringKCoeff left index) (ringKCoeff right (degree - index))
  else
    K.zero

private theorem rawMulCoeffK_eq_sumTerms
    (left right : RingK) (degree : Nat) :
    rawMulCoeffK left right degree =
      sumTerms (List.range ringDegree) (rawTerm left right degree) := by
  unfold rawMulCoeffK
  have stepEquality :
      (fun accumulated index =>
        if index <= degree ∧ degree - index < ringDegree then
          K.add accumulated
            (K.mul (ringKCoeff left index)
              (ringKCoeff right (degree - index)))
        else accumulated) =
      (fun accumulated index =>
        K.add accumulated (rawTerm left right degree index)) := by
    funext accumulated index
    unfold rawTerm
    split
    · rfl
    · exact (ConcreteCarrier.extensionLaws.add_zero accumulated).symm
  rw [stepEquality, foldl_eq_add_sumTerms]
  exact ConcreteCarrier.extensionLaws.zero_add _

private theorem rawTerm_zero
    (left : RingK) (degree index : Nat) :
    rawTerm left ringKZero degree index = K.zero := by
  unfold rawTerm
  split
  · rw [ringKCoeff_zero]
    exact ConcreteCarrier.extensionLaws.mul_zero _
  · rfl

private theorem rawTerm_add
    (left right other : RingK) (degree index : Nat) :
    rawTerm left (ringKAdd right other) degree index =
      K.add (rawTerm left right degree index)
        (rawTerm left other degree index) := by
  unfold rawTerm
  split
  · rw [ringKCoeff_add]
    exact ConcreteCarrier.extensionLaws.left_distrib _ _ _
  · exact (ConcreteCarrier.extensionLaws.zero_add K.zero).symm

private theorem rawTerm_scale
    (left : RingK) (scalar : K) (right : RingK) (degree index : Nat) :
    rawTerm left (scale scalar right) degree index =
      K.mul scalar (rawTerm left right degree index) := by
  unfold rawTerm
  split
  · rw [ringKCoeff_scale]
    calc
      K.mul (ringKCoeff left index)
          (K.mul scalar (ringKCoeff right (degree - index))) =
        K.mul (K.mul (ringKCoeff left index) scalar)
          (ringKCoeff right (degree - index)) :=
            (ConcreteCarrier.extensionLaws.mul_assoc _ _ _).symm
      _ = K.mul (K.mul scalar (ringKCoeff left index))
          (ringKCoeff right (degree - index)) := by
            rw [k_mul_comm (ringKCoeff left index) scalar]
      _ = K.mul scalar
          (K.mul (ringKCoeff left index)
            (ringKCoeff right (degree - index))) :=
            ConcreteCarrier.extensionLaws.mul_assoc _ _ _
  · exact (ConcreteCarrier.extensionLaws.mul_zero scalar).symm

private theorem rawMulCoeffK_zero (left : RingK) (degree : Nat) :
    rawMulCoeffK left ringKZero degree = K.zero := by
  rw [rawMulCoeffK_eq_sumTerms]
  calc
    sumTerms (List.range ringDegree) (rawTerm left ringKZero degree) =
        sumTerms (List.range ringDegree) (fun _ => K.zero) := by
      congr 1
      funext index
      exact rawTerm_zero left degree index
    _ = K.zero := sumTerms_zero _

private theorem rawMulCoeffK_add
    (left right other : RingK) (degree : Nat) :
    rawMulCoeffK left (ringKAdd right other) degree =
      K.add (rawMulCoeffK left right degree)
        (rawMulCoeffK left other degree) := by
  rw [rawMulCoeffK_eq_sumTerms, rawMulCoeffK_eq_sumTerms,
    rawMulCoeffK_eq_sumTerms]
  calc
    sumTerms (List.range ringDegree)
        (rawTerm left (ringKAdd right other) degree) =
      sumTerms (List.range ringDegree)
        (fun index => K.add (rawTerm left right degree index)
          (rawTerm left other degree index)) := by
        congr 1
        funext index
        exact rawTerm_add left right other degree index
    _ = _ := sumTerms_add _ _ _

private theorem rawMulCoeffK_scale
    (left : RingK) (scalar : K) (right : RingK) (degree : Nat) :
    rawMulCoeffK left (scale scalar right) degree =
      K.mul scalar (rawMulCoeffK left right degree) := by
  rw [rawMulCoeffK_eq_sumTerms, rawMulCoeffK_eq_sumTerms]
  calc
    sumTerms (List.range ringDegree) (rawTerm left (scale scalar right) degree) =
      sumTerms (List.range ringDegree)
        (fun index => K.mul scalar (rawTerm left right degree index)) := by
          congr 1
          funext index
          exact rawTerm_scale left scalar right degree index
    _ = _ := sumTerms_scale _ _ _

/-! ## Reduced quotient-ring linearity -/

private theorem reduction_add
    (left0 left1 folded0 folded1 twice0 twice1 : K) :
    K.add
        (K.sub (K.add left0 left1) (K.add folded0 folded1))
        (K.add twice0 twice1) =
      K.add (K.add (K.sub left0 folded0) twice0)
        (K.add (K.sub left1 folded1) twice1) := by
  rcases left0 with ⟨left00, left01⟩
  rcases left1 with ⟨left10, left11⟩
  rcases folded0 with ⟨folded00, folded01⟩
  rcases folded1 with ⟨folded10, folded11⟩
  rcases twice0 with ⟨twice00, twice01⟩
  rcases twice1 with ⟨twice10, twice11⟩
  simp only [K.add, K.sub, K.mk.injEq, Fin.sub_eq_add_neg]
  constructor <;> rw [f_neg_add] <;> ac_rfl

private theorem reduction_add_zero
    (left0 left1 folded0 folded1 : K) :
    K.add (K.sub (K.add left0 left1) (K.add folded0 folded1)) K.zero =
      K.add (K.add (K.sub left0 folded0) K.zero)
        (K.add (K.sub left1 folded1) K.zero) := by
  simpa only [k_zero_add] using
    (reduction_add left0 left1 folded0 folded1 K.zero K.zero)

private theorem mul_neg (left right : K) :
    K.mul left (ConcreteCarrier.extensionOps.neg right) =
      ConcreteCarrier.extensionOps.neg (K.mul left right) := by
  calc
    K.mul left (ConcreteCarrier.extensionOps.neg right) =
        K.mul (ConcreteCarrier.extensionOps.neg right) left :=
      ConcreteCarrier.extensionLaws.mul_comm _ _
    _ = ConcreteCarrier.extensionOps.neg (K.mul right left) :=
      ConcreteCarrier.extensionLaws.neg_mul _ _
    _ = ConcreteCarrier.extensionOps.neg (K.mul left right) := by
      rw [k_mul_comm right left]

private theorem reduction_scale
    (scalar left folded twice : K) :
    K.add (K.sub (K.mul scalar left) (K.mul scalar folded))
        (K.mul scalar twice) =
      K.mul scalar (K.add (K.sub left folded) twice) := by
  rw [← ConcreteCarrier.derived_sub_eq_concrete_sub,
    ← ConcreteCarrier.derived_sub_eq_concrete_sub]
  unfold InterpolationOps.sub
  rw [← mul_neg scalar folded]
  change K.add
      (K.add (K.mul scalar left)
        (K.mul scalar (ConcreteCarrier.extensionOps.neg folded)))
      (K.mul scalar twice) =
    K.mul scalar
      (K.add (K.add left (ConcreteCarrier.extensionOps.neg folded)) twice)
  rw [← k_left_distrib scalar left (ConcreteCarrier.extensionOps.neg folded)]
  exact (k_left_distrib scalar _ twice).symm

private theorem reduction_scale_zero
    (scalar left folded : K) :
    K.add (K.sub (K.mul scalar left) (K.mul scalar folded)) K.zero =
      K.mul scalar (K.add (K.sub left folded) K.zero) := by
  simpa only [k_mul_zero] using
    (reduction_scale scalar left folded K.zero)

/-- Concrete Phi81 multiplication maps the zero right input to zero. -/
theorem ringKMul_right_zero (left : RingK) :
    ringKMul left ringKZero = ringKZero := by
  funext output
  simp only [ringKMul, rawMulCoeffK_zero]
  split <;> split <;> rfl

/-- Concrete Phi81 multiplication is additive in its right input. -/
theorem ringKMul_right_add (left right other : RingK) :
    ringKMul left (ringKAdd right other) =
      ringKAdd (ringKMul left right) (ringKMul left other) := by
  funext output
  change ringKMul left (ringKAdd right other) output =
    K.add (ringKMul left right output) (ringKMul left other output)
  unfold ringKMul
  simp only [rawMulCoeffK_add]
  split <;> split
  all_goals first
    | exact reduction_add _ _ _ _ _ _
    | exact reduction_add_zero _ _ _ _

/-- Concrete Phi81 multiplication commutes with scaling its right input by
an arbitrary quadratic-extension scalar. -/
theorem ringKMul_right_scale
    (left : RingK) (scalar : K) (right : RingK) :
    ringKMul left (scale scalar right) =
      scale scalar (ringKMul left right) := by
  funext output
  change ringKMul left (scale scalar right) output =
    K.mul scalar (ringKMul left right output)
  unfold ringKMul
  simp only [rawMulCoeffK_scale]
  split <;> split
  all_goals first
    | exact reduction_scale _ _ _ _
    | exact reduction_scale_zero _ _ _

private theorem scale_neg_one (value : RingK) :
    scale (ConcreteCarrier.extensionOps.neg ConcreteCarrier.extensionOps.one)
        value = neg value := by
  funext lane
  unfold scale neg
  calc
    K.mul (ConcreteCarrier.extensionOps.neg ConcreteCarrier.extensionOps.one)
        (value lane) =
      ConcreteCarrier.extensionOps.neg
        (K.mul ConcreteCarrier.extensionOps.one (value lane)) :=
          ConcreteCarrier.extensionLaws.neg_mul _ _
    _ = ConcreteCarrier.extensionOps.neg (value lane) := by
      rw [k_one_mul]

private theorem ringKMul_right_neg (left right : RingK) :
    ringKMul left (neg right) = neg (ringKMul left right) := by
  rw [← scale_neg_one right, ringKMul_right_scale,
    scale_neg_one (ringKMul left right)]

private theorem ringKMul_right_sub (left right other : RingK) :
    ringKMul left (sub right other) =
      sub (ringKMul left right) (ringKMul left other) := by
  change ringKMul left (ringKAdd right (neg other)) =
    ringKAdd (ringKMul left right) (neg (ringKMul left other))
  rw [ringKMul_right_add, ringKMul_right_neg]

private theorem ringKMul_interpolate
    (left : RingK) (coordinate : K) (low high : RingK) :
    ringKMul left (interpolate coordinate low high) =
      interpolate coordinate (ringKMul left low) (ringKMul left high) := by
  unfold interpolate
  rw [ringKMul_right_add, ringKMul_right_scale, ringKMul_right_sub]

/-! ## Boolean-MLE action theorem -/

private def evaluateRowsCoordinates {variables : Nat}
    (rows : Rows variables) (coordinates : List K) : RingK :=
  fun lane =>
    (BooleanTable.tabulate (fun vertex => rows vertex lane)).evaluateCoordinates
      ConcreteCarrier.extensionOps coordinates

private theorem evaluateRowsCoordinates_action
    (left : RingK) {variables : Nat}
    (rows : Rows variables) (coordinates : List K) :
    evaluateRowsCoordinates (actRows left rows) coordinates =
      ringKMul left (evaluateRowsCoordinates rows coordinates) := by
  induction variables generalizing coordinates with
  | zero =>
      cases coordinates with
      | nil => rfl
      | cons coordinate coordinates =>
          exact (ringKMul_right_zero left).symm
  | succ variables inductionHypothesis =>
      cases coordinates with
      | nil => exact (ringKMul_right_zero left).symm
      | cons coordinate coordinates =>
          change interpolate coordinate
              (evaluateRowsCoordinates
                (fun tail => ringKMul left (rows (.cons false tail))) coordinates)
              (evaluateRowsCoordinates
                (fun tail => ringKMul left (rows (.cons true tail))) coordinates) =
            ringKMul left
              (interpolate coordinate
                (evaluateRowsCoordinates
                  (fun tail => rows (.cons false tail)) coordinates)
                (evaluateRowsCoordinates
                  (fun tail => rows (.cons true tail)) coordinates))
          have low := inductionHypothesis
            (fun tail => rows (.cons false tail)) coordinates
          have high := inductionHypothesis
            (fun tail => rows (.cons true tail)) coordinates
          unfold actRows at low high
          rw [low, high]
          exact (ringKMul_interpolate left coordinate _ _).symm

/-- Boolean MLE of row-wise multiplication by any fixed `RingK` element is
exactly multiplication of the lane-wise evaluated row ring. -/
theorem evaluateRows_action
    {variables : Nat}
    (left : RingK) (rows : Rows variables) (point : CubePoint K variables) :
    evaluateRows (actRows left rows) point =
      ringKMul left (evaluateRows rows point) := by
  unfold evaluateRows BooleanTable.evaluate
  exact evaluateRowsCoordinates_action left rows point.coordinates

/-- The exact MLE bridge used when a `Pi_RLC` challenge is placed
coefficientwise in `RingK`. This does not assert the separate carrier-action
bridge from complete assignments to row rings. -/
theorem evaluateRows_embeddedChallenge_action
    {variables : Nat}
    (challenge : RingF) (rows : Rows variables)
    (point : CubePoint K variables) :
    evaluateRows (actRows (embedChallenge challenge) rows) point =
      ringKMul (embedChallenge challenge) (evaluateRows rows point) :=
  evaluateRows_action (embedChallenge challenge) rows point

end Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction
