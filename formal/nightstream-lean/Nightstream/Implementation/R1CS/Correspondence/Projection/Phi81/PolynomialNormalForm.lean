import Nightstream.Implementation.R1CS.Correspondence.Projection.Phi81.Carrier
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Profile-neutral exact normal form for one Phi81 projection polynomial.

Assurance tier: model-level. The only semantic premise is exact coefficient
equality; no generated trace, quotient witness, digest, or profile is trusted.

| Stage family | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `projection.identity.quotient` | `X^54 + X^27 + 1` gives one 54-lane remainder | derived | `exact_output_eq_remainder` |
| `projection.identity.product` | embedded schoolbook multiplication is `ringFMul` | derived | `product_remainder_eq_ringFMul` |

Owns: list-polynomial remainder uniqueness and executable Phi81 product
interpretation. Does not own: sampled-root soundness, trace membership, column
binding, costs, or Rust/R1CS conformance. Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.ProjectionPhi81.PolynomialNormalForm

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.ProjectionPhi81

set_option maxRecDepth 16384
set_option maxHeartbeats 4000000

private local instance : Std.Associative ProjectionProgram.K.add :=
  ⟨ProjectionProgram.K.add_assoc⟩

private local instance : Std.Commutative ProjectionProgram.K.add :=
  ⟨ProjectionProgram.K.add_comm⟩

private local instance : Std.Associative (fun left right : Scalar => left + right) :=
  ⟨Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_assoc⟩

private local instance : Std.Commutative (fun left right : Scalar => left + right) :=
  ⟨Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_comm⟩

def shift : Nat → List ProjectionProgram.K → List ProjectionProgram.K
  | 0, polynomial => polynomial
  | count + 1, polynomial => ProjectionProgram.K.zero :: shift count polynomial

private theorem polynomial_add_assoc
    (left middle right : List ProjectionProgram.K) :
    Polynomial.add (Polynomial.add left middle) right =
      Polynomial.add left (Polynomial.add middle right) := by
  induction left generalizing middle right with
  | nil => rfl
  | cons leftHead leftTail inductionHypothesis =>
      cases middle with
      | nil => rfl
      | cons middleHead middleTail =>
          cases right with
          | nil => rfl
          | cons rightHead rightTail =>
              simp only [Polynomial.add, List.cons.injEq]
              exact ⟨ProjectionProgram.K.add_assoc _ _ _,
                inductionHypothesis middleTail rightTail⟩

private theorem polynomial_add_comm
    (left right : List ProjectionProgram.K) :
    Polynomial.add left right = Polynomial.add right left := by
  induction left generalizing right with
  | nil => cases right <;> rfl
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => rfl
      | cons rightHead rightTail =>
          simp only [Polynomial.add, List.cons.injEq]
          exact ⟨ProjectionProgram.K.add_comm _ _,
            inductionHypothesis rightTail⟩

private local instance : Std.Associative Polynomial.add :=
  ⟨polynomial_add_assoc⟩

private local instance : Std.Commutative Polynomial.add :=
  ⟨polynomial_add_comm⟩

private theorem scale_add (scalar : ProjectionProgram.K)
    (left right : List ProjectionProgram.K) :
    Polynomial.scale scalar (Polynomial.add left right) =
      Polynomial.add (Polynomial.scale scalar left)
        (Polynomial.scale scalar right) := by
  induction left generalizing right with
  | nil => rfl
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => rfl
      | cons rightHead rightTail =>
          simp only [Polynomial.add, Polynomial.scale, List.cons.injEq]
          exact ⟨ProjectionProgram.K.mul_add _ _ _,
            inductionHypothesis rightTail⟩

private theorem add_shift (count : Nat)
    (left right : List ProjectionProgram.K) :
    Polynomial.add (shift count left) (shift count right) =
      shift count (Polynomial.add left right) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [shift, Polynomial.add, ProjectionProgram.K.zero_add,
        List.cons.injEq]
      exact ⟨trivial, inductionHypothesis⟩

private theorem mul_add_right (left middle right : List ProjectionProgram.K) :
    Polynomial.mul left (Polynomial.add middle right) =
      Polynomial.add (Polynomial.mul left middle)
        (Polynomial.mul left right) := by
  induction left with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [Polynomial.mul, scale_add, inductionHypothesis]
      have shifted :
          ProjectionProgram.K.zero ::
              Polynomial.add (Polynomial.mul tail middle)
                (Polynomial.mul tail right) =
            Polynomial.add
              (ProjectionProgram.K.zero :: Polynomial.mul tail middle)
              (ProjectionProgram.K.zero :: Polynomial.mul tail right) := by
        simpa [shift] using (add_shift 1
          (Polynomial.mul tail middle) (Polynomial.mul tail right)).symm
      rw [shifted]
      ac_rfl

private theorem mul_one (polynomial : List ProjectionProgram.K) :
    Polynomial.mul polynomial [ProjectionProgram.K.one] = polynomial := by
  induction polynomial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [Polynomial.mul, Polynomial.scale, Polynomial.add,
        inductionHypothesis]

private theorem polynomial_add_nil_right
    (polynomial : List ProjectionProgram.K) :
    Polynomial.add polynomial [] = polynomial := by
  cases polynomial <;> rfl

private theorem mul_cons_zero (left right : List ProjectionProgram.K)
    (leftNonempty : left ≠ []) (rightNonempty : right ≠ []) :
    Polynomial.mul left (ProjectionProgram.K.zero :: right) =
      ProjectionProgram.K.zero :: Polynomial.mul left right := by
  induction left generalizing right with
  | nil => exact (leftNonempty rfl).elim
  | cons head tail inductionHypothesis =>
      cases right with
      | nil => exact (rightNonempty rfl).elim
      | cons rightHead rightTail =>
          cases tail with
          | nil =>
              change ProjectionProgram.K.zero ::
                    Polynomial.add
                      (Polynomial.scale head (rightHead :: rightTail)) [] =
                  ProjectionProgram.K.zero ::
                    Polynomial.add
                      (Polynomial.scale head (rightHead :: rightTail))
                      [ProjectionProgram.K.zero]
              rw [polynomial_add_nil_right]
              simp [Polynomial.scale, Polynomial.add]
              exact (polynomial_add_nil_right _).symm
          | cons tailHead tailTail =>
              change Polynomial.add
                    (ProjectionProgram.K.mul head ProjectionProgram.K.zero ::
                      Polynomial.scale head (rightHead :: rightTail))
                    (ProjectionProgram.K.zero ::
                      Polynomial.mul (tailHead :: tailTail)
                        (ProjectionProgram.K.zero :: rightHead :: rightTail)) =
                  ProjectionProgram.K.zero ::
                    Polynomial.add
                      (Polynomial.scale head (rightHead :: rightTail))
                      (ProjectionProgram.K.zero ::
                        Polynomial.mul (tailHead :: tailTail)
                          (rightHead :: rightTail))
              rw [ProjectionProgram.K.mul_zero,
                inductionHypothesis (rightHead :: rightTail)
                  (by simp) (by simp)]
              rfl

private theorem mul_shift_monomial (count : Nat)
    (polynomial : List ProjectionProgram.K) (nonempty : polynomial ≠ []) :
    Polynomial.mul polynomial (shift count [ProjectionProgram.K.one]) =
      shift count polynomial := by
  induction count with
  | zero => exact mul_one polynomial
  | succ count inductionHypothesis =>
      change Polynomial.mul polynomial
          (ProjectionProgram.K.zero :: shift count [ProjectionProgram.K.one]) =
        ProjectionProgram.K.zero :: shift count polynomial
      rw [mul_cons_zero polynomial _ nonempty (by cases count <;> simp [shift]),
        inductionHypothesis]

private theorem phi81_as_monomials :
    Polynomial.phi81 =
      Polynomial.add [ProjectionProgram.K.one]
        (Polynomial.add (shift 27 [ProjectionProgram.K.one])
          (shift 54 [ProjectionProgram.K.one])) := by
  decide

theorem mul_phi81 (polynomial : List ProjectionProgram.K)
    (nonempty : polynomial ≠ []) :
    Polynomial.mul polynomial Polynomial.phi81 =
      Polynomial.add polynomial
        (Polynomial.add (shift 27 polynomial) (shift 54 polynomial)) := by
  rw [phi81_as_monomials, mul_add_right, mul_add_right, mul_one,
    mul_shift_monomial 27 polynomial nonempty,
    mul_shift_monomial 54 polynomial nonempty]

def coefficient (polynomial : List ProjectionProgram.K) (degree : Nat) :
    ProjectionProgram.K :=
  polynomial.getD degree ProjectionProgram.K.zero

@[simp] private theorem coefficient_nil (degree : Nat) :
    coefficient [] degree = ProjectionProgram.K.zero := by
  simp [coefficient]

private theorem coefficient_add (left right : List ProjectionProgram.K)
    (degree : Nat) :
    coefficient (Polynomial.add left right) degree =
      ProjectionProgram.K.add (coefficient left degree)
        (coefficient right degree) := by
  induction left generalizing right degree with
  | nil => simp [coefficient, Polynomial.add]
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp [coefficient, Polynomial.add]
      | cons rightHead rightTail =>
          cases degree with
          | zero => rfl
          | succ degree =>
              simpa [coefficient, Polynomial.add] using
                inductionHypothesis rightTail degree

private theorem coefficient_shift (count : Nat)
    (polynomial : List ProjectionProgram.K) (degree : Nat) :
    coefficient (shift count polynomial) degree =
      if count ≤ degree then coefficient polynomial (degree - count)
      else ProjectionProgram.K.zero := by
  induction count generalizing degree with
  | zero => simp [shift]
  | succ count inductionHypothesis =>
      cases degree with
      | zero => rfl
      | succ degree =>
          simp only [shift, coefficient, List.getD_cons_succ]
          change coefficient (shift count polynomial) degree = _
          rw [inductionHypothesis]
          simp only [Nat.succ_le_succ_iff, Nat.succ_sub_succ_eq_sub]
          rfl

private theorem coefficient_append (left right : List ProjectionProgram.K)
    (degree : Nat) :
    coefficient (left ++ right) degree =
      if degree < left.length then coefficient left degree
      else coefficient right (degree - left.length) := by
  induction left generalizing degree with
  | nil => simp [coefficient]
  | cons head tail inductionHypothesis =>
      cases degree with
      | zero => rfl
      | succ degree =>
          simp only [List.cons_append, coefficient, List.getD_cons_succ,
            List.length_cons, Nat.succ_lt_succ_iff,
            Nat.succ_sub_succ_eq_sub]
          change coefficient (tail ++ right) degree = _
          exact inductionHypothesis degree

private theorem coefficient_replicate_zero (count degree : Nat) :
    coefficient (List.replicate count ProjectionProgram.K.zero) degree =
      ProjectionProgram.K.zero := by
  induction count generalizing degree with
  | zero => simp [coefficient]
  | succ count inductionHypothesis =>
      cases degree with
      | zero => rfl
      | succ degree =>
          simpa [coefficient, List.replicate_succ] using
            inductionHypothesis degree

private theorem coefficient_append_zeroes
    (polynomial : List ProjectionProgram.K) (count degree : Nat) :
    coefficient
        (polynomial ++ List.replicate count ProjectionProgram.K.zero) degree =
      coefficient polynomial degree := by
  rw [coefficient_append]
  split
  · rfl
  · rename_i outside
    rw [coefficient_replicate_zero]
    apply Eq.symm
    rw [coefficient, List.getD_eq_getElem?_getD,
      List.getElem?_eq_none (Nat.le_of_not_gt outside)]
    rfl

private theorem coefficient_padRight (width : Nat)
    (polynomial : List ProjectionProgram.K) (degree : Nat) :
    coefficient (Polynomial.padRight width polynomial) degree =
      coefficient polynomial degree := by
  exact coefficient_append_zeroes polynomial
    (width - polynomial.length) degree

private theorem coefficient_eq_zero_of_length_le
    (polynomial : List ProjectionProgram.K) (degree : Nat)
    (outside : polynomial.length ≤ degree) :
    coefficient polynomial degree = ProjectionProgram.K.zero := by
  rw [coefficient, List.getD_eq_getElem?_getD,
    List.getElem?_eq_none outside]
  rfl

def remainderRing (raw : List ProjectionProgram.K) : Concrete.RingF :=
  fun output =>
    let degree := output.val
    let rawAt := fun index => (coefficient raw index).c0
    if degree < Concrete.ringMiddleDegree then
      rawAt degree - rawAt (degree + Concrete.ringDegree) +
        rawAt (degree + 81)
    else
      rawAt degree - rawAt (degree + Concrete.ringMiddleDegree)

private theorem add_sub_pair
    (a₁ a₂ b₁ b₂ : Scalar) :
    (a₁ + a₂) - (b₁ + b₂) =
      (a₁ - b₁) + (a₂ - b₂) := by
  simp only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_add]
  ac_rfl

private theorem add_sub_add_triple
    (a₁ a₂ b₁ b₂ c₁ c₂ : Scalar) :
    (a₁ + a₂) - (b₁ + b₂) + (c₁ + c₂) =
      (a₁ - b₁ + c₁) + (a₂ - b₂ + c₂) := by
  rw [add_sub_pair]
  simp only [Fin.sub_eq_add_neg]
  ac_rfl

theorem remainderRing_add
    (left right : List ProjectionProgram.K) :
    remainderRing (Polynomial.add left right) =
      Concrete.ringFAdd (remainderRing left) (remainderRing right) := by
  funext output
  unfold remainderRing Concrete.ringFAdd
  simp only [coefficient_add, ProjectionProgram.K.add]
  split
  · exact add_sub_add_triple _ _ _ _ _ _
  · exact add_sub_pair _ _ _ _

theorem remainderRing_nil :
    remainderRing [] = Concrete.ringFZero := by
  funext output
  unfold remainderRing Concrete.ringFZero
  change (if output.val < Concrete.ringMiddleDegree then
      (0 : Scalar) - 0 + 0 else (0 : Scalar) - 0) = 0
  split <;> simp

private theorem low_cancel (a b output : Scalar) :
    (a + output) - (a + b) + b = output := by
  simp only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_add]
  have cancelA : a + -a = 0 := by
    calc
      a + -a = -a + a := Lean.Grind.Fin.add_comm _ _
      _ = 0 := Lean.Grind.Fin.neg_add_cancel a
  have cancelB : -b + b = 0 := Lean.Grind.Fin.neg_add_cancel b
  calc
    a + output + (-(a) + -b) + b =
        (a + -a) + (output + (-b + b)) := by ac_rfl
    _ = output := by rw [cancelA, cancelB, Fin.zero_add, Fin.add_zero]

private theorem high_cancel (a b output : Scalar) :
    (a + b + output) - (a + b) = output := by
  simp only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_add]
  have cancelA : a + -a = 0 := by
    calc
      a + -a = -a + a := Lean.Grind.Fin.add_comm _ _
      _ = 0 := Lean.Grind.Fin.neg_add_cancel a
  have cancelB : b + -b = 0 := by
    calc
      b + -b = -b + b := Lean.Grind.Fin.add_comm _ _
      _ = 0 := Lean.Grind.Fin.neg_add_cancel b
  calc
    a + b + output + (-(a) + -b) =
        (a + -a) + (b + -b) + output := by ac_rfl
    _ = output := by rw [cancelA, cancelB, Fin.zero_add, Fin.zero_add]

private theorem exact_coefficient_equation
    (raw quotient output : List ProjectionProgram.K)
    (quotientNonempty : quotient ≠ [])
    (exact : raw = Polynomial.add
      (Polynomial.mul quotient Polynomial.phi81)
      (Polynomial.padRight 107 output))
    (degree : Nat) :
    coefficient raw degree = ProjectionProgram.K.add
      (ProjectionProgram.K.add (coefficient quotient degree)
        (ProjectionProgram.K.add
          (if 27 ≤ degree then coefficient quotient (degree - 27)
            else ProjectionProgram.K.zero)
          (if 54 ≤ degree then coefficient quotient (degree - 54)
            else ProjectionProgram.K.zero)))
      (coefficient output degree) := by
  rw [exact, coefficient_add, mul_phi81 quotient quotientNonempty,
    coefficient_add, coefficient_add,
    coefficient_shift, coefficient_shift, coefficient_padRight]

/-- Exact coefficient equality with the fixed quotient/output widths uniquely
determines the concrete Phi81 remainder. -/
theorem exact_output_eq_remainder
    (raw quotient output : List ProjectionProgram.K)
    (quotientWidth : quotient.length = 53)
    (outputWidth : output.length = 54)
    (exact : raw = Polynomial.add
      (Polynomial.mul quotient Polynomial.phi81)
      (Polynomial.padRight 107 output)) :
    List.map ProjectionProgram.K.c0 output =
      List.ofFn (remainderRing raw) := by
  apply List.ext_getElem
  · calc
      (List.map ProjectionProgram.K.c0 output).length = output.length := by
        simp
      _ = 54 := outputWidth
      _ = Concrete.ringDegree := by rfl
      _ = (List.ofFn (remainderRing raw)).length :=
        (List.length_ofFn).symm
  · intro index leftLt rightLt
    have outputLt : index < output.length := by
      simpa only [List.length_map] using leftLt
    have indexLt : index < 54 := by simpa [outputWidth] using outputLt
    let outputIndex : Fin Concrete.ringDegree :=
      ⟨index, by simpa [Concrete.ringDegree] using indexLt⟩
    have outputTailZero : ∀ degree, 54 ≤ degree →
        coefficient output degree = ProjectionProgram.K.zero := by
      intro degree outside
      apply coefficient_eq_zero_of_length_le
      simpa [outputWidth] using outside
    have quotientTailZero : ∀ degree, 53 ≤ degree →
        coefficient quotient degree = ProjectionProgram.K.zero := by
      intro degree outside
      apply coefficient_eq_zero_of_length_le
      simpa [quotientWidth] using outside
    have quotientNonempty : quotient ≠ [] := by
      intro empty
      rw [empty] at quotientWidth
      simp at quotientWidth
    have low := exact_coefficient_equation raw quotient output
      quotientNonempty exact index
    by_cases firstHalf : index < 27
    · have middle := exact_coefficient_equation raw quotient output
        quotientNonempty exact
        (index + 54)
      have high := exact_coefficient_equation raw quotient output
          quotientNonempty exact
          (index + 81)
      have qIndex54 :
          coefficient quotient (index + 54) = ProjectionProgram.K.zero :=
        quotientTailZero _ (by omega)
      have qIndex81 :
          coefficient quotient (index + 81) = ProjectionProgram.K.zero :=
        quotientTailZero _ (by omega)
      have qMiddleTail :
          coefficient quotient (index + 54 - 27) =
            coefficient quotient (index + 27) := by
        congr 2 <;> omega
      have qMiddleBase :
          coefficient quotient (index + 54 - 54) =
            coefficient quotient index := by
        congr 2 <;> omega
      have qHighMiddle :
          coefficient quotient (index + 81 - 27) =
            ProjectionProgram.K.zero := by
        apply quotientTailZero
        omega
      have qHighBase :
          coefficient quotient (index + 81 - 54) =
            coefficient quotient (index + 27) := by
        congr 2 <;> omega
      have outMiddle : coefficient output (index + 54) =
          ProjectionProgram.K.zero := outputTailZero _ (by omega)
      have outHigh : coefficient output (index + 81) =
          ProjectionProgram.K.zero := outputTailZero _ (by omega)
      simp only [if_neg (by omega : ¬ 27 ≤ index),
        if_neg (by omega : ¬ 54 ≤ index),
        ProjectionProgram.K.zero_add] at low
      simp only [if_pos (by omega : 27 ≤ index + 54),
        if_pos (by omega : 54 ≤ index + 54), qIndex54,
        qMiddleTail, qMiddleBase, outMiddle,
        ProjectionProgram.K.zero_add, ProjectionProgram.K.add_zero] at middle
      simp only [if_pos (by omega : 27 ≤ index + 81),
        if_pos (by omega : 54 ≤ index + 81), qIndex81,
        qHighMiddle, qHighBase, outHigh,
        ProjectionProgram.K.zero_add, ProjectionProgram.K.add_zero] at high
      have lowC0 := congrArg ProjectionProgram.K.c0 low
      have middleC0 := congrArg ProjectionProgram.K.c0 middle
      have highC0 := congrArg ProjectionProgram.K.c0 high
      simp only [ProjectionProgram.K.add, ProjectionProgram.K.zero,
        Fin.add_zero] at lowC0 middleC0 highC0
      have middleC0Ordered :
          (coefficient raw (index + 54)).c0 =
            (coefficient quotient index).c0 +
              (coefficient quotient (index + 27)).c0 :=
        middleC0.trans (Lean.Grind.Fin.add_comm _ _)
      simp only [List.getElem_map]
      calc
        output[index].c0 = remainderRing raw outputIndex := by
          have outputCoefficient :
              coefficient output index = output[index] := by
            rw [coefficient, List.getD_eq_getElem?_getD,
              List.getElem?_eq_getElem outputLt]
            rfl
          rw [← outputCoefficient]
          unfold remainderRing
          simp only [outputIndex, firstHalf, Concrete.ringMiddleDegree,
            Concrete.ringDegree, if_true]
          rw [lowC0, middleC0Ordered, highC0]
          exact (low_cancel _ _ _).symm
        _ = (List.ofFn (remainderRing raw))[index] := by
          exact (List.getElem_ofFn (f := remainderRing raw) rightLt).symm
    · have secondHalf : 27 ≤ index := Nat.le_of_not_gt firstHalf
      have middle := exact_coefficient_equation raw quotient output
        quotientNonempty exact
        (index + 27)
      have qMiddleDirect :
          coefficient quotient (index + 27) = ProjectionProgram.K.zero :=
        quotientTailZero _ (by omega)
      have qMiddleShift :
          coefficient quotient (index + 27 - 27) =
            coefficient quotient index := by
        congr 2 <;> omega
      have qMiddleBase :
          coefficient quotient (index + 27 - 54) =
            coefficient quotient (index - 27) := by
        have degreeEquality : index + 27 - 54 = index - 27 := by omega
        rw [degreeEquality]
      have outMiddle : coefficient output (index + 27) =
          ProjectionProgram.K.zero := outputTailZero _ (by omega)
      simp only [if_pos secondHalf,
        if_neg (by omega : ¬ 54 ≤ index)] at low
      simp only [if_pos (by omega : 27 ≤ index + 27),
        if_pos (by omega : 54 ≤ index + 27), qMiddleDirect,
        qMiddleShift, qMiddleBase, outMiddle,
        ProjectionProgram.K.zero_add, ProjectionProgram.K.add_zero] at middle
      have lowC0 := congrArg ProjectionProgram.K.c0 low
      have middleC0 := congrArg ProjectionProgram.K.c0 middle
      simp only [ProjectionProgram.K.add, ProjectionProgram.K.zero,
        Fin.add_zero] at lowC0 middleC0
      simp only [List.getElem_map]
      calc
        output[index].c0 = remainderRing raw outputIndex := by
          have outputCoefficient :
              coefficient output index = output[index] := by
            rw [coefficient, List.getD_eq_getElem?_getD,
              List.getElem?_eq_getElem outputLt]
            rfl
          rw [← outputCoefficient]
          unfold remainderRing
          simp only [outputIndex, Concrete.ringMiddleDegree,
            Concrete.ringDegree, firstHalf, if_false]
          rw [lowC0, middleC0]
          exact (high_cancel _ _ _).symm
        _ = (List.ofFn (remainderRing raw))[index] := by
          exact (List.getElem_ofFn (f := remainderRing raw) rightLt).symm

def embedded (coefficients : List Scalar) : List ProjectionProgram.K :=
  coefficients.map fun coefficient => ⟨coefficient, 0⟩

def scalarCoefficient (coefficients : List Scalar) (degree : Nat) :
    ProjectionProgram.F :=
  coefficients.getD degree 0

def convolutionRange (left right : Nat → ProjectionProgram.F)
    (degree offset count : Nat) : ProjectionProgram.F :=
  match count with
  | 0 => 0
  | count + 1 =>
      (if offset ≤ degree then left offset * right (degree - offset) else 0) +
        convolutionRange left right degree (offset + 1) count

private theorem convolutionRange_shift
    (left right : Nat → ProjectionProgram.F) (degree offset count : Nat) :
    convolutionRange left right (degree + 1) (offset + 1) count =
      convolutionRange (fun index => left (index + 1)) right
        degree offset count := by
  induction count generalizing offset with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [convolutionRange, Nat.succ_le_succ_iff,
        Nat.succ_sub_succ_eq_sub]
      rw [inductionHypothesis]

private theorem convolutionRange_offset_gt
    (left right : Nat → ProjectionProgram.F) (degree offset count : Nat)
    (outside : degree < offset) :
    convolutionRange left right degree offset count = 0 := by
  induction count generalizing offset with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [convolutionRange, if_neg (Nat.not_le.mpr outside)]
      simpa using inductionHypothesis (offset + 1) (by omega)

private theorem scalarCoefficient_eq_zero_of_length_le
    (coefficients : List Scalar) (degree : Nat)
    (outside : coefficients.length ≤ degree) :
    scalarCoefficient coefficients degree = 0 := by
  rw [scalarCoefficient, List.getD_eq_getElem?_getD,
    List.getElem?_eq_none outside]
  rfl

private theorem convolutionRange_eq_zero_of_right_outside
    (left : Nat → ProjectionProgram.F) (right : List Scalar)
    (degree offset count : Nat)
    (outside : right.length + offset + count ≤ degree + 1) :
    convolutionRange left (scalarCoefficient right)
      degree offset count = 0 := by
  induction count generalizing offset with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [convolutionRange]
      by_cases active : offset ≤ degree
      · rw [if_pos active,
          scalarCoefficient_eq_zero_of_length_le right (degree - offset)
            (by omega), Fin.mul_zero, Fin.zero_add]
        exact inductionHypothesis (offset + 1) (by omega)
      · rw [if_neg active, Fin.zero_add]
        exact inductionHypothesis (offset + 1) (by omega)

private theorem coefficient_embedded
    (coefficients : List Scalar) (degree : Nat) :
    (coefficient (embedded coefficients) degree).c0 =
      scalarCoefficient coefficients degree := by
  induction coefficients generalizing degree with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases degree with
      | zero => rfl
      | succ degree =>
          change (coefficient (embedded tail) degree).c0 =
            scalarCoefficient tail degree
          exact inductionHypothesis degree

private theorem coefficient_scale_embedded
    (scalar : ProjectionProgram.F)
    (polynomial : List ProjectionProgram.K) (degree : Nat) :
    (coefficient
      (Polynomial.scale ⟨scalar, 0⟩ polynomial) degree).c0 =
      scalar * (coefficient polynomial degree).c0 := by
  induction polynomial generalizing degree with
  | nil =>
      change (0 : ProjectionProgram.F) = scalar * 0
      exact (Fin.mul_zero scalar).symm
  | cons head tail inductionHypothesis =>
      cases degree with
      | zero =>
          change scalar * head.c0 + 7 * (0 * head.c1) = scalar * head.c0
          rw [Fin.zero_mul, Fin.mul_zero, Fin.add_zero]
      | succ degree =>
          simpa [coefficient, Polynomial.scale] using
            inductionHypothesis degree

private theorem coefficient_cons_zero_succ
    (polynomial : List ProjectionProgram.K) (degree : Nat) :
    coefficient (ProjectionProgram.K.zero :: polynomial) (degree + 1) =
      coefficient polynomial degree := by
  rfl

private theorem coefficient_mul_embedded
    (left right : List Scalar) (degree : Nat) :
    (coefficient (Polynomial.mul (embedded left) (embedded right)) degree).c0 =
      convolutionRange (scalarCoefficient left) (scalarCoefficient right)
        degree 0 left.length := by
  induction left generalizing degree with
  | nil =>
      change (0 : ProjectionProgram.F) = _
      simp [convolutionRange]
  | cons head tail inductionHypothesis =>
      have mulExpand :
          Polynomial.mul (embedded (head :: tail)) (embedded right) =
            Polynomial.add
              (Polynomial.scale ⟨head, 0⟩ (embedded right))
              (ProjectionProgram.K.zero ::
                Polynomial.mul (embedded tail) (embedded right)) := by
        rfl
      cases degree with
      | zero =>
          rw [mulExpand, coefficient_add]
          simp only [ProjectionProgram.K.add,
            coefficient_scale_embedded]
          rw [coefficient_embedded]
          simp [coefficient, convolutionRange, scalarCoefficient,
            convolutionRange_offset_gt, ProjectionProgram.K.zero]
      | succ degree =>
          rw [mulExpand, coefficient_add]
          simp only [ProjectionProgram.K.add, coefficient_scale_embedded,
            coefficient_cons_zero_succ]
          rw [coefficient_embedded, inductionHypothesis]
          simp only [convolutionRange, Nat.zero_le, if_true,
            scalarCoefficient, List.getD_cons_zero]
          have shifted := convolutionRange_shift
            (scalarCoefficient (head :: tail)) (scalarCoefficient right)
            degree 0 tail.length
          simp only [Nat.zero_add] at shifted
          rw [shifted]
          congr 1

private theorem foldl_convolutionRange
    (left right : Nat → ProjectionProgram.F) (degree offset count : Nat)
    (initial : ProjectionProgram.F) :
    (List.range' offset count).foldl
        (fun accumulated index =>
          if index ≤ degree then
            accumulated + left index * right (degree - index)
          else accumulated)
        initial =
      initial + convolutionRange left right degree offset count := by
  induction count generalizing offset initial with
  | zero => simp [convolutionRange]
  | succ count inductionHypothesis =>
      rw [List.range'_succ, List.foldl_cons]
      split
      · rw [inductionHypothesis]
        simp only [convolutionRange, if_pos ‹offset ≤ degree›]
        exact ProjectionProgram.fadd_assoc _ _ _
      · rw [inductionHypothesis]
        simp only [convolutionRange, if_neg ‹¬ offset ≤ degree›,
          Fin.zero_add]

private theorem raw_step_without_right_bound
    (left right : Concrete.RingF) (degree index : Nat)
    (accumulated : Scalar) :
    (if index ≤ degree ∧ degree - index < Concrete.ringDegree then
        accumulated + Concrete.ringFCoeff left index *
          Concrete.ringFCoeff right (degree - index)
      else accumulated) =
      (if index ≤ degree then
        accumulated + Concrete.ringFCoeff left index *
          Concrete.ringFCoeff right (degree - index)
      else accumulated) := by
  by_cases indexLe : index ≤ degree
  · by_cases rightBound : degree - index < Concrete.ringDegree
    · simp [indexLe, rightBound]
    · have rightZero : Concrete.ringFCoeff right (degree - index) = 0 := by
        simp [Concrete.ringFCoeff, rightBound]
      simp only [indexLe, rightBound, and_false, if_false, if_true,
        rightZero, Fin.mul_zero, Fin.add_zero]
  · simp [indexLe]

private theorem ringFCoeff_ringOfList
    (coefficients : List Scalar) (width : coefficients.length = 54) :
    (fun index => Concrete.ringFCoeff (ringOfList coefficients) index) =
      scalarCoefficient coefficients := by
  funext index
  unfold Concrete.ringFCoeff ringOfList scalarCoefficient
  split
  · rfl
  · rename_i outside
    have lengthLe : coefficients.length ≤ index := by
      simpa [width, Concrete.ringDegree] using Nat.le_of_not_gt outside
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none lengthLe]
    rfl

private theorem rawMulCoeffF_eq_embedded_coefficient
    (left right : List Scalar)
    (leftWidth : left.length = 54) (rightWidth : right.length = 54)
    (degree : Nat) :
    Concrete.rawMulCoeffF (ringOfList left) (ringOfList right) degree =
      (coefficient
        (Polynomial.mul (embedded left) (embedded right)) degree).c0 := by
  unfold Concrete.rawMulCoeffF
  have stepEquality :
      (fun accumulated index =>
        if index ≤ degree ∧ degree - index < Concrete.ringDegree then
          accumulated +
            Concrete.ringFCoeff (ringOfList left) index *
              Concrete.ringFCoeff (ringOfList right) (degree - index)
        else accumulated) =
      (fun accumulated index =>
        if index ≤ degree then
          accumulated +
            Concrete.ringFCoeff (ringOfList left) index *
              Concrete.ringFCoeff (ringOfList right) (degree - index)
        else accumulated) := by
    funext accumulated index
    exact raw_step_without_right_bound _ _ degree index accumulated
  rw [stepEquality]
  have folded := foldl_convolutionRange
    (fun index => Concrete.ringFCoeff (ringOfList left) index)
    (fun index => Concrete.ringFCoeff (ringOfList right) index)
    degree 0 Concrete.ringDegree 0
  change (List.range' 0 Concrete.ringDegree).foldl
      (fun accumulated index =>
        if index ≤ degree then
          accumulated +
            Concrete.ringFCoeff (ringOfList left) index *
              Concrete.ringFCoeff (ringOfList right) (degree - index)
        else accumulated) 0 = _
  calc
    _ = convolutionRange
        (fun index => Concrete.ringFCoeff (ringOfList left) index)
        (fun index => Concrete.ringFCoeff (ringOfList right) index)
        degree 0 Concrete.ringDegree := by
          simpa only [Fin.zero_add] using folded
    _ = convolutionRange (scalarCoefficient left) (scalarCoefficient right)
        degree 0 Concrete.ringDegree := by
          rw [ringFCoeff_ringOfList left leftWidth,
            ringFCoeff_ringOfList right rightWidth]
    _ = (coefficient
        (Polynomial.mul (embedded left) (embedded right)) degree).c0 := by
          rw [coefficient_mul_embedded]
          simp [leftWidth, Concrete.ringDegree]

private theorem embedded_product_coefficient_eq_zero_of_degree_gt
    (left right : List Scalar)
    (leftWidth : left.length = 54) (rightWidth : right.length = 54)
    (degree : Nat) (outside : 106 < degree) :
    (coefficient
      (Polynomial.mul (embedded left) (embedded right)) degree).c0 = 0 := by
  rw [coefficient_mul_embedded]
  apply convolutionRange_eq_zero_of_right_outside
  omega

/-- An exact embedded schoolbook product has the same canonical remainder as
the independently defined executable Phi81 multiplication. -/
theorem product_remainder_eq_ringFMul
    (left right : List Scalar)
    (leftWidth : left.length = 54) (rightWidth : right.length = 54) :
    remainderRing
        (Polynomial.mul (embedded left) (embedded right)) =
      Concrete.ringFMul (ringOfList left) (ringOfList right) := by
  funext output
  unfold remainderRing Concrete.ringFMul
  simp only [Concrete.ringMiddleDegree, Concrete.ringDegree]
  split
  · rw [rawMulCoeffF_eq_embedded_coefficient left right leftWidth rightWidth,
      rawMulCoeffF_eq_embedded_coefficient left right leftWidth rightWidth,
      rawMulCoeffF_eq_embedded_coefficient left right leftWidth rightWidth]
    by_cases twice : output.val + 81 ≤ 106
    · simp [twice] <;> rfl
    · have zero := embedded_product_coefficient_eq_zero_of_degree_gt
          left right leftWidth rightWidth (output.val + 81) (by omega)
      simp [twice, zero] <;> rfl
  · rw [rawMulCoeffF_eq_embedded_coefficient left right leftWidth rightWidth,
      rawMulCoeffF_eq_embedded_coefficient left right leftWidth rightWidth]
    have noTwice : ¬ output.val + 81 ≤ 106 := by omega
    simp [noTwice] <;> rfl

end Nightstream.Implementation.R1CS.ProjectionPhi81.PolynomialNormalForm
