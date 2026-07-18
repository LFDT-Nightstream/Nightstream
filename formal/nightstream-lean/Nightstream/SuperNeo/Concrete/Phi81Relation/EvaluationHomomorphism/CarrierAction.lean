import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear

/-!
Complete-carrier algebra behind the Phi81 `Pi_RLC` assignment action.

Protocol: SuperNeo Theorem 5, assignment-side `RingF` action.
Phase: complete 54-lane carrier block to derived Phi81 coefficient image.
Constraint family: semantic coefficient algebra only; this file emits no rows.

Owns: reuse of the canonical complete-carrier block/lane inverse; the fixed
blockwise `Concrete.ringFMul` action; base-field bilinearity of that multiplication;
finite monomial reconstruction of every `RingF` value; and extension of the
canonical basis-defined `phi81Kernel` image to multiplication by `barBasis`.

Does not own: associativity or commutativity of `Concrete.ringFMul`, the final
`Pi_RLC` commutation theorem, `RingF -> RingK` placement, Boolean MLE,
commitments, transcripts, Rust, R1CS, row removal, or counts.

Emits constraints: no.

Authority boundary: every action input is the typed complete assignment.
`carrierColumn` is total because `Shape.carrierWidth` is a whole number of
Phi81 blocks. The coefficient image is derived from the fixed
`phi81Kernel`; no caller supplies a second matrix or evaluation view. That
kernel is itself defined from executable multiplication on coefficient bases,
so this file proves basis-extension consistency, not independent Rust bar/cache
conformance or the quotient-ring law later proved in `RingFLaws`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.assignment_action.carrier.decode` | reuse `Phi81CarrierLayout.carrierColumn`; flattening and decoding are exact inverses | computed | `carrierColumn`, `decode_carrierColumn` |
| `nifs.pi_rlc.verify.assignment_action.block` | one 54-lane block uses the fixed `Concrete.ringFMul` action | computed | `assignmentBlock_act` |
| `nifs.pi_rlc.verify.assignment_action.ringf.linearity` | executable multiplication is additive and base-`F` scalable in both inputs | derived | `ringFMul_add_left`, `ringFMul_add_right`, `ringFMul_scale_left`, `ringFMul_scale_right` |
| `nifs.pi_rlc.verify.assignment_action.kernel_image` | the canonical basis-defined coefficient image extends exactly to `barBasis * block` | derived | `kernelImage_eq_ringFMul` |
| `nifs.pi_rlc.verify.assignment_action.commutation` | `bar * (rho * z) = rho * (bar * z)` | derived | `RingFLaws.ringFMul_barBasis_productOrder` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

/-! ## Fixed complete-carrier action -/

/-- Base-field scaling of every coefficient of a Phi81 ring value. -/
def ringFScale (scalar : F) (value : RingF) : RingF :=
  fun lane => scalar * value lane

/-- A complete assignment depends only on the logical width and its canonical
Phi81 completion, not on public-input or matrix dimensions. -/
abbrev CompleteAssignment (logicalWidth : Nat) :=
  PaperLinearAlgebra.Assignment F
    (Phi81CarrierLayout.carrierWidth logicalWidth)

/-- The complete carrier column owned by one block/lane pair. This is the
canonical layout owner shared with the independent Split-NC semantics. -/
abbrev carrierColumn {logicalWidth : Nat} :=
  @Phi81CarrierLayout.carrierColumn logicalWidth

/-- Read one complete 54-lane assignment block. -/
def assignmentBlock {logicalWidth : Nat}
    (assignment : CompleteAssignment logicalWidth)
    (block : Fin (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth))) : RingF :=
  fun lane => assignment (carrierColumn block lane)

/-- Fixed blockwise `RingF` action on the complete assignment carrier. -/
def act {logicalWidth : Nat}
    (challenge : RingF) (assignment : CompleteAssignment logicalWidth) :
    CompleteAssignment logicalWidth :=
  fun column =>
    let packed := Phi81ColumnLayout.decode column
    ringFMul challenge (assignmentBlock assignment packed.1) packed.2

/-- Complete-carrier flattening followed by decoding recovers the exact
block/lane pair. -/
theorem decode_carrierColumn {logicalWidth : Nat}
    (block : Fin (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth)))
    (lane : Fin ringDegree) :
    Phi81ColumnLayout.decode (carrierColumn block lane) = (block, lane) := by
  exact Phi81CarrierLayout.decode_carrierColumn block lane

/-- Acting on a complete assignment and reading one block is exactly one
application of the fixed Phi81 multiplication. No padding/default case is
possible. -/
theorem assignmentBlock_act {logicalWidth : Nat}
    (challenge : RingF) (assignment : CompleteAssignment logicalWidth)
    (block : Fin (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth))) :
    assignmentBlock (act challenge assignment) block =
      ringFMul challenge (assignmentBlock assignment block) := by
  funext lane
  unfold assignmentBlock act
  rw [decode_carrierColumn]
  rfl

/-! ## Finite-sum support -/

private def sumTerms
    {Value Index : Type}
    (zero : Value) (add : Value -> Value -> Value) :
    List Index -> (Index -> Value) -> Value
  | [], _ => zero
  | index :: indices, term =>
      add (term index) (sumTerms zero add indices term)

private theorem foldl_eq_add_sumTerms
    {Value Index : Type}
    (zero : Value) (add : Value -> Value -> Value)
    (addAssoc : forall left middle right,
      add (add left middle) right = add left (add middle right))
    (addZero : forall value, add value zero = value)
    (indices : List Index) (term : Index -> Value) (initial : Value) :
    indices.foldl (fun accumulated index => add accumulated (term index)) initial =
      add initial (sumTerms zero add indices term) := by
  induction indices generalizing initial with
  | nil => exact (addZero initial).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact addAssoc initial (term index) (sumTerms zero add indices term)

private theorem sumTerms_add
    {Value Index : Type}
    (zero : Value) (add : Value -> Value -> Value)
    (addAssoc : forall left middle right,
      add (add left middle) right = add left (add middle right))
    (addComm : forall left right, add left right = add right left)
    (zeroAdd : forall value, add zero value = value)
    (indices : List Index) (left right : Index -> Value) :
    sumTerms zero add indices (fun index => add (left index) (right index)) =
      add (sumTerms zero add indices left) (sumTerms zero add indices right) := by
  induction indices with
  | nil => exact (zeroAdd zero).symm
  | cons index indices inductionHypothesis =>
      simp only [sumTerms, inductionHypothesis]
      letI : Std.Associative add := ⟨addAssoc⟩
      letI : Std.Commutative add := ⟨addComm⟩
      ac_rfl

private theorem sumTerms_scale
    {Value Scalar Index : Type}
    (zero : Value) (add : Value -> Value -> Value)
    (scale : Scalar -> Value -> Value)
    (zeroScale : forall scalar, scale scalar zero = zero)
    (scaleAdd : forall scalar left right,
      scale scalar (add left right) =
        add (scale scalar left) (scale scalar right))
    (indices : List Index) (scalar : Scalar) (term : Index -> Value) :
    sumTerms zero add indices (fun index => scale scalar (term index)) =
      scale scalar (sumTerms zero add indices term) := by
  induction indices with
  | nil => exact (zeroScale scalar).symm
  | cons index indices inductionHypothesis =>
      simp only [sumTerms, inductionHypothesis, scaleAdd]

private def rawTerm (left right : RingF) (degree index : Nat) : F :=
  if index ≤ degree ∧ degree - index < ringDegree then
    ringFCoeff left index * ringFCoeff right (degree - index)
  else
    0

private theorem rawMulCoeffF_eq_sumTerms
    (left right : RingF) (degree : Nat) :
    rawMulCoeffF left right degree =
      sumTerms 0 (fun a b : F => a + b) (List.range ringDegree)
        (rawTerm left right degree) := by
  unfold rawMulCoeffF
  let add : F -> F -> F := fun a b => a + b
  let indices := List.range ringDegree
  let term := rawTerm left right degree
  have folded :
      indices.foldl (fun accumulated index => add accumulated (term index)) 0 =
        sumTerms 0 add indices term := by
    calc
      _ = 0 + sumTerms 0 add indices term :=
        foldl_eq_add_sumTerms 0 add ConcreteCarrier.baseLaws.add_assoc
          ConcreteCarrier.baseLaws.add_zero indices term 0
      _ = _ := ConcreteCarrier.baseLaws.zero_add _
  have stepEquality :
      (fun accumulated index =>
        if index ≤ degree ∧ degree - index < ringDegree then
          accumulated +
            ringFCoeff left index * ringFCoeff right (degree - index)
        else
          accumulated) =
      (fun accumulated index => add accumulated (term index)) := by
    funext accumulated index
    unfold add term rawTerm
    split
    · rfl
    · exact (ConcreteCarrier.baseLaws.add_zero accumulated).symm
  rw [stepEquality]
  exact folded

private theorem rawMulCoeffF_add_right
    (left right₁ right₂ : RingF) (degree : Nat) :
    rawMulCoeffF left (ringFAdd right₁ right₂) degree =
      rawMulCoeffF left right₁ degree + rawMulCoeffF left right₂ degree := by
  rw [rawMulCoeffF_eq_sumTerms, rawMulCoeffF_eq_sumTerms,
    rawMulCoeffF_eq_sumTerms]
  let add : F -> F -> F := fun a b => a + b
  let leftTerm := rawTerm left right₁ degree
  let rightTerm := rawTerm left right₂ degree
  have termAdd :
      rawTerm left (ringFAdd right₁ right₂) degree =
        fun index => add (leftTerm index) (rightTerm index) := by
    funext index
    by_cases active : index ≤ degree ∧ degree - index < ringDegree
    · simp only [rawTerm, if_pos active, leftTerm, rightTerm, add,
        ringFCoeff, dif_pos active.2, ringFAdd]
      exact ConcreteCarrier.baseLaws.left_distrib _ _ _
    · simp only [rawTerm, if_neg active, leftTerm, rightTerm, add]
      exact (ConcreteCarrier.baseLaws.zero_add 0).symm
  rw [termAdd]
  exact sumTerms_add 0 add ConcreteCarrier.baseLaws.add_assoc
    ConcreteCarrier.baseLaws.add_comm ConcreteCarrier.baseLaws.zero_add
    (List.range ringDegree) leftTerm rightTerm

private theorem rawMulCoeffF_add_left
    (left₁ left₂ right : RingF) (degree : Nat) :
    rawMulCoeffF (ringFAdd left₁ left₂) right degree =
      rawMulCoeffF left₁ right degree + rawMulCoeffF left₂ right degree := by
  rw [rawMulCoeffF_eq_sumTerms, rawMulCoeffF_eq_sumTerms,
    rawMulCoeffF_eq_sumTerms]
  let add : F -> F -> F := fun a b => a + b
  let leftTerm := rawTerm left₁ right degree
  let rightTerm := rawTerm left₂ right degree
  have termAdd :
      rawTerm (ringFAdd left₁ left₂) right degree =
        fun index => add (leftTerm index) (rightTerm index) := by
    funext index
    by_cases active : index ≤ degree ∧ degree - index < ringDegree
    · by_cases indexLt : index < ringDegree
      · simp only [rawTerm, if_pos active, leftTerm, rightTerm, add,
          ringFCoeff, dif_pos indexLt, ringFAdd]
        exact ConcreteCarrier.baseLaws.right_distrib _ _ _
      · simp only [rawTerm, if_pos active, leftTerm, rightTerm, add,
          ringFCoeff, dif_neg indexLt]
        simp only [Fin.zero_mul, Fin.zero_add]
    · simp only [rawTerm, if_neg active, leftTerm, rightTerm, add]
      exact (ConcreteCarrier.baseLaws.zero_add 0).symm
  rw [termAdd]
  exact sumTerms_add 0 add ConcreteCarrier.baseLaws.add_assoc
    ConcreteCarrier.baseLaws.add_comm ConcreteCarrier.baseLaws.zero_add
    (List.range ringDegree) leftTerm rightTerm

private theorem rawMulCoeffF_scale_right
    (left : RingF) (scalar : F) (right : RingF) (degree : Nat) :
    rawMulCoeffF left (ringFScale scalar right) degree =
      scalar * rawMulCoeffF left right degree := by
  rw [rawMulCoeffF_eq_sumTerms, rawMulCoeffF_eq_sumTerms]
  let add : F -> F -> F := fun a b => a + b
  let scale : F -> F -> F := fun a b => a * b
  let term := rawTerm left right degree
  have termScale :
      rawTerm left (ringFScale scalar right) degree =
        fun index => scale scalar (term index) := by
    funext index
    by_cases active : index ≤ degree ∧ degree - index < ringDegree
    · simp only [rawTerm, if_pos active, term, scale,
        ringFCoeff, dif_pos active.2, ringFScale]
      calc
        ringFCoeff left index * (scalar * right ⟨degree - index, active.2⟩) =
            (ringFCoeff left index * scalar) *
              right ⟨degree - index, active.2⟩ :=
          (ConcreteCarrier.baseLaws.mul_assoc _ _ _).symm
        _ = (scalar * ringFCoeff left index) *
              right ⟨degree - index, active.2⟩ := by
          rw [Fin.mul_comm (ringFCoeff left index) scalar]
        _ = scalar * (ringFCoeff left index *
              right ⟨degree - index, active.2⟩) :=
          ConcreteCarrier.baseLaws.mul_assoc _ _ _
    · simp only [rawTerm, if_neg active, term, scale]
      exact (ConcreteCarrier.baseLaws.mul_zero scalar).symm
  rw [termScale]
  exact sumTerms_scale 0 add scale ConcreteCarrier.baseLaws.mul_zero
    ConcreteCarrier.baseLaws.left_distrib (List.range ringDegree) scalar term

private theorem rawMulCoeffF_scale_left
    (scalar : F) (left right : RingF) (degree : Nat) :
    rawMulCoeffF (ringFScale scalar left) right degree =
      scalar * rawMulCoeffF left right degree := by
  rw [rawMulCoeffF_eq_sumTerms, rawMulCoeffF_eq_sumTerms]
  let add : F -> F -> F := fun a b => a + b
  let scale : F -> F -> F := fun a b => a * b
  let term := rawTerm left right degree
  have termScale :
      rawTerm (ringFScale scalar left) right degree =
        fun index => scale scalar (term index) := by
    funext index
    by_cases active : index ≤ degree ∧ degree - index < ringDegree
    · by_cases indexLt : index < ringDegree
      · simp only [rawTerm, if_pos active, term, scale,
          ringFCoeff, dif_pos indexLt, dif_pos active.2, ringFScale]
        exact ConcreteCarrier.baseLaws.mul_assoc _ _ _
      · simp only [rawTerm, if_pos active, term, scale,
          ringFCoeff, dif_neg indexLt]
        simp only [Fin.zero_mul, Fin.mul_zero]
    · simp only [rawTerm, if_neg active, term, scale]
      exact (ConcreteCarrier.baseLaws.mul_zero scalar).symm
  rw [termScale]
  exact sumTerms_scale 0 add scale ConcreteCarrier.baseLaws.mul_zero
    ConcreteCarrier.baseLaws.left_distrib (List.range ringDegree) scalar term

/-! ## Bilinearity of the fixed executable multiplication -/

private theorem add_sub_pair
    (a₁ a₂ b₁ b₂ : F) :
    (a₁ + a₂) - (b₁ + b₂) =
      (a₁ - b₁) + (a₂ - b₂) := by
  simp only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_add]
  letI : Std.Associative (fun (a b : F) => a + b) :=
    ⟨ConcreteCarrier.baseLaws.add_assoc⟩
  letI : Std.Commutative (fun (a b : F) => a + b) :=
    ⟨ConcreteCarrier.baseLaws.add_comm⟩
  ac_rfl

private theorem add_sub_add_triple
    (a₁ a₂ b₁ b₂ c₁ c₂ : F) :
    (a₁ + a₂) - (b₁ + b₂) + (c₁ + c₂) =
      (a₁ - b₁ + c₁) + (a₂ - b₂ + c₂) := by
  rw [add_sub_pair]
  simp only [Fin.sub_eq_add_neg]
  letI : Std.Associative (fun (a b : F) => a + b) :=
    ⟨ConcreteCarrier.baseLaws.add_assoc⟩
  letI : Std.Commutative (fun (a b : F) => a + b) :=
    ⟨ConcreteCarrier.baseLaws.add_comm⟩
  ac_rfl

private theorem scale_sub
    (scalar a b : F) :
    scalar * a - scalar * b = scalar * (a - b) := by
  have mulNeg : scalar * -b = -(scalar * b) := by
    calc
      scalar * -b = (-b) * scalar := Fin.mul_comm _ _
      _ = -(b * scalar) := Lean.Grind.Fin.neg_mul _ _
      _ = -(scalar * b) := by rw [Fin.mul_comm b scalar]
  calc
    scalar * a - scalar * b = scalar * a + -(scalar * b) :=
      Fin.sub_eq_add_neg _ _
    _ = scalar * a + scalar * -b := by rw [mulNeg]
    _ = scalar * (a + -b) :=
      (ConcreteCarrier.baseLaws.left_distrib scalar a (-b)).symm
    _ = scalar * (a - b) := by rw [Fin.sub_eq_add_neg]

private theorem scale_sub_add
    (scalar a b c : F) :
    scalar * a - scalar * b + scalar * c = scalar * (a - b + c) := by
  rw [scale_sub]
  exact (ConcreteCarrier.baseLaws.left_distrib scalar (a - b) c).symm

private theorem add_sub_pair_zero
    (a₁ a₂ b₁ b₂ : F) :
    (a₁ + a₂) - (b₁ + b₂) + 0 =
      (a₁ - b₁ + 0) + (a₂ - b₂ + 0) := by
  rw [Fin.add_zero, Fin.add_zero, Fin.add_zero]
  exact add_sub_pair _ _ _ _

private theorem scale_sub_zero
    (scalar a b : F) :
    scalar * a - scalar * b + 0 = scalar * (a - b + 0) := by
  rw [Fin.add_zero, Fin.add_zero]
  exact scale_sub _ _ _

/-- The fixed Phi81 multiplication is additive in its right input. -/
theorem ringFMul_add_right (left right₁ right₂ : RingF) :
    ringFMul left (ringFAdd right₁ right₂) =
      ringFAdd (ringFMul left right₁) (ringFMul left right₂) := by
  funext output
  simp only [ringFMul, ringFAdd]
  split <;> split <;> simp only [rawMulCoeffF_add_right]
  · exact add_sub_add_triple _ _ _ _ _ _
  · exact add_sub_pair_zero _ _ _ _
  · exact add_sub_add_triple _ _ _ _ _ _
  · exact add_sub_pair_zero _ _ _ _

/-- The fixed Phi81 multiplication is additive in its left input. -/
theorem ringFMul_add_left (left₁ left₂ right : RingF) :
    ringFMul (ringFAdd left₁ left₂) right =
      ringFAdd (ringFMul left₁ right) (ringFMul left₂ right) := by
  funext output
  simp only [ringFMul, ringFAdd]
  split <;> split <;> simp only [rawMulCoeffF_add_left]
  · exact add_sub_add_triple _ _ _ _ _ _
  · exact add_sub_pair_zero _ _ _ _
  · exact add_sub_add_triple _ _ _ _ _ _
  · exact add_sub_pair_zero _ _ _ _

/-- The fixed Phi81 multiplication commutes with base-field scaling in its
right input. -/
theorem ringFMul_scale_right (left : RingF) (scalar : F) (right : RingF) :
    ringFMul left (ringFScale scalar right) =
      ringFScale scalar (ringFMul left right) := by
  funext output
  simp only [ringFMul, ringFScale]
  split <;> split <;> simp only [rawMulCoeffF_scale_right]
  · exact scale_sub_add _ _ _ _
  · exact scale_sub_zero _ _ _
  · exact scale_sub_add _ _ _ _
  · exact scale_sub_zero _ _ _

/-- The fixed Phi81 multiplication commutes with base-field scaling in its
left input. -/
theorem ringFMul_scale_left (scalar : F) (left right : RingF) :
    ringFMul (ringFScale scalar left) right =
      ringFScale scalar (ringFMul left right) := by
  funext output
  simp only [ringFMul, ringFScale]
  split <;> split <;> simp only [rawMulCoeffF_scale_left]
  · exact scale_sub_add _ _ _ _
  · exact scale_sub_zero _ _ _
  · exact scale_sub_add _ _ _ _
  · exact scale_sub_zero _ _ _

/-- Multiplication by the additive identity is the additive identity. -/
theorem ringFMul_zero_right (left : RingF) :
    ringFMul left ringFZero = ringFZero := by
  have zeroScale (value : RingF) : ringFScale 0 value = ringFZero := by
    funext output
    calc
      0 * value output = value output * 0 := Fin.mul_comm _ _
      _ = 0 := ConcreteCarrier.baseLaws.mul_zero _
  calc
    ringFMul left ringFZero = ringFMul left (ringFScale 0 ringFZero) := by
      rw [zeroScale]
    _ = ringFScale 0 (ringFMul left ringFZero) :=
      ringFMul_scale_right left 0 ringFZero
    _ = ringFZero := zeroScale _

/-! ## Finite coefficient-basis reconstruction -/

private def ringFSumRange : Nat -> (Nat -> RingF) -> RingF
  | 0, _ => ringFZero
  | count + 1, term =>
      ringFAdd (ringFSumRange count term) (term count)

private theorem ringFSumRange_apply
    (count : Nat) (term : Nat -> RingF) (output : Fin ringDegree) :
    ringFSumRange count term output =
      sumRange ConcreteCarrier.baseOps count (fun index => term index output) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [ringFSumRange, ringFAdd, sumRange, ConcreteCarrier.baseOps,
        inductionHypothesis]

private theorem ringFSumRange_congr
    (count : Nat) (left right : Nat -> RingF)
    (equal : forall index, index < count -> left index = right index) :
    ringFSumRange count left = ringFSumRange count right := by
  funext output
  rw [ringFSumRange_apply, ringFSumRange_apply]
  exact sumRange_congr ConcreteCarrier.baseOps count
    (fun index => left index output) (fun index => right index output)
    (fun index indexLt => congrFun (equal index indexLt) output)

private theorem ringFMul_ringFSumRange_right
    (left : RingF) (count : Nat) (term : Nat -> RingF) :
    ringFMul left (ringFSumRange count term) =
      ringFSumRange count (fun index => ringFMul left (term index)) := by
  induction count with
  | zero => exact ringFMul_zero_right left
  | succ count inductionHypothesis =>
      rw [ringFSumRange, ringFSumRange, ringFMul_add_right,
        inductionHypothesis]

private def basisTerm (value : RingF) (index : Nat) : RingF :=
  if indexLt : index < ringDegree then
    ringFMonomial index (value ⟨index, indexLt⟩)
  else
    ringFZero

private def basisExpansion (value : RingF) : RingF :=
  ringFSumRange ringDegree (basisTerm value)

/-- Every fixed-size Phi81 value is exactly the finite sum of its 54
coefficient monomials. -/
theorem basisExpansion_eq (value : RingF) :
    basisExpansion value = value := by
  funext output
  unfold basisExpansion
  rw [ringFSumRange_apply]
  calc
    sumRange ConcreteCarrier.baseOps ringDegree
        (fun index => basisTerm value index output) =
      sumRange ConcreteCarrier.baseOps ringDegree
        (fun index => if index = output.val then value output else 0) := by
      apply sumRange_congr
      intro index indexLt
      unfold basisTerm
      rw [dif_pos indexLt]
      unfold ringFMonomial
      by_cases equal : index = output.val
      · rw [if_pos equal, if_pos equal.symm]
        apply congrArg value
        apply Fin.ext
        exact equal
      · rw [if_neg equal, if_neg]
        exact Ne.symm equal
    _ = value output := by
      simpa using
        (sumRange_select ConcreteCarrier.baseOps ConcreteCarrier.baseLaws
          ringDegree output.val (fun _ => value output) output.isLt)

private theorem basisTerm_as_scaled_monomial
    (value : RingF) (index : Nat) (indexLt : index < ringDegree) :
    basisTerm value index =
      ringFScale (value ⟨index, indexLt⟩) (ringFMonomial index 1) := by
  funext output
  unfold basisTerm ringFScale ringFMonomial
  rw [dif_pos indexLt]
  split
  · exact (ConcreteCarrier.baseLaws.mul_one _).symm
  · exact (ConcreteCarrier.baseLaws.mul_zero _).symm

/-! ## Canonical basis-kernel image -/

/-- Finite coefficient image of one transformed matrix-row basis acting on
one complete assignment block. This is the canonical `phi81Kernel`
contraction, not a caller-supplied second product. -/
def kernelImage (row : Fin ringDegree) (block : RingF) : RingF :=
  ringFSumRange ringDegree fun index =>
    if indexLt : index < ringDegree then
      ringFScale (block ⟨index, indexLt⟩) fun output =>
        Phi81CoefficientKernel.phi81Kernel.weight output row
          ⟨index, indexLt⟩
    else
      ringFZero

/-- Extending the basis-defined Phi81 coefficient kernel over all 54 input
lanes gives exactly the fixed executable product `barBasis(row) * block`. -/
theorem kernelImage_eq_ringFMul
    (row : Fin ringDegree) (block : RingF) :
    kernelImage row block =
      ringFMul (Phi81CoefficientKernel.barBasis row) block := by
  calc
    kernelImage row block =
        ringFSumRange ringDegree
          (fun index =>
            ringFMul (Phi81CoefficientKernel.barBasis row)
              (basisTerm block index)) := by
      apply ringFSumRange_congr
      intro index indexLt
      rw [dif_pos indexLt,
        basisTerm_as_scaled_monomial block index indexLt,
        ringFMul_scale_right]
      funext output
      rfl
    _ = ringFMul (Phi81CoefficientKernel.barBasis row)
          (basisExpansion block) := by
      exact (ringFMul_ringFSumRange_right
        (Phi81CoefficientKernel.barBasis row) ringDegree
        (basisTerm block)).symm
    _ = ringFMul (Phi81CoefficientKernel.barBasis row) block := by
      rw [basisExpansion_eq]

end Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction
