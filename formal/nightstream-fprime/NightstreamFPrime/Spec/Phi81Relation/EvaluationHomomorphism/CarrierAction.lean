import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.BaseLinear

/-! SuperNeo v1.2 Theorem 11. -/

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.CarrierAction

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource


def ringFScale (scalar : F) (value : RingF) : RingF :=
  fun lane => scalar * value lane

abbrev CompleteAssignment (logicalWidth : Nat) :=
  PaperLinearAlgebra.Assignment F
    (Phi81CarrierLayout.carrierWidth logicalWidth)

abbrev carrierColumn {logicalWidth : Nat} :=
  @Phi81CarrierLayout.carrierColumn logicalWidth

def assignmentBlock {logicalWidth : Nat}
    (assignment : CompleteAssignment logicalWidth)
    (block : Fin (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth))) : RingF :=
  fun lane => assignment (carrierColumn block lane)

def act {logicalWidth : Nat}
    (challenge : RingF) (assignment : CompleteAssignment logicalWidth) :
    CompleteAssignment logicalWidth :=
  fun column =>
    let packed := Phi81ColumnLayout.decode column
    ringFMul challenge (assignmentBlock assignment packed.1) packed.2

theorem decode_carrierColumn {logicalWidth : Nat}
    (block : Fin (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth)))
    (lane : Fin ringDegree) :
    Phi81ColumnLayout.decode (carrierColumn block lane) = (block, lane) := by
  exact Phi81CarrierLayout.decode_carrierColumn block lane

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
    · simp only [rawTerm, ite_eq_left active, leftTerm, rightTerm, add,
        ringFCoeff, dite_eq_left active.2, ringFAdd]
      exact ConcreteCarrier.baseLaws.left_distrib _ _ _
    · simp only [rawTerm, ite_eq_right active, leftTerm, rightTerm, add]
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
      · simp only [rawTerm, ite_eq_left active, leftTerm, rightTerm, add,
          ringFCoeff, dite_eq_left indexLt, ringFAdd]
        exact ConcreteCarrier.baseLaws.right_distrib _ _ _
      · simp only [rawTerm, ite_eq_left active, leftTerm, rightTerm, add,
          ringFCoeff, dite_eq_right indexLt]
        simp only [Fin.zero_mul, Fin.zero_add]
    · simp only [rawTerm, ite_eq_right active, leftTerm, rightTerm, add]
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
    · simp only [rawTerm, ite_eq_left active, term, scale,
        ringFCoeff, dite_eq_left active.2, ringFScale]
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
    · simp only [rawTerm, ite_eq_right active, term, scale]
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
      · simp only [rawTerm, ite_eq_left active, term, scale,
          ringFCoeff, dite_eq_left indexLt, dite_eq_left active.2, ringFScale]
        exact ConcreteCarrier.baseLaws.mul_assoc _ _ _
      · simp only [rawTerm, ite_eq_left active, term, scale,
          ringFCoeff, dite_eq_right indexLt]
        simp only [Fin.zero_mul, Fin.mul_zero]
    · simp only [rawTerm, ite_eq_right active, term, scale]
      exact (ConcreteCarrier.baseLaws.mul_zero scalar).symm
  rw [termScale]
  exact sumTerms_scale 0 add scale ConcreteCarrier.baseLaws.mul_zero
    ConcreteCarrier.baseLaws.left_distrib (List.range ringDegree) scalar term

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
      rw [dite_eq_left indexLt]
      unfold ringFMonomial
      by_cases equal : index = output.val
      · rw [ite_eq_left equal, ite_eq_left equal.symm]
        apply congrArg value
        apply Fin.ext
        exact equal
      · rw [ite_eq_right equal, ite_eq_right]
        exact Ne.symm equal
    _ = value output := by
      simpa using!
        (sumRange_select ConcreteCarrier.baseOps ConcreteCarrier.baseLaws
          ringDegree output.val (fun _ => value output) output.isLt)

private theorem basisTerm_as_scaled_monomial
    (value : RingF) (index : Nat) (indexLt : index < ringDegree) :
    basisTerm value index =
      ringFScale (value ⟨index, indexLt⟩) (ringFMonomial index 1) := by
  funext output
  unfold basisTerm ringFScale ringFMonomial
  rw [dite_eq_left indexLt]
  split
  · exact (ConcreteCarrier.baseLaws.mul_one _).symm
  · exact (ConcreteCarrier.baseLaws.mul_zero _).symm

def rightCoefficient
    (left : RingF)
    (output input : Fin ringDegree) : F :=
  ringFMul left (ringFMonomial input.val 1) output

theorem ringFMul_apply_eq_rightLinear
    (left right : RingF)
    (output : Fin ringDegree) :
    ringFMul left right output =
      sumRange ConcreteCarrier.baseOps ringDegree fun index =>
        if indexLt : index < ringDegree then
          rightCoefficient left output ⟨index, indexLt⟩ *
            right ⟨index, indexLt⟩
        else
          0 := by
  calc
    ringFMul left right output =
        ringFMul left (basisExpansion right) output := by
      rw [basisExpansion_eq]
    _ = ringFSumRange ringDegree
          (fun index => ringFMul left (basisTerm right index)) output := by
      unfold basisExpansion
      rw [ringFMul_ringFSumRange_right]
    _ = sumRange ConcreteCarrier.baseOps ringDegree
          (fun index => ringFMul left (basisTerm right index) output) :=
      ringFSumRange_apply _ _ _
    _ = sumRange ConcreteCarrier.baseOps ringDegree
          (fun index =>
            if indexLt : index < ringDegree then
              rightCoefficient left output ⟨index, indexLt⟩ *
                right ⟨index, indexLt⟩
            else
              0) := by
      apply sumRange_congr
      intro index indexLt
      rw [basisTerm_as_scaled_monomial right index indexLt,
        ringFMul_scale_right, dite_eq_left indexLt]
      unfold rightCoefficient ringFScale
      exact ConcreteCarrier.baseLaws.mul_comm _ _

def kernelImage (row : Fin ringDegree) (block : RingF) : RingF :=
  ringFSumRange ringDegree fun index =>
    if indexLt : index < ringDegree then
      ringFScale (block ⟨index, indexLt⟩) fun output =>
        Phi81CoefficientKernel.phi81Kernel.weight output row
          ⟨index, indexLt⟩
    else
      ringFZero

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
      rw [dite_eq_left indexLt,
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

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.CarrierAction
