import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.Embedding
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet

/-!
Owns the deterministic extraction algebra for the exact production
`PaperAlgebra` relation. It uses the same assignment, commitment,
public-input, and evaluation actions as `PaperAlgebra.piRlcAlgebra`.

This module does not own the strong-set invertibility theorem, a forking
probability, commitment binding, or any emitted rows.
-/

namespace NightstreamFPrime.Lifecycle.PaperExtractionAlgebra

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiRLC
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkAlgebra
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Lifecycle

private abbrev ScalarRing :=
  Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring

private def ringFModule : ModuleOps RingF RingF where
  zero := ringFZero
  add := ringFAdd
  neg := ScalarRing.neg
  smul := ringFMul

private theorem ringFModuleLaws : ModuleLaws ScalarRing ringFModule where
  add_assoc := Phi81Relation.PiRLCAlgebra.ForkStrongSet.ringLaws.add_assoc
  add_comm := Phi81Relation.PiRLCAlgebra.ForkStrongSet.ringLaws.add_comm
  zero_add := Phi81Relation.PiRLCAlgebra.ForkStrongSet.ringLaws.zero_add
  add_zero := Phi81Relation.PiRLCAlgebra.ForkStrongSet.ringLaws.add_zero
  add_neg := Phi81Relation.PiRLCAlgebra.ForkStrongSet.ringLaws.add_neg
  zero_smul := RingFLaws.ringFMul_zero_left
  add_smul := CarrierAction.ringFMul_add_left
  one_smul := RingFLaws.ringFMul_one_left
  mul_smul := RingFLaws.ringFMul_assoc
  smul_zero := CarrierAction.ringFMul_zero_right
  smul_add := CarrierAction.ringFMul_add_right

private def functionModule {Index Value : Type}
    (module : ModuleOps RingF Value) : ModuleOps RingF (Index -> Value) where
  zero := fun _ => module.zero
  add := fun left right index => module.add (left index) (right index)
  neg := fun value index => module.neg (value index)
  smul := fun scalar value index => module.smul scalar (value index)

private theorem functionModuleLaws {Index Value : Type}
    (module : ModuleOps RingF Value)
    (laws : ModuleLaws ScalarRing module) :
    ModuleLaws ScalarRing (functionModule (Index := Index) module) where
  add_assoc := by
    intro left middle right
    funext index
    exact laws.add_assoc _ _ _
  add_comm := by
    intro left right
    funext index
    exact laws.add_comm _ _
  zero_add := by
    intro value
    funext index
    exact laws.zero_add _
  add_zero := by
    intro value
    funext index
    exact laws.add_zero _
  add_neg := by
    intro value
    funext index
    exact laws.add_neg _
  zero_smul := by
    intro value
    funext index
    exact laws.zero_smul _
  add_smul := by
    intro left right value
    funext index
    exact laws.add_smul _ _ _
  one_smul := by
    intro value
    funext index
    exact laws.one_smul _
  mul_smul := by
    intro left right value
    funext index
    exact laws.mul_smul _ _ _
  smul_zero := by
    intro scalar
    funext index
    exact laws.smul_zero _
  smul_add := by
    intro scalar left right
    funext index
    exact laws.smul_add _ _ _

private def ringKComponent0 (value : RingK) : RingF :=
  fun lane => (value lane).c0

private def ringKComponent1 (value : RingK) : RingF :=
  fun lane => (value lane).c1

private theorem foldEmbedLeftComponent0
    (indices : List Nat) (left : RingF) (right : RingK)
    (degree : Nat) (initial : K) :
    (indices.foldl
        (fun accumulated index =>
          if index <= degree ∧ degree - index < ringDegree then
            K.add accumulated
              (K.mul
                (ringKCoeff (RingKAction.embedChallenge left) index)
                (ringKCoeff right (degree - index)))
          else accumulated)
        initial).c0 =
      indices.foldl
        (fun accumulated index =>
          if index <= degree ∧ degree - index < ringDegree then
            accumulated +
              ringFCoeff left index *
                ringFCoeff (ringKComponent0 right) (degree - index)
          else accumulated)
        initial.c0 := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.foldl_cons]
      by_cases active : index <= degree ∧ degree - index < ringDegree
      · simp only [if_pos active]
        rw [inductionHypothesis]
        congr 1
        unfold RingKAction.embedChallenge ringKComponent0
        simp only [ringKCoeff, ringFCoeff, K.add, K.mul, K.embed]
        split <;> split <;>
          simp [K.zero]
      · simp only [if_neg active]
        exact inductionHypothesis initial

private theorem foldEmbedLeftComponent1
    (indices : List Nat) (left : RingF) (right : RingK)
    (degree : Nat) (initial : K) :
    (indices.foldl
        (fun accumulated index =>
          if index <= degree ∧ degree - index < ringDegree then
            K.add accumulated
              (K.mul
                (ringKCoeff (RingKAction.embedChallenge left) index)
                (ringKCoeff right (degree - index)))
          else accumulated)
        initial).c1 =
      indices.foldl
        (fun accumulated index =>
          if index <= degree ∧ degree - index < ringDegree then
            accumulated +
              ringFCoeff left index *
                ringFCoeff (ringKComponent1 right) (degree - index)
          else accumulated)
        initial.c1 := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.foldl_cons]
      by_cases active : index <= degree ∧ degree - index < ringDegree
      · simp only [if_pos active]
        rw [inductionHypothesis]
        congr 1
        unfold RingKAction.embedChallenge ringKComponent1
        simp only [ringKCoeff, ringFCoeff, K.add, K.mul, K.embed]
        split <;> split <;>
          simp [K.zero]
      · simp only [if_neg active]
        exact inductionHypothesis initial

private theorem rawMulCoeffK_embedLeft_component0
    (left : RingF) (right : RingK) (degree : Nat) :
    (rawMulCoeffK (RingKAction.embedChallenge left) right degree).c0 =
      rawMulCoeffF left (ringKComponent0 right) degree := by
  simpa [rawMulCoeffK, rawMulCoeffF, K.zero] using
    (foldEmbedLeftComponent0 (List.range ringDegree) left right degree K.zero)

private theorem rawMulCoeffK_embedLeft_component1
    (left : RingF) (right : RingK) (degree : Nat) :
    (rawMulCoeffK (RingKAction.embedChallenge left) right degree).c1 =
      rawMulCoeffF left (ringKComponent1 right) degree := by
  simpa [rawMulCoeffK, rawMulCoeffF, K.zero] using
    (foldEmbedLeftComponent1 (List.range ringDegree) left right degree K.zero)

private theorem ringKMul_embedLeft_component0
    (left : RingF) (right : RingK) :
    ringKComponent0
        (ringKMul (RingKAction.embedChallenge left) right) =
      ringFMul left (ringKComponent0 right) := by
  funext lane
  simp only [ringKComponent0, ringKMul, ringFMul]
  split_ifs <;>
    simp [K.add, K.sub, K.zero, rawMulCoeffK_embedLeft_component0]

private theorem ringKMul_embedLeft_component1
    (left : RingF) (right : RingK) :
    ringKComponent1
        (ringKMul (RingKAction.embedChallenge left) right) =
      ringFMul left (ringKComponent1 right) := by
  funext lane
  simp only [ringKComponent1, ringKMul, ringFMul]
  split_ifs <;>
    simp [K.add, K.sub, K.zero, rawMulCoeffK_embedLeft_component1]

private theorem ringK_ext {left right : RingK}
    (component0 : ringKComponent0 left = ringKComponent0 right)
    (component1 : ringKComponent1 left = ringKComponent1 right) :
    left = right := by
  funext lane
  have equal0 := congrFun component0 lane
  have equal1 := congrFun component1 lane
  change (left lane).c0 = (right lane).c0 at equal0
  change (left lane).c1 = (right lane).c1 at equal1
  cases leftValue : left lane with
  | mk left0 left1 =>
      cases rightValue : right lane with
      | mk right0 right1 =>
          rw [leftValue, rightValue] at equal0 equal1
          exact congrArg₂ K.mk equal0 equal1

private def ringKModule : ModuleOps RingF RingK where
  zero := ringKZero
  add := ringKAdd
  neg := RingKAction.neg
  smul := fun scalar value =>
    ringKMul (RingKAction.embedChallenge scalar) value

private theorem ringKModuleLaws : ModuleLaws ScalarRing ringKModule where
  add_assoc := by
    intro left middle right
    change ringKAdd (ringKAdd left middle) right =
      ringKAdd left (ringKAdd middle right)
    apply ringK_ext <;> funext lane <;>
      exact ConcreteCarrier.baseLaws.add_assoc _ _ _
  add_comm := by
    intro left right
    change ringKAdd left right = ringKAdd right left
    apply ringK_ext <;> funext lane <;>
      exact ConcreteCarrier.baseLaws.add_comm _ _
  zero_add := by
    intro value
    change ringKAdd ringKZero value = value
    funext lane
    exact ConcreteCarrier.extensionLaws.zero_add _
  add_zero := by
    intro value
    change ringKAdd value ringKZero = value
    funext lane
    exact ConcreteCarrier.extensionLaws.add_zero _
  add_neg := by
    intro value
    change ringKAdd value (RingKAction.neg value) = ringKZero
    funext lane
    exact ConcreteCarrier.extensionLaws.add_neg _
  zero_smul := by
    intro value
    change ringKMul (RingKAction.embedChallenge ringFZero) value = ringKZero
    apply ringK_ext
    · calc
        ringKComponent0 (ringKMul (RingKAction.embedChallenge ringFZero) value) =
            ringFMul ringFZero (ringKComponent0 value) :=
          ringKMul_embedLeft_component0 _ _
        _ = ringFZero := RingFLaws.ringFMul_zero_left _
        _ = ringKComponent0 ringKZero := rfl
    · calc
        ringKComponent1 (ringKMul (RingKAction.embedChallenge ringFZero) value) =
            ringFMul ringFZero (ringKComponent1 value) :=
          ringKMul_embedLeft_component1 _ _
        _ = ringFZero := RingFLaws.ringFMul_zero_left _
        _ = ringKComponent1 ringKZero := rfl
  add_smul := by
    intro left right value
    change ringKMul (RingKAction.embedChallenge (ringFAdd left right)) value =
      ringKAdd
        (ringKMul (RingKAction.embedChallenge left) value)
        (ringKMul (RingKAction.embedChallenge right) value)
    apply ringK_ext
    · rw [ringKMul_embedLeft_component0,
        CarrierAction.ringFMul_add_left,
        ← ringKMul_embedLeft_component0 left value,
        ← ringKMul_embedLeft_component0 right value]
      rfl
    · rw [ringKMul_embedLeft_component1,
        CarrierAction.ringFMul_add_left,
        ← ringKMul_embedLeft_component1 left value,
        ← ringKMul_embedLeft_component1 right value]
      rfl
  one_smul := by
    intro value
    change ringKMul (RingKAction.embedChallenge ringFOne) value = value
    apply ringK_ext
    · rw [ringKMul_embedLeft_component0, RingFLaws.ringFMul_one_left]
    · rw [ringKMul_embedLeft_component1, RingFLaws.ringFMul_one_left]
  mul_smul := by
    intro left right value
    change ringKMul (RingKAction.embedChallenge (ringFMul left right)) value =
      ringKMul (RingKAction.embedChallenge left)
        (ringKMul (RingKAction.embedChallenge right) value)
    apply ringK_ext
    · rw [ringKMul_embedLeft_component0, ringKMul_embedLeft_component0,
        ringKMul_embedLeft_component0, RingFLaws.ringFMul_assoc]
    · rw [ringKMul_embedLeft_component1, ringKMul_embedLeft_component1,
        ringKMul_embedLeft_component1, RingFLaws.ringFMul_assoc]
  smul_zero := by
    intro scalar
    change ringKMul (RingKAction.embedChallenge scalar) ringKZero = ringKZero
    apply ringK_ext
    · rw [ringKMul_embedLeft_component0,
        show ringKComponent0 ringKZero = ringFZero from rfl,
        CarrierAction.ringFMul_zero_right]
    · rw [ringKMul_embedLeft_component1,
        show ringKComponent1 ringKZero = ringFZero from rfl,
        CarrierAction.ringFMul_zero_right]
  smul_add := by
    intro scalar left right
    change ringKMul (RingKAction.embedChallenge scalar) (ringKAdd left right) =
      ringKAdd
        (ringKMul (RingKAction.embedChallenge scalar) left)
        (ringKMul (RingKAction.embedChallenge scalar) right)
    apply ringK_ext
    · rw [ringKMul_embedLeft_component0,
        show ringKComponent0 (ringKAdd left right) =
          ringFAdd (ringKComponent0 left) (ringKComponent0 right) from rfl,
        CarrierAction.ringFMul_add_right,
        ← ringKMul_embedLeft_component0 scalar left,
        ← ringKMul_embedLeft_component0 scalar right]
      rfl
    · rw [ringKMul_embedLeft_component1,
        show ringKComponent1 (ringKAdd left right) =
          ringFAdd (ringKComponent1 left) (ringKComponent1 right) from rfl,
        CarrierAction.ringFMul_add_right,
        ← ringKMul_embedLeft_component1 scalar left,
        ← ringKMul_embedLeft_component1 scalar right]
      rfl

private def assignmentNeg {shape : Phi81Relation.Shape}
    (assignment : Assignment shape) : Assignment shape :=
  fun column => -assignment column

private def assignmentModule {shape : Phi81Relation.Shape} :
    ModuleOps RingF (Assignment shape) where
  zero := BaseLinear.assignmentZero
  add := BaseLinear.assignmentAdd
  neg := assignmentNeg
  smul := CarrierAction.act

private theorem assignment_ext {shape : Phi81Relation.Shape}
    {left right : Assignment shape}
    (blocks : forall block,
      CarrierAction.assignmentBlock left block =
        CarrierAction.assignmentBlock right block) :
    left = right := by
  funext column
  let decoded := Phi81ColumnLayout.decode column
  have equal := congrFun (blocks decoded.1) decoded.2
  have columnEq : CarrierAction.carrierColumn decoded.1 decoded.2 = column := by
    apply Fin.ext
    exact Phi81ColumnLayout.flatIndex_decode column
  simpa only [CarrierAction.assignmentBlock, columnEq] using equal

private theorem assignmentBlock_add {shape : Phi81Relation.Shape}
    (left right : Assignment shape)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    CarrierAction.assignmentBlock (BaseLinear.assignmentAdd left right) block =
      ringFAdd (CarrierAction.assignmentBlock left block)
        (CarrierAction.assignmentBlock right block) := by
  rfl

private theorem assignmentBlock_zero {shape : Phi81Relation.Shape}
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    CarrierAction.assignmentBlock
        (BaseLinear.assignmentZero : Assignment shape) block = ringFZero := by
  rfl

private theorem assignmentModuleLaws {shape : Phi81Relation.Shape} :
    ModuleLaws ScalarRing (assignmentModule (shape := shape)) where
  add_assoc := by
    intro left middle right
    funext column
    exact ConcreteCarrier.baseLaws.add_assoc _ _ _
  add_comm := by
    intro left right
    funext column
    exact ConcreteCarrier.baseLaws.add_comm _ _
  zero_add := by
    intro value
    funext column
    exact ConcreteCarrier.baseLaws.zero_add _
  add_zero := by
    intro value
    funext column
    exact ConcreteCarrier.baseLaws.add_zero _
  add_neg := by
    intro value
    funext column
    exact ConcreteCarrier.baseLaws.add_neg _
  zero_smul := by
    intro value
    change CarrierAction.act ringFZero value = BaseLinear.assignmentZero
    apply assignment_ext
    intro block
    rw [CarrierAction.assignmentBlock_act,
      RingFLaws.ringFMul_zero_left]
    rfl
  add_smul := by
    intro left right value
    change CarrierAction.act (ringFAdd left right) value =
      BaseLinear.assignmentAdd
        (CarrierAction.act left value) (CarrierAction.act right value)
    apply assignment_ext
    intro block
    simp only [CarrierAction.assignmentBlock_act, assignmentBlock_add]
    exact CarrierAction.ringFMul_add_left left right
      (CarrierAction.assignmentBlock value block)
  one_smul := by
    intro value
    change CarrierAction.act ringFOne value = value
    apply assignment_ext
    intro block
    rw [CarrierAction.assignmentBlock_act, RingFLaws.ringFMul_one_left]
  mul_smul := by
    intro left right value
    change CarrierAction.act (ringFMul left right) value =
      CarrierAction.act left (CarrierAction.act right value)
    apply assignment_ext
    intro block
    rw [CarrierAction.assignmentBlock_act,
      CarrierAction.assignmentBlock_act,
      CarrierAction.assignmentBlock_act,
      RingFLaws.ringFMul_assoc]
  smul_zero := by
    intro scalar
    change CarrierAction.act scalar BaseLinear.assignmentZero =
      BaseLinear.assignmentZero
    apply assignment_ext
    intro block
    simp only [CarrierAction.assignmentBlock_act, assignmentBlock_zero]
    exact CarrierAction.ringFMul_zero_right scalar
  smul_add := by
    intro scalar left right
    change CarrierAction.act scalar (BaseLinear.assignmentAdd left right) =
      BaseLinear.assignmentAdd
        (CarrierAction.act scalar left) (CarrierAction.act scalar right)
    apply assignment_ext
    intro block
    simp only [CarrierAction.assignmentBlock_act, assignmentBlock_add]
    exact CarrierAction.ringFMul_add_right scalar
      (CarrierAction.assignmentBlock left block)
      (CarrierAction.assignmentBlock right block)

private def commitmentModule :
    ModuleOps RingF PaperAlgebra.Commitment :=
  functionModule ringFModule

private theorem commitmentModuleLaws :
    ModuleLaws ScalarRing commitmentModule :=
  functionModuleLaws ringFModule ringFModuleLaws

private def publicNeg {shape : Phi81Relation.Shape}
    (input : PublicInput shape) : PublicInput shape :=
  fun column => -input column

private def publicModule {shape : Phi81Relation.Shape} :
    ModuleOps RingF (PublicInput shape) where
  zero := Phi81Relation.PiRLCAlgebra.PublicInput.publicZero
  add := Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
  neg := publicNeg
  smul := Phi81Relation.PiRLCAlgebra.PublicInput.publicAct

private theorem publicBlock_act {shape : Phi81Relation.Shape}
    (scalar : RingF) (input : PublicInput shape)
    (block : Fin shape.publicRingColumns) :
    Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicAct scalar input) block =
      ringFMul scalar
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock input block) := by
  funext lane
  let column : Fin shape.publicWidth :=
    ⟨block.val * ringDegree + lane.val, by
      have blockLt := block.isLt
      have laneLt := lane.isLt
      simp only [Phi81Relation.Shape.publicWidth, ringDegree] at blockLt laneLt ⊢
      omega⟩
  have blockEq :
      Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex shape column =
        block := by
    apply Fin.ext
    simp only [Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex,
      column, ringDegree]
    omega
  have laneEq :
      Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex column = lane := by
    apply Fin.ext
    simp only [Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex,
      column, ringDegree]
    omega
  change ringFMul scalar
      (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock input
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex shape column))
      (Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex column) =
    ringFMul scalar
      (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock input block) lane
  rw [blockEq, laneEq]

private theorem public_ext {shape : Phi81Relation.Shape}
    {left right : PublicInput shape}
    (blocks : forall block,
      Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock left block =
        Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock right block) :
    left = right := by
  funext column
  let block := Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex shape column
  let lane := Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex column
  have equal := congrFun (blocks block) lane
  have columnEq :
      (⟨block.val * ringDegree + lane.val, by
        have blockLt := block.isLt
        have laneLt := lane.isLt
        simp only [Phi81Relation.Shape.publicWidth, ringDegree] at blockLt laneLt ⊢
        omega⟩ : Fin shape.publicWidth) = column := by
    apply Fin.ext
    dsimp only [block, lane]
    simp only [Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex,
      Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex]
    calc
      column.val / ringDegree * ringDegree + column.val % ringDegree =
          ringDegree * (column.val / ringDegree) + column.val % ringDegree := by
        rw [Nat.mul_comm]
      _ = column.val := Nat.div_add_mod column.val ringDegree
  simpa only [Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock, columnEq]
    using equal

private theorem publicBlock_add {shape : Phi81Relation.Shape}
    (left right : PublicInput shape) (block : Fin shape.publicRingColumns) :
    Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd left right) block =
      ringFAdd
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock left block)
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock right block) := by
  rfl

private theorem publicBlock_zero {shape : Phi81Relation.Shape}
    (block : Fin shape.publicRingColumns) :
    Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicZero : PublicInput shape)
        block = ringFZero := by
  rfl

private theorem publicModuleLaws {shape : Phi81Relation.Shape} :
    ModuleLaws ScalarRing (publicModule (shape := shape)) where
  add_assoc := by
    intro left middle right
    funext column
    exact ConcreteCarrier.baseLaws.add_assoc _ _ _
  add_comm := by
    intro left right
    funext column
    exact ConcreteCarrier.baseLaws.add_comm _ _
  zero_add := by
    intro value
    funext column
    exact ConcreteCarrier.baseLaws.zero_add _
  add_zero := by
    intro value
    funext column
    exact ConcreteCarrier.baseLaws.add_zero _
  add_neg := by
    intro value
    funext column
    exact ConcreteCarrier.baseLaws.add_neg _
  zero_smul := by
    intro value
    change Phi81Relation.PiRLCAlgebra.PublicInput.publicAct ringFZero value =
      Phi81Relation.PiRLCAlgebra.PublicInput.publicZero
    apply public_ext
    intro block
    rw [publicBlock_act, RingFLaws.ringFMul_zero_left]
    rfl
  add_smul := by
    intro left right value
    change Phi81Relation.PiRLCAlgebra.PublicInput.publicAct
        (ringFAdd left right) value =
      Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicAct left value)
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicAct right value)
    apply public_ext
    intro block
    simp only [publicBlock_act, publicBlock_add]
    exact CarrierAction.ringFMul_add_left left right
      (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock value block)
  one_smul := by
    intro value
    change Phi81Relation.PiRLCAlgebra.PublicInput.publicAct ringFOne value = value
    apply public_ext
    intro block
    rw [publicBlock_act, RingFLaws.ringFMul_one_left]
  mul_smul := by
    intro left right value
    change Phi81Relation.PiRLCAlgebra.PublicInput.publicAct
        (ringFMul left right) value =
      Phi81Relation.PiRLCAlgebra.PublicInput.publicAct left
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicAct right value)
    apply public_ext
    intro block
    rw [publicBlock_act, publicBlock_act, publicBlock_act,
      RingFLaws.ringFMul_assoc]
  smul_zero := by
    intro scalar
    change Phi81Relation.PiRLCAlgebra.PublicInput.publicAct scalar
        Phi81Relation.PiRLCAlgebra.PublicInput.publicZero =
      Phi81Relation.PiRLCAlgebra.PublicInput.publicZero
    apply public_ext
    intro block
    simp only [publicBlock_act, publicBlock_zero]
    exact CarrierAction.ringFMul_zero_right scalar
  smul_add := by
    intro scalar left right
    change Phi81Relation.PiRLCAlgebra.PublicInput.publicAct scalar
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd left right) =
      Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicAct scalar left)
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicAct scalar right)
    apply public_ext
    intro block
    simp only [publicBlock_act, publicBlock_add]
    exact CarrierAction.ringFMul_add_right scalar
      (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock left block)
      (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock right block)

private def evaluationModule :
    ModuleOps RingF PaperAlgebra.Evaluation where
  zero := PaperAlgebra.evaluationZero
  add := fun left right => {
    pad := ringKAdd left.pad right.pad
    matrix := fun matrix => ringKAdd (left.matrix matrix) (right.matrix matrix)
  }
  neg := fun value => {
    pad := RingKAction.neg value.pad
    matrix := fun matrix => RingKAction.neg (value.matrix matrix)
  }
  smul := fun scalar value => {
    pad := ringKMul (RingKAction.embedChallenge scalar) value.pad
    matrix := fun matrix =>
      ringKMul (RingKAction.embedChallenge scalar) (value.matrix matrix)
  }

private theorem evaluation_ext {left right : PaperAlgebra.Evaluation}
    (pad : left.pad = right.pad)
    (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

private theorem evaluationModuleLaws :
    ModuleLaws ScalarRing evaluationModule where
  add_assoc := by
    intro left middle right
    apply evaluation_ext
    · exact ringKModuleLaws.add_assoc _ _ _
    · funext matrix
      exact ringKModuleLaws.add_assoc _ _ _
  add_comm := by
    intro left right
    apply evaluation_ext
    · exact ringKModuleLaws.add_comm _ _
    · funext matrix
      exact ringKModuleLaws.add_comm _ _
  zero_add := by
    intro value
    apply evaluation_ext
    · exact ringKModuleLaws.zero_add _
    · funext matrix
      exact ringKModuleLaws.zero_add _
  add_zero := by
    intro value
    apply evaluation_ext
    · exact ringKModuleLaws.add_zero _
    · funext matrix
      exact ringKModuleLaws.add_zero _
  add_neg := by
    intro value
    apply evaluation_ext
    · exact ringKModuleLaws.add_neg _
    · funext matrix
      exact ringKModuleLaws.add_neg _
  zero_smul := by
    intro value
    apply evaluation_ext
    · exact ringKModuleLaws.zero_smul _
    · funext matrix
      exact ringKModuleLaws.zero_smul _
  add_smul := by
    intro left right value
    apply evaluation_ext
    · exact ringKModuleLaws.add_smul _ _ _
    · funext matrix
      exact ringKModuleLaws.add_smul _ _ _
  one_smul := by
    intro value
    apply evaluation_ext
    · exact ringKModuleLaws.one_smul _
    · funext matrix
      exact ringKModuleLaws.one_smul _
  mul_smul := by
    intro left right value
    apply evaluation_ext
    · exact ringKModuleLaws.mul_smul _ _ _
    · funext matrix
      exact ringKModuleLaws.mul_smul _ _ _
  smul_zero := by
    intro scalar
    apply evaluation_ext
    · exact ringKModuleLaws.smul_zero _
    · funext matrix
      exact ringKModuleLaws.smul_zero _
  smul_add := by
    intro scalar left right
    apply evaluation_ext
    · exact ringKModuleLaws.smul_add _ _ _
    · funext matrix
      exact ringKModuleLaws.smul_add _ _ _

private theorem combineEvaluation_eq_linearCombination {count : Nat}
    (coefficients : Fin count -> RingF) (values : Fin count -> RingK) :
    PiRLCFinite.combineEvaluation coefficients values =
      linearCombination ScalarRing ringKModule coefficients values := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [PiRLCFinite.combineEvaluation, linearCombination,
        inductionHypothesis
          (fun index => coefficients index.succ)
          (fun index => values index.succ)]
      rfl

private theorem combineEvaluationFamily_eq_linearCombination {count : Nat}
    (coefficients : Fin count -> RingF)
    (values : Fin count -> PaperAlgebra.Evaluation) :
    PaperAlgebra.combineEvaluationFamily coefficients values =
      linearCombination ScalarRing evaluationModule coefficients values := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [linearCombination]
      apply evaluation_ext
      · change
          PiRLCFinite.combineEvaluation coefficients
              (fun source => (values source).pad) =
            ringKAdd
              (ringKMul (RingKAction.embedChallenge (coefficients 0))
                (values 0).pad)
              ((linearCombination ScalarRing evaluationModule
                (fun index => coefficients index.succ)
                (fun index => values index.succ)).pad)
        have padInduction := congrArg
          (fun family : PaperAlgebra.Evaluation => family.pad)
          (inductionHypothesis
            (fun index => coefficients index.succ)
            (fun index => values index.succ))
        change PiRLCFinite.combineEvaluation
            (fun index => coefficients index.succ)
            (fun index => (values index.succ).pad) =
          (linearCombination ScalarRing evaluationModule
            (fun index => coefficients index.succ)
            (fun index => values index.succ)).pad at padInduction
        rw [PiRLCFinite.combineEvaluation, padInduction]
      · funext matrix
        change
          PiRLCFinite.combineEvaluation coefficients
              (fun source => (values source).matrix matrix) =
            ringKAdd
              (ringKMul (RingKAction.embedChallenge (coefficients 0))
                ((values 0).matrix matrix))
              ((linearCombination ScalarRing evaluationModule
                (fun index => coefficients index.succ)
                (fun index => values index.succ)).matrix matrix)
        have matrixInduction := congrArg
          (fun family : PaperAlgebra.Evaluation => family.matrix matrix)
          (inductionHypothesis
            (fun index => coefficients index.succ)
            (fun index => values index.succ))
        change PiRLCFinite.combineEvaluation
            (fun index => coefficients index.succ)
            (fun index => (values index.succ).matrix matrix) =
          (linearCombination ScalarRing evaluationModule
            (fun index => coefficients index.succ)
            (fun index => values index.succ)).matrix matrix at matrixInduction
        rw [PiRLCFinite.combineEvaluation, matrixInduction]

private theorem mapNegOfMapZeroAdd
    {Source Target : Type}
    (source : ModuleOps RingF Source) (target : ModuleOps RingF Target)
    (sourceLaws : ModuleLaws ScalarRing source)
    (targetLaws : ModuleLaws ScalarRing target)
    (map : Source -> Target)
    (mapZero : map source.zero = target.zero)
    (mapAdd : forall left right,
      map (source.add left right) = target.add (map left) (map right))
    (value : Source) :
    map (source.neg value) = target.neg (map value) := by
  have inverse : target.add (map value) (map (source.neg value)) = target.zero := by
    rw [← mapAdd, sourceLaws.add_neg, mapZero]
  calc
    map (source.neg value) = target.add target.zero (map (source.neg value)) :=
      (targetLaws.zero_add _).symm
    _ = target.add
        (target.add (target.neg (map value)) (map value))
        (map (source.neg value)) := by
      rw [targetLaws.add_comm (target.neg (map value)) (map value),
        targetLaws.add_neg]
    _ = target.add (target.neg (map value))
        (target.add (map value) (map (source.neg value))) :=
      targetLaws.add_assoc _ _ _
    _ = target.add (target.neg (map value)) target.zero := by rw [inverse]
    _ = target.neg (map value) := targetLaws.add_zero _

private def linearMapLawsOfZeroAddSmul
    {Source Target : Type}
    (source : ModuleOps RingF Source) (target : ModuleOps RingF Target)
    (sourceLaws : ModuleLaws ScalarRing source)
    (targetLaws : ModuleLaws ScalarRing target)
    (map : Source -> Target)
    (mapZero : map source.zero = target.zero)
    (mapAdd : forall left right,
      map (source.add left right) = target.add (map left) (map right))
    (mapSmul : forall scalar value,
      map (source.smul scalar value) = target.smul scalar (map value)) :
    LinearMapLaws source target map where
  map_sub := by
    intro left right
    unfold ModuleOps.sub
    rw [mapAdd,
      mapNegOfMapZeroAdd source target sourceLaws targetLaws map mapZero mapAdd]
  map_smul := mapSmul

private theorem evaluationFamily_zero
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : PaperAlgebra.Structure logicalWidth) (point : PaperAlgebra.Point) :
    PaperAlgebra.evaluationFamily (publicFits := publicFits) source
        BaseLinear.assignmentZero point = evaluationModule.zero := by
  apply evaluation_ext
  · exact PiRLC.ExplicitMatrix.evaluate_zero
      (PaperAlgebra.canonicalStructure (publicFits := publicFits) source)
      (PaperAlgebra.padMatrix source) point
  · funext matrix
    exact BaseLinear.matrixEvaluation_zero
      (PaperAlgebra.canonicalStructure (publicFits := publicFits) source)
      point matrix

private theorem evaluationFamily_add
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : PaperAlgebra.Structure logicalWidth)
    (left right : PaperAlgebra.Assignment
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : PaperAlgebra.Point) :
    PaperAlgebra.evaluationFamily (publicFits := publicFits) source
        (BaseLinear.assignmentAdd left right) point =
      evaluationModule.add
        (PaperAlgebra.evaluationFamily (publicFits := publicFits) source left point)
        (PaperAlgebra.evaluationFamily (publicFits := publicFits) source right point) := by
  apply evaluation_ext
  · exact PiRLC.ExplicitMatrix.evaluate_add
      (PaperAlgebra.canonicalStructure (publicFits := publicFits) source)
      (PaperAlgebra.padMatrix source) left right point
  · funext matrix
    exact BaseLinear.matrixEvaluation_add
      (PaperAlgebra.canonicalStructure (publicFits := publicFits) source)
      left right point matrix

private theorem evaluationFamily_act
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : PaperAlgebra.Structure logicalWidth) (scalar : RingF)
    (assignment : PaperAlgebra.Assignment
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : PaperAlgebra.Point) :
    PaperAlgebra.evaluationFamily (publicFits := publicFits) source
        (CarrierAction.act scalar assignment) point =
      evaluationModule.smul scalar
        (PaperAlgebra.evaluationFamily (publicFits := publicFits) source
          assignment point) := by
  apply evaluation_ext
  · exact PiRLC.ExplicitMatrix.evaluate_act
      (PaperAlgebra.canonicalStructure (publicFits := publicFits) source)
      (PaperAlgebra.padMatrix source) scalar (PiRLC.productOrderLaw scalar)
      assignment point
  · funext matrix
    exact PiRLC.matrixEvaluation_act
      (PaperAlgebra.canonicalStructure (publicFits := publicFits) source)
      scalar (PiRLC.productOrderLaw scalar) assignment point matrix

private theorem singleton_getD {Value : Type}
    (value default : Value) (index : Nat) :
    #[value].getD index default = if index = 0 then value else default := by
  cases index <;> simp

private theorem semantic_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (key : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : PaperAlgebra.Structure logicalWidth)
    (assignment : PaperAlgebra.Assignment
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : PaperAlgebra.Point) (index : Nat) :
    ((PaperAlgebra.semantics key).evaluations source assignment point).getD
        index evaluationModule.zero =
      if index = 0 then
        PaperAlgebra.evaluationFamily (publicFits := publicFits)
          source assignment point
      else evaluationModule.zero := by
  change #[PaperAlgebra.evaluationFamily (publicFits := publicFits)
    source assignment point].getD index evaluationModule.zero = _
  exact singleton_getD _ _ index

private def semanticEvaluationMapLaws
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (key : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : PaperAlgebra.Structure logicalWidth)
    (point : PaperAlgebra.Point) (index : Nat) :
    LinearMapLaws (assignmentModule (shape := PaperAlgebra.FullShape logicalWidth publicFits))
      evaluationModule
      (fun assignment =>
        (PaperAlgebra.semantics key).evaluations source assignment point
          |>.getD index evaluationModule.zero) := by
  apply linearMapLawsOfZeroAddSmul _ _ assignmentModuleLaws evaluationModuleLaws
  · rw [semantic_getD]
    split
    · exact evaluationFamily_zero source point
    · rfl
  · intro left right
    rw [semantic_getD, semantic_getD, semantic_getD]
    split
    · exact evaluationFamily_add source left right point
    · exact (evaluationModuleLaws.zero_add evaluationModule.zero).symm
  · intro scalar value
    rw [semantic_getD, semantic_getD]
    split
    · exact evaluationFamily_act source scalar value point
    · exact (evaluationModuleLaws.smul_zero scalar).symm

private theorem combineEvaluations_size
    {count : Nat} (coefficients : Fin count -> RingF)
    (values : Fin count -> Array PaperAlgebra.Evaluation)
    (expectedSize : Nat) (positive : 0 < count)
    (sizes : forall index, (values index).size = expectedSize) :
    (PaperAlgebra.combineEvaluations coefficients values).size = expectedSize := by
  cases count with
  | zero => omega
  | succ count =>
      simp only [PaperAlgebra.combineEvaluations, Array.size_ofFn]
      exact sizes 0

private theorem linearCombination_zero {count : Nat}
    (coefficients : Fin count -> RingF) :
    linearCombination ScalarRing evaluationModule coefficients
        (fun _ => evaluationModule.zero) = evaluationModule.zero := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [linearCombination, evaluationModuleLaws.smul_zero,
        inductionHypothesis, evaluationModuleLaws.zero_add]

private theorem combineEvaluations_getD
    {count : Nat} (coefficients : Fin count -> RingF)
    (values : Fin count -> Array PaperAlgebra.Evaluation)
    (expectedSize index : Nat) (positive : 0 < count)
    (sizes : forall source, (values source).size = expectedSize) :
    (PaperAlgebra.combineEvaluations coefficients values).getD index
        evaluationModule.zero =
      linearCombination ScalarRing evaluationModule coefficients
        (fun source => (values source).getD index evaluationModule.zero) := by
  cases count with
  | zero => omega
  | succ count =>
      by_cases indexLt : index < expectedSize
      · have sourceIndexLt : forall source, index < (values source).size := by
          intro source
          rw [sizes source]
          exact indexLt
        have outputIndexLt :
            index < (PaperAlgebra.combineEvaluations coefficients values).size := by
          rw [combineEvaluations_size coefficients values expectedSize
            (by omega) sizes]
          exact indexLt
        rw [Array.getD_eq_getD_getElem?,
          Array.getElem?_eq_getElem outputIndexLt]
        simp only [Option.getD_some]
        simp only [PaperAlgebra.combineEvaluations, Array.getElem_ofFn]
        rw [combineEvaluationFamily_eq_linearCombination]
        apply congrArg
        funext source
        rw [Array.getD_eq_getD_getElem?,
          Array.getElem?_eq_getElem (sourceIndexLt source)]
        simp only [Option.getD_some]
        rw [Array.getD_eq_getD_getElem?,
          Array.getElem?_eq_getElem (sourceIndexLt source)]
        rfl
      · have sourceIndexGe : forall source, (values source).size <= index := by
          intro source
          rw [sizes source]
          omega
        have outputIndexGe :
            (PaperAlgebra.combineEvaluations coefficients values).size <= index := by
          rw [combineEvaluations_size coefficients values expectedSize
            (by omega) sizes]
          omega
        rw [Array.getD_eq_getD_getElem?,
          Array.getElem?_eq_none outputIndexGe]
        simp only [Option.getD_none]
        have allDefault :
            (fun source => (values source).getD index evaluationModule.zero) =
              fun _ => evaluationModule.zero := by
          funext source
          rw [Array.getD_eq_getD_getElem?,
            Array.getElem?_eq_none (sourceIndexGe source)]
          rfl
        rw [allDefault]
        exact (linearCombination_zero coefficients).symm

/-- The exact deterministic extraction algebra used by the production
`PaperAlgebra.piRlcAlgebra`. The strong-set theorem remains a separate,
explicit security premise. -/
def extractionAlgebra
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (key : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    ExtractionAlgebra (PaperAlgebra.semantics key) productionGlobalParams
      (PaperAlgebra.piRlcAlgebra key) where
  ring := ScalarRing
  ringLaws := Phi81Relation.PiRLCAlgebra.ForkStrongSet.ringLaws
  assignmentModule := assignmentModule
  assignmentLaws := assignmentModuleLaws
  commitmentModule := commitmentModule
  commitmentLaws := commitmentModuleLaws
  publicInputModule := publicModule
  publicInputLaws := publicModuleLaws
  evaluationModule := evaluationModule
  evaluationLaws := evaluationModuleLaws
  combineCommitment_eq := by
    intro count coefficients values
    change Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments
        coefficients values =
      linearCombination ScalarRing commitmentModule coefficients values
    induction count with
    | zero => rfl
    | succ count inductionHypothesis =>
        change Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd
            (Phi81Relation.PiRLCAlgebra.Commitment.commitmentAct
              (coefficients 0) (values 0))
            (Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments
              (fun index => coefficients index.succ)
              (fun index => values index.succ)) =
          commitmentModule.add
            (commitmentModule.smul (coefficients 0) (values 0))
            (linearCombination ScalarRing commitmentModule
              (fun index => coefficients index.succ)
              (fun index => values index.succ))
        rw [
          inductionHypothesis
            (fun index => coefficients index.succ)
            (fun index => values index.succ)]
        rfl
  combinePublicInput_eq := by
    intro count coefficients values
    change Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
        coefficients values =
      linearCombination ScalarRing publicModule coefficients values
    induction count with
    | zero => rfl
    | succ count inductionHypothesis =>
        change Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
            (Phi81Relation.PiRLCAlgebra.PublicInput.publicAct
              (coefficients 0) (values 0))
            (Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
              (fun index => coefficients index.succ)
              (fun index => values index.succ)) =
          publicModule.add
            (publicModule.smul (coefficients 0) (values 0))
            (linearCombination ScalarRing publicModule
              (fun index => coefficients index.succ)
              (fun index => values index.succ))
        rw [
          inductionHypothesis
            (fun index => coefficients index.succ)
            (fun index => values index.succ)]
        rfl
  semanticEvaluations_size_eq := by
    intro system point left right
    rw [PaperAlgebra.semantics_evaluations_size,
      PaperAlgebra.semantics_evaluations_size]
  combineEvaluations_size := combineEvaluations_size
  combineEvaluations_getD := combineEvaluations_getD
  commitMap := linearMapLawsOfZeroAddSmul _ _ assignmentModuleLaws
    commitmentModuleLaws
    (Phi81Relation.PiRLCAlgebra.Commitment.commit key)
    (Phi81Relation.PiRLCAlgebra.Commitment.commit_zero key)
    (Phi81Relation.PiRLCAlgebra.Commitment.commit_add key)
    (Phi81Relation.PiRLCAlgebra.Commitment.commit_act key)
  publicInputMap := linearMapLawsOfZeroAddSmul _ _ assignmentModuleLaws
    publicModuleLaws
    Phi81Relation.projectPublicInput
    Phi81Relation.PiRLCAlgebra.PublicInput.projectPublicInput_zero
    Phi81Relation.PiRLCAlgebra.PublicInput.projectPublicInput_add
    Phi81Relation.PiRLCAlgebra.PublicInput.projectPublicInput_act
  evaluationsMap := semanticEvaluationMapLaws key
  correctedNormCoverage := by
    intro assignment column
    rw [PaperCorrections.production_correctedAmbientBoundFor_eq]
    exact PaperCorrections.all_centeredMagnitude_lt_correctedAmbientBound
      (assignment column)

end NightstreamFPrime.Lifecycle.PaperExtractionAlgebra
