import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKModule
import Nightstream.SuperNeo.Folding.PiRLC.PaperForkAlgebra

/-!
Concrete scalar and evaluation modules used by the paper `Pi_RLC` extractor.

Protocol: SuperNeo Appendix D.5.
Phase: concrete algebra for coordinate-fork extraction.
Constraint family: semantic coefficient algebra only; this file emits no rows.

Owns: a commutative-ring instance for executable Phi81 `RingF`; the module
of full `RingK` evaluations under embedded challenge action; and pointwise
lifting of any proved module to a finite function family.

Does not own: assignment packing, commitments, public inputs, challenge-set
security, transcripts, Rust/R1CS refinement, row removal, or counts.

Emits constraints: no.

| Obligation | Local owner | Emits constraints? | Authority source |
|---|---|---|---|
| Phi81 module laws | `ringFLaws`, assignment and evaluation modules | no | Executable quotient-ring operations |

Authority boundary: all operations are the protocol's executable operations.
No algebraic law is accepted from a caller in this file.
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PaperForkModules

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiRLC
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Coefficientwise additive inverse in `RingF`. -/
def ringFNeg (value : RingF) : RingF :=
  fun lane => -value lane

/-- Exact scalar-ring operations used by extraction. -/
def ringFOps : PaperForkAlgebra.CommutativeRingOps RingF where
  zero := ringFZero
  one := ringFOne
  add := ringFAdd
  mul := ringFMul
  neg := ringFNeg

/-- The executable Phi81 base quotient is a commutative ring. -/
theorem ringFLaws : PaperForkAlgebra.CommutativeRingLaws ringFOps where
  add_assoc := by
    intro left middle right
    funext lane
    exact ConcreteCarrier.baseLaws.add_assoc _ _ _
  add_comm := by
    intro left right
    funext lane
    exact ConcreteCarrier.baseLaws.add_comm _ _
  zero_add := by
    intro value
    funext lane
    exact ConcreteCarrier.baseLaws.zero_add _
  add_zero := by
    intro value
    funext lane
    exact ConcreteCarrier.baseLaws.add_zero _
  add_neg := by
    intro value
    funext lane
    exact ConcreteCarrier.baseLaws.add_neg _
  mul_assoc :=
    EvaluationHomomorphism.RingFLaws.ringFMul_assoc
  mul_comm :=
    EvaluationHomomorphism.RingFLaws.ringFMul_comm
  one_mul :=
    EvaluationHomomorphism.RingFLaws.ringFMul_one_left
  mul_one :=
    EvaluationHomomorphism.RingFLaws.ringFMul_one_right
  left_distrib :=
    EvaluationHomomorphism.CarrierAction.ringFMul_add_right
  right_distrib :=
    EvaluationHomomorphism.CarrierAction.ringFMul_add_left

/-- The scalar ring as a module over itself. -/
def ringFModule : PaperForkAlgebra.ModuleOps RingF RingF where
  zero := ringFZero
  add := ringFAdd
  neg := ringFNeg
  smul := ringFMul

/-- The executable scalar ring forms its regular module. -/
theorem ringFModuleLaws :
    PaperForkAlgebra.ModuleLaws ringFOps ringFModule where
  add_assoc := ringFLaws.add_assoc
  add_comm := ringFLaws.add_comm
  zero_add := ringFLaws.zero_add
  add_zero := ringFLaws.add_zero
  add_neg := ringFLaws.add_neg
  zero_smul := EvaluationHomomorphism.RingFLaws.ringFMul_zero_left
  add_smul := EvaluationHomomorphism.CarrierAction.ringFMul_add_left
  one_smul := EvaluationHomomorphism.RingFLaws.ringFMul_one_left
  mul_smul := EvaluationHomomorphism.RingFLaws.ringFMul_assoc
  smul_zero := EvaluationHomomorphism.CarrierAction.ringFMul_zero_right
  smul_add := EvaluationHomomorphism.CarrierAction.ringFMul_add_right

/-- Pointwise additive inverse of one complete assignment. -/
def assignmentNeg {logicalWidth : Nat}
    (assignment : EvaluationHomomorphism.CarrierAction.CompleteAssignment
      logicalWidth) :
    EvaluationHomomorphism.CarrierAction.CompleteAssignment logicalWidth :=
  fun column => -assignment column

/-- Module operations on the complete Phi81 assignment carrier. -/
def assignmentModule (logicalWidth : Nat) :
    PaperForkAlgebra.ModuleOps RingF
      (EvaluationHomomorphism.CarrierAction.CompleteAssignment logicalWidth) where
  zero := EvaluationHomomorphism.BaseLinear.Raw.assignmentZero
  add := EvaluationHomomorphism.BaseLinear.Raw.assignmentAdd
  neg := assignmentNeg
  smul := EvaluationHomomorphism.CarrierAction.act

/-- Complete assignments form a module under blockwise Phi81 action. -/
theorem assignmentModuleLaws (logicalWidth : Nat) :
    PaperForkAlgebra.ModuleLaws ringFOps (assignmentModule logicalWidth) where
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
    funext column
    change ringFMul ringFZero
        (EvaluationHomomorphism.CarrierAction.assignmentBlock value
          (Phi81ColumnLayout.decode column).1)
        (Phi81ColumnLayout.decode column).2 = 0
    rw [EvaluationHomomorphism.RingFLaws.ringFMul_zero_left]
    rfl
  add_smul := by
    intro left right value
    funext column
    change ringFMul (ringFAdd left right)
        (EvaluationHomomorphism.CarrierAction.assignmentBlock value
          (Phi81ColumnLayout.decode column).1)
        (Phi81ColumnLayout.decode column).2 =
      ringFMul left
          (EvaluationHomomorphism.CarrierAction.assignmentBlock value
            (Phi81ColumnLayout.decode column).1)
          (Phi81ColumnLayout.decode column).2 +
        ringFMul right
          (EvaluationHomomorphism.CarrierAction.assignmentBlock value
            (Phi81ColumnLayout.decode column).1)
          (Phi81ColumnLayout.decode column).2
    rw [EvaluationHomomorphism.CarrierAction.ringFMul_add_left]
    rfl
  one_smul := by
    intro value
    funext column
    change ringFMul ringFOne
        (EvaluationHomomorphism.CarrierAction.assignmentBlock value
          (Phi81ColumnLayout.decode column).1)
        (Phi81ColumnLayout.decode column).2 = value column
    rw [EvaluationHomomorphism.RingFLaws.ringFMul_one_left]
    unfold EvaluationHomomorphism.CarrierAction.assignmentBlock
    apply congrArg value
    apply Fin.ext
    exact Phi81ColumnLayout.flatIndex_decode column
  mul_smul := by
    intro left right value
    funext column
    change ringFMul (ringFMul left right)
        (EvaluationHomomorphism.CarrierAction.assignmentBlock value
          (Phi81ColumnLayout.decode column).1)
        (Phi81ColumnLayout.decode column).2 =
      ringFMul left
        (EvaluationHomomorphism.CarrierAction.assignmentBlock
          (EvaluationHomomorphism.CarrierAction.act right value)
          (Phi81ColumnLayout.decode column).1)
        (Phi81ColumnLayout.decode column).2
    rw [EvaluationHomomorphism.CarrierAction.assignmentBlock_act]
    rw [EvaluationHomomorphism.RingFLaws.ringFMul_assoc]
  smul_zero := by
    intro scalar
    funext column
    change ringFMul scalar ringFZero
        (Phi81ColumnLayout.decode column).2 = 0
    rw [EvaluationHomomorphism.CarrierAction.ringFMul_zero_right]
    rfl
  smul_add := by
    intro scalar left right
    funext column
    change ringFMul scalar
        (ringFAdd
          (EvaluationHomomorphism.CarrierAction.assignmentBlock left
            (Phi81ColumnLayout.decode column).1)
          (EvaluationHomomorphism.CarrierAction.assignmentBlock right
            (Phi81ColumnLayout.decode column).1))
        (Phi81ColumnLayout.decode column).2 =
      ringFMul scalar
          (EvaluationHomomorphism.CarrierAction.assignmentBlock left
            (Phi81ColumnLayout.decode column).1)
          (Phi81ColumnLayout.decode column).2 +
        ringFMul scalar
          (EvaluationHomomorphism.CarrierAction.assignmentBlock right
            (Phi81ColumnLayout.decode column).1)
          (Phi81ColumnLayout.decode column).2
    rw [EvaluationHomomorphism.CarrierAction.ringFMul_add_right]
    rfl

/-- Module operations on one full `RingK` evaluation. -/
def ringKModule : PaperForkAlgebra.ModuleOps RingF RingK where
  zero := ringKZero
  add := ringKAdd
  neg := EvaluationHomomorphism.RingKAction.neg
  smul := EvaluationHomomorphism.RingKModule.act

/-- Full `RingK` evaluations form a module under embedded `RingF` action. -/
theorem ringKModuleLaws :
    PaperForkAlgebra.ModuleLaws ringFOps ringKModule where
  add_assoc := by
    intro left middle right
    funext lane
    exact ConcreteCarrier.extensionLaws.add_assoc _ _ _
  add_comm := by
    intro left right
    funext lane
    exact ConcreteCarrier.extensionLaws.add_comm _ _
  zero_add := by
    intro value
    funext lane
    exact ConcreteCarrier.extensionLaws.zero_add _
  add_zero := by
    intro value
    funext lane
    exact ConcreteCarrier.extensionLaws.add_zero _
  add_neg := by
    intro value
    funext lane
    exact ConcreteCarrier.extensionLaws.add_neg _
  zero_smul := EvaluationHomomorphism.RingKModule.zero_act
  add_smul := EvaluationHomomorphism.RingKModule.add_act
  one_smul := EvaluationHomomorphism.RingKModule.one_act
  mul_smul := EvaluationHomomorphism.RingKModule.mul_act
  smul_zero := EvaluationHomomorphism.RingKModule.act_zero
  smul_add := EvaluationHomomorphism.RingKModule.act_add

/-- Pointwise lifting of a module to an exact function domain. -/
def pointwiseModule
    {Value : Type} (Index : Type)
    (module : PaperForkAlgebra.ModuleOps RingF Value) :
    PaperForkAlgebra.ModuleOps RingF (Index -> Value) where
  zero := fun _ => module.zero
  add := fun left right index => module.add (left index) (right index)
  neg := fun value index => module.neg (value index)
  smul := fun scalar value index => module.smul scalar (value index)

/-- Pointwise lifting preserves all module laws. -/
theorem pointwiseModuleLaws
    {Value : Type} (Index : Type)
    (module : PaperForkAlgebra.ModuleOps RingF Value)
    (laws : PaperForkAlgebra.ModuleLaws ringFOps module) :
    PaperForkAlgebra.ModuleLaws ringFOps (pointwiseModule Index module) where
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

end Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PaperForkModules
