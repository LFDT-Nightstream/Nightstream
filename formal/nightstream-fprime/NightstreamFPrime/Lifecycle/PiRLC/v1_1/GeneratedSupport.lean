import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerGeneratedSupport

/-!
Owns lower-support equality for PiRLC combination outputs and the canonical
public attempt. All generated coordinates are direct variables from the
existing combination children.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation

namespace CombinationFamily

/-- One final combination coordinate is unchanged when environments agree
from the family start onward. -/
theorem output_eval_eq_of_agree_from
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount)
    (left right : Env)
    (agrees : ∀ index, offset ≤ index → left index = right index) :
    (output interface offset block lane cell).eval left =
      (output interface offset block lane cell).eval right := by
  apply Expr.eval_eq_of_agree_satisfy _ (fun index => offset ≤ index)
    left right
  · simp only [output, CombinationStep.output, Expr.VarsSatisfy]
    unfold stepOffset
    omega
  · exact agrees

end CombinationFamily

namespace Semantics

private theorem inputInstance_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : InputBinding.InputInstance logicalWidth publicFits)
    (constraintSystem : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  simp_all

private theorem evaluation_ext (left right : PaperAlgebra.Evaluation)
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) :
    left = right := by
  cases left
  cases right
  simp_all

/-- The complete combined PiRLC output is stable under equality of the shared
point and preservation of the PiRLC-owned suffix. -/
theorem evalOutput_eq_of_point_and_agree_from
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (left right : Env)
    (pointEq : InputBinding.evalPoint (interface.point offset) left =
      InputBinding.evalPoint (interface.point offset) right)
    (agrees : ∀ index, offset ≤ index → left index = right index) :
    evalOutput relation interface offset left =
      evalOutput relation interface offset right := by
  apply inputInstance_ext
  · rfl
  · funext row lane
    exact CombinationFamily.output_eval_eq_of_agree_from
      (CommitmentCombination.familyInterface
        (Formal.commitmentInterface (Formal.atOffset interface offset)))
      (Formal.commitmentOffset offset) row lane CommitmentCombination.cell
      left right (fun index bounded => agrees index (by
        unfold Formal.commitmentOffset Formal.samplerOffset at bounded
        omega))
  · funext column
    exact CombinationFamily.output_eval_eq_of_agree_from
      (PublicInputCombination.familyInterface
        (Formal.publicInputInterface (Formal.atOffset interface offset)))
      (Formal.publicInputOffset offset)
      (PiRLCAlgebra.PublicInput.publicBlockIndex
        (FullShape logicalWidth publicFits) column)
      (PiRLCAlgebra.PublicInput.publicLaneIndex column)
      PublicInputCombination.cell left right (fun index bounded =>
        agrees index (by
          unfold Formal.publicInputOffset Formal.commitmentOffset
            Formal.samplerOffset at bounded
          omega))
  · simpa [evalOutput, OutputBinding.evalOutput,
      Formal.outputBindingInterface, Formal.atOffset] using pointEq
  · apply congrArg (fun value => #[value])
    apply evaluation_ext
    · funext coefficient
      exact congrArg₂ K.mk
        (CombinationFamily.output_eval_eq_of_agree_from
          (RingKCombination.familyInterface
            (EvalKCombination.ringInterface
              (Formal.evalKInterface (Formal.atOffset interface offset))))
          (Formal.evalKOffset offset) EvalKCombination.block
          (Fin.cast EvalKCombination.coefficientCount_eq coefficient)
          RingKCombination.c0Cell left right (fun index bounded =>
            agrees index (by
            unfold Formal.evalKOffset Formal.publicInputOffset
              Formal.commitmentOffset Formal.samplerOffset at bounded
            omega)))
        (CombinationFamily.output_eval_eq_of_agree_from
          (RingKCombination.familyInterface
            (EvalKCombination.ringInterface
              (Formal.evalKInterface (Formal.atOffset interface offset))))
          (Formal.evalKOffset offset) EvalKCombination.block
          (Fin.cast EvalKCombination.coefficientCount_eq coefficient)
          RingKCombination.c1Cell left right (fun index bounded =>
            agrees index (by
            unfold Formal.evalKOffset Formal.publicInputOffset
              Formal.commitmentOffset Formal.samplerOffset at bounded
            omega)))
    · funext matrix coefficient
      exact congrArg₂ K.mk
        (CombinationFamily.output_eval_eq_of_agree_from
          (RingKCombination.familyInterface
            (EvalACombination.ringInterface
              (Formal.evalAInterface (Formal.atOffset interface offset))))
          (Formal.evalAOffset offset) matrix
          (Fin.cast EvalKCombination.coefficientCount_eq coefficient)
          RingKCombination.c0Cell left right (fun index bounded =>
            agrees index (by
            unfold Formal.evalAOffset Formal.evalKOffset
              Formal.publicInputOffset Formal.commitmentOffset
              Formal.samplerOffset at bounded
            omega)))
        (CombinationFamily.output_eval_eq_of_agree_from
          (RingKCombination.familyInterface
            (EvalACombination.ringInterface
              (Formal.evalAInterface (Formal.atOffset interface offset))))
          (Formal.evalAOffset offset) matrix
          (Fin.cast EvalKCombination.coefficientCount_eq coefficient)
          RingKCombination.c1Cell left right (fun index bounded =>
            agrees index (by
            unfold Formal.evalAOffset Formal.evalKOffset
              Formal.publicInputOffset Formal.commitmentOffset
              Formal.samplerOffset at bounded
            omega)))
  · rfl

/-- Equality of the three canonical attempt components is equality of the
public PiRLC attempt itself. -/
theorem attempt_eq_of_components
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (left right : Env)
    (inputsEq : evalInputs relation interface offset left =
      evalInputs relation interface offset right)
    (challengesEq : evalChallenges interface offset left =
      evalChallenges interface offset right)
    (outputEq : evalOutput relation interface offset left =
      evalOutput relation interface offset right) :
    attempt relation interface offset left =
      attempt relation interface offset right := by
  unfold attempt
  rw [inputsEq, challengesEq, outputEq]

/-- Cross-interface equality of the public PiRLC attempt follows from exact
equality of its three canonical components. -/
theorem attempt_eq_of_cross_components
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (leftInterface rightInterface : Formal.Interface logicalWidth publicFits)
    (leftOffset rightOffset : Nat) (left right : Env)
    (inputsEq : evalInputs relation leftInterface leftOffset left =
      evalInputs relation rightInterface rightOffset right)
    (challengesEq : evalChallenges leftInterface leftOffset left =
      evalChallenges rightInterface rightOffset right)
    (outputEq : evalOutput relation leftInterface leftOffset left =
      evalOutput relation rightInterface rightOffset right) :
    attempt relation leftInterface leftOffset left =
      attempt relation rightInterface rightOffset right := by
  unfold attempt
  rw [inputsEq, challengesEq, outputEq]

end Semantics

end NightstreamFPrime.Lifecycle.PiRLC.v1_1
