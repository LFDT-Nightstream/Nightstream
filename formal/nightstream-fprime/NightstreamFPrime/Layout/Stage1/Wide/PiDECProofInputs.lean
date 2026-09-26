import NightstreamFPrime.Layout.Stage1.PiDECProofInputs
import NightstreamFPrime.Layout.Stage1.Wide.PiDECInputs

/-! Reuse the proved PiDEC data loader at the wide layout's input interval.
The four families keep their order. Translation preserves every other value. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.PiDECProofInputs

open NightstreamFPrime.Circuit NightstreamFPrime.Circuit.Quadratic NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint
open PiDECInputs

private def shift : Nat := 925480
private def referenceEnv (env : Env) : Env := fun index => env (index - shift)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

def load {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) : Env :=
  fun index => Stage1.PiDECProofInputs.load (referenceEnv env) proof parent (index + shift)

theorem load_agreesOutside {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    AgreesOutside env (load env proof parent) proofInputStart proofInputColumnCount := by
  intro index outside
  have oldOutside : index + shift < Stage1.PiDECInputs.proofInputStart ∨
      Stage1.PiDECInputs.proofInputStart + Stage1.PiDECInputs.proofInputColumnCount ≤ index + shift := by
    change index < 27496062 ∨ 27545310 ≤ index at outside
    change index + 925480 < 28421542 ∨ 28470790 ≤ index + 925480
    omega
  have old := Stage1.PiDECProofInputs.load_agreesOutside (referenceEnv env) proof parent (index + shift) oldOutside
  exact old.trans (congrArg env (Nat.add_sub_cancel_right index shift))

private theorem at_shift {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (left right : Nat) (same : left + shift = right) :
    load env proof parent left = Stage1.PiDECProofInputs.load (referenceEnv env) proof parent right :=
  congrArg (Stage1.PiDECProofInputs.load (referenceEnv env) proof parent) same

theorem eval_childCommitment {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (row : Fin productionProfile.commitmentWidth) (lane : Fin ringDegree) :
    (childCommitment child row lane).eval (load env proof parent) = proof.piDecCommitments child row lane := by
  refine (at_shift env proof parent _ _ ?_).trans
    (Stage1.PiDECProofInputs.eval_childCommitment (referenceEnv env) proof parent child row lane)
  change (27496062 + child.val * 1188 + row.val * 54 + lane.val) + 925480 =
    28421542 + child.val * 1188 + row.val * 54 + lane.val
  omega

theorem eval_childEvalK {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex) (coefficient : Fin productionShape.coefficientCount) :
    (childEvalK child coefficient).eval (load env proof parent) = (proof.piDecEvaluations child).pad coefficient := by
  have same : (childEvalK child coefficient).eval (load env proof parent) =
      (Stage1.PiDECInputs.childEvalK child coefficient).eval (Stage1.PiDECProofInputs.load (referenceEnv env) proof parent) := by
    change K.mk _ _ = K.mk _ _
    apply congrArg₂ K.mk
    all_goals
      apply at_shift
      first
      | change (27515070 + child.val * 108 + coefficient.val * 2) + 925480 =
          28440550 + child.val * 108 + coefficient.val * 2
      | change (27515070 + child.val * 108 + coefficient.val * 2 + 1) + 925480 =
          28440550 + child.val * 108 + coefficient.val * 2 + 1
      all_goals omega
  exact same.trans (Stage1.PiDECProofInputs.eval_childEvalK (referenceEnv env) proof parent child coefficient)

theorem eval_childEvalA {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (matrix : Fin productionShape.matrixCount) (coefficient : Fin productionShape.coefficientCount) :
    (childEvalA child matrix coefficient).eval (load env proof parent) =
      (proof.piDecEvaluations child).matrix matrix coefficient := by
  have same : (childEvalA child matrix coefficient).eval (load env proof parent) =
      (Stage1.PiDECInputs.childEvalA child matrix coefficient).eval (Stage1.PiDECProofInputs.load (referenceEnv env) proof parent) := by
    change K.mk _ _ = K.mk _ _
    apply congrArg₂ K.mk
    all_goals
      apply at_shift
      first
      | change (27516798 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2) + 925480 =
          28442278 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2
      | change (27516798 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2 + 1) + 925480 =
          28442278 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2 + 1
      all_goals omega
  exact same.trans (Stage1.PiDECProofInputs.eval_childEvalA (referenceEnv env) proof parent child matrix coefficient)

theorem eval_childPublicInput {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex) (coordinate : Fin 270) :
    (childPublicInput child coordinate).eval (load env proof parent) =
      Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput parent child coordinate := by
  refine (at_shift env proof parent _ _ ?_).trans
    (Stage1.PiDECProofInputs.eval_childPublicInput (referenceEnv env) proof parent child coordinate)
  change (27540990 + child.val * 270 + coordinate.val) + 925480 =
    28466470 + child.val * 270 + coordinate.val
  omega

end NightstreamFPrime.Layout.Stage1.Wide.PiDECProofInputs
