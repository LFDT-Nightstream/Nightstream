import NightstreamFPrime.Layout.Stage1.PiRLCGeneratedRelocation

/-!
Owns uniform relocation of the complete PiRLC combined output. The shared
point is supplied as an exact evaluated equality. Every commitment, public,
and evaluation coordinate is a direct final combination variable.
-/

namespace NightstreamFPrime.Layout.Stage1.PiRLCOutputRelocation

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
open NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.WiringShift
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem commitmentOutput_shift
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Formal.Interface logicalWidth publicFits)
    (leftOffset delta : Nat) (row : Fin CommitmentCombination.blockCount)
    (lane : Fin ringDegree) :
    CommitmentCombination.output
        (Formal.commitmentInterface
          (Formal.atOffset right (leftOffset + delta)))
        (Formal.commitmentOffset (leftOffset + delta)) row lane =
      expression delta
        (CommitmentCombination.output
          (Formal.commitmentInterface (Formal.atOffset left leftOffset))
          (Formal.commitmentOffset leftOffset) row lane) := by
  unfold CommitmentCombination.output
  rw [show Formal.commitmentOffset (leftOffset + delta) =
      Formal.commitmentOffset leftOffset + delta by
    unfold Formal.commitmentOffset Formal.samplerOffset
    omega]
  exact PiRLCGeneratedRelocation.combinationOutput_shift _ _ _ _ _ _ _

private theorem publicInputOutput_shift
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Formal.Interface logicalWidth publicFits)
    (leftOffset delta : Nat)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    PublicInputCombination.output
        (Formal.publicInputInterface
          (Formal.atOffset right (leftOffset + delta)))
        (Formal.publicInputOffset (leftOffset + delta)) column =
      expression delta
        (PublicInputCombination.output
          (Formal.publicInputInterface (Formal.atOffset left leftOffset))
          (Formal.publicInputOffset leftOffset) column) := by
  unfold PublicInputCombination.output
  rw [show Formal.publicInputOffset (leftOffset + delta) =
      Formal.publicInputOffset leftOffset + delta by
    unfold Formal.publicInputOffset Formal.commitmentOffset
      Formal.samplerOffset
    omega]
  exact PiRLCGeneratedRelocation.combinationOutput_shift _ _ _ _ _ _ _

private theorem ringKOutput_shift
    {blockCount : Nat}
    (left right : RingKCombination.Interface blockCount)
    (leftOffset delta : Nat) (block : Fin blockCount)
    (lane : Fin ringDegree) :
    RingKCombination.output right (leftOffset + delta) block lane =
      quadratic delta (RingKCombination.output left leftOffset block lane) := by
  unfold RingKCombination.output quadratic
  congr 1
  · exact PiRLCGeneratedRelocation.combinationOutput_shift
      (RingKCombination.familyInterface left)
      (RingKCombination.familyInterface right) _ _ _ _ _
  · exact PiRLCGeneratedRelocation.combinationOutput_shift
      (RingKCombination.familyInterface left)
      (RingKCombination.familyInterface right) _ _ _ _ _

private theorem evalKOutput_shift
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Formal.Interface logicalWidth publicFits)
    (leftOffset delta : Nat)
    (coefficient : Fin productionShape.coefficientCount) :
    EvalKCombination.output
        (Formal.evalKInterface (Formal.atOffset right (leftOffset + delta)))
        (Formal.evalKOffset (leftOffset + delta)) coefficient =
      quadratic delta
        (EvalKCombination.output
          (Formal.evalKInterface (Formal.atOffset left leftOffset))
          (Formal.evalKOffset leftOffset) coefficient) := by
  unfold EvalKCombination.output
  rw [show Formal.evalKOffset (leftOffset + delta) =
      Formal.evalKOffset leftOffset + delta by
    unfold Formal.evalKOffset Formal.publicInputOffset
      Formal.commitmentOffset Formal.samplerOffset
    omega]
  exact ringKOutput_shift _ _ _ _ _ _

private theorem evalAOutput_shift
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Formal.Interface logicalWidth publicFits)
    (leftOffset delta : Nat) (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    EvalACombination.output
        (Formal.evalAInterface (Formal.atOffset right (leftOffset + delta)))
        (Formal.evalAOffset (leftOffset + delta)) matrix coefficient =
      quadratic delta
        (EvalACombination.output
          (Formal.evalAInterface (Formal.atOffset left leftOffset))
          (Formal.evalAOffset leftOffset) matrix coefficient) := by
  unfold EvalACombination.output
  rw [show Formal.evalAOffset (leftOffset + delta) =
      Formal.evalAOffset leftOffset + delta by
    unfold Formal.evalAOffset Formal.evalKOffset Formal.publicInputOffset
      Formal.commitmentOffset Formal.samplerOffset
    omega]
  exact ringKOutput_shift _ _ _ _ _ _

private theorem commitmentOutput_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (row : Fin CommitmentCombination.blockCount) (lane : Fin ringDegree) :
    (CommitmentCombination.output
      (Formal.commitmentInterface (Formal.atOffset interface offset))
      (Formal.commitmentOffset offset) row lane).VarsSatisfy
        (SupportRange.Extend (fun _ => False) offset
          (offset + Formal.logicalPrivateCount)) := by
  unfold CommitmentCombination.output
  apply PiRLCGeneratedRelocation.combinationOutput_supported
  · unfold Formal.commitmentOffset Formal.samplerOffset
    omega
  · rw [CommitmentCombination.logicalPrivateCount_eq]
    unfold Formal.commitmentOffset Formal.samplerOffset
    rw [SamplerChain.logicalPrivateCount_eq]
    norm_num [Formal.logicalPrivateCount]

private theorem publicInputOutput_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    (PublicInputCombination.output
      (Formal.publicInputInterface (Formal.atOffset interface offset))
      (Formal.publicInputOffset offset) column).VarsSatisfy
        (SupportRange.Extend (fun _ => False) offset
          (offset + Formal.logicalPrivateCount)) := by
  unfold PublicInputCombination.output
  apply PiRLCGeneratedRelocation.combinationOutput_supported
  · unfold Formal.publicInputOffset Formal.commitmentOffset
      Formal.samplerOffset
    omega
  · rw [PublicInputCombination.logicalPrivateCount_eq]
    unfold Formal.publicInputOffset Formal.commitmentOffset
      Formal.samplerOffset
    rw [SamplerChain.logicalPrivateCount_eq]
    norm_num [Formal.logicalPrivateCount]

private theorem evalKOutput_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (coefficient : Fin productionShape.coefficientCount) :
    KSupported
      (EvalKCombination.output
        (Formal.evalKInterface (Formal.atOffset interface offset))
        (Formal.evalKOffset offset) coefficient)
      (SupportRange.Extend (fun _ => False) offset
        (offset + Formal.logicalPrivateCount)) := by
  let family := RingKCombination.familyInterface
    (EvalKCombination.ringInterface
      (Formal.evalKInterface (Formal.atOffset interface offset)))
  have startLe : offset ≤ Formal.evalKOffset offset := by
    unfold Formal.evalKOffset Formal.publicInputOffset
      Formal.commitmentOffset Formal.samplerOffset
    omega
  have finishLe : Formal.evalKOffset offset +
      CombinationFamily.logicalPrivateCount EvalKCombination.blockCount
        RingKCombination.cellCount ≤
      offset + Formal.logicalPrivateCount := by
    rw [EvalKCombination.logicalPrivateCount_eq]
    unfold Formal.evalKOffset Formal.publicInputOffset
      Formal.commitmentOffset Formal.samplerOffset
    rw [SamplerChain.logicalPrivateCount_eq]
    norm_num [Formal.logicalPrivateCount]
  unfold KSupported EvalKCombination.output RingKCombination.output
  exact ⟨
    PiRLCGeneratedRelocation.combinationOutput_supported family offset
      (Formal.evalKOffset offset) (offset + Formal.logicalPrivateCount)
      startLe finishLe EvalKCombination.block
      (Fin.cast EvalKCombination.coefficientCount_eq coefficient)
      RingKCombination.c0Cell,
    PiRLCGeneratedRelocation.combinationOutput_supported family offset
      (Formal.evalKOffset offset) (offset + Formal.logicalPrivateCount)
      startLe finishLe EvalKCombination.block
      (Fin.cast EvalKCombination.coefficientCount_eq coefficient)
      RingKCombination.c1Cell⟩

private theorem evalAOutput_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    KSupported
      (EvalACombination.output
        (Formal.evalAInterface (Formal.atOffset interface offset))
        (Formal.evalAOffset offset) matrix coefficient)
      (SupportRange.Extend (fun _ => False) offset
        (offset + Formal.logicalPrivateCount)) := by
  let family := RingKCombination.familyInterface
    (EvalACombination.ringInterface
      (Formal.evalAInterface (Formal.atOffset interface offset)))
  have startLe : offset ≤ Formal.evalAOffset offset := by
    unfold Formal.evalAOffset Formal.evalKOffset Formal.publicInputOffset
      Formal.commitmentOffset Formal.samplerOffset
    omega
  have finishLe : Formal.evalAOffset offset +
      CombinationFamily.logicalPrivateCount EvalACombination.blockCount
        RingKCombination.cellCount ≤
      offset + Formal.logicalPrivateCount := by
    rw [EvalACombination.logicalPrivateCount_eq]
    unfold Formal.evalAOffset Formal.evalKOffset Formal.publicInputOffset
      Formal.commitmentOffset Formal.samplerOffset
    rw [SamplerChain.logicalPrivateCount_eq]
    norm_num [Formal.logicalPrivateCount]
  unfold KSupported EvalACombination.output RingKCombination.output
  exact ⟨
    PiRLCGeneratedRelocation.combinationOutput_supported family offset
      (Formal.evalAOffset offset) (offset + Formal.logicalPrivateCount)
      startLe finishLe matrix
      (Fin.cast EvalKCombination.coefficientCount_eq coefficient)
      RingKCombination.c0Cell,
    PiRLCGeneratedRelocation.combinationOutput_supported family offset
      (Formal.evalAOffset offset) (offset + Formal.logicalPrivateCount)
      startLe finishLe matrix
      (Fin.cast EvalKCombination.coefficientCount_eq coefficient)
      RingKCombination.c1Cell⟩

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

/-- The complete combined PiRLC output relocates uniformly with the phase. -/
theorem evalOutput_eq_of_shift_agreement
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (left right : Formal.Interface logicalWidth publicFits)
    (leftOffset delta : Nat) (leftEnv rightEnv : Env)
    (pointEq : InputBinding.evalPoint (left.point leftOffset) leftEnv =
      InputBinding.evalPoint (right.point (leftOffset + delta)) rightEnv)
    (agrees : ∀ index,
      SupportRange.Extend (fun _ => False) leftOffset
          (leftOffset + Formal.logicalPrivateCount) index →
        rightEnv (index + delta) = leftEnv index) :
    Semantics.evalOutput relation left leftOffset leftEnv =
      Semantics.evalOutput relation right (leftOffset + delta) rightEnv := by
  apply inputInstance_ext
  · rfl
  · funext row lane
    change
      (CommitmentCombination.output
        (Formal.commitmentInterface (Formal.atOffset left leftOffset))
        (Formal.commitmentOffset leftOffset) row lane).eval leftEnv =
      (CommitmentCombination.output
        (Formal.commitmentInterface
          (Formal.atOffset right (leftOffset + delta)))
        (Formal.commitmentOffset (leftOffset + delta)) row lane).eval rightEnv
    rw [commitmentOutput_shift]
    exact (expression_eval_eq_of_shift_agreement delta _ _ leftEnv rightEnv
      (commitmentOutput_supported left leftOffset row lane) agrees).symm
  · funext column
    change
      (PublicInputCombination.output
        (Formal.publicInputInterface (Formal.atOffset left leftOffset))
        (Formal.publicInputOffset leftOffset) column).eval leftEnv =
      (PublicInputCombination.output
        (Formal.publicInputInterface
          (Formal.atOffset right (leftOffset + delta)))
        (Formal.publicInputOffset (leftOffset + delta)) column).eval rightEnv
    rw [publicInputOutput_shift]
    exact (expression_eval_eq_of_shift_agreement delta _ _ leftEnv rightEnv
      (publicInputOutput_supported left leftOffset column) agrees).symm
  · change InputBinding.evalPoint (left.point leftOffset) leftEnv =
      InputBinding.evalPoint (right.point (leftOffset + delta)) rightEnv
    exact pointEq
  · apply congrArg (fun value => #[value])
    apply evaluation_ext
    · funext coefficient
      change
        (EvalKCombination.output
          (Formal.evalKInterface (Formal.atOffset left leftOffset))
          (Formal.evalKOffset leftOffset) coefficient).eval leftEnv =
        (EvalKCombination.output
          (Formal.evalKInterface
            (Formal.atOffset right (leftOffset + delta)))
          (Formal.evalKOffset (leftOffset + delta)) coefficient).eval rightEnv
      rw [evalKOutput_shift]
      exact (quadratic_eval_eq_of_shift_agreement delta _ _ leftEnv rightEnv
        (evalKOutput_supported left leftOffset coefficient) agrees).symm
    · funext matrix coefficient
      change
        (EvalACombination.output
          (Formal.evalAInterface (Formal.atOffset left leftOffset))
          (Formal.evalAOffset leftOffset) matrix coefficient).eval leftEnv =
        (EvalACombination.output
          (Formal.evalAInterface
            (Formal.atOffset right (leftOffset + delta)))
          (Formal.evalAOffset (leftOffset + delta)) matrix coefficient).eval
            rightEnv
      rw [evalAOutput_shift]
      exact (quadratic_eval_eq_of_shift_agreement delta _ _ leftEnv rightEnv
        (evalAOutput_supported left leftOffset matrix coefficient) agrees).symm
  · rfl

end NightstreamFPrime.Layout.Stage1.PiRLCOutputRelocation
