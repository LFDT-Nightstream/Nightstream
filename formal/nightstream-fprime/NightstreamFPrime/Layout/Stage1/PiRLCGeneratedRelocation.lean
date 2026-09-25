import NightstreamFPrime.Gadgets.Poseidon2.Duplex.WiringShift
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.GeneratedSupport

/-!
Owns uniform offset relocation for PiRLC sampler-generated expressions.
All results use the existing direct fresh-variable formulas. This module
changes no sampler, transcript, circuit, row, or semantic predicate.
-/

namespace NightstreamFPrime.Layout.Stage1.PiRLCGeneratedRelocation

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.WiringShift
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem samplerOutputState_shift
    (left right : Sampler.Interface) (coordinate leftOffset delta : Nat) :
    Sampler.outputState right coordinate (leftOffset + delta) =
      state delta (Sampler.outputState left coordinate leftOffset) := by
  funext lane
  simp only [Sampler.outputState, DigestWindow.output,
    Permutation.Owned.output, Permutation.scheduleOutput,
    Permutation.freshState, state, expression]
  congr 1
  unfold Sampler.windowOffset Sampler.windowBase
    DigestWindow.permutationOffset
  omega

theorem samplerOutputWord_shift
    (leftOffset delta : Nat) (position : Fin ringDegree) :
    Sampler.outputWord (leftOffset + delta) position =
      expression delta (Sampler.outputWord leftOffset position) := by
  simp only [Sampler.outputWord, First54.output, First54ValueStep.output,
    expression]
  congr 1
  unfold First54.valueOffset First54.positionOffset Sampler.selectorOffset
    Sampler.windowBase
  omega

theorem stateAtExpr_shift
    (left right : SamplerChain.Interface) (leftOffset delta count : Nat)
    (initialShift : right.initialState (leftOffset + delta) =
      state delta (left.initialState leftOffset)) :
    SamplerChain.stateAtExpr right (leftOffset + delta) count =
      state delta (SamplerChain.stateAtExpr left leftOffset count) := by
  cases count with
  | zero => exact initialShift
  | succ source =>
      unfold SamplerChain.stateAtExpr
      rw [show SamplerChain.sourceOffset (leftOffset + delta) source =
          SamplerChain.sourceOffset leftOffset source + delta by
        unfold SamplerChain.sourceOffset
        omega]
      exact samplerOutputState_shift
        { initialState := fun _ =>
            SamplerChain.stateAtExpr left leftOffset source }
        { initialState := fun _ =>
            SamplerChain.stateAtExpr right (leftOffset + delta) source }
        source (SamplerChain.sourceOffset leftOffset source) delta

theorem challengeExpr_shift
    (left right : SamplerChain.Interface) (leftOffset delta : Nat)
    (source : Fin SamplerChain.sourceCount) (lane : Fin ringDegree) :
    SamplerChain.challengeExpr right (leftOffset + delta) source lane =
      expression delta
        (SamplerChain.challengeExpr left leftOffset source lane) := by
  unfold SamplerChain.challengeExpr Sampler.outputChallenge
  rw [show SamplerChain.sourceOffset (leftOffset + delta) source.val =
      SamplerChain.sourceOffset leftOffset source.val + delta by
    unfold SamplerChain.sourceOffset
    omega]
  rw [samplerOutputWord_shift]
  rfl

/-- A final combination output moves uniformly with its family start. The
family interface is absent from the output index by definition. -/
theorem combinationOutput_shift
    {blockCount cellCount : Nat} [NeZero cellCount]
    (left right : CombinationFamily.Interface blockCount cellCount)
    (leftOffset delta : Nat) (block : Fin blockCount)
    (lane : Fin ringDegree) (cell : Fin cellCount) :
    CombinationFamily.output right (leftOffset + delta) block lane cell =
      expression delta
        (CombinationFamily.output left leftOffset block lane cell) := by
  simp only [CombinationFamily.output, CombinationStep.output, expression]
  congr 1
  unfold CombinationFamily.stepOffset
  omega

/-- A final combination output is supported by an enclosing phase interval
whenever that interval contains the complete combination family. -/
theorem combinationOutput_supported
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : CombinationFamily.Interface blockCount cellCount)
    (phaseOffset familyOffset phaseFinish : Nat)
    (startLe : phaseOffset ≤ familyOffset)
    (finishLe : familyOffset +
        CombinationFamily.logicalPrivateCount blockCount cellCount ≤
      phaseFinish)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount) :
    (CombinationFamily.output interface familyOffset block lane cell).VarsSatisfy
      (SupportRange.Extend (fun _ => False) phaseOffset phaseFinish) := by
  simp only [CombinationFamily.output, CombinationStep.output,
    Expr.VarsSatisfy]
  apply Or.inr
  constructor
  · unfold CombinationFamily.stepOffset
    omega
  · let step := CombinationFamily.stepSize blockCount cellCount
    have indexLt : (CombinationStep.indexOf block lane cell).val < step := by
      simpa [step, CombinationFamily.stepSize] using
        (CombinationStep.indexOf block lane cell).isLt
    have sourceLt := CombinationFamily.finalSource.isLt
    have beforeNext :
        CombinationFamily.finalSource.val * step +
            (CombinationStep.indexOf block lane cell).val <
          (CombinationFamily.finalSource.val + 1) * step := by
      rw [Nat.add_mul]
      simpa using Nat.add_lt_add_left indexLt
        (CombinationFamily.finalSource.val * step)
    have beforeFamily :
        (CombinationFamily.finalSource.val + 1) * step ≤
          CombinationFamily.sourceCount * step :=
      Nat.mul_le_mul_right step (Nat.succ_le_of_lt sourceLt)
    have localLt :
        familyOffset + CombinationFamily.finalSource.val * step +
            (CombinationStep.indexOf block lane cell).val <
          familyOffset + CombinationFamily.sourceCount * step := by
      simpa [Nat.add_assoc] using
        Nat.add_lt_add_left (beforeNext.trans_le beforeFamily) familyOffset
    apply Nat.lt_of_lt_of_le localLt
    simpa [CombinationFamily.logicalPrivateCount, step] using finishLe

end NightstreamFPrime.Layout.Stage1.PiRLCGeneratedRelocation
