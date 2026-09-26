import NightstreamFPrime.Lifecycle.PiRLC.Wide.Batch
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily

/-! Connect the checked wide-sampler bits directly to the existing ring
combination families. The challenge view allocates no copy or scratch value.
The family retains its existing soundness and constructive completeness. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.Wide.Combination

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open v1_1

variable {blockCount cellCount : Nat} [NeZero cellCount]

abbrev Inputs (blockCount cellCount : Nat) :=
  Nat → Fin Batch.sourceCount → Fin blockCount → Fin ringDegree → Fin cellCount → Expr

def interface (samplerOffset : Nat) (inputs : Inputs blockCount cellCount) :
    CombinationFamily.Interface blockCount cellCount where
  challenge := fun _ => Batch.outputChallenge samplerOffset
  input := inputs

theorem assumptions (samplerOffset offset : Nat) (inputs : Inputs blockCount cellCount)
    (afterSampler : samplerOffset + Batch.privateCount ≤ offset)
    (inputsBelow : ∀ source block lane cell, (inputs offset source block lane cell).VarsBelow offset)
    (env : Env) : CombinationFamily.Assumptions (interface samplerOffset inputs) offset env := by
  constructor
  · intro source lane
    exact Expr.VarsBelow.mono _ (Batch.outputChallenge_below samplerOffset source lane) afterSampler
  · exact inputsBelow

theorem soundness (sampler : Batch.Interface) (samplerOffset offset : Nat)
    (inputs : Inputs blockCount cellCount) (env : Env)
    (sampled : Batch.SpecHolds sampler samplerOffset env)
    (afterSampler : samplerOffset + Batch.privateCount ≤ offset)
    (inputsBelow : ∀ source block lane cell, (inputs offset source block lane cell).VarsBelow offset)
    (rows : holds env (Circuit.ops (CombinationFamily.circuit (interface samplerOffset inputs)).main offset)) :
    ∀ block cell,
      CombinationFamily.evalOutput (interface samplerOffset inputs) offset env block cell =
        CombinationFamily.rightCombination (count := 17) (fun source =>
          ringFMul
            (Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt
              (Scalar.evalState env (sampler.initialState samplerOffset)) source.val)
            (fun lane => (inputs offset (Fin.cast Batch.sourceCount_eq.symm source) block lane cell).eval env)) := by
  intro block cell
  have result := CombinationFamily.soundness (interface samplerOffset inputs) offset env
    (assumptions samplerOffset offset inputs afterSampler inputsBelow env) rows block cell
  rw [result]
  unfold CombinationFamily.orderedCombination
  apply congrArg CombinationFamily.rightCombination
  funext source
  unfold CombinationFamily.term CombinationFamily.challengeValue CombinationFamily.inputValue
  change ringFMul (fun lane => (Batch.outputChallenge samplerOffset _ lane).eval env) _ = _
  rw [Batch.outputChallenge_eval sampler env samplerOffset sampled]
  rfl

theorem complete (samplerOffset offset : Nat) (inputs : Inputs blockCount cellCount)
    (afterSampler : samplerOffset + Batch.privateCount ≤ offset)
    (inputsBelow : ∀ source block lane cell, (inputs offset source block lane cell).VarsBelow offset)
    (env : Env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (CombinationFamily.circuit (interface samplerOffset inputs)).main offset)) ∧
      holdsFlat completed (Circuit.ops (CombinationFamily.circuit (interface samplerOffset inputs)).main offset) :=
  CombinationFamily.complete (interface samplerOffset inputs) offset env
    (assumptions samplerOffset offset inputs afterSampler inputsBelow env)

end NightstreamFPrime.Lifecycle.PiRLC.Wide.Combination
