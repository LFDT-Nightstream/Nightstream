import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1

/-! Exact evaluation of the application's constant ten-block hash prefix. -/

namespace NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1Prefix

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Poseidon2
open Poseidon2HashChainV1

theorem absorbBlocksFast_append (prefixBlocks suffixBlocks : Nat)
    (state : State) (headWords suffix : List F)
    (length : headWords.length = prefixBlocks * rate) :
    absorbBlocksFast (prefixBlocks + suffixBlocks) state (headWords ++ suffix) =
      absorbBlocksFast suffixBlocks (absorbBlocksFast prefixBlocks state headWords) suffix := by
  induction prefixBlocks generalizing state headWords with
  | zero =>
      have empty : headWords = [] := by
        cases headWords with
        | nil => rfl
        | cons value rest => simp at length
      simp [empty, absorbBlocksFast]
  | succ count ih =>
      have enough : rate ≤ headWords.length := by rw [length, Nat.succ_mul]; omega
      have restLength : (headWords.drop rate).length = count * rate := by
        rw [List.length_drop, length, Nat.succ_mul]
        omega
      simp only [Nat.succ_add, absorbBlocksFast]
      rw [List.take_append_of_le_length enough, List.drop_append_of_le_length enough]
      exact ih _ _ restLength

def constantState : State := absorbBlocksFast 10 zeroState domainTag

def suffixHash (priorState message : List F) : List F :=
  let absorbed := absorbBlocksFast 2 constantState (priorState ++ message)
  let padded := permute ((List.range width).map fun i =>
    if i = 0 then absorbed.getD 0 0 + 1 else absorbed.getD i 0)
  padded.take digestLen

theorem suffixHash_eq_step (priorState message : List F)
    (priorLength : priorState.length = Application.stateWordCount)
    (messageLength : message.length = messageWordCount) :
    suffixHash priorState message = step priorState message := by
  unfold step
  rw [hash_eq_hashFast]
  unfold hashFast
  rw [preimage_length priorState message priorLength messageLength]
  change suffixHash priorState message =
    (permute ((List.range width).map fun i =>
      if i = 0 then
        (absorbBlocksFast 12 zeroState (preimage priorState message)).getD 0 0 + 1
      else (absorbBlocksFast 12 zeroState (preimage priorState message)).getD i 0)).take digestLen
  have prefixLength : domainTag.length = 10 * rate := by
    rw [domainTag_length]
    rfl
  have split := absorbBlocksFast_append 10 2 zeroState domainTag
    (priorState ++ message) prefixLength
  change absorbBlocksFast 12 zeroState (domainTag ++ (priorState ++ message)) = _ at split
  simp only [preimage, List.append_assoc]
  rw [split]
  rfl

/-- The two variable blocks and the unchanged padding permutation. -/
def threePermutations (priorState message : List F) : List F :=
  let second := absorbBlock (absorbBlock constantState priorState) message
  (permute ((List.range width).map fun i =>
    if i = 0 then second.getD 0 0 + 1 else second.getD i 0)).take digestLen

theorem threePermutations_eq_step (priorState message : List F)
    (priorLength : priorState.length = 4) (messageLength : message.length = 4) :
    threePermutations priorState message = step priorState message := by
  rw [← suffixHash_eq_step priorState message priorLength messageLength]
  unfold suffixHash threePermutations
  have takePrior : (priorState ++ message).take rate = priorState := by
    rw [show rate = priorState.length by simpa only [rate] using priorLength.symm]
    exact List.take_left
  have dropPrior : (priorState ++ message).drop rate = message := by
    rw [show rate = priorState.length by simpa only [rate] using priorLength.symm]
    exact List.drop_left
  have takeMessage : message.take rate = message := by
    rw [show rate = message.length by simpa only [rate] using messageLength.symm]
    exact List.take_length
  simp only [absorbBlocksFast, takePrior, dropPrior, takeMessage]

end NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1Prefix
