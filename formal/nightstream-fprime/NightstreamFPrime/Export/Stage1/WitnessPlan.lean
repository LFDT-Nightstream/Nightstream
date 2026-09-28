import NightstreamFPrime.Export.Stage1.WitnessProgram

/-!
Owns the compact witness plan for the PiRLC sampler, PiDEC, and running-
transition suffix.

Each scalar carries the current reduction's canonical witness batches. The
structural expansion proof fixes source order without materializing the
complete witness schedule.
-/

namespace NightstreamFPrime.Export.Stage1.WitnessPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

inductive Block where
  | batches (values : List WitnessBatch)
deriving Repr

def Block.format : Format Block where
  encode
    | .batches values => .array [.atom 1, (list WitnessBatch.format).encode values]
  decode
    | .array [.atom 1, values] => do
      pure (.batches (← (list WitnessBatch.format).decode values))
    | _ => .error "invalid witness plan block"
  decode_encode := by
    intro block
    cases block
    simp only
    rw [(list WitnessBatch.format).decode_encode]
    rfl

def Block.expand : Block → List WitnessBatch
  | .batches values => values

private theorem flatMap_flatMap_expand {Alpha : Type}
    (values : List Alpha) (blocks : Alpha → List Block) :
    (values.flatMap blocks).flatMap Block.expand =
      values.flatMap fun value => (blocks value).flatMap Block.expand := by
  induction values with
  | nil => rfl
  | cons value rest inductionHypothesis => simp [inductionHypothesis]

def sourceBlocks (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth)
    (source : Nat) : List Block :=
  [.batches (WitnessProgram.piRlcSourceBatches logicalWidth publicFits source)]

theorem sourceBlocks_expand (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth)
    (source : Nat) :
    (sourceBlocks logicalWidth publicFits source).flatMap Block.expand =
      WitnessProgram.piRlcSourceBatches logicalWidth publicFits source := by
  simp only [sourceBlocks, List.flatMap_cons, List.flatMap_nil, Block.expand, List.append_nil]

def piRlcBlocks
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Block :=
  (List.range 17).flatMap
    (sourceBlocks logicalWidth publicFits)

theorem piRlcBlocks_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (piRlcBlocks logicalWidth publicFits).flatMap Block.expand =
      WitnessProgram.piRlcSamplerBatches logicalWidth publicFits := by
  unfold piRlcBlocks WitnessProgram.piRlcSamplerBatches
  rw [flatMap_flatMap_expand]
  simp_rw [sourceBlocks_expand]

def piDecBlock
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Block :=
  .batches (WitnessProgram.piDecBatches logicalWidth publicFits)

@[simp] theorem piDecBlock_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (piDecBlock logicalWidth publicFits).expand =
      WitnessProgram.piDecBatches logicalWidth publicFits := by
  rfl

def runningTransitionBlock
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Block :=
  .batches
    (WitnessProgram.directRunningTransitionBatches logicalWidth publicFits)

@[simp] theorem runningTransitionBlock_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (runningTransitionBlock logicalWidth publicFits).expand =
      WitnessProgram.runningTransitionBatches logicalWidth publicFits := by
  exact WitnessProgram.directRunningTransitionBatches_eq
    logicalWidth publicFits

def canonicalBlocks
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Block :=
  piRlcBlocks logicalWidth publicFits ++
    [piDecBlock logicalWidth publicFits,
      runningTransitionBlock logicalWidth publicFits]

theorem canonicalBlocks_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (canonicalBlocks logicalWidth publicFits).flatMap Block.expand =
      WitnessProgram.piRlcSamplerBatches logicalWidth publicFits ++
        (WitnessProgram.piDecBatches logicalWidth publicFits ++
          WitnessProgram.runningTransitionBatches logicalWidth publicFits) := by
  rw [canonicalBlocks, List.flatMap_append, piRlcBlocks_expand]
  simp only [List.flatMap_cons, List.flatMap_nil, piDecBlock_expand,
    runningTransitionBlock_expand, List.append_nil]

end NightstreamFPrime.Export.Stage1.WitnessPlan
