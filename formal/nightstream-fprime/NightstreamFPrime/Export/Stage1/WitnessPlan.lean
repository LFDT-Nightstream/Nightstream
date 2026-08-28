import NightstreamFPrime.Export.Stage1.WitnessProgram

/-!
Owns the compact witness plan for the PiRLC sampler, PiDEC, and running-
transition suffix.

Each lane carries its logical start and source expression once. The fixed Lean
expansion reconstructs the nine canonical-u64 and candidate-decoder batches.
The structural expansion proof fixes all source, round, and lane bounds and
their order without evaluating the closed schedule.
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
  | digestLane (logicalStart : Nat) (source : Expr)
  | batches (values : List WitnessBatch)
deriving Repr

def Block.format : Format Block where
  encode
    | .digestLane logicalStart source => .array [
        .atom 0,
        .atom logicalStart,
        exprFormat.encode source]
    | .batches values => .array [
        .atom 1,
        (list WitnessBatch.format).encode values]
  decode
    | .array [.atom 0, .atom logicalStart, source] => do
      pure (.digestLane logicalStart (← exprFormat.decode source))
    | .array [.atom 1, values] => do
      pure (.batches (← (list WitnessBatch.format).decode values))
    | _ => .error "invalid witness plan block"
  decode_encode := by
    intro block
    cases block
    · simp only
      rw [exprFormat.decode_encode]
      rfl
    · simp only
      rw [(list WitnessBatch.format).decode_encode]
      rfl

def Block.expand : Block → List WitnessBatch
  | .digestLane logicalStart source =>
      WitnessProgram.digestLaneBatches source logicalStart
  | .batches values => values

def laneBlock
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source round : Nat) (lane : Fin 4) : Block :=
  let logicalStart :=
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart
      source round lane.val
  .digestLane logicalStart
    (PiRLCSamplerOrdinaryRows.fastLaneSource
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round lane)

@[simp] theorem laneBlock_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source round : Nat) (lane : Fin 4) :
    (laneBlock logicalWidth publicFits source round lane).expand =
      WitnessProgram.piRlcDigestLaneBatches logicalWidth publicFits
        source round lane := by
  unfold laneBlock Block.expand
    WitnessProgram.piRlcDigestLaneBatches
  rw [PiRLCSamplerOrdinaryRows.fastLaneSource_eq]

private theorem flatMap_map_expand {Alpha : Type}
    (values : List Alpha) (make : Alpha → Block) :
    (values.map make).flatMap Block.expand =
      values.flatMap fun value => (make value).expand := by
  induction values with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp [inductionHypothesis]

private theorem flatMap_flatMap_expand {Alpha : Type}
    (values : List Alpha) (blocks : Alpha → List Block) :
    (values.flatMap blocks).flatMap Block.expand =
      values.flatMap fun value => (blocks value).flatMap Block.expand := by
  induction values with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp [inductionHypothesis]

def windowBlocks
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source round : Nat) : List Block :=
  (List.finRange 4).map
    (laneBlock logicalWidth publicFits source round)

theorem windowBlocks_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source round : Nat) :
    (windowBlocks logicalWidth publicFits source round).flatMap Block.expand =
      WitnessProgram.piRlcWindowBatches logicalWidth publicFits
        source round := by
  unfold windowBlocks WitnessProgram.piRlcWindowBatches
  rw [flatMap_map_expand]
  simp_rw [laneBlock_expand]

def sourceBlocks
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source : Nat) : List Block :=
  (List.range 8).flatMap
    (windowBlocks logicalWidth publicFits source)

theorem sourceBlocks_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source : Nat) :
    (sourceBlocks logicalWidth publicFits source).flatMap Block.expand =
      WitnessProgram.piRlcSourceBatches logicalWidth publicFits source := by
  unfold sourceBlocks WitnessProgram.piRlcSourceBatches
  rw [flatMap_flatMap_expand]
  simp_rw [windowBlocks_expand]

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
