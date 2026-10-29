import CompressionLean.Replay.Gkr

namespace CompressionLean.Replay

def readOpening (shape : WhirShape) (o : Opening) : Except String Unit := do
  ensure (o.rounds.size == shape.rounds.size && o.evals.size == shape.batches.size &&
    o.initialOod.size == shape.oodSamples &&
    (shape.batches.zip o.evals).all fun pair => pair.2.size == pair.1.size) "WHIR proof shape"
  ensure (o.initialSumcheck.size == shape.initialSumcheck.rounds) "WHIR sum-check shape"
  for pair in shape.rounds.zip o.rounds do
    ensure (pair.2.ood.size == pair.1.oodSamples) "WHIR proof shape"
    ensure (pair.2.sumcheck.size == pair.1.sumcheck.rounds) "WHIR sum-check shape"
  ensure (o.finalPoly.size == shape.finalPolyLen) "WHIR proof shape"
  ensure (o.finalSumcheck.size == shape.finalSumcheck.rounds) "WHIR sum-check shape"

def whirSumcheck (shape : SumcheckShape) (pairs : Array (Ext × Ext)) : ReplayM Unit := do
  unless pairs.isEmpty do
    observeAll shape.seed
    pairs.toList.forM fun pair => do
      observeExts #[pair.1, pair.2]
      let _ ← sampleExt

def whirQueries (draws : Nat) : ReplayM Unit :=
  (List.range draws).forM fun _ => do
    let word ← sampleWord
    ensure (!(word + 1 == 0)) "WHIR query sample p - 1"
    record (.sample word)

def whirTranscript (shape : WhirShape) (o : Opening) : ReplayM Unit := do
  o.initialOod.toList.forM fun answer => do
    observeAll shape.virtualSeed
    let _ ← sampleExt
    observeExt answer
  (shape.batches.zip o.evals).toList.forM fun pair => do
    observeAll pair.1.seed
    observeExts pair.2
  observeAll shape.whirSeed
  observeAll shape.batchingSeed
  let _ ← sampleExt
  whirSumcheck shape.initialSumcheck o.initialSumcheck
  (shape.rounds.zip o.rounds).toList.forM fun pair => do
    observeAll pair.2.root
    pair.2.ood.toList.forM fun answer => do
      let _ ← sampleExt
      observeExt answer
    whirQueries pair.1.queryDraws
    let _ ← sampleExt
    whirSumcheck pair.1.sumcheck pair.2.sumcheck
  observeExts o.finalPoly
  whirQueries shape.finalQueryDraws
  whirSumcheck shape.finalSumcheck o.finalSumcheck

end CompressionLean.Replay
