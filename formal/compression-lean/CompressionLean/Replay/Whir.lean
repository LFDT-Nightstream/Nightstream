import CompressionLean.Replay.Gkr

/-!
Owns the WHIR openings as the model idealizes them (`Soundness.lean`,
`Opens`): the opened values are taken as given, and the transcript of
`whir::verify` is replayed so that every WHIR draw matches Rust's. The WHIR
checks themselves (Merkle paths, folds, the final polynomial and the final
sum-check) are not replayed.
-/

namespace CompressionLean.Replay

/-- The opening has the shape's counts (`OpeningView::read`). -/
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

/-- A quadratic sum-check: its separator, then per round `(h(0), h(∞))` and a draw. -/
def whirSumcheck (shape : SumcheckShape) (pairs : Array (Ext × Ext)) : ReplayM Unit := do
  unless pairs.isEmpty do
    observeAll shape.seed
    for pair in pairs do
      observeExt pair.1
      observeExt pair.2
      discard sampleExt

/-- `draws` query indices: a sample, rejected when it is `p - 1` (`query_index`). -/
def whirQueries (draws : Nat) : ReplayM Unit := do
  for _ in [0:draws] do
    let word ← sampleWord
    ensure (!(word + 1 == 0)) "WHIR query sample p - 1"
    record (.sample word)

/-- The transcript of one WHIR opening; returns the opened batch values. -/
def whirTranscript (shape : WhirShape) (o : Opening) : ReplayM (Array (Array Ext)) := do
  for answer in o.initialOod do
    observeAll shape.virtualSeed
    discard sampleExt
    observeExt answer
  for pair in shape.batches.zip o.evals do
    observeAll pair.1.seed
    pair.2.forM observeExt
  observeAll shape.whirSeed
  observeAll shape.batchingSeed
  discard sampleExt
  whirSumcheck shape.initialSumcheck o.initialSumcheck
  for pair in shape.rounds.zip o.rounds do
    observeAll pair.2.root
    for answer in pair.2.ood do
      discard sampleExt
      observeExt answer
    whirQueries pair.1.queryDraws
    discard sampleExt
    whirSumcheck pair.1.sumcheck pair.2.sumcheck
  o.finalPoly.forM observeExt
  whirQueries shape.finalQueryDraws
  whirSumcheck shape.finalSumcheck o.finalSumcheck
  return o.evals

end CompressionLean.Replay
