import Lean.Data.Json
import CompressionLean.Circuit.Hash
import CompressionLean.Replay.Gkr

namespace CompressionLean.Circuit

open Lean (FromJson)
open CompressionLean.Replay (TreeShape WhirShape layerFactors layerDegree layerCount foldVars)

structure ClaimView where
  commitment : Array (Array Var)
  publicInput : Array (Array Var)
  point : Array KVar
  evalK : Array KVar
  evalA : Array (Array KVar)

def ClaimView.words (c : ClaimView) : Array Var :=
  c.commitment.flatten ++ c.publicInput.flatten ++
    (c.point ++ c.evalK ++ c.evalA.flatten).flatMap fun value => #[value.1, value.2]

def privateExts (count : ℕ) : Build (Array ExtVar) :=
  (Array.range count).mapM fun _ => Algebra.privateExt

def privatePairs (count : ℕ) : Build (Array (ExtVar × ExtVar)) :=
  (Array.range count).mapM fun _ => do
    let a ← Algebra.privateExt
    pure (a, ← Algebra.privateExt)

def privateDigest : Build DigestVar := do
  let words ← (Array.range 4).mapM fun _ => Build.priv
  pure fun lane => words[lane]!

structure GkrLayerView where
  rounds : Array (Array ExtVar)
  children : Array (Array ExtVar)

structure GkrView where
  roots : Array (ExtVar × ExtVar)
  layers : Array GkrLayerView

def GkrView.read (shapes : Array TreeShape) : Build GkrView := do
  let roots ← privatePairs shapes.size
  let layers ← (Array.range (layerCount shapes)).mapM fun index => do
    let k := index + 1
    let factors := layerFactors shapes k
    let degree := layerDegree factors
    let rounds ← (Array.range (k - 1)).mapM fun _ => privateExts degree
    let children ← factors.mapM fun f => privateExts (2 * (f + 1))
    pure ({ rounds, children } : GkrLayerView)
  pure { roots, layers }

structure LeafView where
  values : Array Var
  path : Array DigestVar

def LeafView.read (oracles setupVars : ℕ) : Build LeafView := do
  let values ← (Array.range (oracles * 2 ^ foldVars)).mapM fun _ => Build.priv
  let path ← (Array.range (setupVars + 1 - foldVars)).mapM fun _ => privateDigest
  pure { values, path }

structure QueryRows where
  indexBits : ℕ
  foldingFactor : ℕ
deriving FromJson, Inhabited

structure OpeningRows where
  rounds : Array QueryRows
  final : QueryRows
deriving FromJson, Inhabited

structure RoundView where
  root : DigestVar
  ood : Array ExtVar
  queries : Array (Array Var)
  sumcheck : Array (ExtVar × ExtVar)

structure OpeningView where
  initialOod : Array ExtVar
  evals : Array (Array ExtVar)
  initialSumcheck : Array (ExtVar × ExtVar)
  rounds : Array RoundView
  finalPoly : Array ExtVar
  finalQueries : Array (Array Var)
  finalSumcheck : Array (ExtVar × ExtVar)

def queryDraws (count indexBits : ℕ) : ℕ := if count = 0 then 2 ^ indexBits else count

def queryRows (draws width : ℕ) (base : Bool) : Build (Array (Array Var)) :=
  let rowLen := if base then width else 3 * width
  (Array.range draws).mapM fun _ => (Array.range rowLen).mapM fun _ => Build.priv

def OpeningView.read (shape : WhirShape) (rows : OpeningRows) : Build OpeningView := do
  let initialOod ← privateExts shape.oodSamples
  let evals ← shape.batches.mapM fun batch => privateExts batch.size
  let initialSumcheck ← privatePairs shape.initialSumcheck.rounds
  let rounds ← shape.rounds.zipIdx.mapM fun (round, r) => do
    let set := rows.rounds[r]!
    let root ← privateDigest
    let ood ← privateExts round.oodSamples
    let queries ← queryRows (queryDraws round.queryDraws set.indexBits) (2 ^ set.foldingFactor) (r == 0)
    let sumcheck ← privatePairs round.sumcheck.rounds
    pure ({ root, ood, queries, sumcheck } : RoundView)
  let finalPoly ← privateExts shape.finalPolyLen
  let finalQueries ← queryRows (queryDraws shape.finalQueryDraws rows.final.indexBits)
    (2 ^ rows.final.foldingFactor) shape.rounds.isEmpty
  let finalSumcheck ← privatePairs shape.finalSumcheck.rounds
  pure { initialOod, evals, initialSumcheck, rounds, finalPoly, finalQueries, finalSumcheck }

end CompressionLean.Circuit
