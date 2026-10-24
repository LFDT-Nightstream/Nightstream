import Lean.Data.Json
import CompressionLean.Ports.Circuit.Hash
import CompressionLean.Ports.Replay.Gkr

/-!
Owns the views of the statement and of the parts of a layer-1 proof that
later verifier stages read, as handles: the statement (`ClaimWords`,
`lib.rs::ClaimWords`), a GKR proof (`gkr.rs::GkrView`), a queried setup leaf
(`setup.rs::LeafView`) and one WHIR opening (`whir/mod.rs::OpeningView`). Each
`read` is the shape run of its Rust counterpart (`proof = None`): it writes
one `private` per proof word, in Rust's order.

The sizes come from the replay's relation (`Ports/Replay/Data.lean`). A WHIR
opening also needs the index bits and the folding factor of each query set
(`QueryRows`), which the replay does not read. A relation whose shapes Rust
rejects is out of scope.
-/

namespace CompressionLean.Circuit

open Lean (FromJson)
open CompressionLean.Replay (TreeShape WhirShape layerFactors layerDegree layerCount foldVars)

/-- `ClaimWords` over handles (`lib.rs::ClaimWords`); a `K` value is `(re, im)`. -/
structure ClaimView where
  commitment : Array (Array Var)
  publicInput : Array (Array Var)
  point : Array KVar
  evalK : Array KVar
  evalA : Array (Array KVar)

/-- The ring rows, then the `K` values (`ClaimWords::words`, `lib.rs::ClaimWords::words`). -/
def ClaimView.words (c : ClaimView) : Array Var :=
  c.commitment.flatten ++ c.publicInput.flatten ++
    (c.point ++ c.evalK ++ c.evalA.flatten).flatMap fun value => #[value.1, value.2]

/-- `count` extension proof values. -/
def privateExts (count : ℕ) : Build (Array ExtVar) :=
  (Array.range count).mapM fun _ => Algebra.privateExt

/-- `count` pairs of extension proof values. -/
def privatePairs (count : ℕ) : Build (Array (ExtVar × ExtVar)) :=
  (Array.range count).mapM fun _ => do
    let a ← Algebra.privateExt
    pure (a, ← Algebra.privateExt)

/-- A digest of proof words. -/
def privateDigest : Build DigestVar := do
  let words ← (Array.range 4).mapM fun _ => Build.priv
  pure fun lane => words[lane]!

/-- One GKR layer: its round messages and each active tree's children. -/
structure GkrLayerView where
  rounds : Array (Array ExtVar)
  children : Array (Array ExtVar)

/-- A batched GKR proof (`GkrView`, `gkr.rs::GkrView`). -/
structure GkrView where
  roots : Array (ExtVar × ExtVar)
  layers : Array GkrLayerView

/-- `GkrView::read` (`gkr.rs::GkrView::read`): the roots, then per layer `k` its
`k - 1` round messages of the layer degree and each active tree's
`2·(factors + 1)` children. -/
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

/-- A queried setup leaf: its values and its path (`setup.rs::LeafView`). -/
structure LeafView where
  values : Array Var
  path : Array DigestVar

/-- `LeafView::read` (`setup.rs::LeafView::read`): `oracles·2^FOLD` values, then
`setupVars + 1 - FOLD` path digests. -/
def LeafView.read (oracles setupVars : ℕ) : Build LeafView := do
  let values ← (Array.range (oracles * 2 ^ foldVars)).mapM fun _ => Build.priv
  let path ← (Array.range (setupVars + 1 - foldVars)).mapM fun _ => privateDigest
  pure { values, path }

/-- What `OpeningView::read` needs of one query set beyond the replay's
`WhirShape`: the index bits (the draws when the query count is zero) and the
folding factor (the row width is `2^foldingFactor`). -/
structure QueryRows where
  indexBits : ℕ
  foldingFactor : ℕ
deriving FromJson, Inhabited

/-- The query sets of one opening: one per round, then the final one. -/
structure OpeningRows where
  rounds : Array QueryRows
  final : QueryRows
deriving FromJson, Inhabited

/-- One WHIR round (`RoundView`, `whir/mod.rs::RoundView`). -/
structure RoundView where
  root : DigestVar
  ood : Array ExtVar
  queries : Array (Array Var)
  sumcheck : Array (ExtVar × ExtVar)

/-- One WHIR opening as handles (`whir/mod.rs::OpeningView`). -/
structure OpeningView where
  initialOod : Array ExtVar
  evals : Array (Array ExtVar)
  initialSumcheck : Array (ExtVar × ExtVar)
  rounds : Array RoundView
  finalPoly : Array ExtVar
  finalQueries : Array (Array Var)
  finalSumcheck : Array (ExtVar × ExtVar)

/-- The query count, or every index when it is zero (`whir/mod.rs::draws`). -/
def queryDraws (count indexBits : ℕ) : ℕ := if count = 0 then 2 ^ indexBits else count

/-- `draws` rows of `width` values, or of `width` extension values as three
words each (`rows`, `whir/mod.rs::rows`). -/
def queryRows (draws width : ℕ) (base : Bool) : Build (Array (Array Var)) :=
  let rowLen := if base then width else 3 * width
  (Array.range draws).mapM fun _ => (Array.range rowLen).mapM fun _ => Build.priv

/-- `OpeningView::read` (`whir/mod.rs::OpeningView::read`). Round 0 opens base rows, and
the final set does when there are no rounds. -/
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
