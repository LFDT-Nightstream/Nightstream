import CompressionLean.Circuit.Early
import CompressionLean.Circuit.Fold
import CompressionLean.Circuit.Honest
import CompressionLean.Circuit.Late
import CompressionLean.Circuit.Linear
import CompressionLean.Circuit.Openings

/-!
Owns the layer-1 verifier of `crates/neo-spartan/src/verifier.rs` as a
builder program: the view of a proof as handles (`ProofView`, its shape run
`ProofView.read`) and `verify`, the whole of `verifier.rs:136-316`: the
seed, the relation words and the statement words in the duplex, then the
early stage (`earlyStage`), the linear stage (`linearStage`), the honest
fold (`honestStage`), the late stage (`lateStage`), the query points and the
openings (`openingsStage`).

The relation is the replay's (`Replay/Data.lean`): its sizes, tree shapes,
WHIR shapes and transcript words. What the openings read of each `Pcs`
beyond its `WhirShape` is a `WhirKey`. The statement is a `ClaimView`.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Relation D runOracles foldVars)

/-- A layer-1 proof as handles (`verifier.rs:24-39`). -/
structure ProofView where
  p0 : DigestVar
  histogram : Array Var
  early : GkrView
  earlyValues : EarlyValuesView
  quotient : Array ExtVar
  linear : Array (Array ExtVar)
  finals : Array ExtVar
  partials : Array (Array ExtVar)
  p1 : DigestVar
  leaves : Array LeafView
  late : GkrView
  lateValues : Array (ExtVar × ExtVar)
  p1Opening : OpeningView
  p0Opening : OpeningView

/-- `ProofView::read` with `proof = None` (`verifier.rs:43-122`): every proof
word in Rust's order. `p1Rows` and `p0Rows` are the query sets of the two
openings. -/
def ProofView.read (rel : Relation) (p1Rows p0Rows : OpeningRows) : Build ProofView := do
  let p0 ← privateDigest
  let histogram ← (Array.range (2 * rel.normBound + 1)).mapM fun _ => Build.priv
  let early ← GkrView.read rel.earlyShapes
  let runs ← privateExts 3
  let pairs ← privatePairs rel.matrices
  let block ← Algebra.privateExt
  let quotient ← privateExts (D - 1)
  let linear ← (Array.range rel.cubeVars).mapM fun _ => privateExts 2
  let finals ← privateExts 3
  let partials ← (Array.range 2).mapM fun _ => privateExts (2 ^ foldVars)
  let p1 ← privateDigest
  let leaves ← (Array.range rel.queries).mapM fun _ => LeafView.read (rel.kappa + runOracles) rel.setupVars
  let late ← GkrView.read rel.lateShapes
  let lateValues ← privatePairs rel.matrices
  let p1Opening ← OpeningView.read rel.p1Whir p1Rows
  let p0Opening ← OpeningView.read rel.p0Whir p0Rows
  pure { p0, histogram, early, earlyValues := { runs, pairs, block }, quotient, linear, finals, partials,
         p1, leaves, late, lateValues, p1Opening, p0Opening }

/-- What the early stage leaves to later stages: the norm tree's leaf point,
`z~` at it (`β - q~`) and the matrix openings. -/
structure EarlyOut where
  normPoint : Array ExtVar
  zLeaf : ExtVar
  openings : EarlyOpeningsView

/-- The early stage (`verifier.rs:153-168`): bind P0's root and the
histogram, draw `β` and the lookup and scatter challenges, check the norm
root against the histogram's fraction sum, then the matrix leaf claims, and
bind the early values. -/
def earlyStage (rel : Relation) (statement : ClaimView) (proof : ProofView) : Transcript EarlyOut := do
  observeRoot rel.commitmentSeed proof.p0
  Duplex.observeSlice proof.histogram
  let beta ← Duplex.sampleExt
  let lookupBeta ← Duplex.sampleExt
  let gamma ← Duplex.sampleExt
  let scatterBeta ← Duplex.sampleExt
  let mu ← Duplex.sampleExt
  let challenges : EarlyChallengesView := { lookup := (lookupBeta, gamma), scatter := (scatterBeta, mu) }
  let total ← tableSum proof.histogram rel.normBound beta
  let claims ← gkrVerify proof.early rel.earlyShapes
  let (p, q) := claims[0]!.root
  let _ ← Build.extInv q
  let expected ← Build.extMul q total
  Build.assertExtEqual p expected
  let zLeaf ← Build.extSub beta claims[0]!.values[0]!
  let openings ← checkEarly rel.key.layout statement.point challenges (claims.extract 1 claims.size)
    proof.earlyValues
  observeEarly proof.earlyValues
  pure { normPoint := claims[0]!.point, zLeaf, openings }

/-- `verify` (`verifier.rs:136-316`): a new duplex binds the seed, the
relation words as constants and the statement words; then the early stage,
the linear stage, the honest fold, the late stage, each query's point
(`verifier.rs:262-265`) and the two openings. -/
def verify (rel : Relation) (p1Key p0Key : WhirKey) (seed : DigestVar) (statement : ClaimView)
    (proof : ProofView) : Build Unit := do
  let duplex ← Duplex.new
  let run : Transcript Unit := do
    Duplex.observeSlice (Array.ofFn seed)
    for word in rel.words do
      Duplex.observe (← Build.const word)
    Duplex.observeSlice statement.words
    let early ← earlyStage rel statement proof
    let lin ← linearStage rel statement proof.quotient proof.linear proof.finals
    let honest ← honestStage rel
      { lanePoint := lin.lanePoint, blockPoint := lin.blockPoint, runPoint := early.openings.runPoint,
        setup := early.openings.setup, block := proof.earlyValues.block,
        commitmentRows := (Array.range rel.kappa).map lin.mixing.commitment, keyValue := lin.keyValue,
        partials := proof.partials, p1 := proof.p1, leaves := proof.leaves }
    let late ← lateStage rel lin.tau ((Array.range rel.matrices).map lin.mixing.evalA) proof.late proof.lateValues
    let queryPoints ← (honest.indexBits.mapM (queryPoint rel.setupVars) : Build _)
    openingsStage rel p1Key p0Key
      { queryPoints, keyPoint := honest.keyPoint, runPoint := honest.runPoint, blockPoint := lin.blockPoint,
        lateBlockPoint := late.blockPoint, alpha := honest.alpha, queried := honest.queried,
        omegaValue := lin.omegaValue, lateOmega := late.omega, normPoint := early.normPoint,
        linearPoint := lin.linearPoint, earlyRunPoint := early.openings.runPoint,
        pairPoint := early.openings.pairPoint, latePairPoint := late.pairPoint, rowPoint := early.openings.rowPoint,
        zLeaf := early.zLeaf, zValue := lin.zValue, lateBlock := late.block, mult := early.openings.mult,
        p1 := proof.p1, p0 := proof.p0, partials := proof.partials, earlyValues := proof.earlyValues,
        lateValues := proof.lateValues, p1Opening := proof.p1Opening, p0Opening := proof.p0Opening }
  let _ ← run.run duplex

end CompressionLean.Circuit
