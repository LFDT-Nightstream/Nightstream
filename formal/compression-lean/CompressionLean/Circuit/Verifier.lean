import CompressionLean.Circuit.Fold
import CompressionLean.Circuit.Sumcheck
import CompressionLean.Circuit.Views

/-!
Owns the layer-1 verifier of `crates/neo-spartan/src/verifier.rs` as a
builder program: the view of a proof as handles (`ProofView`, its shape run
`ProofView.read`) and `verify`, ported so far up to `verifier.rs:151` (the
seed, the relation words and the statement words in the duplex).

The relation is the replay's (`Replay/Data.lean`): its sizes, tree shapes,
WHIR shapes and transcript words. The statement is a `ClaimView`.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Relation D runOracles foldVars)

/-- `ClaimWords` over handles (`lib.rs:776-783`); a `K` value is `(re, im)`. -/
structure ClaimView where
  commitment : Array (Array Var)
  publicInput : Array (Array Var)
  point : Array KVar
  evalK : Array KVar
  evalA : Array (Array KVar)

/-- The ring rows, then the `K` values (`ClaimWords::words`, `lib.rs:834-843`). -/
def ClaimView.words (c : ClaimView) : Array Var :=
  c.commitment.flatten ++ c.publicInput.flatten ++
    (c.point ++ c.evalK ++ c.evalA.flatten).flatMap fun value => #[value.1, value.2]

/-- The prover's early values (`matrix::EarlyValues`): `e_re`, `e_im` and
the row column at the run point, the matrix pairs and the slot blocks. -/
structure EarlyValuesView where
  runs : Array ExtVar
  pairs : Array (ExtVar × ExtVar)
  block : ExtVar

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

/-- `verify` (`verifier.rs:136-151`): a new duplex binds the seed, the
relation words as constants and the statement words. -/
def verify (rel : Relation) (seed : DigestVar) (statement : ClaimView) : Build Unit := do
  let duplex ← Duplex.new
  let run : Transcript Unit := do
    Duplex.observeSlice (Array.ofFn seed)
    for word in rel.words do
      Duplex.observe (← Build.const word)
    Duplex.observeSlice statement.words
  let _ ← run.run duplex

end CompressionLean.Circuit
