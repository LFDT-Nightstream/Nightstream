import CompressionLean.Circuit.Gkr

/-!
Owns the parts of the layer-1 verifier's early stage that are not GKR, as
builder programs, each in the call order of its Rust counterpart:

- the root binding of a commitment (`whir/mod.rs:221-232`);
- the histogram's fraction sum (`norm.rs:36-49`);
- the closed forms that the early leaf checks read (`mle.rs:36-107`): the
  interval and step functions, the identity and `χ_r`;
- the early leaf checks (`Structure::check_early`, `matrix.rs:261-317`,
  with `tuple`, `check_pair` and `matrix_step`);
- the binding of the early values (`observe_early`, `verifier.rs:318-331`).

The late stage reads `checkPair`, `Mle.identity`, `Mle.step` and
`observePairs` too.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl Ext Layout)

/-- The prover's early values (`matrix::EarlyValues`, `matrix.rs:103-113`):
`e_re`, `e_im` and the row column at the run point, the matrix pairs and the
slot blocks. -/
structure EarlyValuesView where
  runs : Array ExtVar
  pairs : Array (ExtVar × ExtVar)
  block : ExtVar

/-- The lookup challenges `(β, γ)` and the scatter challenges `(β, μ)`
(`matrix::EarlyChallenges`, `matrix.rs:96-101`). -/
structure EarlyChallengesView where
  lookup : ExtVar × ExtVar
  scatter : ExtVar × ExtVar

/-- What the early leaf checks leave to openings (`matrix::EarlyOpenings`,
`matrix.rs:115-123`): `setup` is the run columns `i`, `σ`, `m` at the run
point. -/
structure EarlyOpeningsView where
  runPoint : Array ExtVar
  pairPoint : Array ExtVar
  rowPoint : Array ExtVar
  mult : ExtVar
  setup : ExtVar × ExtVar × ExtVar

/-- `observe_root`: the commitment separator `seed` as constants, then the
root (`whir/mod.rs:221-232`). -/
def observeRoot (seed : Array Gl) (root : DigestVar) : Transcript Unit := do
  for word in seed do
    Duplex.observe (← Build.const word)
  Duplex.observeSlice (Array.ofFn root)

/-- `table_sum`: `Σ_t m_t / (β - t)` over `t ∈ [-bound, bound]`, slot
`t + bound` (`norm.rs:36-49`). -/
def tableSum (counts : Array Var) (bound : ℕ) (beta : ExtVar) : Build ExtVar := do
  let mut total ← Algebra.extZero
  for (count, slot) in counts.zipIdx do
    let value ← Algebra.extConstant (Ext.ofGl (Gl.ofNat slot - Gl.ofNat bound))
    let denominator ← Build.extSub beta value
    let inverse ← Build.extInv denominator
    let term ← Build.extScale inverse count
    total ← Build.extAdd total term
  pure total

namespace Mle

/-- `less_than`: `Σ_{x < bound} eq(point, x)`, high bit first
(`mle.rs:36-55`). -/
def lessThan (point : Array ExtVar) (bound : ℕ) : Build ExtVar := do
  if bound >>> point.size ≠ 0 then return ← Algebra.extOne
  let mut total ← Algebra.extZero
  let mut head ← Algebra.extOne
  for t in (List.range point.size).reverse do
    let low ← Algebra.oneMinus point[t]!
    if bound.testBit t then
      let term ← Build.extMul head low
      total ← Build.extAdd total term
      head ← Build.extMul head point[t]!
    else
      head ← Build.extMul head low
  pure total

/-- `step`: `values[s]` on `[starts[s], starts[s + 1])`, the last segment
ending at `stop` (`mle.rs:57-72`). -/
def step (point : Array ExtVar) (starts : Array ℕ) (values : Array ExtVar) (stop : ℕ) : Build ExtVar := do
  let mut total ← Algebra.extZero
  let mut below ← lessThan point (starts[0]?.getD stop)
  for (value, s) in values.zipIdx do
    let upper ← lessThan point (starts[s + 1]?.getD stop)
    let interval ← Build.extSub upper below
    let term ← Build.extMul value interval
    total ← Build.extAdd total term
    below := upper
  pure total

/-- `identity`: `Σ_t 2^t point_t` (`mle.rs:74-82`). -/
def identity (point : Array ExtVar) : Build ExtVar := do
  let mut total ← Algebra.extZero
  for coordinate in point.reverse do
    let doubled ← Build.extAdd total total
    total ← Build.extAdd doubled coordinate
  pure total

/-- `chi`: `Π_{t<n} eq(r_t, point_t) · Π_{t≥n} (1 - r_t)` in `K ⊗ Ext`, with
`eq(r, x) = (1 - x) + r·(2x - 1)` (`mle.rs:84-107`). -/
def chi (r : Array KVar) (point : Array ExtVar) : Build KxVar := do
  let mut total ← KxVar.one
  for (value, t) in r.zipIdx do
    let rt ← KxVar.fromK value
    let factor ← match point[t]? with
      | some x => do
          let twice ← Build.extAdd x x
          let one ← Algebra.extOne
          let slope ← Build.extSub twice one
          let factor ← rt.scale slope
          let low ← Algebra.oneMinus x
          let re ← Build.extAdd factor.re low
          pure ⟨re, factor.im⟩
      | none => rt.oneMinus
    total ← total.mul factor
  pure total

end Mle

/-- `tuple`: `i + γ·(re + γ·im)` (`matrix.rs:394-400`). -/
def tuple (index gamma re im : ExtVar) : Build ExtVar := do
  let a ← Build.extMul gamma im
  let c ← Build.extAdd re a
  let d ← Build.extMul gamma c
  Build.extAdd index d

/-- `check_pair`: `P_x·Q_y = P_y·Q_x` with both denominators nonzero
(`matrix.rs:402-415`). -/
def checkPair (x y : TreeClaimView) : Build Unit := do
  let (px, qx) := x.root
  let (py, qy) := y.root
  let _ ← Build.extInv qx
  let _ ← Build.extInv qy
  let left ← Build.extMul px qy
  let right ← Build.extMul py qx
  Build.assertExtEqual left right

/-- `matrix_step`: `j~` over runs, matrix `j` on its run range and zero on
padding runs (`matrix.rs:368-380`). -/
def matrixStep (layout : Layout) (point : Array ExtVar) : Build ExtVar := do
  let values ← (Array.range layout.matrices).mapM fun j => Algebra.extConstant (Ext.ofGl (Gl.ofNat j))
  Mle.step point (layout.runBounds.extract 0 layout.matrices) values layout.runBounds[layout.matrices]!

/-- `Structure::check_early` (`matrix.rs:261-317`): the claims are the
lookup runs, lookup rows, scatter runs and scatter pairs, in that order. -/
def checkEarly (layout : Layout) (r : Array KVar) (challenges : EarlyChallengesView)
    (claims : Array TreeClaimView) (values : EarlyValuesView) : Build EarlyOpeningsView := do
  let (lookupRuns, lookupRows) := (claims[0]!, claims[1]!)
  let (scatterRuns, scatterPairs) := (claims[2]!, claims[3]!)
  checkPair lookupRuns lookupRows
  checkPair scatterRuns scatterPairs
  let (beta, gamma) := challenges.lookup
  let (eRe, eIm, row) := (values.runs[0]!, values.runs[1]!, values.runs[2]!)
  let combined ← tuple row gamma eRe eIm
  let expected ← Build.extSub beta combined
  Build.assertExtEqual lookupRuns.values[0]! expected
  let rowPoint := lookupRows.point
  let chi ← Mle.chi r rowPoint
  let index ← Mle.identity rowPoint
  let combined ← tuple index gamma chi.re chi.im
  let expected ← Build.extSub beta combined
  Build.assertExtEqual lookupRows.values[1]! expected
  let (beta, mu) := challenges.scatter
  let mixed ← Build.extMul mu eIm
  let e ← Build.extAdd eRe mixed
  Build.assertExtEqual scatterRuns.values[1]! e
  let runPoint := scatterRuns.point
  let matrix ← matrixStep layout runPoint
  let tag ← Algebra.extScaleConstant matrix (Gl.ofNat (2 ^ layout.slotVariables))
  let rest ← Build.extSub beta scatterRuns.values[2]!
  let slot ← Build.extSub rest tag
  let pairPoint := scatterPairs.point
  let eq ← Algebra.eqTable (pairPoint.extract layout.slotVariables pairPoint.size)
  let mut combined ← Algebra.extZero
  for ((re, im), weight) in values.pairs.zip eq do
    let mixed ← Build.extMul mu im
    let value ← Build.extAdd re mixed
    let term ← Build.extMul weight value
    combined ← Build.extAdd combined term
  Build.assertExtEqual scatterPairs.values[0]! combined
  let index ← Mle.identity pairPoint
  let expected ← Build.extSub beta index
  Build.assertExtEqual scatterPairs.values[1]! expected
  pure { runPoint, pairPoint := pairPoint.extract 0 layout.slotVariables, rowPoint,
         mult := lookupRows.values[0]!, setup := (row, slot, scatterRuns.values[0]!) }

/-- `observe_pairs` (`verifier.rs:326-331`). -/
def observePairs (pairs : Array (ExtVar × ExtVar)) : Transcript Unit :=
  pairs.forM fun (a, b) => do
    Duplex.observeExt a
    Duplex.observeExt b

/-- `observe_early` (`verifier.rs:318-324`). -/
def observeEarly (values : EarlyValuesView) : Transcript Unit := do
  values.runs.forM Duplex.observeExt
  observePairs values.pairs
  Duplex.observeExt values.block

end CompressionLean.Circuit
