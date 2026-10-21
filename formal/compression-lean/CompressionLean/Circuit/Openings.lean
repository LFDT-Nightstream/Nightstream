import CompressionLean.Circuit.Honest
import CompressionLean.Circuit.Whir

/-!
Owns the openings of the layer-1 verifier (`crates/neo-spartan/src/verifier.rs:266-315`)
as a builder program (`openingsStage`): the P1 opening and its equality
checks, then the P0 opening and its equality checks. It calls `Build` in
Rust's order.

The stage starts from the duplex after the late stage (`observe_pairs`,
`verifier.rs:260`), with the query points of `verifier.rs:262-265`. It reads
only `OpeningsInput`: what the earlier stages compute and the proof parts
the checks name.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Ext Relation foldVars)

/-- The `Ext` value whose coordinate columns opened to `values`
(`from_coordinates`, `verifier.rs:352-362`). -/
def fromCoordinates (values : Array ExtVar) : Build ExtVar := do
  let mut total ← Algebra.extZero
  for (value, c) in values.zipIdx do
    let basis ← Algebra.extConstant (Ext.basis c)
    let term ← Build.extMul basis value
    total ← Build.extAdd total term
  pure total

/-- What the openings read, by their names in `verifier.rs`. -/
structure OpeningsInput where
  /-- Each setup query's point (`query_points`, `verifier.rs:262-265`). -/
  queryPoints : Array (Array ExtVar)
  /-- The honest fold's points on the setup variables (`key_point`, `run_point`). -/
  keyPoint : Array ExtVar
  runPoint : Array ExtVar
  /-- The block part of the linear point and the late block point. -/
  blockPoint : Array ExtVar
  lateBlockPoint : Array ExtVar
  /-- The honest fold's challenges (`alpha`) and the queried group values (`queried`). -/
  alpha : Array ExtVar
  queried : Array ExtVar
  /-- `Ω^A~(s_b)` from the finals and `late.omega`. -/
  omegaValue : ExtVar
  lateOmega : ExtVar
  /-- The norm tree's point (`claims[0].point`), the linear point, the early
  run, pair and row points and the late pair point. -/
  normPoint : Array ExtVar
  linearPoint : Array ExtVar
  earlyRunPoint : Array ExtVar
  pairPoint : Array ExtVar
  latePairPoint : Array ExtVar
  rowPoint : Array ExtVar
  /-- `z~` at the norm leaf and at the linear point, `late.block`, `early.mult`. -/
  zLeaf : ExtVar
  zValue : ExtVar
  lateBlock : ExtVar
  mult : ExtVar
  /-- The proof's roots, partials, early and late values and openings. -/
  p1 : DigestVar
  p0 : DigestVar
  partials : Array (Array ExtVar)
  earlyValues : EarlyValuesView
  lateValues : Array (ExtVar × ExtVar)
  p1Opening : OpeningView
  p0Opening : OpeningView

/-- `verifier.rs:266-315`: P1 opens the fold groups at the fold claims and at
every query point, and `Ω^A` at the two block points; each value, from its
coordinates, equals the stage value. P0 opens at the early, linear and late
points; each value equals the stage value or the proof value. -/
def openingsStage (rel : Relation) (p1Key p0Key : WhirKey) (o : OpeningsInput) : Transcript Unit := do
  let points := #[o.keyPoint.extract foldVars o.keyPoint.size, o.runPoint.extract foldVars o.runPoint.size] ++
    o.queryPoints ++ #[o.blockPoint, o.lateBlockPoint]
  let opened ← whirVerify rel.p1Whir p1Key o.p1 o.p1Opening points
  let keyGroup ← combinePartials o.partials[0]! o.alpha
  let runGroup ← combinePartials o.partials[1]! o.alpha
  let expected := #[keyGroup, runGroup] ++ o.queried ++ #[o.omegaValue, o.lateOmega]
  let groups := opened.flatMap fun batch => (Array.range ((batch.size + 2) / 3)).map fun c =>
    batch.extract (3 * c) (3 * c + 3)
  let values ← (groups.mapM fromCoordinates : Build _)
  for (value, expected) in values.zip expected do
    Build.assertExtEqual value expected
  let slots := o.earlyRunPoint.extract 0 rel.key.layout.slotVariables
  let points := #[o.normPoint, o.linearPoint, o.earlyRunPoint, o.pairPoint, slots, o.latePairPoint, o.rowPoint]
  let opened ← whirVerify rel.p0Whir p0Key o.p0 o.p0Opening points
  let flat (pairs : Array (ExtVar × ExtVar)) : Array ExtVar := pairs.flatMap fun (a, b) => #[a, b]
  let v := o.earlyValues
  let expected := #[#[o.zLeaf], #[o.zValue], #[v.runs[0]!, v.runs[1]!], flat v.pairs, #[v.block],
    flat o.lateValues ++ #[o.lateBlock], #[o.mult]]
  for (value, expected) in opened.flatten.zip expected.flatten do
    Build.assertExtEqual value expected

end CompressionLean.Circuit
