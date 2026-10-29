import CompressionLean.Circuit.Honest
import CompressionLean.Circuit.Whir

namespace CompressionLean.Circuit

open CompressionLean.Replay (Ext Relation foldVars)

def fromCoordinates (values : Array ExtVar) : Build ExtVar := do
  let mut total ← Algebra.extZero
  for (value, c) in values.zipIdx do
    let basis ← Algebra.extConstant (Ext.basis c)
    let term ← Build.extMul basis value
    total ← Build.extAdd total term
  pure total

structure OpeningsInput where
  queryPoints : Array (Array ExtVar)
  keyPoint : Array ExtVar
  runPoint : Array ExtVar
  blockPoint : Array ExtVar
  lateBlockPoint : Array ExtVar
  alpha : Array ExtVar
  queried : Array ExtVar
  omegaValue : ExtVar
  lateOmega : ExtVar
  normPoint : Array ExtVar
  linearPoint : Array ExtVar
  earlyRunPoint : Array ExtVar
  pairPoint : Array ExtVar
  latePairPoint : Array ExtVar
  rowPoint : Array ExtVar
  zLeaf : ExtVar
  zValue : ExtVar
  lateBlock : ExtVar
  mult : ExtVar
  p1 : DigestVar
  p0 : DigestVar
  partials : Array (Array ExtVar)
  earlyValues : EarlyValuesView
  lateValues : Array (ExtVar × ExtVar)
  p1Opening : OpeningView
  p0Opening : OpeningView

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
