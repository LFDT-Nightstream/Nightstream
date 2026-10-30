import CompressionLean.Ports.Circuit.Mle
import CompressionLean.Ports.Circuit.Early

namespace CompressionLean.Circuit

open CompressionLean.Replay (Word Layout Relation D)

def laneTables (layout : Layout) (tau : Array ExtVar) : Build (Array (Array ExtVar × Array ExtVar)) :=
  layout.classes.mapM fun c => Mle.laneSplit tau c.len c.ratio

def slotStep (layout : Layout) (point : Array ExtVar) (lanes : Array (Array ExtVar × Array ExtVar)) (t : ℕ) :
    Build ExtVar :=
  Mle.step point (layout.segments.map (·.first))
    (layout.segments.map fun s => let pair := lanes[s.classIndex]!; (if t = 0 then pair.1 else pair.2)[s.lane]!)
    layout.slots

structure LateOpeningsView where
  pairPoint : Array ExtVar
  block : ExtVar
  blockPoint : Array ExtVar
  omega : ExtVar

def checkLate (layout : Layout) (beta : ExtVar) (evalA : Array (ExtVar × ExtVar))
    (lanes : Array (Array ExtVar × Array ExtVar)) (claims : Array TreeClaimView) (pairs : Array (ExtVar × ExtVar)) :
    Build LateOpeningsView := do
  let (blockPairs, blocks) := (claims[0]!, claims[1]!)
  checkPair blockPairs blocks
  let slotPart := blockPairs.point.extract 0 layout.slotVariables
  let t := blockPairs.point[layout.slotVariables]!
  let mut ubar ← Algebra.extZero
  for ((re, im), (wRe, wIm)) in pairs.zip evalA do
    let a ← Build.extMul wRe re
    let c ← Build.extMul wIm im
    let term ← Build.extAdd a c
    ubar ← Build.extAdd ubar term
  let head ← slotStep layout slotPart lanes 0
  let tail ← slotStep layout slotPart lanes 1
  let low ← Algebra.oneMinus t
  let a ← Build.extMul low head
  let c ← Build.extMul t tail
  let lane ← Build.extAdd a c
  Build.assertExtEqual blockPairs.values[0]! ubar
  Build.assertExtEqual blockPairs.values[1]! lane
  let index ← Mle.identity blocks.point
  let expected ← Build.extSub beta index
  Build.assertExtEqual blocks.values[1]! expected
  let rest ← Build.extSub beta blockPairs.values[2]!
  let block ← Build.extSub rest t
  pure { pairPoint := slotPart, block, blockPoint := blocks.point, omega := blocks.values[0]! }

def lateStage (rel : Relation) (tau : Array ExtVar) (evalA : Array (ExtVar × ExtVar)) (late : GkrView)
    (lateValues : Array (ExtVar × ExtVar)) : Transcript LateOpeningsView := do
  let beta ← Duplex.sampleExt
  let claims ← gkrVerify late rel.lateShapes
  let lanes ← laneTables rel.key.layout tau
  let openings ← checkLate rel.key.layout beta evalA lanes claims lateValues
  observePairs lateValues
  pure openings

end CompressionLean.Circuit
