import CompressionLean.Circuit.Early

/-!
Owns the late stage of the layer-1 verifier (`verifier.rs:251-260`) as a
builder program (`lateStage`): the late `β`, the late GKR (`gkrVerify` at
`Structure::late_shapes`), the lane tables (`lane_split`, `mle.rs:109-140`),
the late leaf checks (`Structure::check_late`, `matrix.rs:319-361`, with
`slot_step`) and the binding of the late pairs. Each part calls `Build` in
the order of its Rust counterpart.

The stage starts from the duplex after the setup queries
(`verifier.rs:244-249`). From earlier stages it reads only the lane weights
`τ` and the Eval_A weights of the mixing (`mixing.eval_a()`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Word Layout Relation D)

namespace Mle

/-- `lane_split`: the head and tail lane tables of a run class
`(length ≤ D, ratio)` for the lane weights `tau` (`mle.rs:109-140`). -/
def laneSplit (tau : Array ExtVar) (length : ℕ) (ratio : Word) : Build (Array ExtVar × Array ExtVar) := do
  let ratioPower := ratio.pow length
  let mut head := Array.replicate D tau[D - 1]!
  for lane in (List.range (D - 1)).reverse do
    let carried ← Algebra.extScaleConstant head[lane + 1]! ratio
    let mut value ← Build.extAdd tau[lane]! carried
    if lane + length < D then
      let dropped ← Algebra.extScaleConstant tau[lane + length]! ratioPower
      value ← Build.extSub value dropped
    head := head.set! lane value
  let mut sums := Array.replicate D tau[0]!
  let mut acc ← Algebra.extZero
  let mut power : Word := 1
  for (value, k) in tau.zipIdx do
    let term ← Algebra.extScaleConstant value power
    acc ← Build.extAdd acc term
    sums := sums.set! k acc
    power := power * ratio
  let zero ← Algebra.extZero
  let tail ← (Array.range D).mapM fun lane =>
    if D + 1 ≤ lane + length then Algebra.extScaleConstant sums[lane + length - (D + 1)]! (ratio.pow (D - lane))
    else pure zero
  pure (head, tail)

end Mle

/-- `Structure::lane_tables`: `[H0, H1]` of every class (`matrix.rs:217-224`). -/
def laneTables (layout : Layout) (tau : Array ExtVar) : Build (Array (Array ExtVar × Array ExtVar)) :=
  layout.classes.mapM fun c => Mle.laneSplit tau c.len c.ratio

/-- `slot_step`: `Ĥ_t~` over slots, the class's lane table `t` at the slot's
lane (`matrix.rs:382-392`). -/
def slotStep (layout : Layout) (point : Array ExtVar) (lanes : Array (Array ExtVar × Array ExtVar)) (t : ℕ) :
    Build ExtVar :=
  Mle.step point (layout.segments.map (·.first))
    (layout.segments.map fun s => let pair := lanes[s.classIndex]!; (if t = 0 then pair.1 else pair.2)[s.lane]!)
    layout.slots

/-- What the late leaf checks leave to openings (`matrix::LateOpenings`,
`matrix.rs:125-131`). -/
structure LateOpeningsView where
  pairPoint : Array ExtVar
  block : ExtVar
  blockPoint : Array ExtVar
  omega : ExtVar

/-- `Structure::check_late` (`matrix.rs:319-361`): the claims are the block
pairs and the blocks, in that order; `pairs` are the `U_j` values sent at the
slot part of the pair point. -/
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

/-- The late stage (`verifier.rs:251-260`): draw `β`, verify the late GKR,
build the lane tables from `tau`, check the late leaf claims against
`lateValues` and bind them. -/
def lateStage (rel : Relation) (tau : Array ExtVar) (evalA : Array (ExtVar × ExtVar)) (late : GkrView)
    (lateValues : Array (ExtVar × ExtVar)) : Transcript LateOpeningsView := do
  let beta ← Duplex.sampleExt
  let claims ← gkrVerify late rel.lateShapes
  let lanes ← laneTables rel.key.layout tau
  let openings ← checkLate rel.key.layout beta evalA lanes claims lateValues
  observePairs lateValues
  pure openings

end CompressionLean.Circuit
