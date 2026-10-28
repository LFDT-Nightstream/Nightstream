import CompressionLean.Circuit.Mle
import CompressionLean.Circuit.Ring
import CompressionLean.Circuit.Sumcheck

/-!
Owns the linear stage of the layer-1 verifier (`crates/neo-spartan/src/verifier.rs:170-206`)
as a builder program: `λ` and the mixing, the quotient and `ζ`, the lifted
target, the linear sum-check, the finals and the linear claim
(`linearStage`), with `lane_value` (`laneValue`). It calls `Build` in Rust's
order and returns what later stages read (`LinearView`).

The stage starts from the duplex after the early stage (`observe_early`,
`verifier.rs:168`) and reads the statement and three parts of the proof
view; it reads nothing else that the early stage computes.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Relation D laneVars)

/-- `Λ~(x) = Σ_{l<D} ζ^l·eq(x, l)` at the lane part of the linear point
(`lane_value`, `verifier.rs:340-350`). -/
def laneValue (zeta : ExtVar) (point : Array ExtVar) : Build ExtVar := do
  let eq ← Algebra.eqTable point
  let mut total ← Algebra.extZero
  let mut power ← Algebra.extOne
  for weight in eq.extract 0 D do
    let term ← Build.extMul weight power
    total ← Build.extAdd total term
    power ← Build.extMul power zeta
  pure total

/-- What later stages read of the linear stage. -/
structure LinearView where
  mixing : MixingVar
  tau : Array ExtVar
  blockPoint : Array ExtVar
  lanePoint : Array ExtVar
  /-- The lane bits, then the block bits. -/
  linearPoint : Array ExtVar
  keyValue : ExtVar
  omegaValue : ExtVar
  zValue : ExtVar

/-- `verifier.rs:170-206` on the proof's `quotient`, `linear` round messages
and `finals`. -/
def linearStage (rel : Relation) (statement : ClaimView) (quotient : Array ExtVar)
    (linear : Array (Array ExtVar)) (finals : Array ExtVar) : Transcript LinearView := do
  let lambda ← Duplex.sampleExt
  let mixing ← MixingVar.new lambda rel
  for value in quotient do
    Duplex.observeExt value
  let zeta ← Duplex.sampleExt
  let targets ← Ring.targets statement mixing
  let value ← Ring.liftedTarget targets quotient zeta
  let (point, last) ← sumcheckReplay linear value
  -- The block bits bind first, then the lane bits (`verifier.rs:179-181`).
  let blockPoint := point.extract 0 rel.blockVars
  let lanePart := point.extract rel.blockVars point.size
  let (keyValue, omegaValue, zValue) := (finals[0]!, finals[1]!, finals[2]!)
  for value in finals do
    Duplex.observeExt value
  let (scale, lanePoint) ← Mle.scaledEqPoint zeta laneVars
  let tau ← Ring.tau zeta
  let evalK ← Mle.evalK statement.point blockPoint tau rel.key.blocks
  let (kRe, kIm) := mixing.evalK
  let mut omega ← Build.extMul scale keyValue
  for (weight, value) in [(kRe, evalK.re), (kIm, evalK.im)] do
    let term ← Build.extMul weight value
    omega ← Build.extAdd omega term
  omega ← Build.extAdd omega omegaValue
  for block in [0:rel.publicBlocks] do
    let eq ← Algebra.eqAt blockPoint block
    let term ← Build.extMul eq (mixing.public block)
    omega ← Build.extAdd omega term
  let lanes ← laneValue zeta lanePart
  let weighted ← Build.extMul omega lanes
  let expected ← Build.extMul weighted zValue
  Build.assertExtEqual last expected
  pure { mixing, tau, blockPoint, lanePoint, linearPoint := lanePart ++ blockPoint, keyValue, omegaValue,
         zValue }

end CompressionLean.Circuit
