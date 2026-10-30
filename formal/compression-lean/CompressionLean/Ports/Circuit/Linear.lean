import CompressionLean.Ports.Circuit.Mle
import CompressionLean.Ports.Circuit.Ring
import CompressionLean.Ports.Circuit.Sumcheck

namespace CompressionLean.Circuit

open CompressionLean.Replay (Relation D laneVars)

def laneValue (zeta : ExtVar) (point : Array ExtVar) : Build ExtVar := do
  let eq ← Algebra.eqTable point
  let mut total ← Algebra.extZero
  let mut power ← Algebra.extOne
  for weight in eq.extract 0 D do
    let term ← Build.extMul weight power
    total ← Build.extAdd total term
    power ← Build.extMul power zeta
  pure total

structure LinearView where
  mixing : MixingVar
  tau : Array ExtVar
  blockPoint : Array ExtVar
  lanePoint : Array ExtVar
  linearPoint : Array ExtVar
  keyValue : ExtVar
  omegaValue : ExtVar
  zValue : ExtVar

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
