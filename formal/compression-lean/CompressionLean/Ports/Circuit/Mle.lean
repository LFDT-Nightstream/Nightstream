import CompressionLean.Ports.Circuit.Algebra
import CompressionLean.Ports.Replay.Data

namespace CompressionLean.Circuit

open CompressionLean.Replay (D Word)

namespace Mle

def scaledEqPoint (c : ExtVar) (bits : ℕ) : Build (ExtVar × Array ExtVar) := do
  let one ← Algebra.extOne
  let mut kappa := one
  let mut point := #[]
  let mut power := c
  for _ in [0:bits] do
    let denominator ← Build.extAdd one power
    kappa ← Build.extMul kappa denominator
    let inverse ← Build.extInv denominator
    point := point.push (← Build.extMul power inverse)
    power ← Build.extMul power power
  pure (kappa, point)

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

def identity (point : Array ExtVar) : Build ExtVar := do
  let mut total ← Algebra.extZero
  for coordinate in point.reverse do
    let doubled ← Build.extAdd total total
    total ← Build.extAdd doubled coordinate
  pure total

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

abbrev DivisionStates := Array (Array (Option KxVar))

def evalK (r : Array KVar) (sBlock tau : Array ExtVar) (blocks : ℕ) : Build KxVar := do
  let blockBits := sBlock.size
  let coordinateBits := Nat.log2 (D * 2 ^ blockBits).nextPowerOfTwo
  let zero ← KxVar.zero
  let empty : DivisionStates := Array.replicate D #[none, none]
  let mut states := empty.set! 0 #[some (← KxVar.one), none]
  for t in (List.range coordinateBits).reverse do
    let rT ← KxVar.fromK r[t]!
    let rLow ← rT.oneMinus
    let boundBit := blocks / 2 ^ t % 2
    let mut next := empty
    for remainder in [0:D] do
      for below in [0:2] do
        let some value := (states[remainder]!)[below]! | continue
        for bit in [0:2] do
          let doubled := 2 * remainder + bit
          let quotient := if doubled ≥ D then 1 else 0
          if t ≥ blockBits ∧ quotient = 1 then continue
          let nextBelow : Option ℕ :=
            if below = 1 ∨ quotient < boundBit then some 1 else if quotient = boundBit then some 0 else none
          let some nextBelow := nextBelow | continue
          let mut factor := if bit = 0 then rLow else rT
          if t < blockBits then
            let s := sBlock[t]!
            let weight ← if quotient = 1 then pure s else Algebra.oneMinus s
            factor ← factor.scale weight
          let product ← value.mul factor
          let slot := doubled - D * quotient
          let merged ← match (next[slot]!)[nextBelow]! with
            | some existing => existing.add product
            | none => pure product
          next := next.set! slot (next[slot]!.set! nextBelow (some merged))
    states := next
  let mut total := zero
  for (accumulated, remainder) in states.zipIdx do
    if let some value := accumulated[1]! then
      let term ← value.scale tau[remainder]!
      total ← total.add term
  for value in r.extract coordinateBits r.size do
    let factor ← (← KxVar.fromK value).oneMinus
    total ← total.mul factor
  pure total

end Mle

end CompressionLean.Circuit
