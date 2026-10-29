import CompressionLean.Circuit.Algebra
import CompressionLean.Replay.Data

namespace CompressionLean.Circuit

open CompressionLean.Replay (D)

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
