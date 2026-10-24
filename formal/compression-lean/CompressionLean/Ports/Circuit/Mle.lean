import CompressionLean.Ports.Circuit.Algebra
import CompressionLean.Ports.Replay.Data

/-!
Owns the closed forms of `crates/neo-spartan/src/mle.rs` that the verifier
stages evaluate, as builder programs, in the order of `mle.rs`: the
scaled-eq point of a power weight (`scaledEqPoint`), the interval and step
functions, the identity and `χ_r` that the leaf checks read (`lessThan`,
`step`, `identity`, `chi`), the lane split (`laneSplit`) and the Eval_K
(Pad) weight by long division (`evalK`). Each one calls `Build` in the order
of its Rust counterpart. Points are low bit first.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (D Word)

namespace Mle

/-- `κ = Π_t (1 + c^{2^t})` and `x_t = c^{2^t}/(1 + c^{2^t})`
(`scaled_eq_point`, `mle.rs:21-34`). The inverse rejects a pole. -/
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

/-- The long division's states: `states[remainder][below]`, `none` where the
state is not reachable. -/
abbrev DivisionStates := Array (Array (Option KxVar))

/-- `Σ_{b<blocks} eq(s_block, b)·Σ_{l<D} τ_l·χ_r(D·b + l)` by long division of
the coordinate by `D`, high bit first (`eval_k`, `mle.rs:146-204`). -/
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
