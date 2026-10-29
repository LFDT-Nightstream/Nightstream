import CompressionLean.Ports.Circuit.Sumcheck
import CompressionLean.Ports.Circuit.Views

namespace CompressionLean.Circuit

open CompressionLean.Replay (TreeShape)

structure TreeClaimView where
  root : ExtVar × ExtVar
  point : Array ExtVar
  values : Array ExtVar
deriving Inhabited

def gkrCombine (children : Array ExtVar) (mu : ExtVar) : Build ExtVar := do
  let factors := children.size / 2 - 1
  let numerator (side : ℕ) : Build ExtVar := do
    let mut total ← Algebra.extOne
    for j in [0:factors] do
      total ← Build.extMul total children[2 * j + side]!
    pure total
  let p0 ← numerator 0
  let p1 ← numerator 1
  let (q0, q1) := (children[2 * factors]!, children[2 * factors + 1]!)
  let a ← Build.extMul p0 q1
  let c ← Build.extMul p1 q0
  let qq ← Build.extMul q0 q1
  let d ← Build.extMul mu qq
  let sum ← Build.extAdd a c
  Build.extAdd sum d

def gkrVerify (view : GkrView) (shapes : Array TreeShape) : Transcript (Array TreeClaimView) := do
  for (p, q) in view.roots do
    Duplex.observeExt p
    Duplex.observeExt q
  let mut current := view.roots
  let mut claims : Array TreeClaimView := Array.replicate shapes.size default
  let mut point : Array ExtVar := #[]
  for (layer, index) in view.layers.zipIdx do
    let k := index + 1
    let active := (Array.range shapes.size).filter fun i => k ≤ shapes[i]!.depth
    let mu ← Duplex.sampleExt
    let nu ← Duplex.sampleExt
    let mut claim ← Algebra.extZero
    let mut expected ← Algebra.extZero
    let mut weight ← Algebra.extOne
    for (i, values) in active.zip layer.children do
      let (p, q) := current[i]!
      let mq ← Build.extMul mu q
      let pq ← Build.extAdd p mq
      let term ← Build.extMul weight pq
      claim ← Build.extAdd claim term
      let combined ← gkrCombine values mu
      let term ← Build.extMul weight combined
      expected ← Build.extAdd expected term
      weight ← Build.extMul weight nu
    let (next, last) ← sumcheckReplay layer.rounds claim
    let eq ← Algebra.eqEval point next
    let target ← Build.extMul eq expected
    Build.assertExtEqual last target
    for values in layer.children do
      for value in values do
        Duplex.observeExt value
    let tau ← Duplex.sampleExt
    let next := next.push tau
    for (i, values) in active.zip layer.children do
      let reduced ← (Array.range (values.size / 2)).mapM fun j => do
        let difference ← Build.extSub values[2 * j + 1]! values[2 * j]!
        let step ← Build.extMul tau difference
        Build.extAdd values[2 * j]! step
      if shapes[i]!.depth = k then
        claims := claims.set! i { root := view.roots[i]!, point := next, values := reduced }
      else
        current := current.set! i (reduced[0]!, reduced[1]!)
    point := next
  pure claims

end CompressionLean.Circuit
